"""Lockstep block-pivot dual on the compiled CROWN form: a fixed number R of rounds per call, B boxes (lanes) at once,
fixed shapes, no host synchronisation, one round capturable as one CUDA graph; the state resumes (crown_absorb/lockstep).

The LP of a lane (one box): min c z + c0 s.t. R z <= h (the face rows of the compiled form and its output rows), l <= z
<= u, rows A = [R; -I; I], b = [h, -l, u]. Columns: the inputs, then the face nodes' elements level by level (a face row
of an element has its last nonzero, the tail, at the element; its other nonzeros on lower levels). A row basis I (n
rows) is represented as CROWN represents it: per column at most one basis row with that tail (a selected face or a box
end), scaled to a unit diagonal, gives the level-triangular T (the forward/backward through the selected faces); the
other basis rows correct T in k Schur slots: B^-1 q = v - W K^-1 V v, v = T^-1 q, W = T^-1 E_J, K = I + V W (k x k, the
only factorisation, cuSOLVER's batched LU). Dual feasible bases (y = -B^-T c >= 0) are kept; one round:
    factor the basis; y, x; rho = b - A x, the violated rows; up to q = 8 entering rows (the most violated, scaled, or
    Bland's smallest label once patience = n rounds have not lowered the violation count; permanent); D = B^-T A_J^T;
    per entering row the ratio test (smallest label among ties), a candidate Farkas ray when no positive entry; the
    joint exchange of the q', q'/2, ... first compatible pairs (the exchange kernel H = D[P, J] nonsingular, t = H^-1
    y_P >= 0, every new mass >= 0, the basis not seen in this call unless in Bland mode, the new basis factorable and
    dual feasible), the four sizes tried side by side, the first accepted wins;
    one rigorous certificate per round (the compiled CROWN certify: the round's bound, every candidate ray, the
    accepted exchange's bound), its lower bound >= T or a positive Farkas value closes the lane (masked thereafter).
Warm start (a child from its parent's state after the parent's own R rounds, never an optimal basis): the parent's
row labels, the scalar bound inherited, the parent's multipliers re-certified on the child; the basis repaired by box
end flips and, where real negative masses remain, a dual phase 1 (artificial rows -a_s of the negative rows, b = 0 on
the real rows and 1 on the artificial ones, then the artificial rows cleaned out of the basis), all inside the same R.
A numerical failure (no factor, negative mass, stationarity) restarts the lane at its CROWN basis (paid); a failed
exchange only shrinks the block. The semantics follow the reference block mechanism (scaled LP, Basis, tolerances) operation for operation; the certificate, not the search, decides every prune.

Opt-in: nothing else imports this module."""
import collections
import math

import torch

from .compiled_crown import CompiledGraph, Data, Node, Captured, INF, NF, fuse, column_scale
from .small_inverse import inverse

# the reference's tolerances
TOL_PHASE_NEG = 1e-12     # a real negative mass entering phase 1 (relative to 1 + max |y|)
TOL_MASS = 2e-8           # a negative mass of the round's basis: a numerical failure
TOL_STAT = 1e-6           # stationarity |A_I^T y + c| / (1 + |c|)
TOL_VIO = 1e-10           # a violated row: rho < -TOL_VIO (1 + |b| + |A| |x|)
TOL_RATE = 1e-10          # an artificial row's replacement in the cleanup
TOL_PIV = 1e-11           # a positive entry of the ratio test
TOL_TIE = 1e-12           # ratio ties
TOL_COND = 1e12           # the exchange kernel's condition number
TOL_JOINT = 2e-9          # the joint exchange's and the trial basis's masses
TOL_KCOND = INF           # the Schur kernel's condition number beyond which a lane falls back to fp64 (fp64: none)
TOLS64 = dict(PHASE_NEG=TOL_PHASE_NEG, MASS=TOL_MASS, STAT=TOL_STAT, VIO=TOL_VIO, RATE=TOL_RATE, PIV=TOL_PIV,
              TIE=TOL_TIE, COND=TOL_COND, JOINT=TOL_JOINT, KCOND=TOL_KCOND)
# an fp32 search: every tolerance the square root of the fp64 one (half the digits; eps32 ~ 2^-24 ~ sqrt(eps64) / 8);
# a Schur kernel with cond_1 > 1e6 sends its lane to fp64
TOLS32 = {k: (v ** 0.5 if k != 'KCOND' else 1e6) for k, v in TOLS64.items()}
Q = 8                     # the block size
ATTEMPTS = 4              # q', q' // 2, q' // 4, q' // 8 (q' <= 8)


def compiled_from_forms(n, nodes, CM, Cc, sparse=False):
    """A CompiledGraph (certify, R_A, F_A, F_T, R_T) on given operand forms instead of a traced network: n inputs; nodes
    a list of dicts (level order) name, E, P (1 or 2), M (P, E, ncol) the operand forms' coefficients over the columns
    [inputs | elements in node order], c (P, E) their constants, level, Fl, Fu (the face counts); CM (S, ncol), Cc (S,)
    the rows of C (row 0 the objective) as affine forms of the columns. The forms are taken as exact (no fold error:
    dMop = dCM = 0), the premise of a frozen binary64 graph; no base rows."""
    dt, dev = CM.dtype, CM.device
    cg = CompiledGraph.__new__(CompiledGraph)
    cg.net, cg.K, cg.free_tangent, cg.sparse = None, 5, False, sparse
    cg.margins, cg.nf, cg.dE = False, NF, None
    cg._Mop = cg._dMop = cg._CM = cg._dCM = None
    cg.in_shape, cg.n = (n,), n
    cg.Et = Et = sum(nd['E'] for nd in nodes)
    cg.ncol = ncol = n + Et
    cg.S = S = CM.shape[0]
    cg.rows = torch.arange(1, S, device=dev)
    cg.R = 0
    out, s, levels = [], 0, []
    for nd in nodes:
        out.append(Node(name=nd['name'], E=nd['E'], shape=(nd['E'],), c0=n + s, P=nd['P'], same=False,
                        operands=[f"{nd['name']}.op{p}" for p in range(nd['P'])], level=nd['level'], own=False, s=s,
                        Fl=nd['Fl'], Fu=nd['Fu']))
        if levels and levels[-1][2] == nd['level']:
            levels[-1][1] = s + nd['E']
        else:
            assert not levels or levels[-1][2] < nd['level'], 'nodes must come in level order'
            levels.append([s, s + nd['E'], nd['level']])
        s += nd['E']
    cg.nodes, cg.levels = out, [(a, b) for a, b, _ in levels]
    cg.k2c = cg.c2k = torch.arange(ncol, device=dev)
    cg.Mop = torch.zeros(2, Et, ncol, dtype=dt, device=dev)
    cg.cop = torch.zeros(2, Et, dtype=dt, device=dev)
    for nd, o in zip(nodes, out):
        cg.Mop[:nd['P'], o.s:o.s + o.E] = torch.as_tensor(nd['M'], dtype=dt, device=dev)
        cg.cop[:nd['P'], o.s:o.s + o.E] = torch.as_tensor(nd['c'], dtype=dt, device=dev)
    cg.dMop, cg.dcop = torch.zeros_like(cg.Mop), torch.zeros_like(cg.cop)
    cg.CM, cg.Cc = CM.clone(), Cc.clone()
    cg.dCM, cg.dCc = torch.zeros_like(cg.CM), torch.zeros_like(cg.Cc)
    cg.E = torch.zeros(0, n, dtype=dt, device=dev)
    cg._mats()
    cg.fill = {'nnz_folded': int((cg.Mop != 0).sum()) + int((cg.CM != 0).sum())}
    cg._private = {}
    cg._set_counts({o.name: (o.Fl, o.Fu) for o in out})
    cg._row_maps()
    return cg


class CompiledCertifier:
    """The compiled CROWN certificate of B lanes on static per-lane Data (capturable): masses (B, Q, m) on the table
    rows (row i -> its place rowmap[i] in the compiled flat layout [lower faces | upper faces | output rows | base
    rows]) and an objective weight objw (Q,) per right-hand side (1: the bound D(y); 0: the Farkas value). The objective
    enters as an output row c z + c0 <= 0 (rows of C: [objective, objective as output row, the table's output rows]),
    so one certify (objective 0) serves bounds and rays alike. Box masses are not needed: certify prices the column
    boxes. rowmask (B, m), 1 by default: a lane's masses are multiplied by it before certify (0: the row is not part of
    that lane's LP, e.g. a cutoff row without an incumbent; its threshold is then never read)."""

    def __init__(self, cg, rowmap, obj_flat, B):
        self.cg, self.B = cg, B
        dt, dev = cg.cop.dtype, cg.cop.device
        self.rowmap = torch.as_tensor(rowmap, dtype=torch.long, device=dev)
        self.obj_flat = int(obj_flat)
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        Fl, Fu, Et, ncol, S, R = cg.Fl, cg.Fu, cg.Et, cg.ncol, cg.S, cg.R
        self.d = Data(lo=z(B, ncol), hi=z(B, ncol), lw=z(B, Fl, 2, Et), lb=z(B, Fl, Et), uw=z(B, Fu, 2, Et),
                      ub=z(B, Fu, Et), thr=z(B, S), h=z(B, R), undefined=torch.zeros(B, dtype=torch.bool, device=dev))
        self.d.lbc, self.d.ubc = z(B, Fl, Et), z(B, Fu, Et)
        self.d.bflat = z(B, cg.nflat)
        self.d.slack = (z(B, Fl, Et), z(B, Fu, Et), z(B, len(cg.rows)), z(B), z(B, R))
        self.d.cbl, self.d.cbu, self.d.cbla, self.d.cbua = z(B, Fl, Et), z(B, Fu, Et), z(B, Fl, Et), z(B, Fu, Et)
        self.rowmask = torch.ones(B, len(self.rowmap), dtype=dt, device=dev)

    def load(self, i, lw, lb, uw, ub, lo, hi, thr, h=None, undefined=None, rowmask=None):
        """lanes [i, i + len) of the raw data: lw (b, Fl, 2, Et), lb (b, Fl, Et), uw, ub, lo, hi (b, ncol), thr (b, S),
        h (b, R) the base rows' right-hand sides, undefined (b,) the boxes without a valid relaxation, rowmask (b, m)"""
        d = self.d
        for buf, v in ((d.lw, lw), (d.lb, lb), (d.uw, uw), (d.ub, ub), (d.lo, lo), (d.hi, hi), (d.thr, thr),
                       (d.h, h), (d.undefined, undefined), (self.rowmask, rowmask)):
            if v is None:
                continue
            v = torch.as_tensor(v, dtype=buf.dtype)
            buf[i:i + v.shape[0]].copy_(v, non_blocking=True)

    def prepare(self):
        """the derived data (faces' constants in column terms, b of every row, the fold's slack and the slack-relaxed
        face b): capturable"""
        cg, d = self.cg, self.d
        d.lbc.copy_(d.lb + (d.lw * cg.cop).sum(2))
        d.ubc.copy_(d.ub + (d.uw * cg.cop).sum(2))
        B = d.lo.shape[0]
        d.bflat.copy_(torch.cat([-d.lbc.reshape(B, -1), d.ubc.reshape(B, -1), d.thr[:, cg.rows] - cg.Ccr, d.h], 1))
        for buf, v in zip(d.slack, cg._fold_slack(d)):
            buf.copy_(v)
        for buf, v in zip((d.cbl, d.cbu, d.cbla, d.cbua), cg._cert_b(d, d.slack)):
            buf.copy_(v)

    def __call__(self, mass, objw):
        cg = self.cg
        B, Qn, m = mass.shape
        mass = mass * self.rowmask[:, None]
        flat = mass.new_zeros(B, Qn, cg.nflat)
        flat.scatter_(2, self.rowmap.expand(B, Qn, m), mass)
        flat[..., self.obj_flat] = objw
        Fl, Fu, Et, nr = cg.Fl, cg.Fu, cg.Et, len(cg.rows)
        y = (flat[..., :Fl * Et].unflatten(-1, (Fl, Et)), flat[..., Fl * Et:(Fl + Fu) * Et].unflatten(-1, (Fu, Et)),
             flat[..., (Fl + Fu) * Et:(Fl + Fu) * Et + nr], flat[..., (Fl + Fu) * Et + nr:])
        return cg.certify(self.d, y, objective=0.0)[0]

    @staticmethod
    def diagonal_qp(rows, q, c, c0, h, lower, upper, mass):
        """Rigorous quadratic certificate for an already enclosed sparse table.

        The graph adapter must enclose its affine folds/rows before using
        this entry point, just as load/prepare do for the compiled layout.
        Search proposals and scaling are never trusted as certificates.
        """
        from nlp_solver.rigorous_dual import quadratic_lower_bound
        return quadratic_lower_bound(rows, q, c, c0, h, lower, upper, mass)[0]


def _mv(A, x):
    """A x for A (..., p, q), x (..., q): an elementwise product and a sum (fusable), not a batched gemv"""
    return (A * x.unsqueeze(-2)).sum(-1)


def _tmv(A, x):
    """A^T x for A (..., q, p), x (..., q)"""
    return (A * x.unsqueeze(-1)).sum(-2)


def _first(mask):
    """index of the first True of each row of mask (B, n) (n where none) and whether any"""
    n = mask.shape[-1]
    idx = torch.arange(n, device=mask.device).expand_as(mask)
    return torch.where(mask, idx, n).amin(-1), mask.any(-1)


Factor = collections.namedtuple('Factor', 'rows slot inv scale T W V LU piv Kinv ok k kcond direct', defaults=(None,))
Factor.__doc__ = """a factored row basis of B' lanes (see the module doc): rows (B', n, n) the basis rows, slot (B', n) the basis
position of each column's T row, scale, T, W, V, the k x k kernel (LU, piv, or its inverse Kinv), ok (B',) (capacity
and a nonsingular K), k (B',); direct (optional B' mask) selects LU on lanes unsafe for explicit inversion."""


class UpdatedRound:
    """Eager rank-q factor maintenance followed by a captured pivot/certificate.

    Rebuilds select a dynamic subset of lanes, so they run outside capture.
    Persistent factor buffers allow the rest of the round to remain captured.
    Each adaptive width owns its buffers for the entire graph lifetime.
    """

    def __init__(self, engine, capture):
        self.engine, self.kap = engine, engine.kap
        F = engine.factor(engine.ids, engine._ar, engine.active & ~engine.closed)
        if F.Kinv is None:
            G = torch.linalg.lu_solve(F.LU, F.piv, engine._eye_k.expand(engine.B, -1, -1))
            F = F._replace(Kinv=G, direct=torch.zeros_like(F.ok))
        else:
            piv = torch.arange(1, engine.kap + 1, dtype=torch.int32, device=engine.dev).expand(engine.B, -1)
            F = F._replace(LU=engine._eye_k.expand(engine.B, -1, -1), piv=piv,
                           direct=torch.zeros_like(F.ok))
        self.factor = type(F)(*(v.clone() if torch.is_tensor(v) else v for v in F))
        engine._round_factor = self.factor
        try:
            self.graph = capture(engine.round)
        finally:
            engine._round_factor = None

    def replay(self):
        e = self.engine
        e.set_kap(self.kap)
        F = e.factor(e.ids, e._ar, e.active & ~e.closed, incremental=True)
        for name, dst, src in zip(F._fields, self.factor, F):
            if torch.is_tensor(dst):
                if src is not None:
                    dst.copy_(src)
                elif name == 'LU':
                    dst.copy_(e._eye_k)
                elif name == 'piv':
                    dst.copy_(torch.arange(1, e.kap + 1, dtype=dst.dtype, device=e.dev))
                elif name == 'direct':
                    dst.zero_()
        return self.graph.replay()


class LockstepDual:
    """B lanes of one table pattern. Static buffers throughout (load writes the inputs; setup, round, finish are
    capturable). Inputs per lane: R (m, n) the table rows, h (m,), c (n,), c0, l, u (n,) the column boxes, dlow, dup
    (n,) the default lower / upper face row of each element (-1 on the inputs), T the cutoff; a parent state (ids (n,)
    row labels, best, y (m,) masses) or none. Pattern: kind (m,) 0 output row, 1 lower face, 2 upper face; levels the
    column ranges [(0, nin), level 1, ...]; certifier: CompiledCertifier on the same lanes. cap: the Schur capacity
    (default min(64, n - 1), the reference's); ring: the seen-basis history (>= the rounds of a call reproduces the
    reference's per-call set). k_update (default False): rank-q inverse maintenance on main rounds; k_rebuild bounds
    the updates between full rebuilds. With capture, factor maintenance selects its rebuild lanes outside the graph;
    the rest of the round stays captured (UpdatedRound). structure/scc select a
    bounded sparse peeling/block backend; subopt selects 32 directions and up
    to eight sequential exchanges. strict_precision gives empty pricing UNKNOWN
    semantics and requests an fp64 replay of fp32 stalls. All default off."""

    def __init__(self, m, n, levels, kind, certifier, B, cap=None, ring=32, device='cpu', dtype=torch.float64,
                 kernel=None, bland='astra', trial='factor', perturb=0.0, dense=True,
                 k_update=False, k_rebuild=32, structure=False, scc=False, subopt=False, strict_precision=False):
        self.m, self.n, self.B = m, n, B
        # perturb > 0: the search's costs are perturbed at every setup so that the start basis is strictly dual
        # feasible, its masses raised by perturb (1 + max |y|) u (u in [1/2, 1), fixed per lane and basis position):
        # dual degenerate ties broken. Only the search sees it: every bound is certified with the true objective
        # (any y >= 0 is a certificate), so the perturbation costs strength, never validity. 0: off (as before).
        self.perturb = float(perturb)
        self.k_update, self.k_rebuild = bool(k_update), int(k_rebuild)
        self.structure, self.scc, self.subopt = bool(structure), bool(scc), bool(subopt)
        self.strict_precision = bool(strict_precision)
        if self.scc and not self.structure:
            raise ValueError('scc requires structure')
        if self.structure and self.k_update:
            raise ValueError('structured bases and Schur updates are separate backends')
        if self.k_rebuild < 1:
            raise ValueError('k_rebuild must be positive')
        self._k_cache = None
        self._round_factor = None
        self.sign_rays = True            # the sign start's output-row rays (the sparse engine: none)
        # the accepted exchange's check: 'factor' (the reference: factor each trial basis, its masses y = -B^-T c
        # >= 0) or 'exchange' (the joint exchange's masses, exact in exact arithmetic since a nonsingular H keeps the
        # basis nonsingular, and the trial basis's Schur capacity; the next round's factor is the safety net)
        assert trial in ('factor', 'exchange')
        self.trial = trial
        # anti-cycling: 'astra' (the reference block mechanism: permanent Bland after n rounds without a lower
        # violation count, or after any round without an accepted exchange); 'fable' (the dual simplex reference: a
        # failed block takes one Bland step, then back to blocks; permanent Bland after n rounds without an increase
        # of the call's dual objective; a failure in Bland is numerical: restart at the CROWN basis, flagged)
        assert bland in ('astra', 'fable')
        self.bland = bland
        # the small kernels (K, H): 'gj' the batched Gauss-Jordan inverse (small_inverse, CUDA only), 'lapack'
        # torch.linalg's LU (cuBLAS batched getrf on CUDA under preferred_linalg_library('cusolver'))
        self.kernel = kernel or ('gj' if torch.device(device).type == 'cuda' else 'lapack')
        self.M = M = m + 2 * n                       # real rows: table, lower box ends (-z <= -l), upper (z <= u)
        self.Mx = Mx = M + n                         # + n artificial slots of phase 1
        self.levels = [tuple(map(int, lv)) for lv in levels]
        self.nin = self.levels[0][1]
        self.cap = min(64, n - 1) if cap is None else cap
        self.kap = max(1, self.cap)                  # the K width (padding: identity)
        self.ring = ring
        self.cert = certifier
        self.dev, self.dt = torch.device(device), dtype
        # the search's dtype dt (fp32 or fp64) and its tolerances; certified quantities (bounds, T, the certificate's
        # masses) in the certificate's dtype cdt (the compiled certify, fp64)
        self.cdt = torch.float64
        self.tol = TOLS32 if dtype == torch.float32 else TOLS64
        dev, dt, cdt = self.dev, dtype, self.cdt
        z = lambda *s: torch.zeros(*s, dtype=dt, device=dev)
        zc = lambda *s: torch.zeros(*s, dtype=cdt, device=dev)
        zl = lambda *s: torch.zeros(*s, dtype=torch.long, device=dev)
        zb = lambda *s: torch.zeros(*s, dtype=torch.bool, device=dev)
        kind = torch.as_tensor(kind, dtype=torch.long, device=dev)
        self.local = torch.cat([(kind == 1) | (kind == 2), torch.ones(2 * n, dtype=torch.bool, device=dev)])
        # inputs
        # (dense False: the sparse engine (lockstep_sparse) keeps no (m x n) table and no (Mx x n) scaled rows)
        self.R = z(B, m, n) if dense else None
        self.h, self.c, self.c0 = z(B, m), z(B, n), z(B)
        self.l, self.u, self.T = z(B, n), z(B, n), zc(B)
        self.dlow, self.dup = zl(B, n), zl(B, n)
        self.active = zb(B)
        self.pids, self.has_pids, self.pbest, self.py, self.has_py = zl(B, n), zb(B), zc(B), zc(B, m), zb(B)
        # the scaled LP and its rows with the artificial slots
        self.A = z(B, Mx, n) if dense else None
        self.b = z(B, Mx)
        self.rs = z(B, M)
        self.cs = z(B, n)
        self.tail = zl(B, Mx)
        self.loc = zb(B, Mx)
        self.init_ids = zl(B, n)
        # the lane state
        self.ids, self.src = zl(B, n), zl(B, n)
        self.aux, self.nart = zb(B), zl(B)
        self.safe, self.hmin, self.bad = zb(B), zl(B), zl(B)
        self.seen, self.nseen = zl(B, ring, n), zl(B)
        self.best, self.besty = zc(B), zc(B, m)
        self.ray, self.closed = zb(B), zb(B)
        self.cbest, self.stall, self.once = zc(B), zl(B), zb(B)        # the call's best certificate, stall, Bland once
        self.want64 = zb(B)                                            # an fp32 lane to redo in fp64 (ill-conditioned
                                                                       # Schur kernel or a numerical failure)
        self.optimal, self.numerical = zb(B), zb(B)                    # no violated row (out of phase 1); a restart
        self.stalled, self.resource = zb(B), zb(B)
        self.shift, self.width = z(B, n), z(B, n)
        self.cs_true = z(B, n)                                         # the unperturbed scaled costs
        self.yfin = zc(B, m)                                           # final_masses: the last basis's masses
        self.xw = z(B, n)                                              # the last basic point (original units)
        self.rayy = zc(B, m)                                           # the first certified Farkas ray (row masses)
        self.rounds_closed = zl(B)                                     # the round of the closing certificate
        self.stats = {k: zl(B) for k in ('rounds', 'phase', 'backoffs', 'fallbacks', 'exchanges', 'maxk', 'exch_rounds',
                                          'cond_band', 'w64_kcond', 'w64_mass', 'w64_stat', 'w64_factor', 'w64_bland')}
        if self.k_update:
            self._k_epoch = zl(1)
            self.stats.update({k: zl(B) for k in ('k_updates', 'k_rebuilds', 'k_residual', 'k_lu')})
        if self.structure or self.subopt or self.strict_precision:
            self.stats.update({k: zl(B) for k in ('direction_rhs', 'positive_steps', 'zero_steps', 'certificates',
                                                'peel_core', 'maxblock', 'dependency_depth', 'resource')})
        self.out_ids = zl(B, n)
        self.kind_out = zl(B)
        self._eye_n = torch.eye(n, dtype=dt, device=dev) if dense else None
        self._eye_k = torch.eye(self.kap, dtype=dt, device=dev)
        self._pos = torch.arange(n, device=dev)
        gen = torch.Generator(device='cpu').manual_seed(0)
        self._pert_u = (0.5 + 0.5 * torch.rand(B, n, generator=gen, dtype=torch.float64)).to(dev, dt)
        self._ar = torch.arange(B, device=dev)
        self._objw_setup = torch.tensor([1.0, 0.0, 1.0], dtype=cdt, device=dev)      # (no host copy under capture)
        self._objw_one = torch.ones(1, dtype=cdt, device=dev)
        self._objw_round = torch.cat([torch.ones(1, dtype=cdt, device=dev), torch.zeros(Q, dtype=cdt, device=dev),
                                      torch.ones(1, dtype=cdt, device=dev)])
        self.graphs = None
        self.debug = None                            # a list: eager rounds append their intermediate tensors

    # ------------------------------------------------------------------------------------------------ inputs
    def load(self, i, R, h, c, c0, l, u, dlow, dup, T, pids=None, pbest=None, py=None):
        """lanes [i, i + b) (numpy or torch); a parent state per lane or none (pids None, or a row of -1)"""
        def put(buf, v):
            v = torch.as_tensor(v)
            buf[i:i + v.shape[0]].copy_(v.to(buf.dtype), non_blocking=True)
        for buf, v in ((self.R, R), (self.h, h), (self.c, c), (self.c0, c0), (self.l, l), (self.u, u),
                       (self.dlow, dlow), (self.dup, dup), (self.T, T)):
            if buf is not None and v is not None:
                put(buf, v)
        b = torch.as_tensor(c).shape[0]
        self.active[i:i + b] = True
        if pids is None:
            self.has_pids[i:i + b] = False
            self.has_py[i:i + b] = False
            self.pbest[i:i + b] = -INF
        else:
            pids = torch.as_tensor(pids)
            put(self.pids, pids.clamp(min=0))
            self.has_pids[i:i + b] = (pids >= 0).all(-1).to(self.dev)
            put(self.pbest, pbest)
            if py is None:                           # (labels and bound only: the basis is refactored here)
                self.has_py[i:i + b] = False
            else:
                put(self.py, py)
                self.has_py[i:i + b] = self.has_pids[i:i + b]

    def deactivate(self, i):
        """lanes [i, B) padding: a copy of lane 0's inputs, masked from the start"""
        if i >= self.B:
            return
        for buf in (self.R, self.h, self.c, self.c0, self.l, self.u, self.dlow, self.dup, self.T, self.pids,
                    self.has_pids, self.pbest, self.py, self.has_py):
            if buf is not None:
                buf[i:] = buf[:1]
        self.active[i:] = False

    # ------------------------------------------------------------------------------------------------ the basis
    def _F_T(self, T, q):
        """T^-1 q, level by level (the forward through the selected faces): q (B', n, k) or (B', n)"""
        v = q.clone()
        for s, e in self.levels[1:]:
            v[:, s:e] = q[:, s:e] - (_mv(T[:, s:e, :s], v[:, :s]) if q.ndim == 2 else T[:, s:e, :s] @ v[:, :s])
        return v

    def _R_T(self, T, q):
        """T^-T q, level by level (the backward through the selected faces)"""
        u = q.clone()
        for s, e in reversed(self.levels[1:]):
            u[:, :s] = u[:, :s] - (_tmv(T[:, s:e, :s], u[:, s:e]) if q.ndim == 2 else
                                   T[:, s:e, :s].transpose(1, 2) @ u[:, s:e])
        return u

    def factor(self, ids, lane, valid=None, incremental=False):
        """Factor of the row bases ids (B', n) of lanes lane (B',) (block_crown.Basis: per column the first basis
        position whose local row has that tail, the unused positions to the missing columns in order, unit scaling,
        W, K = I + V W). valid (B',): lanes to factor (K = I elsewhere)."""
        if self.structure:
            return self._structured_factor(ids, lane, valid)
        n, kap, dev = self.n, self.kap, self.dev
        Bp = ids.shape[0]
        L = lane[:, None]
        rows = self.A[L, ids]                                                   # (B', n, n)
        tl = self.tail[L, ids]
        lc = self.loc[L, ids] & (tl >= 0)
        pos = self._pos.expand(Bp, n)
        tgt = torch.where(lc, tl, n)
        slot_x = torch.full((Bp, n + 1), n, dtype=torch.long, device=dev).scatter_reduce(
            1, tgt, pos, 'amin', include_self=True)
        slot = slot_x[:, :n]
        retained = slot < n
        used = (tgt < n) & (slot_x.gather(1, tgt) == pos)
        missing, extra = ~retained, ~used
        k = missing.sum(1)
        rank_m = missing.cumsum(1) - 1
        rank_e = extra.cumsum(1) - 1
        by_rank = torch.full((Bp, n + 1), 0, dtype=torch.long, device=dev).scatter(
            1, torch.where(extra, rank_e, n), pos)[:, :n]
        fslot = torch.where(retained, slot, by_rank.gather(1, rank_m.clamp(min=0)))
        J = torch.zeros(Bp, kap + 1, dtype=torch.long, device=dev).scatter(
            1, torch.where(missing & (rank_m < kap), rank_m, kap), pos)[:, :kap]
        jv = torch.arange(kap, device=dev)[None] < k[:, None]
        rows_s = rows.gather(1, fslot[:, :, None].expand(Bp, n, n))
        diag = torch.diagonal(rows_s, dim1=1, dim2=2)
        scale = torch.where(retained, 1.0 / torch.where(retained, diag, 1.0), 1.0)
        bar = rows_s * scale[:, :, None]
        T = torch.where(retained[:, :, None], bar, self._eye_n)
        EJ = torch.nn.functional.one_hot(J, n).to(self.dt) * jv[:, :, None]    # (B', kap, n)
        V = (bar.gather(1, J[:, :, None].expand(Bp, kap, n)) - EJ) * jv[:, :, None]
        W = self._F_T(T, EJ.transpose(1, 2))                                    # (B', n, kap)
        if incremental and self.k_update and lane is self._ar:
            ok = k <= min(self.cap, kap)
            if valid is not None:
                ok = ok & valid
            F = Factor(rows, fslot, torch.zeros_like(fslot).scatter(1, fslot, pos), scale, T, W, V,
                       None, None, None, ok, k, torch.zeros(Bp, dtype=self.dt, device=dev))
            return self._update_factor(F, ids, lane, J)
        if getattr(self, 'identity_backbone', False):
            # PlaneCROWN has only box faces in its local backbone: T = I,
            # W = E_J^T. Gather the Schur columns instead of a dense n*k*k
            # product with an explicitly materialized selection matrix.
            K = self._eye_k + V.gather(2, J[:, None].expand(Bp, kap, kap)) * jv[:, None]
        else:
            K = self._eye_k + V @ W
        ok = k <= min(self.cap, self.kap)             # (an adaptive K: the variant's own width)
        if valid is not None:
            ok = ok & valid
        K = torch.where(ok[:, None, None], K, self._eye_k)
        if getattr(self, 'finite_factors', False):
            finite = torch.isfinite(K).all(dim=(1, 2))
            finite &= K.abs().amax(dim=(1, 2)) <= math.sqrt(torch.finfo(K.dtype).max) / kap
            ok = ok & finite
            K = torch.where(finite[:, None, None], K, self._eye_k)
        if self.kernel == 'gj':
            Kinv, good = inverse(K, torch.where(ok, k, 0))
            LU = piv = None
            kcond = K.abs().sum(1).amax(1) * Kinv.abs().sum(1).amax(1)          # cond_1 (the identity padding: 1)
        else:
            LU, piv, info = torch.linalg.lu_factor_ex(K)
            Kinv, good = None, info == 0
            if getattr(self, 'finite_factors', False):
                good &= torch.isfinite(LU).all(dim=(1, 2)) & ((piv > 0) & (piv <= kap)).all(1)
                LU = torch.where(good[:, None, None], LU, self._eye_k)
                piv = torch.where(good[:, None], piv, torch.arange(1, kap+1, device=dev, dtype=piv.dtype))
            kcond = torch.zeros(Bp, dtype=K.dtype, device=dev)
        ok = ok & good
        inv = torch.zeros_like(fslot).scatter(1, fslot, pos)                     # the inverse of the slot map
        return Factor(rows=rows, slot=fslot, inv=inv, scale=scale, T=T, W=W, V=V, LU=LU, piv=piv, Kinv=Kinv, ok=ok,
                      k=k, kcond=torch.where(ok, kcond, 0.0))

    def _basis_rows(self, ids, lane):
        return self.A[lane[:, None], ids]

    def _basis_entries(self, ids, lane):
        rows = self._basis_rows(ids, lane)
        cols = torch.arange(self.n, device=self.dev).expand_as(rows)
        return cols, rows

    def _structured_factor(self, ids, lane, valid):
        from .structured_basis import factor
        cols, vals = self._basis_entries(ids, lane)
        return factor(ids, cols, vals.double(), min(self.cap, self.kap), valid, scc=self.scc)

    def _update_factor(self, F, ids, lane, J):
        """Update K^-1 through a rank-q change of the *row basis* B.

        K^-1 = B^-1[J, P], where J are missing local columns and P their
        extra basis positions (unscaled). Thus B' = B + E_S delta gives
        K'^{-1} = B^{-1}[J', P'] - D[J'] (I + delta D)^{-1}
        (delta B^{-1})[:, P'], D = B^{-1} E_S. Common J/P entries come
        straight from the old inverse. Only new rows/columns of this
        submatrix need additional basis solves. This handles permutations,
        growing/shrinking k and changes of T, including multi-level graphs.

        The small solves cost O(q n k + q k^2); no k^3 work between rebuilds.
        W/V are still assembled by factor(), in the current layout. A fresh
        table, phase 1, excessive changes, a periodic refresh, or failed
        residual probes rebuild only the affected lanes. This orchestration
        is eager; UpdatedRound captures the remainder of a CUDA round.
        Lanes whose inverse/RHS amplification is unsafe retain direct LU.
        Search inverses never certify pruning: CompiledCertifier still does.
        """
        B, n, kap = ids.shape[0], self.n, self.kap
        eye = self._eye_k.expand(B, -1, -1)
        G, age = eye.clone(), torch.zeros(B, dtype=torch.long, device=self.dev)
        epoch = int(self._k_epoch[0])
        cache = self._k_cache
        used = torch.zeros_like(F.ok)
        residual_bad = torch.zeros_like(F.ok)
        if cache is not None and cache[0] == epoch:
            _, old, oldids, oldJ, oldage = cache
            changed = ids != oldids
            count = changed.sum(1)
            can = F.ok & old.ok & (count <= Q) & (oldage < self.k_rebuild) & ~self.aux
            if old.direct is not None:
                can = can & ~old.direct
            # Artificial rows can change their meaning without changing labels.
            can = can & ~(oldids >= self.M).any(1) & ~(ids >= self.M).any(1)
            if bool(can.any()):
                q = max(1, int(torch.where(can, count, 0).max()))
                def pick(mask, count):
                    rank = torch.arange(mask.shape[1], device=self.dev).expand_as(mask)
                    pos = torch.where(mask, rank, mask.shape[1]).topk(count, largest=False).values
                    return pos.clamp(max=mask.shape[1] - 1), pos < mask.shape[1]

                oldkap = oldJ.shape[1]
                ov = torch.arange(oldkap, device=self.dev)[None] < old.k[:, None]
                nv = torch.arange(kap, device=self.dev)[None] < F.k[:, None]
                oldP, P = old.slot.gather(1, oldJ), F.slot.gather(1, J)
                def match(oldpos, newpos):
                    mapping = torch.full((B, n + 1), oldkap, dtype=torch.long, device=self.dev)
                    mapping.scatter_(1, torch.where(ov, oldpos, n),
                                     torch.arange(oldkap, device=self.dev).expand(B, -1))
                    ix = mapping.gather(1, newpos)
                    return ix.clamp(max=oldkap - 1), (ix < oldkap) & nv
                ir, rows_common = match(oldJ, J)
                ic, cols_common = match(oldP, P)
                nr, nc = nv & ~rows_common, nv & ~cols_common
                can = can & (nr.sum(1) <= 2 * Q) & (nc.sum(1) <= 2 * Q)
                nr, nc = nr & can[:, None], nc & can[:, None]
                G0 = old.Kinv.gather(1, ir[:, :, None].expand(-1, -1, oldkap)).gather(
                    2, ic[:, None].expand(-1, kap, -1))
                # New extra basis positions: columns of the old B^-1.
                edge = int(nc.sum(1).max())
                if edge:
                    cp, cm = pick(nc, edge)
                    crhs = torch.nn.functional.one_hot(P.gather(1, cp), n).to(self.dt) * cm[:, :, None]
                    col = self.solve(old, crhs.transpose(1, 2)).gather(1, J[:, :, None].expand(-1, -1, edge))
                    cmap = torch.zeros(B, kap + 1, dtype=torch.long, device=self.dev).scatter(
                        1, torch.where(cm, cp, kap), torch.arange(edge, device=self.dev).expand(B, -1))[:, :kap]
                    G0 = torch.where(nc[:, None], col.gather(2, cmap[:, None].expand(-1, kap, -1)), G0)
                # New missing columns: rows of the old B^-1.
                edge = int(nr.sum(1).max())
                if edge:
                    rp, rm = pick(nr, edge)
                    rrhs = torch.nn.functional.one_hot(J.gather(1, rp), n).to(self.dt) * rm[:, :, None]
                    row = self.solve_t(old, rrhs.transpose(1, 2)).gather(1, P[:, :, None].expand(-1, -1, edge))
                    rmap = torch.zeros(B, kap + 1, dtype=torch.long, device=self.dev).scatter(
                        1, torch.where(rm, rp, kap), torch.arange(edge, device=self.dev).expand(B, -1))[:, :kap]
                    G0 = torch.where(nr[:, :, None], row.transpose(1, 2).gather(
                        1, rmap[:, :, None].expand(-1, -1, kap)), G0)
                pos, on = pick(changed, q)
                on = on & can[:, None]
                erhs = torch.nn.functional.one_hot(pos, n).to(self.dt) * on[:, :, None]
                delta = (self._basis_rows(ids.gather(1, pos), lane) -
                         self._basis_rows(oldids.gather(1, pos), lane)) * on[:, :, None]
                D = self.solve(old, erhs.transpose(1, 2))
                adj = self.solve_t(old, delta.transpose(1, 2))
                H = torch.eye(q, dtype=self.dt, device=self.dev) + delta @ D
                rhs = adj.gather(1, P[:, :, None].expand(-1, -1, q)).transpose(1, 2)
                LU, piv, info = torch.linalg.lu_factor_ex(H)
                G = G0 - D.gather(1, J[:, :, None].expand(-1, -1, q)) @ torch.linalg.lu_solve(LU, piv, rhs)
                G = torch.where(nv[:, :, None] & nv[:, None], G, eye)
                # Probe both solve directions without forming a cubic K @ G.
                # The search also checks stationarity of its actual solves.
                probe = torch.stack([torch.ones(kap, dtype=self.dt, device=self.dev),
                                     ((torch.arange(kap, device=self.dev) % 2) * 2 - 1).to(self.dt)], 1)
                x = G @ probe
                r = x + F.V @ (F.W @ x) - probe
                xt = G.transpose(1, 2) @ probe
                rt = xt + F.W.transpose(1, 2) @ (F.V.transpose(1, 2) @ xt) - probe
                err = torch.maximum(r.abs().amax((1, 2)), rt.abs().amax((1, 2)))
                # Probe the actual primal/dual solves as well. A small kernel
                # residual alone can hide amplification by an ill-conditioned
                # basis, especially on large tables. Refresh before the round
                # can mistake an inaccurate basic point for an optimum.
                trial = F._replace(Kinv=G)
                y = self.solve_t(trial, -self.cs)
                yr = (self._basis_ty(trial, y) + self.cs).abs() / (1 + self.cs.abs())
                brhs = self.b.gather(1, ids)
                xb = self.solve(trial, brhs)
                xr = (self._Ax(xb).gather(1, ids) - brhs).abs() / (1 + brhs.abs())
                err = torch.maximum(err, torch.maximum(yr.amax(1), xr.amax(1)))
                residual_bad = can & (~torch.isfinite(err) | (err > self.tol['STAT'] * .01))
                used = can & (info == 0) & torch.isfinite(G).all((1, 2)) & ~residual_bad
                age = torch.where(used, oldage + 1, age)
        rebuild = F.ok & ~used
        sel = torch.nonzero(rebuild).flatten()
        ok, cond = F.ok.clone(), F.kcond.clone()
        lu_buf = eye.clone()
        piv_buf = torch.arange(1, kap + 1, dtype=torch.int32, device=self.dev).expand(B, -1).clone()
        if sel.numel():
            K = self._eye_k + F.V[sel] @ F.W[sel]
            if self.kernel == 'gj':
                full, good = inverse(K, F.k[sel])
            else:
                LU, piv, info = torch.linalg.lu_factor_ex(K)
                full = torch.linalg.lu_solve(LU, piv, self._eye_k.expand(len(sel), -1, -1))
                good = (info == 0) & torch.isfinite(full).all((1, 2))
                lu_buf[sel], piv_buf[sel] = LU, piv
            G[sel] = full
            ok[sel] = ok[sel] & good
        G = torch.where(ok[:, None, None], G, eye)
        # Inverting an ill-conditioned K explicitly can lose forward accuracy
        # even immediately after a full rebuild. Keep those lanes on LU solves.
        # cond_1 is bounded without forming K. Include the actual RHS scale:
        # cancellation can leave a near-zero solution component even for a
        # large RHS, so a small relative kernel residual is insufficient.
        # Keep two digits of safety relative to the search tolerance.
        norm = 1 + (F.V.abs().sum(1)[:, None] @ F.W.abs()).amax((1, 2))
        condition = norm * G.abs().sum(1).amax(1)
        rhs_scale = torch.maximum(self.cs.abs().amax(1), self.b.gather(1, ids).abs().amax(1)).clamp(min=1)
        risk = condition * rhs_scale
        direct = ok & (F.k > 0) & (~torch.isfinite(risk) | (risk * torch.finfo(self.dt).eps > self.tol['STAT'] * .01))
        need_lu = direct & (~rebuild if self.kernel != 'gj' else True)
        ls = torch.nonzero(need_lu).flatten()
        if ls.numel():
            K = self._eye_k + F.V[ls] @ F.W[ls]
            LU, piv, info = torch.linalg.lu_factor_ex(K)
            lu_buf[ls], piv_buf[ls] = LU, piv
            ok[ls] = ok[ls] & (info == 0)
        rebuild = rebuild | direct
        used = used & ~direct
        age = torch.where(used, age, 0)
        if self.kernel == 'gj':
            cond = condition
        has_direct = bool(direct.any())
        F = F._replace(Kinv=G, LU=lu_buf if has_direct else None, piv=piv_buf if has_direct else None,
                       direct=direct if has_direct else None, ok=ok, kcond=torch.where(ok, cond, 0.))
        self.stats['k_updates'] += used.long()
        self.stats['k_rebuilds'] += rebuild.long()
        self.stats['k_residual'] += residual_bad.long()
        self.stats['k_lu'] += direct.long()
        # ids is the mutable lane state; no cached tensor may alias it.
        oldids = ids.clone()
        saved = F._replace(ids=oldids) if 'ids' in F._fields else F
        self._k_cache = (epoch, saved, oldids, J, age)
        return F

    @staticmethod
    def _ksolve(F, r, adjoint=False):
        """K^-1 r (adjoint: K^-T r) of a factor F, r (B', kap, k) or (B', kap)"""
        if F.Kinv is not None:
            if r.ndim == 2:
                out = _tmv(F.Kinv, r) if adjoint else _mv(F.Kinv, r)
            else:
                out = (F.Kinv.transpose(1, 2) if adjoint else F.Kinv) @ r
            if F.direct is not None:
                rhs = r[..., None] if r.ndim == 2 else r
                if rhs.is_cuda and F.LU.shape[0] == 1:
                    # Captured cuSOLVER getrs with a singleton batch fails on
                    # this runtime. Apply its pivots and triangular factors
                    # directly, only on the opt-in update LU safety path.
                    def permute(v, reverse=False):
                        n = F.LU.shape[-1]
                        pos = torch.arange(n, device=v.device)[None]
                        for i in (range(n-1, -1, -1) if reverse else range(n)):
                            j = F.piv[:, i:i+1].long()-1
                            order = torch.where(pos == i, j, torch.where(pos == j, i, pos))
                            v = v.gather(1, order[..., None].expand(-1, -1, v.shape[-1]))
                        return v
                    if adjoint:
                        lu_t = F.LU.transpose(1, 2)
                        v = torch.linalg.solve_triangular(lu_t, rhs, upper=False)
                        direct = permute(torch.linalg.solve_triangular(lu_t, v, upper=True, unitriangular=True), True)
                    else:
                        v = torch.linalg.solve_triangular(F.LU, permute(rhs), upper=False, unitriangular=True)
                        direct = torch.linalg.solve_triangular(F.LU, v, upper=True)
                else:
                    direct = torch.linalg.lu_solve(F.LU, F.piv, rhs, adjoint=adjoint)
                out = torch.where(F.direct[:, None] if r.ndim == 2 else F.direct[:, None, None],
                                  direct[..., 0] if r.ndim == 2 else direct, out)
            return out
        if r.ndim == 2:
            return torch.linalg.lu_solve(F.LU, F.piv, r[..., None], adjoint=adjoint)[..., 0]
        return torch.linalg.lu_solve(F.LU, F.piv, r, adjoint=adjoint)

    def solve(self, F, rhs):
        """B^-1 rhs: rhs (B', n) or (B', n, k)"""
        if self.structure:
            from .structured_basis import solve
            return solve(F, rhs)
        if rhs.ndim == 2:
            q = rhs.gather(1, F.slot) * F.scale
            v = self._F_T(F.T, q)
            return v - _mv(F.W, self._ksolve(F, _mv(F.V, v)))
        q = rhs.gather(1, F.slot[:, :, None].expand(-1, -1, rhs.shape[2])) * F.scale[:, :, None]
        v = self._F_T(F.T, q)
        return v - F.W @ self._ksolve(F, F.V @ v)

    def solve_t(self, F, rhs):
        """B^-T rhs: rhs (B', n) or (B', n, k)"""
        if self.structure:
            from .structured_basis import solve
            return solve(F, rhs, transpose=True)
        if rhs.ndim == 2:
            q = rhs - _tmv(F.V, self._ksolve(F, _tmv(F.W, rhs), adjoint=True))
            u = self._R_T(F.T, q) * F.scale
            return u.gather(1, F.inv)
        q = rhs - F.V.transpose(1, 2) @ self._ksolve(F, F.W.transpose(1, 2) @ rhs, adjoint=True)
        u = self._R_T(F.T, q) * F.scale[:, :, None]
        return u.gather(1, F.inv[:, :, None].expand(-1, -1, u.shape[2]))

    def masses(self, ids, y, lane=None):
        """the table-row masses (B', m) (original units, >= 0) of the basis ids with dual values y"""
        m, M = self.m, self.M
        L = (self._ar if lane is None else lane)[:, None]
        tab = ids < m
        rs = self.rs[L, ids.clamp(max=M - 1)]
        val = torch.where(tab, y.clamp(min=0) / rs, 0.0)
        return y.new_zeros(ids.shape[0], m + 1).scatter_add(1, torch.where(tab, ids, m), val)[:, :m]

    # ------------------------------------------------------------------------------------------------ setup
    def _sign(self):
        """CROWN's sign backward through the default faces of the objective and of every output row (sign_start):
        Y (B, Q, m) the face masses, the input coefficients q (B, Q, n), and the initial row basis of the objective
        (lp.initial(): each element's chosen face, each input's box end by the sign of its final coefficient)"""
        B, m, n = self.B, self.m, self.n
        ext = self._ext
        specs = torch.cat([self.c[:, None], self.R[:, ext]], 1)                 # (B, 1 + next, n)
        q = specs.clone()
        Y = q.new_zeros(B, specs.shape[1], m)
        init = self._pos.expand(B, n).clone()
        for s, e in reversed(self.levels[1:]):
            A = q[:, :, s:e]
            pos = A >= 0
            mass = A.abs()
            rl, ru = self.dlow[:, s:e], self.dup[:, s:e]
            Y.scatter_add_(2, torch.where(pos, rl[:, None], ru[:, None]), mass)
            Rl = self.R.gather(1, rl[:, :, None].expand(B, e - s, n))
            Ru = self.R.gather(1, ru[:, :, None].expand(B, e - s, n))
            q = q + (mass * pos) @ Rl + (mass * ~pos) @ Ru
            q[:, :, s:e] = 0.0
            init[:, s:e] = torch.where(pos[:, 0], rl, ru)
        q0 = q[:, 0, :self.nin]
        init[:, :self.nin] = torch.where(q0 >= 0, m + self._pos[:self.nin], m + n + self._pos[:self.nin])
        return specs, Y, q, init

    def _multi_cert(self, specs, const, Y):
        """mechanisms/model.multi_cert: rigorous lower bounds (B, Q) of spec . z + const on the face rows' multipliers
        Y (B, Q, m) (only to choose the sign ray's output row, as the reference; the chosen ray is certified apart).
        In the search's dtype: an fp64 search reproduces the reference; an fp32 one evaluates the same enclosure in
        fp32 (its unit roundoff and underflow unit): the choice is a heuristic, the products are (B, Q, m) x (m, n)"""
        cd = self.dt
        U = 2.0 ** -53 if cd == torch.float64 else 2.0 ** -24
        ETA = math.ulp(0.0) if cd == torch.float64 else torch.finfo(torch.float32).tiny   # (a normal fp32: inductor keeps fp32)
        gam = lambda k: k * U / (1 - k * U)
        dn = lambda x: torch.nextafter(x, torch.full_like(x, -INF))
        upp = lambda x: torch.nextafter(x, torch.full_like(x, INF))
        R, h, l, u = (t.to(cd) for t in (self.R, self.h, self.l, self.u))
        specs, const, Y = specs.to(cd), const.to(cd), Y.to(cd).clamp(min=0)
        r = specs + Y @ R
        rho = specs.abs() + Y @ R.abs()
        k = 2 * self.m + 8
        g = gam(k)
        e = upp((g * rho + k * ETA) / (1 - g))
        rl, ru = dn(r - e), upp(r + e)
        l, u = l[:, None], u[:, None]
        terms = torch.minimum(torch.minimum(dn(rl * l), dn(rl * u)), torch.minimum(dn(ru * l), dn(ru * u)))
        hy = (Y @ h[:, :, None])[..., 0]
        hr = (Y @ h.abs()[:, :, None])[..., 0]
        eh = upp((g * hr + k * ETA) / (1 - g))
        base = dn(const - upp(hy + eh))
        k = 2 * self.n + 8
        g = gam(k)
        st = terms.sum(-1)
        er = upp((g * terms.abs().sum(-1) + k * ETA) / (1 - g))
        return dn(base + dn(st - er))

    def setup(self):
        """a new LP epoch for every lane: the scaled LP, CROWN's sign start (bound, output-row ray, initial basis),
        the inherited bound and the parent's multipliers re-certified, the parent's labels repaired (box end flips,
        dual phase 1); capturable"""
        B, m, n, M = self.B, self.m, self.n, self.M
        if self.k_update:
            self._k_epoch.add_(1)
        dt, dev = self.dt, self.dev
        self.cert.prepare()
        # ---- the scaled LP (block_crown.LP): shift, width, row scale
        self._scaled_lp()
        self.cs_true.copy_(self.cs)
        # ---- the sign start
        specs, Y, q, init = self._sign()
        self.init_ids.copy_(init)
        ne = self._ext.numel() if self.sign_rays else 0
        if ne:
            const = torch.cat([self.c0[:, None], -self.h[:, self._ext]], 1)
            v = self._multi_cert(specs, const, Y)
        best_row = v[:, 1:].argmax(1) if ne else torch.zeros(B, dtype=torch.long, device=dev)
        ray = Y[self._ar, 1 + best_row].clone() if ne else Y.new_zeros(B, m)
        if ne:
            ray[self._ar, self._ext[best_row]] += 1.0
        mass = torch.stack([Y[:, 0].to(self.cdt), ray.to(self.cdt), self.py], 1)
        vals = self.cert(mass, self._objw_setup)
        best = torch.where(self.has_pids, self.pbest, torch.full_like(self.pbest, -INF))
        besty = torch.zeros_like(self.besty)
        up0 = vals[:, 0] > best
        best = torch.where(up0, vals[:, 0], best)
        besty = torch.where(up0[:, None], Y[:, 0], besty)
        rayf = (vals[:, 1] > 0) & torch.full_like(self.active, ne > 0)
        up2 = self.has_py & (vals[:, 2] > best)
        best = torch.where(up2, vals[:, 2], best)
        besty = torch.where(up2[:, None], self.py, besty)
        if self.strict_precision:
            besty = torch.where((self.has_py & (vals[:, 2] >= best))[:, None], self.py, besty)
        closed = (best >= self.T) | rayf
        self.best.copy_(best)
        self.besty.copy_(besty)
        self.ray.copy_(rayf)
        self.rayy.copy_(torch.where(rayf[:, None], ray, 0.0))
        self.closed.copy_(closed | ~self.active)
        # ---- start_phase: the parent's labels (else the CROWN basis), box end flips, phase 1 where needed
        go = self.active & ~closed
        ids0 = torch.where(self.has_pids[:, None], self.pids, init)
        F1 = self.factor(ids0, self._ar, go)
        y1 = self.solve_t(F1, -self.cs)
        if self.perturb > 0:
            eps = self.perturb * (1 + y1.abs().amax(1, keepdim=True).nan_to_num(0.0, 0.0, 0.0)) * self._pert_u
            cs_p = self.cs - self._basis_ty(F1, eps)
            self.cs.copy_(torch.where((go & F1.ok)[:, None] & torch.isfinite(cs_p).all(1, keepdim=True), cs_p,
                                      self.cs))
            y1 = self.solve_t(F1, -self.cs)
        flip = (y1 < 0) & (ids0 >= m) & (ids0 < M)
        ids1 = torch.where(flip, m + (ids0 - m + n) % (2 * n), ids0)
        F2 = self.factor(ids1, self._ar, go)
        y2 = self.solve_t(F2, -self.cs)
        neg = y2 < -self.tol['PHASE_NEG'] * (1 + y2.abs().amax(1, keepdim=True))
        fail = go & ~(F1.ok & F2.ok)
        neg = neg & (go & ~fail)[:, None]
        nart = neg.sum(1)
        rank = neg.cumsum(1) - 1
        src = torch.zeros(B, n + 1, dtype=torch.long, device=dev).scatter(1, torch.where(neg, rank, n), ids1)[:, :n]
        ids = torch.where(neg, M + rank, ids1)
        ids = torch.where(fail[:, None], init, ids)
        self.ids.copy_(torch.where(go[:, None], ids, ids0))
        self.src.copy_(src)
        self.nart.copy_(nart)
        self.aux.copy_(nart > 0)
        self._art_rows()
        self.safe.zero_()
        self.cbest.fill_(-INF)
        self.stall.zero_()
        self.once.zero_()
        self.optimal.zero_()
        self.stalled.zero_()
        self.resource.zero_()
        self.numerical.copy_(fail)
        self.want64.copy_(go & (fail | (F1.kcond > self.tol['KCOND']) | (F2.kcond > self.tol['KCOND'])))
        self.xw.fill_(float('nan'))
        self.rounds_closed.zero_()
        self.hmin.fill_(M + 1)
        self.bad.zero_()
        self.nseen.zero_()
        for st in self.stats.values():
            st.zero_()
        self.stats['fallbacks'].copy_(fail.long())
        self.stats['maxk'].copy_(torch.maximum(torch.where(F1.ok, F1.k, 0), torch.where(F2.ok, F2.k, 0)))

    def _scaled_lp(self):
        """the scaled LP of every lane (block_crown.LP): v = shift + width x (compiled_crown.column_scale, A71), rows
        A = R width / rs (rs the row's largest entry), b, their tails and locality; the scaled costs cs"""
        B, n, M = self.B, self.n, self.M
        l, u = self.l, self.u
        shift = (l + u) / 2
        width = column_scale(l, u)                  # (A71: a near-fixed column scales as a fixed one)
        self.shift.copy_(shift)
        self.width.copy_(width)
        eye = self._eye_n.expand(B, n, n)
        orig = torch.cat([self.R, -eye, eye], 1)                                # (B, M, n)
        borig = torch.cat([self.h, -l, u], 1)
        raw = orig * width[:, None]
        rs = raw.abs().amax(2).clamp(min=1e-100)
        A = raw / rs[:, :, None]
        b = (borig - (orig @ shift[:, :, None])[..., 0]) / rs
        self.rs.copy_(rs)
        self.cs.copy_(self.c * width)
        self.A[:, :M].copy_(A)
        self.A[:, M:].zero_()
        self.b[:, :M].copy_(b)
        self.b[:, M:].zero_()
        nz = A != 0
        last = torch.where(nz, self._pos, -1).amax(2)
        self.tail[:, :M].copy_(last)
        self.tail[:, M:] = -1
        self.loc[:, :M].copy_(self.local.expand(B, M))
        self.loc[:, M:] = False

    # the scaled rows' products (the sparse engine, lockstep_sparse, computes them from the compiled form)
    def _Ax(self, x):
        """A x of every row (B, Mx), x (B, n)"""
        return _mv(self.A, x)

    def _absAx(self, xa):
        """|A| xa of every row (B, Mx)"""
        return _mv(self.A.abs(), xa)

    def _rows(self, lab):
        """the rows lab (B, k) as a dense (B, k, n)"""
        return self.A[self._ar[:, None], lab]

    def _basis_ty(self, F, y):
        """A_I^T y (B, n) of the factored basis rows"""
        if self.structure:
            from .structured_basis import product
            return product(F.cols, F.vals, y.to(F.vals.dtype)[..., None], transpose=True)[..., 0].to(y.dtype)
        return _tmv(F.rows, y)

    def kcount(self, ids):
        """the Schur slots k (B,) of the bases ids (B, n): the columns no local basis row has as its tail"""
        B, n = ids.shape
        L = self._ar[:B, None]
        tl = self.tail[L, ids]
        lc = self.loc[L, ids] & (tl >= 0)
        cov = torch.zeros(B, n + 1, dtype=torch.bool, device=self.dev).scatter_(1, torch.where(lc, tl, n), True)
        return n - cov[:, :n].sum(1)

    def set_kap(self, kap):
        """the K width of the next captured round (adaptive K; the identity padding below it)"""
        # (every width keeps its own identity alive, the first one included: a captured graph reads the tensor it
        # was captured with, and a replaced one was freed under the setup / round / finish graphs of the full width)
        eyes = self.__dict__.setdefault('_eyes', {})
        eyes.setdefault(self._eye_k.shape[0], self._eye_k)
        self.kap = max(1, int(kap))
        if self.kap not in eyes:
            eyes[self.kap] = torch.eye(self.kap, dtype=self.dt, device=self.dev)
        self._eye_k = eyes[self.kap]

    def dict_values(self, Y, w):
        """the dictionary's D / Phi (B, K) in the search's dtype: Y (K, m) masses on the table rows, w (K,) objective
        weights (LockstepBounding's pick-then-certify)"""
        r = w[None, :, None] * self.c[:, None] + torch.einsum('km,bmn->bkn', Y, self.R)
        return w[None] * self.c0[:, None] - self.h @ Y.T + torch.minimum(r * self.l[:, None], r * self.u[:, None]).sum(2)

    def _art_rows(self):
        """the artificial rows -a_src (slots i < nart) of the lanes in phase 1"""
        n, M = self.n, self.M
        L = self._ar[:, None]
        on = (self._pos[None] < self.nart[:, None]) & self.aux[:, None]
        self.A[:, M:].copy_(torch.where(on[:, :, None], -self.A[L, self.src], 0.0))
        self.tail[:, M:].copy_(torch.where(on, self.tail[L, self.src], -1))
        self.loc[:, M:].copy_(on & self.loc[L, self.src])

    # ------------------------------------------------------------------------------------------------ one round
    def round(self):
        """one lockstep round of every lane not closed (capturable)"""
        B, m, n, M, Mx = self.B, self.m, self.n, self.M, self.Mx
        dt, dev, ar = self.dt, self.dev, self._ar
        st = self.stats
        live = self.active & ~self.closed
        if self.strict_precision:
            live = live & ~self.stalled & ~self.want64
        aux = self.aux
        ids = self.ids
        st['rounds'] += live.long()
        st['phase'] += (live & aux).long()
        # ---- the round's basis
        F = self._round_factor if self._round_factor is not None else self.factor(
            ids, ar, live, incremental=self.k_update)
        if self.structure:
            for key, value in (('peel_core', F.peel_k), ('maxblock', F.maxblock), ('dependency_depth', F.depth)):
                st[key].copy_(torch.maximum(st[key], torch.where(live, value, 0)))
            res = live & ~F.ok
            self.resource.copy_(self.resource | res)
            st['resource'] += res.long()
        y = self.solve_t(F, -self.cs)
        if self.dt == torch.float32:     # an fp32 search: one step of iterative refinement of the basis's masses
            y = y - self.solve_t(F, self._basis_ty(F, y) + self.cs)
        art_row = torch.cat([torch.zeros(M, dtype=torch.bool, device=dev), torch.ones(n, dtype=torch.bool, device=dev)])
        bL = torch.where(aux[:, None], art_row.to(dt).expand(B, Mx), self.b)
        # the basic point and the cleanup direction of phase 1 (B^-1 e_p, p the first artificial position): one solve
        artp = ids >= M
        p_cl, has_art = _first(artp)
        e_p = torch.nn.functional.one_hot(p_cl.clamp(max=n - 1), n).to(dt)
        xv = self.solve(F, torch.stack([bL.gather(1, ids), e_p], 2))
        x, v = xv[..., 0], xv[..., 1]
        mass0 = self.masses(ids, y)
        ymax = y.abs().amax(1)
        badmass = (y.amin(1) < -self.tol['MASS'] * (1 + ymax)) | ~torch.isfinite(y).all(1)
        stat = (self._basis_ty(F, y) + self.cs).abs() / (1 + self.cs.abs())
        badstat = stat.amax(1) > self.tol['STAT']
        if self.structure:
            from .structured_basis import product
            rhs = bL.gather(1, ids).double()
            bx = product(F.cols, F.vals, x.double()[..., None])[..., 0]
            ax = product(F.cols, F.vals.abs(), x.double().abs()[..., None])[..., 0]
            primal = (bx-rhs).abs() / (1 + rhs.abs() + ax)
            badstat = badstat | ~torch.isfinite(primal).all(1) | (primal.amax(1) > self.tol['STAT'])
        fail = live & (~F.ok | badmass | badstat)
        cert0 = live & F.ok
        go = live & ~fail
        st['maxk'].copy_(torch.maximum(st['maxk'], torch.where(cert0, F.k, 0)))
        # ---- the violated rows
        valid = torch.cat([torch.ones(B, M, dtype=torch.bool, device=dev),
                           (self._pos[None] < self.nart[:, None]) & aux[:, None]], 1)
        inb = torch.zeros(B, Mx, dtype=torch.bool, device=dev).scatter(1, ids, True)
        rho = bL - self._Ax(x)
        norm = 1 + bL.abs() + self._absAx(x.abs())                             # (one read of A)
        vio = (rho < -self.tol['VIO'] * norm) & valid & ~inb
        nvio = vio.sum(1)
        novio = go & (nvio == 0)
        onvio = go & (nvio > 0)
        # ---- phase 1 without violations: one artificial row out (the first), by the largest real rate
        rates = self._Ax(v)[:, :M]
        cc = (rates.abs() > self.tol['RATE']) & ~inb[:, :M]
        newrow = torch.where(cc, rates.abs(), -1.0).argmax(1)
        clean = novio & aux & has_art
        clfail = clean & ~cc.any(1)
        fail = fail | clfail
        # ---- patience, entering rows
        hmin_new = torch.where(onvio & (nvio < self.hmin), nvio, self.hmin)
        bad_new = torch.where(onvio, torch.where(nvio < self.hmin, 0, self.bad + 1), self.bad)
        fable = self.bland == 'fable'
        if fable:      # the call's dual objective stalled for n rounds (through the previous basis: one round late)
            safe = self.safe | (onvio & (self.stall >= n))
            bl = safe | self.once
        else:          # n rounds without a lower violation count
            safe = self.safe | (onvio & (bad_new >= n))
            bl = safe
        if self.subopt:
            from .candidate_steps import candidate_round
            return candidate_round(self, F, y, x, mass0, bL, rho, norm, vio, go, fail, novio, onvio,
                                   clean, clfail, p_cl, newrow, hmin_new, bad_new, safe, bl)
        key = torch.where(vio, rho / norm, INF)
        order = torch.sort(key, dim=1, stable=True).indices[:, :Q]
        okq = torch.gather(vio, 1, order)
        bland, _ = _first(vio)
        cand = torch.where(bl[:, None], torch.cat([bland[:, None].clamp(max=Mx - 1), order[:, 1:]], 1), order)
        candv = torch.where(bl[:, None], torch.arange(Q, device=dev)[None] == 0, okq) & onvio[:, None]
        Ac = self._rows(cand)                                                   # (B, Q, n)
        D = self.solve_t(F, Ac.transpose(1, 2))                                 # (B, n, Q)
        # ---- ratio test per entering row; candidate rays
        Dt = D.transpose(1, 2).contiguous()                                     # (B, Q, n): reductions over n
        posd = Dt > self.tol['PIV']
        haspos = posd.any(2) & candv
        ratio = torch.where(posd, y.clamp(min=0)[:, None] / torch.where(posd, Dt, 1.0), INF)
        theta = ratio.amin(2)
        tied = posd & (ratio <= (theta + self.tol['TIE'] * (1 + theta.abs()))[:, :, None])
        p = torch.where(tied, ids[:, None], 2 ** 62).argmin(2)                 # the tied position of least label
        rayc = candv & ~haspos & ~aux[:, None]
        rmass = self.masses(ids[:, None].expand(B, Q, n).reshape(B * Q, n),
                            (-Dt).reshape(B * Q, n), ar.repeat_interleave(Q)).reshape(B, Q, m)
        jtab = cand < m
        rsj = self.rs.gather(1, cand.clamp(max=M - 1))
        rmass = rmass.scatter_add(2, torch.where(jtab, cand, 0)[:, :, None], torch.where(jtab, 1.0 / rsj, 0.0)[:, :, None])
        # ---- deduplicate leaving positions, compact
        same = (p[:, :, None] == p[:, None, :]) & haspos[:, None, :]
        earlier = torch.tril(torch.ones(Q, Q, dtype=torch.bool, device=dev), -1)
        keep = haspos & ~(same & earlier).any(2)
        corder = torch.sort((~keep).to(torch.int8), dim=1, stable=True).indices
        p0 = keep.sum(1)
        ps, cs = p.gather(1, corder), corder
        js = cand.gather(1, corder)
        # ---- the joint exchanges of sizes p0 >> a, side by side
        sz = (p0[:, None] >> torch.arange(ATTEMPTS, device=dev)[None])                # (B, A)
        AT = ATTEMPTS
        iq = torch.arange(Q, device=dev)
        inz = iq[None, None] < sz[:, :, None]                                    # (B, A, Q)
        Dc = D.gather(2, cs[:, None, :].expand(B, n, Q))                         # (B, n, Q) columns in cs order
        H = Dc.gather(1, ps[:, :, None].expand(B, Q, Q))                         # (B, Q, Q) rows ps, columns cs
        msk = inz[:, :, :, None] & inz[:, :, None, :]                            # (B, A, Q, Q)
        eyeq = torch.eye(Q, dtype=dt, device=dev)
        HA = torch.where(msk, H[:, None], eyeq)                                  # identity padding
        yps = y.gather(1, ps)                                                    # (B, Q)
        rhsA = torch.where(inz, yps[:, None], 0.0)
        HAf = HA.reshape(B * AT, Q, Q)
        if self.kernel == 'gj':
            Hinv, good = inverse(HAf, sz.reshape(-1))
            inf1 = torch.where(good, 0, 1)
        else:
            Hinv, inf1 = torch.linalg.solve_ex(HAf, eyeq.expand(B * AT, Q, Q))
        n1 = lambda Z: torch.where(msk.reshape(B * AT, Q, Q), Z, 0.0).abs().sum(1).amax(1)
        cond = n1(HAf) * n1(Hinv)
        cond = torch.where((inf1 == 0) & torch.isfinite(cond), cond, INF).reshape(B, AT)
        st['cond_band'] += ((cond > self.tol['COND'] / Q) & (cond <= self.tol['COND'] * Q) & (sz > 0) & onvio[:, None]).sum(1)
        t = _mv(Hinv.reshape(B, AT, Q, Q), rhsA)                                  # (B, A, Q), 0 on the padding
        t = torch.where(inz, t, 0.0)
        ny = y[:, None] - _mv(Dc[:, None], t)                                      # (B, A, n)
        psA = torch.where(inz, ps[:, None], n)
        ny = torch.cat([ny, ny.new_zeros(B, AT, 1)], 2).scatter(2, psA, t)[..., :n]
        ni = torch.cat([ids[:, None].expand(B, AT, n), ids.new_zeros(B, AT, 1)], 2).scatter(
            2, psA, torch.where(inz, js[:, None], 0))[..., :n]
        tol = self.tol['JOINT'] * (1 + ymax)
        jfail = ny.amin(2) < -tol[:, None]
        keyA = torch.sort(ni, dim=2).values                                      # (B, A, n)
        hist = torch.arange(self.ring, device=dev)[None] < self.nseen.clamp(max=self.ring)[:, None]
        rep = ((keyA[:, :, None] == self.seen[:, None]).all(3) & hist[:, None]).any(2) & ~safe[:, None]
        pre = onvio[:, None] & (sz > 0) & (cond <= self.tol['COND']) & ~jfail & ~rep
        if self.trial == 'factor':      # the reference's check: factor the trial basis, its masses >= 0
            lanes4 = ar[:, None].expand(B, AT).reshape(-1)
            Ft = self.factor(ni.reshape(B * AT, n), lanes4, pre.reshape(-1))
            yy = self.solve_t(Ft, -self.cs[lanes4]).reshape(B, AT, n)
            k4 = Ft.k.reshape(B, AT)
            okt = Ft.ok.reshape(B, AT)
        else:                           # the exchange's own masses; only the Schur capacity of the trial basis
            tl4 = self.tail[ar[:, None, None], ni]
            lc4 = self.loc[ar[:, None, None], ni] & (tl4 >= 0)
            cov = torch.zeros(B, AT, n + 1, dtype=torch.bool, device=dev).scatter_(   # columns with a local row
                2, torch.where(lc4, tl4, n), True)[..., :n]
            k4 = n - cov.sum(2)
            yy, okt = ny, (torch.ones_like(k4, dtype=torch.bool) if self.structure else k4 <= self.cap)
        yyok = yy.amin(2) >= -self.tol['JOINT'] * (1 + yy.abs().amax(2))
        acc = pre & okt & yyok
        a_star, accepted = _first(acc)
        a_c = a_star.clamp(max=AT - 1)
        tried = torch.where(accepted, a_star, (sz > 0).sum(1))
        ni_acc = ni[ar, a_c]
        yy_acc = yy[ar, a_c]
        mass1 = self.masses(ni_acc, yy_acc)
        # ---- one certificate: the round's bound, the candidate rays, the exchange's bound
        cm = torch.cat([mass0[:, None], rmass, mass1[:, None]], 1).to(self.cdt)
        vals = self.cert(cm, self._objw_round)
        v0 = torch.where(cert0, vals[:, 0], -INF)
        up0 = v0 > self.best
        best = torch.where(up0, v0, self.best)
        besty = torch.where(up0[:, None], mass0, self.besty)
        rayhit = ((vals[:, 1:1 + Q] > 0) & rayc).any(1) & onvio
        closed_mid = (best >= self.T) | self.ray | rayhit
        accept = onvio & accepted & ~closed_mid & (p0 > 0)
        v1 = torch.where(accept, vals[:, 1 + Q], -INF)
        up1 = v1 > best
        best = torch.where(up1, v1, best)
        besty = torch.where(up1[:, None], mass1, besty)
        ray = self.ray | rayhit
        closed = self.closed | (live & ((best >= self.T) | ray))
        rok = (vals[:, 1:1 + Q] > 0) & rayc
        ridx, _ = _first(rok)
        rayy = torch.where((rayhit & ~self.ray)[:, None], rmass[ar, ridx.clamp(max=Q - 1)], self.rayy)
        nacc = onvio & ~accept & ~closed_mid                                    # no exchange accepted
        if fable:      # a failed block: one Bland step next round; a failure in Bland: numerical (restart)
            numfail = nacc & bl
            once = nacc & ~bl
            safe_next = safe
        else:          # any round without an accepted exchange: permanent Bland
            numfail = torch.zeros_like(nacc)
            once = self.once
            safe_next = safe | (onvio & ~accept)
        fail2 = fail | numfail
        prog = (self.cbest == -INF) | (v0 > self.cbest + 1e-10 * (1 + self.cbest.abs()))
        stall = torch.where(fail2, 0, torch.where(go, torch.where(prog, 0, self.stall + 1), self.stall))
        cbest = torch.where(go, torch.maximum(self.cbest, v0), self.cbest)
        if self.debug is not None:
            self.debug.append({k: v.clone() for k, v in dict(ids=ids, y=y, x=x, vio=vio, cand=cand, candv=candv, D=D, p=p,
                                                          ps=ps, js=js, sz=sz, pre=pre, acc=acc, a_star=a_star, cond=cond,
                                                          fail=fail, vals=vals, safe=safe, rho=rho, norm=norm).items()})
        # ---- the state
        st['backoffs'] += torch.where(onvio & ~closed_mid & (p0 > 0), tried, 0)
        st['exchanges'] += torch.where(accept, sz.gather(1, a_c[:, None])[:, 0], 0)
        st['exch_rounds'] += accept.long()
        st['fallbacks'] += fail2.long()
        st['maxk'].copy_(torch.maximum(st['maxk'], torch.where(acc, k4, 0).amax(1)))
        ids_cl = ids.scatter(1, p_cl.clamp(max=n - 1)[:, None], newrow[:, None])
        nids = torch.where(accept[:, None], ni_acc, ids)
        nids = torch.where((clean & ~clfail)[:, None], ids_cl, nids)
        nids = torch.where(fail2[:, None], self.init_ids, nids)
        if self.strict_precision and self.dt == torch.float32:
            nids = torch.where(fail2[:, None], ids, nids)
        left = (nids >= M).any(1)
        naux = torch.where(fail2, False, torch.where(novio & aux, left, aux))
        if self.strict_precision and self.dt == torch.float32:
            naux = torch.where(fail2, aux, naux)
        self.ids.copy_(torch.where(live[:, None], nids, ids))
        self.aux.copy_(torch.where(live, naux, aux))
        self.hmin.copy_(torch.where(live, hmin_new, self.hmin))
        self.bad.copy_(torch.where(live, bad_new, self.bad))
        reset = fail2 if fable else torch.zeros_like(fail2)          # (astra keeps its Bland state over a restart)
        self.safe.copy_(torch.where(live, safe_next & ~reset, self.safe))
        self.once.copy_(torch.where(live, once & ~reset, self.once))
        self.stall.copy_(stall)
        self.cbest.copy_(cbest)
        self.optimal.copy_(torch.where(live, novio & ~aux, self.optimal))
        if self.strict_precision:
            self.optimal.zero_()
            self.stalled.copy_(self.stalled | (novio & ~aux))
            if self.dt == torch.float32:
                self.want64.copy_(self.want64 | (novio & ~aux))
        self.numerical.copy_(self.numerical | (live & fail2))
        self.want64.copy_(self.want64 | (live & (fail2 | (F.kcond > self.tol['KCOND']))))
        st['w64_kcond'] += (live & (F.kcond > self.tol['KCOND'])).long()
        st['w64_mass'] += (live & F.ok & badmass).long()
        st['w64_stat'] += (live & F.ok & badstat).long()
        st['w64_factor'] += (live & ~F.ok).long()
        st['w64_bland'] += (live & numfail).long()
        self.xw.copy_(torch.where((go & ~aux)[:, None], self.shift + self.width * x, self.xw))
        self.rayy.copy_(rayy)
        self.rounds_closed.copy_(torch.where(closed & ~self.closed, st['rounds'], self.rounds_closed))
        slotr = (self.nseen % self.ring)[:, None, None].expand(B, 1, n)
        self.seen.copy_(torch.where(accept[:, None, None], self.seen.scatter(1, slotr, keyA[ar, a_c][:, None]),
                                    self.seen))
        self.nseen.copy_(self.nseen + accept.long())
        self.best.copy_(torch.where(live, best, self.best))
        self.besty.copy_(torch.where(live[:, None], besty, self.besty))
        self.ray.copy_(torch.where(live, ray, self.ray))
        self.closed.copy_(closed)
        self._art_rows()

    def final_masses(self):
        """the table-row masses (yfin, B x m) of every lane's last basis for its own true costs (clipped at 0; 0 for a
        lane in phase 1 or without a factor): an OBBT lane's multipliers of its objective (capturable)"""
        live = self.active & ~self.aux
        F = self.factor(self.ids, self._ar, live)
        y = self.solve_t(F, -self.cs_true)
        mass = self.masses(self.ids, torch.nan_to_num(y, 0.0, 0.0, 0.0)).to(self.cdt)
        self.yfin.copy_(torch.where((live & F.ok)[:, None], mass, 0.0))

    def finish(self):
        """the state labels (the artificial rows mapped back to their sources), the kind (0 open, 1 bound, 2 Farkas);
        with a perturbed search, first the last basis's masses of the true costs certified (clipped at 0): the bound
        the perturbation would otherwise cost"""
        M = self.M
        if self.perturb > 0:
            live = self.active & ~self.closed & ~self.aux
            F = self.factor(self.ids, self._ar, live)
            y = self.solve_t(F, -self.cs_true)
            mass = self.masses(self.ids, torch.nan_to_num(y, 0.0, 0.0, 0.0)).to(self.cdt)
            val = self.cert(mass[:, None], self._objw_one)[:, 0]
            up = live & F.ok & (val > self.best)
            self.best.copy_(torch.where(up, val, self.best))
            self.besty.copy_(torch.where(up[:, None], mass, self.besty))
            self.closed.copy_(self.closed | (up & (self.best >= self.T)))
        srcm = self.src.gather(1, (self.ids - M).clamp(min=0, max=self.n - 1))
        self.out_ids.copy_(torch.where(self.ids >= M, srcm, self.ids))
        self.kind_out.copy_(torch.where(self.ray, 2, torch.where(self.closed & self.active, 1, 0)))

    # ------------------------------------------------------------------------------------------------ driving
    def bind(self, ext):
        """the output rows of the table (kind 0), shared by the lanes"""
        self._ext = torch.as_tensor(ext, dtype=torch.long, device=self.dev)

    def capture(self, fused=False):
        """setup, round and finish each as one CUDA graph on the static buffers; fused: each first traced whole by
        dynamo and fused by inductor (compiled_crown.fuse)"""
        wrap = fuse if fused else (lambda f: f)
        if self.k_update:
            self.graphs = {'setup': Captured(wrap(self.setup)),
                           'round': UpdatedRound(self, lambda fn: Captured(wrap(fn))),
                           'finish': Captured(wrap(self.finish))}
        else:
            self.graphs = {name: Captured(wrap(fn)) for name, fn in (('setup', self.setup), ('round', self.round),
                                                                       ('finish', self.finish))}

    def run(self, R, eager=None):
        """a new epoch (setup), R rounds, finish; on the captured graphs when captured (eager: force eager)"""
        g = self.graphs if (eager is not True and self.graphs is not None) else None
        (g['setup'].replay if g else self.setup)()
        for _ in range(R):
            (g['round'].replay if g else self.round)()
        (g['finish'].replay if g else self.finish)()

    def resume(self, R, eager=None):
        """R more rounds on the current state, finish"""
        g = self.graphs if (eager is not True and self.graphs is not None) else None
        for _ in range(R):
            (g['round'].replay if g else self.round)()
        (g['finish'].replay if g else self.finish)()
