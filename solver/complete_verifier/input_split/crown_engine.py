"""bound_engine: crown_lockstep -- the BaB's bounding as one compiled CROWN (crown_absorb/ARCH_ONE_TREE.md). Opt-in
(solver.nlp.bound_engine); with the default 'gear' nothing here runs.

Built once at the root from the root box, GEAR's spec C, its relaxed thresholds, base rows and rational rows: the
extended graph (nlp_solver.polynomial_graph: links, reduced RLT, cleared denominators) compiled by
auto_LiRPA.compiled_crown (sparse, A68 margins: its own bounds and the BoundMul faces hold for the exact graph) with
the spec rows [objective, objective as a row (the certificate's objective slot), the cutoff c v <= T, GEAR's constraint
rows of finite threshold, the derived rows]. One bounding call on D domains, in chunks of the widest lane bucket:
  1. compiled CROWN on the boxes: intermediate bounds, faces, node boxes, the rigorous bound of every row;
  2. the dense lockstep tables of those faces and rows (the search's copy) and the certificate's data (fp64);
  3. auto_LiRPA.lockstep_bounding: the dictionary's certificates, R rounds of block pivots per lane in fp32 (the
     columns scaled by compiled_crown.column_scale, A71), an fp64 redo of the lanes the fp32 search flags; every
     prune is decided by the fp64 rigorous certificate (compiled CROWN certify, its fold slack and A68 margins);
  4. back to GEAR: row 0 the certified bound (+inf on every row for a Farkas certificate), the other rows compiled
     CROWN's rigorous row bounds; CROWN's input planes moved so that their minimum on the box is the rigorous bound
     (GEAR's branching and clipping); the domains' new states (basis labels, best bound, multipliers) as
     history.engine_state, which the domain list stores with the domains and hands to their children.
The cutoff T = the incumbent less GEAR's gap allowance (the smallest T the gap rule still closes against the
incumbent); GEAR then prunes against T, so the bound it reports for a discarded box is T. Without an incumbent the
cutoff row is a vacuous row of the search and is masked out of the certificate. Not in the certificate's scope: a unary
operator's own tangent and secant faces (computed from rigorous bounds by its code; their rounding and libm's).

Also here: the CPU LP call counter (install_lp_counter), every call of scipy's linprog and highspy's run counted,
attributed to the primal side (root LP start point, lp_alternation) or else to the dual side."""
import atexit
import collections
import json
import math
import os
import sys
import time
import traceback

import numpy as np
import torch

import arguments
from auto_LiRPA.cuda_memory import MemoryBudgetExceeded
from auto_LiRPA.utils import select_rows

INF = float('inf')


class EngineUnavailable(Exception):
    """the engine does not apply to this problem (the reason is printed; GEAR's bounding runs instead)"""


def gap_cutoff(P, abs_tol, rel_tol):
    """the smallest T <= P that GEAR's gap rule (solving_utils.is_primal_dual_converged) closes against P"""
    from solving_utils import is_primal_dual_converged
    if not math.isfinite(P):
        return P
    T = P - max(abs_tol, rel_tol * max(1.0, abs(P)))
    conv = lambda t: is_primal_dual_converged(torch.tensor([P], dtype=torch.float64),
                                              torch.tensor([t], dtype=torch.float64), abs_tol=abs_tol, rel_tol=rel_tol)
    while not conv(T):
        T = math.nextafter(T, INF)
    return T


class FBBT:
    """Feasibility-based bound tightening on the compiled form of a CompiledGraph (sparse), batched over boxes, fixed
    sweeps, every step rounded outward. Linear rows over the compiled columns: the output rows (the cutoff c v <= T,
    GEAR's constraint rows, the derived rows; b = threshold - C c_out + the fold's slack) and the base rows (b = h);
    the operand forms of the BoundMul elements (op_p = M_p v + c_p, their fold error added). A sweep: (1) every
    linear row backward onto every column it holds (a_j v_j <= b - sum_{k != j} min a_k v_k); (2) every BoundMul
    element backward (op_0 in z / op_1 where op_1's interval excludes 0, and the other way), each operand interval
    then a pair of linear rows backward onto the columns of the levels below; (3) forward, the elements' intervals
    from their operands' (corner products). With crown_engine_propagate, square and pole-free reciprocal columns
    also propagate in both directions. Other unary columns keep their bound() interval. Every bound is intersected with the current one, so it only ever
    shrinks; a column with lower > upper proves the box holds no point that satisfies the rows (with the cutoff: no
    point better than T). The arithmetic: sums with their gamma_k error bound of the magnitudes, quotients and
    products one step outward."""

    def __init__(self, cg, row_ids, mul):
        self.cg, dev = cg, cg.cop.device
        self.n, self.Et, self.ncol = cg.n, cg.Et, cg.ncol
        r, c, v = (t.to(dev) for t in cg._cm())
        keep = torch.isin(r, row_ids)
        pos = torch.full((cg.S,), -1, dtype=torch.long, device=dev)
        pos[row_ids] = torch.arange(len(row_ids), device=dev)
        self.out = (pos[r[keep]], c[keep], v[keep].double())             # (row in row_ids order, column, value)
        self.row_ids = row_ids
        E = cg.E
        er, ec = torch.nonzero(E, as_tuple=True)
        self.base = (er, ec, E[er, ec].double())
        orr, oc, ov = (t.to(dev) for t in cg._op())
        self.op = (orr, oc, ov.double())                                 # rows p Et + e
        self.mul = mul

    @staticmethod
    def _down(x):
        return torch.nextafter(x, torch.full_like(x, -INF))

    @staticmethod
    def _up(x):
        return torch.nextafter(x, torch.full_like(x, INF))

    def _activity(self, lo, hi, coo, nrows):
        """per row the lower activity sum_j min(a_j l_j, a_j u_j) (finite part), its magnitude, the count of its
        infinite terms, and the per-entry term"""
        r, c, a = coo
        B = lo.shape[0]
        term = torch.where(a > 0, a * lo[:, c], a * hi[:, c])            # (B, nnz)
        inf = ~torch.isfinite(term)
        t = torch.where(inf, 0.0, term)
        z = lambda: torch.zeros(B, nrows, dtype=torch.float64, device=lo.device)
        S, M, N = z().index_add_(1, r, t), z().index_add_(1, r, t.abs()), z().index_add_(1, r, inf.double())
        return S, M, N, t, inf

    def _backward(self, lo, hi, coo, b, nrows):
        """one backward pass of the rows sum_j a_j v_j <= b_r (b (B, nrows), +inf: no row): the columns' new
        bounds (B, ncol)"""
        r, c, a = coo
        B = lo.shape[0]
        Lnew = torch.full((B, self.ncol), -INF, dtype=torch.float64, device=lo.device)
        Unew = torch.full((B, self.ncol), INF, dtype=torch.float64, device=lo.device)
        if not r.numel():
            return Lnew, Unew
        S, M, N, t, inf = self._activity(lo, hi, coo, nrows)
        cnt = torch.zeros(nrows, dtype=torch.float64, device=lo.device).index_add_(0, r, torch.ones_like(a))
        br = b[:, r]
        R = br - (S[:, r] - t)
        k = cnt[r] + 4                                                   # (gamma_k of the row's terms, + 4)
        err = k * 2.0 ** -53 / (1 - k * 2.0 ** -53) * (br.abs() + M[:, r] + t.abs()) + 2.0 ** -1000
        Rup = self._up(R + err)
        ok = (N[:, r] - inf.double() == 0) & torch.isfinite(br)
        q = Rup / a
        Unew.scatter_reduce_(1, c.expand(B, -1), torch.where(ok & (a > 0), self._up(q), INF), 'amin')
        Lnew.scatter_reduce_(1, c.expand(B, -1), torch.where(ok & (a < 0), self._down(q), -INF), 'amax')
        return Lnew, Unew

    def _operands(self, lo, hi, err):
        """the operand intervals (B, 2 Et) of every element from the column box (fold error err (B, 2 Et))"""
        r, c, a = self.op
        n2 = 2 * self.Et
        S, M, N, t, inf = self._activity(lo, hi, self.op, n2)
        neg = (r, c, -a)
        S2, M2, N2, _, _ = self._activity(lo, hi, neg, n2)
        cst = self.cg.cop.reshape(-1).double()
        cnt = torch.zeros(n2, dtype=torch.float64, device=lo.device).index_add_(0, r, torch.ones_like(a))
        g = (cnt + 4) * 2.0 ** -53 / (1 - (cnt + 4) * 2.0 ** -53)
        olo = torch.where(N > 0, -INF, self._down(S + cst - g * (M + cst.abs()) - err - 2.0 ** -1000))
        ohi = torch.where(N2 > 0, INF, self._up(-S2 + cst + g * (M2 + cst.abs()) + err + 2.0 ** -1000))
        return olo, ohi

    @staticmethod
    def _corners(al, au, bl, bu, up):
        c = torch.stack([al * bl, al * bu, au * bl, au * bu]).nan_to_num(nan=0.0)
        return torch.nextafter(c.amax(0), torch.full_like(al, INF)) if up else \
            torch.nextafter(c.amin(0), torch.full_like(al, -INF))

    def run(self, lo, hi, b_out, h_base, sweeps):
        """lo, hi (B, ncol) the column box; b_out (B, len(row_ids)) the output rows' right-hand sides (with the fold
        slack; +inf: no row), h_base (B, R) the base rows'. Returns the tightened box and empty (B,)"""
        n, Et = self.n, self.Et
        lo, hi = lo.clone(), hi.clone()
        e_idx = torch.arange(Et, device=lo.device)
        mul = self.mul
        for _ in range(sweeps):
            vmax = torch.maximum(lo.abs(), hi.abs())
            fin = torch.isfinite(vmax)
            cg = self.cg
            t = cg.LdG.right(torch.where(fin, vmax, 0.0)) + cg.dg
            bad = cg.LdGi.right((~fin).double())
            t = torch.where(bad > 0, INF, t * (1 + 2.0 ** -40) + 2.0 ** -1000)
            op_err = t[:, :2 * Et]
            row_err = t[:, 2 * Et:][:, self.row_ids]
            # (1) the linear rows
            for coo, b, nr in ((self.out, b_out + row_err, len(self.row_ids)),
                               (self.base, h_base, h_base.shape[1])):
                L, U = self._backward(lo, hi, coo, b, nr)
                lo, hi = torch.maximum(lo, L), torch.minimum(hi, U)
            # (2) the BoundMul elements backward, (3) forward
            olo, ohi = self._operands(lo, hi, op_err)
            zl, zu = lo[:, n:], hi[:, n:]
            al, au, bl, bu = olo[:, :Et], ohi[:, :Et], olo[:, Et:], ohi[:, Et:]
            def quot(zl, zu, yl, yu):
                ok = mul & ((yl > 0) | (yu < 0)) & torch.isfinite(zl) & torch.isfinite(zu) & torch.isfinite(yl) & \
                    torch.isfinite(yu)
                c = torch.stack([zl / yl, zl / yu, zu / yl, zu / yu])
                return (torch.where(ok, self._down(c.amin(0)), -INF), torch.where(ok, self._up(c.amax(0)), INF))
            ql0, qu0 = quot(zl, zu, bl, bu)
            ql1, qu1 = quot(zl, zu, al, au)
            sq = getattr(self, 'sq', None)
            if sq is not None:
                # z=a^2: z<=zu gives |a|<=sqrt(zu); z>=zl gives
                # a one-sided lower magnitude only when a's sign is known.
                rad_u = self._up(torch.sqrt(zu.clamp(min=0)))
                rad_l = self._down(torch.sqrt(zl.clamp(min=0))).clamp(min=0)
                ql0 = torch.where(sq, torch.where(al >= 0, rad_l, -rad_u), ql0)
                qu0 = torch.where(sq, torch.where(au <= 0, -rad_l, rad_u), qu0)
            recip = getattr(self, 'recip', None)
            if recip is not None:
                pole_free = recip & ((zl > 0) | (zu < 0))
                ql0 = torch.where(pole_free, self._down(1/zu), ql0)
                qu0 = torch.where(pole_free, self._up(1/zl), qu0)
            nl = torch.cat([torch.maximum(al, ql0), torch.maximum(bl, ql1)], 1)
            nu = torch.cat([torch.minimum(au, qu0), torch.minimum(bu, qu1)], 1)
            # operand intervals as rows M v <= nu - c + err, -M v <= -nl + c + err
            cst = cg.cop.reshape(-1).double()
            r, c, a = self.op
            both = (torch.cat([r, r + 2 * Et]), torch.cat([c, c]), torch.cat([a, -a]))
            b = torch.cat([self._up(nu - cst + op_err), self._up(-nl + cst + op_err)], 1)
            L, U = self._backward(lo, hi, both, torch.where(torch.isfinite(b), b, INF), 4 * Et)
            lo, hi = torch.maximum(lo, L), torch.minimum(hi, U)
            olo, ohi = self._operands(lo, hi, op_err)
            fz_l = self._corners(olo[:, :Et], ohi[:, :Et], olo[:, Et:], ohi[:, Et:], False)
            fz_u = self._corners(olo[:, :Et], ohi[:, :Et], olo[:, Et:], ohi[:, Et:], True)
            lo[:, n:] = torch.where(mul, torch.maximum(lo[:, n:], fz_l), lo[:, n:])
            hi[:, n:] = torch.where(mul, torch.minimum(hi[:, n:], fz_u), hi[:, n:])
            if sq is not None:
                al, au = olo[:, :Et], ohi[:, :Et]
                near = torch.where(al > 0, al, torch.where(au < 0, au, 0.0))
                sl = self._down(near.square()).clamp(min=0)
                su = self._up(torch.maximum(al.square(), au.square()))
                lo[:, n:] = torch.where(sq, torch.maximum(lo[:, n:], sl), lo[:, n:])
                hi[:, n:] = torch.where(sq, torch.minimum(hi[:, n:], su), hi[:, n:])
            if recip is not None:
                al, au = olo[:, :Et], ohi[:, :Et]
                pole_free = recip & ((al > 0) | (au < 0))
                lo[:, n:] = torch.where(pole_free, torch.maximum(lo[:, n:], self._down(1/au)), lo[:, n:])
                hi[:, n:] = torch.where(pole_free, torch.minimum(hi[:, n:], self._up(1/al)), hi[:, n:])
        empty = (lo > hi).any(1)
        return lo, hi, empty



def table_precheck(net, n, n_spec, n_base):
    """Raise EngineUnavailable, before CompiledGraph's expensive build, when the GraphQP table would exceed its fixed
    capacity. Lower bounds only (never skips a table that fits): the columns are exactly n + the face nodes'
    elements (CompiledGraph.ncol) less at most the BoundSqr elements GraphQP can eliminate; the rows at least one
    per finite spec row and base row. The nonzeros are not bounded."""
    from auto_LiRPA.compiled_crown import face_nodes
    from auto_LiRPA.curvature_qp import check_capacity
    names = face_nodes(net)
    size = lambda k: math.prod(net[k].output_shape[1:])
    ncol = n + sum(size(k) for k in names)
    squares = sum(size(k) for k in names if type(net[k]).__name__ == 'BoundSqr')
    try:
        check_capacity('graph_qp', n_spec + n_base, ncol - squares, 0, 'skip CompiledGraph build, M1/M2 off')
    except ValueError as e:
        raise EngineUnavailable(str(e)) from None


class CrownEngine:
    """see the module doc. net: GEAR's BoundedModule; C (S, n_out) its spec (row 0 the objective); thr (S,) the relaxed
    thresholds; x_L, x_U (1, n) the root box; E (R, n), h (R,) the base rows; rational_rows: their rational forms."""

    tiered = False

    def __init__(self, net, C, thr, x_L, x_U, E=None, h=None, rational_rows=None, cfg=None, tangent=None,
                 report=True, seed=None):
        from auto_LiRPA.cuda_memory import CudaMemoryBudget
        from contextlib import nullcontext
        cfg = arguments.Config['solver']['nlp'] if cfg is None else cfg
        if cfg.get('crown_engine_m1', False):
            from input_split.m1_qp import configure
            cfg = dict(cfg)
            configure(cfg)
        if cfg.get('crown_engine_absorb', False):
            from input_split.crown_absorb import configure
            cfg = dict(cfg)
            configure(cfg)
        reserve = float(arguments.Config['bab'].get('domain_storage_gpu_mb', 0)) * 2**20
        if cfg.get('crown_engine_obbt_dsx', False) or cfg.get('crown_engine_node_dsx', False):
            # No tree storage exists during root OBBT. engine_main2's 16 GiB
            # future storage ceiling otherwise excludes every V100 root LP.
            # Admission still protects current driver/cache headroom;
            # subsequent calls recheck actual residency as the tree grows.
            reserve = 0
        self.memory = (CudaMemoryBudget(x_L.device, float(cfg.get('crown_engine_mem_gb', 8.)) * 2**30, reserve,
                                        owner=(lambda: self) if cfg.get('crown_engine_m1', False) else None)
                       if x_L.device.type == 'cuda' else None)
        with self.memory.allocations() if self.memory is not None else nullcontext():
            try:
                self._initialize(net, C, thr, x_L, x_U, E, h, rational_rows, cfg, tangent, report, seed)
            except (ValueError, NotImplementedError) as error:
                # Strict M1 domains/operators are an admission condition. The
                # native GEAR bound remains available for a rejected root.
                if cfg.get('crown_engine_m1', False) and str(error).startswith('strict '):
                    raise EngineUnavailable(str(error)) from None
                raise

    def _initialize(self, net, C, thr, x_L, x_U, E=None, h=None, rational_rows=None, cfg=None, tangent=None,
                    report=True, seed=None):
        from nlp_solver.polynomial_graph import PolynomialGraph
        from auto_LiRPA.compiled_crown import CompiledGraph
        from auto_LiRPA.lockstep_bounding import LockstepBounding
        from auto_LiRPA.lockstep_dual import CompiledCertifier
        cfg = arguments.Config['solver']['nlp'] if cfg is None else cfg
        dsx_obbt = bool(cfg.get('crown_engine_obbt_dsx', False) or cfg.get('crown_engine_node_dsx', False))
        if cfg.get('crown_engine_m1', False):
            from input_split.m1_qp import M1CompiledGraph
            CompiledGraph = M1CompiledGraph
        if dsx_obbt:
            from auto_LiRPA.obbt_unary import OutwardCompiledGraph
            CompiledGraph = OutwardCompiledGraph
            if cfg.get('crown_engine_absorb', False):
                from input_split.crown_absorb import AbsorbCompiledGraph
                CompiledGraph = AbsorbCompiledGraph
            cfg = dict(cfg, crown_engine_squares='true', crown_engine_mixed_squares='true',
                       crown_engine_propagate='true', crown_engine_obbt_nodes='true',
                       crown_engine_ibp='true')
        t0 = time.time()
        dev = x_L.device
        self.dev = dev
        if self.memory is not None:
            self.memory.require(3 * C.numel() * C.element_size())
        lo, hi = x_L.reshape(1, -1).double(), x_U.reshape(1, -1).double()
        self.root_lo, self.root_hi = lo.clone(), hi.clone()
        self.Sg = C.shape[0]
        self.abs_tol = float(cfg.get('bab_primal_dual_abs_tol', 1e-5))
        self.rel_tol = float(cfg.get('bab_primal_dual_rel_tol', 1e-4))
        Eh = None if E is None else (E.double().to(dev), h.double().to(dev))
        curvature_names = ()
        if cfg.get('crown_engine_m1', False):
            from auto_LiRPA.compiled_crown import face_nodes
            if not face_nodes(net):
                raise EngineUnavailable('M1 has no nonlinear face nodes; native linear bounds apply')
            # The exact symbolic polynomial walker can return UNKNOWN for
            # an affine output operator that the compiled fold supports.
            # Inspect the original graph before adding optional link rows;
            # their duplicate monomials must not create artificial coupling.
            original = CompiledGraph(net, lo, hi, C.to(lo), thr.to(lo),
                                     *(None, None) if Eh is None else Eh, sparse=True)
            rr, cc, vv = original._cm()
            finite_rows = torch.isfinite(thr.reshape(-1).to(dev))
            used = set(original._op()[1].cpu().tolist())
            used.update(cc[(rr > 0) & finite_rows[rr] & (vv != 0)].cpu().tolist())
            curvature_names = [(nd.name, k) for nd in original.nodes
                if type(net[nd.name]).__name__ == 'BoundSqr'
                for k in range(nd.E) if original.CM0[original.n+nd.s+k] > 0
                and original.n+nd.s+k not in used]
            del original
        self.pg = pg = PolynomialGraph(net, C, thr, lo.cpu(), hi.cpu(), *(None, None) if Eh is None else
                                       (Eh[0].cpu(), Eh[1].cpu()), rational_rows=rational_rows,
                                       squares=str(cfg.get('crown_engine_squares', 'false')) == 'true',
                                       mixed_squares=str(cfg.get('crown_engine_mixed_squares', 'false')) == 'true',
                                       curvature_squares=bool(cfg.get('crown_engine_m1', False)),
                                       curvature_leaf_names=curvature_names)
        # the objective's tangent row (GEAR's quadratic / convex tangent, frozen at its root anchor): every point
        # with objective <= T satisfies g x <= T - b(box), b(box) the tangent plane's offset on the box (a base row
        # of per-box right-hand side, last of the base rows)
        self.tangent = tangent
        if tangent is not None:
            g = tangent.g.reshape(1, -1).to(device=dev, dtype=torch.float64)
            Eh = (g, torch.zeros(1, dtype=torch.float64, device=dev)) if Eh is None else \
                (torch.cat([Eh[0], g]), torch.cat([Eh[1], torch.zeros(1, dtype=torch.float64, device=dev)]))
        te = pg.thresholds(lo, hi)[0]
        keep = [k for k in range(1, te.shape[0]) if math.isfinite(float(te[k]))]
        self.keep = torch.tensor(keep, dtype=torch.long, device=dev)
        Cx = pg.C.to(device=dev, dtype=torch.float64)
        Cc = torch.cat([Cx[:1], Cx[:1], Cx[:1], Cx[keep]])
        self.S = S = Cc.shape[0]
        # GEAR row k (>= 1) <- certificate row 3 + j (keep[j] == k); -1: none (an infinite threshold)
        self.grow = torch.full((self.Sg,), -1, dtype=torch.long, device=dev)
        self.grow[0] = 0
        for j, k in enumerate(keep):
            if k < self.Sg:
                self.grow[k] = 3 + j
        graph_hit = (seed is not None and torch.equal(pg.C, seed.pg.C)
                     and pg.mono_new == seed.pg.mono_new and pg.squares_rewritten == seed.pg.squares_rewritten
                     and [(r.q, r.q0, r.tag, r.rlt) for r in pg.rows]
                         == [(r.q, r.q0, r.tag, r.rlt) for r in seed.pg.rows]
                     and torch.equal(self.keep, seed.keep))
        if graph_hit:
            self.pg = pg = seed.pg
            self.cg = cg = seed.cg.with_base_rows(*(None, None) if Eh is None else Eh)
            cg.memory = self.memory
            print('[crown-engine-cache] compiled graph reused; base-row shape updated', flush=True)
        else:
            if cfg.get('crown_engine_m1', False):
                table_precheck(pg.net, lo.shape[1], len(keep), 0 if Eh is None else Eh[0].shape[0])
            self.cg = cg = CompiledGraph(pg.net, lo, hi, Cc, self._thr(lo, hi, torch.full((1,), INF, device=dev)),
                                         *(None, None) if Eh is None else Eh, rows=list(range(1, S)), sparse=True,
                                         free_tangent=bool(cfg.get('crown_engine_tiered', False)),
                                         trig_tangents=int(cfg.get('crown_engine_trig_tangents', 0)), memory=self.memory)
        self.tiered = bool(cfg.get('crown_engine_tiered', False))
        self.cfg = dict(cfg)
        self.budget_estimates = {}
        self.root_seed = None
        self.cg.diagnostics = self.tiered
        if cg.trig_tangents:
            self._trig_ranges(Cc, self._thr(lo, hi, lo.new_tensor([INF]))[0], E, h)
        # every level's operand bounds also from the interval of their forms over the lower levels' node boxes
        # (compiled_crown bound()'s ibp_operands; the faces rebuilt level by level from the tighter one)
        cg.ibp_operands = str(cfg.get('crown_engine_ibp', 'false')) == 'true'
        self.hb = None if Eh is None else Eh[1]
        n, Et, ncol, Fl, Fu, R = cg.n, cg.Et, cg.ncol, cg.Fl, cg.Fu, cg.R
        # the BoundMul elements (two operands): FBBT's and the mcc rule's
        self.mul = torch.zeros(Et, dtype=torch.bool, device=dev)
        for nd in cg.nodes:
            if nd.P == 2:
                self.mul[nd.s:nd.s + nd.E] = True
        sweeps = int(cfg.get('crown_engine_fbbt', 0))
        self.fbbt = FBBT(cg, torch.arange(2, S, device=dev), self.mul) if sweeps > 0 else None
        self.fbbt_sweeps = sweeps
        ar = lambda k: torch.arange(k, device=dev)
        fl = ar(Fl * Et)[cg.vl.flatten()]
        fu = Fl * Et + ar(Fu * Et)[cg.vu.flatten()]
        base, nr = (Fl + Fu) * Et, len(cg.rows)
        fo, fb = base + 1 + ar(nr - 1), base + nr + ar(R)                # (the objective's slot `base` not a row)
        self.sel = torch.cat([fl, fu, fo, fb])
        self.m = m = len(self.sel)
        kind = torch.cat([torch.ones(len(fl)), 2 * torch.ones(len(fu)), torch.zeros(len(fo) + len(fb))]).long()
        self.cut_row = len(fl) + len(fu)
        self.inv = torch.full((cg.nflat,), -1, dtype=torch.long, device=dev)
        self.inv[self.sel] = ar(m)
        self.n, self.ncol, self.Et, self.Fl, self.Fu = n, ncol, Et, Fl, Fu
        if cfg.get('crown_engine_m1', False):
            from input_split.m1_qp import initialize
            self.sparse = True
            initialize(self, t0, report)
            return
        # memory: the dense table and the lockstep buffers per lane (fp32 search, fp64 redo), the table build's
        # fp64 temporaries; the widest bucket the budget allows
        self.sparse = str(cfg.get('crown_engine_sparse', 'false')) == 'true'
        capc = int(cfg.get('crown_engine_cap', 0))
        cap = min(64 if capc <= 0 else capc, ncol - 1)
        # If the entire legacy kernel fits, there is no capacity problem for
        # structured bases/suboptimization to solve. Preserve the off path,
        # including its memory buckets and precision/probe scheduling.
        requested = any(str(cfg.get('crown_engine_' + key, 'false')) == 'true'
                        for key in ('structure', 'subopt'))
        if requested and cap == ncol - 1:
            cfg = dict(cfg)
            for key in ('structure', 'scc', 'subopt', 'strict_precision'):
                cfg['crown_engine_' + key] = 'false'
        if self.sparse:
            # the sparse lanes: vectors over the rows and columns, the Schur slots (2 n cap), the seen ring, the
            # certificate's per-lane data; x 1.5 for the rounds' temporaries (wastewater13m1, 256 lanes: 12.7 GB)
            Mx, nflat = m + 3 * ncol, cg.nflat
            per32 = 1.5 * (24 * Mx + 8 * ncol * cap + 8 * 32 * ncol + 64 * (Fl + Fu) * Et + 48 * nflat
                         + 4 * ncol * (3 * cap + 32) + 16 * cap * cap)          # (+ K and its LU: cap x cap)
            if str(cfg.get('crown_engine_k_update', 'false')) == 'true':
                # Previous factor plus persistent captured-round factors across
                # adaptive widths (their total width is less than 2*cap).
                per32 += 1.5 * (24 * ncol * cap + 24 * cap * cap + 48 * ncol)
            if str(cfg.get('crown_engine_structure', 'false')) == 'true':
                # Sparse selected-entry tapes, fp64 block workspaces, and the
                # largest signed RHS tile. This is allocation accounting at
                # engine construction, outside the captured device algorithms.
                op_rows = cg._op()[0]
                face_degree = int(torch.bincount(op_rows % Et, minlength=Et).max()) if Et else 0
                cm_rows = cg._cm()[0]
                cm_rows = cm_rows[cm_rows >= 2] - 2
                entries = ncol * (face_degree + 2) + len(cm_rows) + int((cg.E != 0).sum())
                per32 += 1.5 * (96*entries + 16*32*entries + 8*24*cap*cap + 16*32*ncol + 8*26*m)
            elif str(cfg.get('crown_engine_subopt', 'false')) == 'true':
                per32 += 1.5 * (8*4*32*ncol + 8*26*m)
            build = 8 * (6 * nflat + 4 * ncol)
        else:
            per32 = 4 * (3 * m * ncol + 12 * ncol * ncol) + 8 * 32 * ncol
            build = 8 * ((Fl + Fu) * Et * ncol + 2 * m * ncol)
        # certify prices Q+2 candidates through padded sparse reductions, not
        # just nnz. These gathers (and their absolute-error companions) were
        # absent from the old estimate; graph pools retained them across calls.
        from auto_LiRPA.compiled_crown import Mat
        from auto_LiRPA.lockstep_dual import Q
        padded = max((idx.numel() for value in vars(cg).values() if isinstance(value, Mat)
                      for reduction in (getattr(value, 'cl', None), getattr(value, 'cr', None))
                      if reduction is not None for idx, _ in reduction.buckets), default=0)
        self.scratch_bytes = per32 + build + 8 * 4 * (Q + 2) * padded
        budget = (self.memory.available() if self.memory is not None else
                  float(cfg.get('crown_engine_mem_gb', 8.0)) * 2 ** 30)
        top = int(cfg.get('crown_engine_lanes', 256))
        while top > 1 and (top * self.scratch_bytes if self.memory is not None else
                          top * (per32 + build) + (top // 4) * 2 * per32) > budget:
            top //= 2
        if self.memory is not None:
            self.memory.require(self.scratch_bytes)
            print('[crown-memory] ' + json.dumps(dict(self.memory.snapshot(), top=top,
                                                     scratch_per_lane=int(self.scratch_bytes))), flush=True)
        self.tan_row = m - 1 if tangent is not None else None             # (table row of the tangent row)
        self.info = dict(n=n, Et=Et, ncol=ncol, m=m, Fl=Fl, Fu=Fu, R=R, levels=len(cg.levels), rows=nr - 1,
                         tangent=tangent is not None, squares=pg.squares_rewritten,
                         derived=len(pg.rows), gear_rows=self.Sg, gear_rows_kept=int((self.grow >= 0).sum()),
                         top=top, per_lane_mb=round((per32 + build) / 2 ** 20, 2))
        if os.environ.get('CROWN_ENGINE_DEBUG'):
            print(f'[crown-engine] root box lo {lo.flatten().tolist()}\nhi {hi.flatten().tolist()}', flush=True)
        self.calls = 0
        self.stats = dict(lanes=0, bound=0, farkas=0, fp64=0, numerical=0, optimal=0, seconds=0.0,
                          seconds_build=0.0, seconds_lockstep=0.0, fbbt_tightened=0, fbbt_empty=0, seconds_fbbt=0.0)
        self.lb = None
        if top < 8 and self.memory is None:
            # (a minimum of 64 lanes lost foulds3pq, which the engine closes with 32 lanes in 130 s, and won none of
            # foulds4pq / 5pq / 5stp back: the memory budget alone decides)
            if self.fbbt is None:
                raise EngineUnavailable(f'the dense lockstep table is too large for the lane budget: {self.info}')
            self.branching, self.width = 'gear', ncol + 1
            print(f'[crown-engine] FBBT only (GEAR bounds the boxes): the dense lockstep table is too large for '
                  f'the lane budget: {self.info}', flush=True)
            if report:
                atexit.register(self.report)
            return
        buckets = sorted({max(1, top // 16), max(1, top // 4), top})
        self.top = top
        if not self.sparse:
            self.Mop = cg.Mop                                              # dense operand forms (2, Et, ncol)
            self.unit = torch.zeros(Et, ncol, dtype=torch.float64, device=dev)
            self.unit[ar(Et), n + ar(Et)] = 1.0
            Epad = torch.cat([cg.E, torch.zeros(R, Et, dtype=torch.float64, device=dev)], 1)
            self.rest = torch.cat([cg.CM[cg.rows], Epad])                 # flat rows from `base` on
        self.c, self.c0 = (cg.CM0 if cg.sparse else cg.CM[0]).clone(), float(cg.Cc[0])
        levels = [(0, n)] + [(n + a, n + b) for a, b in cg.levels]
        ext = torch.nonzero(kind == 0).flatten()
        certifier = lambda B: CompiledCertifier(cg, self.sel, base, B)
        self.lb = LockstepBounding(m, ncol, levels, kind, ext, certifier, R=int(cfg.get('crown_engine_rounds', 8)),
                                   R_max=int(cfg.get('crown_engine_rounds_max', 8)),
                                   cap=(None if capc <= 0 else cap),
                                   sparse=(cg, self.sel[:len(fl) + len(fu)]) if self.sparse else None,
                                   kadapt=str(cfg.get('crown_engine_kadapt', 'false')) == 'true',
                                   k_update=str(cfg.get('crown_engine_k_update', 'false')) == 'true',
                                   k_rebuild=int(cfg.get('crown_engine_k_rebuild', 32)),
                                   structure=str(cfg.get('crown_engine_structure', 'false')) == 'true',
                                   scc=str(cfg.get('crown_engine_scc', 'false')) == 'true',
                                   subopt=str(cfg.get('crown_engine_subopt', 'false')) == 'true',
                                   strict_precision=str(cfg.get('crown_engine_strict_precision', 'false')) == 'true',
                                   near_cap=(max(1, cap - 16) if str(cfg.get('crown_engine_kadapt', 'false')) ==
                                             'true' else None),
                                   perturb=float(cfg.get('crown_engine_perturb', 0.0)),
                                   buckets=buckets, buckets64=sorted({max(1, top // 16), max(1, top // 4)}),
                                   dtype=torch.float64 if str(cfg.get('crown_engine_search', 'fp32')) == 'fp64'
                                   else torch.float32, dict_K=int(cfg.get('crown_engine_dict', 64)), trial='exchange',
                                   bland='fable', fused=str(cfg.get('crown_engine_fused', 'false')) == 'true',
                                   device=dev, memory=self.memory, scratch_bytes=self.scratch_bytes,
                                   cache_buckets=bool(cfg.get('crown_engine_cache_buckets', False)))
        # the packed state of a domain: its basis labels and certified bound (its multipliers are not kept: the
        # child refactors the labels on its own table; m floats more per domain made kall_diffcircles_7's 5.9M
        # open domains run out of memory)
        # the node targets of OBBT and the domains' node intervals as caps (crown_engine_obbt_nodes): the packed
        # state then also holds every element's interval, inherited by the children
        self.node_caps = str(cfg.get('crown_engine_obbt_nodes', 'false')) == 'true'
        self.width = ncol + 1 + (2 * Et if self.node_caps else 0)
        if self.tiered:
            self.pending_col = self.width
            self.width += 1
        if cfg.get('crown_engine_node_dsx', False):
            self.node_beta_col = self.width
            self.width += 2*ncol + 2  # sparse beta masses/labels, lineage gain/work
        if cfg.get('crown_engine_absorb', False):
            self.absorb_scope_col = self.width
            self.width += 1 + 2*n  # exact cutoff and ancestor input box
            self.absorb_faces_col = self.width
            self.width += 6*ncol + 1  # numeric faces and an explicit initialized flag
        from auto_LiRPA.operators import BoundSqr, BoundReciprocal
        self.sq = torch.zeros(Et, dtype=torch.bool, device=dev)
        self.recip = torch.zeros_like(self.sq)
        for nd in cg.nodes:
            if isinstance(pg.net[nd.name], BoundSqr):
                self.sq[nd.s:nd.s + nd.E] = True
            if isinstance(pg.net[nd.name], BoundReciprocal):
                self.recip[nd.s:nd.s + nd.E] = True
        self.branching = 'mcc' if self.tiered else str(cfg.get('crown_engine_branching', 'gear'))
        self.stay = str(cfg.get('crown_engine_stay', 'false')) == 'true'
        self.obbt_targets = int(cfg.get('crown_engine_obbt', 0))
        self.propagate = str(cfg.get('crown_engine_propagate', 'false')) == 'true'
        if self.fbbt is not None and self.propagate:
            self.fbbt.sq = self.sq
            self.fbbt.recip = self.recip
        if self.propagate or self.branching == 'mcc':
            r, c, v = cg._op()
            from auto_LiRPA.compiled_crown import Mat
            self.op_abs = Mat(coo=(r, c, v.abs()), shape=(2*Et, ncol), sparse=True)
        self._share, self.root_caps = True, None
        self.obbt_warm = str(cfg.get('crown_engine_obbt_warm', 'true')) == 'true'
        self.obbt_full = str(cfg.get('crown_engine_obbt_full', 'false')) == 'true'
        if self.obbt_targets:
            self._obbt_cert = CompiledCertifier(cg, self.sel, base, top)
            self._objw0 = torch.zeros(1, dtype=torch.float64, device=dev)
        for k in ('obbt_lanes', 'obbt_tightened', 'obbt_closed'):
            self.stats[k] = 0
        self.stats['seconds_obbt'] = 0.0
        if graph_hit and seed.lb is not None and self.m-self.cg.R == seed.m-seed.cg.R:
            oldE, oldh = seed.source_rows
            prefix = oldE is None or (E is not None and len(E) >= len(oldE)
                     and torch.equal(E[:len(oldE)], oldE) and torch.equal(h[:len(oldh)], oldh))
            if prefix and self.m >= seed.m and self.lb.K == seed.lb.K:
                self.lb.dictm[:, :seed.m].copy_(seed.lb.dictm)
                self.lb.dictv.copy_(seed.lb.dictv)
                self.root_caps = seed.root_caps
                if seed.root_seed is not None:
                    sl, su, packed = seed.root_seed
                    state, caps = seed._unpack(packed)
                    ids = torch.where(state.ids >= seed.m, state.ids+self.m-seed.m, state.ids)
                    self.root_seed = (sl, su, self._pack(state._replace(ids=ids), caps))
                if cfg.get('crown_engine_absorb', False):
                    self.root_cap_cutoff = getattr(seed, 'root_cap_cutoff', -INF)
                    if not hasattr(seed, 'root_cap_cutoff'):
                        self.root_caps, self.root_seed = None, None
                print('[crown-engine-cache] seed basis/caps/dictionary migrated with row-label map', flush=True)
        self.info['seconds_setup'] = round(time.time() - t0, 2)
        print(f'[crown-engine] ready: {self.info} buckets {buckets}', flush=True)
        if report:
            atexit.register(self.report)

    # ------------------------------------------------------------------------------------------------ data
    def _trig_ranges(self, C, thresholds, E, h):
        """Root-valid argument ranges from proportional affine constraint rows.

        Exact Fraction identities: if row = s*(argument-a0)+r0 <= t,
        then argument <= (t-r0)/s+a0 for s>0 (>= for s<0).
        Only input-affine forms are used; the outward-rounded root ranges
        hold on every feasible child. No LP and no tolerance matching.
        """
        from fractions import Fraction
        from nlp_solver.polynomial_graph import _Fold, Aff
        from auto_LiRPA.operators import BoundSin
        fold, cg = _Fold(self.pg.net), self.cg
        out = fold.array(self.pg.net[self.pg.net.final_name]).reshape(-1)
        rows = []
        for i in range(3, len(C)):
            f = sum((out[j]*Fraction(float(C[i, j])) for j in torch.nonzero(C[i]).flatten().tolist()), Aff({}, 0))
            if f.t and all(j < cg.n for j in f.t):
                rows.append((f, Fraction(float(thresholds[i]))))
        if E is not None:
            for er, hr in zip(E, h):
                rows.append((Aff({j: Fraction(float(er[j])) for j in torch.nonzero(er).flatten().tolist()}, 0),
                             Fraction(float(hr))))
        caps = (torch.full((1, 2, cg.Et), -INF, dtype=torch.float64, device=self.dev),
                torch.full((1, 2, cg.Et), INF, dtype=torch.float64, device=self.dev))
        for nd in cg.nodes:
            node = self.pg.net[nd.name]
            if not isinstance(node, BoundSin):
                continue
            for k, arg in enumerate(fold.array(node.inputs[0]).reshape(-1)):
                if not arg.t or any(j >= cg.n for j in arg.t):
                    continue
                j0 = next(iter(arg.t))
                for row, rhs in rows:
                    if row.t.keys() != arg.t.keys():
                        continue
                    s = Fraction(row.t[j0])/Fraction(arg.t[j0])
                    if not all(row.t[j] == s*arg.t[j] for j in arg.t):
                        continue
                    value = (rhs-row.k)/s+arg.k
                    f = float(value)
                    side = 1 if s > 0 else 0
                    if (side == 1 and Fraction(f) < value) or (side == 0 and Fraction(f) > value):
                        f = math.nextafter(f, INF if side else -INF)
                    old = float(caps[side][0, 0, nd.s+k])
                    caps[side][0, 0, nd.s+k] = min(old, f) if side else max(old, f)
        cg.operand_caps = caps

    def trig_anchor(self, x):
        """Use an incumbent as a slope proposal; support validity is independent of its feasibility."""
        from auto_LiRPA.operators import BoundSin, BoundExp
        if (not self.cg.trig_tangents and not self.tiered) or x is None:
            return
        with torch.no_grad():
            self.pg.net(x.reshape(1, *self.cg.in_shape).to(self.root_lo))
        for nd in self.cg.nodes:
            if isinstance(self.pg.net[nd.name], BoundSin):
                value = self.pg.net.get_forward_value(self.pg.net[nd.name].inputs[0])
                self.cg._private[nd.name].trig_anchor = value.detach().clone()
            if self.tiered and isinstance(self.pg.net[nd.name], BoundExp):
                value = self.pg.net.get_forward_value(self.pg.net[nd.name].inputs[0])
                self.cg._private[nd.name].face_anchor = value.detach().clone()

    def _thr(self, lo, hi, T):
        """the certificate rows' thresholds (B, S) on the boxes: [objective (unused), its slot (0), the cutoff (T, 0
        where none), GEAR's rows and the derived rows (PolynomialGraph.thresholds)]"""
        te = self.pg.thresholds(lo, hi).to(torch.float64)
        z = te.new_zeros(te.shape[0], 1)
        cut = torch.where(torch.isfinite(T), T, 0.0).to(te).reshape(-1, 1).expand(te.shape[0], 1)
        return torch.cat([z, z, cut, te[:, self.keep]], 1)

    def _obbt(self, xl, xu, s, T, tabs, res, lb, caps=None):
        """OBBT lanes on the chunk s: for the open, LP-optimal domains of lowest bound (at most P* / (2 K)), their K input
        columns of largest width x dual-weighted coefficient (sum_i |y_i R_ij| at the domain's certified masses), the
        objectives +-e_j as lanes of the same lockstep dual, warm from the domain's basis; each lane's last basis's
        masses y give the Lagrangian value D of its objective, and the slab beyond D (less tau (1 + |D|), tau =
        2^-26) is certified empty by the Farkas value of y on the slab's box (compiled CROWN certify): the domain's
        bound on x_j moves there. A lane that closes (bound >= T or Farkas: its certificates hold for the domain)
        prunes the domain. In place: xl, xu (the domains' boxes) and lb (B, S)."""
        from auto_LiRPA.lockstep_bounding import DomainState
        from auto_LiRPA.lockstep_dual import CompiledCertifier
        K, n, dev = self.obbt_targets, self.n, self.dev
        Ts = T[s]
        # (only domains whose LP the search has solved and that still cannot be pruned: there the bound has
        # nothing more to give on the box, and the basis is a good start; on tables too slow to reach the optimum,
        # OBBT lanes would only take the output lanes' time, as on foulds3pq)
        ready = ~res.numerical if getattr(self.lb, 'strict_precision', False) else res.optimal
        cand = torch.nonzero((res.kind == 0) & ready & torch.isfinite(lb[:, 0]) & (lb[:, 0] < Ts)).flatten()
        if not cand.numel():
            return
        t0 = time.time()
        k = min(K, n)
        kn = min(K, self.Et) if (self.node_caps and caps is not None) else 0      # node targets per domain
        cand = cand[torch.argsort(lb[cand, 0])[:max(1, self.top // (2 * (k + kn)))]]
        y = res.state.y[cand].to(torch.float64)
        if self.sparse:
            op64 = next((bucket.eng for bucket in self.lb.buckets64 + self.lb.buckets
                         if bucket.eng is not None), None)
            if op64 is None:
                return
            ym = torch.cat([y.abs(), y.new_zeros(len(cand), 2 * self.ncol)], 1)
            sens = op64._orig_ATy(ym, select_rows(tabs['lw'], cand).to(op64.dt), select_rows(tabs['uw'], cand).to(op64.dt),
                                  absolute=True).double()[:, :n]
        else:
            sens = (y.abs()[:, :, None] * select_rows(tabs['R'], cand)[:, :, :n].abs()).sum(1)
        w = xu[s][cand] - xl[s][cand]
        # width x dual-weighted coefficient; where no mass touches the inputs (an LP bound at the box), the width
        # relative to the root box breaks the tie (eigmaxb: the bound is the objective's own box end)
        rw = (self.root_hi - self.root_lo).clamp(min=1e-300)
        fallback = sens.amax(1, keepdim=True) <= 0
        score = torch.where(w > 0, torch.where(fallback, w / rw, w * sens), -1.0)
        if self.propagate and self._share:
            # The objective may sit at its box end with no useful dual input
            # sensitivity. Probe inputs involved in violated nonlinear faces,
            # including squares/reciprocals, as in lifted node probing.
            violation = self._violation(res.witness[cand])
            vs = self.op_abs.left(violation.repeat(1, 2))[:, :n]*w
            # Violation is a fallback for a zero-sensitivity objective. Replacing
            # useful dual sensitivities spends the limited OBBT lanes on large
            # nonlinear residuals even when their inputs cannot be tightened.
            score = torch.where(fallback & (vs.amax(1, keepdim=True) > 0), vs, score)
        if self.propagate and not self._share:  # a full root pass must not drop zero-mass targets
            score = torch.where(w > 0, w / rw, -1.0)
        sc, tj = score.topk(k, 1)
        if kn:
            # node targets: the products and squares of largest relaxation gap at the domain's LP solution
            xw = res.witness[cand].to(torch.float64)
            op = (self.cg.Lop.right(torch.nan_to_num(xw)).unflatten(-1, (2, self.Et)) + self.cg.cop[None])
            zc = xw[:, n:]
            gap = torch.where(self.mul, (zc - op[:, 0] * op[:, 1]).abs(),
                              torch.where(self.sq, (zc - op[:, 0] * op[:, 0]).abs(), 0.0))
            if self.propagate:
                gap = self._violation(xw)
            wz = caps[1][cand] - caps[0][cand]
            gap = torch.where(torch.isfinite(xw).all(1, keepdim=True) & (wz > 0), gap.nan_to_num(0.0), -1.0)
            if self.propagate and not self._share:
                gap = torch.where(wz > 0, wz, -1.0)
            scn, tn = gap.topk(kn, 1)
            sc, tj = torch.cat([sc, scn], 1), torch.cat([tj, n + tn], 1)
            k = k + kn
        keep = (sc > 0).flatten().repeat_interleave(2)
        dom = cand.repeat_interleave(2 * k)[keep]
        jj = tj.repeat_interleave(2, 1).flatten()[keep]
        side = torch.tensor([1.0, -1.0], dtype=torch.float64, device=dev).repeat(len(cand) * k)[keep]
        L = dom.numel()
        if not L:
            return
        ar = torch.arange(L, device=dev)
        sub = {key: select_rows(v, dom) for key, v in tabs.items()}
        c = torch.zeros(L, self.ncol, dtype=torch.float64, device=dev)
        c[ar, jj] = side
        sub['c'], sub['c0'] = c, torch.zeros(L, dtype=torch.float64, device=dev)
        # warm start: the domain's basis (dual infeasible for the new objective: phase 1), or (obbt_warm false)
        # CROWN's sign start of the objective itself, dual feasible from the first round
        st = DomainState(res.state.ids[dom], res.state.best[dom], None) if self.obbt_warm else None
        # anytime: the OBBT lanes' blocks of rounds stop once they have taken as long as the chunk's own bounding
        # (time-shared 1:1 with the output lanes; at least R rounds, each lane certified from its last basis)
        self.lb.want_yfin, self.lb.deadline = True, (self._deadline if self.tiered else min(time.time() + self._chunk_seconds, self._deadline)
                                                     if self._share and not self.obbt_full else self._deadline)
        try:
            r = self.lb.bound(sub, st, Ts[dom], self.c, self.c0)
        finally:
            self.lb.want_yfin, self.lb.deadline = False, self._deadline
        # closed lanes: their certificates are of the main objective (bound >= T) or Farkas rays
        cl = torch.zeros(lb.shape[0], dtype=torch.bool, device=dev)
        cl[dom[r.kind > 0]] = True
        lb[cl] = INF
        # the Lagrangian of each lane's objective at its last basis's masses (plain fp64; certified below)
        yv = r.yfin.to(torch.float64)
        if self.sparse:
            ym = torch.cat([yv, yv.new_zeros(L, 2 * self.ncol)], 1)
            rr = c + op64._orig_ATy(ym.to(op64.dt), sub['lw'].to(op64.dt), sub['uw'].to(op64.dt)).double()
        else:
            rr = c + (yv[:, :, None] * sub['R'].to(torch.float64)).sum(1)
        D = -(sub['h'].to(torch.float64) * yv).sum(1) + torch.minimum(rr * sub['l'], rr * sub['u']).nan_to_num(
            nan=0.0).sum(1)
        tau = 2.0 ** -26
        prop = torch.where(side > 0, D - tau * (1 + D.abs()), -D + tau * (1 + D.abs()))
        lj, uj = sub['l'][ar, jj], sub['u'][ar, jj]
        useful = torch.isfinite(D) & torch.where(side > 0, prop > lj, prop < uj) & (r.kind == 0) & (yv.abs().sum(1) > 0)
        lo_s, hi_s = sub['l'].clone(), sub['u'].clone()
        hi_s[ar, jj] = torch.where(side > 0, torch.minimum(prop, uj), uj)
        lo_s[ar, jj] = torch.where(side < 0, torch.maximum(prop, lj), lj)
        phi = torch.full((L,), -INF, dtype=torch.float64, device=dev)
        cert = self._obbt_cert
        for a in range(0, L, cert.B):
            q = slice(a, min(L, a + cert.B))
            m_ = q.stop - q.start
            cert.load(0, sub['lw'][q], sub['lb'][q], sub['uw'][q], sub['ub'][q], lo_s[q], hi_s[q], sub['thr'][q],
                      h=sub.get('hb', None) if 'hb' not in sub else sub['hb'][q], undefined=sub['undefined'][q],
                      rowmask=sub['rowmask'][q])
            cert.prepare()
            mass = torch.zeros(cert.B, 1, self.m, dtype=torch.float64, device=dev)
            mass[:m_, 0] = yv[q]
            phi[q] = cert(mass, self._objw0)[:m_, 0]
        ok = useful & (phi > 0)
        gi = s.start + dom
        nc = self.ncol
        newl = torch.full((lb.shape[0], nc), -INF, dtype=torch.float64, device=dev)
        newu = torch.full((lb.shape[0], nc), INF, dtype=torch.float64, device=dev)
        lo_ok, up_ok = ok & (side > 0), ok & (side < 0)
        newl.index_put_((dom[lo_ok], jj[lo_ok]), prop[lo_ok], accumulate=False)
        newu.index_put_((dom[up_ok], jj[up_ok]), prop[up_ok], accumulate=False)
        rows = slice(s.start, s.start + lb.shape[0])
        xl[rows] = torch.maximum(xl[rows], newl[:, :n])
        xu[rows] = torch.minimum(xu[rows], newu[:, :n])
        emp = (xl[rows] > xu[rows]).any(1)
        if caps is not None:            # (node targets: the domain's node caps)
            caps[0].copy_(torch.maximum(caps[0], newl[:, n:]))
            caps[1].copy_(torch.minimum(caps[1], newu[:, n:]))
            emp = emp | (caps[0] > caps[1]).any(1)
        lb[emp] = INF
        if bool(ok.any()):
            Tf = Ts[torch.isfinite(Ts)]
            if Tf.numel():
                cap = arguments.Globals.get('discarded_threshold_min')
                arguments.Globals['discarded_threshold_min'] = float(Tf.min()) if cap is None else \
                    min(cap, float(Tf.min()))
        self.stats['obbt_lanes'] += L
        self.stats['obbt_tightened'] += int(ok.sum())
        self.stats['obbt_closed'] += int(cl.sum())
        self.stats['seconds_obbt'] += time.time() - t0

    def _fbbt_box(self, d, T):
        """FBBT on Data d's column boxes (cutoff T per box): the tightened column box and empty (B,)"""
        cg, rid = self.cg, self.fbbt.row_ids
        b_out = d.thr[:, rid] - cg.Cc[rid]
        b_out[:, 0] = torch.where(torch.isfinite(T), T - cg.Cc[2], INF)          # (row 2: the cutoff)
        b_out = torch.where(torch.isfinite(b_out), b_out, INF)
        h = d.h if d.h.shape[1] == cg.R else d.h.expand(d.lo.shape[0], cg.R)
        return self.fbbt.run(d.lo, d.hi, b_out, h.to(torch.float64), self.fbbt_sweeps)

    def _bound_fbbt(self, xl, xu, s, T, caps_in=None):
        """compiled CROWN on the boxes xl[s], xu[s]; with FBBT, the boxes tightened in place (xl, xu: the inputs'
        bounds, kept by the domains for their children) and the bound rebuilt on them, its node columns intersected
        with FBBT's (A71: _widen keeps them off degenerate widths); the regions cut off by the cutoff row hold no
        point better than T, which then caps the reported dual. Returns (Data, row bounds, empty (b,))"""
        cg, n = self.cg, self.n
        d, out = cg.bound(xl[s], xu[s], self._thr(xl[s], xu[s], T[s]), self._hb(xl[s], xu[s], T[s]), caps=caps_in)
        if self.cfg.get('crown_engine_absorb', False) and self.tan_row is not None:
            # Targets select/repeat lanes, and later box bounds overwrite the
            # engine scratch mask. Retain this table's own immutable mask.
            d.tan_ok = self._tan_ok
        b = d.lo.shape[0]
        empty = torch.zeros(b, dtype=torch.bool, device=self.dev)
        caps = None
        if self.node_caps:              # (the domain's node intervals: the inherited caps, tightened below)
            caps = (d.lo[:, n:].clone(), d.hi[:, n:].clone()) if caps_in is None else \
                (torch.maximum(caps_in[0], d.lo[:, n:]), torch.minimum(caps_in[1], d.hi[:, n:]))
        if caps is not None:            # (an element whose interval is empty: the box holds no point)
            empty = empty | (caps[0] > caps[1]).any(1)
        if self.fbbt is None:
            return d, out, empty, caps
        if self.tiered and self.dev.type == 'cuda':
            torch.cuda.synchronize(self.dev)
        tf = time.time()
        lo, hi, emp_f = self._fbbt_box(d, T[s])
        empty = empty | emp_f
        if caps is not None:
            caps = (torch.maximum(caps[0], lo[:, n:]), torch.minimum(caps[1], hi[:, n:]))
            empty = empty | (caps[0] > caps[1]).any(1)
        nl, nu = torch.maximum(lo[:, :n], xl[s]), torch.minimum(hi[:, :n], xu[s])
        tight = ((nl > xl[s]) | (nu < xu[s])).any(1) & ~empty
        self.stats['fbbt_tightened'] += int(tight.sum())
        self.stats['fbbt_empty'] += int(empty.sum())
        node_tight = ((lo[:, n:] > d.lo[:, n:]) | (hi[:, n:] < d.hi[:, n:])).any(1) & ~empty
        if bool(tight.any()) or (self.propagate and bool(node_tight.any())):
            xl[s] = torch.where(tight[:, None], nl, xl[s])
            xu[s] = torch.where(tight[:, None], nu, xu[s])
            changed = tight | empty | (node_tight if self.propagate else False)
            Tf = T[s][changed]
            Tf = Tf[torch.isfinite(Tf)]
            if Tf.numel():
                cap = arguments.Globals.get('discarded_threshold_min')
                arguments.Globals['discarded_threshold_min'] = float(Tf.min()) if cap is None else \
                    min(cap, float(Tf.min()))
            # the bound rebuilt on the tightened boxes, FBBT's node intervals as caps level by level (the faces of
            # every level from the capped intervals below it; _widen after the caps, A71)
            cz = (lo[:, n:], hi[:, n:]) if caps is None else caps
            cz = (torch.where(empty[:, None], -INF, cz[0]), torch.where(empty[:, None], INF, cz[1]))
            d, out = cg.bound(xl[s], xu[s], self._thr(xl[s], xu[s], T[s]), self._hb(xl[s], xu[s], T[s]), caps=cz)
            if self.cfg.get('crown_engine_absorb', False) and self.tan_row is not None:
                d.tan_ok = self._tan_ok
        if self.tiered and self.dev.type == 'cuda':
            torch.cuda.synchronize(self.dev)
        self.stats['seconds_fbbt'] += time.time() - tf
        return d, out, empty, caps

    def tighten(self, x_L, x_U, thresholds):
        """FBBT alone (the lockstep does not fit; GEAR bounds the boxes): the tightened boxes and empty (D,)"""
        D = x_L.shape[0]
        xl = x_L.reshape(D, -1).to(self.dev, torch.float64).clone()
        xu = x_U.reshape(D, -1).to(self.dev, torch.float64).clone()
        T = thresholds.reshape(D, -1)[:, 0].to(self.dev, torch.float64)
        top = 256
        empties = []
        for a in range(0, D, top):
            empties.append(self._bound_fbbt(xl, xu, slice(a, min(D, a + top)), T)[2])     # (no node caps here)
        return xl.reshape(x_L.shape).to(x_L), xu.reshape(x_U.shape).to(x_U), torch.cat(empties)

    def _rowmask_out(self, T):
        """(b, S - 2) which spec rows 2.. hold on the lanes (the cutoff only with an incumbent)"""
        m = torch.ones(T.shape[0], self.S - 2, dtype=torch.bool, device=self.dev)
        m[:, 0] = torch.isfinite(T)
        return m

    def _partial(self, tabs, state, T, good):
        """LockstepBounding.bound on the lanes `good` only; the others: bound -inf, open, their state as given
        (labels -1 where none), no point"""
        from auto_LiRPA.lockstep_bounding import BoundResult, DomainState
        b = good.shape[0]
        ids = state.ids.clone() if state is not None else torch.full((b, self.ncol), -1, dtype=torch.long,
                                                                     device=self.dev)
        best = state.best.clone() if state is not None else torch.full((b,), -INF, dtype=torch.float64,
                                                                       device=self.dev)
        out = BoundResult(lower=torch.full((b,), -INF, dtype=torch.float64, device=self.dev),
                          kind=torch.zeros(b, dtype=torch.long, device=self.dev),
                          state=DomainState(ids, best, torch.zeros(b, self.m, dtype=torch.float64, device=self.dev)),
                          witness=torch.full((b, self.ncol), float('nan'), dtype=torch.float64, device=self.dev),
                          optimal=torch.zeros(b, dtype=torch.bool, device=self.dev),
                          numerical=torch.zeros(b, dtype=torch.bool, device=self.dev),
                          fp64=torch.zeros(b, dtype=torch.bool, device=self.dev))
        g = torch.nonzero(good).flatten()
        reasons = torch.full((b,), 4, dtype=torch.long, device=self.dev) if self.lb.strict_precision else None
        if not g.numel():
            self.lb.last_processed = torch.zeros(b, dtype=torch.bool, device=self.dev)
            if reasons is not None:
                self.lb.last_reasons = reasons
            return out
        sub = None if state is None else DomainState(state.ids[g], state.best[g], None)
        r = self.lb.bound({k: v[g] for k, v in tabs.items()}, sub, T[g], self.c, self.c0)
        if getattr(self.lb, 'last_processed', None) is not None:
            processed = torch.zeros(b, dtype=torch.bool, device=self.dev)
            processed[g] = self.lb.last_processed
            self.lb.last_processed = processed
        if reasons is not None:
            reasons[g] = self.lb.last_reasons
            self.lb.last_reasons = reasons
        for name in ('lower', 'kind', 'witness', 'optimal', 'numerical', 'fp64'):
            getattr(out, name)[g] = getattr(r, name).to(getattr(out, name).dtype)
        out.state.ids[g], out.state.best[g], out.state.y[g] = r.state.ids, r.state.best, r.state.y.to(torch.float64)
        return out

    def _hb(self, lo, hi, T):
        """the base rows' right-hand sides (B, R) on the boxes: GEAR's, then the tangent row's T - b(box) (where T
        is finite and the plane exists; else a vacuous value above g x on the box, the row masked)"""
        if self.tangent is None:
            return self.hb
        B = lo.shape[0]
        g = self.cg.E[-1]
        try:
            b = self.tangent.plane(lo, hi)[1].to(lo)
        except ValueError:
            b = torch.full((B,), -INF, dtype=lo.dtype, device=lo.device)
        ok = torch.isfinite(T) & torch.isfinite(b)
        vac = torch.maximum(g * lo, g * hi).sum(1)
        vac = vac + 1e-6 * (vac.abs() + 1)
        ht = torch.where(ok, T - b, vac)
        self._tan_ok = ok
        return torch.cat([self.hb[:-1].expand(B, -1), ht[:, None]], 1)

    def empty_states(self, B):
        """packed states of B domains without one (labels -1: a cold start; no node caps)"""
        if self.cfg.get('crown_engine_m1', False):
            return torch.full((B, 1), -INF, dtype=torch.float32, device=self.dev)
        st = torch.zeros(B, self.width, dtype=(torch.float64 if self.cfg.get('crown_engine_absorb', False)
                                              else torch.float32), device=self.dev)
        st[:, :self.ncol] = -1
        st[:, self.ncol] = -INF
        if self.node_caps:
            o = self.ncol + 1
            if self.root_caps is None:
                st[:, o:o + self.Et] = -INF
                st[:, o + self.Et:o + 2*self.Et] = INF
            else:                       # (the root OBBT's node caps hold in every domain of the root box)
                st[:, o:o + self.Et] = self._f32(self.root_caps[0], False).expand(B, -1)
                st[:, o + self.Et:o + 2*self.Et] = self._f32(self.root_caps[1], True).expand(B, -1)
        if hasattr(self, 'absorb_scope_col'):
            o = self.absorb_scope_col
            st[:, o] = getattr(self, 'root_cap_cutoff', INF) if self.root_caps is not None else INF
            st[:, o+1:o+1+self.n] = self.root_lo
            st[:, o+1+self.n:o+1+2*self.n] = self.root_hi
        return st

    @staticmethod
    def _f32(t, up):
        """t in fp32, rounded outward (up: never below t)"""
        r = t.to(torch.float32)
        if up:
            return torch.where(r.double() < t, torch.nextafter(r, torch.full_like(r, INF)), r)
        return torch.where(r.double() > t, torch.nextafter(r, torch.full_like(r, -INF)), r)

    def _pack(self, s, caps=None):
        parts = [s.ids.float(), self._f32(s.best, False)[:, None]]
        if self.node_caps:
            parts += [self._f32(caps[0], False), self._f32(caps[1], True)]
        if hasattr(self, 'pending_col'):
            parts.append(s.best.new_zeros(len(s.best), 1).float())
        if hasattr(self, 'node_beta_col'):
            parts.append(s.best.new_zeros(len(s.best), 2*self.ncol + 2).float())
        if hasattr(self, 'absorb_scope_col'):
            parts = [p.double() for p in parts]
            scope = s.best.new_full((len(s.best), 1), getattr(self, 'root_cap_cutoff', INF))
            parts += [scope, self.root_lo.expand(len(s.best), -1), self.root_hi.expand(len(s.best), -1)]
        if hasattr(self, 'absorb_faces_col'):
            parts.append(s.best.new_zeros(len(s.best), 6*self.ncol+1))
        return torch.cat(parts, 1)

    def _unpack(self, st):
        """(DomainState, node caps (lo, hi) (b, Et) or None)"""
        from auto_LiRPA.lockstep_bounding import DomainState
        st = st.to(self.dev)
        caps = None
        if self.node_caps:
            o = self.ncol + 1
            caps = (st[:, o:o + self.Et].double(), st[:, o + self.Et:o + 2*self.Et].double())
        return DomainState(st[:, :self.ncol].long(), st[:, self.ncol].double(), None), caps

    def child_states(self, history, B):
        """the packed states of the B domains being added: history.engine_state when it is theirs, else empty"""
        st = getattr(history, 'engine_state', None)
        history.engine_state = None
        if (torch.is_tensor(st) and st.shape[0] == B
                and (not self.cfg.get('crown_engine_node_dsx', False) or st.shape[1] == self.width)):
            return st
        if (self.tiered and self.root_seed is not None
                and (not self.cfg.get('crown_engine_node_dsx', False)
                     or self.root_seed[2].shape[1] == self.width)):
            return self.root_seed[2].expand(B, -1).clone()
        return self.empty_states(B)

    def cutoff_thresholds(self, thresholds):
        """GEAR's thresholds with the objective row at the cutoff T of the incumbent P it holds"""
        P = float(thresholds.reshape(-1, thresholds.shape[-1])[0, 0])
        if not math.isfinite(P):
            return thresholds
        out = thresholds.clone()
        out[..., 0] = gap_cutoff(P, self.abs_tol, self.rel_tol)
        return out

    # ------------------------------------------------------------------------------------------------ bounding
    def _tables(self, d, T):
        cg, Fl, Fu, Et, n = self.cg, self.Fl, self.Fu, self.Et, self.n
        B = d.lo.shape[0]
        # the cutoff row without an incumbent: a vacuous row of the search (above the objective's maximum over the
        # column boxes), masked out of the certificate
        noinc = ~torch.isfinite(T)
        ub = torch.maximum(self.c * d.lo, self.c * d.hi).nan_to_num(nan=0.0).sum(1) + self.c0
        ub = (ub + 1e-6 * (ub.abs() + 1)).clamp(max=1e30)
        d.thr[:, 2] = torch.where(noinc, ub, d.thr[:, 2])
        d.rhs(cg.rows, cg.Ccr)
        rowmask = torch.ones(B, self.m, dtype=torch.float64, device=self.dev)
        rowmask[:, self.cut_row] = (~noinc).double()
        if self.tan_row is not None:
            rowmask[:, self.tan_row] = getattr(d, 'tan_ok', self._tan_ok).double()
        if not self.sparse:
            Al = torch.einsum('bfpe,pec->bfec', d.lw, self.Mop) - self.unit
            Au = self.unit - torch.einsum('bfpe,pec->bfec', d.uw, self.Mop)
            A = torch.cat([Al.reshape(B, Fl * Et, -1), Au.reshape(B, Fu * Et, -1), self.rest.expand(B, -1, -1)], 1)
        ar = torch.arange(Et, device=self.dev)
        dlow = self.inv[d.dl.argmax(1) * Et + ar]
        dup = self.inv[(Fl + d.du.argmax(1)) * Et + ar]
        pad = torch.full((B, n), -1, dtype=torch.long, device=self.dev)
        dense = {} if self.sparse else {'R': A[:, self.sel]}
        return dict(**dense, h=d.bflat[:, self.sel], l=d.lo, u=d.hi, dlow=torch.cat([pad, dlow], 1),
                    dup=torch.cat([pad, dup], 1), lw=d.lw, lb=d.lb, uw=d.uw, ub=d.ub, thr=d.thr,
                    undefined=d.undefined, rowmask=rowmask, **({} if self.hb is None else {'hb': d.h}))

    def _bound_budgeted(self, x_L, x_U, dm_lb, thresholds, history, deadline):
        """Optional tier: preserve every lane when work or its LP is unavailable."""
        from auto_LiRPA.work_budget import WorkBudget
        D, cg = len(x_L), self.cg
        deadline = INF if deadline is None else float(deadline)
        budget = WorkBudget(deadline, dict(work=max(0., self.cfg.get('crown_engine_upgrade_seconds', 5.)),
                            capture=max(0., self.cfg.get('crown_engine_capture_seconds', 10.))),
                            self.dev, self.budget_estimates)
        budget.sync()
        start = time.perf_counter()
        xl, xu = (v.reshape(D, -1).to(self.dev, torch.float64).clone() for v in (x_L, x_U))
        T = thresholds.reshape(D, -1)[:, 0].to(xl)
        old = xl.new_full((D, self.Sg), -INF) if dm_lb is None else dm_lb.reshape(D, -1).to(xl)
        old = torch.where(torch.isnan(old), -INF, old)
        lb = old.clone()
        lA, lB = xl.new_zeros(D, self.Sg, self.n), xl.new_full((D, self.Sg), -INF)
        st = getattr(history, 'engine_state', None)
        states = st.clone().to(self.dev) if torch.is_tensor(st) and len(st) == D else self.empty_states(D)
        witness = xl.new_full((D, self.ncol), float('nan'))
        pending = torch.ones(D, dtype=torch.bool, device=self.dev)
        valid = torch.zeros_like(pending)
        reasons = torch.full((D,), 6, dtype=torch.long, device=self.dev)
        bad_nodes = torch.full((D,), -1, dtype=torch.long, device=self.dev)
        retry_before = self.lb.fp64_retries
        rounds_before = len(self.lb.rounds_run)
        self._deadline = deadline
        self.lb.deadline, self.lb.budget = deadline, budget
        max_domains = max(0, int(self.cfg.get('crown_engine_upgrade_domains', 256)))
        obbt_seconds = max(0., self.cfg.get('crown_engine_root_obbt_seconds', 2.) if not self._share
                           else self.cfg.get('crown_engine_obbt_seconds', 0.))
        obbt_spent = 0.
        try:
            for a in range(0, min(D, max_domains), self.top):
                s = slice(a, min(D, max_domains, a + self.top))
                key = s.stop - s.start
                if not budget.allows('build', key):
                    break
                initial_l, initial_u = xl[s].clone(), xu[s].clone()
                sub, caps_in = self._unpack(states[s])
                fbbt_before = self.stats['seconds_fbbt']
                with budget.measure('build', key):
                    d, out, empty, caps = self._bound_fbbt(xl, xu, s, T, caps_in)
                    tabs = self._tables(d, T[s])
                    plane_b, g, _ = cg.output_bound(d)
                # Undefined operator faces never justify pruning or shrinking.
                empty = empty & ~d.undefined
                crown_far = ((out[:, 2:] > d.thr[:, 2:]) & torch.isfinite(d.thr[:, 2:])
                             & self._rowmask_out(T[s])).any(1) & ~d.undefined
                empty |= crown_far
                finite_box = torch.isfinite(d.lo).all(1) & torch.isfinite(d.hi).all(1)
                good = ~d.undefined & finite_box & ~empty
                rr = torch.where(d.undefined, 1, torch.where(~finite_box, 2, 0))
                rr = torch.where(empty, 5, rr)
                if hasattr(d, 'bad_node'):
                    bad_nodes[s] = d.bad_node
                # The partial solver preserves parent labels on invalid lanes.
                res = self._partial(tabs, sub, T[s], good)
                processed = good & self.lb.last_processed
                pending[s] = good & ~processed
                rr = torch.where(good & ~processed, 6, rr)
                rr = torch.where(processed & res.numerical, 4, rr)
                rr = torch.where(processed & ~torch.isfinite(res.lower), 3, rr)
                rr = torch.where(processed & (res.kind == 2), 5, rr)
                rr = torch.where(processed & (res.kind == 1), 0, rr)
                reasons[s] = rr
                usable = processed & torch.isfinite(res.lower)
                # A non-finite proposal is no certificate. Explicit Farkas and
                # cutoff proofs retain their separate, rigorous semantics.
                proposed = torch.where(usable, res.lower, -INF)
                lb[s, 0] = torch.maximum(old[s, 0], proposed)
                rows = self.grow >= 0
                row_values = out[:, self.grow[rows]].double()
                row_values = torch.where(torch.isfinite(row_values) & ~d.undefined[:, None], row_values, -INF)
                lb[s, rows] = torch.maximum(lb[s, rows], row_values)
                bound_hit = processed & (res.kind == 1) & torch.isfinite(T[s])
                lb[s, 0] = torch.where(bound_hit, torch.maximum(lb[s, 0], T[s]), lb[s, 0])
                far = (processed & (res.kind == 2)) | empty
                lb[s] = torch.where(far[:, None], INF, lb[s])
                g = g.expand(key, -1, -1).double()
                price = torch.minimum(g * d.lo[:, None, :self.n], g * d.hi[:, None, :self.n]).sum(-1)
                intercept = plane_b - price
                ga, gb = g[:, self.grow[rows]], intercept[:, self.grow[rows]]
                plane_ok = torch.isfinite(ga).all(-1) & torch.isfinite(gb) & ~d.undefined[:, None]
                lA[s, rows] = torch.where(plane_ok[:, :, None], ga, 0.)
                lB[s, rows] = torch.where(plane_ok, gb, -INF)
                # Caps and input tightening from an unavailable LP are discarded;
                # the native lane and its parent's certificates remain intact.
                xl[s] = torch.where(good[:, None], xl[s], initial_l)
                xu[s] = torch.where(good[:, None], xu[s], initial_u)
                if caps is not None:
                    caps = tuple(torch.where(good[:, None] & ~torch.isnan(v), v, orig)
                                 for v, orig in zip(caps, caps_in))
                state_best = torch.where(torch.isfinite(res.state.best), res.state.best, sub.best)
                packed = self._pack(res.state._replace(best=state_best), caps)
                states[s] = torch.where(processed[:, None], packed, states[s])
                witness_ok = usable & res.optimal & torch.isfinite(res.witness).all(1) & ~res.numerical
                witness[s] = torch.where(witness_ok[:, None], res.witness, float('nan'))
                valid[s] = witness_ok
                if self.obbt_targets and obbt_spent < obbt_seconds and time.time() < deadline:
                    obudget = WorkBudget(min(deadline, time.time()+obbt_seconds-obbt_spent), dict(work=obbt_seconds-obbt_spent,
                                        capture=max(0., budget.limits['capture']-budget.spent['capture'])),
                                        self.dev, self.budget_estimates)
                    self.lb.budget = obudget
                    tick = time.perf_counter()
                    obbt_before = self.stats['seconds_obbt']
                    self._obbt(xl, xu, s, T, tabs, res, lb[s], caps)
                    obudget.sync()
                    obbt_elapsed = time.perf_counter()-tick
                    obbt_spent += obbt_elapsed
                    self.stats['seconds_obbt'] = obbt_before + obbt_elapsed-obudget.spent['capture']
                    budget.spent['capture'] += obudget.spent['capture']
                    budget.spent['obbt'] += obbt_elapsed-obudget.spent['capture']
                    self.lb.budget = budget
                    states[s] = torch.where(processed[:, None], self._pack(res.state._replace(best=state_best), caps), states[s])
                for k, value in (('lanes', key), ('bound', int(bound_hit.sum())), ('farkas', int(far.sum())),
                                 ('numerical', int(res.numerical.sum())), ('optimal', int(res.optimal.sum()))):
                    self.stats[k] += value
                budget.spent['fbbt'] += self.stats['seconds_fbbt'] - fbbt_before
                # FBBT is nested in build; keep the ledger disjoint.
                budget.spent['build'] -= self.stats['seconds_fbbt'] - fbbt_before
        finally:
            self.lb.deadline, self.lb.budget = INF, None
        if hasattr(self, 'pending_col'):
            states[:, self.pending_col] = pending.to(states)
        history.engine_state = states
        history.engine_boxes = (xl.reshape_as(x_L).to(x_L), xu.reshape_as(x_U).to(x_U))
        history.engine_witness, history.engine_valid = witness, valid
        history.engine_pending, history.engine_reasons = pending, reasons
        history.engine_stay = pending.clone()
        self.calls += 1
        budget.sync()
        elapsed = time.perf_counter() - start
        retries = self.lb.fp64_retries-retry_before
        self.stats['seconds'] += elapsed
        self.stats['fp64'] += retries
        for phase, value in budget.spent.items():
            if phase not in ('fbbt', 'obbt'):
                k = 'seconds_' + phase
                self.stats[k] = self.stats.get(k, 0.) + value
        ledger = dict(call=self.calls, domains=D, processed=int((reasons != 6).sum()), seconds=elapsed,
                      phases={k:budget.spent[k] for k in ('build','fbbt','dict','capture','rounds','obbt','setup','finish')},
                      fp64_retries=retries,
                      rounds=self.lb.rounds_run[rounds_before:],
                      reasons=dict(collections.Counter(reasons.cpu().tolist())),
                      reason_codes={0:'valid', 1:'undefined_operator', 2:'nonfinite_box', 3:'nonfinite_bound',
                                    4:'numerical', 5:'proved_empty', 6:'budget_pending'},
                      bad_node=bad_nodes.cpu().tolist(), node_names=[n.name for n in cg.nodes],
                      lower0=old[:, 0].cpu().tolist(), lower1=lb[:, 0].cpu().tolist())
        self.last_ledger = ledger
        path = os.environ.get('CROWN_ENGINE_LEDGER')
        if path:
            with open(path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(ledger, allow_nan=True) + '\n')
        print('[crown-engine-ledger] ' + json.dumps({k:v for k,v in ledger.items()
              if k not in ('lower0', 'lower1', 'bad_node', 'node_names')}, sort_keys=True), flush=True)
        return lb.to(thresholds), lA.to(thresholds).reshape(D, self.Sg, *x_L.shape[1:]), lB.to(thresholds)

    def bound(self, x_L, x_U, dm_lb, thresholds, history, deadline=None):
        """GEAR's bounding step on the domains x_L, x_U: (lower bounds (D, S), lA (D, S, n), lbias (D, S)); the new
        states go to history.engine_state"""
        from auto_LiRPA.cuda_memory import memory_error, release_cuda
        if self.cfg.get('crown_engine_m1', False):
            from input_split.m1_qp import bounded_nodes
            return bounded_nodes(self, x_L, x_U, dm_lb, thresholds, history, deadline)
        if getattr(self, 'memory', None) is None:
            return self._bound_impl(x_L, x_U, dm_lb, thresholds, history, deadline)
        if getattr(self.lb, 'memory_blocked', False):
            self.memory_disabled = True
        while not getattr(self, 'memory_disabled', False):
            try:
                with self.memory.allocations():
                    self.memory.require(self.top * self.scratch_bytes)
                    return self._bound_impl(x_L, x_U, dm_lb, thresholds, history, deadline)
            except Exception as error:
                if not memory_error(error):
                    raise
                print(f'[crown-memory] table fallback top={self.top}: {error}', flush=True)
            for bucket in self.lb.buckets + self.lb.buckets64:
                bucket.graphs, bucket.eng = None, None
            self.lb.capture_enabled = False
            release_cuda(self.dev)
            if self.top > 1:
                self.top = max(1, self.top // 2)
            else:
                self.memory_disabled = True
        # Native/inherited certificates survive a failed optional upgrade.
        D = len(x_L)
        history.engine_valid = torch.zeros(D, dtype=torch.bool, device=x_L.device)
        keep_pending = any(self.cfg.get('crown_engine_'+key, False)
                           for key in ('cascade', 'continue', 'nested_faces', 'dual_obbt'))
        history.engine_pending = history.engine_stay = torch.full_like(history.engine_valid, keep_pending)
        history.engine_witness = None
        if self.cfg.get('crown_engine_wbranch', 'off') in ('W', 'WV'):
            history.engine_wbranch_point = x_L.new_full((D, self.n), float('nan'))
            history.engine_wbranch_score = x_L.new_zeros(D, self.n)
        history.engine_boxes = (x_L, x_U)
        lower = thresholds.new_full((D, self.Sg), -INF) if dm_lb is None else dm_lb.clone()
        return (lower, x_L.new_zeros(()).expand(D, self.Sg, *x_L.shape[1:]),
                thresholds.new_full((), -INF).expand(D, self.Sg))

    def _bound_impl(self, x_L, x_U, dm_lb, thresholds, history, deadline=None):
        if self.cfg.get('crown_engine_cascade', False) and not getattr(history, 'engine_root', False):
            from input_split.cascade_cpu import bound_nodes
            return bound_nodes(self, x_L, x_U, dm_lb, thresholds, history, deadline)
        if (not getattr(history, 'engine_root', False) and any(self.cfg.get('crown_engine_'+key, False)
                for key in ('continue', 'nested_faces', 'dual_obbt'))):
            from input_split.cascade_gpu import bound_nodes
            return bound_nodes(self, x_L, x_U, dm_lb, thresholds, history, deadline)
        if self.cfg.get('crown_engine_absorb', False) and not getattr(history, 'engine_root', False):
            from input_split.crown_absorb import bound_nodes
            return bound_nodes(self, x_L, x_U, dm_lb, thresholds, history, deadline)
        if self.cfg.get('crown_engine_node_dsx', False) and not getattr(history, 'engine_root', False):
            from input_split.node_dsx import bound_nodes
            return bound_nodes(self, x_L, x_U, dm_lb, thresholds, history, deadline)
        if self.tiered:
            return self._bound_budgeted(x_L, x_U, dm_lb, thresholds, history, deadline)
        t0 = time.time()
        cg = self.cg
        D = x_L.shape[0]
        xl = x_L.reshape(D, -1).to(self.dev, torch.float64).clone()
        xu = x_U.reshape(D, -1).to(self.dev, torch.float64).clone()
        T = thresholds.reshape(D, -1)[:, 0].to(self.dev, torch.float64)
        st = getattr(history, 'engine_state', None)
        st = st if torch.is_tensor(st) and st.shape[0] == D else None
        lbs, lAs, lbiases, states, witness, stays = [], [], [], [], [], []
        # the solve's deadline: no block of rounds and no OBBT past it (anytime: every lane certified as it stands)
        self._deadline = INF if deadline is None else float(deadline)
        self.lb.deadline = self._deadline
        for a in range(0, D, self.top):
            s = slice(a, min(D, a + self.top))
            tb = time.time()
            sub, caps_in = (None, None) if st is None else self._unpack(st[s])
            d, out, empty, caps = self._bound_fbbt(xl, xu, s, T, caps_in)
            tabs = self._tables(d, T[s])
            plane_b, g, bias = cg.output_bound(d)
            self.stats['seconds_build'] += time.time() - tb
            tl = time.time()
            # a box on which an operator's relaxation is undefined (e.g. a reciprocal's operand interval holding 0)
            # has no LP: no search there (it keeps its parent's bound and state); hs104's level-4 reciprocal sent
            # every lane of 382k to fp64 and back
            # a row of the spec (cutoff, GEAR's, derived) whose rigorous CROWN lower bound exceeds its threshold
            # proves the box empty (the dense engine's sign-start rays; the sparse one has none)
            thr_r = d.thr[:, 2:]
            crown_far = ((out[:, 2:] > thr_r) & torch.isfinite(thr_r) & (self._rowmask_out(T[s]))).any(1)
            empty = empty | (crown_far & ~d.undefined)
            good = ~d.undefined & torch.isfinite(d.lo).all(1) & torch.isfinite(d.hi).all(1) & ~empty
            if bool(good.all()):
                res = self.lb.bound(tabs, sub, T[s], self.c, self.c0)
            else:
                res = self._partial(tabs, sub, T[s], good)
            witness_ready, searching = res.optimal, ~res.optimal
            if self.lb.strict_precision:
                # Snapshot before probe calls reuse the bounding object.
                # A finite fp64 stalled point is a primal hint, never a proof.
                witness_ready = (self.lb.last_reasons == 3) & torch.isfinite(res.witness).all(1)
                searching = self.lb.last_reasons == 0
            self.stats['seconds_lockstep'] += time.time() - tl
            self._chunk_seconds = time.time() - tb
            self.stats['undefined'] = self.stats.get('undefined', 0) + int((~good).sum())
            b = res.lower.shape[0]
            low = res.lower
            if dm_lb is not None:
                low = torch.maximum(low, dm_lb.reshape(D, -1)[s, 0].to(low))      # a child is in its parent's box
            lb = torch.full((b, self.Sg), -INF, dtype=torch.float64, device=self.dev)
            ok = self.grow >= 0
            lb[:, ok] = out[:, self.grow[ok]].to(torch.float64)
            lb[:, 0] = low
            far = (res.kind == 2) | empty                                   # (FBBT's empty box: no point)
            lb[far] = INF
            # CROWN's input planes, moved so that each one's minimum on the box is its rigorous bound
            g = g.expand(b, -1, -1)
            # (priced on the box the bound was computed on: FBBT / OBBT may shrink the domain's box afterwards)
            price = torch.minimum(g * d.lo[:, None, :self.n], g * d.hi[:, None, :self.n]).sum(-1)
            lbias = (plane_b - price).nan_to_num(nan=-1e300, neginf=-1e300)
            lA = torch.zeros(b, self.Sg, self.n, dtype=torch.float64, device=self.dev)
            lB = torch.full((b, self.Sg), -1e300, dtype=torch.float64, device=self.dev)
            lA[:, ok], lB[:, ok] = g[:, self.grow[ok]], lbias[:, self.grow[ok]]
            if self.obbt_targets and time.time() < self._deadline:
                self._obbt(xl, xu, s, T, tabs, res, lb, caps)
            lbs.append(lb), lAs.append(lA), lbiases.append(lB), states.append(self._pack(res.state, caps))
            # (a basic point is the LP's solution only where the lane is optimal; elsewhere it is primal
            # infeasible, often far outside the box: no point)
            witness.append(torch.where(witness_ready[:, None], res.witness, float('nan')))
            # a domain whose LP the search has not solved yet, and whose bound the call raised, stays unsplit
            # (ARCH_ONE_TREE 2.4: branch the optimal and the numerical ones; the others go on in place)
            pb = sub.best if sub is not None else torch.full_like(res.lower, -INF)
            stays.append((res.kind == 0) & searching & ~res.numerical & good & (res.lower > pb))
            for k, v in (('lanes', b), ('bound', int((res.kind == 1).sum())), ('farkas', int(far.sum())),
                         ('fp64', int(res.fp64.sum())), ('numerical', int(res.numerical.sum())),
                         ('optimal', int(res.optimal.sum()))):
                self.stats[k] += v
        self.lb.deadline = INF
        history.engine_state = torch.cat(states)
        history.engine_boxes = (xl.reshape(x_L.shape).to(x_L), xu.reshape(x_U.shape).to(x_U))
        history.engine_witness = torch.cat(witness)
        history.engine_stay = torch.cat(stays) if self.stay else None
        self.calls += 1
        self.stats['seconds'] += time.time() - t0
        lb = torch.cat(lbs)
        if self.calls <= 3 or self.calls % 20 == 0:
            fin = lb[:, 0][torch.isfinite(lb[:, 0])]
            print(f'[crown-engine] call {self.calls}: D={D} pruned bound {int((lb[:, 0] >= T).sum())} '
                  f'farkas {int(torch.isposinf(lb[:, 0]).sum())} '
                  f'min lb {float(fin.min()) if fin.numel() else math.nan:.10g} T {float(T[0]):.10g} '
                  f'rounds {self.lb.rounds_run[-4:]} {time.time() - t0:.3f}s', flush=True)
        dt = thresholds.dtype
        return lb.to(dt), torch.cat(lAs).to(dt).reshape(D, self.Sg, *x_L.shape[1:]), torch.cat(lbiases).to(dt)

    def shave(self, lo, hi, T, passes, deadline=INF, fractions=6):
        """root probing: for every input and side, the slabs [l, l + f w] and [u - f w, u] (f = 1/2 ... 2^-fractions
        of the width w) bounded as boxes of their own (warm from the box's own state); the widest slab whose bound
        reaches the cutoff T, or that has a Farkas certificate, holds no point better than T and is cut off. Up to
        `passes` passes while the box shrinks. Returns the box and whether it shrank (then the caller records T as a
        discarded threshold)."""
        class _H:
            engine_root = True
            engine_state = None
        n, dev = self.n, self.dev
        lo, hi = lo.reshape(1, n).to(dev, torch.float64).clone(), hi.reshape(1, n).to(dev, torch.float64).clone()
        f = 2.0 ** -torch.arange(1, fractions + 1, dtype=torch.float64, device=dev)        # (K,)
        K, shrank = len(f), False
        thr1 = torch.full((1, self.Sg), INF, dtype=torch.float64, device=dev)
        thr1[:, 0] = T
        for p in range(passes):
            if time.time() > deadline:
                break
            self.bound(lo, hi, None, thr1, _H)                                 # the box's own state
            root_state = _H.engine_state
            w = (hi - lo)[0]
            J = torch.nonzero(w > 0).flatten()
            if not J.numel():
                break
            nJ = J.numel()
            jj = J.repeat_interleave(K).repeat(2)                              # (2 nJ K,)
            ff = f.repeat(nJ).repeat(2)
            side = torch.cat([torch.zeros(nJ * K), torch.ones(nJ * K)]).bool().to(dev)
            xl, xu = lo.expand(len(jj), n).clone(), hi.expand(len(jj), n).clone()
            ar = torch.arange(len(jj), device=dev)
            cut = torch.where(side, hi[0, jj] - ff * w[jj], lo[0, jj] + ff * w[jj])
            xu[ar[~side], jj[~side]] = cut[~side]                              # lower slabs [l, l + f w]
            xl[ar[side], jj[side]] = cut[side]                                 # upper slabs [u - f w, u]
            thr = thr1.expand(len(jj), -1).clone()
            _H.engine_state = root_state.expand(len(jj), -1).clone()
            lb = self.bound(xl, xu, None, thr, _H)[0][:, 0]
            gone = (lb >= T) | torch.isposinf(lb)
            # per input and side the widest slab cut off
            best = torch.where(gone, ff, 0.0).reshape(2, nJ, K).amax(2)        # (2, nJ)
            keepdim = best.sum(0) < 1                                          # (the whole box: left to the BaB)
            newl = torch.where(keepdim & (best[0] > 0), lo[0, J] + best[0] * w[J], lo[0, J])
            newu = torch.where(keepdim & (best[1] > 0), hi[0, J] - best[1] * w[J], hi[0, J])
            ch = (newl > lo[0, J]) | (newu < hi[0, J])
            if not bool(ch.any()):
                break
            lo[0, J], hi[0, J] = newl, newu
            shrank = True
            print(f'[crown-engine] shave pass {p}: {int(ch.sum())} of {nJ} inputs tightened, '
                  f'mean width ratio {float(((hi - lo)[0, J] / w[J]).mean()):.4f}', flush=True)
        return lo, hi, shrank

    def root_obbt(self, lo, hi, T, passes, deadline):
        """OBBT of the root box on every input and every element (both sides, the cutoff T), as lanes of the lockstep
        dual (the same certificates as in the tree), up to `passes` passes while the box shrinks and before the
        deadline. Returns the box and the node caps (kept as self.root_caps: the first domains start from them)."""
        if getattr(self, 'cfg', {}).get('crown_engine_m1', False):
            # Production M2 is serviced only at committed tree boundaries.
            return lo, hi
        if getattr(self, 'cfg', {}).get('crown_engine_obbt_dsx', False):
            from input_split.dsx_obbt import root_wave
            return root_wave(self, lo, hi, T, deadline)
        class _H:
            engine_root = True
            engine_state = None
        lo, hi = lo.reshape(1, -1).to(self.dev, torch.float64).clone(), hi.reshape(1, -1).to(self.dev, torch.float64).clone()
        thr = torch.full((1, self.Sg), INF, dtype=torch.float64, device=self.dev)
        thr[:, 0] = T
        keep = (self.obbt_targets, self._share)
        self.obbt_targets, self._share = max(self.n, self.Et), False
        _H.engine_state = self.empty_states(1)
        if self.tiered and self.root_seed is not None:
            seed_lo, seed_hi, seed_state = self.root_seed
            if bool((lo >= seed_lo).all() and (hi <= seed_hi).all()):
                _H.engine_state = seed_state.clone()
        try:
            for p in range(passes):
                if time.time() > deadline:
                    break
                w0 = (hi - lo).clone()
                c0 = None if self.root_caps is None else (self.root_caps[1] - self.root_caps[0]).clone()
                if not self.propagate and not self.tiered:
                    _H.engine_state = self.empty_states(1)
                optimal_before = self.stats['optimal']
                lb, _, _ = self.bound(lo, hi, None, thr, _H, deadline=deadline)
                if getattr(self, 'tier_root_record', False):
                    self.root_obbt_lower = max(self.root_obbt_lower, min(T, float(lb[0, 0])))
                if not bool(torch.isfinite(lb[0, 0])):
                    # the root box itself holds no point better than T (or none at all): nothing to tighten; the
                    # first bounding of the BaB prunes it again (its tightened box and caps would be empty)
                    print(f'[crown-engine] root OBBT pass {p}: root pruned (lb {float(lb[0, 0])})', flush=True)
                    break
                lo, hi = _H.engine_boxes[0].reshape(1, -1).double(), _H.engine_boxes[1].reshape(1, -1).double()
                if self.tiered:
                    self.root_seed = (lo.clone(), hi.clone(), _H.engine_state.clone())
                if self.node_caps:
                    st, caps = self._unpack(_H.engine_state)
                    self.root_caps = (caps[0].clone(), caps[1].clone())
                shrunk = bool(((hi - lo) < w0).any()) or (c0 is not None and self.root_caps is not None and
                                                          bool(((self.root_caps[1] - self.root_caps[0]) < c0).any()))
                if self.propagate and c0 is None and self.root_caps is not None:
                    shrunk = True  # first node caps need a rebuild even when the input hull did not shrink
                print(f'[crown-engine] root OBBT pass {p}: lb {float(lb[0, 0]):.10g} box width ratio '
                      f'{float(((hi - lo) / w0.clamp(min=1e-300))[w0 > 0].mean()) if bool((w0 > 0).any()) else 1:.4f}',
                      flush=True)
                if not shrunk and (not self.propagate or self.stats['optimal'] > optimal_before):
                    break
        finally:
            self.obbt_targets, self._share = keep
        return lo, hi

    def lp_vertex(self, lo, hi, deadline=None):
        """the LP solution of the box (one lane, no cutoff, its search to the optimum, OBBT off): its input columns,
        or None where the lane does not reach the optimum"""
        class _H:
            engine_root = True
            engine_state = None
        thr = torch.full((1, self.Sg), INF, dtype=torch.float64, device=self.dev)
        keep, self.obbt_targets = self.obbt_targets, 0
        try:
            self.bound(lo.reshape(1, -1), hi.reshape(1, -1), None, thr, _H, deadline=deadline)
            if self.tiered:
                self.root_seed = (lo.clone(), hi.clone(), _H.engine_state.clone())
        finally:
            self.obbt_targets = keep
        xw = getattr(_H, 'engine_witness', None)
        if xw is None or not bool(torch.isfinite(xw[0, :self.n]).all()):
            return None
        return torch.minimum(torch.maximum(xw[0, :self.n], lo.reshape(-1).to(xw)), hi.reshape(-1).to(xw))

    def branch(self, x_L, x_U, history):
        """the split dimension (B, 1) of each domain by the engine's rule: 'widest' the largest width relative to the
        root box; 'mcc' at the domain's last basic point (history.engine_witness) the product of largest McCormick
        violation |z - a b|, then of its operands' inputs the one whose width contributes most to the product's
        relaxation (|a_j| w_j W_b + |b_j| w_j W_a, W the operand widths); 'widest' where no point or no violation"""
        if self.cfg.get('crown_engine_m1', False):
            from input_split.m1_qp import branch
            return branch(self, x_L, x_U, history)
        if self.cfg.get('crown_engine_absorb', False):
            from input_split.crown_absorb import branch
            return branch(self, x_L, x_U, history)
        B = x_L.shape[0]
        lo, hi = x_L.reshape(B, -1).to(self.dev, torch.float64), x_U.reshape(B, -1).to(self.dev, torch.float64)
        w = hi - lo
        root = getattr(history, 'root_width', None)
        rel = w / root.to(w).reshape(1, -1).clamp(min=1e-300) if root is not None else w
        rel = torch.where(w > 0, rel, -1.0)
        pick = rel.argmax(1)
        xw = getattr(history, 'engine_witness', None)
        if self.branching == 'mcc' and torch.is_tensor(xw) and xw.shape[0] == B:
            xw = xw.to(self.dev, torch.float64)
            n = self.n
            viol = self._violation(xw)
            e = viol.argmax(1)
            M = self.cg.Mop[:, :, :n].abs()                               # (2, Et, n) on the inputs
            Me = M[:, e].transpose(0, 1)                                   # (B, 2, n)
            Wop = (Me * w[:, None]).sum(-1)                                # (B, 2) operand widths
            score = Me[:, 0] * w * Wop[:, 1:2] + Me[:, 1] * w * Wop[:, 0:1]
            score = torch.where((self.sq | self.recip)[e, None], Me[:, 0]*w, score)
            j = torch.where(w > 0, score, -1.0).argmax(1)
            ok = (viol.amax(1) > 0) & (score.amax(1) > 0)
            pick = torch.where(ok, j, pick)
        return pick[:, None].to(x_L.device)

    def _violation(self, xw):
        """Nonlinear residuals at a finite optimal witness; used only to choose work."""
        op = self.cg.Lop.right(torch.nan_to_num(xw)).unflatten(-1, (2, self.Et)) + self.cg.cop[None]
        z = xw[:, self.n:]
        value = torch.where(self.mul, op[:, 0]*op[:, 1],
                            torch.where(self.sq, op[:, 0].square(), 1/op[:, 0]))
        gap = torch.where(self.mul | self.sq | self.recip, (z-value).abs(), 0.0)
        return torch.where(torch.isfinite(xw).all(1, keepdim=True), gap.nan_to_num(0.0, 0.0, 0.0), 0.0)

    def report(self):
        s = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.stats.items()}
        print(f'[crown-engine] calls={self.calls} {s}', flush=True)
        if self.lb is not None and self.lb.k_stats is not None:
            print(f'[crown-engine] Schur {self.lb.k_stats}', flush=True)
        if self.lb is not None and self.lb.work_stats:
            print(f'[crown-engine] structured/subopt {dict(self.lb.work_stats)}', flush=True)


def root_lp_vertex(root):
    """An optional GPU CROWN primal proposal; unfinished starts supply no point."""
    from contextlib import nullcontext
    from auto_LiRPA.deadline import BoundDeadline, bound_deadline
    from auto_LiRPA.cuda_memory import release_cuda
    t0 = time.time()
    cfg = getattr(root, 'nlp_config', {})
    scheduled = cfg.get('crown_engine_plane_schedule', False)
    allowance = (max(0., float(cfg.get('crown_engine_capture_seconds', 10.)))
                 + max(0., float(cfg.get('crown_engine_upgrade_seconds', 5.))))
    deadline = min(root.deadline, t0+allowance) if scheduled else getattr(root, 'deadline', None)
    eng = x = None
    try:
        with bound_deadline(deadline) if scheduled else nullcontext():
            eng = build_engine(root, tangent=False, report=False)
            if eng is not None and eng.lb is not None:
                x = eng.lp_vertex(root.dm_l[:1].double(), root.dm_u[:1].double(),
                                  deadline=(deadline if scheduled else root.deadline) if eng.tiered else None)
    except BoundDeadline:
        if not scheduled or time.time() >= root.deadline:
            raise
        print('[crown-engine] root LP start budget exhausted; using native primal starts', flush=True)
    finally:
        if eng is not None and eng.tiered:
            root._seed_engine = eng
        else:
            eng = None
        if scheduled:
            release_cuda(root.dm_l.device)
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    print(f'[crown-engine] root LP vertex: {"found" if x is not None else "none"} {time.time() - t0:.2f}s', flush=True)
    return None if x is None else x.reshape(root.dm_l[:1].shape).to(root.dm_l)


def build_engine(root, tangent=True, report=True, cfg=None):
    """the engine of a GEAR root (RootState), or None (the reason printed); tangent: the objective's tangent row
    from the root's QTB / convex tangent when there is one"""
    try:
        net = root.net.net
        lo, hi = root.dm_l[:1].reshape(1, -1).double(), root.dm_u[:1].reshape(1, -1).double()
        n = lo.shape[1]
        C = root.c[0].double()
        thr = root.rowpool.thresholds(root.rhs[:1].double()).reshape(-1)
        E = h = None
        base = root.base_input_constraints
        if base is not None:
            E, h = base[0][:1].reshape(-1, n).double(), -base[1][:1].reshape(-1).double()
        tiered = bool(root.nlp_config.get('crown_engine_tiered', False))
        cached = getattr(root, '_seed_engine', None) if tiered else None
        q = getattr(root.history, 'qtb', None) if tangent else None
        need_tangent = q is not None and not hasattr(q, 'evaluator') and hasattr(q, 'g') and hasattr(q, 'plane')
        if cached is not None:
            oldE, oldh = cached.source_rows
            same_rows = ((E is None and oldE is None) or (E is not None and oldE is not None
                         and torch.equal(E, oldE) and torch.equal(h, oldh)))
            if (not need_tangent and same_rows and torch.equal(C, cached.source_C)
                    and torch.equal(thr[1:], cached.source_thr[1:])
                    and bool((lo >= cached.root_lo).all() and (hi <= cached.root_hi).all())):
                root._seed_engine = None
                if report:
                    atexit.register(cached.report)
                print('[crown-engine-cache] root seed graph/basis/caps/dictionary/captures reused', flush=True)
                if cached.cfg.get('crown_engine_cascade'):
                    from input_split.cascade_cpu import initialize
                    initialize(cached, root)
                return cached
            # Changed rows/tangent change the LP pattern. Never reuse their
            # labels or certificates as if shape alone established validity.
            print('[crown-engine-cache] root pattern changed; new tables required', flush=True)
            root._seed_engine = None
            if not (torch.equal(C, cached.source_C) and torch.equal(thr[1:], cached.source_thr[1:])
                    and bool((lo >= cached.root_lo).all() and (hi <= cached.root_hi).all())):
                cached = None
        rr = None
        try:
            from nlp_solver.rational_rows import extract_rational_rows
            pts = (lo + (hi - lo) * torch.as_tensor(np.random.default_rng(0).random((8, n)), dtype=lo.dtype,
                                                     device=lo.device))
            rr = extract_rational_rows(net, root.net.model_ori, ((lo + hi) / 2).to(root.dm_l), C[None].to(lo.device),
                                       pts.to(root.dm_l))
        except Exception as e:  # pylint: disable=broad-except
            print(f'[crown-engine] no rational rows ({type(e).__name__}: {e})', flush=True)
        want_tangent, tangent = tangent, None
        q = getattr(root.history, 'qtb', None) if want_tangent else None
        if q is not None and not hasattr(q, 'evaluator') and hasattr(q, 'g') and hasattr(q, 'plane'):
            import copy
            tangent = copy.copy(q)                        # (frozen at its root anchor: GEAR may move the original)
            for k in ('x_hat', 'g', 'g_error'):
                if torch.is_tensor(getattr(tangent, k, None)):
                    setattr(tangent, k, getattr(tangent, k).clone())
        engine = CrownEngine(net, C.to(lo.device), thr.to(lo.device), lo, hi, E, h, rational_rows=rr, tangent=tangent,
                             report=report, cfg=cfg, **(dict(seed=cached) if tiered else {}))
        if tiered:
            engine.source_C, engine.source_thr = C.clone(), thr.clone()
            engine.source_rows = (None, None) if E is None else (E.clone(), h.clone())
        if engine.cfg.get('crown_engine_cascade'):
            from input_split.cascade_cpu import initialize
            initialize(engine, root)
        return engine
    except (EngineUnavailable, MemoryBudgetExceeded) as e:
        print(f'[crown-engine] unavailable, GEAR bounding instead: {e}', flush=True)
    except Exception as e:  # pylint: disable=broad-except
        print(f'[crown-engine] build failed, GEAR bounding instead: {type(e).__name__}: {e}\n'
              f'{traceback.format_exc()[-3000:]}', flush=True)
    return None


# ---------------------------------------------------------------------------------------------------- CPU LP counter
LP_CALLS = {'dual': 0, 'primal': 0, 'primal_lanes': 0}
_SITES = set()
_PRIMAL = ('lp_alternation.py', 'compute_root_lp_point')


def _side():
    f = sys._getframe(2)
    while f is not None:
        if f.f_code.co_filename.endswith(_PRIMAL[0]) or f.f_code.co_name == _PRIMAL[1]:
            return 'primal', f
        f = f.f_back
    return 'dual', sys._getframe(3)


def _counted(fn, name):
    def wrapper(*a, **k):
        side, f = _side()
        LP_CALLS[side] += 1
        site = (side, name, f.f_code.co_filename.rsplit('/', 1)[-1], f.f_code.co_name)
        if site not in _SITES:
            _SITES.add(site)
            print(f'[cpu-lp] first call: side={side} solver={name} site={site[2]}:{site[3]}', flush=True)
        return fn(*a, **k)
    wrapper._lp_counted = True
    return wrapper


def install_lp_counter():
    """count every CPU LP solve of this process (scipy.optimize.linprog, highspy.Highs.run) and every lp_alternation
    child lane (a separate process of HiGHS LPs, primal side); the totals are printed at exit"""
    import scipy.optimize
    if getattr(scipy.optimize.linprog, '_lp_counted', False):
        return
    scipy.optimize.linprog = _counted(scipy.optimize.linprog, 'linprog')
    try:
        import highspy
        highspy.Highs.run = _counted(highspy.Highs.run, 'highs')
    except ImportError:
        pass
    try:
        from nlp_solver import lp_alternation
        start = lp_alternation.start

        def lanes(*a, **k):
            LP_CALLS['primal_lanes'] += 1
            return start(*a, **k)
        lp_alternation.start = lanes
    except ImportError:
        pass
    atexit.register(lambda: print(f'[cpu-lp] calls dual={LP_CALLS["dual"]} primal={LP_CALLS["primal"]} '
                                  f'primal_lanes={LP_CALLS["primal_lanes"]} pid={os.getpid()}', flush=True))
