"""Sparse graph-to-diagonal-QP adapter, with enclosed affine folds.

Eligible BoundSqr leaves occur only in positive objective/cutoff rows and
their own faces. Single-column affine operands expand into diagonal,
linear and constant terms with an outward fold-error enclosure. A multi-column operand gets one auxiliary
column and two enclosed affine identity rows; its curvature stays diagonal.
All other graph rows remain polyhedral. No ancestor face pool is retained.
"""
import math
import sys
import torch
from .restarted_qp import product
from .cap_log import cap_log

def available_bytes(device=None):
    """free device memory in bytes, or None on CPU (no cap). device None: the current CUDA device if any"""
    device = torch.device(device) if device is not None else (
        torch.device('cuda', torch.cuda.current_device()) if torch.cuda.is_available() else None)
    if device is None or device.type != 'cuda':
        return None
    return torch.cuda.mem_get_info(device)[0]


def table_bytes(m, n, nnz):
    """resident bytes of one table: per nonzero row, col and slope (int64) and value (float64); per row ref and
    virtual-row id (int64)"""
    return 8*(4*nnz+2*m)


def lane_bytes(m, n, nnz):
    """bytes of one lane built from the table (build): its values, rhs, c, q, lower, upper"""
    return 8*(nnz+m+4*n)


def check_capacity(site, m, n, nnz, action='raise, M1/M2 tables off', device=None):
    """raise (after one [cap] line) when the table plus one lane do not fit the available device memory; never on CPU"""
    free = available_bytes(device)
    need = table_bytes(m, n, nnz)+lane_bytes(m, n, nnz)
    if free is not None and need > free:
        cap_log(site, 'bytes', free, need, action)
        raise ValueError(('graph QP' if site == 'graph_qp' else 'epigraph')+' exceeds fixed table capacity')


class GraphQP:
    @staticmethod
    def _normal_expansion(weight, scale, shift):
        # Relative-error bounds require normal intermediate products. Keep
        # the affine identity instead when expansion could under/overflow.
        def multiply(a, b):
            v = a*b
            if not math.isfinite(v) or (a != 0 and b != 0 and abs(v) < sys.float_info.min):
                raise ArithmeticError
            return v
        try:
            multiply(multiply(weight, scale), scale)
            multiply(multiply(multiply(2., weight), scale), shift)
            multiply(multiply(weight, shift), shift)
            return True
        except ArithmeticError:
            return False

    def __init__(self, cg, square_elements=None):
        from nlp_solver.rigorous_dual import Rows
        self.cg = cg; E, N = cg.Et, cg.ncol
        op_r, op_c, op_v = [t.detach().cpu() for t in cg._op()]
        cr, cc, cv = [t.detach().cpu() for t in cg._cm()]
        cm0 = cg.CM0.detach().cpu()
        if square_elements is None:
            square_elements = [e for nd in cg.nodes if type(cg.net[nd.name]).__name__ == 'BoundSqr'
                               for e in range(nd.s, nd.s+nd.E)]
        consumed = set(op_c[op_v != 0].tolist()) | set(cc[(cr >= 3) & (cv != 0)].tolist())
        chosen = [e for e in square_elements if cm0[cg.n+e] > 0 and cg.n+e not in consumed]
        self.chosen = torch.tensor(chosen, dtype=torch.long, device=cg.cop.device)
        removed = {cg.n+e for e in chosen}
        kept = [j for j in range(N) if j not in removed]
        index = {j: k for k, j in enumerate(kept)}
        op = [[] for _ in range(2*E)]
        for r, c, v in zip(op_r.tolist(), op_c.tolist(), op_v.tolist()):
            if v: op[r].append((c, v))
        # Singleton affine operands need no additional column. Multi-column
        # operands use a band enclosing the compiled affine-fold error.
        direct, virtual, scales, shifts = {}, [], [], []
        for e in chosen:
            terms = op[e]
            if len(terms) == 1 and self._normal_expansion(
                    float(cm0[cg.n+e]), terms[0][1], float(cg.cop[0, e])):
                direct[e] = index[terms[0][0]]
                scales.append(terms[0][1]); shifts.append(float(cg.cop[0, e]))
            else:
                virtual.append(e); scales.append(1.); shifts.append(0.)
        self.virtual = torch.tensor(virtual, dtype=torch.long, device=cg.cop.device)
        destination = [direct[e] if e in direct else len(kept)+virtual.index(e) for e in chosen]
        self.destination = torch.tensor(destination, dtype=torch.long, device=cg.cop.device)
        self.scales = torch.tensor(scales, dtype=torch.float64, device=cg.cop.device)
        self.shifts = torch.tensor(shifts, dtype=torch.float64, device=cg.cop.device)
        self.singleton = torch.tensor([e in direct for e in chosen], dtype=torch.bool, device=cg.cop.device)
        self.keep = torch.tensor(kept, dtype=torch.long, device=cg.cop.device)
        self.n = len(kept)+len(virtual)
        rr, cols, vals, slopes, refs, virtual_row = [], [], [], [], [], []

        def entry(r, col, value, slope=-1):
            rr.append(r); cols.append(col); vals.append(value); slopes.append(slope)

        # Faces remain factorised by their sparse operand forms. Duplicate
        # COO entries are intentional: no dense m*n matrix is assembled.
        for upper, mask in ((False, cg.vl.cpu()), (True, cg.vu.cpu())):
            for f, e in mask.nonzero().tolist():
                if e in chosen: continue
                row = len(refs); refs.append((cg.Fl+f if upper else f)*E+e); virtual_row.append(-1)
                sign = -1 if upper else 1
                for p in range(2):
                    slot = ((cg.Fl+f if upper else f)*2+p)*E+e
                    for j, v in op[p*E+e]: entry(row, index[j], sign*v, slot)
                entry(row, index[cg.n+e], -sign)
        base = (cg.Fl+cg.Fu)*E
        # one stable argsort per entry list; rows read contiguous segments (entry order within a row is unchanged)
        slot_of = torch.full((N,), -1, dtype=torch.long); slot_of[torch.tensor(kept, dtype=torch.long)] = torch.arange(len(kept))
        specs = cg.rows.cpu()
        order = torch.argsort(cr, stable=True)
        cr_s, cc_s, cv_s = cr[order], slot_of[cc[order]], cv[order]
        used = torch.isin(cr_s, specs[(specs != 1) & (specs != 2)])
        assert not (used & (cc_s < 0)).any(), 'row refers to an eliminated column'
        lo_s, hi_s = torch.searchsorted(cr_s, specs), torch.searchsorted(cr_s, specs, right=True)
        cc_l, cv_l, lo_l, hi_l = cc_s.tolist(), cv_s.tolist(), lo_s.tolist(), hi_s.tolist()
        for pos, spec in enumerate(specs.tolist()):
            if spec in (1, 2): continue   # reserved objective and cutoff
            row = len(refs); refs.append(base+pos); virtual_row.append(-1)
            for q in range(lo_l[pos], hi_l[pos]): entry(row, cc_l[q], cv_l[q])
        er, ec = (cg.E.detach().cpu() != 0).nonzero(as_tuple=True)
        order = torch.argsort(er, stable=True)
        er_s, ec_s = er[order], ec[order]
        ev_l = cg.E.detach().cpu()[er_s, ec_s].tolist(); ec_l = slot_of[ec_s].tolist()
        rows_r = torch.arange(cg.R)
        lo_l, hi_l = torch.searchsorted(er_s, rows_r).tolist(), torch.searchsorted(er_s, rows_r, right=True).tolist()
        for r in range(cg.R):
            row = len(refs); refs.append(base+len(cg.rows)+r); virtual_row.append(-1)
            for q in range(lo_l[r], hi_l[r]): entry(row, ec_l[q], ev_l[q])
        for k, e in enumerate(virtual):
            for sign in (1., -1.):
                row = len(refs); refs.append(-1); virtual_row.append(k)
                entry(row, len(kept)+k, sign)
                for j, v in op[e]: entry(row, index[j], -sign*v)
        dev = cg.cop.device
        integer = lambda a: torch.tensor(a, dtype=torch.long, device=dev)
        self.row, self.col = integer(rr), integer(cols)
        self.values = torch.tensor(vals, dtype=torch.float64, device=dev)
        self.slopes, self.refs, self.virtual_row = integer(slopes), integer(refs), integer(virtual_row)
        self.m = len(refs)
        self.virtual_count = len(virtual)*2
        self.virtual_rows = integer([k for k,v in enumerate(virtual_row) if v >= 0])
        self.op_rows = Rows(op_r.to(dev), op_c.to(dev), op_v.to(dev), 2*E, N)
        self.rows_type = Rows
        check_capacity('graph_qp', self.m, self.n, len(rr), device=dev)

    def build(self, d):
        from nlp_solver.rigorous_dual import gamma
        cg, B, E = self.cg, len(d.lo), self.cg.Et
        up = lambda x: torch.nextafter(x, torch.full_like(x, float('inf')))
        down = lambda x: torch.nextafter(x, torch.full_like(x, -float('inf')))
        M = torch.maximum(d.lo.abs(), d.hi.abs())
        g = gamma(cg.ncol+16)
        operand_error = up((cg.LdG.right(M)+cg.dg)*(1+g)+2.**-1000)
        opv = self.op_rows.val
        terms_l = torch.minimum(opv*d.lo[:, self.op_rows.col], opv*d.hi[:, self.op_rows.col])
        terms_u = torch.maximum(opv*d.lo[:, self.op_rows.col], opv*d.hi[:, self.op_rows.col])
        ol = d.lo.new_zeros(B, 2*E).index_add(1, self.op_rows.row, terms_l)
        ou = d.lo.new_zeros(B, 2*E).index_add(1, self.op_rows.row, terms_u)
        oa = d.lo.new_zeros(B, 2*E).index_add(1, self.op_rows.row,
                                             torch.maximum(terms_l.abs(), terms_u.abs()))
        cop = cg.cop.flatten()
        error = up(operand_error[:, :2*E]+g*(oa+cop.abs())+2.**-1000)
        ol, ou = down(ol+cop-error), up(ou+cop+error)
        lower = torch.cat([d.lo[:, self.keep], ol[:, self.virtual]], 1)
        upper = torch.cat([d.hi[:, self.keep], ou[:, self.virtual]], 1)
        scale = torch.cat([d.lw, d.uw], 1).flatten(1)
        multipliers = (torch.where(self.slopes[None] >= 0, scale[:, self.slopes.clamp_min(0)], 1.)
                       if scale.shape[1] else d.lo.new_ones(B, len(self.values)))
        values = self.values[None]*multipliers
        rows = self.rows_type(self.row, self.col, values, self.m, self.n)
        d.derive(cg.cop).rhs(cg.rows, cg.Ccr)
        slack = cg._fold_slack(d)
        slack_flat = torch.cat([slack[0].flatten(1), slack[1].flatten(1), slack[2], slack[4]], 1)
        ref = self.refs.clamp_min(0)
        rhs = up(d.bflat[:, ref]+slack_flat[:, ref])
        # Enclose slope*M coefficient products and folded row constants.
        constmag = torch.cat([(d.lb.abs()+(d.lw*cg.cop).abs().sum(2)).flatten(1),
            (d.ub.abs()+(d.uw*cg.cop).abs().sum(2)).flatten(1),
            d.thr[:, cg.rows].abs()+cg.Ccr.abs(), d.h.abs()], 1)
        mag = product(rows, values.abs(), torch.maximum(lower.abs(), upper.abs()))
        rhs = up(rhs+up(gamma(12)*(mag+constmag[:, ref])+2.**-1000))
        if self.virtual.numel():
            vr = self.virtual_rows
            ids = self.virtual_row[vr]
            # Rows were emitted as t-a*z <= b+error, -t+a*z <= -b+error.
            sign = (1-2*(torch.arange(self.virtual_count, device=d.lo.device) % 2)).to(d.lo.dtype)
            ee = self.virtual[ids]
            rhs[:, vr] = up(sign*cg.cop[0, ee]+operand_error[:, ee]
                            +gamma(8)*(cg.cop[0, ee].abs()+operand_error[:, ee])+2.**-1000)
        c = torch.cat([cg.CM0[self.keep][None].expand(B, -1), lower.new_zeros(B, len(self.virtual))], 1).clone()
        q = torch.zeros_like(c)
        c0 = down(cg.Cc[0]-slack[3])
        if self.chosen.numel():
            weights = cg.CM0[cg.n+self.chosen][None].expand(B, -1)
            # A one-column affine operand has no cross curvature: expand
            # w=(a*z+b)^2 directly, and enclose all fold/arithmetic error.
            qt = weights*self.scales*self.scales
            ct = 2*weights*self.scales*self.shifts
            kt = weights*self.shifts*self.shifts
            base_c = c.clone()
            c.index_add_(1, self.destination, ct)
            q.index_add_(1, self.destination, qt)
            gq = gamma(len(self.chosen)+16)
            q = down(q*(1-gq)).clamp_min(0.)
            cmag = base_c.abs().index_add(1, self.destination, ct.abs())
            linear_error = (up(gq*cmag)*torch.maximum(lower.abs(), upper.abs())).sum(1)
            delta = torch.where(self.singleton, operand_error[:, self.chosen], 0.)
            weighted_delta = weights*delta
            unsafe = (weights != 0) & (delta != 0) & (
                (weighted_delta.abs() < torch.finfo(torch.float64).tiny) | ~torch.isfinite(weighted_delta))
            if hasattr(cg, '_capture_invalid'):
                cg._capture_invalid.logical_or_(unsafe.any(1))
            elif bool(unsafe.any()):
                raise ValueError('strict QP affine-fold product is unsupported')
            reach = torch.maximum(ol[:, self.chosen].abs(), ou[:, self.chosen].abs())
            fold_error = (weights*delta*(2*reach+delta)).sum(1)
            const = kt.sum(1)
            ge = gamma(cg.ncol+2*len(self.chosen)+64)
            err = up((linear_error+fold_error+gamma(len(self.chosen)+32)*
                      (cg.Cc[0].abs()+const.abs()+slack[3])+2.**-1000)*(1+ge))
            c0 = down((cg.Cc[0]+const-slack[3])-err)
        return rows, c, c0, rhs, lower, upper, q

    def expand(self, z):
        full = z.new_zeros(len(z), self.cg.ncol)
        full[:, self.keep] = z[:, :len(self.keep)]
        if self.chosen.numel():
            full[:, self.cg.n+self.chosen] = (z[:, self.destination]*self.scales+self.shifts).square()
        return full
