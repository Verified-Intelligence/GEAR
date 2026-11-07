"""Scoped LP-to-CROWN transport. Row banks are compressed once, never per domain.

The LP producer owns row validity/tolerances. This module treats its float data
as exact, encloses compression error, and checks box/cutoff scope at every use.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import copy
import hashlib

import numpy as np
import torch

from nlp_solver.rigorous_dual import Rows, dual_lower_bound, round_down
from nlp_solver.structure.rowpool import round_up


def _fingerprint(*parts):
    digest = hashlib.sha256()
    for part in parts:
        digest.update(np.asarray(part).tobytes())
    return digest.hexdigest()


def exact_residual(data, y):
    c, c0, G, h, _, _ = data
    y = np.asarray(y, dtype=np.float64)
    if not np.isfinite(y).all():
        raise ValueError('nonfinite LP multiplier')
    y = np.maximum(y, 0)
    r, kappa = list(map(lambda v: F(float(v)), c)), F(float(c0))
    G = G.tocsr()
    for i in np.flatnonzero(y):
        yy = F(float(y[i]))
        kappa -= yy * F(float(h[i]))
        for k in range(G.indptr[i], G.indptr[i + 1]):
            r[G.indices[k]] += yy * F(float(G.data[k]))
    return r, kappa


class LiftedMap:
    """Stable column identities and interval definitions, shared by all scopes."""
    def __init__(self, lp):
        self.n, self.nb, self.N = lp.n, lp.nb, lp.N
        self.a, self.b = lp.a.detach().cpu().clone(), lp.b.detach().cpu().clone()
        self.levels = [v.detach().cpu().clone() for v in lp.levels]
        self.atoms = copy.deepcopy(lp.atoms)
        self.column_version = _fingerprint(self.a, self.b, str(getattr(self.atoms, 'args', None)),
                                           str(getattr(self.atoms, 'kind', None)),
                                           str(getattr(self.atoms, 'expo', None)), self.N)
        self.column_map = tuple([('input', j) for j in range(self.n)] +
                                [('atom', self.atoms.kind[j], self.atoms.expo[j],
                                  tuple(self.atoms.args[j][0]), tuple(self.atoms.args[j][1]),
                                  self.atoms.args[j][2]) for j in range(self.nb - self.n)] +
                                [('product', int(a), int(b)) for a, b in zip(self.a, self.b)] +
                                [('epigraph', j) for j in range(self.N - self.nb - len(self.a))])

    def intervals(self, lower, upper, aux_lower, aux_upper):
        """Outward interval propagation, intersecting each column before its users."""
        l, u = lower.double().flatten(1), upper.double().flatten(1)
        B = len(l)
        al, au = aux_lower.to(l), aux_upper.to(l)
        zl, zu = al.expand(B, -1).clone(), au.expand(B, -1).clone()
        zl[:, :self.n] = torch.maximum(l, al[:self.n])
        zu[:, :self.n] = torch.minimum(u, au[:self.n])
        if self.atoms is not None:
            from nlp_solver.lifted_atoms import _value_box
            constants = self.atoms.arrays(torch, zl)
            for ks in constants['levels']:
                lo, hi = self.atoms._range(constants, ks, zl[:, :self.nb], zu[:, :self.nb], torch)
                low, high = _value_box(constants['code'][ks], constants['expo'][ks], lo, hi, torch)
                zl[:, self.n + ks] = torch.maximum(low, al[self.n + ks])
                zu[:, self.n + ks] = torch.minimum(high, au[self.n + ks])
        a, b = self.a.to(l.device), self.b.to(l.device)
        for ks in self.levels:
            ks = ks.to(l.device)
            ia, ib = a[ks], b[ks]
            la, ua, lb, ub = zl[:, ia], zu[:, ia], zl[:, ib], zu[:, ib]
            corners = torch.stack((la * lb, la * ub, ua * lb, ua * ub), -1)
            low = torch.nextafter(corners.amin(-1), l.new_tensor(-torch.inf))
            high = torch.nextafter(corners.amax(-1), l.new_tensor(torch.inf))
            square = ia == ib
            low = torch.where(square & (la <= 0) & (ua >= 0), 0., low)
            zl[:, self.nb + ks] = torch.maximum(low, al[self.nb + ks])
            zu[:, self.nb + ks] = torch.minimum(high, au[self.nb + ks])
        return zl, zu


@dataclass
class DomainCertificate:
    scope_id: int
    ancestor_id: object
    scope_lower: torch.Tensor
    scope_upper: torch.Tensor
    P0: float
    cutoff_version: object
    row_version: str
    columns: LiftedMap
    scalar_bound: float
    kappa: float
    kappa_error: float
    residual_r: torch.Tensor
    residual_error: torch.Tensor
    aux_lower: torch.Tensor
    aux_upper: torch.Tensor
    y: np.ndarray
    input_cut: object = None

    @property
    def column_version(self):
        return self.columns.column_version

    @property
    def column_map(self):
        return self.columns.column_map

    @classmethod
    def export(cls, scope_id, columns, lower, upper, data, y, cutoff=None, ancestor_id=None):
        r, kappa = exact_residual(data, y)
        rf = np.array([float(v) for v in r])
        error = np.array([round_up(abs(v - F(float(w)))) for v, w in zip(r, rf)])
        kf = float(kappa)
        ke = round_up(abs(kappa - F(kf)))
        c, c0, G, h, lz, uz = data
        tensor = lambda v: torch.as_tensor(np.array(v, copy=True), dtype=torch.float64)
        scalar, source_error, _ = dual_lower_bound(
            Rows.from_scipy(G), tensor(c)[None], tensor([c0]), tensor(h)[None],
            tensor(lz)[None], tensor(uz)[None], tensor(np.maximum(y, 0))[None])
        scalar = float(scalar[0])
        # Retain the source evaluator's outward error budget. Compression must
        # not recover numerical slack that the producer deliberately enclosed.
        ke = (round_up(F(ke) + F(float(source_error[0])))
              if torch.isfinite(source_error[0]) else float('inf'))
        return cls(scope_id, ancestor_id, tensor(lower).flatten(), tensor(upper).flatten(),
                   float('inf') if cutoff is None else float(cutoff), cutoff,
                   _fingerprint(G.data, G.indices, G.indptr, h, c, c0), columns,
                   scalar, kf, ke, tensor(rf), tensor(error), tensor(lz), tensor(uz),
                   np.maximum(np.asarray(y), 0).copy())

    def applies(self, lower, upper, cutoff):
        l, u = lower.flatten(1).double(), upper.flatten(1).double()
        use = (l >= self.scope_lower.to(l)).all(1) & (u <= self.scope_upper.to(u)).all(1)
        if np.isfinite(self.P0):
            use &= torch.as_tensor(cutoff, device=l.device, dtype=torch.float64).reshape(-1) <= self.P0
        return use

    def evaluate(self, lower, upper):
        l, u = self.columns.intervals(lower, upper, self.aux_lower, self.aux_upper)
        B, N = l.shape
        r = self.residual_r.to(l).expand(B, -1)
        empty = l.new_empty(B, 0)
        rows = Rows(torch.empty(0, dtype=torch.long, device=l.device),
                    torch.empty(0, dtype=torch.long, device=l.device), empty, 0, N)
        result = dual_lower_bound(rows, r, l.new_full((B,), self.kappa), empty, l, u, empty,
                                  c_err=self.residual_error.to(l).expand(B, -1),
                                  c0_err=l.new_full((B,), self.kappa_error))[0]
        # The scalar is another certificate on this same scope. Its monotonic
        # inheritance is exact even when the compressed sum has more rounding.
        result = torch.maximum(result, result.new_tensor(self.scalar_bound))
        result[(l > u).any(1)] = torch.inf
        return result.clamp(max=self.P0)


class LeafCertificates:
    """Uncompressed proposals owned by one open LP leaf, never its dead parents."""
    def __init__(self, ancestor_id=None):
        self.ancestor_id = ancestor_id
        self.records = []
        self.last_id = ancestor_id
        self.order = -1

    def capture(self, lower, upper, data, y, cutoff):
        # OBBT mutates these endpoints after the objective solve. The row bank,
        # objective and multipliers are immutable and can be retained by reference.
        data = (*data[:4], data[4].copy(), data[5].copy())
        self.records.append((lower.copy(), upper.copy(), data, y, cutoff))

    def export(self, registry):
        for lower, upper, data, y, cutoff in self.records:
            cert = registry.add(lower, upper, data, y, cutoff, self.last_id)
            self.last_id = cert.scope_id
        self.records.clear()


class ScopeIndex:
    """A box hierarchy: containment queries visit only intersecting subtrees.

    Exact scope/cutoff checks still happen in applies(); this index only rejects
    impossible candidates. Overlapping scopes and shared split faces are retained.
    """
    def __init__(self, scopes):
        self.lower = np.stack([s.scope_lower.numpy() for s in scopes])
        self.upper = np.stack([s.scope_upper.numpy() for s in scopes])
        self.cutoff = np.array([s.P0 for s in scopes])
        centers = self.lower / 2 + self.upper / 2

        def build(ids):
            lo, hi = self.lower[ids].min(0), self.upper[ids].max(0)
            if len(ids) <= 8:
                return lo, hi, ids, None
            spread = np.ptp(centers[ids], axis=0) / np.maximum(hi - lo, 1e-300)
            axis = int(spread.argmax())
            ids = ids[np.argsort(centers[ids, axis], kind='stable')]
            mid = len(ids) // 2
            return lo, hi, build(ids[:mid]), build(ids[mid:])
        self.root = build(np.arange(len(scopes)))

    def matches(self, lower, upper, cutoff):
        """Sparse (domain, scope) pairs, with exact containment and cutoff checks.

        Keep the pairs found by the hierarchy instead of throwing away the domain
        indices and testing every candidate against the entire batch a second time.
        All comparisons are float64, as in DomainCertificate.applies.
        """
        l = lower.detach().flatten(1).cpu().double().numpy()
        u = upper.detach().flatten(1).cpu().double().numpy()
        p = torch.as_tensor(cutoff, dtype=torch.float64).detach().cpu().numpy().reshape(-1)
        p = np.broadcast_to(p, (len(l),))
        domains, scopes = [], []

        def visit(node, batch):
            lo, hi, left, right = node
            batch = batch[((l[batch] >= lo) & (u[batch] <= hi)).all(1)]
            if not len(batch):
                return
            if right is None:
                use = ((l[batch, None] >= self.lower[left]) &
                       (u[batch, None] <= self.upper[left])).all(2)
                # A nonfinite scope cutoff imposes no restriction, including
                # when the caller's cutoff is NaN (the original applies rule).
                use &= ((p[batch, None] <= self.cutoff[left]) |
                        ~np.isfinite(self.cutoff[left]))
                rows, cols = np.nonzero(use)
                domains.append(batch[rows])
                scopes.append(left[cols])
            else:
                visit(left, batch)
                visit(right, batch)
        visit(self.root, np.arange(len(l)))
        if not domains:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(domains), np.concatenate(scopes)

    def candidates(self, lower, upper):
        return np.unique(self.matches(lower, upper, -float('inf'))[1]).tolist()


class CertificateRegistry:
    def __init__(self, lp, input_cutoff_cuts=False):
        self.input_cutoff_cuts = input_cutoff_cuts
        self.columns = LiftedMap(lp)
        self._scopes = []
        self.auxiliary_scopes = []
        self._frontier = None
        self._materialized = True
        self._leaf_order = 0
        self._indexes = {}
        self._graph_maps = {}
        self._graph_tables = {}

    @property
    def scopes(self):
        self.materialize()
        return self._scopes

    def defer(self, frontier):
        self._frontier = frontier
        self._materialized = False

    def new_leaf(self, leaf=None):
        leaf = LeafCertificates() if leaf is None else leaf
        leaf.order = self._leaf_order
        self._leaf_order += 1
        return leaf

    def materialize(self):
        if not self._materialized and self._frontier is not None:
            leaves = self._frontier() if callable(self._frontier) else self._frontier
            # Input clipping chooses the latest applicable proposal. Preserve
            # solve order even when the heap returns leaves in a different order.
            for leaf in sorted((s for s in leaves if s is not None), key=lambda s: s.order):
                leaf.export(self)
            self._materialized = True

    def add(self, lower, upper, data, y, cutoff=None, ancestor_id=None):
        cert = DomainCertificate.export(len(self._scopes), self.columns, lower, upper,
                                        data, y, cutoff, ancestor_id)
        if self.input_cutoff_cuts and cutoff is not None and np.isfinite(cutoff):
            cert.input_cut = project_cutoff(self.columns, data, y, cutoff)
        self._scopes.append(cert)
        return cert

    def add_auxiliary(self, lower, upper, aux_lower, aux_upper, cutoff=None, proof=None):
        tensor = lambda v: torch.as_tensor(np.array(v, copy=True), dtype=torch.float64).flatten()
        scope = AuxiliaryBounds(len(self.auxiliary_scopes), tensor(lower), tensor(upper),
                                float('inf') if cutoff is None else float(cutoff), self.columns,
                                tensor(aux_lower), tensor(aux_upper), proof)
        self.auxiliary_scopes.append(scope)
        return scope

    def graph_references(self, bounded_module, example, lower, upper, cutoff):
        self.materialize()
        result = {}
        cap = lower.new_full((len(lower),), torch.inf)
        for bank, scopes in (('certificates', self._scopes), ('auxiliary', self.auxiliary_scopes)):
            rows, ids = self._matches(scopes, lower, upper, cutoff, bank)
            if not len(ids):
                continue
            index = self._indexes[bank][1]
            row_ids = torch.as_tensor(rows, device=lower.device)
            caps = torch.as_tensor(index.cutoff[ids], device=lower.device, dtype=lower.dtype)
            cap.scatter_reduce_(0, row_ids, caps, reduce='amin', include_self=True)
            # Different column versions (e.g. before and after root RLT) retain
            # separate proved graph maps. Scope IDs remain bank-local.
            for version in dict.fromkeys(scopes[i].columns.column_version for i in np.unique(ids)):
                selected = np.array([s.columns.column_version == version for s in scopes])
                use = selected[ids]
                graph_key = (id(bounded_module), version)
                if graph_key not in self._graph_maps:
                    scope = scopes[int(ids[np.flatnonzero(use)[0]])]
                    self._graph_maps[graph_key] = GraphReferenceMap(scope.columns, bounded_module, example)
                key = (graph_key, bank, lower.device, lower.dtype)
                if key not in self._graph_tables:
                    self._graph_tables[key] = GraphReferenceTable(self._graph_maps[graph_key])
                table = self._graph_tables[key]
                table.extend(scopes, np.unique(ids[use]), lower)
                table.reduce(rows[use], ids[use], lower, upper, result)
        return result, cap

    def select_scopes(self, lower, upper, cutoff):
        self.materialize()
        rows, scopes = self._matches(self._scopes, lower, upper, cutoff, 'certificates')
        ids = np.full(len(lower), -1, dtype=np.int64)
        # The previous ordered writes select the last applicable certificate.
        np.maximum.at(ids, rows, scopes)
        return torch.as_tensor(ids[:, None], device=lower.device)

    def references(self, lower, upper, cutoff):
        """Domain references are stable scope IDs, independent of queue/batch order."""
        self.materialize()
        return self._references(self._scopes, lower, upper, cutoff, 'certificates')

    def _matches(self, scopes, lower, upper, cutoff, bank):
        if not scopes:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        if bank not in self._indexes or self._indexes[bank][0] != len(scopes):
            self._indexes[bank] = (len(scopes), ScopeIndex(scopes))
        return self._indexes[bank][1].matches(lower, upper, cutoff)

    def _references(self, scopes, lower, upper, cutoff, bank):
        rows, ids = self._matches(scopes, lower, upper, cutoff, bank)
        if not len(ids):
            return []
        # Compatibility for evaluators that consume one certificate at a time.
        # The hot graph and queue paths consume sparse pairs directly.
        order = np.argsort(ids, kind='stable')
        unique, starts = np.unique(ids[order], return_index=True)
        refs = []
        for scope_id, indices in zip(unique, np.split(rows[order], starts[1:])):
            mask = torch.zeros(len(lower), dtype=torch.bool, device=lower.device)
            mask[torch.as_tensor(indices, device=lower.device)] = True
            refs.append((scopes[int(scope_id)], mask))
        return refs

    def bound(self, lower, upper, cutoff):
        result = lower.new_full((len(lower),), -torch.inf, dtype=torch.float64)
        refs = torch.full((len(lower), 1), -1, dtype=torch.long, device=lower.device)
        for cert, mask in self.references(lower, upper, cutoff):
            refs[mask] = cert.scope_id
            if mask.any():
                result[mask] = torch.maximum(result[mask], cert.evaluate(lower[mask], upper[mask]))
        return round_down(result, lower.dtype), refs


@dataclass
class AuxiliaryBounds:
    scope_id: int
    scope_lower: torch.Tensor
    scope_upper: torch.Tensor
    P0: float
    columns: LiftedMap
    aux_lower: torch.Tensor
    aux_upper: torch.Tensor
    proof: object
    applies = DomainCertificate.applies


class GraphReferenceMap:
    """Proved affine identities over lifted columns, from exact symbolic forms."""
    def __init__(self, columns, bounded_module, example):
        from nlp_solver.structure.forms import FormAlgebra
        from nlp_solver.structure.walk import build_forms
        from nlp_solver.lifted_atoms import NotLifted
        algebra = FormAlgebra(columns.n, example.detach().cpu().double().numpy().reshape(-1))
        forms = build_forms(bounded_module, example, algebra)
        atoms = copy.deepcopy(columns.atoms)
        if atoms is not None:
            # Resolve freshly interned symbolic atoms by their actual definitions,
            # never by a coincidentally equal atom number or tensor name.
            atoms.info, atoms.of = algebra.atom_info, {}
        monomials = [(j,) for j in range(columns.nb)]
        for a, b in zip(columns.a, columns.b):
            monomials.append(tuple(sorted(monomials[int(a)] + monomials[int(b)])))
        lookup = {m: j for j, m in enumerate(monomials)}
        self.nodes = {}
        for node in bounded_module.nodes():
            fs = forms.forms.get(node.name)
            if not node.perturbed or not node.inputs or not isinstance(fs, np.ndarray):
                continue
            elements = []
            for form in fs.flat:
                if form.poly is None or form.inexact:
                    break
                terms = []
                try:
                    for mono, coefficient in form.poly.items():
                        if not mono:
                            terms.append((None, F(coefficient)))
                            continue
                        if atoms is not None:
                            physical = tuple(sorted(atoms.columns(mono)))
                        else:
                            if any(v >= columns.n or type(e) is not int or e <= 0 for v, e in mono):
                                raise NotLifted('unmapped monomial')
                            physical = tuple(v for v, exponent in mono for _ in range(exponent))
                        if physical not in lookup:
                            raise NotLifted('unmapped column')
                        terms.append((lookup[physical], F(coefficient)))
                except NotLifted:
                    break
                elements.append(terms)
            if len(elements) == fs.size:
                self.nodes[node.name] = (fs.shape, elements)

    def bounds_batch(self, lower, upper):
        """Exact affine interval images for a bank of scopes, rounded once.

        Binary64 endpoints have power-of-two denominators. Lift them to one
        integer lattice and evaluate the sparse rational identities across all
        scopes using Python's unbounded integers (NumPy object arrays). This is
        the same rational sum as bounds(), without Fraction objects per term
        per scope. Repeated graph expressions share the exact calculation.
        """
        import math
        expressions = [tuple(terms) for _, elements in self.nodes.values() for terms in elements]
        unique = list(dict.fromkeys(expressions))
        used = sorted({j for terms in unique for j, _ in terms if j is not None})
        positions = {j: k for k, j in enumerate(used)}
        L = lower.detach().cpu().double().numpy()[:, used]
        U = upper.detach().cpu().double().numpy()[:, used]
        ratios = [float(v).as_integer_ratio() for v in np.r_[L.ravel(), U.ravel()]]
        lattice = max((d for _, d in ratios), default=1)
        values = np.array([n * (lattice // d) for n, d in ratios], dtype=object)
        li, ui = values.reshape(2, len(lower), len(used))
        denominator = math.lcm(*(c.denominator for terms in unique for _, c in terms))
        denominator = max(1, denominator)
        exact_denominator = lattice * denominator
        lows, highs = [], []

        def directed(n, up):
            n = int(n)
            try:
                value = n / exact_denominator
            except OverflowError:
                if n > 0:
                    return float('inf') if up else np.finfo(np.float64).max
                return -np.finfo(np.float64).max if up else -float('inf')
            a, b = value.as_integer_ratio()
            comparison = a * exact_denominator - n * b
            if (up and comparison < 0) or (not up and comparison > 0):
                value = np.nextafter(value, np.inf if up else -np.inf)
            return value

        for terms in unique:
            lo, hi = np.zeros(len(lower), dtype=object), np.zeros(len(lower), dtype=object)
            for j, coefficient in terms:
                c = coefficient.numerator * (denominator // coefficient.denominator)
                if j is None:
                    lo += c * lattice
                    hi += c * lattice
                else:
                    a, b = li[:, positions[j]], ui[:, positions[j]]
                    lo += c * (a if c >= 0 else b)
                    hi += c * (b if c >= 0 else a)
            lows.append(np.fromiter((directed(v, False) for v in lo), dtype=np.float64, count=len(lo)))
            highs.append(np.fromiter((directed(v, True) for v in hi), dtype=np.float64, count=len(hi)))
        if not expressions:
            empty = torch.empty((len(lower), 0), dtype=torch.float64)
            return empty, empty.clone()
        lookup = {terms: j for j, terms in enumerate(unique)}
        order = [lookup[terms] for terms in expressions]
        return (torch.from_numpy(np.stack(lows, 1)[:, order]),
                torch.from_numpy(np.stack(highs, 1)[:, order]))

    def bounds(self, lower, upper):
        lo, hi = self.bounds_batch(lower[None], upper[None])
        out, offset = {}, 0
        for name, (shape, elements) in self.nodes.items():
            end = offset + len(elements)
            out[name] = (lo[0, offset:end].reshape(shape), hi[0, offset:end].reshape(shape))
            offset = end
        return out


class GraphReferenceTable:
    """Device/dtype-cached outward bounds, packed across graph nodes.

    Only newly used scopes require symbolic conversion and a host-to-device copy.
    Sparse indexed extrema replace the scope x graph-node masked update loop.
    """
    def __init__(self, graph):
        self.graph = graph
        self.layout = []
        offset = 0
        for name, (shape, _) in graph.nodes.items():
            size = int(np.prod(shape))
            self.layout.append((name, shape, offset, offset + size))
            offset += size
        self.width = offset
        self.ready = np.zeros(0, dtype=bool)
        self.lower = self.upper = None

    def extend(self, scopes, ids, like):
        if len(self.ready) < len(scopes):
            extra = len(scopes) - len(self.ready)
            lo = like.new_full((extra, self.width), -torch.inf)
            hi = like.new_full((extra, self.width), torch.inf)
            self.lower = lo if self.lower is None else torch.cat((self.lower, lo))
            self.upper = hi if self.upper is None else torch.cat((self.upper, hi))
            self.ready = np.r_[self.ready, np.zeros(extra, dtype=bool)]
        missing = ids[~self.ready[ids]]
        if not len(missing):
            return
        lows, highs = self.graph.bounds_batch(
            torch.stack([scopes[int(i)].aux_lower for i in missing]),
            torch.stack([scopes[int(i)].aux_upper for i in missing]))
        positions = torch.as_tensor(missing, device=like.device)
        self.lower[positions] = round_down(lows.to(like.device), like.dtype)
        self.upper[positions] = -round_down(-highs.to(like.device), like.dtype)
        self.ready[missing] = True

    def reduce(self, rows, ids, lower, upper, result):
        if not self.width:
            return
        lo = lower.new_full((len(lower), self.width), -torch.inf)
        hi = upper.new_full((len(upper), self.width), torch.inf)
        # Bound temporary gather memory independently of batch and scope count.
        # This is an allocation budget, not a work-admission or instance cutoff.
        chunk = max(1, (1 << 20) // self.width)
        for start in range(0, len(ids), chunk):
            scope_ids = torch.as_tensor(ids[start:start + chunk], device=lower.device)
            row_ids = torch.as_tensor(rows[start:start + chunk], device=lower.device)
            indices = row_ids[:, None].expand(-1, self.width)
            lo.scatter_reduce_(0, indices, self.lower[scope_ids], reduce='amax', include_self=True)
            hi.scatter_reduce_(0, indices, self.upper[scope_ids], reduce='amin', include_self=True)
        for name, shape, start, end in self.layout:
            l = lo[:, start:end].reshape(len(lower), *shape[1:])
            u = hi[:, start:end].reshape(len(upper), *shape[1:])
            if name in result:
                l, u = torch.maximum(l, result[name][0]), torch.minimum(u, result[name][1])
            result[name] = (l, u)


def merge_reference_bounds(first, second):
    out = dict(first or {})
    for key, (lo, hi) in (second or {}).items():
        if key in out:
            lo, hi = torch.maximum(lo, out[key][0]), torch.minimum(hi, out[key][1])
        out[key] = (lo, hi)
    return out


def project_cutoff(columns, data, y, cutoff):
    """Theorem 4: a lower input estimator of r'z <= P-kappa.

    Exact rational accumulation protects sign decisions, recursive product
    substitution and cancellation. Unmapped atoms/epigraphs use safe endpoints.
    The returned RHS already includes every source tolerance and rounding error.
    """
    r, kappa = exact_residual(data, y)
    lower, upper = data[4:]
    constant = F(0)
    for j in range(len(r) - 1, columns.n - 1, -1):
        q = r[j]
        if not q:
            continue
        if columns.nb <= j < columns.nb + len(columns.a):
            k = j - columns.nb
            a, b = int(columns.a[k]), int(columns.b[k])
            la, ua, lb, ub = (F(float(v)) for v in (lower[a], upper[a], lower[b], upper[b]))
            if a == b:
                t = (la + ua) / 2
                ca, cb, cc = (2*t, F(0), -t*t) if q >= 0 else (la+ua, F(0), -la*ua)
            else:
                bank = ([(lb, la, -la*lb), (ub, ua, -ua*ub)] if q >= 0 else
                        [(lb, ua, -ua*lb), (ub, la, -la*ub)])
                values = [aa*(la+ua)/2 + ab*(lb+ub)/2 + bc for aa, ab, bc in bank]
                index = int(values[1] > values[0]) if q >= 0 else int(values[1] < values[0])
                ca, cb, cc = bank[index]
            r[a] += q*ca
            r[b] += q*cb
            constant += q*cc
        else:
            constant += q*F(float(lower[j] if q >= 0 else upper[j]))
        r[j] = F(0)
    A = np.array([float(v) for v in r[:columns.n]])
    constant += sum(min((v-F(float(a)))*F(float(lo)), (v-F(float(a)))*F(float(hi)))
                    for v, a, lo, hi in zip(r, A, lower, upper))
    return A, round_up(F(float(cutoff))-kappa-constant)
