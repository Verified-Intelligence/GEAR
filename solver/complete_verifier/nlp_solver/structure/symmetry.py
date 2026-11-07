"""Interchangeable variable blocks of the NLP, found on the exact forms.

A chain is a list of m >= 2 disjoint blocks of k variables (tuples, corresponding coordinates
aligned) with equal boxes such that every swap of two consecutive blocks maps the objective to itself
and the multiset of constraint rows (the spec rows at their right-hand sides and the input linear rows)
to itself, compared exactly (atoms by their canonical key). Consecutive transpositions generate the full
symmetric group on the blocks, so every feasible point can be permuted, at the same objective, to
satisfy x[B_1[0]] <= x[B_2[0]] <= ... <= x[B_m[0]]: the m-1 ordering rows keep the optimal value.
Chains are disjoint and their permutation groups commute (disjoint supports), so the rows of all
chains hold together.

Candidate blocks are only proposals (single variables of equal signature; supports of rows, atom
arguments and monomials) and the exact swap test decides, so a wrong proposal costs time, never
soundness. Work is bounded by a count of term operations, never by time; on exhaustion the chains
already verified are returned (each is a proved symmetry on its own)."""
import collections
import itertools

import numpy as np

from auto_LiRPA.cap_log import cap_log

from .forms import exact

BUDGET = 1_000_000         # term operations over all swap tests (each test also costs TEST_COST)
TEST_COST = 100
MAX_TERMS = 300_000        # terms of all rows and atom arguments; more: no detection
MAX_BLOCK = 64             # variables per block
MAX_MISSES = 16            # candidates a block may fail to match before it is taken to be alone
MAX_ALIGNMENTS = 24        # coordinate orders tried when a block's coordinates are indistinguishable


def row_items(poly, thr):
    """the row poly <= thr as one canonical poly <= 0 (items sorted)"""
    items = dict(poly)
    items[()] = items.get((), 0) - thr
    return tuple(sorted((m, c) for m, c in items.items() if c))


class Symmetry:
    def __init__(self, report, rhs, lo, hi, input_rows):
        """report: StructureReport (exact forms); rhs: right-hand sides of the spec rows (row 0 unused);
        lo, hi: root box; input_rows: polys of the rows A x + b <= 0, {monomial: coefficient}."""
        self.n, self.lo, self.hi = report.n, lo, hi
        self.info = report.atoms
        self.by_key = {(k, p, a): v for v, (k, p, a, _) in report.atoms.items()}
        self.objective = tuple(sorted(report.forms[0].poly.items()))
        self.rows = [row_items(f.poly, exact(float(t))) for f, t in zip(report.forms[1:], rhs)] + [
            row_items(p, 0) for p in input_rows]
        self.objective_terms = set(self.objective)
        self.count = collections.Counter(self.rows)
        self.terms = (len(self.objective) + sum(map(len, self.rows))
                      + sum(len(a[2]) for a in report.atoms.values()))
        self.touch = collections.defaultdict(list)       # variable -> rows containing it
        for r, items in enumerate(self.rows):
            for v in self._support(items):
                self.touch[v].append(r)
        self.otouch = collections.defaultdict(list)      # variable -> objective terms containing it
        for t, term in enumerate(self.objective):
            for v in self._support([term]):
                self.otouch[v].append(t)
        self.spent = 0

    def _support(self, items):
        out = set()
        for m, _ in items:
            for v, _ in m:
                if v < self.n:
                    out.add(v)
                else:
                    out |= self.info[v][3]
        return out

    # ---- the exact swap test -------------------------------------------------------------
    def _atom(self, v, sigma, memo):
        """the atom v with its variables renamed by sigma, as an atom id (None: it does not exist)"""
        if v not in memo:
            kind, param, arg, support = self.info[v]
            if not support & sigma.keys():
                memo[v] = v
            else:
                image = self._items(arg, sigma, memo)
                memo[v] = None if image is None else self.by_key.get((kind, param, image))
                if memo[v] is None and image is not None and kind == 'pow' and type(param) is int and param % 2 == 0:
                    # (-p)^even is the atom of p: a renamed difference may be stored with the other sign
                    memo[v] = self.by_key.get((kind, param, tuple(sorted((m, -c) for m, c in image))))
        return memo[v]

    def _items(self, items, sigma, memo):
        """the terms with their variables renamed by sigma (sorted), None if an atom has no image"""
        out = []
        for m, c in items:
            image = []
            for v, e in m:
                v = sigma.get(v, v) if v < self.n else self._atom(v, sigma, memo)
                if v is None:
                    return None
                image.append((v, e))
            out.append((tuple(sorted(image)), c))
        return tuple(sorted(out))

    def swappable(self, a, b):
        """the exchange of blocks a and b (tuples, coordinate by coordinate) is a symmetry. The exchange
        sigma is an involution, so it maps the rows touching a or b onto themselves, with multiplicities,
        iff each such row's image is a row occurring as often; likewise for the objective's terms"""
        if any(self.lo[i] != self.lo[j] or self.hi[i] != self.hi[j] for i, j in zip(a, b)):
            return False
        sigma = {**dict(zip(a, b)), **dict(zip(b, a))}
        memo = {}
        self.spent += TEST_COST
        for v in sigma:
            for t in self.otouch[v]:
                self.spent += 1
                image = self._items([self.objective[t]], sigma, memo)
                if image is None or image[0] not in self.objective_terms:
                    return False
            for r in self.touch[v]:
                self.spent += len(self.rows[r])
                image = self._items(self.rows[r], sigma, memo)
                if self.count.get(image) != self.count[self.rows[r]]:
                    return False
        return True

    # ---- candidate blocks ------------------------------------------------------------------
    def signatures(self):
        """variable -> class id; variables of one class share box and term shapes summed over all rows,
        refined twice by the classes met in the same objective terms and rows: symmetries preserve all of it"""
        sig = collections.defaultdict(collections.Counter)
        for kind, rows in ((0, [self.objective]), (1, self.rows)):
            for items in rows:
                for m, c in items:
                    for v, e in m:
                        if v < self.n:
                            sig[v][(kind, len(m), sum(x for _, x in m), e, float(c))] += 1
        ids = {}
        cls = [ids.setdefault((self.lo[v], self.hi[v], tuple(sorted(sig[v].items(), key=repr))), len(ids))
               for v in range(self.n)]
        places = [self._support([t]) for t in self.objective] + [self._support(r) for r in self.rows]
        for _ in range(2):
            ids, seen = {}, collections.defaultdict(list)
            for place in places:
                key = tuple(sorted(collections.Counter(cls[v] for v in place).items()))
                for v in place:
                    seen[v].append(ids.setdefault(key, len(ids)))
            ids = {}
            cls = [ids.setdefault((cls[v], tuple(sorted(seen[v]))), len(ids)) for v in range(self.n)]
        return cls

    def candidate_sets(self):
        """supports of 2..MAX_BLOCK variables of the rows, atom arguments and their terms, with how often
        each occurs and a key per variable (its coefficients in the first poly it came from)"""
        found = {}

        def add(items):
            support = frozenset(self._support(items))
            if 2 <= len(support) <= MAX_BLOCK:
                entry = found.setdefault(support, [0, {v: tuple(sorted(
                    (len(m), e, c) for m, c in items for u, e in m if u == v)) for v in support}])
                entry[0] += 1

        for items in (self.objective, *self.rows, *(a[2] for a in self.info.values())):
            add(items)
            if len(items) > 1:
                for term in items:
                    add([term])
        return found

    def blocks(self, cls, used):
        """one layout of disjoint candidate blocks per block shape; the variables of a block are ordered
        by class and key, so equal shapes align coordinate by coordinate"""
        groups = collections.defaultdict(list)
        for support, (count, key) in self.candidate_sets().items():
            if not used.intersection(support):
                order = sorted(support, key=lambda v: (cls[v], key[v], v))
                groups[tuple((cls[v], key[v]) for v in order)].append((-count, tuple(order)))
        for shape in sorted((s for s in groups if len(groups[s]) > 1), key=lambda s: (len(s), repr(s))):
            taken, layout = set(), []
            for _, block in sorted(groups[shape]):
                if not taken.intersection(block):
                    taken.update(block)
                    layout.append(block)
            yield layout

    # ---- chains ----------------------------------------------------------------------------
    @staticmethod
    def alignments(block, cls):
        """orders of a block to try: its coordinates permuted inside runs of equal class (a run of
        interchangeable coordinates, e.g. x, y, z of a point); only the given order if there are too many"""
        runs = [list(g) for _, g in itertools.groupby(block, key=lambda v: cls[v])]
        count = 1
        for run in runs:
            count *= len(list(itertools.permutations(run)))
        if count > MAX_ALIGNMENTS:
            return [block]
        return [sum(p, ()) for p in itertools.product(*(itertools.permutations(r) for r in runs))]

    def chains(self, blocks, cls):
        """greedily split blocks into chains whose consecutive members are swappable. A block that
        matches none of MAX_MISSES successive candidates is taken to be alone (a false give-up only
        loses a chain, never soundness): this bounds the tests of a large class with no symmetry"""
        out, pending = [], list(blocks)
        while len(pending) > 1 and self.spent <= BUDGET:
            chain, rest, misses = [pending[0]], [], 0
            for k, blk in enumerate(pending[1:], 1):
                if self.spent > BUDGET or (len(chain) == 1 and misses == MAX_MISSES):
                    rest.extend(pending[k:])
                    break
                match = next((c for c in self.alignments(blk, cls) if self.swappable(chain[-1], c)), None)
                if match is None:
                    rest.append(blk)
                    misses += 1
                else:
                    chain.append(match)
                    misses = 0
            if len(chain) > 1:
                out.append(chain)
            pending = rest
        if self.spent > BUDGET:
            cap_log('structure.symmetry', 'term_ops', BUDGET, self.spent, 'chains found so far kept')
        return out

    def find(self):
        """list of chains (lists of blocks): single variables of one class first, then proposed blocks"""
        cls = self.signatures()
        classes = collections.defaultdict(list)
        for v, c in enumerate(cls):
            classes[c].append(v)
        found, used = [], set()

        def take(layouts):
            for blocks in layouts:
                blocks = [b for b in blocks if not used.intersection(b)]
                for chain in self.chains(blocks, cls):
                    found.append(chain)
                    used.update(v for b in chain for v in b)

        singles = [[(v,) for v in c if self.touch[v] or self.otouch[v]] for c in classes.values()]
        take(singles)
        take(self.blocks(cls, used))
        return found


def symmetry_chains(report, rhs, lo, hi, input_A=None, input_b=None):
    """Chains of interchangeable blocks of the problem of `report`, with the input rows
    input_A x + input_b <= 0 (numpy): [] when a row is not an exact polynomial form (a rounded
    transcendental constant included), a right-hand side is not finite, or the forms exceed MAX_TERMS."""
    forms = report.forms
    if not forms or any(f.poly is None or f.inexact for f in forms) or not np.isfinite(rhs).all() \
            or input_b is not None and not np.isfinite(input_b).all():
        return []
    input_rows = []
    if input_A is not None:
        for a, b in zip(input_A, input_b):
            poly = {((int(j), 1),): exact(float(a[j])) for j in np.flatnonzero(a)}
            if b:
                poly[()] = exact(float(b))
            input_rows.append(poly)
    sym = Symmetry(report, rhs, lo, hi, input_rows)
    if sym.terms > MAX_TERMS:
        cap_log('structure.symmetry', 'terms', MAX_TERMS, sym.terms, 'no symmetry detection')
    return sym.find() if sym.terms <= MAX_TERMS else []


def ordering_pairs(chains):
    """(a, b) for every x[a] <= x[b] that orders consecutive blocks of a chain by their first coordinates"""
    return [(c[t][0], c[t + 1][0]) for c in chains for t in range(len(c) - 1)]


def sort_blocks(x, chains):
    """the flat point x with the blocks of every chain permuted so that their first coordinates ascend
    (the same objective and feasibility, as the blocks are interchangeable)"""
    y = x.clone()
    for chain in chains:
        for src, dst in zip(sorted(chain, key=lambda b: float(x[b[0]])), chain):
            y[list(dst)] = x[list(src)]
    return y
