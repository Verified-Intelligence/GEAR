"""Certified frontier potential for forecasts, never additional ledger reward."""
import math

import numpy as np


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


class GapMass:
    """Integral of (cutoff - checked leaf lb)+ over normalized root-box measure.

    Only committed open covers are sampled. A picked parent remains logically
    present until its children commit; callers must not sample inside replacement.
    Point coordinates have unit measure. Unknown bounds/underflow disable mass
    forecasting; zero numerical mass is never a closure certificate.
    """
    def __init__(self, lo, hi, cutoff):
        self.width = (array(hi) - array(lo)).reshape(-1)
        self.cutoff = float(cutoff)
        self.mass = 0.
        self.valid = math.isfinite(self.cutoff) and bool(np.isfinite(self.width).all())
        self.valid = self.valid and bool((self.width >= 0).all())

    def terms(self, lo, hi, bounds):
        bounds = array(bounds).reshape(-1)
        if not len(bounds):
            return 0.
        lo, hi = array(lo).reshape(len(bounds), -1), array(hi).reshape(len(bounds), -1)
        widths = hi - lo
        if not (np.isfinite(bounds).all() and np.isfinite(widths).all()):
            self.valid = False
            return 0.
        positive = self.width > 0
        ratios = np.maximum(0., widths[:, positive]) / self.width[positive]
        volumes = np.prod(ratios, axis=1)
        # A nonempty cell that underflows is unknown, never resolved for free.
        if ((volumes == 0) & (ratios > 0).all(axis=1)).any():
            self.valid = False
        gaps = np.maximum(0., self.cutoff - bounds)
        terms = volumes * gaps
        if not np.isfinite(terms).all():
            self.valid = False
            return 0.
        return math.fsum(terms.tolist())

    def add(self, lo, hi, bounds, sign=1):
        self.mass += sign * self.terms(lo, hi, bounds)

    def snapshot(self):
        return (self.cutoff, max(0., self.mass)) if self.valid and math.isfinite(self.mass) else None


def domain_snapshot(domains, root):
    """Incremental between batches; rescan only when the verified cutoff changes.

    add/pick_out update only their batch. Pending parents are sampled only after
    successful replacement. Rebase on external incumbent changes earns no gain.
    """
    P = root.values()[0]
    measure = getattr(domains, 'controller_measure', None)
    if measure is None or P != measure.cutoff:
        measure = GapMass(root.dm_l[:1], root.dm_u[:1], P)
        if len(domains):
            measure.add(domains.dm_l._storage[:len(domains)],
                        domains.dm_u._storage[:len(domains)],
                        domains.lb._storage[:len(domains), 0])
        domains.controller_measure = measure
    snapshot = measure.snapshot()
    # The minimum is its own certified frontier, independent of another lane's D.
    import arguments
    bound = min(P, float(domains.lb._storage[:len(domains), 0].min())) if len(domains) else P
    cap = arguments.Globals.get('discarded_threshold_min')
    if cap is not None:
        bound = min(bound, cap)
    return (P, None if snapshot is None else snapshot[1], bound)


class PendingRate:
    """A positive decrement settles all its prefix; flat work dilutes it."""
    def __init__(self):
        self.value = self.anchor = None
        self.rate = self.previous_work = self.pending = self.gain = 0.
        self.threshold = 0.

    def account(self, seconds, value, rebase=False, retain=False):
        self.pending += seconds
        if value is None or not math.isfinite(value):
            return
        if rebase or self.value is None or value > self.value:
            self.value = self.anchor = value
            if not retain:
                self.rate = self.previous_work = 0.
            self.gain = 0.
            return
        self.gain += self.value - value
        self.value = value
        if self.gain > 0 and self.gain >= self.threshold and self.pending > 0:
            self.previous_work = self.pending
            self.rate = self.gain / self.pending
            self.pending = self.gain = 0.

    @property
    def velocity(self):
        return self.rate * self.previous_work / (self.previous_work + self.pending) if self.previous_work else 0.


class ProgressForecast(PendingRate):
    """Two views of the same paid tree work, combined by max, never added.

    The minimum-frontier gap gives a time-to-closure forecast even below global D.
    Weighted mass resolves partial/tied layers. Its remaining opportunity is
    discounted by M/M_initial: diminishing box measure cannot amplify a score.
    Cutoff changes rebase mass without renewing its anchor. Compatible bound
    forecasts survive, but the external cutoff jump earns no tree progress.
    """
    def __init__(self):
        super().__init__()
        self.cutoff = None
        self.frontier = PendingRate()
        self.bound = None

    @property
    def mass(self):
        return self.value

    def account(self, seconds, snapshot, tolerance=0.):
        rebase = snapshot is not None and self.cutoff != snapshot[0]
        if snapshot is not None:
            self.cutoff = snapshot[0]
            if len(snapshot) > 2:
                self.bound = snapshot[2]
        anchor = self.anchor
        super().account(seconds, None if snapshot is None else snapshot[1], rebase)
        if anchor is not None and self.anchor is not None:
            self.anchor = max(anchor, self.anchor)  # a cutoff cannot renew opportunity
        value = (-snapshot[2] if snapshot is not None and len(snapshot) > 2
                 and math.isfinite(snapshot[2]) else None)
        if rebase or self.frontier.value is None:
            self.frontier.threshold = tolerance
        # A tighter cutoff preserves old certificates. Keep the paid forecast,
        # but exclude the external cutoff's jump in the new frontier minimum.
        self.frontier.account(seconds, value, rebase, retain=rebase)

    def invalidate(self):
        self.rate = self.previous_work = 0.

    def score(self, gap, cutoff=None):
        cutoff = self.cutoff if cutoff is None else cutoff
        own_gap = (cutoff-self.bound if cutoff is not None and self.bound is not None
                   and math.isfinite(cutoff) and math.isfinite(self.bound) else 0.)
        mass_rate = self.velocity / self.anchor if self.anchor and self.anchor > 0 else 0.
        bound_rate = (self.frontier.velocity / own_gap
                      if own_gap > 0 else 0.)
        # No potential value, including numerical zero, certifies closure.
        return gap * max(mass_rate, bound_rate)
