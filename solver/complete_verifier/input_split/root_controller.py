"""Certified receipts and owner-thread publication shared by the S4 controller.

Production scheduling uses slice_controller. The original live-stack adapter is
retained for its direct API callers; it is not used by input_bab_parallel when
the controller is enabled. Work is measured host service, not reference work.
"""
from contextlib import contextmanager
import math
import queue
import threading
import time
from types import SimpleNamespace

import torch
import arguments
from gear import report as _report

from input_split.batch_branch_and_bound import OpLedger, _BAB_STEPS
from input_constraints import standard_form_to_input_constraints


class RootClosed(BaseException):
    """Unwind primal work at a model-evaluation boundary after certified closure."""
    def __init__(self, source):
        self.source = source


class RootDeadline(BaseException):
    """Cancel root work while retaining the last committed endpoints/frontier."""


def finish_timeout(root):
    # A proof accepted before the alarm remains a solved result even when the
    # loop's next time check skips its ordinary checkpoint.
    terminal = getattr(root, 'controller_terminal', None)
    if isinstance(terminal, RootClosed):
        return root.finish(terminal.source, root.values()[1])
    from input_split.batch_branch_and_bound import export_best_primal
    P, D = root.values()
    primal = root.rhs.new_tensor([P])
    sign = -1 if arguments.Config['solving']['is_maximization'] else 1
    print(f'Primal Solution {sign * primal}, Dual Solution {sign * D}')
    print(f'Solution gap: {P - D}')
    _report.record_pd(sign * primal, sign * D)
    _report.record_event('time_limit')
    export_best_primal(root.net, getattr(root, 'domains', None), root.best_primal_x,
                        primal, root.primal_solution_found)
    root.ledger.commit_current('deadline', 0.0, root.values)
    return primal.new_tensor(D), root.stats.visited, 'unknown'


class ReceiptLedger(OpLedger):
    """Rewards only come from accepted global endpoints; never inf-inf or widths."""
    def __init__(self, config=None):
        super().__init__()
        self.seq = 0
        self.config = config or {}
        self.pending_work = {}
        self.episodes = {}

    def line(self, name, seconds, work=1, dD=0.0, dP=0.0):
        self.commit(name, seconds, self.P, self.D)

    @contextmanager
    def publication(self):
        """Defer owner signal handlers across endpoint capture and publication.

        Workers only enqueue evidence; the owner is the sole writer. A mutex
        would not exclude a signal handler reentering that same Python thread.
        """
        was_busy = getattr(self, 'busy', False)
        self.busy = True
        try:
            yield
        finally:
            self.busy = was_busy

    def commit_current(self, name, seconds, values):
        """Capture shared endpoints inside the publication boundary."""
        with self.publication():
            self.commit(name, seconds, *values())

    def commit(self, name, seconds, P=None, D=None, work=None, accounting='transaction', parent=None):
        with self.publication():
            # Arguments are proposals, never replacements for the incumbent.
            # Stale/repeated evidence retains its cost but cannot earn again.
            P = self.P if P is None else min(self.P, P)
            D = self.D if D is None else max(self.D, D)
            self.seq += 1
            dP = max(0.0, self.P - P) if math.isfinite(self.P) and math.isfinite(P) else 0.0
            dD = max(0.0, D - self.D) if math.isfinite(self.D) and math.isfinite(D) else 0.0
            first_P = not math.isfinite(self.P) and math.isfinite(P)
            first_D = not math.isfinite(self.D) and math.isfinite(D)
            recipe = ('alm_bab' if name.startswith('bab_alm_') else
                      'root_lifted' if name.startswith(('early_lifted', 'lifted_', 'root_lp', 'obbt', 'lp_tree')) else
                      'crown_tree' if name.startswith('gpu_tree') else
                      'ipopt' if name.startswith('ipopt') else
                      'lp_alternation' if name.startswith('lp_alternation') else
                      name.removesuffix('_publish').removesuffix('_slice').removesuffix('_done').removesuffix('_cancelled')
                      if name.startswith(('gp_certificate_', 'exp_certificate_', 'tangent_', 'convex_cuts_')) else
                      'alm_root' if name == 'nlp' or name.startswith(('alm_', 'nlp_', 'restore_', 'polish_', 'sqp_')) else name)
            pending = self.pending_work.get(recipe, 0.0)
            if accounting == 'transaction':
                pending += max(0.0, seconds)
            terminal = (name in ('alm_publish', 'nlp', 'ipopt_publish', 'deadline')
                        or name.endswith(('_cancelled', '_done_publish', '_done', '_episode')))
            settled = pending if dP + dD > 0 or first_P or first_D or terminal else 0.0
            episode = self.episodes.get(recipe, 0)
            self.pending_work[recipe] = 0.0 if settled else pending
            self.episodes[recipe] = episode + int(settled > 0)
            from solving_utils import is_primal_dual_converged
            closed = is_primal_dual_converged(
                torch.tensor(P, dtype=torch.float64), torch.tensor(D, dtype=torch.float64),
                abs_tol=float(self.config.get('bab_primal_dual_abs_tol', 1e-5)),
                rel_tol=float(self.config.get('bab_primal_dual_rel_tol', 1e-4)))
            closed = closed and time.monotonic() < getattr(self, 'deadline', math.inf)
            endpoint = lambda v: f'{v:.12g}' if math.isfinite(v) else 'null'
            print(f'[Op] {name} t={time.time() - self.origin:.3f} s={seconds:.4f} '
                  f'cpu_total={time.process_time():.9g} '
                  f'work={max(0.0, seconds):.6g} unit=host_seconds job={name}:{self.seq} '
                  f'accounting={accounting} parent={parent or "none"} '
                  f'transaction={recipe}:{episode} pending_work={self.pending_work[recipe]:.6g} '
                  f'settled_work={settled:.6g} state={"settled" if settled or terminal else "pending"} '
                  f'dD={dD:.12g} dP={dP:.12g} P={endpoint(P)} D={endpoint(D)} '
                  f'first_feasible={int(not math.isfinite(self.P) and math.isfinite(P))} '
                  f'first_bound={int(not math.isfinite(self.D) and math.isfinite(D))} closed={int(closed)}', flush=True)
            self.P, self.D = P, D
            observer = getattr(self, "observer", None)
            if observer is not None:
                observer(name, max(0.0, seconds), dP + dD)

    def iteration(self, timer, work, P, D):
        # A local LP gain or box contraction is never the frontier's gain.
        seconds = sum(max(0.0, timer.time_sum.get(k, 0.0) - self.sums.get(k, 0.0))
                      for k in _BAB_STEPS) + sum(n[1] for n in self.notes)
        self.commit('bab_batch', seconds, P, D)
        parent = f'bab_batch:{self.seq}'
        for name, seconds, _, _ in self.notes:
            self.commit(name, seconds, P, D, accounting='component', parent=parent)
        self.sums, self.notes = dict(timer.time_sum), []


def offer(root, family, point, seconds):
    """Thread-safe message delivery. Only the root thread verifies/commits."""
    root.controller_inbox.put((family, None if point is None else
                               torch.as_tensor(point).detach().clone(), seconds))


def poll(root, service=True):
    was_committing = getattr(root, 'controller_committing', False)
    root.controller_committing = True
    try:
        from input_split.lifted_bound import make_verifier
        while True:
            try:
                family, point, seconds = root.controller_inbox.get_nowait()
            except queue.Empty:
                break
            t0 = time.monotonic()
            accepted_P = None
            if point is not None:
                verify = make_verifier(root.net.model_ori, root.x.ptb.x_L, root.x.ptb.x_U,
                                       root.c, root.rhs, root.or_spec_size,
                                       root.base_input_linear_constraints,
                                       dict(root.nlp_config, constr_viol_tol=1e-6))
                ok, obj, used = verify(torch.as_tensor(point).reshape(-1).cpu().double().numpy())
                if ok and math.isfinite(obj) and obj < root.values()[0]:
                    root.primal_solution = torch.tensor([obj], device=root.rhs.device, dtype=root.rhs.dtype)
                    root.best_primal_x = torch.as_tensor(used, device=root.dm_l.device,
                                                         dtype=root.dm_l.dtype).reshape(root.dm_l[:1].shape)
                    root.primal_solution_found = True
                    accepted_P = root.values()[0]
                    history = getattr(getattr(root, 'domains', None), 'history', None)
                    if history is not None:
                        # Export prefers this history slot. A publication can
                        # close inside a paused tree batch, before its next loop
                        # refresh, so the accepted value and coordinates commit
                        # together here.
                        history.alm_best_primal_x = root.best_primal_x
            checking_s = time.monotonic() - t0
            root.controller_nested_s = getattr(root, 'controller_nested_s', 0.0) + checking_s
            # This message proposes only its own verified primal evidence.
            # A dual awaiting the interrupted producer's receipt belongs there.
            root.ledger.commit(family + '_publish', seconds + checking_s, P=accepted_P)
            if root.closes(root.values()[1]):
                raise RootClosed(family + '_publish')
        if service:
            service_incumbent(root)
            independent_bab(root)
    finally:
        root.controller_committing = was_committing


def service_incumbent(root):
    setup = getattr(root, 'early_lifted_setup', None)
    P = root.values()[0]
    if setup is not None and math.isfinite(P) and P < getattr(root, 'early_lifted_cutoff', math.inf):
        root.early_lifted_cutoff = P
        t0 = time.monotonic()
        bound = setup.bound(root.primal_solution, root.time_left()) if root.time_left() > 0 else None
        if bound is not None and math.isfinite(bound):
            root.certified_dual = max(root.certified_dual, bound)
        seconds = time.monotonic() - t0
        root.controller_nested_s += seconds
        root.ledger.commit_current('early_lifted_cutoff', seconds, root.values)
        if root.closes(root.values()[1]):
            raise RootClosed('early_lifted_cutoff')
        advance_lifted(root, setup, P)


def independent_bab(root):
    """The existing lifted batch can complete independently of the primal join."""
    from input_split.lifted_bound import run_lifted_bab, make_verifier, make_local_solver
    setup = getattr(root, 'early_lifted_setup', None)
    if setup is None or not getattr(root, 'early_strengthened', False) or root.time_left() <= 0:
        return
    P = root.values()[0]
    history = setup.history
    last = getattr(history, 'lifted_bab_cutoff', None)
    if last is not None and not getattr(history, 'lifted_bab_paid', False) and not P < last - 1e-4 * abs(last) - 1e-9:
        return
    if float(root.nlp_config.get('lifted_lp_bab_time', 0.0)) <= 0:
        return
    verify = make_verifier(root.net.model_ori, root.x.ptb.x_L, root.x.ptb.x_U,
                           root.c, root.rhs, root.or_spec_size,
                           root.base_input_linear_constraints, root.nlp_config)
    local = (make_local_solver(root.net.model_ori, root.x.ptb.x_L, root.x.ptb.x_U,
                               root.c, root.rhs, root.or_spec_size,
                               root.base_input_linear_constraints, root.nlp_config, verify)
             if root.nlp_config.get('lifted_lp_leaf_ipopt', False) else None)
    def batch():
        D = run_lifted_bab(history, setup.lo, setup.hi, root.primal_solution,
                           root.nlp_config, root.time_left(), verify_fn=verify,
                           incumbent_x=root.best_primal_x, local_fn=local,
                           floor=root.values()[1], time_share=1.0,
                           checkpoint=getattr(root, 'controller_checkpoint', None))
        D = getattr(history, 'lifted_bab_D', D)
        if D is not None and math.isfinite(D):
            root.certified_dual = max(root.certified_dual, min(P, D))
    paid_service(root, 'early_lifted_bab', batch)
    if root.closes(root.values()[1]):
        raise RootClosed('early_lifted_bab')
    candidate = getattr(history, 'lifted_bab_new_incumbent', None)
    history.lifted_bab_new_incumbent = None
    if candidate is not None:
        offer(root, 'early_lifted_bab', candidate[0], 0.0)
        poll(root)


def paid_service(root, name, fn):
    start, nested = time.monotonic(), root.controller_nested_s
    try:
        return fn()
    except (RootClosed, RootDeadline):
        name += '_cancelled'
        raise
    finally:
        seconds = max(0.0, time.monotonic() - start - (root.controller_nested_s - nested))
        root.controller_nested_s += seconds
        root.ledger.commit_current(name, seconds, root.values)


def advance_lifted(root, setup, P):
    """Run the existing eligible strengthening recipe without waiting for ALM.

    ALM's live call stack is retained while this root-thread service executes.
    The legacy OBBT budgets and incumbent-triggered re-OBBT limits are unchanged.
    """
    from input_split.lifted_bound import _relaxes_problem, reobbt_on_improvement
    if not _relaxes_problem(setup.history, setup.lp, setup.rhs_rows):
        return
    setup.cutoff = P

    def service(name, fn):
        if root.time_left() <= 0:
            return
        paid_service(root, name, fn)
        if root.closes(root.values()[1]):
            raise RootClosed(name)

    def bound():
        D = setup.bound(root.primal_solution, root.time_left())
        if D is not None and math.isfinite(D):
            root.certified_dual = max(root.certified_dual, D)

    def obbt():
        setup.obbt(root.best_primal_x, root.time_left())
        box = getattr(setup.history, 'lifted_box', None)
        if box is not None:
            setup.lo, setup.hi = (t.reshape(1, -1).double() for t in box)

    if not getattr(root, 'early_strengthened', False):
        root.early_strengthened = True
        service('early_lifted_warm_up', lambda: setup.warm_up(root.time_left()))
        service('early_lifted_rlt', setup.add_rlt)
        service('early_lifted_bound', bound)
        service('early_lifted_obbt', obbt)
    else:
        service('early_lifted_reobbt', lambda: reobbt_on_improvement(
            setup.history, root.primal_solution, root.nlp_config, root.time_left(),
            checkpoint=getattr(root, 'controller_checkpoint', None)))
        box = getattr(setup.history, 'lifted_box', None)
        if box is not None:
            setup.lo, setup.hi = (t.reshape(1, -1).double() for t in box)
    service('early_lifted_bound', bound)


def early_lp(root):
    """A complete-row relaxation needs no incumbent, ALM join, tangent or OBBT."""
    from input_split.lifted_bound import RootLifted
    history = SimpleNamespace(rowpool=root.rowpool, qtb=None)
    t0 = time.monotonic()
    setup = RootLifted.build(root.net, history, root.dm_l, root.dm_u, root.c, root.rhs,
                             standard_form_to_input_constraints(root.base_input_constraints), root.nlp_config,
                             allow_obbt=getattr(root, "use_reordered_bab", False))
    root.early_lifted_setup = setup
    root.ledger.commit_current('early_lifted_build', time.monotonic() - t0, root.values)
    if setup is not None and root.time_left() > 0:
        t0 = time.monotonic()
        bound = setup.bound(root.primal_solution, root.time_left())
        if bound is not None and math.isfinite(bound):
            root.certified_dual = max(root.certified_dual, bound)
        root.ledger.commit_current('early_lifted_bound', time.monotonic() - t0, root.values)
        if root.closes(root.values()[1]):
            raise RootClosed('early_lifted_bound')


def run_nlp(root, runner, *args, **kwargs):
    """Poll publications inside ALM's inner work, including restoration forwards.

    The hook only polls on the owning thread. Reentrancy is suppressed while the
    original-model verifier runs. Unwinding destroys the current ALM stack; this
    is cancellation after closure, never a claim of resumable ALM state.
    """
    stop = threading.Event()
    config = dict(kwargs['nlp_config'])
    def publish(family, point, seconds):
        if family == 'alm':
            # ALM's host interval contains the synchronous dual service above.
            # Charge that work once, to its own receipts, and carry ALM's entire
            # zero-gain prefix to the candidate it finally returns.
            seconds = max(0.0, seconds - (root.controller_nested_s -
                              getattr(root, 'controller_alm_nested_start', root.controller_early_s)))
            root.controller_nested_s += seconds
        offer(root, family, point, seconds)
    config['_controller_publish'] = publish
    config['_controller_alm_start'] = lambda: setattr(root, 'controller_alm_nested_start',
                                                      root.controller_nested_s)
    config['_controller_stop'] = stop
    config['_controller_deadline'] = root.deadline
    config['_controller_monotonic_deadline'] = time.monotonic() + root.time_left()
    kwargs['nlp_config'] = config
    owner, busy = threading.get_ident(), [False]

    def checkpoint(module, inputs):
        if threading.get_ident() != owner or busy[0]:
            return
        busy[0] = True
        try:
            if root.time_left() <= 0:
                raise RootDeadline()
            poll(root)
            if root.time_left() <= 0:
                raise RootDeadline()
        except (RootClosed, RootDeadline):
            stop.set()
            raise
        finally:
            busy[0] = False

    def producer_checkpoint():
        was_busy, busy[0] = busy[0], True
        try:
            if root.time_left() <= 0:
                raise RootDeadline()
            poll(root, service=False)
            if root.time_left() <= 0:
                raise RootDeadline()
        except (RootClosed, RootDeadline):
            stop.set()
            raise
        finally:
            busy[0] = was_busy

    t0 = time.monotonic()
    try:
        early_lp(root)
    finally:
        root.controller_early_s = time.monotonic() - t0
        root.controller_nested_s = getattr(root, 'controller_nested_s', 0.0) + root.controller_early_s
    if root.time_left() <= 0:
        raise RootDeadline()
    kwargs['root_wall_budget'] = max(0.001, root.time_left())
    root.controller_checkpoint = producer_checkpoint
    setup = getattr(root, 'early_lifted_setup', None)
    if setup is not None:
        setup.checkpoint = producer_checkpoint
    hook = root.net.model_ori.register_forward_pre_hook(checkpoint)
    try:
        result = runner(*args, **kwargs)
        busy[0] = True
        if root.time_left() <= 0:
            raise RootDeadline()
        poll(root)
        return result
    except (RootClosed, RootDeadline):
        stop.set()
        raise
    finally:
        stop.set()
        hook.remove()
        root.controller_checkpoint = None
        if setup is not None:
            setup.checkpoint = None


def settle_cancelled(root):
    """Charge outstanding messages without changing a result already committed."""
    while True:
        try:
            family, _, seconds = root.controller_inbox.get_nowait()
        except queue.Empty:
            return
        root.ledger.commit_current(family + '_cancelled', seconds, root.values)
