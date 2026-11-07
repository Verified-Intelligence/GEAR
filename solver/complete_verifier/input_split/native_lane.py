"""Independent, cancellable native atoms; only the owner publishes certificates.

The installed cyipopt holds the GIL. Fresh interpreters also avoid inheriting a
live Torch/OpenMP pool. Two persistent CPU workers overlap their imports with
model setup; pipes carry small commands and NumPy results, never owner state.
"""
from concurrent.futures import ThreadPoolExecutor
import atexit
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time


def _worker(read_fd, write_fd):
    boot_cpu = time.process_time()
    from multiprocessing.connection import Connection
    import torch
    from joblib.externals import cloudpickle
    import nlp_solver  # noqa: F401; warm the native libraries before the first task
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
    incoming, outgoing = Connection(read_fd, readable=True, writable=False), Connection(write_fd, readable=False, writable=True)
    outgoing.send_bytes(pickle.dumps(('ready', time.monotonic(), time.process_time()-boot_cpu)))
    while True:
        try:
            path, paths, dtype, threads = pickle.loads(incoming.recv_bytes())
        except EOFError:
            break
        sys.path[:] = paths
        torch.set_default_dtype(getattr(torch, dtype))
        torch.set_num_threads(threads)
        started = time.monotonic()
        try:
            with open(path, 'rb') as source:
                fn, args = cloudpickle.load(source)
            if len(args) > 7 and isinstance(args[7], dict):
                args = list(args)
                def publish(point):
                    outgoing.send_bytes(pickle.dumps(('candidate', point, time.monotonic()-started)))
                args[7] = dict(args[7], _controller_point=publish)
            result = fn(*args)
            if hasattr(result, 'x') and isinstance(result.x, torch.Tensor):
                result.x = result.x.detach().cpu().numpy().copy()
            outgoing.send_bytes(pickle.dumps(('result', True, result, time.monotonic()-started)))
        except BaseException as error:
            outgoing.send_bytes(pickle.dumps(('result', False, repr(error), time.monotonic()-started)))
        finally:
            # Do not retain one model/graph while another job is pending.
            fn = args = result = None
    incoming.close()
    outgoing.close()


class _Slot:
    def __init__(self):
        from multiprocessing.connection import Connection
        task_read, task_write = os.pipe()
        result_read, result_write = os.pipe()
        self.started = time.monotonic()
        paths = [os.path.dirname(os.path.dirname(__file__))] + sys.path
        env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in paths if p))
        self.process = subprocess.Popen(
            [sys.executable, '-u', '-m', 'input_split.native_lane', '--worker', str(task_read), str(result_write)],
            pass_fds=(task_read, result_write), env=env)
        os.close(task_read)
        os.close(result_write)
        self.send = Connection(task_write, readable=False, writable=True)
        self.receive = Connection(result_read, readable=True, writable=False)
        self.bootstrap = None

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.send.close()
        self.receive.close()


class _Workers:
    def __init__(self):
        self.slots = [_Slot(), _Slot()]
        self.available = queue.Queue()
        for slot in self.slots:
            self.available.put(slot)

    def close(self):
        for slot in self.slots:
            slot.close()


_BOOT_LOCK = threading.Lock()
_WARM = None


def prewarm():
    """Start fresh interpreters early; no native or model state is forked."""
    global _WARM
    with _BOOT_LOCK:
        if _WARM is None:
            _WARM = _Workers()


def _take_workers():
    global _WARM
    prewarm()
    with _BOOT_LOCK:
        workers, _WARM = _WARM, None
        return workers


def _close_unused():
    global _WARM
    with _BOOT_LOCK:
        if _WARM is not None:
            _WARM.close()
            _WARM = None


atexit.register(_close_unused)


class NativeExecutor:
    def __init__(self, root, stop):
        self.root, self.stop = root, stop
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.workers = _take_workers()

    def submit(self, fn, *args):
        was_committing = getattr(self.root, 'controller_committing', False)
        self.root.controller_committing = True
        try:
            return self._submit(fn, *args)
        finally:
            self.root.controller_committing = was_committing

    def _submit(self, fn, *args):
        import torch
        clean = []
        for arg in args:
            if isinstance(arg, dict):
                arg = {k: v for k, v in arg.items() if k not in
                       ('_controller_publish', '_controller_stop', '_controller_native_wait', '_controller_native_submit')}
            clean.append(arg)
        from nlp_solver.ipopt_solver import NNProblemIPOPT
        rows = (clean[8] if fn.__name__ == 'solve_seed' else clean[10]) if len(clean) > 8 else None
        constraints = clean[5].shape[-2] + (0 if rows is None else rows[0].shape[-2]) if len(clean) > 5 else 0
        if len(clean) > 1 and clean[1].is_cuda and clean[1].numel() * max(constraints, 1) <= NNProblemIPOPT.CPU_EVAL_MAX_ENTRIES:
            import copy
            def cpu_arg(value):
                if isinstance(value, torch.Tensor):
                    return value.detach().cpu()
                if isinstance(value, tuple):
                    return tuple(cpu_arg(v) for v in value)
                return value
            clean = [cpu_arg(arg) for arg in clean]
            clean[0] = copy.deepcopy(clean[0]).cpu()
        from joblib.externals import cloudpickle
        fd, path = tempfile.mkstemp(prefix='s4_native_', suffix='.pkl')
        try:
            with os.fdopen(fd, 'wb') as target:
                cloudpickle.dump((fn, tuple(clean)), target)
        except BaseException:
            os.unlink(path)
            raise
        family = 'ipopt_seed' if fn.__name__ == 'solve_seed' else 'ipopt'
        return self.pool.submit(self._call, path, str(torch.get_default_dtype()).split('.')[-1],
                                torch.get_num_threads(), family, list(sys.path))

    def _call(self, path, dtype, threads, family, paths):
        slot = self.workers.available.get()
        if slot.process.poll() is not None and not self.stop.is_set() and self.root.time_left() > 0:
            slot = _Slot()
            self.workers.slots.append(slot)
        started = time.monotonic()
        point, result = None, None
        bootstrap, charged = 0., 0.
        completed, service = False, 0.
        try:
            if self.stop.is_set() or self.root.time_left() <= 0:
                return None
            slot.send.send_bytes(pickle.dumps((path, paths, dtype, threads)))
            while not self.stop.is_set() and self.root.time_left() > 0:
                if slot.receive.poll(min(.05, max(0.0, self.root.time_left()))):
                    try:
                        message = pickle.loads(slot.receive.recv_bytes())
                    except EOFError:
                        print(f'[NativeAtom] failed exit={slot.process.poll()} no_result=1', flush=True)
                        break
                    if message[0] == 'ready':
                        slot.bootstrap = message[1]-slot.started
                        bootstrap = slot.bootstrap
                        print(f'[NativeAtom] bootstrap_s={bootstrap:.9g} bootstrap_cpu_s={message[2]:.9g}', flush=True)
                        continue
                    if message[0] == 'candidate':
                        _, candidate, service = message
                        work = service + bootstrap
                        self._publish(family, candidate, max(0., work-charged))
                        charged = work
                        continue
                    _, ok, value, service = message
                    completed = True
                    if not ok:
                        print(f'[NativeAtom] failed error={value}', flush=True)
                        break
                    result = value
                    point = getattr(result, 'x', getattr(result, 'x_solution', None))
                    break
                if slot.process.poll() is not None:
                    print(f'[NativeAtom] failed exit={slot.process.returncode} no_result=1', flush=True)
                    break
        finally:
            if not completed:
                # IPOPT cannot indefinitely hold the owner's shared deadline.
                slot.close()
            os.unlink(path)
            self.workers.available.put(slot)
            prior = started-slot.started if slot.bootstrap is None else min(started-slot.started, bootstrap)
            work = service + bootstrap if completed else time.monotonic()-started+prior
            self._publish(family, point, max(0., work-charged))
        return result

    def _publish(self, family, point, work):
        self.root.controller_inbox.put((family, point, work))
        notify = getattr(self.root, 'controller_notify', None)
        if notify is not None:
            notify()

    def shutdown(self, wait=True):
        self.pool.shutdown(wait=wait)
        self.workers.close()


def solve_seed(model, lo, hi, C_obj, rhs_obj, C_constr, rhs_constr, config, rows, seed):
    from nlp_solver.ipopt_solver import IPOPTSolver, NNProblemIPOPT
    problem = NNProblemIPOPT(model, lo.squeeze(0), hi.squeeze(0), C_obj, rhs_obj, C_constr, rhs_constr,
                             input_constr_A=None if rows is None else rows[0],
                             input_constr_b=None if rows is None else rows[1], device=None, dtype=None)
    return IPOPTSolver(config).solve_single(problem, x_init=seed, timeout=config['ipopt_timeout'])


if __name__ == '__main__':
    _worker(int(sys.argv[2]), int(sys.argv[3]))
