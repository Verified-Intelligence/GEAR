"""Admission for optional CUDA work, including allocator and graph-pool residency."""
import gc
import functools
import re
import types
from contextlib import contextmanager

import torch


class MemoryBudgetExceeded(RuntimeError):
    pass


def memory_error(error):
    return (isinstance(error, (MemoryError, torch.OutOfMemoryError, MemoryBudgetExceeded))
            or isinstance(error, RuntimeError) and any(message in str(error).lower() for message in (
                'out of memory', "can't allocate memory", 'not enough memory', 'bad_alloc', 'alloc_failed')))


def cuda_launch_error(error):
    """Recoverable launch/capture failures, excluding poisoned CUDA contexts."""
    return isinstance(error, RuntimeError) and any(message in str(error) for message in (
        'invalid configuration argument', 'too many resources requested for launch',
        'previous error during capture'))


@functools.lru_cache(maxsize=None)
def _capture_stream(device):
    return torch.cuda.Stream(device=device)  # one per device: captures sharing a pool must share a stream


def capture(graph, fn, *, pool=None, stream=None):
    """Capture fn into graph like torch.cuda.graph, without its gc.collect/empty_cache.

    A failed capture leaves the graph reset and its pool released, so a retry can reuse the pool.
    Automatic gc is off meanwhile: finalizers must not run on the capturing thread.
    """
    side = stream if stream is not None else _capture_stream(torch.cuda.current_device())
    pool = torch.cuda.graph_pool_handle() if pool is None else pool
    torch.cuda.synchronize()
    side.wait_stream(torch.cuda.current_stream())
    begun, collecting = False, gc.isenabled()
    gc.disable()
    try:
        with torch.cuda.stream(side):
            graph.capture_begin(pool, capture_error_mode='thread_local')
            begun = True
            try: out = fn()
            finally: graph.capture_end()
    except BaseException:
        try:
            graph.pool()  # available only after a completed capture_end
        except RuntimeError:
            index = torch.cuda.current_device()
            allocating = begun  # capture_begin may also fail after registering its pool
            try:
                torch._C._cuda_endAllocateToPool(index, pool)
                allocating = True
            except RuntimeError as error:
                if "not currently recording to mempool_id" not in str(error):
                    raise
            if allocating:
                torch._C._cuda_releasePool(index, pool)
        graph.reset()
        raise
    finally:
        if collecting: gc.enable()
    torch.cuda.current_stream().wait_stream(side)
    return out


def release_cuda(device):
    gc.collect()
    if torch.device(device).type == 'cuda':
        torch.cuda.empty_cache()


def dense_chunk(device, per_lane_bytes, requested):
    if torch.device(device).type != 'cuda':
        return requested
    free = torch.cuda.mem_get_info(device)[0]
    lanes = min(requested, int(free // (2 * max(1, per_lane_bytes))))
    if lanes < 1:
        raise MemoryBudgetExceeded('one dense LP table and its workspaces exceed free-memory allowance')
    return lanes


def optional_cuda_work(fn):
    """An allocation failure in an optional proof path supplies no new proof."""
    @functools.wraps(fn)
    def run(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as error:
            if not memory_error(error) and not cuda_launch_error(error):
                raise
            print(f'[crown-memory] {fn.__name__} skipped: {error}', flush=True)
        release_cuda('cuda')
        return None
    return run


def bounded_intermediate(fn):
    """A failed optional backward tightening keeps the interval enclosure."""
    @functools.wraps(fn)
    def run(self, node, *args, **kwargs):
        try:
            return fn(self, node, *args, **kwargs)
        except Exception as error:
            if not memory_error(error):
                raise
            print(f'[crown-memory] intermediate {node.name}: interval fallback', flush=True)
        release_cuda(self.device)
        interval = self.IBP_general(node=node)
        node.lower, node.upper = interval.lower, interval.upper
    return run


def owned_cuda_bytes(owner, segments):
    """Count live owned storage once, including a captured pool's unused blocks.

    Cache entries include closures, so their captured inputs/workspaces must be
    visited too. Native networks and the symbolic host graph are caller state.
    """
    seen, storages, pools, alive = set(), {}, set(), []
    def visit(value):
        if id(value) in seen: return
        seen.add(id(value))
        if torch.is_tensor(value):
            # Sparse tensors have no storage of their own; count their parts.
            # The part views are temporaries: keep them alive so that their
            # id() is not reused while `seen` still holds it.
            parts = ((value.crow_indices(), value.col_indices(), value.values())
                     if value.layout == torch.sparse_csr else
                     (value._indices(), value._values()) if value.layout == torch.sparse_coo else None)
            if parts is not None:
                alive.append(parts)
                for part in parts: visit(part)
            elif value.is_cuda:
                storage = value.untyped_storage()
                storages[storage.data_ptr()] = storage.nbytes()
        elif isinstance(value, torch.cuda.CUDAGraph):
            pools.add(value.pool())
        elif isinstance(value, dict):
            for item in value.values(): visit(item)
        elif isinstance(value, (tuple, list)):
            for item in value: visit(item)
        elif isinstance(value, types.FunctionType):
            for cell in value.__closure__ or ():
                try: contents = cell.cell_contents
                except ValueError: continue  # variable not assigned on this branch
                visit(contents)
        elif (not isinstance(value, (torch.nn.Module, types.ModuleType, type))
              and hasattr(value, '__dict__')):
            for name, item in vars(value).items():
                if name not in ('net', 'pg', 'memory'): visit(item)
    visit(owner)
    private = [s for s in segments if tuple(s.get('segment_pool_id', (0, 0))) in pools]
    return sum(s['total_size'] for s in private) + sum(
        size for address, size in storages.items() if not any(
            s['address'] <= address < s['address'] + s['total_size'] for s in private))


class CudaMemoryBudget:
    """Charge owned residency and this call, independently of tree growth.

    Driver free bytes and reusable process cache fund new allocations; the
    configured tree allowance remains untouched. No engine-creation baseline
    limits subsequent native/tree allocations. CPU admission is unrestricted.
    """
    def __init__(self, device, cap_bytes, reserve_bytes=0, owner=None):
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda',torch.cuda.current_device())
        self.reserve = int(reserve_bytes)
        self.enabled = self.device.type == 'cuda'
        self.limit = cap_bytes
        self.owner = owner
        self.resident = 0
        self.entry = None
        self.failed_need = self.failed_owned = self.requested = 0
        if self.enabled: release_cuda(self.device)

    def _measure(self):
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        segments = [s for s in torch.cuda.memory_snapshot() if s['device'] == self.device.index]
        # Graph-private cache cannot be lent to a different allocation pool.
        slack = sum(s['total_size']-s['allocated_size'] for s in segments
                    if tuple(s.get('segment_pool_id', (0, 0))) != (0, 0))
        own = owned_cuda_bytes(self.owner(), segments) if self.owner is not None else self.resident
        if self.entry is not None:
            own = max(own, self.entry[0] + allocated + slack - self.entry[1])
        cached = (sum(s['total_size']-s['active_size'] for s in segments
                      if tuple(s.get('segment_pool_id',(0,0))) == (0,0))
                  if segments else max(0,reserved-allocated-slack))
        return max(0, own), allocated+slack, reserved, cached

    def _available(self, own, cached):
        free = torch.cuda.mem_get_info(self.device)[0]
        return max(0, min(self.limit-own, free+cached-self.reserve))

    def available(self):
        if not self.enabled: return float('inf')
        own, _, _, cached = self._measure()
        return self._available(own,cached)

    def require(self, size):
        self.requested = max(self.requested, size)
        if not self.enabled: return
        own, _, _, cached = self._measure()
        available = self._available(own,cached)
        if size > available or own > self.limit:
            self.failed_need = max(self.failed_need,own+size)
            raise MemoryBudgetExceeded(f'need={int(size)} available={int(available)} limit={int(self.limit)}')

    def retry_need(self, workspace):
        """Additional bytes needed after evicting optional residency."""
        own = self._measure()[0] if self.enabled else 0
        # Cached graphs present at entry are optional. Their eviction must
        # not turn a recoverable single-lane OOM into a permanent cap failure.
        evicted = max(0,self.failed_owned-own)
        return max(workspace,self.failed_need-own-evicted)

    def snapshot(self):
        if not self.enabled: return {}
        return dict(limit_bytes=int(self.limit), available_bytes=int(self.available()),
                    owned_bytes=int(self._measure()[0]),
                    allocated_bytes=torch.cuda.memory_allocated(self.device),
                    reserved_bytes=torch.cuda.memory_reserved(self.device),
                    free_bytes=torch.cuda.mem_get_info(self.device)[0])

    @contextmanager
    def allocations(self):
        """Guard only the call's incremental reservation; restore the native cap."""
        if not self.enabled or self.entry is not None:
            yield
            return
        own, process, reserved, cached = self._measure()
        total = torch.cuda.mem_get_info(self.device)[1]
        previous = torch.cuda.memory.get_per_process_memory_fraction(self.device.index)
        fraction = min(previous, (reserved+self._available(own,cached))/total)
        torch.cuda.memory.set_per_process_memory_fraction(fraction, self.device.index)
        self.entry = own, process
        self.failed_need = self.requested = 0
        self.failed_owned = own
        try:
            yield
            self.require(0)
        except Exception as error:
            if memory_error(error):
                # The allocator reports its failed request, not a fitted margin.
                match = re.search(r'(?:allocate|allocating) ([0-9.]+)\s*([KMGT]?i?B|bytes)', str(error), re.I)
                request = 0
                if match:
                    unit = match[2].lower()
                    power = 'kmgt'.find(unit[0])+1 if unit[0] in 'kmgt' else 0
                    request = float(match[1])*1024**power
                self.failed_need = max(self.failed_need, own+self.requested, self._measure()[0]+request)
            raise
        finally:
            self.resident = self._measure()[0]
            self.entry = None
            torch.cuda.memory.set_per_process_memory_fraction(previous, self.device.index)
