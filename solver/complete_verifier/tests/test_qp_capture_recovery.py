"""Model the PyTorch failure that leaves a generator capturing and skips stream exit."""
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from nlp_solver.alm_polish import _capture_qp_step, _make_qp_solver, _qp_ipm


@pytest.mark.parametrize('failure', ['begin', 'begin_allocated', 'body', 'body_clean', 'end', 'end_after_allocator', 'instantiate', None])
def test_failed_capture_restores_stream_generator_and_allocator(monkeypatch, failure):
    cuda = torch.cuda
    state = SimpleNamespace(stream='owner', capturing=False, pool_refs=0, allocating=False)
    original = SimpleNamespace(capturing=False, offset=28)
    class Generator:
        current = original
        def graphsafe_get_state(self):
            return self.current
        def clone_state(self):
            return SimpleNamespace(capturing=False, offset=self.current.offset)
        def graphsafe_set_state(self, other):
            self.current = other
        def rand(self):
            assert not self.current.capturing, 'Offset increment outside graph capture'
            self.current.offset += 4
            return self.current.offset
    generator = Generator()
    pool = (0, 123)
    class Graph:
        completed = False
        def pool(self):
            if not self.completed:
                raise RuntimeError('capture_end must have been called')
            return pool
        def reset(self):
            if self.completed:
                state.pool_refs -= 1
    graph = Graph()
    class Stream:
        def wait_stream(self, other): pass
    side = Stream()
    @contextmanager
    def stream(selected):
        previous, state.stream = state.stream, selected
        try:
            yield
        finally:
            state.stream = previous
    @contextmanager
    def device(selected):
        yield
    class Capture:
        def __enter__(self):
            generator.current.capturing = True
            if failure == 'begin':
                raise RuntimeError('begin failed')
            state.pool_refs += 1
            state.allocating = True
            if failure == 'begin_allocated':
                raise RuntimeError('begin failed after pool registration')
            state.capturing = True
        def __exit__(self, *args):
            # cudaStreamEndCapture always ends the stream capture, including
            # invalidation. PyTorch's epilogue/stream guard can still be skipped.
            state.capturing = False
            if failure in ('body', 'end'):
                raise RuntimeError('capture invalidated')
            state.allocating = False
            if failure == 'end_after_allocator':
                raise RuntimeError('end failed after allocator exit')
            generator.current.capturing = False
            graph.completed = True
            if failure == 'instantiate':
                raise RuntimeError('instantiate failed')
    def end_pool(index, token):
        if not state.allocating:
            raise RuntimeError('endAllocatePool: not currently recording to mempool_id')
        state.allocating = False
    def release_pool(index, token):
        state.pool_refs -= 1
    monkeypatch.setattr(cuda, 'device', device)
    monkeypatch.setattr(cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(cuda, 'default_generators', (generator,))
    monkeypatch.setattr(cuda, 'stream', stream)
    monkeypatch.setattr(cuda, 'graph_pool_handle', lambda: pool)
    monkeypatch.setattr(cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(cuda, 'current_stream', lambda *a: side)
    graph.capture_begin = lambda *a, **k: Capture().__enter__()
    graph.capture_end = lambda: Capture().__exit__()
    monkeypatch.setattr(torch._C, '_cuda_endAllocateToPool', end_pool)
    monkeypatch.setattr(torch._C, '_cuda_releasePool', release_pool)
    def step():
        assert state.stream is side and state.capturing
        assert generator.current is not original
        if failure in ('body', 'body_clean'):
            raise KeyboardInterrupt('body interrupted')
    if failure:
        with pytest.raises(KeyboardInterrupt if failure == 'body_clean' else RuntimeError):
            _capture_qp_step(graph, step, 'cuda:0', side)
        assert state.pool_refs == 0
    else:
        _capture_qp_step(graph, step, 'cuda:0', side)
        assert graph.completed and state.pool_refs == 1
    assert state.stream == 'owner' and not state.capturing and not state.allocating
    assert generator.current is original and original.offset == 28
    assert generator.rand() == 32


def test_qp_fallback_runs_after_recovery(monkeypatch):
    def fail(*args):
        raise RuntimeError('capture failed')
    calls = []
    monkeypatch.setattr('nlp_solver.alm_polish._GraphedQP', fail)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda device: calls.append(device))
    device = torch.device('cuda:0')
    assert _make_qp_solver(1, 2, 1, torch.float64, device) is _qp_ipm
    assert calls == [device]


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required; CPU job only models failure')
def test_cuda_invalid_capture_leaves_rng_and_next_capture_usable():
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    owner = torch.cuda.current_stream(device)
    generator = torch.cuda.default_generators[0]
    expected = generator.clone_state()
    x = torch.ones(8, device=device)
    def invalid():
        x.add_(1)
        torch.cuda.synchronize(device)  # prohibited inside capture
    for _ in range(2):
        with pytest.raises(RuntimeError):
            _capture_qp_step(torch.cuda.CUDAGraph(), invalid, device, stream)
        assert torch.cuda.current_stream(device) == owner
        assert not torch.cuda.is_current_stream_capturing()
        assert torch.equal(torch.rand(16, device=device), torch.rand(16, device=device, generator=expected))
        graph = torch.cuda.CUDAGraph()
        _capture_qp_step(graph, lambda: x.add_(1), device, stream)
        graph.replay()
        torch.cuda.synchronize(device)


@pytest.mark.parametrize('enabled', [True, False])
@pytest.mark.parametrize('failure', ['end', None])
def test_capture_cleans_up_a_failed_end_and_restores_gc(monkeypatch, failure, enabled):
    import gc
    from auto_LiRPA import cuda_memory
    calls, seen = [], []
    class Graph:
        def capture_begin(self, *a, **k): calls.append('begin')
        def capture_end(self):
            seen.append(gc.isenabled())
            if failure: raise RuntimeError('end failed')
        def pool(self): raise RuntimeError('capture_end must have been called')
        def reset(self): calls.append('reset')
    class Stream:
        def wait_stream(self, other): pass
    @contextmanager
    def stream(selected): yield
    monkeypatch.setattr(torch.cuda, 'stream', stream)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'graph_pool_handle', lambda: (0, 7))
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: None)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda *a: Stream())
    monkeypatch.setattr(torch._C, '_cuda_endAllocateToPool', lambda index, token: calls.append('end_pool'))
    monkeypatch.setattr(torch._C, '_cuda_releasePool', lambda index, token: calls.append('release'))
    (gc.enable if enabled else gc.disable)()
    try:
        if failure:
            with pytest.raises(RuntimeError):
                cuda_memory.capture(Graph(), lambda: None, stream=Stream())
            assert calls == ['begin', 'end_pool', 'release', 'reset']
        else:
            cuda_memory.capture(Graph(), lambda: None, stream=Stream())
            assert calls == ['begin']
        assert seen == [False] and gc.isenabled() == enabled
    finally:
        gc.enable()
