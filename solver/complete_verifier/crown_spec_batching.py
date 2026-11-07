"""Utilities for batching independent final specification rows."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import torch

try:
    from auto_LiRPA.patches import Patches
except ImportError:  # pragma: no cover - auto_LiRPA is present in verifier runs.
    Patches = ()  # type: ignore[assignment]


def _should_batch_specs(C: torch.Tensor | None, spec_batch_size: int | float | None) -> bool:
    if C is None or not isinstance(C, torch.Tensor) or C.ndim < 3:
        return False
    if spec_batch_size is None:
        return False
    try:
        spec_batch_size = int(spec_batch_size)
    except OverflowError:
        return False
    return spec_batch_size > 0 and C.shape[1] > spec_batch_size


def _slice_spec_kwarg(value: Any, start: int, end: int, C: torch.Tensor) -> Any:
    if (
        isinstance(value, torch.Tensor)
        and value.ndim >= 2
        and value.shape[0] == C.shape[0]
        and value.shape[1] == C.shape[1]
    ):
        return value[:, start:end]
    return value


def _same_unstable_idx(values: list[Any]) -> bool:
    first = values[0]
    for value in values[1:]:
        if first is None or value is None:
            if first is not value:
                return False
        elif isinstance(first, torch.Tensor) and isinstance(value, torch.Tensor):
            if first.shape != value.shape or not torch.equal(first, value):
                return False
        elif isinstance(first, tuple) and isinstance(value, tuple):
            if len(first) != len(value):
                return False
            for first_item, value_item in zip(first, value):
                if first_item.shape != value_item.shape or not torch.equal(first_item, value_item):
                    return False
        else:
            return False
    return True


def _merge_unstable_idx(values: list[Any], chunk_starts: list[int]) -> Any:
    if all(value is None for value in values):
        return None
    if _same_unstable_idx(values):
        return values[0]

    first = next(value for value in values if value is not None)
    if isinstance(first, torch.Tensor):
        return torch.cat([
            value + start for value, start in zip(values, chunk_starts)
            if value is not None
        ], dim=0)
    if isinstance(first, tuple):
        if not first:
            return None
        return tuple(
            torch.cat([
                value[idx] for value in values if value is not None
            ], dim=0)
            for idx in range(len(first))
        )
    return first


def _merge_patches(values: list[Any], chunk_starts: list[int]) -> Any:
    first = values[0]
    if not isinstance(first, Patches):
        return first
    unstable_idx = _merge_unstable_idx(
        [value.unstable_idx for value in values], chunk_starts
    )
    return first.create_similar(
        torch.cat([value.patches for value in values], dim=0),
        unstable_idx=unstable_idx,
    )


def _merge_values(
    values: list[Any],
    batch_size: int,
    chunk_starts: list[int],
    key: str | None = None,
) -> Any:
    if all(value is None for value in values):
        return None
    if key == "unstable_idx":
        return _merge_unstable_idx(values, chunk_starts)

    first = values[0]
    if isinstance(first, torch.Tensor):
        if first.ndim >= 2 and first.shape[0] == batch_size:
            return torch.cat(values, dim=1)
        return torch.cat(values, dim=0)

    if isinstance(first, Patches):
        return _merge_patches(values, chunk_starts)

    if isinstance(first, dict):
        merged = {}
        for child_key in first:
            merged[child_key] = _merge_values(
                [value[child_key] for value in values],
                batch_size,
                chunk_starts,
                child_key,
            )
        return merged

    return first


def _merge_compute_bounds_returns(
    rets: list[tuple[Any, ...]],
    batch_size: int,
    chunk_starts: list[int],
) -> tuple[Any, ...]:
    merged = []
    for idx in range(len(rets[0])):
        merged.append(_merge_values([ret[idx] for ret in rets], batch_size, chunk_starts))
    return tuple(merged)


def _get_optimize_bound_args(net: Any) -> dict[str, Any] | None:
    try:
        optimize_bound_args = net.bound_opts["optimize_bound_args"]
    except (AttributeError, KeyError, TypeError):
        return None
    return optimize_bound_args if isinstance(optimize_bound_args, dict) else None


def _extract_spec_threshold(
    stop_criterion_func: Any,
    C: torch.Tensor,
) -> torch.Tensor | None:
    closure = getattr(stop_criterion_func, "__closure__", None)
    if closure is None:
        return None
    for cell in closure:
        try:
            value = cell.cell_contents
        except ValueError:
            continue
        if (
            isinstance(value, torch.Tensor)
            and value.ndim >= 2
            and value.shape[0] == C.shape[0]
            and value.shape[1] == C.shape[1]
        ):
            return value
    return None


def _infer_stop_criterion_kind(
    stop_criterion_func: Any,
    threshold: torch.Tensor,
) -> str | None:
    if threshold.ndim < 2:
        return None

    try:
        probe = torch.where(
            torch.isfinite(threshold),
            threshold - 1,
            torch.zeros_like(threshold),
        )
        probe[:, 0] = torch.where(
            torch.isfinite(threshold[:, 0]),
            threshold[:, 0] + 1,
            torch.ones_like(threshold[:, 0]),
        )
        out = stop_criterion_func(probe)
    except Exception:  # pragma: no cover - best-effort compatibility path.
        return None

    if not isinstance(out, torch.Tensor):
        return None
    if out.shape == threshold.shape:
        return "elementwise"
    if out.ndim >= 2 and out.shape[0] == threshold.shape[0] and out.shape[1] == 1:
        return "any" if bool(out.any().item()) else "all"
    return None


def _make_stop_criterion_func(
    kind: str | None,
    threshold: torch.Tensor,
) -> Any:
    if kind == "elementwise":
        return lambda x: x > threshold
    if kind == "any":
        return lambda x: (x > threshold).any(dim=1, keepdim=True)
    if kind == "all":
        return lambda x: (x > threshold).all(dim=1, keepdim=True)
    return None


@contextmanager
def _chunk_stop_criterion(
    optimize_bound_args: dict[str, Any] | None,
    original_stop_criterion_func: Any,
    threshold: torch.Tensor | None,
    kind: str | None,
    start: int,
    end: int,
):
    if optimize_bound_args is None or threshold is None:
        yield
        return

    chunk_stop_criterion_func = _make_stop_criterion_func(
        kind, threshold[:, start:end])
    if chunk_stop_criterion_func is None:
        yield
        return

    optimize_bound_args["stop_criterion_func"] = chunk_stop_criterion_func
    try:
        yield
    finally:
        optimize_bound_args["stop_criterion_func"] = original_stop_criterion_func


def _index_select_spec_dim(
    value: Any,
    keep_indices: torch.Tensor,
    batch_size: int,
    num_specs: int,
    key: str | None = None,
) -> Any:
    if value is None:
        return None
    if key == "unstable_idx":
        return value

    if isinstance(value, torch.Tensor):
        indices = keep_indices.to(value.device)
        if value.ndim >= 2 and value.shape[0] == batch_size and value.shape[1] == num_specs:
            return value.index_select(1, indices)
        if value.shape[0] == num_specs and value.shape[0] != batch_size:
            return value.index_select(0, indices)
        return value

    if isinstance(value, Patches):
        if value.patches.shape[0] != num_specs:
            return value
        return value.create_similar(
            value.patches.index_select(0, keep_indices.to(value.patches.device)),
            unstable_idx=value.unstable_idx,
        )

    if isinstance(value, dict):
        return {
            child_key: _index_select_spec_dim(
                child_value, keep_indices, batch_size, num_specs, child_key)
            for child_key, child_value in value.items()
        }

    if isinstance(value, tuple):
        return tuple(
            _index_select_spec_dim(item, keep_indices, batch_size, num_specs)
            for item in value
        )

    return value


def _filter_verified_specs_from_merged_ret(
    merged_ret: tuple[Any, ...],
    C: torch.Tensor,
    threshold: torch.Tensor | None,
) -> tuple[Any, ...]:
    if threshold is None or len(merged_ret) == 0:
        return merged_ret

    lb = merged_ret[0]
    if (
        not isinstance(lb, torch.Tensor)
        or lb.ndim < 2
        or C.shape[0] != 1
        or lb.shape[0] != C.shape[0]
        or lb.shape[1] != C.shape[1]
    ):
        return merged_ret

    try:
        threshold = threshold.to(device=lb.device, dtype=lb.dtype)
        verified = lb > threshold
    except RuntimeError:
        return merged_ret
    if verified.shape != lb.shape:
        return merged_ret

    keep_mask = verified[0].logical_not()
    if keep_mask.all():
        return merged_ret

    keep_indices = keep_mask.nonzero(as_tuple=True)[0]
    num_specs = C.shape[1]
    return tuple(
        _index_select_spec_dim(value, keep_indices, C.shape[0], num_specs)
        for value in merged_ret
    )


def _restore_return_A_state(net: Any, merged_ret: tuple[Any, ...]) -> None:
    if len(merged_ret) < 3 or not isinstance(merged_ret[-1], dict):
        return
    A_dict = merged_ret[-1]
    try:
        net.A_dict = A_dict
        output_name = net.output_name[0]
        input_name = net.input_name[0]
        input_A = A_dict[output_name][input_name]
        input_node = net[input_name]
    except (AttributeError, KeyError, IndexError, TypeError):
        return

    lA = input_A.get("lA")
    uA = input_A.get("uA")
    if isinstance(lA, torch.Tensor):
        input_node.lA = lA.transpose(0, 1).detach()
    if isinstance(uA, torch.Tensor):
        input_node.uA = uA.transpose(0, 1).detach()


from nlp_solver.continuation import Atom, retained
import time

@retained

def compute_bounds_in_spec_batches(
    net: Any,
    *,
    C: torch.Tensor | None,
    spec_batch_size: int | float | None,
    oom_retry: bool = False,
    _sliced: bool = False,
    _work_cap: Any = None,
    **compute_bounds_kwargs: Any,
) -> tuple[Any, ...]:
    """``_compute_bounds_in_spec_batches``; with ``oom_retry``, a call that runs out
    of GPU memory is repeated with half as many spec rows per chunk.

    Chunking does not change the bound of any row, so this only trades time for
    memory. It is meant for the root bound (one domain, so the BaB batch retry
    cannot help): pooling_sppb5pq/5tp died there on a 2 GiB A matrix. A single
    row per chunk that still does not fit raises.
    """
    if not oom_retry:
        return (yield from _compute_bounds_in_spec_batches(
            net, C=C, spec_batch_size=spec_batch_size, _sliced=_sliced, _work_cap=_work_cap,
            _iterate=True, **compute_bounds_kwargs))
    size = spec_batch_size
    while True:
        try:
            return (yield from _compute_bounds_in_spec_batches(
                net, C=C, spec_batch_size=size, _sliced=_sliced, _work_cap=_work_cap,
                _iterate=True, **compute_bounds_kwargs))
        except RuntimeError as error:
            from nlp_solver.utils import is_cuda_oom
            if not is_cuda_oom(error) or not isinstance(C, torch.Tensor) or C.ndim < 3:
                raise
            current = int(size) if _should_batch_specs(C, size) else C.shape[1]
            if current <= 1:
                raise
            size = max(1, current // 2)
            print(f"CROWN spec batching: {current} rows per chunk ran out of GPU "
                  f"memory; retrying with {size}.")
        # Outside except: its traceback keeps the failed A tensors alive.
        torch.cuda.empty_cache()


@retained
def _compute_bounds_in_spec_batches(
    net: Any,
    *,
    C: torch.Tensor | None,
    spec_batch_size: int | float | None,
    _sliced: bool = False,
    _work_cap: Any = None,
    **compute_bounds_kwargs: Any,
) -> tuple[Any, ...]:
    """Call ``compute_bounds`` with final C rows split along the spec dimension.

    Rows in ``C[:, i, :]`` are independent linear specifications. Splitting them
    reduces the size of the backward A chain without changing the bound being
    computed for any row.
    """
    if _sliced and C is not None and C.shape[1] > 1:
        spec_batch_size = min(C.shape[1], int(spec_batch_size or C.shape[1]))
    elif not _should_batch_specs(C, spec_batch_size):
        from input_split.tree_slice import region
        with region('crown_specs', 1 if C is None else C.shape[1]) if _sliced else nullcontext():
            ret = net.compute_bounds(C=C, **compute_bounds_kwargs)
        if _sliced:
            yield Atom('gpu_tree')
        return ret

    assert C is not None
    spec_batch_size = int(spec_batch_size)
    num_specs = C.shape[1]
    total_batches = (num_specs + spec_batch_size - 1) // spec_batch_size
    optimize_bound_args = _get_optimize_bound_args(net)
    original_stop_criterion_func = (
        optimize_bound_args.get("stop_criterion_func")
        if optimize_bound_args is not None else None
    )
    spec_threshold = _extract_spec_threshold(original_stop_criterion_func, C)
    stop_criterion_kind = (
        _infer_stop_criterion_kind(original_stop_criterion_func, spec_threshold)
        if spec_threshold is not None else None
    )
    print(
        "CROWN spec batching: "
        f"{num_specs} final spec rows in {total_batches} chunks "
        f"(chunk size {spec_batch_size})."
    )

    _in_name = net.input_name[0] if isinstance(net.input_name, (list, tuple)) else net.input_name
    _chunk_A_entries: list = []
    merged_ret = None
    start, chunk_size = 0, 1 if _sliced else spec_batch_size
    if _sliced:
        from input_split.tree_slice import ServiceSizer
        sizer = ServiceSizer()
    while start < num_specs:
        if _sliced:
            chunk_size = sizer.size(spec_batch_size, _work_cap() if _work_cap is not None else None)
        end = min(start + chunk_size, num_specs)
        started = time.monotonic()
        chunk_kwargs = {
            key: _slice_spec_kwarg(value, start, end, C)
            for key, value in compute_bounds_kwargs.items()
        }
        chunk_C = C[:, start:end]
        with _chunk_stop_criterion(
            optimize_bound_args,
            original_stop_criterion_func,
            spec_threshold,
            stop_criterion_kind,
            start,
            end,
        ):
            from input_split.tree_slice import region
            with region('crown_specs', end-start) if _sliced else nullcontext():
                chunk_ret = net.compute_bounds(C=chunk_C, **chunk_kwargs)
        # Snapshot the input node's per-chunk lA/uA. compute_bounds writes these on the
        # node (shape (spec, batch, *in)); each chunk overwrites them, so without this
        # only the LAST chunk survives. BatchedlA.from_net reads input_node.lA, so we
        # must stitch the full-spec version back after the loop or input-split BaB / clip
        # sees a truncated lA (e.g. 241 of 3313 rows) that mismatches the full global_lb.
        try:
            _inn = net[_in_name]
            _chunk_A_entries.append((
                _inn.lA.detach().clone() if torch.is_tensor(getattr(_inn, "lA", None)) else None,
                _inn.uA.detach().clone() if torch.is_tensor(getattr(_inn, "uA", None)) else None,
            ))
        except (AttributeError, KeyError, TypeError):
            _chunk_A_entries.append((None, None))
        if merged_ret is None:
            merged_ret = chunk_ret
        else:
            merged_ret = _merge_compute_bounds_returns(
                [merged_ret, chunk_ret],
                batch_size=C.shape[0],
                chunk_starts=[0, start],
            )
        del chunk_ret
        if _sliced:
            sizer.account(end-start, time.monotonic()-started)
            yield Atom('gpu_tree')
        start = end
        if chunk_C.is_cuda:
            torch.cuda.empty_cache()

    assert merged_ret is not None
    if _sliced:
        total_batches = len(_chunk_A_entries)
    # Stitch per-chunk input-node lA/uA into full-spec tensors (concat on the spec axis 0
    # of (spec, batch, *in)) and write them back so BatchedlA.from_net sees all rows.
    def _cat_chunks(idx):
        vals = [e[idx] for e in _chunk_A_entries]
        if any(v is None for v in vals) or len(vals) != total_batches:
            return None
        return torch.cat(vals, dim=0)
    if total_batches > 1 and len(_chunk_A_entries) == total_batches:
        try:
            _inn = net[_in_name]
            _full_lA = _cat_chunks(0)
            _full_uA = _cat_chunks(1)
            if _full_lA is not None:
                _inn.lA = _full_lA
            if _full_uA is not None:
                _inn.uA = _full_uA
        except (AttributeError, KeyError, TypeError, RuntimeError):
            pass

    if not _sliced:
        merged_ret = _filter_verified_specs_from_merged_ret(
            merged_ret, C, spec_threshold)
    _restore_return_A_state(net, merged_ret)
    final_shape = C.size()[:2]
    if isinstance(merged_ret[0], torch.Tensor):
        final_shape = merged_ret[0].size()[:2]
    if hasattr(net, "final_shape"):
        net.final_shape = final_shape
    if hasattr(net, "bound_opts"):
        net.bound_opts.update({"final_shape": final_shape})
    return merged_ret
