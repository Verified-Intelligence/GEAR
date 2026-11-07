#########################################################################
##   This file is part of the auto_LiRPA library, a core part of the   ##
##   α,β-CROWN (alpha-beta-CROWN) neural network verifier developed    ##
##   by the α,β-CROWN Team                                             ##
##                                                                     ##
##   Copyright (C) 2020-2026 The α,β-CROWN Team                        ##
##   Team leaders:                                                     ##
##          Faculty:   Huan Zhang <huan@huan-zhang.com> (UIUC)         ##
##          Student:   Xiangru Zhong <xiangru4@illinois.edu> (UIUC)    ##
##                                                                     ##
##   See CONTRIBUTORS for all current and past developers in the team. ##
##                                                                     ##
##     This program is licensed under the BSD 3-Clause License,        ##
##        contained in the LICENCE file in this directory.             ##
##                                                                     ##
#########################################################################
from .base import *
from ..patches import Patches, patches_to_matrix
from torch.nn import Module


class BoundGather(Bound):
    def __init__(self, attr, x, output_index, options):
        super().__init__(attr, x, output_index, options)
        self.axis = attr['axis'] if 'axis' in attr else 0

    def forward(self, x, indices):
        self.indices = indices
        if self.axis == -1:
            self.axis = len(x.shape) - 1
        # BoundShape.shape() will return values on cpu only
        x = x.to(self.indices.device)
        if indices.ndim == 0:
            if indices == -1:
                self.indices = x.shape[self.axis] + indices
            return torch.index_select(x, dim=self.axis, index=self.indices).squeeze(self.axis)
        elif indices.ndim == 1:
            if self.axis == 0:
                assert not self.perturbed
            # `index_select` requires `indices` to be a 1-D tensor
            return torch.index_select(x, dim=self.axis, index=indices)

        raise ValueError('Unsupported shapes in Gather: '
                         f'data {x.shape}, indices {indices.shape}, '
                         f'axis {self.axis}')

    def bound_backward(self, last_lA, last_uA, *args, **kwargs):
        assert self.batch_dim != -1

        def _expand_A_with_zeros(A, axis, idx, max_axis_size):
            # Need to recreate A with three parts: before the gathered element, gathered element, and after gathered element.
            tensors = []
            if idx < 0:
                idx = max_axis_size + idx
            if idx > 0:
                shape_pre = list(A.shape)
                shape_pre[axis] *= idx
                # Create the same shape as A, except for the dimension to be gathered.
                tensors.append(torch.zeros(
                    shape_pre, dtype=A.dtype, device=A.device))
            # The gathered element itself, in the middle.
            tensors.append(A)
            if max_axis_size - idx - 1 > 0:
                shape_next = list(A.shape)
                shape_next[axis] *= max_axis_size - idx - 1
                # Create the rest part of A.
                tensors.append(torch.zeros(
                    shape_next, dtype=A.dtype, device=A.device))
            # Concatenate all three parts together.
            return torch.cat(tensors, dim=axis)

        def _bound_oneside(A):
            if A is None:
                return None

            if isinstance(A, torch.Tensor):
                if self.indices.ndim == 0:
                    A = A.unsqueeze(self.axis + 1)
                    idx = int(self.indices)
                    return _expand_A_with_zeros(A, self.axis + 1, idx, self.input_shape[self.axis])
                else:
                    shape = list(A.shape)
                    final_A = torch.zeros(
                        *shape[:self.axis + 1], self.input_shape[self.axis],
                        *shape[self.axis + 2:], dtype=A.dtype, device=A.device)
                    idx = self.indices.view([*[1]*(self.axis+1), -1, *[1]*len(shape[self.axis + 2:])])
                    idx = idx.repeat([*A.shape[:self.axis+1], 1, *A.shape[self.axis+2:]])
                    final_A.scatter_add_(dim=self.axis+1, index=idx, src=A)
                    return final_A
            elif isinstance(A, Patches):
                if self.indices.ndim == 0:
                    idx = int(self.indices)
                    assert len(self.input_shape) == 4 and self.axis == 1, "Gather is only supported on the channel dimension for Patches mode."
                    # For gather in the channel dimension, we only need to deal with the in_c dimension (-3) in patches.
                    patches = A.patches
                    # -3 is the in_c dimension.
                    new_patches = _expand_A_with_zeros(patches, axis=-3, idx=idx, max_axis_size=self.input_shape[self.axis])
                    return A.create_similar(new_patches)
                else:
                    raise NotImplementedError
            else:
                raise ValueError(f'Unknown last_A type {type(A)}')

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def bound_forward(self, dim_in, x, indices):
        assert self.indices.ndim <= 1 and (self.indices >= 0).all()
        select_indices = self.indices.reshape(-1)
        if isinstance(x, torch.Size):
            lb = ub = torch.index_select(
                torch.tensor(x, device=self.device),
                dim=self.axis, index=select_indices)
            if self.indices.ndim == 0:
                lb = ub = lb.squeeze(self.axis)
            lw = uw = torch.zeros(
                dim_in, *lb.shape, device=self.device)
        else:
            # x.lw/x.uw have an extra leading dimension for forward-mode
            # coefficients, so the concrete gather axis is shifted by one.
            axis = self.axis + 1
            lw = torch.index_select(x.lw, dim=axis, index=select_indices)
            uw = torch.index_select(x.uw, dim=axis, index=select_indices)
            lb = torch.index_select(x.lb, dim=self.axis, index=select_indices)
            ub = torch.index_select(x.ub, dim=self.axis, index=select_indices)
            if self.indices.ndim == 0:
                lw = lw.squeeze(axis)
                uw = uw.squeeze(axis)
                lb = lb.squeeze(self.axis)
                ub = ub.squeeze(self.axis)
        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return Interval(
            self.forward(v[0].lower, v[1].lower),
            self.forward(v[0].upper, v[1].lower))

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        self.solver_vars = self.forward(v[0], v[1])

    def build_gradient_node(self, grad_upstream):
        return [(GatherGrad(self.axis, self.indices, self.input_shape), (grad_upstream,), []), None]


class GatherGrad(Module):
    def __init__(self, axis, indices, input_shape):
        super().__init__()
        self.axis = axis
        self.indices = indices
        self.input_shape = input_shape
    
    def forward(self, grad_last):
        # TODO: It's better to use scatter_add_ instead of cat.
        # This is a workaround for the fact that scatter_add_ does not support negative indices.

        # Scalar indices case (ndim == 0)
        if self.indices.ndim == 0:
            grad_unsq = grad_last.unsqueeze(self.axis)
            
            # Get the scalar index and adjust if negative.
            idx = int(self.indices)
            if idx < 0:
                idx = self.input_shape[self.axis] + idx
            
            # Build the gradient by concatenating three parts along self.axis:
            tensors = []
            # 1. Zeros block before the gathered element (if idx > 0)
            if idx > 0:
                shape_pre = list(grad_unsq.shape)
                shape_pre[self.axis] = idx  # pre-block has size idx along self.axis
                zeros_pre = torch.zeros(shape_pre, dtype=grad_last.dtype, device=grad_last.device)
                tensors.append(zeros_pre)
            
            # 2. The gathered gradient slice (already in grad_unsq)
            tensors.append(grad_unsq)
            
            # 3. Zeros block after the gathered element
            num_after = self.input_shape[self.axis] - idx - 1
            if num_after > 0:
                shape_post = list(grad_unsq.shape)
                shape_post[self.axis] = num_after
                zeros_post = torch.zeros(shape_post, dtype=grad_last.dtype, device=grad_last.device)
                tensors.append(zeros_post)
            
            # Concatenate all parts along self.axis to form the full gradient tensor.
            grad_input = torch.cat(tensors, dim=self.axis)
            return grad_input

        # 1-D indices case (ndim == 1)
        elif self.indices.ndim == 1:
            grad_slices = []
            # Iterate over each position in the original input along self.axis.
            for i in range(self.input_shape[self.axis]):
                # matching: tensor of indices (in grad_last) where the gathered index equals i.
                matching = (self.indices == i).nonzero(as_tuple=False).squeeze(-1)
                
                if matching.numel() == 0:
                    # No matching index: create a zeros slice with the same shape as one slice of grad_last.
                    slice_shape = list(grad_last.shape)
                    slice_shape[self.axis] = 1  # single slice along self.axis
                    grad_slice = torch.zeros(slice_shape, dtype=grad_last.dtype, device=grad_last.device)
                else:
                    # There are one or more matching positions.
                    # For each matching index j, extract the corresponding slice from grad_last.
                    slice_list = []
                    for j in matching.tolist():
                        # Build slicing object：select all elements, but at self.axis take index j.
                        slicer = [slice(None)] * grad_last.dim()
                        slicer[self.axis] = j
                        # Extract the slice and add back the missing dimension.
                        slice_j = grad_last[tuple(slicer)].unsqueeze(self.axis)
                        slice_list.append(slice_j)
                    # Concatenate all slices along self.axis; if there are duplicates, sum them.
                    cat_slices = torch.cat(slice_list, dim=self.axis)
                    # Sum along self.axis to accumulate contributions from duplicate indices.
                    grad_slice = cat_slices.sum(dim=self.axis, keepdim=True)
                # Append the slice corresponding to position i.
                grad_slices.append(grad_slice)
            
            # Concatenate all slices in order along self.axis to form the final gradient tensor.
            grad_input = torch.cat(grad_slices, dim=self.axis)
            return grad_input

        else:
            raise ValueError("Unsupported indices dimensions in gradient for Gather")


class BoundGatherElements(Bound):
    def __init__(self, attr, input, output_index, options):
        super().__init__(attr, input, output_index, options)
        self.axis = attr['axis']

    def forward(self, x, index):
        self.index = index
        return torch.gather(x, dim=self.axis, index=index)

    def bound_backward(self, last_lA, last_uA, x, index, **kwargs):
        assert self.batch_dim != -1

        dim = self._get_dim()

        def _bound_oneside(last_A):
            if last_A is None:
                return None
            A = torch.zeros(
                last_A.shape[0], last_A.shape[1], *x.output_shape[1:],
                dtype=last_A.dtype, device=last_A.device)
            A.scatter_(
                dim=dim + 1,
                index=self.index.unsqueeze(0).repeat(A.shape[0], *([1] * (A.ndim - 1))),
                src=last_A)
            return A

        return [(_bound_oneside(last_lA), _bound_oneside(last_uA)), (None, None)], 0, 0

    def interval_propagate(self, *v):
        assert not self.is_input_perturbed(1)
        return Interval(
            self.forward(v[0].lower, v[1].lower),
            self.forward(v[0].upper, v[1].upper))

    def bound_forward(self, dim_in, x, index):
        assert self.axis != 0
        dim = self._get_dim()
        return LinearBound(
            torch.gather(x.lw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.lb, dim=dim, index=self.index),
            torch.gather(x.uw, dim=dim + 1, index=self.index.unsqueeze(1).repeat(1, dim_in, 1)),
            torch.gather(x.ub, dim=dim, index=self.index))

    def _get_dim(self):
        dim = self.axis
        if dim < 0:
            dim = len(self.output_shape) + dim
        return dim


class BoundScatterElements(Bound):
    """Bound operator for scatter_reduce with reduction='add'/'sum'.

    Implements: y = data.scatter_add(dim, index, src)
    Semantics:  y[i] = data[i] + sum(src[j] for j where index[j]==i)

    This is affine in both data and src (index is a constant), so bound
    propagation has zero bias — similar to BoundAdd.
    """

    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axis = attr.get('axis', 0)
        self.reduction = attr.get('reduction', 'none')
        if isinstance(self.reduction, bytes):
            self.reduction = self.reduction.decode()

    def _check_reduction(self):
        if self.reduction not in ('add', 'sum'):
            raise NotImplementedError(
                f'BoundScatterElements: currently only supports reduction '
                f'in {{add, sum}}, got "{self.reduction}".')

    def _get_dim(self, ndim):
        dim = self.axis
        if dim < 0:
            dim = ndim + dim
        return dim

    @staticmethod
    def _normalize_index(index, dim, dim_size):
        index = index.to(dtype=torch.long)
        if bool((index < 0).any()):
            index = torch.where(index < 0, index + dim_size, index)
        return index

    @staticmethod
    def _expand_data(data, dim, *others):
        target = list(data.shape)
        for axis in range(data.ndim):
            if axis == dim:
                continue
            size = data.size(axis)
            for other in others:
                if other is not None and other.ndim == data.ndim:
                    size = max(size, other.size(axis))
            target[axis] = size
        return data.expand(target).clone()

    def forward(self, data, index, src):
        self._check_reduction()
        dim = self._get_dim(data.ndim)
        self.index = self._normalize_index(index, dim, data.shape[dim])
        output = self._expand_data(data, dim, self.index, src)
        return output.scatter_add_(dim, self.index, src)

    def bound_backward(self, last_lA, last_uA, data, index, src, **kwargs):
        self._check_reduction()
        assert self.batch_dim != -1
        dim = self._get_dim(len(self.output_shape))
        index_value = self._normalize_index(
            self.index, dim, data.output_shape[dim])

        def _bound_data(last_A):
            if last_A is None:
                return None
            if isinstance(last_A, Patches):
                raise NotImplementedError(
                    'BoundScatterElements does not support Patches mode.')
            return self.broadcast_backward(last_A, data)

        def _bound_src(last_A):
            if last_A is None:
                return None
            if isinstance(last_A, Patches):
                raise NotImplementedError(
                    'BoundScatterElements does not support Patches mode.')
            gather_index = index_value.unsqueeze(0).repeat(
                last_A.shape[0], *([1] * index_value.ndim))
            return torch.gather(last_A, dim=dim + 1, index=gather_index)

        lA_data, lA_src = _bound_data(last_lA), _bound_src(last_lA)
        uA_data, uA_src = _bound_data(last_uA), _bound_src(last_uA)

        return [(lA_data, uA_data), (None, None), (lA_src, uA_src)], 0, 0

    def bound_forward(self, dim_in, data, index, src):
        self._check_reduction()
        assert self.axis != 0, (
            'BoundScatterElements: axis=0 (batch dimension) is not supported.')
        dim = self._get_dim(data.lb.ndim)
        index_value = self._normalize_index(
            self.index, dim, data.lb.shape[dim])

        # Index for weight matrices: [batch, dim_in, *src_spatial]
        index_w = index_value.unsqueeze(1).expand(
            -1, dim_in, *self.index.shape[1:])

        def _scatter_add_w(w_data, w_src):
            if w_data is None and w_src is None:
                return None
            if w_data is not None:
                w = w_data.clone()
            else:
                # data has no weight; create zeros matching data output shape
                w = torch.zeros(
                    data.lb.shape[0], dim_in, *data.lb.shape[1:],
                    device=data.lb.device)
            if w_src is not None:
                w.scatter_add_(dim + 1, index_w, w_src)
            return w

        lw = _scatter_add_w(data.lw, src.lw)
        uw = _scatter_add_w(data.uw, src.uw)

        lb = self.forward(data.lb, self.index, src.lb)
        ub = self.forward(data.ub, self.index, src.ub)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        self._check_reduction()
        # v[0]=(data_lb, data_ub), v[1]=(index, index), v[2]=(src_lb, src_ub)
        assert not self.is_input_perturbed(1), (
            'BoundScatterElements: index must not be perturbed.')
        dim = self._get_dim(v[0].lower.ndim)
        index = self._normalize_index(v[1].lower, dim, v[0].lower.shape[dim])
        lower = self._expand_data(v[0].lower, dim, index, v[2].lower)
        upper = self._expand_data(v[0].upper, dim, index, v[2].upper)
        lower.scatter_add_(dim, index, v[2].lower)
        upper.scatter_add_(dim, index, v[2].upper)
        return Interval(lower, upper)


class BoundATenIndex_add(Bound):
    """Bound operator for torch.Tensor.index_add.

    Implements: y = data.index_add(dim, index, src, alpha)
    Semantics:  y[i] = data[i] + alpha * sum(src[j] for j where index[j]==i)

    The key advantage over scatter_add is that index is always 1D and is
    implicitly broadcast across all non-scatter dimensions (including batch).
    This avoids batch dimension mismatch issues.

    This is affine in both data and src (index is constant), so bound
    propagation has zero bias.

    Traced as aten::ATen with inputs: [data, dim, index, src, alpha].
    """

    def __init__(self, attr, inputs, output_index, options):
        super().__init__(attr, inputs, output_index, options)
        self.axis = None
        self.alpha = 1.0

    def forward(self, data, dim, index, src, alpha):
        self.axis = int(dim)
        if self.axis < 0:
            self.axis += data.ndim
        self.index = index
        self.alpha = float(alpha)
        # Handle batch broadcasting: index_add requires data and src to have
        # the same shape on all dims except the scatter dim. When one has a
        # larger batch than the other, expand the smaller one.
        if data.ndim == src.ndim and data.shape != src.shape:
            target_shape_data = list(data.shape)
            target_shape_src = list(src.shape)
            for i in range(data.ndim):
                if i == self.axis:
                    continue
                mx = max(data.shape[i], src.shape[i])
                target_shape_data[i] = mx
                target_shape_src[i] = mx
            data = data.expand(target_shape_data)
            src = src.expand(target_shape_src)
        return data.index_add(self.axis, index, src, alpha=self.alpha)

    def _get_dim(self):
        dim = self.axis
        if dim < 0:
            dim = len(self.output_shape) + dim
        return dim

    def bound_backward(self, last_lA, last_uA, data, dim, index, src, alpha,
                       **kwargs):
        assert self.batch_dim != -1
        ax = self._get_dim()

        def _bound_oneside(last_A):
            if last_A is None:
                return None, None
            if isinstance(last_A, Patches):
                raise NotImplementedError(
                    'BoundATenIndex_add does not support Patches mode.')
            # A_data: identity pass-through (output shape == data shape).
            A_data = last_A

            # A_src: gather from last_A at positions given by the 1D index.
            # last_A shape: [spec, batch, *output_spatial]
            # We need to gather along ax+1 (offset by the spec dim).
            # Expand the 1D index to match all other dims of last_A.
            idx = self.index
            # Build shape for index: 1 everywhere except at ax+1.
            view_shape = [1] * last_A.ndim
            view_shape[ax + 1] = len(idx)
            expand_shape = list(last_A.shape)
            expand_shape[ax + 1] = len(idx)
            idx = idx.view(view_shape).expand(expand_shape)
            A_src = torch.gather(last_A, dim=ax + 1, index=idx)
            if self.alpha != 1.0:
                A_src = A_src * self.alpha
            return A_data, A_src

        lA_data, lA_src = _bound_oneside(last_lA)
        uA_data, uA_src = _bound_oneside(last_uA)

        # Returns: [(lA, uA) for data, None for dim, None for index,
        #           (lA, uA) for src, None for alpha]
        return [
            (lA_data, uA_data), (None, None), (None, None),
            (lA_src, uA_src), (None, None)
        ], 0, 0

    def bound_forward(self, dim_in, data, dim, index, src, alpha):
        ax = self._get_dim()
        assert ax != 0, (
            'BoundATenIndex_add: axis=0 (batch dimension) is not supported.')

        # Expand 1D index for weight matrices: [batch, dim_in, *spatial].
        # Weight dim ax+1 is the scatter axis (shifted by dim_in dim).
        idx = self.index
        # For weights: shape [1, 1, ...idx_len..., 1, ...]
        w_view = [1] * (data.lb.ndim + 1)  # +1 for dim_in
        w_view[ax + 1] = len(idx)

        def _index_add_w(w_data, w_src):
            if w_data is None and w_src is None:
                return None
            if w_data is not None:
                w = w_data.clone()
            else:
                w = torch.zeros(
                    data.lb.shape[0], dim_in, *data.lb.shape[1:],
                    device=data.lb.device)
            if w_src is not None:
                w_s = w_src if self.alpha == 1.0 else w_src * self.alpha
                # Expand index to match w's shape for scatter_add along ax+1.
                expand_shape = list(w_s.shape)
                expand_shape[ax + 1] = len(idx)
                idx_expanded = idx.view(w_view).expand(expand_shape)
                w.scatter_add_(ax + 1, idx_expanded, w_s)
            return w

        lw = _index_add_w(data.lw, src.lw)
        uw = _index_add_w(data.uw, src.uw)

        lb = self.forward(data.lb, self.axis, self.index, src.lb, self.alpha)
        ub = self.forward(data.ub, self.axis, self.index, src.ub, self.alpha)

        return LinearBound(lw, lb, uw, ub)

    def interval_propagate(self, *v):
        # v: [data, dim, index, src, alpha]
        # data=(lb, ub), dim=(val, val), index=(val, val), src=(lb, ub),
        # alpha=(val, val)
        assert not self.is_input_perturbed(2), (
            'BoundATenIndex_add: index must not be perturbed.')
        return Interval(
            self.forward(v[0].lower, v[1].lower, v[2].lower, v[3].lower, v[4].lower),
            self.forward(v[0].upper, v[1].lower, v[2].lower, v[3].upper, v[4].lower),
        )
