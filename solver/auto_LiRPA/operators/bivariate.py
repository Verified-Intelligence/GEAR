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
""" Bivariate operators"""
import torch
from torch import Tensor
from torch.nn import Module
from typing import Dict, Optional
from .base import *
from .activation_base import BoundOptimizableActivation
from .convex_concave import BoundSqrt
from .clampmult import multiply_by_A_signs
from ..utils import *


class MulHelper:
    """Handle linear relaxation for multiplication.

    This helper can be used by BoundMul, BoundMatMul,
    BoundLinear (with weight perturbation).
    """

    def __init__(self):
        pass

    @staticmethod
    def interpolated_relaxation(x_l: Tensor, x_u: Tensor,
                                y_l: Tensor, y_u: Tensor,
                                r_l: Optional[Tensor] = None,
                                r_u: Optional[Tensor] = None,
                                middle: bool = False,
                               ) -> Tuple[Tensor, Tensor, Tensor,
                                          Tensor, Tensor, Tensor]:
        """Interpolate two optimal linear relaxations for optimizable bounds."""
        if r_l is None and r_u is None:
            if middle:
                # This option is equivalent to optimized linear relaxation
                # with 0.5 as the fixed parameter.
                # It interpolates two valid linear relaxations.
                # See Appendix C in https://openreview.net/pdf?id=BJxwPJHFwS
                alpha_l = (y_l - y_u) * 0.5 + y_u
                beta_l = (x_l - x_u) * 0.5 + x_u
                gamma_l = (y_u * x_u - y_l * x_l) * 0.5 - y_u * x_u
                alpha_u = (y_u - y_l) * 0.5 + y_l
                beta_u = (x_l - x_u) * 0.5 + x_u
                gamma_u = (y_l * x_u - y_u * x_l) * 0.5 - y_l * x_u
            else:
                alpha_l, beta_l, gamma_l = y_l, x_l, -y_l * x_l
                alpha_u, beta_u, gamma_u = y_u, x_l, -y_u * x_l
            return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u
        else:
            assert isinstance(r_l, Tensor) and isinstance(r_u, Tensor)
            # TODO (for zhouxing/qirui): this function may benefit from JIT,
            # because it has many element-wise operation which can be fused.
            # Need to benchmark and see performance.
            alpha_l = (y_l - y_u) * r_l + y_u
            beta_l = (x_l - x_u) * r_l + x_u
            gamma_l = (y_u * x_u - y_l * x_l) * r_l - y_u * x_u
            alpha_u = (y_u - y_l) * r_u + y_l
            beta_u = (x_l - x_u) * r_u + x_u
            gamma_u = (y_l * x_u - y_u * x_l) * r_u - y_l * x_u
            return alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u

    @staticmethod
    def get_relaxation(x_l: Tensor, x_u: Tensor, y_l: Tensor, y_u: Tensor,
                       opt_stage: Optional[str],
                       alphas: Optional[Dict[str, Tensor]],
                       start_name: Optional[str],
                       middle: bool = False,
                       adaptive_r: Optional[Tuple[Tensor, Tensor]] = None,
                      ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        if opt_stage in ['opt', 'reuse'] and alphas is not None and start_name in alphas:
            assert x_l.ndim == y_l.ndim
            ns = start_name
            alphas[ns].data[:] = alphas[ns].data[:].clamp(min=0, max=1)
            return MulHelper.interpolated_relaxation(
                x_l, x_u, y_l, y_u, alphas[ns][:2], alphas[ns][2:4])
        elif adaptive_r is not None:
            r_l, r_u = adaptive_r
            ret = MulHelper.interpolated_relaxation(
                x_l, x_u, y_l, y_u, r_l, r_u)
            if opt_stage in ['opt', 'reuse']:
                return tuple(r.unsqueeze(0).unsqueeze(0).expand(2, 1, *r.shape) for r in ret)
            return ret
        else:
            ret = MulHelper.interpolated_relaxation(
                x_l, x_u, y_l, y_u, middle=middle)
            if opt_stage in ['opt', 'reuse']:
                # Expand to (2, 1, *shape) for compatibility with optimized bound backward
                return tuple(r.unsqueeze(0).unsqueeze(0).expand(2, 1, *r.shape) for r in ret)
            return ret

    @staticmethod
    def get_forward_relaxation(x_l, x_u, y_l, y_u, opt_stage, alpha, start_name):
        # Broadcast
        # FIXME perhaps use a more efficient way
        x_l = x_l + torch.zeros_like(y_l)
        x_u = x_u + torch.zeros_like(y_u)
        y_l = y_l + torch.zeros_like(x_l)
        y_u = y_u + torch.zeros_like(x_u)
        return MulHelper.get_relaxation(x_l, x_u, y_l, y_u, opt_stage, alpha, start_name)

    @staticmethod
    def _get_gap(x, y, alpha, beta):
        return x * y - alpha * x - beta * y


class BoundMul(BoundOptimizableActivation):
    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.mul_helper = MulHelper()
        if options is None:
            options = {}
        # middle=True averages the two McCormick envelopes, halving the worst-case
        # pointwise gap versus picking a single vertex envelope. It is the safer
        # default for pure CROWN (no alpha). adaptive=True overrides middle and
        # picks between the two McCormick planes per-element based on the forward
        # operating point. coeff_aware=True (highest precedence among the three)
        # picks per-element by maximizing min_box(P_l) for the lower plane and
        # minimizing max_box(P_u) for the upper plane, independent of the upstream
        # last_A. All choices are sound McCormick envelopes, so any per-element
        # pick is sound.
        mul_opts = options.get('mul', {})
        self.coeff_aware = mul_opts.get('coeff_aware', True)
        self.middle = mul_opts.get('middle', True)
        self.adaptive = mul_opts.get('adaptive', False)
        if self.coeff_aware:
            # coeff_aware wins over both middle and adaptive
            self.middle = False
            self.adaptive = False

    def forward(self, x, y):
        self.x_shape = x.shape
        self.y_shape = y.shape
        return x * y

    def get_relaxation_opt(self, x_l, x_u, y_l, y_u):
        if self.output_dtype is None:
            raise RuntimeError(
                f'Output dtype is unavailable for node {self.name}')
        # Multiplication promotes its floating-point operands during forward.
        # Apply the same promotion locally before combining bounds in one
        # relaxation; the bounds stored on the input nodes remain unchanged.
        x_l = x_l.to(dtype=self.output_dtype)
        x_u = x_u.to(dtype=self.output_dtype)
        y_l = y_l.to(dtype=self.output_dtype)
        y_u = y_u.to(dtype=self.output_dtype)
        adaptive_r = None
        if self.opt_stage not in ['opt', 'reuse']:
            if self.coeff_aware:
                adaptive_r = self._compute_coeff_aware_r(x_l, x_u, y_l, y_u)
            elif self.adaptive:
                adaptive_r = self._compute_adaptive_r(x_l, x_u, y_l, y_u)
        return self.mul_helper.get_relaxation(
            x_l, x_u, y_l, y_u, self.opt_stage, getattr(self, 'alpha', None),
            getattr(self, '_start', None), middle=self.middle,
            adaptive_r=adaptive_r)

    def _compute_adaptive_r(self, x_l, x_u, y_l, y_u):
        """Pick McCormick lower/upper plane per element based on forward operating point.

        Both candidate planes are valid envelopes of x*y on the box; selecting
        one of them element-wise is still a valid relaxation.

        Lower planes:
          P1 (r_l=1): z ≥ y_l·x + x_l·y − x_l·y_l, pointwise gap (x−x_l)(y−y_l).
          P2 (r_l=0): z ≥ y_u·x + x_u·y − x_u·y_u, pointwise gap (x_u−x)(y_u−y).
        Upper planes:
          P1 (r_u=1): z ≤ y_u·x + x_l·y − y_u·x_l, pointwise gap (x−x_l)(y_u−y).
          P2 (r_u=0): z ≤ y_l·x + x_u·y − y_l·x_u, pointwise gap (x_u−x)(y−y_l).

        Returns None (caller falls back to middle envelope) when either forward
        value is unavailable — at the box center the criterion is degenerate,
        and middle strictly dominates a binary vertex pick in max-gap.
        """
        x_star = getattr(self.inputs[0], 'forward_value', None)
        y_star = getattr(self.inputs[1], 'forward_value', None)
        if x_star is None or y_star is None:
            return None
        x_star = x_star + torch.zeros_like(x_l)
        y_star = y_star + torch.zeros_like(y_l)
        # Clamp operating point into [l, u] so gap computations are non-negative.
        x_star = torch.min(torch.max(x_star, x_l), x_u)
        y_star = torch.min(torch.max(y_star, y_l), y_u)
        dxl, dxu = x_star - x_l, x_u - x_star
        dyl, dyu = y_star - y_l, y_u - y_star
        r_l = (dxl * dyl <= dxu * dyu).to(x_l.dtype)
        r_u = (dxl * dyu <= dxu * dyl).to(x_l.dtype)
        return r_l, r_u

    @staticmethod
    def _affine_min_box(a, b, c, x_l, x_u, y_l, y_u):
        """min over the box of P(x,y) = a*x + b*y + c. Each linear term hits its
        min at x_l if a>=0 else x_u (and analogously for b)."""
        x_min = torch.where(a >= 0, x_l, x_u)
        y_min = torch.where(b >= 0, y_l, y_u)
        return a * x_min + b * y_min + c

    @staticmethod
    def _affine_max_box(a, b, c, x_l, x_u, y_l, y_u):
        """max over the box of P(x,y) = a*x + b*y + c."""
        x_max = torch.where(a >= 0, x_u, x_l)
        y_max = torch.where(b >= 0, y_u, y_l)
        return a * x_max + b * y_max + c

    def _compute_coeff_aware_r(self, x_l, x_u, y_l, y_u):
        """Pick McCormick lower/upper plane per element by box-aware CROWN-side score.

        Lower candidates (sound: P_l <= x*y on the box):
          P1_l: a=y_l, b=x_l, c=-x_l*y_l   (touches at corner (x_l, y_l))
          P2_l: a=y_u, b=x_u, c=-x_u*y_u   (touches at corner (x_u, y_u))
        Upper candidates (sound: P_u >= x*y on the box):
          P1_u: a=y_u, b=x_l, c=-y_u*x_l   (touches at (x_l, y_u))
          P2_u: a=y_l, b=x_u, c=-y_l*x_u   (touches at (x_u, y_l))

        Through one CROWN backward step the local lower-bound contribution is
          last_lA_pos . P_l + last_lA_neg . P_u
        followed by min/max over the upstream box. To maximize that local
        contribution independently of last_A magnitude, pick the lower plane
        with the larger min over the box, and the upper plane with the smaller
        max over the box.
        """
        # Broadcast bounds to a common shape so the elementwise selection works
        # regardless of broadcast dimensions in x and y.
        x_l_b = x_l + torch.zeros_like(y_l)
        x_u_b = x_u + torch.zeros_like(y_u)
        y_l_b = y_l + torch.zeros_like(x_l)
        y_u_b = y_u + torch.zeros_like(x_u)

        # Lower planes
        a1l, b1l, c1l = y_l_b, x_l_b, -x_l_b * y_l_b
        a2l, b2l, c2l = y_u_b, x_u_b, -x_u_b * y_u_b
        min_p1l = BoundMul._affine_min_box(a1l, b1l, c1l, x_l_b, x_u_b, y_l_b, y_u_b)
        min_p2l = BoundMul._affine_min_box(a2l, b2l, c2l, x_l_b, x_u_b, y_l_b, y_u_b)
        # r_l == 1 selects P1_l (consistent with interpolated_relaxation convention
        # where r_l interpolates from y_u toward y_l as r_l goes from 0 to 1).
        r_l = (min_p1l >= min_p2l).to(x_l_b.dtype)

        # Upper planes
        a1u, b1u, c1u = y_u_b, x_l_b, -y_u_b * x_l_b
        a2u, b2u, c2u = y_l_b, x_u_b, -y_l_b * x_u_b
        max_p1u = BoundMul._affine_max_box(a1u, b1u, c1u, x_l_b, x_u_b, y_l_b, y_u_b)
        max_p2u = BoundMul._affine_max_box(a2u, b2u, c2u, x_l_b, x_u_b, y_l_b, y_u_b)
        # r_u == 1 selects P1_u (interpolated_relaxation goes from y_l toward y_u
        # as r_u goes from 0 to 1).
        r_u = (max_p1u <= max_p2u).to(x_l_b.dtype)

        return r_l, r_u

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        x_l = self.inputs[0].lower
        y_l = self.inputs[1].lower
        if self.output_dtype is None:
            raise RuntimeError(
                f'Output dtype is unavailable for node {self.name}')
        assert x_l.ndim == y_l.ndim
        shape = [max(x_l.shape[i], y_l.shape[i]) for i in range(x_l.ndim)]
        alpha = torch.ones(
            4, size_spec, *shape, dtype=self.output_dtype, device=x_l.device)
        return alpha

    def _is_softmax(self):
        """This multiplication comes from softmax.

        It is the division converted to BoundMul + BoundReciprocal.
        """
        return (
            self.from_complex_node == 'BoundSoftmax'
            and type(self.inputs[0]).__name__ == 'BoundExp'
            and type(self.inputs[1]).__name__ == 'BoundReciprocal'
            and type(self.inputs[1].inputs[0]).__name__ == 'BoundReduceSum'
            and type(self.inputs[1].inputs[0].inputs[0]).__name__ == 'BoundExp')

    def bound_relax(self, x, y, init=False, dim_opt=None):
        if init:
            pass
        if not self._is_softmax():
            self.require_finite_relaxation_bounds(x.lower, x.upper, operator='mul')
            self.require_finite_relaxation_bounds(
                y.lower, y.upper, operator='mul', input_index=1)
        (alpha_l, beta_l, gamma_l,
         alpha_u, beta_u, gamma_u) = self.get_relaxation_opt(
            x.lower, x.upper, y.lower, y.upper)

        # Check NaN which can happen in softmax if Exp's bounds are too loose
        if self._is_softmax():
            assert alpha_l.shape[:-1] == beta_l.shape[:-1]
            assert alpha_l.shape[-1] == 1 or alpha_l.shape[-1] == beta_l.shape[-1]
            assert beta_l.shape == gamma_l.shape
            mask = (alpha_l.isnan().expand(beta_l.shape)
                    | alpha_l.isinf().expand(beta_l.shape)
                    | beta_l.isnan() | beta_l.isinf()
                    | gamma_l.isnan() | gamma_l.isinf())
            if mask.any():
                alpha_l = alpha_l.clone()
                alpha_l[mask.any(dim=-1)] = 0
                beta_l = beta_l.clone()
                beta_l[mask] = 0
                gamma_l = gamma_l.clone()
                gamma_l[mask] = 0

            assert alpha_u.shape[:-1] == beta_u.shape[:-1]
            assert alpha_u.shape[-1] == 1 or alpha_u.shape[-1] == beta_u.shape[-1]
            assert beta_u.shape == gamma_u.shape
            mask = (alpha_u.isnan().expand(beta_u.shape)
                    | alpha_u.isinf().expand(beta_u.shape)
                    | beta_u.isnan() | beta_u.isinf()
                    | gamma_u.isnan() | gamma_u.isinf())
            if mask.any():
                alpha_u = alpha_u.clone()
                alpha_u[mask.any(dim=-1)] = 0
                beta_u = beta_u.clone()
                beta_u[mask] = 0
                gamma_u = gamma_u.clone()
                gamma_u[mask] = 1.

        self.lw = [alpha_l, beta_l]
        self.lb = gamma_l
        self.uw = [alpha_u, beta_u]
        self.ub = gamma_u

    @staticmethod
    def _multiply_by_const(x, const):
        if isinstance(x, torch.Tensor):
            return x * const
        elif isinstance(x, Patches):
            # Multiplies patches by a const. Assuming const is a tensor, and it must be in nchw format.
            assert isinstance(const, torch.Tensor) and const.ndim == 4
            if (const.size(0) == 1 or const.size(0) == x.patches.size(1)) and const.size(1) == x.patches.size(-3) and const.size(2) == const.size(3) == 1:
                # The case that we can do channel-wise broadcasting multiplication
                # Shape of const: (batch, in_c, 1, 1)
                # Shape of patches when unstable_idx is None: (spec, batch, in_c, patch_h, patch_w)
                # Shape of patches when unstable_idx is not None: (out_c, batch, out_h, out_w, in_c, patch_h, patch_w)
                const_reshaped = const
            else:
                assert x.unstable_idx is None and (x.padding == 0 or x.padding == [0,0,0,0]) and x.stride == 1 and x.patches.size(-1) == x.patches.size(-2) == 1
                # The assumed dimension is (out_c, N, out_h, out_w, in_c, 1, 1) with padding =1 and stride = 0.
                # In this special case we can directly multiply.
                # After reshape it is (1, N, H, W, C, 1, 1)
                const_reshaped = const.permute(0, 2, 3, 1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            return x.create_similar(x.patches * const_reshaped)
        else:
            raise ValueError(f'Unsupported x type {type(x)}')

    def bound_backward_constant(self, last_lA, last_uA, x, y, op=None,
                                reduce_bias=True, **kwargs):
        assert reduce_bias
        op = BoundMul._multiply_by_const if op is None else op

        def _constant_forward_value(node):
            value = getattr(node, 'forward_value', None)
            if value is not None:
                return value
            value = getattr(node, 'value', None)
            if value is not None:
                return value.to(node.device) if hasattr(value, 'to') else value
            raise AttributeError(
                f'{node.__class__.__name__} has neither forward_value nor value')

        # Handle the case of multiplication by a constant.
        factor = None
        if x.perturbed:
            factor = _constant_forward_value(y)
        if y.perturbed:
            factor = _constant_forward_value(x)
        # No need to compute A matrix if it is Constant.
        lAx = (None if not x.perturbed or last_lA is None
               else self.broadcast_backward(op(last_lA, factor), x))
        uAx = (None if not x.perturbed or last_uA is None
               else self.broadcast_backward(op(last_uA, factor), x))
        lAy = (None if not y.perturbed or last_lA is None
               else self.broadcast_backward(op(last_lA, factor), y))
        uAy = (None if not y.perturbed or last_uA is None
               else self.broadcast_backward(op(last_uA, factor), y))
        return [(lAx, uAx), (lAy, uAy)], 0., 0.

    def bound_backward(self, last_lA, last_uA, x, y, start_node=None, **kwargs):
        if start_node is not None:
            self._start = start_node.name
        if self.is_linear_op:
            ret = self.bound_backward_constant(last_lA, last_uA, x, y, **kwargs)
        else:
            ret = self.bound_backward_both_perturbed(
                last_lA, last_uA, x, y, **kwargs)
        return ret

    def bound_backward_both_perturbed(self, last_lA, last_uA, x, y,
                                      reduce_bias=True, **kwargs):
        # See BoundActivation.relaxation_cache_key: the McCormick planes depend
        # on the two input boxes only while no alpha is optimised.
        key = self.relaxation_cache_key(x, y)
        if not self.relaxation_cache_lookup(key):
            self.bound_relax(x, y)
            self.relaxation_cache_store(key)

        def _bound_oneside(last_A, alpha_pos, beta_pos, gamma_pos,
                           alpha_neg, beta_neg, gamma_neg, opt=False):
            if last_A is None:
                return None, None, 0

            if type(last_A) == Patches:
                assert reduce_bias
                assert last_A.identity == 0
                # last_A shape: [out_c, batch_size, out_h, out_w, in_c, H, W].
                # Here out_c is the spec dimension.
                # for patches mode, we need to unfold the alpha_pos/neg and beta_pos/neg
                alpha_pos = maybe_unfold_patches(alpha_pos, last_A)
                alpha_neg = maybe_unfold_patches(alpha_neg, last_A)
                beta_pos = maybe_unfold_patches(beta_pos, last_A)
                beta_neg = maybe_unfold_patches(beta_neg, last_A)
                gamma_pos = maybe_unfold_patches(gamma_pos, last_A)
                gamma_neg = maybe_unfold_patches(gamma_neg, last_A)
                A_x, bias = multiply_by_A_signs(
                    last_A, alpha_pos, alpha_neg, gamma_pos, gamma_neg,
                    patches_padding_pos_masked=self.get_bound_opt(
                        'clampmult_patches_padding_pos_masked', False))
                A_y, _ = multiply_by_A_signs(
                    last_A, beta_pos, beta_neg, None, None,
                    patches_padding_pos_masked=self.get_bound_opt(
                        'clampmult_patches_padding_pos_masked', False))
            elif type(last_A) == Tensor:
                last_A_pos, last_A_neg = last_A.clamp(min=0), last_A.clamp(max=0)
                A_x, _ = multiply_by_A_signs(last_A, alpha_pos, alpha_neg, None, None)
                A_y, _ = multiply_by_A_signs(last_A, beta_pos, beta_neg, None, None)
                A_x = self.broadcast_backward(A_x, x)
                A_y = self.broadcast_backward(A_y, y)
                if reduce_bias:
                    if opt:
                        bias = (torch.einsum('sb...,sb...->sb',
                                             last_A_pos, gamma_pos)
                                + torch.einsum('sb...,sb...->sb',
                                               last_A_neg, gamma_neg))
                    else:
                        bias = (self.get_bias(last_A_pos, gamma_pos.squeeze(0)) +
                            self.get_bias(last_A_neg, gamma_neg.squeeze(0)))
                else:
                    assert not opt
                    bias = last_A_pos * gamma_pos + last_A_neg * gamma_neg
                    assert len(x.output_shape) == bias.ndim - 1
                    assert len(y.output_shape) == bias.ndim - 1
                    bias_x = bias_y = bias
                    for i in range(2, bias.ndim):
                        if bias_x.shape[i] != x.output_shape[i - 1]:
                            assert x.output_shape[i - 1] == 1
                            bias_x = bias_x.sum(i, keepdim=True)
                    for i in range(2, bias.ndim):
                        if bias_y.shape[i] != y.output_shape[i - 1]:
                            assert y.output_shape[i - 1] == 1
                            bias_y = bias_y.sum(i, keepdim=True)
                    bias = (bias_x, bias_y)
            else:
                raise NotImplementedError(last_A)
            return A_x, A_y, bias

        alpha_l, beta_l, gamma_l = self.lw[0], self.lw[1], self.lb
        alpha_u, beta_u, gamma_u = self.uw[0], self.uw[1], self.ub

        if self.opt_stage in ['opt', 'reuse']:
            lA_x, lA_y, lbias = _bound_oneside(
                last_lA, alpha_l[0], beta_l[0], gamma_l[0],
                alpha_u[0], beta_u[0], gamma_u[0], opt=True)
            uA_x, uA_y, ubias = _bound_oneside(
                last_uA, alpha_u[1], beta_u[1], gamma_u[1],
                alpha_l[1], beta_l[1], gamma_l[1], opt=True)
        else:
            alpha_l, alpha_u = alpha_l.unsqueeze(0), alpha_u.unsqueeze(0)
            beta_l, beta_u = beta_l.unsqueeze(0), beta_u.unsqueeze(0)
            gamma_l, gamma_u = gamma_l.unsqueeze(0), gamma_u.unsqueeze(0)
            lA_x, lA_y, lbias = _bound_oneside(
                last_lA, alpha_l, beta_l, gamma_l, alpha_u, beta_u, gamma_u)
            uA_x, uA_y, ubias = _bound_oneside(
                last_uA, alpha_u, beta_u, gamma_u, alpha_l, beta_l, gamma_l)

        return [(lA_x, uA_x), (lA_y, uA_y)], lbias, ubias

    def bound_forward(self, dim_in, x, y):
        if self.is_linear_op:
            if not self.inputs[0].perturbed:
                return self.bound_forward_constant(x, y, self.inputs[0].batch_dim != -1)
            elif not self.inputs[1].perturbed:
                return self.bound_forward_constant(y, x, self.inputs[1].batch_dim != -1)
            else:
                assert False, "When is_linear_op is True, at least one input should be constant."
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='mul')
        self.require_finite_relaxation_bounds(y.lower, y.upper, operator='mul')
        return self.bound_forward_both_perturbed(dim_in, x, y)
    
    def bound_forward_constant(self, x, y, batched_constant):
        # x is constant
        const = x.lb
        const_pos, const_neg = const.clamp(min=0), const.clamp(max=0)
        lb = const_pos * y.lb + const_neg * y.ub
        ub = const_pos * y.ub + const_neg * y.lb
        if batched_constant:
            # If x is batched, its first dimension will be the batch dimension
            # We need to unsqueeze an extra dimension to align the batch dimension
            # x and y both have shape (B, a_1, a_2, ..., a_n)
            # lw/uw has shape (B, dim_in, a_1, a_2, ..., a_n)
            const_pos = const_pos.unsqueeze(1)
            const_neg = const_neg.unsqueeze(1)
        lw = const_pos * y.lw + const_neg * y.uw
        uw = const_pos * y.uw + const_neg * y.lw
        return LinearBound(lw, lb, uw, ub)

    def bound_forward_both_perturbed(self, dim_in, x, y):
        return LinearBound(None, None, None, None)

    @staticmethod
    def interval_propagate_constant(x, y, op=lambda x, const: x * const):
        # x is constant
        const = x.lower
        inp_lb = y.lower
        inp_ub = y.upper
        # Select the endpoint per sign instead of adding a masked-out product:
        # the masked product is 0 * inf = NaN at an infinite endpoint, which
        # used to widen the whole interval to (-inf, inf). A zero constant
        # times any real value is exactly 0.
        zero = torch.zeros((), dtype=inp_lb.dtype, device=inp_lb.device)
        lb = torch.where(const > 0, op(inp_lb, const),
                         torch.where(const < 0, op(inp_ub, const), zero))
        ub = torch.where(const > 0, op(inp_ub, const),
                         torch.where(const < 0, op(inp_lb, const), zero))
        return Interval(lb, ub)

    def interval_propagate(self, x, y):
        from ..range_gap import scale_gap
        gap = (scale_gap(x.range_gap, y.lower, y.upper) if x.range_gap is not None
               else scale_gap(y.range_gap, x.lower, x.upper))
        if self.is_linear_op:
            if not self.inputs[0].perturbed:
                result = self.interval_propagate_constant(x, y)
            elif not self.inputs[1].perturbed:
                result = self.interval_propagate_constant(y, x)
            else:
                assert False, "When is_linear_op is True, at least one input should be constant."
            return Interval(result.lower, result.upper, gap)
        else:
            result = self.interval_propagate_both_perturbed(x, y)
            lower, upper = result.lower, result.upper
            if self._is_softmax():
                # Endpoint multiplication may contain 0 * inf. Use the known
                # softmax range instead of letting NaNs discard the IBP bound.
                lower = torch.where(torch.isnan(lower), 0., lower)
                upper = torch.where(torch.isnan(upper), 1., upper)
                lower = lower.clamp(min=0)
                upper = upper.clamp(max=1)
            return Interval(lower, upper, gap)

    @staticmethod
    def interval_propagate_both_perturbed(*v):
        x, y = v[0], v[1]
        if x is y:
            # A shortcut for x * x.
            h_L, h_U = v[0].lower, v[0].upper
            r0 = h_L * h_L
            r1 = h_U * h_U
            # When h_L < 0, h_U > 0, lower bound is 0.
            # When h_L < 0, h_U < 0, lower bound is h_U * h_U.
            # When h_L > 0, h_U > 0, lower bound is h_L * h_L.
            l = F.relu(h_L) - F.relu(-h_U)
            upper = torch.fmax(r0, r1)
            upper = torch.where(
                upper.isnan(), torch.full_like(upper, float('inf')), upper)
            return Interval(l * l, upper)

        r0 = x.lower * y.lower
        r1 = x.lower * y.upper
        r2 = x.upper * y.lower
        r3 = x.upper * y.upper
        # 0 * (±inf) → NaN. torch.min / torch.max propagate NaN, which would
        # taint both sides of the interval. Ignoring the NaN corner can be
        # unsound (for example [0, 1] * [inf, inf] would get lower=+inf), so
        # fail closed to an unbounded interval for affected entries.
        corner_nan = r0.isnan() | r1.isnan() | r2.isnan() | r3.isnan()
        lower = torch.fmin(torch.fmin(r0, r1), torch.fmin(r2, r3))
        upper = torch.fmax(torch.fmax(r0, r1), torch.fmax(r2, r3))
        if deferring() or corner_nan.any():
            lower = torch.where(
                corner_nan, torch.full_like(lower, float('-inf')), lower)
            upper = torch.where(
                corner_nan, torch.full_like(upper, float('inf')), upper)

        return Interval(lower, upper)

    def build_solver(self, *v, model, C=None, model_type="mip", solver_pkg="gurobi"):
        # both operands constant -> exact forward
        if isinstance(v[0], Tensor) and isinstance(v[1], Tensor):
            self.solver_vars = self.forward(*v)
            return
        # exactly one operand constant -> affine (original behavior)
        if isinstance(v[0], Tensor) or isinstance(v[1], Tensor):
            if isinstance(v[1], Tensor):
                gvar_array = np.array(v[0]) * v[1].cpu().numpy()
            else:
                gvar_array = np.array(v[1]) * v[0].cpu().numpy()
            self.solver_vars = gvar_array.tolist()
            return
        # both operands are variables -> bilinear McCormick relaxation
        import gurobipy as grb
        import arguments
        xv, yv = list(v[0]), list(v[1])
        xl = self.inputs[0].lower.flatten().detach().cpu().numpy()
        xu = self.inputs[0].upper.flatten().detach().cpu().numpy()
        yl = self.inputs[1].lower.flatten().detach().cpu().numpy()
        yu = self.inputs[1].upper.flatten().detach().cpu().numpy()
        out_lb = self.lower.flatten().detach().cpu().numpy()
        out_ub = self.upper.flatten().detach().cpu().numpy()
        multi = arguments.Config["solver"]["multi_facet"]["enabled"]
        n = max(len(xv), len(yv))
        zs = []
        for i in range(n):
            xi, yi = xv[i % len(xv)], yv[i % len(yv)]
            z = model.addVar(lb=float(out_lb[i]), ub=float(out_ub[i]),
                             vtype=grb.GRB.CONTINUOUS, name=f"mul{self.name}_{i}")
            model.addConstr(z >= yl[i] * xi + xl[i] * yi - xl[i] * yl[i])   # lower L1
            model.addConstr(z <= yu[i] * xi + xl[i] * yi - yu[i] * xl[i])   # upper U1
            if multi:
                model.addConstr(z >= yu[i] * xi + xu[i] * yi - xu[i] * yu[i])  # lower L2
                model.addConstr(z <= yl[i] * xi + xu[i] * yi - yl[i] * xu[i])  # upper U2
            zs.append(z)
        model.update()
        self.solver_vars = zs

    def update_input_perturbation_state(self):
        self.splittable = all(node.perturbed for node in self.inputs)
        self.is_linear_op = False
        for inp in self.inputs:
            if not inp.perturbed:
                # If any of the two inputs are constant, we do not need input bounds.
                self.is_linear_op = True
        if self.is_linear_op:
            # One input is constant; no bounds required.
            self.requires_input_bounds = []
        else:
            # Both inputs are perturbed. Need relaxation.
            self.requires_input_bounds = [0, 1]
        
    def build_gradient_node(self, grad_upstream):
        grad_node_0 = MulGrad(self.inputs[0].output_shape if self.inputs[0].batch_dim != -1 else
                              torch.Size((1,) + self.inputs[0].output_shape))
        grad_node_1 = MulGrad(self.inputs[1].output_shape if self.inputs[1].batch_dim != -1 else
                              torch.Size((1,) + self.inputs[1].output_shape))
        return [(grad_node_0, (grad_upstream, self.inputs[1].forward_value), [self.inputs[1]]),
                (grad_node_1, (grad_upstream, self.inputs[0].forward_value), [self.inputs[0]])]

    def faces(self, x, y, exact_ties=None):
        """The four McCormick faces, MulHelper.interpolated_relaxation at r in {1, 0} on each side, and the default
        relaxation as weights on them (r on the face r=1, 1 - r on r=0). exact_ties: coeff_aware's choice by
        _coeff_aware_r_exact (None: the node's coeff_aware_exact). Returns (lower, upper, weights, weights)."""
        x_l, x_u, y_l, y_u = (t.to(dtype=self.output_dtype) for t in (x.lower, x.upper, y.lower, y.upper))
        zero = torch.zeros_like(x_l) + torch.zeros_like(y_l)
        x_l, x_u, y_l, y_u = x_l + zero, x_u + zero, y_l + zero, y_u + zero
        one = torch.ones_like(zero)
        f1 = MulHelper.interpolated_relaxation(x_l, x_u, y_l, y_u, one, one)
        f0 = MulHelper.interpolated_relaxation(x_l, x_u, y_l, y_u, zero, zero)
        r = None
        if self.coeff_aware:
            r = self._compute_coeff_aware_r(x_l, x_u, y_l, y_u)
        elif self.adaptive:
            r = self._compute_adaptive_r(x_l, x_u, y_l, y_u)
        if r is None:
            r = (one * (0.5 if self.middle else 1.0),) * 2
        r_l, r_u = r
        lower = [((f1[0], f1[1]), f1[2]), ((f0[0], f0[1]), f0[2])]
        upper = [((f1[3], f1[4]), f1[5]), ((f0[3], f0[4]), f0[5])]
        return lower, upper, torch.stack([r_l, 1 - r_l]), torch.stack([r_u, 1 - r_u])



class MulGrad(Module):
    def __init__(self, input_shape):
        super().__init__()
        # We need the input shape to handle broadcasting
        self.input_shape = input_shape

    def forward(self, grad_last, y):
        # z = x * y
        # ∂z/∂x = y
        if y.ndim > 0:
            # If y is not a constant scalar, its second dimension is for spec
            y = y.unsqueeze(1)
        return reduce_broadcast_dims(grad_last * y, self.input_shape)


class BoundDiv(Bound):

    def forward(self, x, y):
        # FIXME (05/11/2022): ad-hoc implementation for layer normalization
        if isinstance(self.inputs[1], BoundSqrt) and hasattr(self.inputs[0], "inputs") and len(self.inputs[0].inputs) > 0:
            input = self.inputs[0].inputs[0]
            x = input.forward_value
            n = input.forward_value.shape[-1]

            dev = x * (1. - 1. / n) - (x.sum(dim=-1, keepdim=True) - x) / n
            dev_sqr = dev ** 2
            s = (dev_sqr.sum(dim=-1, keepdim=True) - dev_sqr) / dev_sqr.clamp(min=epsilon)
            sqrt = torch.sqrt(1. / n * (s + 1))
            return torch.sign(dev) * (1. / sqrt)

        return x / y
