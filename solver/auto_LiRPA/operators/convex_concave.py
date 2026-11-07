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
"""Nonlinear functions that are either convex or convave within the entire domain."""
import torch
from torch.nn import Module
from .base import *
from .activation_base import BoundActivation, BoundOptimizableActivation


class _DomainRestricted:
    """Read-only view of a bound node with its lower bound tightened to the
    operator's real domain. Attribute access falls through to the original."""

    __slots__ = ('_node', 'lower')

    def __init__(self, node, lower):
        object.__setattr__(self, '_node', node)
        object.__setattr__(self, 'lower', lower)

    def __getattr__(self, item):
        return getattr(object.__getattribute__(self, '_node'), item)


class BoundLog(BoundActivation):

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.range_l = 1e-6

    def _check_real_domain(self, lower):
        if (lower <= 0).any():
            # Intersect with the closure of the real domain x > 0 only: a positive
            # floor (range_l) would cut off feasible arguments in (0, floor) and
            # could invert the interval (hs025: argument 3.3e-16, upper 4.4e-16).
            restricted = self.restrict_input_domain(
                lower, min_lower=0.0, operator='log')
            if restricted is not None:
                return restricted
            raise OperatorDomainError(
                'log',
                'input bounds require a strictly positive lower bound',
            )

    def forward(self, x):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            return torch.logsumexp(self.inputs[0].inputs[0].inputs[0].forward_value, dim=-1)
        return torch.log(x)

    def bound_relax(self, x, init=False):
        restricted = self._check_real_domain(x.lower)
        if restricted is not None:
            x = _DomainRestricted(x, restricted)
        # log is unbounded below at 0+: no affine relaxation exists on an interval
        # whose lower end is 0 (after the restriction above: was <= 0).
        self.require_finite_relaxation_bounds(
            torch.where(x.lower > 0, x.lower, torch.full_like(x.lower, -torch.inf)),
            x.upper, operator='log')
        if init:
            self.init_linear_relaxation(x)
        rl, ru = self.forward(x.lower), self.forward(x.upper)
        du = x.upper - x.lower
        ku = (ru - rl) / du.clamp(min=epsilon)
        self.add_linear_relaxation(mask=None, type='lower', k=ku, x0=x.lower, y0=rl)
        # Parallel-tangent upper bound: make tangent slope equal to secant slope,
        # minimising max pointwise gap. For log, slope 1/m = (log(u)-log(l))/(u-l),
        # so m = (u-l)/log(u/l) — the logarithmic mean of l and u.
        dlog = (ru - rl).clamp(min=epsilon)
        m_parallel = du / dlog
        m_mid = (x.lower + x.upper) / 2
        # Fall back to midpoint when u≈l (both numerator and denominator tiny).
        m = torch.where(du > 1e-6, m_parallel, m_mid)
        m = torch.min(torch.max(m, x.lower), x.upper)
        # Clamping a positive tangent point changes its derivative and can put
        # the upper plane below log(x) on tiny intervals.
        k = torch.reciprocal(m)
        rm = self.forward(m)
        self.add_linear_relaxation(mask=None, type='upper', k=k, x0=m, y0=rm)

    def interval_propagate(self, *v):
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            par = self.inputs[0].inputs[0].inputs[0]
            lower = torch.logsumexp(par.lower, dim=-1)
            upper = torch.logsumexp(par.upper, dim=-1)
            return Interval(lower, upper)
        restricted = self._check_real_domain(v[0].lower)
        if restricted is not None:
            v = (Interval(restricted, v[0].upper),) + tuple(v[1:])
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        A, lbias, ubias = super().bound_backward(last_lA, last_uA, x)
        # NOTE adhoc implementation for loss fusion
        if self.loss_fusion:
            assert A[0][0] is None
            exp_module = self.inputs[0].inputs[0]
            ubias = ubias + self.get_bias(A[0][1], exp_module.max_input.squeeze(-1))
        return A, lbias, ubias


class BoundSqrt(BoundOptimizableActivation):
    face_tangent_alpha = True

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.use_prior_constraint = True
        self.has_constraint = True
        self.range_l = 1e-6

    def _check_real_domain(self, lower):
        if (lower < 0).any():
            restricted = self.restrict_input_domain(
                lower, min_lower=0.0, operator='sqrt')
            if restricted is not None:
                return restricted
            raise OperatorDomainError(
                'sqrt',
                'input bounds require a non-negative lower bound',
            )

    def forward(self, x):
        return torch.sqrt(x)

    def bound_relax(self, x, init=False, dim_opt=None):
        restricted = self._check_real_domain(x.lower)
        if restricted is not None:
            x = _DomainRestricted(x, restricted)
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='sqrt')
        if init:
            self.init_linear_relaxation(x, dim_opt)

        # Check if alpha exists for this start_node (it may not exist when
        # final_layer_alpha_only=True in init_alpha).
        if self.opt_stage in ['opt', 'reuse'] and self._start in self.alpha:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], x.lower), x.upper)
            mid = self.alpha[self._start]
        else:
            # Parallel-tangent: tangent slope = secant slope ⇒ 1/(2√m) = 1/(√l+√u),
            # so m = ((√l + √u) / 2)². Minimises the max pointwise tangent-to-f gap.
            sqrt_l_tmp = torch.sqrt(x.lower.clamp(min=0))
            sqrt_u_tmp = torch.sqrt(x.upper.clamp(min=0))
            mid = ((sqrt_l_tmp + sqrt_u_tmp) / 2) ** 2
            mid = torch.min(torch.max(mid, x.lower), x.upper)
        # The square-root derivative is infinite at zero, but a zero-width
        # interval at zero is represented exactly by the constant line y = 0.
        fixed_zero = (x.lower == 0) & (x.upper == 0)
        safe_mid = torch.where(fixed_zero, torch.ones_like(mid), mid)
        k = 0.5 / self.forward(safe_mid)
        k = torch.where(fixed_zero, torch.zeros_like(k), k)
        mid_value = self.forward(safe_mid)
        mid_value = torch.where(
            fixed_zero, torch.zeros_like(mid_value), mid_value)
        self.add_linear_relaxation(
            mask=None, type='upper', k=k, x0=mid, y0=mid_value)

        sqrt_l = self.forward(x.lower)
        sqrt_u = self.forward(x.upper)
        k = (sqrt_u - sqrt_l) / (x.upper - x.lower).clamp(min=1e-8)
        self.add_linear_relaxation(mask=None, type='lower', k=k, x0=x.lower)

    def interval_propagate(self, *v):
        restricted = self._check_real_domain(v[0].lower)
        if restricted is not None:
            v = (Interval(restricted, v[0].upper),) + tuple(v[1:])
        return super().interval_propagate(*v)

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        if self.use_prior_constraint and self.check_constraint_available(x):
            if hasattr(x, 'cstr_interval'):
                del x.cstr_interval
                del x.cstr_lower
                del x.cstr_upper

            constraint = self._ibp_constraint(
                x, delete_bounds_after_use=True)
            x_l, x_u = constraint.lower, constraint.upper
            x_u = torch.max(x_u, x_l + 1e-8)
        return super().bound_backward(last_lA, last_uA, x, **kwargs)

    def clamp_interim_bounds(self):
        self.cstr_lower = self.lower.clamp(min=0)
        self.cstr_upper = self.upper.clamp(min=0)
        self.cstr_interval = Interval(
            self.cstr_lower, self.cstr_upper)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(
            2, size_spec, *l.shape, dtype=l.dtype, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha


class BoundReciprocal(BoundOptimizableActivation):
    face_tangent_alpha = True


    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        self.splittable = True
        self.range_l = 1e-6

    def forward(self, x):
        return torch.reciprocal(x)

    def bound_backward(self, last_lA, last_uA, x, start_node=None,
                       start_shape=None, reduce_bias=True, **kwargs):
        # On [0, u] the reciprocal is unbounded above but bounded below (a tangent is
        # a valid lower plane on all of (0, u]); on [l, 0] the other way round. Such
        # an interval is admissible exactly when no backward coefficient selects the
        # unbounded side: a lower bound uses the upper relaxation where last_lA < 0,
        # an upper bound where last_uA > 0. The relaxation is then built on the
        # half interval [u/2, u] ([l, l/2]), whose bounded side stays valid on the
        # whole interval; the other side is multiplied by zero coefficients.
        touch_pos = (x.lower == 0) & (x.upper > 0)
        touch_neg = (x.upper == 0) & (x.lower < 0)
        if self.get_bound_opt('undefined_relaxation_rows', False):
            # Report every element whose interval contains zero; backward_general
            # discards the rows that select an unbounded side and reruns this on
            # the replacement interval. The half interval [u/2, u] ([l, l/2])
            # keeps the bounded side valid on (0, u] ([l, 0)); an interval that
            # crosses zero (or is {0}) has no bounded side at all.
            contains_zero = (x.lower <= 0) & (x.upper >= 0)
            if bool(contains_zero.any()):
                both = contains_zero & ~touch_pos & ~touch_neg
                dummy_l, dummy_u = UndefinedRelaxation.DUMMY
                sub_lower = torch.where(
                    touch_pos, x.upper / 2,
                    torch.where(touch_neg, x.lower, torch.full_like(x.lower, dummy_l)))
                sub_upper = torch.where(
                    touch_pos, x.upper,
                    torch.where(touch_neg, x.lower / 2, torch.full_like(x.upper, dummy_u)))
                error = OperatorDomainError(
                    'reciprocal',
                    'linear bound propagation input interval contains zero')
                error.undefined = UndefinedRelaxation(
                    no_lower=touch_neg | both, no_upper=touch_pos | both,
                    lower=sub_lower, upper=sub_upper)
                raise error
        if not (bool(touch_pos.any()) or bool(touch_neg.any())):
            return super().bound_backward(
                last_lA, last_uA, x, start_node=start_node, start_shape=start_shape,
                reduce_bias=reduce_bias, **kwargs)

        def selects(A, positive):
            if A is None:
                return torch.zeros_like(touch_pos)
            if not isinstance(A, Tensor):
                return torch.ones_like(touch_pos)
            return ((A > 0) if positive else (A < 0)).any(0).reshape(touch_pos.shape)

        needs_upper = selects(last_lA, False) | selects(last_uA, True)
        needs_lower = selects(last_lA, True) | selects(last_uA, False)
        if bool((touch_pos & needs_upper).any()) or bool((touch_neg & needs_lower).any()):
            raise OperatorDomainError(
                'reciprocal',
                'linear bound propagation input interval contains zero',
            )
        saved_lower, saved_upper = x.lower, x.upper
        try:
            x.lower = torch.where(touch_pos, saved_upper / 2, saved_lower)
            x.upper = torch.where(touch_neg, saved_lower / 2, saved_upper)
            return super().bound_backward(
                last_lA, last_uA, x, start_node=start_node, start_shape=start_shape,
                reduce_bias=reduce_bias, **kwargs)
        finally:
            x.lower, x.upper = saved_lower, saved_upper

    def interval_propagate(self, *v):
        h_L = v[0].lower
        h_U = v[0].upper
        lower = torch.reciprocal(h_U)
        upper = torch.reciprocal(h_L)

        # reciprocal(h_U), reciprocal(h_L) already handle intervals strictly
        # on either side of zero and [0, h_U], since reciprocal(+0) is +inf.
        # Correct the remaining singular cases: crossing zero needs the full
        # real hull, approaching zero from below needs a -inf lower bound, and
        # the zero-only interval is conservatively treated as fully unbounded.
        crosses_zero = (h_L < 0) & (h_U > 0)
        negative_to_zero = (h_L < 0) & (h_U == 0)
        zero_only = (h_L == 0) & (h_U == 0)

        lower = torch.where(
            crosses_zero | negative_to_zero | zero_only,
            torch.full_like(lower, -torch.inf),
            lower)
        upper = torch.where(
            crosses_zero | zero_only | ((h_L == 0) & (h_U > 0)),
            torch.full_like(upper, torch.inf),
            upper)
        from ..range_gap import reciprocal_gap
        # The compiled consumer uses the hull only and rejects pole intervals
        # through its deferred validity mask; constructing range-gap metadata
        # would introduce a host reduction into that device graph.
        return Interval(lower, upper, None if deferring() else reciprocal_gap(h_L, h_U))

    def bound_relax(self, x, init=False, dim_opt=None):
        invalid = (x.lower <= 0) & (x.upper >= 0)
        if not defer_check(invalid) and invalid.any():
            raise OperatorDomainError(
                'reciprocal',
                'linear bound propagation input interval contains zero',
            )
        positive = x.lower > 0
        negative = x.upper < 0

        # Use the finite endpoint to choose a tangent for one-sided unbounded
        # intervals. Their asymptotic relaxations are assigned below.
        unbounded_lower = torch.isneginf(x.lower)
        unbounded_upper = torch.isposinf(x.upper)
        finite_lower = torch.where(
            unbounded_lower, x.upper, x.lower)
        finite_upper = torch.where(
            unbounded_upper, x.lower, x.upper)
        self.require_finite_relaxation_bounds(
            finite_lower, finite_upper, operator='reciprocal')
        if init:
            self.init_linear_relaxation(x, dim_opt)

        # Reciprocal is convex on (0, +inf) and concave on (-inf, 0), so the
        # secant and tangent exchange their upper/lower roles across zero.
        secant_k = -1. / (x.lower * x.upper)
        secant_x0 = torch.where(positive, x.lower, x.upper)
        self.add_linear_relaxation(
            mask=positive, type='upper', k=secant_k, x0=secant_x0)
        self.add_linear_relaxation(
            mask=negative, type='lower', k=secant_k, x0=secant_x0)

        # Check if alpha exists for this start_node (it may not exist when
        # final_layer_alpha_only=True in init_alpha).
        if self.opt_stage in ['opt', 'reuse'] and self._start in self.alpha:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], finite_lower), finite_upper)
            mid = torch.where(
                positive,
                self.alpha[self._start].clamp(min=0.01),
                self.alpha[self._start].clamp(max=-0.01),
            )
        else:
            mid = (finite_lower + finite_upper) / 2

        tangent_k = -1. / (mid**2)
        self.add_linear_relaxation(
            mask=positive, type='lower', k=tangent_k, x0=mid)
        self.add_linear_relaxation(
            mask=negative, type='upper', k=tangent_k, x0=mid)

        if deferring():
            self.lw.masked_fill_(unbounded_upper, 0)
            self.lb.masked_fill_(unbounded_upper, 0)
            self.uw.masked_fill_(unbounded_lower, 0)
            self.ub.masked_fill_(unbounded_lower, 0)
        else:
            self.lw[..., unbounded_upper] = 0
            self.lb[..., unbounded_upper] = 0
            self.uw[..., unbounded_lower] = 0
            self.ub[..., unbounded_lower] = 0

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(
            2, size_spec, *l.shape, dtype=l.dtype, device=l.device)
        finite_lower = torch.where(torch.isneginf(l), u, l)
        finite_upper = torch.where(torch.isposinf(u), l, u)
        alpha.data[:2] = (finite_lower + finite_upper) / 2
        return alpha

    def build_gradient_node(self, grad_upstream):
        return [(ReciprocalGrad(), (grad_upstream, self.inputs[0].forward_value), [self.inputs[0]])]


class ReciprocalGrad(Module):
    def __init__(self):
        super().__init__()

    def forward(self, g, x):
        # partial derivative of 1/x is -1/x^2
        return -g / torch.square(x).unsqueeze(1)


class BoundExp(BoundOptimizableActivation):
    face_tangent_alpha = True

    def __init__(self, attr=None, inputs=None, output_index=0, options=None):
        super().__init__(attr, inputs, output_index, options)
        if options is None:
            options = {}
        self.options = options.get('exp', {})
        self.max_input = 0
        self.has_constraint = True

    def forward(self, x):
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(x, dim=-1, keepdim=True)[0].detach()
            return torch.exp(x - self.max_input)
        return torch.exp(x)

    def interval_propagate(self, *v):
        assert len(v) == 1
        # unary monotonous functions only
        h_L, h_U = v[0].lower, v[0].upper
        if self.loss_fusion and self.options != 'no-max-input':
            self.max_input = torch.max(h_U, dim=-1, keepdim=True)[0]
            h_L, h_U = h_L - self.max_input, h_U - self.max_input
        else:
            self.max_input = 0
        return Interval(torch.exp(h_L), torch.exp(h_U))

    # Note: BoundExp.bound_forward intentionally inherits BoundActivation's
    # implementation, which calls bound_relax(x, init=True). bound_relax uses
    # the parallel-tangent point m = log((exp(u)-exp(l))/(u-l)), which gives a
    # tighter (steepest-sound) lower tangent than the previous midpoint
    # override and matches the formulation used by backward CROWN. The base
    # bound_forward composes self.lw/self.lb/self.uw/self.ub (populated by
    # bound_relax) with the input's forward-linear form.
    #
    # Loss-fusion / log-softmax adjustment lives only in bound_backward (see
    # below); no forward-mode override is needed for that path.

    def bound_backward(self, last_lA, last_uA, x, **kwargs):
        # Special case when computing log_softmax (FIXME: find a better solution, this trigger condition is not reliable).
        if self.loss_fusion and last_lA is None and last_uA is not None and torch.min(
                last_uA) >= 0 and x.batch_dim != -1:
            self.require_finite_relaxation_bounds(x.lower, x.upper, operator='exp')
            # Adding an extra bias term to the input. This is equivalent to adding a constant and subtract layer before exp.
            # Note that we also need to adjust the bias term at the end.
            if self.options == 'no-detach':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0]
            elif self.options != 'no-max-input':
                self.max_input = torch.max(x.upper, dim=-1, keepdim=True)[0].detach()
            else:
                self.max_input = 0
            adjusted_lower = x.lower - self.max_input
            adjusted_upper = x.upper - self.max_input
            # relaxation for upper bound only (used in loss fusion)
            exp_l, exp_u = torch.exp(adjusted_lower), torch.exp(adjusted_upper)
            k = (exp_u - exp_l) / (adjusted_upper - adjusted_lower).clamp(min=1e-8)
            if k.requires_grad:
                k = k.clamp(min=1e-8)
            uA = last_uA * k.unsqueeze(0)
            ubias = last_uA * (-adjusted_lower * k + exp_l).unsqueeze(0)

            if ubias.ndim > 2:
                ubias = torch.sum(ubias, dim=tuple(range(2, ubias.ndim)))
            # Also adjust the missing ubias term.
            if uA.ndim > self.max_input.ndim:
                A = torch.sum(uA, dim=tuple(range(self.max_input.ndim, uA.ndim)))
            else:
                A = uA

            # These should hold true in loss fusion
            assert self.batch_dim == 0
            assert A.shape[0] == 1

            batch_size = A.shape[1]
            ubias -= (A.reshape(batch_size, -1) * self.max_input.reshape(batch_size, -1)).sum(dim=-1).unsqueeze(0)
            return [(None, uA)], 0, ubias
        else:
            As, lbias, ubias = super().bound_backward(last_lA, last_uA, x, **kwargs)
            lA, uA = As[0]
            lA, lbias = self._check_nan(
                lA, lbias, last_lA, const_bound=0.,
                invalid_bound=-torch.inf)
            uA, ubias = self._check_nan(
                uA, ubias, last_uA, const_bound=torch.inf,
                invalid_bound=torch.inf)
            return [(lA, uA)], lbias, ubias

    def _check_nan(self, A, bias, last_A, const_bound, invalid_bound):
        """Handle non-finite backward coefficients for exp.

        The original local NaN case is ``0 * inf`` from a very loose exp
        relaxation. When ``last_A`` is finite and nonnegative, replacing the
        affected row by the global exp bound is sound: lower side uses 0,
        upper side uses +inf unless the row coefficient is exactly zero.

        If upstream coefficients are already non-finite, the local relaxation
        can no longer certify a finite affine bound. In that case we keep the
        affine row zero and set the row bias to the side-specific conservative
        bound (-inf for lower, +inf for upper).
        """
        if A is None:
            return A, bias
        if not isinstance(bias, torch.Tensor):
            return A, bias
        if not bias.isnan().any() and (
                not isinstance(A, torch.Tensor) or torch.isfinite(A).all()):
            return A, bias

        if not isinstance(A, torch.Tensor):
            bias = torch.where(
                bias.isnan(),
                torch.full_like(bias, invalid_bound),
                bias)
            return A, bias

        A_ = A.reshape(-1, *A.shape[2:]).clone()
        bias_ = bias.reshape(-1).clone()
        row_count = A_.shape[0]
        if bias_.numel() != row_count:
            # Unexpected shape; fail closed for every non-finite A/bias entry.
            if (not torch.isfinite(A_).all()) or bias_.isnan().any():
                A_.zero_()
                bias_ = torch.full_like(bias_, invalid_bound)
            return A_.reshape(A.shape), bias_.reshape(bias.shape)

        row_bad_A = ~torch.isfinite(A_).reshape(row_count, -1).all(dim=1)
        row_bias_nan = bias_.isnan()

        row_bad_last = torch.zeros(row_count, dtype=torch.bool, device=A.device)
        row_last_nonnegative = torch.zeros(
            row_count, dtype=torch.bool, device=A.device)
        row_last_nonzero = torch.zeros(
            row_count, dtype=torch.bool, device=A.device)
        if isinstance(last_A, torch.Tensor):
            last_A_ = last_A.reshape(-1, *last_A.shape[2:])
            if last_A_.shape[0] == row_count:
                last_A_flat = last_A_.reshape(row_count, -1)
                row_bad_last = ~torch.isfinite(last_A_flat).all(dim=1)
                row_last_nonnegative = (last_A_flat >= 0).all(dim=1)
                row_last_nonzero = (last_A_flat != 0).any(dim=1)
            else:
                row_bad_last[:] = True
        else:
            row_bad_last[:] = True

        relaxation_has_nan = any(
            isinstance(t, torch.Tensor) and t.isnan().any()
            for t in (self.lw, self.uw, self.lb, self.ub))
        local_ok = row_bias_nan & ~row_bad_last & row_last_nonnegative
        if relaxation_has_nan:
            local_ok = torch.zeros_like(local_ok)
        invalid = row_bad_last | (row_bias_nan & ~local_ok) | (
            row_bad_A & ~local_ok)

        if local_ok.any():
            A_[local_ok] = 0
            bias_[local_ok] = torch.where(
                row_last_nonzero[local_ok],
                torch.full_like(bias_[local_ok], const_bound),
                torch.zeros_like(bias_[local_ok]))
        if invalid.any():
            A_[invalid] = 0
            bias_[invalid] = invalid_bound

        A = A_.reshape(A.shape)
        bias = bias_.reshape(bias.shape)
        return A, bias

    def bound_relax(self, x, init=False, dim_opt=None):
        self.require_finite_relaxation_bounds(x.lower, x.upper, operator='exp')
        if init:
            self.init_linear_relaxation(x, dim_opt)
        l, u = x.lower, x.upper
        # Check if alpha exists for this start_node (it may not exist when
        # final_layer_alpha_only=True in init_alpha).
        if self.opt_stage in ['opt', 'reuse'] and self._start in self.alpha:
            self.alpha[self._start].data[:2] = torch.min(torch.max(
                self.alpha[self._start].data[:2], x.lower), x.upper)
            m = self.alpha[self._start]
            # The cap is a heuristic: every point of [l, u] gives a valid tangent of the convex exp.
            # bound_opts {'exp': {'free_tangent': True}} places it where alpha says (e.g. an incumbent).
            if not (isinstance(self.options, dict) and self.options.get('free_tangent', False)):
                m = torch.min(m, x.lower + 0.99)
        else:
            # Parallel-tangent: lower tangent slope = upper secant slope ⇒
            # exp(m) = (exp(u)-exp(l))/(u-l), so m = log((exp(u)-exp(l))/(u-l)).
            # Stable form: m = u + log(1 - exp(l-u)) - log(u-l)
            #               = u + log(-expm1(l-u)) - log(u-l).
            # Falls back to midpoint when u≈l (0/0 limit is (l+u)/2).
            du = (x.upper - x.lower)
            neg_expm1_lu = -torch.expm1(x.lower - x.upper)  # (0, 1]
            m_parallel = x.upper + torch.log(neg_expm1_lu.clamp(min=epsilon)) \
                - torch.log(du.clamp(min=epsilon))
            m_mid = (x.lower + x.upper) / 2
            m = torch.where(du > 1e-6, m_parallel, m_mid)
            m = torch.min(torch.max(m, x.lower), x.upper)
        exp_l, exp_m, exp_u = torch.exp(x.lower), torch.exp(m), torch.exp(x.upper)
        # Lower bound: tangent line at m. If m is so large that exp(m)
        # overflows, fall back to the global lower bound exp(x) >= 0. This is
        # sound and avoids inf - inf in the tangent intercept.
        lower_k = torch.where(torch.isfinite(exp_m), exp_m, torch.zeros_like(exp_m))
        lower_x0 = torch.where(torch.isfinite(m), m, torch.zeros_like(m))
        lower_b = -lower_x0 * lower_k + lower_k
        lower_safe = torch.isfinite(lower_k) & torch.isfinite(lower_b)
        self.lw[:] = torch.where(lower_safe, lower_k, torch.zeros_like(lower_k))
        self.lb[:] = torch.where(lower_safe, lower_b, torch.zeros_like(lower_b))

        # Upper bound: secant line. If exp(u) overflows, there is no finite
        # floating-point secant available; use a conservative +inf upper bias
        # instead of manufacturing NaNs or clipped, unsound finite constants.
        # The secant of the true interval: clamping l at -1e9 gave a secant that dips below
        # exp(x) < 1 further left (U(-1e10) = -8.9), and clamping the width at 1e-8 under-
        # estimated the slope of narrower intervals. exp(l) * expm1(d) / d is accurate for
        # small widths d, where the difference quotient cancels.
        d = u - l
        k = torch.where(
            d < 1,
            exp_l * torch.where(d > 0, torch.expm1(d.clamp(max=1)) / d.clamp(min=1e-300),
                                torch.ones_like(d)),
            (exp_u - exp_l) / d.clamp(min=1))
        upper_b = -l * k + exp_l
        upper_safe = torch.isfinite(k) & torch.isfinite(upper_b)
        self.uw[:] = torch.where(upper_safe, k, torch.zeros_like(k))
        self.ub[:] = torch.where(
            upper_safe, upper_b, torch.full_like(upper_b, torch.inf))

    def clamp_interim_bounds(self):
        self.cstr_lower = self.lower.clamp(min=0)
        self.cstr_upper = self.upper.clamp(min=0)
        self.cstr_interval = (self.cstr_lower, self.cstr_upper)

    def _init_opt_parameters_impl(self, size_spec, **kwargs):
        """Implementation of init_opt_parameters for each start_node."""
        l, u = self.inputs[0].lower, self.inputs[0].upper
        alpha = torch.empty(
            2, size_spec, *l.shape, dtype=l.dtype, device=l.device)
        alpha.data[:2] = (l + u) / 2
        return alpha

    def build_gradient_node(self, grad_upstream):
        if self.loss_fusion:
            raise NotImplementedError('Gradient computation for exp with loss fusion is not supported.')
        return [(ExpGrad(), (grad_upstream, self.inputs[0].forward_value), [self.inputs[0]])]


class ExpGrad(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, g, preact):
        # exp'(x) = exp(x)
        return g * torch.exp(preact).unsqueeze(1)
