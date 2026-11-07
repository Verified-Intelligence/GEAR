"""Excluded open intervals for real quotients at a pole.

A gap (left, right) means every defined value is <= left or >= right.
Empty/reversed gaps carry no information. Endpoints are rounded inward so
rounding can only shrink the excluded set. Unsupported operators drop it.
"""
import torch


def _inward(value, direction):
    # Interval certificates are constants in the relaxation. nextafter has no
    # derivative; do not send alpha/intermediate-bound gradients through it.
    return torch.nextafter(value.detach(), torch.full_like(value, direction))


def reciprocal_gap(lower, upper):
    valid = (lower <= 0) & (upper >= 0) & (lower < upper)
    if not bool(valid.any()):
        return None
    left = torch.where(lower < 0, _inward(1 / lower, torch.inf), -torch.inf)
    right = torch.where(upper > 0, _inward(1 / upper, -torch.inf), torch.inf)
    return torch.where(valid, left, 0), torch.where(valid, right, 0)


def negate_gap(gap):
    return None if gap is None else (-gap[1], -gap[0])


def shift_gap(gap, lower, upper):
    if gap is None:
        return None
    left, right = gap
    valid = (lower == upper) & torch.isfinite(lower) & (left < right)
    # An absent ray stays absent. Finite endpoints that overflow are rounded
    # back to the largest finite value on the appropriate side.
    shifted_l = torch.where(torch.isneginf(left), left,
                            _inward(left + lower, torch.inf))
    shifted_r = torch.where(torch.isposinf(right), right,
                            _inward(right + lower, -torch.inf))
    return torch.where(valid, shifted_l, 0), torch.where(valid, shifted_r, 0)


def scale_gap(gap, lower, upper):
    if gap is None:
        return None
    left, right = gap
    positive = (lower >= 0) & (upper > 0)
    negative = (upper <= 0) & (lower < 0)
    valid = (positive | negative) & (lower <= upper) & (left < right)
    # For each possible factor the scaled rays exclude a different gap.
    # Their intersection is the gap excluded for ALL factors, also when the
    # input gap has been shifted away from zero.
    out_l = torch.where(positive, torch.maximum(left * lower, left * upper),
                        torch.maximum(right * lower, right * upper))
    out_r = torch.where(positive, torch.minimum(right * lower, right * upper),
                        torch.minimum(left * lower, left * upper))
    # Preserve infinite boundaries only when the input ray was absent.
    absent_l = torch.where(positive, torch.isneginf(left), torch.isposinf(right))
    absent_r = torch.where(positive, torch.isposinf(right), torch.isneginf(left))
    out_l = torch.where(absent_l, -torch.inf, _inward(out_l, torch.inf))
    out_r = torch.where(absent_r, torch.inf, _inward(out_r, -torch.inf))
    valid = valid & ~torch.isnan(out_l) & ~torch.isnan(out_r)
    return torch.where(valid, out_l, 0), torch.where(valid, out_r, 0)


def square_gap_lower(gap):
    left, right = gap
    radius = torch.minimum(-left, right).clamp(min=0)
    return _inward(radius * radius, -torch.inf).clamp(min=0)
