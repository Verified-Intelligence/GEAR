"""One rigorous weak-duality evaluator (WP7).

For  min c^T z + c0  s.t.  G z <= h,  l <= z <= u  and ANY multiplier y >= 0 (the weak-duality value)

    D(y) = c0 - h^T y + sum_j min(r_j l_j, r_j u_j),      r = c + G^T y,

is a lower bound of the LP optimum. This module returns a floating-point number lb <= D(y), in exact arithmetic on the
float data it is given, from fp64 operations only. Optimisers (IPM, simplex, PDHG, propagation) only propose y; their
precision never touches soundness. Every bound the verifier trusts goes through `dual_lower_bound`, every infeasibility
verdict through `certify_infeasible` (a Farkas certificate: the same value of the homogeneous LP is > 0).

Float model (Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., Lemma 3.1, eq. 3.4-3.5): u = 2^-53,
gamma_k = k u / (1 - k u), every operation is correctly rounded with an extra absolute error <= eta = 2^-1074 on
underflow, and a sum of N floats in ANY order (blocked, pairwise, atomic scatter-add, fused multiply-add) has error
<= gamma_{N-1} sum |x_i|; terms that are exactly zero do not count.

Error bound (all quantities are the computed floats, y >= 0, M_j = max(|l_j|, |u_j|)):
  1. r_hat_j = fl(c_j + sum_i G_ij y_i):  |r_hat_j - r_j| <= eps_j := gamma_{k_j+1} rho_j,  rho_j = |c_j| + sum_i |G_ij| y_i,
     k_j = number of nonzero products in column j (rho_hat is summed from the rounded products and |c_j|; rho <= rho_hat / (1 - u),
     absorbed by step 4).
  2. t_hat_j = fl(min(r_hat_j l_j, r_hat_j u_j)): rounding is monotone, so it commutes with min, and
     |t_hat_j - t_j| <= M_j eps_j + u |t_hat_j| + eta   (x -> min(x l, x u) is M_j-Lipschitz).
  3. val = fl(c0 - sum_i fl(h_i y_i) + sum_j t_hat_j), N = 1 + #{y_i > 0} + #{t_hat_j != 0} terms:
         |val - D(y)| <= E := gamma_{N+2} (|c0| + sum |h_i| y_i + sum |t_hat_j|) + sum_j M_j eps_j + (m + n + 3) eta.
     The underflow term counts every product h_i y_i and r_hat_j l_j, also those that round to 0 (they err by up to
     eta/2 each and are not among the N terms); N + 2 <= m + n + 3.
  4. E and the final subtraction are themselves computed in floats: E is inflated by 1 + gamma_{2(m+n)+16} and by
     4u(|val| + E), so that fl(val - E) <= val - E_true <= D(y).
Where E > tol/10 the reduced costs are recomputed in double-double (Ogita-Rump-Oishi "Dot2", SIAM J. Sci. Comput. 26
(2005): |r_tilde - r| <= u |r| + gamma_{k+1}^2 rho), the same formula is evaluated with eps_j := 2u|r_tilde_j| +
gamma_{k_j+1}^2 rho_j, and the larger of the two rigorous bounds is returned. This removes the cancellation term
(|G|^T y)_j M_j, the only one that grows with |y|.

Infinite box sides: a column whose side is infinite contributes 0 only when r_j is exactly 0 (no nonzero product, c_j = 0);
it contributes r_j l_j (r_j u_j) only when the sign of r_j is certified (|r_hat_j| > eps_j); otherwise the bound is -inf.
Any NaN/inf in the data or in the value gives -inf. y < 0 is clamped to 0. Rows with y_i = 0 contribute exactly 0 (a row
with h_i = +inf, G_i = 0 and y_i = 0 is inactive and harmless).

Scope: the LP data (c, c0, G, h, l, u) are taken exactly as given, float64 only (other dtypes raise TypeError); building rows
and planes with outward cushions is the caller's job, as is the rounding of c and c0 when it forms them (it passes that
error as c_err, c0_err), and so is rounding a returned bound toward -inf (`round_down`) when it goes to a narrower
dtype. A returned lb is a certificate for that LP and nothing else.

Batched: G is a Rows object (one sparsity pattern, values per domain); everything is a tensor op over (B, .) so the
same code runs on the GPU. The double-double step runs on the flagged domains only, as max-column-nnz sequential
vectorised steps.
"""
import torch

U = 2.0 ** -53                  # unit roundoff of binary64
ETA = 2.0 ** -1074              # absolute error unit of an underflowing operation
BIG = 2.0 ** 400                # the Dekker splitting of the double-double step needs |a| < 2^996; stay far below


def gamma(k):
    return k * U / (1.0 - k * U)


def round_down(x, dtype):
    """x cast to dtype, rounded toward -inf: the cast alone rounds to nearest and can lift a bound above its LP."""
    y = x.to(dtype)
    return torch.where(y.double() > x, torch.nextafter(y, torch.full_like(y, -float('inf'))), y)


class Rows:
    """G (m x n) in COO form: a pattern (row, col) shared by the batch, values val of shape (B, nnz) or (1, nnz)."""

    def __init__(self, row, col, val, m, n):
        self.row, self.col, self.m, self.n = row.long(), col.long(), int(m), int(n)
        self.val = val if val.dim() == 2 else val.unsqueeze(0)
        self._slots = None

    @classmethod
    def from_dense(cls, G):
        """G: (B, m, n) tensor; the pattern is the union of the nonzeros over the batch."""
        B, m, n = G.shape
        row, col = (G != 0).any(0).nonzero(as_tuple=True)
        return cls(row, col, G[:, row, col], m, n)

    @classmethod
    def from_scipy(cls, G):
        G = G.tocoo()
        return cls(torch.as_tensor(G.row, dtype=torch.long), torch.as_tensor(G.col, dtype=torch.long),
                   torch.as_tensor(G.data, dtype=torch.float64), G.shape[0], G.shape[1])

    def slots(self):
        """Entries grouped by their rank inside their column: one group per sequential double-double step, every
        group has at most one entry per column."""
        if self._slots is None:
            order = torch.argsort(self.col, stable=True)
            cnt = torch.bincount(self.col, minlength=self.n)
            start = torch.cumsum(cnt, 0) - cnt
            rank = torch.arange(order.numel(), device=order.device) - start[self.col[order]]
            by_rank = torch.argsort(rank, stable=True)
            sizes = torch.bincount(rank, minlength=1).tolist() if order.numel() else []
            self._slots = list(torch.split(order[by_rank], sizes))
        return self._slots

    def prepare_segmented(self):
        """Opt-in O(nnz) segmented pricing for immutable first-order tables.

        Avoid atomic contention in both certificate channels. k is the
        number of nonzero stored entries per column, an upper bound on the
        active products for every multiplier. Baseline Rows are unchanged.
        Call after masking rows and before proposing any multipliers.
        """
        if not self.row.numel():
            return
        from auto_LiRPA.compiled_crown import Mat
        self._pricing = (Mat(coo=(self.row, self.col, self.val), shape=(self.m, self.n), sparse=True),
                         Mat(coo=(self.row, self.col, self.val.abs()), shape=(self.m, self.n), sparse=True))
        self._pricing_k = self.val.new_zeros(self.val.shape[0], self.n).index_add(
            1, self.col, (self.val != 0).to(self.val.dtype))


# ------------------------------------------------------------------------------------------------ error-free transforms
def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _two_prod(a, b):                              # Dekker/Veltkamp: p + e = a b exactly (no overflow, no underflow)
    p = a * b
    t = 134217729.0 * a
    ah = t - (t - a)
    al = a - ah
    t = 134217729.0 * b
    bh = t - (t - b)
    bl = b - bh
    return p, ((ah * bh - p) + ah * bl + al * bh) + al * bl


def _reduced_float(rows, c, y):
    """r = c + G^T y in fp64, rho = |c| + |G|^T y, k = number of nonzero products per column. Rows with y == 0 add 0."""
    if hasattr(rows, '_pricing'):
        mat, magnitude = rows._pricing
        return mat.left(y)+c, magnitude.left(y)+c.abs(), rows._pricing_k.expand_as(c)
    ya = y[:, rows.row]
    act = (ya != 0) & (rows.val != 0)
    p = torch.where(act, rows.val * ya, 0.0)
    z = torch.zeros_like(c)
    r = z.clone().index_add_(1, rows.col, p) + c
    rho = z.clone().index_add_(1, rows.col, p.abs()) + c.abs()           # from the rounded products: rho <= rho_hat / (1 - u)
    k = z.clone().index_add_(1, rows.col, act.to(c.dtype))
    return r, rho, k


def _reduced_dd(rows, val, c, y):
    """r = c + G^T y with TwoProduct/TwoSum accumulation (Dot2), rounded once at the end. c, y, val: (B, .)."""
    hi, lo = c.clone(), torch.zeros_like(c)
    for idx in rows.slots():
        col = rows.col[idx]
        ya = y[:, rows.row[idx]]
        a = torch.where(ya != 0, val[:, idx], 0.0)
        p, pe = _two_prod(a, ya)
        s, se = _two_sum(hi[:, col], p)
        hi[:, col] = s
        lo[:, col] = lo[:, col] + (se + pe)
    return hi + lo


# ------------------------------------------------------------------------------------------------ the bound
def _bound(c0, h, l, u, y, r, rho, k, dd, infl, c_err, c0_err):
    """(val, E, lb) for a given float reduced cost r (see the module docstring, steps 1-4). dd: r is a Dot2 result."""
    kk = torch.where(k > 0, gamma(k + 1.0), torch.zeros_like(k))       # no nonzero product: r_hat = c_j exactly
    eps = (kk * kk * rho + 2 * U * r.abs() if dd else kk * rho) + 4 * ETA * (k + 2) + c_err
    eps_sign = eps * infl
    fl, fu = torch.isfinite(l), torch.isfinite(u)
    both = fl & fu
    l0, u0 = torch.where(fl, l, 0.0), torch.where(fu, u, 0.0)
    pos, neg = (r > eps_sign) & fl, (r < -eps_sign) & fu
    zero = (rho == 0) & (k == 0) & (c_err == 0)
    ok_col = both | zero | pos | neg
    t = torch.where(both, torch.minimum(r * l0, r * u0), torch.where(pos, r * l0, torch.where(neg, r * u0, 0.0)))
    M = torch.where(both, torch.maximum(l0.abs(), u0.abs()),
                    torch.where(pos, l0.abs(), torch.where(neg, u0.abs(), 0.0)))
    hy = torch.where(y > 0, h * y, 0.0)
    val = c0 - hy.sum(1) + t.sum(1)
    N = 1.0 + (y > 0).sum(1) + (t != 0).sum(1)
    under = (h.shape[1] + l.shape[1] + 3) * ETA                         # every product, also one rounded to 0 (step 3)
    E = gamma(N + 2) * (c0.abs() + hy.abs().sum(1) + t.abs().sum(1)) + (M * eps).sum(1) + under + c0_err
    E = E * infl
    E = E + 4 * U * (val.abs() + E)
    lb = val - E
    good = ok_col.all(1) & torch.isfinite(lb)
    return val, E, torch.where(good, lb, torch.full_like(lb, -float('inf')))


def _evaluate(rows, c, c0, h, l, u, y, need, c_err, c0_err):
    """The two-stage evaluation. need(val, E) -> bool (B,): flag the domains whose reduced costs are recomputed."""
    if any(t.dtype != torch.float64 for t in (c, c0, h, l, u, y, rows.val)):
        raise TypeError('rigorous_dual: every input must be float64 (the bound is exact for the data as given)')
    y = y.clamp_min(0.0)
    B = y.shape[0]
    infl = 1.0 + gamma(2 * (rows.m + rows.n) + 16)
    valid = (torch.isfinite(c).all(1) & torch.isfinite(y).all(1) & torch.isfinite(c0)
             & ~l.isnan().any(1) & ~u.isnan().any(1))
    r, rho, k = _reduced_float(rows, c, y)
    val, E, lb = _bound(c0, h, l, u, y, r, rho, k, False, infl, c_err, c0_err)
    flag = (valid & need(val, E, lb)) if rows.val.shape[1] else torch.zeros_like(valid)   # no entry: r = c exactly
    if not getattr(rows, "_single_pass", False) and flag.any():
        S = flag.nonzero().flatten()
        vals = rows.val if len(rows.val)==1 else rows.val[S]
        vmax = vals.abs().amax(1)
        safe = (vmax < BIG) & (y[S].amax(1) < BIG) & (c[S].abs().amax(1) < BIG)
        S = S[safe]
        if S.numel():
            r2 = _reduced_dd(rows, rows.val if len(rows.val)==1 else rows.val[S], c[S], y[S])
            val2, E2, lb2 = _bound(c0[S], h[S], l[S], u[S], y[S], r2, rho[S], k[S], True, infl, c_err[S], c0_err[S])
            use = lb2 > lb[S]
            lb[S] = torch.where(use, lb2, lb[S])
            E[S] = torch.where(use, E2, E[S])
            val[S] = torch.where(use, val2, val[S])
        flag = torch.zeros_like(flag)
        flag[S] = True
    return lb, E, val, flag


def dual_lower_bound(rows, c, c0, h, l, u, y, tol=None, c_err=None, c0_err=None):
    """Rigorous lower bound lb <= D(y) of min c^T z + c0 s.t. G z <= h, l <= z <= u, for y >= 0.
    rows: Rows; c, l, u: (B, n); c0: (B,); h, y: (B, m); tol: float or (B,), the verifier's tolerance for this problem
    (default max(1e-5, 1e-4 |value|), the defaults of bab_primal_dual_abs_tol and bab_primal_dual_rel_tol); it only
    decides where the double-double step runs, never the soundness. c_err (B, n), c0_err (B,): absolute errors the
    caller made forming c and c0 (default 0); the bound then holds for every c, c0 within them.
    Returns (lb (B,), E (B,): the error term subtracted, dd (B,): the double-double step was used)."""
    def need(val, E, lb):
        t = val.abs().mul(1e-4).clamp_min(1e-5) if tol is None else torch.as_tensor(tol, dtype=val.dtype, device=val.device)
        return ~(E <= t / 10) | ~torch.isfinite(lb)

    c_err = torch.zeros_like(c) if c_err is None else c_err
    c0_err = torch.zeros_like(c0) if c0_err is None else c0_err
    lb, E, _, dd = _evaluate(rows, c, c0, h, l, u, y, need, c_err, c0_err)
    return lb, E, dd


def certify_infeasible(rows, h, l, u, y):
    """(B,) bool: True only where y >= 0 proves {G z <= h, l <= z <= u} empty, i.e. the rigorous lower bound of
    min 0 s.t. the same constraints at y is > 0 (Farkas). The double-double step runs when it could change the verdict;
    never for y = 0 (after clamping), whose exact value is 0, so it proves nothing however it is evaluated."""
    B, n = y.shape[0], rows.n
    zero_c, zero_c0 = torch.zeros(B, n, dtype=y.dtype, device=y.device), torch.zeros(B, dtype=y.dtype, device=y.device)
    ray = (y > 0).any(1)

    def need(val, E, lb):
        return (val + E > 0) & (lb <= 0) & ray

    lb, _, _, _ = _evaluate(rows, zero_c, zero_c0, h, l, u, y, need, zero_c, zero_c0)
    return lb > 0


def quadratic_lower_bound(rows, q, c, c0, h, l, u, y, c_err=None, c0_err=None):
    """Outward diagonal-QP dual bound, for finite boxes and q >= 0.

    Any stored a gives q*z^2 >= 2*q*a*z-q*a^2. Choose the clipped
    approximate quadratic minimizer and certify that supporting affine
    objective with the existing evaluator. Coefficient/constant rounding is
    enclosed explicitly. This avoids treating evaluation at an approximate
    minimizer (an upper bound) as a lower bound. No division accuracy is
    needed for soundness. Returns the rigorous bound, error and anchor.
    """
    single_pass = getattr(rows, "_single_pass", False)
    if q.dtype != torch.float64 or (not single_pass and bool((q < 0).any())):
        raise ValueError('quadratic_lower_bound requires nonnegative binary64 curvature')
    r, _, _ = _reduced_float(rows, c, y.clamp_min(0))
    a = torch.where(q > 0, -r / torch.where(q > 0, 2*q, 1.), 0.)
    a = torch.maximum(l, torch.minimum(u, a))
    t = 2*q*a
    cc = c+t
    k = q*a*a
    ksum = k.sum(1)
    offset = c0-ksum
    infl = 1+gamma(2*rows.n+32)
    ce = (gamma(4)*(c.abs()+t.abs())+16*ETA)*infl
    c0e = (gamma(rows.n+6)*(c0.abs()+k.abs().sum(1))+32*(rows.n+1)*ETA)*infl
    if c_err is not None:
        ce = (ce+c_err)*infl
    if c0_err is not None:
        c0e = (c0e+c0_err)*infl
    lb, error, _ = dual_lower_bound(rows, cc, offset, h, l, u, y,
                                     c_err=ce, c0_err=c0e)
    good = torch.isfinite(l).all(1) & torch.isfinite(u).all(1) & torch.isfinite(q).all(1)
    if single_pass:      # no host sync: a violated q >= 0 (or NaN) makes the row's bound -inf
        good = good & (q >= 0).all(1)
    return torch.where(good, lb, -float('inf')), error, a
