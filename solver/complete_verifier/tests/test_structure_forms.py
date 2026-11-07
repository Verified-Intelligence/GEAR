import os
import sys
from fractions import Fraction

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from nlp_solver.structure.forms import (ZERO, BudgetExceeded, Form, FormAlgebra, constant,
                                        constant_value, exact)


def var(j):
    return Form({((j, 1),): 1})


def test_exact_uses_the_binary_value_of_floats():
    assert exact(0.1) == Fraction(0.1)          # not Fraction(1, 10)
    assert exact(3.0) == 3 and isinstance(exact(3.0), int)


def test_ring_identities():
    a = FormAlgebra(2, None)
    x = var(0)
    p = a.mul(a.add(x, constant(1)), a.add(x, constant(-1)))
    assert p.poly == {((0, 2),): 1, (): -1}
    assert a.add(p, a.neg(p)).poly == {}


def test_fractional_power_of_monomial_needs_positive_box():
    pos = FormAlgebra(1, np.array([0.5]))
    mixed = FormAlgebra(1, np.array([-1.0]))
    sq = pos.mul(var(0), var(0))
    assert pos.pow(sq, Fraction(1, 2)).poly == {((0, 1),): 1}
    root = mixed.pow(mixed.mul(var(0), var(0)), Fraction(1, 2))
    assert root.poly != {((0, 1),): 1}          # |x| is not x: stays an atom


def test_exp_of_log_power_and_log_domain():
    a = FormAlgebra(1, np.array([0.5]))
    cubed = a.exp(a.scale(a.log(var(0)), 3))
    assert cubed.poly == {((0, 3),): 1}
    assert a.log(constant(-2)).poly is None


def test_negative_power_records_a_pole():
    a = FormAlgebra(1, None)
    a.pow(var(0), -1)
    assert 0 in a.poles


def test_budget_ends_work_with_an_exception_not_a_hang():
    a = FormAlgebra(3, None, budget=10)
    s = a.lincomb([(var(j), 1) for j in range(3)])
    with pytest.raises(BudgetExceeded):
        a.mul(a.mul(s, s), a.mul(s, s))


def test_constant_value():
    assert constant_value(constant(Fraction(3, 2))) == Fraction(3, 2)
    assert constant_value(ZERO) == 0
    assert constant_value(var(0)) is None


def test_an_inexact_zero_keeps_the_inexact_flag():
    a = FormAlgebra(1, None)
    tiny = a.exp(constant(-1000))                    # rounds to 0.0 but is not 0
    assert tiny.poly == {} and tiny.inexact
    assert a.add(var(0), tiny).inexact
    assert a.add(tiny, var(0)).inexact
    assert a.lincomb([(var(0), 1), (tiny, 1)]).inexact
    assert a.scale(tiny, 0).inexact
    assert a.mul(tiny, var(0)).inexact
    assert a.pow(tiny, 2).inexact
    assert a.exp(tiny).inexact                       # exp(rounded zero) is not exactly 1


def test_rounded_constants_taint_expansion_division_and_exp():
    a = FormAlgebra(2, np.array([0.5, 0.5]))
    root2 = a.pow(constant(2), Fraction(1, 2))               # rounded
    assert root2.inexact
    with_atom = a.add(a.mul(root2, var(0)), a.pow(a.add(a.mul(var(0), var(1)), constant(1)), 2))
    assert any(v >= 2 for m in with_atom.poly for v, _ in m) and with_atom.inexact      # a kept pow atom
    assert a.expand(with_atom).inexact                       # expand rebuilt its terms exact
    assert a.div(var(0), root2).inexact
    assert a.exp(a.mul(root2, a.log(var(0)))).inexact       # x0**1.41..: a rounded exponent
    assert not a.exp(a.scale(a.log(var(0)), 3)).inexact
