import pytest

from oupic.base_calculator import BaseCalculator, ContrastInfo, ValueWithCoef
from oupic.bm_calculator import BMCalculator
from oupic.ou_calculator import OUCalculator

TOLERANCE = 1e-5


def approx_equal(lhs: ValueWithCoef, rhs: ValueWithCoef) -> bool:
    for attr in ValueWithCoef.__slots__:
        if not lhs.__getattribute__(attr) == pytest.approx(
            lhs.__getattribute__(attr), TOLERANCE
        ):
            return False
    return True


def test_bm_calculator(left_info: ContrastInfo, right_info: ContrastInfo) -> None:
    calculator = BMCalculator()

    assert calculator.calc_contrast_variance(left_info, right_info) == pytest.approx(
        60.0, TOLERANCE
    )

    assert calculator.calc_addition_dist_to_parent(
        left_info, right_info
    ) == pytest.approx(8.3333333, TOLERANCE)

    assert approx_equal(
        calculator.calc_contrast(left_info, right_info, standardized=True),
        ValueWithCoef(value=0.1290994, left_par=0.1290994, right_par=-0.1290994),
    )
    assert approx_equal(
        calculator.calc_contrast(left_info, right_info, standardized=False),
        ValueWithCoef(value=1.0, left_par=1.0, right_par=1.0),
    )
    assert approx_equal(
        calculator.calc_nd_value(left_info, right_info),
        ValueWithCoef(value=4.8333333, left_par=0.8333333, right_par=0.1666666),
    )


def test_ou_calculator(left_info: ContrastInfo, right_info: ContrastInfo) -> None:
    calculator = OUCalculator(0.1)

    assert calculator.calc_contrast_variance(left_info, right_info) == pytest.approx(
        0.067684, TOLERANCE
    )

    assert calculator.calc_addition_dist_to_parent(
        left_info, right_info
    ) == pytest.approx(9.998746, TOLERANCE)

    assert approx_equal(
        calculator.calc_contrast(left_info, right_info, standardized=True),
        ValueWithCoef(value=-5.526667, left_par=0.025899, right_par=-1.414041),
    )
    assert approx_equal(
        calculator.calc_contrast(left_info, right_info, standardized=False),
        ValueWithCoef(value=-1.437828, left_par=0.006737, right_par=-0.367879),
    )
    assert approx_equal(
        calculator.calc_nd_value(left_info, right_info),
        ValueWithCoef(value=5.062516, left_par=0.999835, right_par=0.015834),
    )


def test_base_calculator(left_info: ContrastInfo, right_info: ContrastInfo) -> None:
    BaseCalculator.__abstractmethods__ = frozenset()

    calculator = BaseCalculator()

    with pytest.raises(NotImplementedError):
        calculator.calc_addition_dist_to_parent(left_info, right_info)
    with pytest.raises(NotImplementedError):
        calculator.calc_contrast(left_info, right_info, standardized=True)
    with pytest.raises(NotImplementedError):
        calculator.calc_contrast_variance(left_info, right_info)
    with pytest.raises(NotImplementedError):
        calculator.calc_nd_value(left_info, right_info)
