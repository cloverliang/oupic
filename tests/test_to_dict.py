from oupic.base_calculator import ValueWithCoef


def test_contrast_info_to_dict(left_info):
    assert left_info.to_dict() == {
        "is_contrast": True,
        "contrast_standardized": 2.0,
        "dist_to_parent": 10.0,
        "nd_value": 5.0,
    }


def test_value_with_coef_to_dict():
    value_with_coef = ValueWithCoef(1.0, 1.0, -1.0)
    assert value_with_coef.to_dict() == {
        "value": 1.0,
        "left_par": 1.0,
        "right_par": -1.0,
    }
