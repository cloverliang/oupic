import pytest

from oupic.base_calculator import ContrastInfo


@pytest.fixture
def left_info():
    return ContrastInfo(
        is_contrast=True,
        contrast_standardized=2.0,
        dist_to_parent=10.0,
        nd_value=5.0,
    )


@pytest.fixture
def right_info():
    return ContrastInfo(
        is_contrast=True,
        contrast_standardized=4.0,
        dist_to_parent=50.0,
        nd_value=4.0,
    )
