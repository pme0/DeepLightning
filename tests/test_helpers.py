import pytest

from deeplightning.transforms.helpers import (
    none_or_zero, 
    all_none_or_zero,
)


@pytest.mark.parametrize(
    "value", 
    [
        [], 
        {},
    ]
)
def test_none_or_zero_error(value):
    with pytest.raises(ValueError) as e_info:
        none_or_zero(value)


@pytest.mark.parametrize(
    ("value", "EXPECTED"), 
    [
        (None, True),
        (0, True),
        (0.0, True),
        (1, False),
    ]
)
def test_none_or_zero_value(value, EXPECTED):
    assert none_or_zero(value) is EXPECTED


@pytest.mark.parametrize(
    ("value", "EXPECTED"), 
    [
        (None, True), 
        (0, True),
        (0.0, True),
        (1, False),
        ([None, None], True),
        ([0, 0], True),
        ([0.0, 0.0], True),
        ([0, 0.0], True),
        ([0, 1], False),
        ([0.0, 1.0], False),
        ([0, 1.0], False),
    ]
)
def test_all_none_or_zero_value(value, EXPECTED):
    assert all_none_or_zero(value) is EXPECTED
