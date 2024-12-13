import pytest
from torchvision import transforms as T
from typing import Any

from deeplightning.transforms.ops import \
(
    RandomAffine,
    CenterCrop,
    ColorJitter,
)


def _test_error(fn, expected, *args):
    with pytest.raises(expected) as e_info:
        fn(*args)


def _test_ok(fn, expected, *args):
    t = fn(*args)
    if expected.value is None:
        assert t is None
    else:
        assert isinstance(t, expected.value)


class Ok:
    def __init__(self, value: Any = None):
        self.value = value

    def __hash__(self):
        # All instances of Ok hash to the same value
        return hash('Ok')

    def __eq__(self, other):
        # All instances of Ok are considered equal
        return isinstance(other, Ok)
    

hashmap = {
    ValueError: _test_error,
    TypeError: _test_error,
    Ok(): _test_ok,
}


@pytest.mark.parametrize(
    ("degrees", "translate", "scale", "shear", "EXPECTED"), 
    [
        (None, None, None, None, Ok(None)),
        ([5,7], None, None, None, Ok(T.RandomAffine)),
        ([5,7], [0,1], None, None, Ok(T.RandomAffine)),
        ([5,7], None, [5,7], None, Ok(T.RandomAffine)),
        ([5,7], None, None, [5,7], Ok(T.RandomAffine)),
        ([5,7], [0,1], [5,7], [5,7], Ok(T.RandomAffine)),
        (None, [5,7], None, None, ValueError),
        (5, None, None, None, TypeError),
        ([5,7], 5, None, None, TypeError),
        ([5,7], None, 5, None, TypeError),
        ([5,7], None, None, 5, TypeError),
    ]
)
def test_affine_transform(
    degrees, translate, scale, shear, EXPECTED
):
    args = [degrees, translate, scale, shear]
    hashmap[EXPECTED](
        RandomAffine, EXPECTED, *args
    )


@pytest.mark.parametrize(
    ("size", "EXPECTED"), 
    [
        (None, Ok(None)),
        ([5,7], Ok(T.CenterCrop)),
        (5, TypeError),
    ]
)
def test_centercrop_transform(
    size, EXPECTED
):
    args = [size]
    hashmap[EXPECTED](
        CenterCrop, EXPECTED, *args
    )


@pytest.mark.parametrize(
    ("brightness", "contrast", "saturation", "hue", "EXPECTED"), 
    [
        (None, None, None, None, Ok(None)),
        (0., None, None, None, Ok(None)),
        (None, 0., None, None, Ok(None)),
        (None, None, 0., None, Ok(None)),
        (None, None, None, 0., Ok(None)),
        (0., 0., 0., 0., Ok(None)),
        (0., None, None, None, Ok(None)),
        (None, 0., None, None, Ok(None)),
        (None, None, 0., None, Ok(None)),
        (None, None, None, 0., Ok(None)),
        (0., 0., 0., 0., Ok(None)),
        (1., 0., 0., 0., TypeError), # brightness must be 2d sequence
        (0., 1., 0., 0., TypeError), # contrast must be 2d sequence
        (0., 0., 1., 0., TypeError), # saturation must be 2d sequence
        (0., 0., 0., 1., TypeError), # hue must be 2d sequence
        ([-1.,1.], [-1.,1.], [-1.,1.], [-1.,1.], ValueError), # brightness must be (0, inf)
        ([0.,1.], [-1.,1.], [-1.,1.], [-1.,1.], ValueError), # contrast must be (0, inf)
        ([0.,1.], [0.,1.], [-1.,1.], [-1.,1.], ValueError), # saturation must be (0, inf)
        ([0.,1.], [0.,1.], [0.,1.], [-1.,1.], ValueError), # hue must be (-0.5, 0.5)
        ([0.,1.], [0.,1.], [0.,1.], [-.5,.5], Ok(T.ColorJitter)),
    ]
)
def test_colorjitter_transform(
    brightness, contrast, saturation, hue, EXPECTED
):
    args = [brightness, contrast, saturation, hue]
    hashmap[EXPECTED](
        ColorJitter, EXPECTED, *args
    )