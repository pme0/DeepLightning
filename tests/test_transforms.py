import sys
sys.path.insert(0, "..")
import PIL
import omegaconf
import torch
import pytest

from deeplightning.data.transforms.transforms import __TransformsDict__


CFG = omegaconf.OmegaConf.create(
    {
        "normalize_1d": {
            "mean": [0.5], 
            "std": [1.0]
        },
        "normalize_3d": {
            "mean": [0.5, 0.5, 0.5], 
            "std": [1.0, 1.0, 1.0]
        },
        "flip": 1.0,
    })


#====================================
#       Normalize
#------------------------------------
@pytest.mark.parametrize("kwargs",
    (
        pytest.param(dict(cfg=CFG["normalize_1d"], size=(1,1,2,2))),
        pytest.param(dict(cfg=CFG["normalize_3d"], size=(1,3,2,2))),
    ))
def test_normalize(kwargs):

    transform = __TransformsDict__["normalize"](kwargs["cfg"])
    image_tensor = torch.ones(kwargs["size"])
    image_tensor = transform(image_tensor)

    assert torch.mean(image_tensor) == 0.5
    assert torch.std(image_tensor) == 0.0


#====================================
#       Flip
#------------------------------------
@pytest.mark.parametrize("kwargs",
    (
        pytest.param(dict(f="hflip", p=CFG["flip"])),
        pytest.param(dict(f="vflip", p=CFG["flip"])),
    ))
def test_flip(kwargs):

    image_tensor = torch.ones((1,3,2,2))
    for k in range(3):
        image_tensor[0,k,0,0] = 0

    transform = __TransformsDict__[kwargs["f"]](kwargs["p"])
    image_tensor_trf = transform(image_tensor)

    if kwargs["f"] == "hflip":
        assert image_tensor_trf[0,0,0,1].item() == 0
        assert image_tensor_trf[0,1,0,1].item() == 0
        assert image_tensor_trf[0,2,0,1].item() == 0
    elif kwargs["f"] == "vflip":
        assert image_tensor_trf[0,0,1,0].item() == 0
        assert image_tensor_trf[0,1,1,0].item() == 0
        assert image_tensor_trf[0,2,1,0].item() == 0
    else:
        raise ValueError


#====================================
#       ?
#------------------------------------




