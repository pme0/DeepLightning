import numpy as np
from omegaconf import ListConfig



def pair(x):
    return x if isinstance(x, tuple) or isinstance(x, ListConfig) else (x, x)


def none_or_value(x, v):
    if not (
        x is None
        or isinstance(x, int)
        or isinstance(x, float)
    ):
        raise ValueError(
            f"Input must be [None, int, float], found {x} (type {type(x)})"
        )
    return x is None or x == v


def all_none_or_value(x, v):
    if (
        x is None 
        or isinstance(x, int) 
        or isinstance(x, float)
    ):
        x = [x]
    aux = True
    for i in x:
        aux *= none_or_value(i, v)
        if aux == 0:
            return False
    return True


def all_none_or_zero(x):
    return all_none_or_value(x, v=0)


def all_none_or_one(x):
    return all_none_or_value(x, v=1)
