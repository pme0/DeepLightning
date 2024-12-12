import numpy as np
from omegaconf import ListConfig


ErrCode = "ErrCode"


def is_false_or_none(x):
    return x is None or x is False

def is_all_false_or_none(x):
    aux = True
    for i in x:
        aux *= is_false_or_none(i)
        if aux == 0:
            return False
    return True

def is_all_constant(x, c):
    return np.all(np.array([i == c for i in x]))



#=====

def pair(x):
    return x if isinstance(x, tuple) or isinstance(x, ListConfig) else (x, x)


def none_or_zero(x):
    if not (
        x is None 
        or isinstance(x, int) 
        or isinstance(x, float)
        or isinstance(x, list)
        or isinstance(x, ListConfig)
    ):
        raise ValueError(f"{ErrCode}: Input must be [None, int, float], found {x} (type {type(x)})")
    return x is None or x == 0.0


def all_none_or_zero(x):
    if (
        x is None 
        or isinstance(x, int) 
        or isinstance(x, float)
    ):
        x = [x]
    aux = True
    for i in x:
        aux *= none_or_zero(i)
        if aux == 0:
            return False
    return True