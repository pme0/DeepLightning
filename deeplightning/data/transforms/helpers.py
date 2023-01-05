import numpy as np


def is_false_or_none(x):
    return x is None or x is False

def is_all_false_or_none(x):
    out = True
    for i in x:
        out *= is_false_or_none(i)
    return bool(out)

def is_all_constant(x, c):
    return np.all(np.array([i == c for i in x]))
