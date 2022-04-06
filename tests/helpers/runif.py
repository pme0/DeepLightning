import torch


def single_gpu_available(fn):
    def wrapper():
        if torch.cuda.device_count() > 0:
            return fn
        else:
            return None
    return wrapper


def multi_gpu_available(fn):
    def wrapper():
        if torch.cuda.device_count() > 1:
            return fn
        else:
            return None
    return wrapper