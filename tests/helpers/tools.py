import torch


def compare_model_params(modelA, modelB):
    passed = True
    for key in modelA.state_dict():
        if (modelA.state_dict()[key] != modelB.state_dict()[key]).sum().item() > 0:
            passed = False
    return passed

    
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