import torch



class Accumulator:
    """
    """
    def __init__(self):
        self.items = []

    def update(self, item):
        self.items.append(item)

    def compute(self):
        return torch.stack(self.items).mean()

    def reset(self):
        self.items.clear()