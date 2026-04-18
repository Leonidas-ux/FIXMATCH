
"""Exponential Moving Average (EMA) of model parameters."""
import torch
from copy import deepcopy


class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.ema = deepcopy(model)
        self.ema.eval()
        self.decay = decay

        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema.state_dict()
            for k in esd.keys():
                if esd[k].dtype.is_floating_point:
                    esd[k].mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
                else:
                    esd[k].copy_(msd[k])