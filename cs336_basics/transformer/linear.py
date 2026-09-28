import torch
import torch.nn as nn
import math

class Linear(nn.Module):
    def __init__(
        self, 
        in_features, 
        out_features, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.din = in_features
        self.dout = out_features
        self.w = nn.Parameter(torch.empty(self.dout, self.din, device=device, dtype=dtype))
        std = math.sqrt(2 / (self.dout + self.din))
        nn.init.trunc_normal_(self.w, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.T