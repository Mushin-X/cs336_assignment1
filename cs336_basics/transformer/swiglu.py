import torch
import torch.nn as nn
import math

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        std = math.sqrt(2 / (self.d_model + self.d_ff))
        self.w1 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty(self.d_ff, self.d_model, device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty(self.d_model, self.d_ff, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w1, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w2, mean=0, std=std, a=-3*std, b=3*std)
        nn.init.trunc_normal_(self.w3, mean=0, std=std, a=-3*std, b=3*std)

    def silu(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.silu(x @ self.w1.T) * (x @ self.w3.T)) @ self.w2.T