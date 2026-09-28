import torch
import torch.nn as nn
from cs336_basics.transformer.softmax import softmax 

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, 
        Q: torch.Tensor, 
        K: torch.Tensor, 
        V: torch.Tensor, 
        mask: torch.Tensor | None = None
    ):
        d_k = Q.shape[-1]
        scores = Q @ K.transpose(-1, -2) / torch.sqrt(torch.tensor(d_k))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = softmax(scores, dim=-1)

        out = attn_weights @ V
        return out