import torch
import torch.nn as nn
import math
from cs336_basics.transformer.scaled_dot_product_attention import ScaledDotProductAttention

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("输入维度不能被头数整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        std = math.sqrt(2 / (2 * self.d_model))
        self.q_weight = nn.Parameter(torch.empty((self.d_model, self.d_model)))
        nn.init.trunc_normal_(self.q_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.k_weight = nn.Parameter(torch.empty((self.d_model, self.d_model)))
        nn.init.trunc_normal_(self.k_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.v_weight = nn.Parameter(torch.empty((self.d_model, self.d_model)))
        nn.init.trunc_normal_(self.v_weight, mean=0, std=std, a=-3 * std, b=3 * std)
        self.o_weight = nn.Parameter(torch.empty((self.d_model, self.d_model)))
        nn.init.trunc_normal_(self.o_weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, in_features: torch.Tensor) -> torch.Tensor:
        #参数的维度形状约定是(out_features, in_features)，因此记得转置
        q = in_features @ self.q_weight.T
        k = in_features @ self.k_weight.T
        v = in_features @ self.v_weight.T
        q = q.view(*q.shape[:-1], self.num_heads, self.d_head)
        k = k.view(*k.shape[:-1], self.num_heads, self.d_head)
        v = v.view(*v.shape[:-1], self.num_heads, self.d_head)
        #(..., num_heads, seq_len, d_head)
        q = q.transpose(-2, -3) 
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)

        seq_len = q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=in_features.device))

        scaled_dot_product_attention = ScaledDotProductAttention()
        out = scaled_dot_product_attention(q, k, v, mask) #(..., num_heads, seq_len, d_head)

        out = out.transpose(-2, -3)
        out = out.flatten(-2)
        out = out @ self.o_weight.T
        return out
