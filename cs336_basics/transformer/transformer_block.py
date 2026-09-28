import torch
import torch.nn as nn
from cs336_basics.transformer.multihead_self_attention_with_rope import MultiheadSelfAttentionWithRoPE
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.swiglu import SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
    ) -> torch.Tensor:
        super().__init__()

        self.multihead_att_with_rope = MultiheadSelfAttentionWithRoPE(d_model, num_heads, max_seq_len, theta)
        self.rmsnorm1 = RMSNorm(d_model)
        self.swiglu = SwiGLU(d_model, d_ff)
        self.rmsnorm2 = RMSNorm(d_model)

    def forward(self, in_features: torch.Tensor):
        x = in_features
        x = x + self.multihead_att_with_rope(self.rmsnorm1(x))
        out = x + self.swiglu(self.rmsnorm2(x))
        return out