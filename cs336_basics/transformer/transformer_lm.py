import torch
import torch.nn as nn
from cs336_basics.transformer.transformer_block import TransformerBlock
from cs336_basics.transformer.linear import Linear
from cs336_basics.transformer.embedding import Embedding
from cs336_basics.transformer.rmsnorm import RMSNorm
from cs336_basics.transformer.softmax import softmax

class TransformerLm(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float
    ):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.block = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta) for _ in range(num_layers)
        ])
        self.rmsnorm = RMSNorm(d_model)
        self.linear = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)
        for b in self.block:
            x = b(x)
        x = self.rmsnorm(x)
        x = self.linear(x)
        # x = softmax(x, dim=-1) #作业要求返回未归一化的logits
        return x
