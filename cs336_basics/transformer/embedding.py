import torch
import torch.nn as nn

class Embedding(nn.Module):
    def __init__(
        self, 
        num_embeddings, 
        embedding_dim, 
        device=None, 
        dtype=None
    ):
        super().__init__()
        self.num_em = num_embeddings
        self.em_dim = embedding_dim
        self.w = nn.Parameter(torch.empty(self.num_em, self.em_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.w, mean=0, std=1, a = -3, b = 3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.w[token_ids]