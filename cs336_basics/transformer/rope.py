import torch
import torch.nn as nn

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 == 1:
            raise ValueError("d_k为奇数, 其必须为偶数")

        freq_index = torch.arange(0, d_k, 2) #公式中的2i
        freqs = 1.0 / (theta ** (freq_index / d_k))
        position = torch.arange(max_seq_len)
        angles = torch.outer(position, freqs)
        self.register_buffer("cos_cache", angles.cos(), persistent=False)
        self.register_buffer("sin_cache", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        #注意：输入x的序列长度可能要小于max_seq_len，因此不能直接用上述创建的cos和sin，而要进行挑选
        #token_positions就是输入x中每个序列应对应对cos和sin缓存中的下标

        now_cos = self.cos_cache[token_positions]
        now_sin = self.sin_cache[token_positions]

        #(..., seq_len, d_k / 2)
        #注意这里不是[:, 0::2]这是针对二维矩阵，对于多维应该是[..., 0::2]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        #(..., seq_len, d_k / 2)
        new_x_even = now_cos * x_even - now_sin * x_odd
        new_x_odd = now_sin * x_even + now_cos * x_odd

        #(..., seq_len, d_k / 2, 2)
        out = torch.stack([new_x_even, new_x_odd], dim=-1) #按列堆叠，使得奇偶交错

        out = out.flatten(-2) # (..., seq_len, d_k)
        return out