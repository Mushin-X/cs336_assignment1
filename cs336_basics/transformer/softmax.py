import torch

def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    #max会返回(values, index)元组， values即最大值，index是最大值在该维度中的下标
    mx = x.max(dim=dim, keepdim=True).values
    x = torch.exp(x - mx)
    return x / x.sum(dim=dim, keepdim=True)