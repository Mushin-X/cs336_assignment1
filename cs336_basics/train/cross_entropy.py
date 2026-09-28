import torch

#输入的inputs：(batch_size, vocab_size)，targets：(batch_size)
def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    mx = inputs.max(dim=-1, keepdim=True).values #(batch_size, 1)
    target_x = inputs[torch.arange(inputs.shape[0], device=inputs.device), targets].view(-1, 1) #(batch_size, 1)
    target_x -= mx #(batch_size, 1)
    ex = torch.exp(inputs - mx) #(batch_size, 1)

    #这里没有直接调用之前写的softmax是因为这里有一个log可以进一步优化即logsoftmax
    return -(target_x - torch.log(ex.sum(dim=-1, keepdim=True))).mean() #返回一个类型为torch.float的标量

def perplexity(cross_entropy_value: torch.Tensor) -> torch.Tensor:
    return torch.exp(cross_entropy_value.mean())