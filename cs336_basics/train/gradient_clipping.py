import torch

class GradientClipping:
    def __init__(self, params, max_l2_norm):
        self.params = params
        self.max_l2_norm = max_l2_norm
        self.eps = 1e-6

    def cal(self):
        s = 0
        for p in self.params:
            if p.grad is not None:
                s += p.grad.pow(2).sum()

        s = torch.sqrt(s)
        if s <= self.max_l2_norm:
            return 

        for p in self.params:
            if p.grad is not None:
                p.grad.mul_(self.max_l2_norm / (s + self.eps))
        