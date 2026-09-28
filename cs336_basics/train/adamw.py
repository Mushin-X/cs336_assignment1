from typing import Callable, Optional
import torch

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr, betas, eps, weight_decay):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps":eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                state = self.state[p] 
                g = p.grad.data

                #state的键有：'m', 'v', 'step'
                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['step'] = 0

                #更新m, v, t
                state['step'] += 1
                m, v = state['m'], state['v'] #这两个变量为张量，虽然是引用变量，但由于赋值操作是重新绑定，因此要用字典赋值
                state['m'] = b1 * m + (1 - b1) * g
                state['v'] = b2 * v + (1 - b2) * g**2
                
                # 偏差修正，注意下面的改动不需要修改状态字典中的m和v
                m, v, t = state['m'], state['v'], state['step']
                m = m / (1 - b1**t)
                v = v / (1 - b2**t)

                #梯度更新
                p.data *= 1 - lr * weight_decay
                p.data -= lr / (torch.sqrt(v) + eps) * m