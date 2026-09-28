import numpy as np
import torch

class DataLoading:
    def __init__(self, dataset, batch_size, context_length, device):
        self.x = dataset
        self.batch_size = batch_size
        self.seq_len = context_length
        self.device = device

    def get_batch(self):
        starts = np.random.randint(0, len(self.x) - self.seq_len, size=self.batch_size)
        inputs = np.stack([self.x[i:i+self.seq_len] for i in starts])
        targets = np.stack([self.x[i+1:i+self.seq_len+1] for i in starts])
        inputs = torch.from_numpy(inputs) #(batch, seq_len)
        targets = torch.from_numpy(targets) #(batch, seq_len)
        # 注意.to()不是原地修改
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets