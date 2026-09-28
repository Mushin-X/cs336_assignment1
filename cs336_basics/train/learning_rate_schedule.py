import math

class LearningRateSchedule:
    def __init__(self, it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
        self.t = it
        self.a_max = max_learning_rate
        self.a_min = min_learning_rate
        self.T_w = warmup_iters
        self.T_c = cosine_cycle_iters

    def get_lr(self):
        if self.t < self.T_w:
            return self.t / self.T_w * self.a_max
        elif self.t > self.T_c:
            return self.a_min

        p = math.cos((self.t - self.T_w) / (self.T_c - self.T_w) * math.pi)
        return self.a_min + (1 + p) / 2 * (self.a_max - self.a_min)