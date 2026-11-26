import math

import torch


def warmup_cosine(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0, stepped once per batch."""
    def f(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)
