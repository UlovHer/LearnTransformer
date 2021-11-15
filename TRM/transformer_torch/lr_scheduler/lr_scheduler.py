import torch

from torch import nn, optim
from torch.optim.lr_scheduler import _LRScheduler, StepLR


class WarmUpLR(_LRScheduler):
    """warmup_training learning rate scheduler
    Args:
        optimizer: optimzier(e.g. SGD)
        total_iters: totoal_iters of warmup phase
    """

    def __init__(self, optimizer, total_iters, last_epoch=-1):
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """we will use the first m batches, and set the learning
        rate to base_lr * m / total_iters
        """
        return [base_lr * self.last_epoch / (self.total_iters + 1e-8) for base_lr in self.base_lrs]


if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optimizer = optim.SGD(model, lr=0.001, momentum=0.9, weight_decay=5e-4)

    iter_per_epoch = 10
    warmup_epoch = 5
    scheduler_stplr = StepLR(optimizer, step_size=10, gamma=0.1)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * warmup_epoch)
    optimizer.zero_grad()
    optimizer.step()
    for epoch in range(1, 1000):
        warmup_scheduler.step(epoch)
        warm_lr = warmup_scheduler.get_lr()
        print("epoch:%d,warm_lr:%s" % (epoch, warm_lr))
