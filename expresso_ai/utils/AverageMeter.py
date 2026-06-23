
import numpy as np

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.accumulator = None

    def update(self, val, n=1, accumulate=False):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        if accumulate:
            if self.accumulator is None:
                self.accumulator = val
            else:
                self.accumulator = np.concatenate((self.accumulator, val), axis=0)
