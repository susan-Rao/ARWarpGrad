"""

Based on code for WarpGrad:
https://github.com/flennerhag/warpgrad.

"""
import torch
from torch.utils.data.sampler import Sampler


class RandomSampler(Sampler):
    def __init__(self, data_source, iterations, batch_size):
        self.data_source = data_source
        self.iterations = iterations
        self.batch_size = batch_size

    def __iter__(self):
        if self.data_source._train:
            idx = torch.randperm(self.iterations * self.batch_size) % len(
                self.data_source)
        else:
            idx = torch.randperm(len(self.data_source))
        return iter(idx.tolist())

    def __len__(self):  # pylint: disable=protected-access
        return self.iterations * self.batch_size if self.data_source._train \
            else len(self.data_source)