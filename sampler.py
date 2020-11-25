from __future__ import absolute_import
from collections import defaultdict
import math

import numpy as np
import copy
import random
import torch
from torch.utils.data.sampler import (
    Sampler, SequentialSampler, RandomSampler, SubsetRandomSampler,
    WeightedRandomSampler)


def No_index(a, b):
    assert isinstance(a, list)
    return [i for i, j in enumerate(a) if j != b]


class RandomMultipleSampler(Sampler):
    def __init__(self, data_source, num_instances=16):
        self.data_source = zip(data_source.data,data_source.targets)
        self.index_class = defaultdict(int)
        self.class_index = defaultdict(list)
        self.num_instances = num_instances

        for index, (img, label) in enumerate(self.data_source):
            self.index_class[index] = label
            self.class_index[label].append(index)

        self.classes = list(self.class_index.keys())
        self.num_samples = len(self.classes)

    def __len__(self):
        return self.num_samples * self.num_instances

    def __iter__(self):
        indices = torch.randperm(len(self.classes)).tolist()
        ret = []

        for kid in indices:
            i = random.choice(self.class_index[self.classes[kid]])

            # img, label = self.data_source[i]

            ret.append(i)

            class_i = self.index_class[i]
            index = self.class_index[class_i]


            select_indexes = No_index(index, i)
            if (not select_indexes): continue
            if len(select_indexes) >= self.num_instances:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=False)
            else:
                ind_indexes = np.random.choice(select_indexes, size=self.num_instances-1, replace=True)

            for kk in ind_indexes:
                ret.append(index[kk])


        return iter(ret)
