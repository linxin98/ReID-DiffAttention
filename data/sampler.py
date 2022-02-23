from collections import defaultdict
import random
import copy

import numpy as np
from torch.utils.data.sampler import Sampler


class TripletSampler(Sampler):
    def __init__(self, labels, batch_size, p, k):
        super(TripletSampler, self).__init__(None)
        self.labels = labels
        self.batch_size = batch_size
        self.p = p
        self.k = k
        # print(len(self.labels))
        # print(self.labels)
        # Create label dict.
        self.label_dict = defaultdict(list)
        for index in range(len(self.labels)):
            if self.labels[index] > 0:
                self.label_dict[self.labels[index]].append(index)
        # Estimate number of examples in an epoch.
        length = 0
        self.label_list = list(np.unique(self.labels))
        self.label_list = [label for label in self.label_list if label > 0]
        for label in self.label_list:
            num = len(self.label_dict[label])
            if num < self.k:
                num = self.k
            length += num - num % self.k
        self.length = length

    def __iter__(self):
        # Make up mini batchs.
        batch_idxs_dict = defaultdict(list)
        for label in self.label_list:
            idxs = copy.deepcopy(self.label_dict[label])
            if len(idxs) < self.k:
                idxs = np.random.choice(idxs, size=self.k, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.k:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []
        # Make up available batchs.
        avai_labels = copy.deepcopy(self.label_list)
        final_idxs = []
        while len(avai_labels) >= self.p:
            selected_labels = random.sample(avai_labels, self.p)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
