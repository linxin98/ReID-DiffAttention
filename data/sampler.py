from collections import defaultdict
import random
import copy

import numpy as np
from torch.utils.data.sampler import Sampler


class TripletSampler(Sampler):
    def __init__(self, pids, pid_index, batch_size, p, k):
        super(TripletSampler, self).__init__(None)
        self.pids = pids
        self.pid_index = pid_index
        self.batch_size = batch_size
        self.p = p
        self.k = k

        # estimate number of examples in an epoch
        self.length = 0
        for pid in self.pids:
            idxs = self.pid_index[pid]
            num = len(idxs)
            if num < self.k:
                num = self.k
            self.length += num - num % self.k

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = copy.deepcopy(self.pid_index[pid])
            if len(idxs) < self.k:
                idxs = np.random.choice(idxs, size=self.k, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.k:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []

        while len(avai_pids) >= self.p:
            selected_pids = random.sample(avai_pids, self.p)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length