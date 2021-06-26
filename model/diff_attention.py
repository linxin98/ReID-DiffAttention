import copy

import torch
from torch import nn
import torch.nn.functional as F


class DiffAttentionNet(nn.Module):
    def __init__(self, num_feature, in_transform, diff_ratio, out_transform):
        super(DiffAttentionNet, self).__init__()
        self.num_feature = num_feature
        self.in_transform = in_transform
        self.diff_ratio = diff_ratio
        self.out_transform = out_transform

        self.fc1 = nn.Linear(self.num_feature, self.num_feature // self.diff_ratio)
        self.fc2 = nn.Linear(self.num_feature // self.diff_ratio, self.num_feature)

    def forward(self, x, y, keep_dim=True):
        diff = x - y
        if self.in_transform == 'abs':
            diff_attention = torch.abs(diff)
        elif self.in_transform == 'square':
            diff_attention = torch.square(diff)
        else:
            diff_attention = copy.deepcopy(diff)

        diff_attention = F.relu(self.fc1(diff_attention))
        diff_attention = self.fc2(diff_attention)

        if self.out_transform == 'sigmoid':
            diff_attention = torch.sigmoid(diff_attention)
        else:
            pass

        if keep_dim:
            x_feature = x.mul(diff_attention)
            y_feature = y.mul(diff_attention)
            return x_feature, y_feature
        else:
            diff = diff.mul(diff_attention)
            distance = torch.sqrt(torch.sum(torch.square(diff), 1))
            return distance
