import copy

import torch
from torch import nn
import torch.nn.functional as F


class DiffAttentionNet(nn.Module):
    def __init__(self, num_feature, in_transform, diff_ratio, out_transform, aggregate=True):
        super(DiffAttentionNet, self).__init__()
        self.num_feature = num_feature
        self.in_transform = in_transform
        self.diff_ratio = diff_ratio
        self.out_transform = out_transform
        self.aggregate = aggregate
        if self.aggregate:
            self.conv1 = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1, padding='same')
        self.fc1 = nn.Linear(self.num_feature, self.num_feature // self.diff_ratio, bias=False)
        self.fc2 = nn.Linear(self.num_feature // self.diff_ratio, self.num_feature, bias=False)

    def forward(self, x, y, keep_dim=True):
        # Make diff with input transform.
        diff = x - y
        if self.in_transform == 'abs':
            diff = torch.abs(diff)
        elif self.in_transform == 'square':
            diff = torch.square(diff)
        else:
            pass
        # Aggregate.
        if self.aggregate:
            feature = torch.stack((diff, x, y), 1)
            feature = self.conv1(feature)
            feature = feature.view(feature.shape[0], -1)
        else:
            feature = diff
        # Calculate attention.
        diff_attention = F.relu(self.fc1(feature))
        diff_attention = self.fc2(diff_attention)
        # Transform output.
        if self.out_transform == 'sigmoid':
            diff_attention = torch.sigmoid(diff_attention)
        else:
            pass
        # Output result.
        if keep_dim:
            x_feature = x.mul(diff_attention)
            y_feature = y.mul(diff_attention)
            return x_feature, y_feature
        else:
            return diff_attention
