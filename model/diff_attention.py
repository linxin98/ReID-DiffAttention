import copy

import torch
from torch import nn
import torch.nn.functional as F


class DiffAttentionNet(nn.Module):
    def __init__(self, num_feature, in_transform, diff_ratio, out_transform, use_origin=True):
        super(DiffAttentionNet, self).__init__()
        self.num_feature = num_feature
        self.in_transform = in_transform
        self.diff_ratio = diff_ratio
        self.out_transform = out_transform
        self.use_origin = use_origin
        if self.use_origin:
            in_first_channel = 3
        else:
            in_first_channel = 1

        self.conv1 = nn.Conv1d(in_channels=in_first_channel, out_channels=1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(1)
        self.conv2 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm1d(1)

    def forward(self, x, y, keep_dim=True):
        # Make diff.
        diff = x - y
        if self.in_transform == 'abs':
            diff = torch.abs(diff)
        elif self.in_transform == 'square':
            diff = torch.square(diff)
        else:
            pass
        # Concentrate.
        if self.use_origin:
            input = torch.stack((diff, x, y), 1)
        else:
            input = torch.stack((diff), 1)
        # Calculate loacl attention.
        diff_attention = F.relu(self.bn1(self.conv1(input)))
        diff_attention = self.bn2(self.conv2(diff_attention))
        diff_attention = torch.squeeze(diff_attention)
        # Transform output.
        if self.out_transform == 'sigmoid':
            diff_attention = torch.sigmoid(diff_attention)
        else:
            pass
        # Output
        if keep_dim:
            x_feature = x.mul(diff_attention)
            y_feature = y.mul(diff_attention)
            return x_feature, y_feature
        else:
            return diff_attention
