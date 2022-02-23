import sys
import numpy as np
import torch
from torch import nn

sys.path.append("")
from util import tool

class TripletLoss(nn.Module):
    def __init__(self, batch_size, p, k, margin=0.3, soft_margin=False):
        super(TripletLoss, self).__init__()
        self.batch_size = batch_size
        self.p = p
        self.k = k
        self.margin = margin
        self.soft_margin = soft_margin

        self.template = torch.zeros((self.batch_size, self.batch_size))
        for x in range(self.batch_size):
            min = x // self.k * self.k
            max = min + self.k
            for y in range(min, max):
                self.template[x, y] = 1

    def forward(self, features1, features2):
        distance_matrix = tool.get_distance_matrix(
            features1, features2, mode='template', shape=(self.batch_size, self.batch_size))
        positive_distance_matrix = distance_matrix.mul(self.template)
        negative_distance_matrix = distance_matrix - positive_distance_matrix
        positive_distance = torch.amax(positive_distance_matrix, dim=1)
        negative_distance, _ = torch.sort(negative_distance_matrix)
        negative_distance = negative_distance[:, self.k]
        one = -torch.ones(self.batch_size)
        if self.soft_margin:
            soft_margin_loss = nn.SoftMarginLoss()
            loss = soft_margin_loss(positive_distance - negative_distance, one)
        else:
            losses = positive_distance - negative_distance + self.margin
            losses = torch.clamp(losses, min=0)
            loss = torch.mean(losses)
        return loss


if __name__ == '__main__':
    features1 = torch.Tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    features2 = torch.Tensor([[4, 5, 6], [7, 8, 9], [7, 8, 9]])
    triplet_loss_function = TripletLoss(3, 3, 1)
    triplet_loss = triplet_loss_function(features1, features2)
    print(triplet_loss)
