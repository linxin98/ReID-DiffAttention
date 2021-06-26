import torch
from torch import nn


class TripletLoss(nn.Module):
    def __init__(self, margin, batch_size, p, k, soft_margin):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.batch_size = batch_size
        self.p = p
        self.k = k
        self.soft_margin = soft_margin

    def forward(self, distance_matrix):
        num_batch_images = distance_matrix.shape[0]
        num_images_per_class = num_batch_images // num_batch_classes
        template = torch.zeros((self.batch_size, self.batch_size))
        for x in range(self.batch_size):
            min = x // self.k * self.k
            max = min + self.k
            for y in range(min, max):
                if x != y:
                    template[x, y] = 1
        positive_distance_matrix = distance_matrix.mul(template)
        # print(positive_distance_matrix[:10,:10])
        negative_distance_matrix = distance_matrix - positive_distance_matrix
        # print(negative_distance_matrix[:10,:10])
        positive_distance = torch.amax(positive_distance_matrix, dim=1)
        # print(positive_distance[:10], positive_distance.shape)
        negative_distance, _ = torch.sort(negative_distance_matrix)
        # print(negative_distance[:10, :10])
        negative_distance = negative_distance[:, num_images_per_class]
        # print(negative_distance[:10], negative_distance.shape)
        one = torch.ones(self.batch_size)
        one = -one
        if self.soft_margin:
            soft_margin_loss = nn.SoftMarginLoss()
            loss = soft_margin_loss(positive_distance - negative_distance, one)
        else:
            losses = positive_distance - negative_distance + self.margin
            # print(losses[:10], losses.shape)
            losses = torch.clamp(losses, min=0)
            # print(losses[:10], losses.shape)
            loss = torch.mean(losses)
        return loss
