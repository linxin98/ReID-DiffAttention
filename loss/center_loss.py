import torch
from torch import nn


class CenterLoss(nn.Module):

    def __init__(self, num_class, feat_dim, use_gpu=False, device=None):
        super(CenterLoss, self).__init__()
        self.num_class = num_class
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu
        self.device = device

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim).to(self.device))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_class, self.feat_dim))

    def forward(self, x, labels):
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_class) + torch.pow(
            self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_class, batch_size).t()
        distmat.addmm_(beta=1, alpha=-2, mat1=x, mat2=self.centers.t())

        classes = torch.arange(self.num_class).long()
        if self.use_gpu:
            classes = classes.to(self.device)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_class)
        mask = labels.eq(classes.expand(batch_size, self.num_class))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
        return loss
