import sys
import torch
from torch import nn

sys.path.append("")
from util import tool


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + \
        1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


class WeightedRegularizedTriplet(nn.Module):

    def __init__(self, batch_size, use_gpu=False, device=None):
        super(WeightedRegularizedTriplet, self).__init__()
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.device = device
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, features1, features2, labels):
        # if normalize_feature:
        #     global_feat = normalize(global_feat, axis=-1)
        # dist_mat = euclidean_dist(global_feat, global_feat)
        dist_mat = tool.get_distance_matrix(
            features1, features2, mode='template', shape=(self.batch_size, self.batch_size))

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = labels.expand(N, N).eq(labels.expand(N, N).t()).float()
        is_neg = labels.expand(N, N).ne(labels.expand(N, N).t()).float()
        
        if self.use_gpu:
          dist_mat = dist_mat.to(self.device)
          is_pos = is_pos.to(self.device)
          is_neg = is_neg.to(self.device)
          # print(is_pos)
          # print(dist_mat)

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # return loss, furthest_positive, closest_negative
        return loss

if __name__ == '__main__':
    features1 = torch.Tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    features2 = torch.Tensor([[4, 5, 6], [7, 8, 9], [7, 8, 9]])
    labels = torch.Tensor([0,1,2])
    triplet_loss_function = WeightedRegularizedTriplet(3)
    triplet_loss = triplet_loss_function(features1, features2, labels)
    print(triplet_loss)