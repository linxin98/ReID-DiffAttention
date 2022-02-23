import random
from re import template

import numpy as np
import torch
from torch import nn


def setup_random_seed(seed):
    # Python
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    # CUDA
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel()
                        for p in model.parameters() if p.requires_grad)
    return {'Total Params': total_num, 'Trainable Params': trainable_num}


def get_templates(m, n, mode='normal'):
    template1 = []
    template2 = []
    if mode == 'val':
        for x in range(0, m):
            for y in range(0, n):
                template1.append(x)
                template2.append(y)
    else:
        for x in range(0, m - 1):
            for y in range(x + 1, n):
                template1.append(x)
                template2.append(y)
    return template1, template2


def get_distance_matrix(features1, features2, mode='normal', cpu=False, shape=None):
    m, n = features1.size(0), features2.size(0)
    if mode == 'template':
        # Mainly for different distance functions.
        assert shape is not None, 'If use template mode, shape should not be None.'
        m, n = shape[0], shape[1]
        distance = nn.functional.pairwise_distance(features1, features2)
        distance_matrix = torch.zeros((m, n))
        index = 0
        for x in range(0, m - 1):
            for y in range(x + 1, n):
                distance_matrix[x, y] = distance[index]
                distance_matrix[y, x] = distance[index]
                index += 1
    elif mode == 'aggregate':
        # Mainly for re-ranking.
        features = torch.cat([features1, features2])
        all_num = m + n
        distance_matrix = torch.pow(features, 2).sum(dim=1, keepdim=True).expand(
            all_num, all_num) + torch.pow(features, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t()
        distance_matrix.addmm_(features, features.t(), beta=1, alpha=-2)
        distance_matrix = distance_matrix.clamp(min=0).sqrt()
    else:
        # Mainly for evaluating.
        distance_matrix = torch.pow(features1, 2).sum(dim=1, keepdim=True).expand(
            m, n) + torch.pow(features2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distance_matrix.addmm_(features1, features2.t(), beta=1, alpha=-2)
        distance_matrix = distance_matrix.clamp(min=0).sqrt()
    if cpu:
        distance_matrix = distance_matrix.cpu().numpy()
    return distance_matrix


if __name__ == '__main__':
    print('Util Test')
    # template
    template1, template2 = get_templates(4, 4)
    print(template1)
    print(template2)
    template1, template2 = get_templates(4, 4, mode='val')
    print(template1)
    print(template2)
    # distance
    features1 = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    features2 = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distance_matrix = get_distance_matrix(
        features1, features2, mode='normal')
    print(distance_matrix)
    features1 = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    features2 = torch.Tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    distance_matrix = get_distance_matrix(
        features1, features2, mode='aggregate')
    print(distance_matrix)
    features1 = torch.Tensor([[1, 2, 3], [1, 2, 3], [4, 5, 6]])
    features2 = torch.Tensor([[4, 5, 6], [7, 8, 9], [7, 8, 9]])
    distance_matrix = get_distance_matrix(
        features1, features2, mode='template', shape=(3, 3))
    print(distance_matrix)
