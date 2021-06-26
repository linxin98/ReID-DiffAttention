import torch
from torch import nn


class Regularization(nn.Module):
    def __init__(self, p):
        super(Regularization, self).__init__()
        self.p = p

    def forward(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_list.append(param)

        reg_loss = torch.tensor(0)

        for w in weight_list:
            reg = torch.pow(torch.norm(w, p=self.p), self.p)
            reg_loss = reg_loss + reg

        return reg_loss
