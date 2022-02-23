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
        reg_loss = torch.tensor(0.).cuda()
        for w in weight_list:
            reg_loss += torch.pow(torch.norm(w, p=self.p), self.p)
        return reg_loss

