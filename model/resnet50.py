import torch
from torch import nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet50, self).__init__()
        self.pretrained = pretrained
        model = resnet50(pretrained=self.pretrained)
        self.feature_model = nn.Sequential(*(list(model.children())[:-1]))

    def forward(self, x):
        feature = self.feature_model(x)
        feature = torch.flatten(feature, 1)
        return feature
