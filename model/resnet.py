import torch
from torch import nn
from torchvision.models import resnet50


class ResNet50(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(ResNet50, self).__init__()
        self.pretrained = pretrained
        self.num_classes = num_classes
        model = resnet50(pretrained=self.pretrained)
        self.feature_model = nn.Sequential(*(list(model.children())[:-1]))
        self.fc = nn.Linear(2048, self.num_classes)

    def forward(self, x):
        feature = self.feature_model(x)
        feature = torch.flatten(feature, 1)
        if self.training:
            return self.fc(feature), feature
        else:
            return feature
