from torch import nn


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


class Classifier(nn.Module):
    def __init__(self, num_feature, num_class, bias=True, init='normal'):
        super(Classifier, self).__init__()
        self.num_feature = num_feature
        self.num_class = num_class
        self.bias = bias
        self.init = init
        self.classifier = nn.Linear(
            self.num_feature, self.num_class, bias=self.bias)
        if self.init == 'kaiming':
            self.classifier.apply(weights_init_kaiming)
        else:
            self.classifier.apply(weights_init_classifier)

    def forward(self, feat):
        cls_score = self.classifier(feat)
        return cls_score
