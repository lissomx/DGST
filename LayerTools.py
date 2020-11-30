import torch
import torch.nn as nn
from torch.nn import functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ReShape(nn.Module):
    def __init__(self, dims=(1,2)):
        super(ReShape, self).__init__()
        self.dims = dims
    def forward(self, data):
        return data.transpose(*self.dims)

class OneHot(nn.Module):
    def __init__(self, vocb_size):
        super(OneHot, self).__init__()
        self.vocb_size = vocb_size
    def forward(self, data):
        return F.one_hot(data, self.vocb_size).float()