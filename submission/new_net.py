from net2d import ResNet2D18
from net3d import ResNet3D18

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class NewModel(nn.Module):

    def __init__(self, channel_2d=3, channel_3d=1, num_classes=3, num_node=10, ):
        super(NewModel, self).__init__()
        self.resnet2d = ResNet2D18(num_node, channel_2d)
        self.resnet3d = ResNet3D18(num_node, channel_3d)
        self.linear = nn.Linear(num_node*2, num_classes)

    def forward(self, x):
        out3d = self.resnet3d(x[0])
        out2d = self.resnet2d(x[1])
        out = torch.cat([out3d, out2d], dim=1)
        out = self.linear(out)

        return out


if __name__ == '__main__':
    net = NewModel()
    x3d = torch.randn(2, 1, 160, 192, 160).cuda()
    x2d = torch.randn(2, 8, 3, 32, 32).cuda()
    net.cuda()
    y = net([x3d, x2d])
    print(y.size())