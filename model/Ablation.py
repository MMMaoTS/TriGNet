import torch
import torch.nn as nn
from model.gddr import GDDRBlock
from model.layers import BNActConv, SEBasicBlock
import config


class Layer1(nn.Module):
    def __init__(self):
        super(Layer1, self).__init__()

        self.br1 = BNActConv(1, 1, kernel_size=125, padding="same")
        self.br2 = BNActConv(1, 1, kernel_size=25, padding="same")
        self.br3 = BNActConv(1, 1, kernel_size=5, padding="same")

    def forward(self, x):
        x1 = self.br1(x)
        x2 = self.br2(x)
        x3 = self.br3(x)
        x = torch.cat([x1, x2, x3], dim=1)
        x = x.unsqueeze(1)
        return x


class Layer2(nn.Module):
    def __init__(self, scale=0.1):
        super(Layer2, self).__init__()
        self.scale = scale
        self.layer1 = BNActConv(
            1, 32, kernel_size=(1, 50), stride=(1, 2), padding=(0, 0), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d
        )
        self.layer2 = BNActConv(
            32, 32, kernel_size=(1, 16), stride=(1, 2), padding=(0, 0), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d
        )
        self.maxpool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2))

    def forward(self, x):
        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.maxpool(x)
        return x


class Layer3(nn.Module):
    def __init__(self, scale=1):
        super(Layer3, self).__init__()
        self.scale = scale
        self.br1 = nn.Sequential(
            BNActConv(32, 32, kernel_size=(1, 7), padding=(0, 3), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d),
            BNActConv(32, 32, kernel_size=(7, 1), padding=(3, 0), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d),
        )
        self.br2 = nn.Sequential(
            BNActConv(32, 32, kernel_size=(1, 5), padding=(0, 2), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d),
            BNActConv(32, 32, kernel_size=(5, 1), padding=(2, 0), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d),
        )
        self.br3 = nn.Sequential(
            BNActConv(32, 32, kernel_size=(1, 3), padding=(0, 1), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d),
            BNActConv(32, 32, kernel_size=(3, 1), padding=(1, 0), norm_layer=nn.BatchNorm2d, conv_layer=nn.Conv2d),
        )

    def forward(self, x):
        res = x

        x1 = self.br1(x)
        x1 = x1 + self.scale * res

        x2 = self.br2(x)
        x2 = x2 + self.scale * res

        x3 = self.br3(x)
        x3 = x3 + self.scale * res

        x = torch.cat([x1, x2, x3], dim=1)
        x = torch.flatten(x, -2)
        return x


class Layer4(nn.Module):
    def __init__(self, threshold=0.025):
        super(Layer4, self).__init__()
        in_channels = 96
        self.br1 = self.make_blocks(in_channels, 7, 1, config.DC_NUM_LAYERS, threshold)
        self.br2 = self.make_blocks(in_channels, 5, 1, config.DC_NUM_LAYERS, threshold)
        self.br3 = self.make_blocks(in_channels, 3, 1, config.DC_NUM_LAYERS, threshold)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.se1 = SEBasicBlock(in_channels, in_channels)
        self.se2 = SEBasicBlock(in_channels, in_channels)
        self.se3 = SEBasicBlock(in_channels, in_channels)

    def make_blocks(self, in_channels, kernel_size, num_blocks, num_layers, threshold):
        layers = []
        for i in range(num_layers):
            layers.append(BNActConv(in_channels, in_channels, kernel_size=kernel_size, padding="same"))
        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.br1(x)
        x1 = self.avgpool(x1)
        x1 = self.se1(x1)

        x2 = self.br2(x)
        x2 = self.avgpool(x2)
        x2 = self.se2(x2)

        x3 = self.br3(x)
        x3 = self.avgpool(x3)
        x3 = self.se3(x3)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.avgpool(x)
        return x


class Layer5(nn.Module):
    def __init__(self):
        super(Layer5, self).__init__()
        self.in_planes = 288
        self.se = self._make_layer(SEBasicBlock, 288, 1, 1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.se(x)
        return x


class Ablation(nn.Module):
    def __init__(self):
        super(Ablation, self).__init__()
        self.ly1 = Layer1()
        self.ly2 = Layer2()
        self.ly3 = Layer3()
        self.ly4 = Layer4()
        self.ly5 = Layer5()

        self.clf = nn.Sequential(nn.Flatten(), nn.Linear(288, 512), nn.Linear(512, config.NUM_CLASSES))

    def forward(self, x):
        x = self.ly1(x)
        x = self.ly2(x)
        x = self.ly3(x)
        x = self.ly4(x)
        x = self.ly5(x)
        x = self.clf(x)
        return x
