import torch.nn as nn
from torch.nn import functional as F


class BNActConv(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        groups=1,
        norm_layer=nn.BatchNorm1d,
        activation_layer=nn.ReLU,
        dilation=1,
        inplace=True,
        bias=None,
        conv_layer=nn.Conv1d,
        first_conv=True,
    ):
        if first_conv:
            norm_channels = out_channels
        else:
            norm_channels = in_channels

        if bias is None:
            bias = norm_layer is None
        layers = []

        if norm_layer is not None:
            layers.append(norm_layer(norm_channels))

        if activation_layer is not None:
            if activation_layer == nn.Tanh:
                layers.append(activation_layer())
            else:
                layers.append(activation_layer(inplace=inplace))

        if first_conv:
            layers.insert(
                0,
                conv_layer(
                    in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias
                ),
            )
        else:
            layers.append(
                conv_layer(
                    in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, groups=groups, bias=bias
                )
            )
        super(BNActConv, self).__init__(*layers)
        self.out_channels = out_channels


class SELayer(nn.Module):
    expansion = 1

    def __init__(self, in_channels, reudction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.se = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reudction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reudction, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1)
        y = self.se(y).view(b, c, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
        *,
        reduction=16
    ):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
