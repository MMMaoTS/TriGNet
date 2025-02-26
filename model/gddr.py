import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import BNActConv
import config


def mss(x):
    return x.pow(2).mean()

class GateUnitModule(nn.Module):
    def __init__(self, channels, b=1, gamma=2):
        super(GateUnitModule, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 16, channels, 1, bias=False),
        )

        self.conv = nn.Conv1d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_x = self.mlp(self.max_pool(x))
        avg_x = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_x + avg_x)
        x = x * channel_out

        max_x, _ = torch.max(x, dim=1, keepdim=True)
        avg_x = torch.mean(x, dim=1, keepdim=True)
        prob = self.sigmoid(self.conv(torch.cat([max_x, avg_x], dim=1)))
        return prob.mean()


class DRModule(nn.Module):
    def __init__(self, channels, num_layers, threshold):
        super(DRModule, self).__init__()
        self.channels = channels
        self.num_layers = num_layers
        self.threshold = threshold
        self.residual = []
        self.gates = self.make_gate_layer()
        self.init_res()

    def init_res(self):
        for i in range(self.num_layers):
            self.residual.append(0.0)

    def update(self, x, index):
        self.residual[index] = x

    def make_gate_layer(self):
        layers = []
        for i in range(self.num_layers):
            layers.append(GateUnitModule(self.channels))
        return nn.Sequential(*layers)

    def fit_residual(self, x, proj, scale, index):
        if proj > 0.5:
            weight = F.softmax(torch.arange(1, index + 1, dtype=torch.float32), dim=0)
            weight = 1 - weight
            res = 0.0
            for i in range(index):
                res = res + self.residual[i] * weight[i]

            scale = scale if scale > 0.1 else 0.1
            x = x + scale * res
        return x

    def forward(self, x, index):
        proj = self.gates[index](x)
        x = self.fit_residual(x, proj, config.DC_SCALE, index)
        return x


class GDDRBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, num_layers, threshold):
        super(GDDRBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        self.gatedDC = DRModule(self.in_channels, self.num_layers, threshold)
        self.layers = self._make_layer()

    def _make_layer(self):
        layers = []
        for i in range(self.num_layers):
            layers.append(BNActConv(self.in_channels, self.out_channels, kernel_size=self.kernel_size, padding="same"))
        return nn.Sequential(*layers)

    def forward(self, x):
        for index in range(self.num_layers):
            self.gatedDC.update(x, index)
            x = self.layers[index](x)
            x = self.gatedDC(x, index)
        return x
