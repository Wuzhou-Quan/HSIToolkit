import torch
from .attentions import ChannelAttention


class MultilayerPerceptronBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        inter_channels=None,
        act_layer=torch.nn.GELU,
        drop_rate=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        inter_channels = inter_channels or in_channels
        self.fc1 = torch.nn.Linear(in_channels, inter_channels)
        self.act = act_layer()
        self.fc2 = torch.nn.Linear(inter_channels, out_channels)
        self.drop = torch.nn.Dropout(drop_rate) if drop_rate > 0.0 else torch.nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FullyConnectionNetworkBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        inter_channels=None,
        act_layer=torch.nn.ReLU,
        drop_rate=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        inter_channels = inter_channels or in_channels
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            torch.nn.BatchNorm2d(inter_channels),
            act_layer(),
            torch.nn.Dropout(drop_rate) if drop_rate > 0.0 else torch.nn.Identity(),
            torch.nn.Conv2d(inter_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.block(x)


class ChannelAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        proj_inter_channels=None,
        attn_inter_channels=None,
        out_channels=None,
    ):
        super().__init__()
        proj_inter_channels = proj_inter_channels or in_channels // 4
        attn_inter_channels = attn_inter_channels or in_channels // 32
        out_channels = out_channels or in_channels
        self.proj = CAC3x3(in_channels, inter_channels=proj_inter_channels)
        self.ca = ChannelAttention(in_channels, inter_channels=attn_inter_channels)

    def forward(self, x):
        x = self.proj(x)
        x = self.ca(x) * x
        return x


class CAC3x3(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels=None,
        inter_channels=None,
        act_layer=torch.nn.GELU,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        inter_channels = inter_channels or in_channels // 4
        self.conv1 = torch.nn.Conv2d(in_channels, inter_channels, 3, 1, 1)
        self.act = act_layer()
        self.conv2 = torch.nn.Conv2d(inter_channels, out_channels, 3, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        return x


class ResidualBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, force_residual_align: bool = False):
        super().__init__()
        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # Channel align if needed
        if in_channels != out_channels or force_residual_align:
            self.channel_align = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), torch.nn.BatchNorm2d(out_channels))
        else:
            self.channel_align = torch.nn.Identity()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main body of the residual block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Channel align
        residual = self.channel_align(residual)

        # Add residual connection and apply ReLU
        out = (x + residual).relu()
        return out, x


class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, force_residual_align: bool = False):
        super().__init__()
        # First convolutional layer
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(out_channels)

        # Channel align if needed
        if in_channels != out_channels or force_residual_align:
            self.channel_align = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), torch.nn.BatchNorm2d(out_channels))
        else:
            self.channel_align = torch.nn.Identity()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main body of the residual block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Channel align
        residual = self.channel_align(residual)

        # Add residual connection and apply ReLU
        out = (x + residual).relu()
        return out, x


class ResidualBlock3D(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, force_residual_align: bool = False):
        super().__init__()
        # First convolutional layer
        self.conv1 = torch.nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm3d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)

        # Second convolutional layer
        self.conv2 = torch.nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm3d(out_channels)

        # Channel align if needed
        if in_channels != out_channels or force_residual_align:
            self.channel_align = torch.nn.Sequential(torch.nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False), torch.nn.BatchNorm3d(out_channels))
        else:
            self.channel_align = torch.nn.Identity()

        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # Main body of the residual block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        # Channel align
        residual = self.channel_align(residual)

        # Add residual connection and apply ReLU
        out = (x + residual).relu()
        return out, x


import collections
from itertools import repeat


def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse


_pair = _ntuple(2, "_pair")


class GatedSpatialConv2d(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, ref_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False):
        """
        :param in_channels:
        :param out_channels:
        :param kernel_size:
        :param stride:
        :param padding:
        :param dilation:
        :param groups:
        :param bias:
        """

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, "zeros")

        self._gate_conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(in_channels + ref_channels),
            torch.nn.Conv2d(in_channels + ref_channels, in_channels * 2, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels * 2, in_channels, 1),
            torch.nn.BatchNorm2d(in_channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, input_features, gating_features):
        """
        :param input_features:  [NxCxHxW]  featuers comming from the shape branch (canny branch).
        :param gating_features: [Nx1xHxW] features comming from the texture branch (resnet). Only one channel feature map.
        :return:
        """
        x = torch.cat([input_features, gating_features], dim=1)
        alphas = self._gate_conv(x)

        input_features = input_features * (alphas + 1)
        return torch.nn.functional.conv2d(input_features, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)
