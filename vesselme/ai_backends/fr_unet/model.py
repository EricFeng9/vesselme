from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """FR-UNet 的基础卷积块：两层 3x3 卷积用于提取局部血管纹理。"""

    def __init__(self, channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        # 属性名必须保持为 conv，才能和官方 checkpoint 中的 block*.conv.conv.* 对齐。
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FeatureFuse(nn.Module):
    """把不同感受野的特征相加，保留细血管和较粗血管的响应。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 属性名沿用官方实现，确保权重 key 可 strict 加载。
        self.conv11 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv33 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv33_di = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=2,
            dilation=2,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.conv11(x) + self.conv33(x) + self.conv33_di(x))


class UpBlock(nn.Module):
    """上采样块，用转置卷积把低分辨率语义特征还原到更高分辨率。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 属性名必须保持为 up，才能匹配官方 checkpoint。
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class DownBlock(nn.Module):
    """下采样块，用步长卷积扩大感受野并压缩空间尺寸。"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # 属性名必须保持为 down，才能匹配官方 checkpoint。
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(x)


class FrBlock(nn.Module):
    """FR-UNet 网格中的一个节点，可选择输出上采样或下采样分支。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        is_up: bool = False,
        is_down: bool = False,
        fuse: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_up = is_up
        self.is_down = is_down
        self.fuse = FeatureFuse(in_channels, out_channels) if fuse else nn.Conv2d(in_channels, out_channels, 1)
        self.conv = ConvBlock(out_channels, dropout=dropout)
        if self.is_up:
            self.up = UpBlock(out_channels, out_channels // 2)
        if self.is_down:
            self.down = DownBlock(out_channels, out_channels * 2)

    def forward(self, x: torch.Tensor):
        if self.in_channels != self.out_channels:
            x = self.fuse(x)
        x = self.conv(x)
        if self.is_up and self.is_down:
            return x, self.up(x), self.down(x)
        if self.is_up:
            return x, self.up(x)
        if self.is_down:
            return x, self.down(x)
        return x


class FRUNet(nn.Module):
    """官方 FR-UNet 结构的推理版本，默认单通道输入、单通道血管 logits 输出。"""

    def __init__(
        self,
        num_classes: int = 1,
        num_channels: int = 1,
        feature_scale: int = 2,
        dropout: float = 0.2,
        fuse: bool = True,
        out_average: bool = True,
    ) -> None:
        super().__init__()
        self.out_average = out_average
        filters = [int(v / feature_scale) for v in [64, 128, 256, 512, 1024]]

        self.block1_3 = FrBlock(num_channels, filters[0], dropout, is_down=True, fuse=fuse)
        self.block1_2 = FrBlock(filters[0], filters[0], dropout, is_down=True, fuse=fuse)
        self.block1_1 = FrBlock(filters[0] * 2, filters[0], dropout, is_down=True, fuse=fuse)
        self.block10 = FrBlock(filters[0] * 2, filters[0], dropout, is_down=True, fuse=fuse)
        self.block11 = FrBlock(filters[0] * 2, filters[0], dropout, is_down=True, fuse=fuse)
        self.block12 = FrBlock(filters[0] * 2, filters[0], dropout, fuse=fuse)
        self.block13 = FrBlock(filters[0] * 2, filters[0], dropout, fuse=fuse)
        self.block2_2 = FrBlock(filters[1], filters[1], dropout, is_up=True, is_down=True, fuse=fuse)
        self.block2_1 = FrBlock(filters[1] * 2, filters[1], dropout, is_up=True, is_down=True, fuse=fuse)
        self.block20 = FrBlock(filters[1] * 3, filters[1], dropout, is_up=True, is_down=True, fuse=fuse)
        self.block21 = FrBlock(filters[1] * 3, filters[1], dropout, is_up=True, fuse=fuse)
        self.block22 = FrBlock(filters[1] * 3, filters[1], dropout, is_up=True, fuse=fuse)
        self.block3_1 = FrBlock(filters[2], filters[2], dropout, is_up=True, is_down=True, fuse=fuse)
        self.block30 = FrBlock(filters[2] * 2, filters[2], dropout, is_up=True, fuse=fuse)
        self.block31 = FrBlock(filters[2] * 3, filters[2], dropout, is_up=True, fuse=fuse)
        self.block40 = FrBlock(filters[3], filters[3], dropout, is_up=True, fuse=fuse)

        self.final1 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final2 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final3 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final4 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        self.final5 = nn.Conv2d(filters[0], num_classes, kernel_size=1)
        # 官方实现保留了 fuse 层，虽然 forward 默认不用；checkpoint 中含有该层参数。
        self.fuse = nn.Conv2d(5, num_classes, kernel_size=1)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1_3, x_down1_3 = self.block1_3(x)
        x1_2, x_down1_2 = self.block1_2(x1_3)
        x2_2, x_up2_2, x_down2_2 = self.block2_2(x_down1_3)
        x1_1, x_down1_1 = self.block1_1(torch.cat([x1_2, x_up2_2], dim=1))
        x2_1, x_up2_1, x_down2_1 = self.block2_1(torch.cat([x_down1_2, x2_2], dim=1))
        x3_1, x_up3_1, x_down3_1 = self.block3_1(x_down2_2)
        x10, x_down10 = self.block10(torch.cat([x1_1, x_up2_1], dim=1))
        x20, x_up20, x_down20 = self.block20(torch.cat([x_down1_1, x2_1, x_up3_1], dim=1))
        x30, x_up30 = self.block30(torch.cat([x_down2_1, x3_1], dim=1))
        _, x_up40 = self.block40(x_down3_1)
        x11, x_down11 = self.block11(torch.cat([x10, x_up20], dim=1))
        x21, x_up21 = self.block21(torch.cat([x_down10, x20, x_up30], dim=1))
        _, x_up31 = self.block31(torch.cat([x_down20, x30, x_up40], dim=1))
        x12 = self.block12(torch.cat([x11, x_up21], dim=1))
        _, x_up22 = self.block22(torch.cat([x_down11, x21, x_up31], dim=1))
        x13 = self.block13(torch.cat([x12, x_up22], dim=1))

        if self.out_average:
            return (
                self.final1(x1_1)
                + self.final2(x10)
                + self.final3(x11)
                + self.final4(x12)
                + self.final5(x13)
            ) / 5
        return self.final5(x13)


def _init_weights(module: nn.Module) -> None:
    """初始化只在无权重 smoke test 时使用；正式推理会加载官方权重覆盖参数。"""

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(module.weight, a=1e-2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=1e-2)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
