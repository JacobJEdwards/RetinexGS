from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FiLMLayer(nn.Module):
    def __init__(self: Self, embed_dim: int, feature_channels: int):
        super(FiLMLayer, self).__init__()
        self.layer = nn.Linear(embed_dim, feature_channels * 2)

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        gamma_beta = self.layer(embedding)
        gamma_beta = gamma_beta.view(gamma_beta.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        return gamma * x + beta

class RetinexNet(nn.Module):
    def __init__(
            self: Self, in_channels: int = 3, out_channels: int = 1, embed_dim: int = 32
    ) -> None:
        super(RetinexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=32)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid() we operate in log space, no need for sigmoid

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        c1 = self.relu(self.conv1(x))

        c1_modulated = self.film1(c1, embedding)

        p1 = self.pool(c1_modulated)
        c2 = self.relu(self.conv2(p1))
        p2 = self.pool(c2)

        up1 = self.upconv1(p2)
        up1_resized = F.interpolate(
            up1, size=p1.shape[2:], mode="bilinear", align_corners=False
        )

        merged = torch.cat([up1_resized, p1], dim=1)
        c3 = self.relu(self.conv3(merged))
        log_illumination = self.upconv2(c3)
        final_illumination = F.interpolate(
            log_illumination, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return final_illumination


class MultiScaleRetinexNet(nn.Module):
    def __init__(
            self: Self, in_channels: int = 3, out_channels: int = 3, embed_dim: int = 32, use_refinement: bool = False
    ) -> None:
        super(MultiScaleRetinexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=32)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.upconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        self.output_head_medium = nn.Conv2d(32, out_channels, kernel_size=3, padding=1)
        self.output_head_coarse = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.combination_layer = nn.Conv2d(
            # in_channels=num_scales * out_channels,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        self.use_refinement = use_refinement
        if self.use_refinement:
            # The RefinementNet now takes the output channels of the MSRNet as its input channels
            self.refinement_net = RefinementNet(out_channels, out_channels, embed_dim)

        self.relu = nn.ReLU()


        # self.sigmoid = nn.Sigmoid()

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        c1 = self.relu(self.conv1(x))
        c1_modulated = self.film1(c1, embedding)
        p1 = self.pool(c1_modulated)
        c2 = self.relu(self.conv2(p1))
        p2 = self.pool(c2)

        up1 = self.upconv1(p2)
        up1 = F.interpolate(
            up1, size=p1.shape[2:], mode="bilinear", align_corners=False
        )
        merged = torch.cat([up1, p1], dim=1)
        c3 = self.relu(self.conv3(merged))

        log_illumination_full_res = self.upconv2(c3)
        log_illumination_full_res = F.interpolate(
            log_illumination_full_res,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        log_illumination_medium_res = self.output_head_medium(c3)
        log_illumination_medium_res = F.interpolate(
            log_illumination_medium_res,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        log_illum_coarse_res = self.output_head_coarse(
            p2
        )  # [B, out_channels, H/4, W/4]
        log_illum_coarse_res = F.interpolate(
            log_illum_coarse_res, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        # concatenated_maps = torch.add(
        #     [
        #         log_illum_coarse_res,
        #         log_illumination_medium_res,
        #         log_illumination_full_res,
        #     ],
        #     dim=1,
        # )
        concatenated_maps = log_illum_coarse_res + log_illumination_medium_res + log_illumination_full_res

        final_illumination = self.combination_layer(concatenated_maps)

        # if self.use_refinement:
        #     illumination_residual = self.refinement_net(final_illumination, embedding)
        #     final_illumination = final_illumination + illumination_residual
        #     final_illumination = torch.clamp(final_illumination, 0, 1)

        return final_illumination

# class DenoisingNet(nn.Module):
#     def __init__(self: Self, in_channels: int, out_channels: int, embed_dim: int = 0) -> None:
#         super(DenoisingNet, self).__init__()
# 
#         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu1 = nn.LeakyReLU(0.2, inplace=True)
#         self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=64) if embed_dim > 0 else None
# 
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.relu2 = nn.LeakyReLU(0.2, inplace=True)
#         self.film2 = FiLMLayer(embed_dim=embed_dim, feature_channels=64) if embed_dim > 0 else None
# 
#         self.conv3 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
#         self.sigmoid = nn.Sigmoid()
#         self.tanh = nn.Tanh()
# 
#         nn.init.constant_(self.conv3.weight, 0.)
#         if self.conv3.bias is not None:
#             nn.init.constant_(self.conv3.bias, 0.)
# 
#         self.embed_dim = embed_dim
# 
#     def forward(self: Self, x: Tensor, embedding: Tensor | None = None) -> Tensor:
#         identity = x
# 
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu1(out)
#         if self.film1 is not None:
#             out = self.film1(out, embedding)
# 
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu2(out)
#         if self.film2 is not None:
#             out = self.film2(out, embedding)
# 
#         residual = self.conv3(out)
# 
#         denoised_output = identity + self.tanh(residual)
#         return denoised_output
    
    
class RefinementNet(nn.Module):
    def __init__(self: Self, in_channels: int, out_channels: int, embed_dim: int):
        super(RefinementNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.film2 = FiLMLayer(embed_dim=embed_dim, feature_channels=64)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.film3 = FiLMLayer(embed_dim=embed_dim, feature_channels=64)


        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1) # To output residual

        nn.init.constant_(self.output_conv.weight, 0.)
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0.)

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.film1(out, embedding)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.film2(out, embedding)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.film3(out, embedding)

        residual = self.output_conv(out)

        return residual

import torch
import torch.nn as nn


@torch.no_grad()
def init_weights(init_type='xavier'):
    if init_type == 'xavier':
        init = nn.init.xavier_normal_
    elif init_type == 'he':
        init = nn.init.kaiming_normal_
    else:
        init = nn.init.orthogonal_

    def initializer(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            init(m.weight)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight, 1.0, 0.01)
            nn.init.zeros_(m.bias)

    return initializer


class DownsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsampleBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.actv = nn.PReLU(out_channels)

    def forward(self, x):
        return self.actv(self.conv(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, cat_channels, out_channels):
        super(UpsampleBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels + cat_channels, out_channels, 3, padding=1)
        self.conv_t = nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2)
        self.actv = nn.PReLU(out_channels)
        self.actv_t = nn.PReLU(in_channels)

    def forward(self, x):
        upsample, concat = x
        upsample = self.actv_t(self.conv_t(upsample))
        return self.actv(self.conv(torch.cat([concat, upsample], 1)))


class InputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(out_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class OutputBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.actv_1 = nn.PReLU(in_channels)
        self.actv_2 = nn.PReLU(out_channels)

    def forward(self, x):
        x = self.actv_1(self.conv_1(x))
        return self.actv_2(self.conv_2(x))


class DenoisingBlock(nn.Module):
    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(in_channels + inner_channels, inner_channels, 3, padding=1)
        self.conv_2 = nn.Conv2d(in_channels + 2 * inner_channels, inner_channels, 3, padding=1)
        self.conv_3 = nn.Conv2d(in_channels + 3 * inner_channels, out_channels, 3, padding=1)

        self.actv_0 = nn.PReLU(inner_channels)
        self.actv_1 = nn.PReLU(inner_channels)
        self.actv_2 = nn.PReLU(inner_channels)
        self.actv_3 = nn.PReLU(out_channels)

    def forward(self, x):
        out_0 = self.actv_0(self.conv_0(x))

        out_0 = torch.cat([x, out_0], 1)
        out_1 = self.actv_1(self.conv_1(out_0))

        out_1 = torch.cat([out_0, out_1], 1)
        out_2 = self.actv_2(self.conv_2(out_1))

        out_2 = torch.cat([out_1, out_2], 1)
        out_3 = self.actv_3(self.conv_3(out_2))

        return out_3 + x


class DenoisingNet(nn.Module):
    r"""
    Residual-Dense U-net for image denoising.
    """
    def __init__(self, **kwargs):
        super().__init__()

        channels = kwargs['channels']
        filters_0 = kwargs['base_filters']
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        # Encoder:
        # Level 0:
        self.input_block = InputBlock(channels, filters_0)
        self.block_0_0 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_1 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.down_0 = DownsampleBlock(filters_0, filters_1)

        # Level 1:
        self.block_1_0 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_1 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.down_1 = DownsampleBlock(filters_1, filters_2)

        # Level 2:
        self.block_2_0 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_1 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.down_2 = DownsampleBlock(filters_2, filters_3)

        # Level 3 (Bottleneck)
        self.block_3_0 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)
        self.block_3_1 = DenoisingBlock(filters_3, filters_3 // 2, filters_3)

        # Decoder
        # Level 2:
        self.up_2 = UpsampleBlock(filters_3, filters_2, filters_2)
        self.block_2_2 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)
        self.block_2_3 = DenoisingBlock(filters_2, filters_2 // 2, filters_2)

        # Level 1:
        self.up_1 = UpsampleBlock(filters_2, filters_1, filters_1)
        self.block_1_2 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)
        self.block_1_3 = DenoisingBlock(filters_1, filters_1 // 2, filters_1)

        # Level 0:
        self.up_0 = UpsampleBlock(filters_1, filters_0, filters_0)
        self.block_0_2 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)
        self.block_0_3 = DenoisingBlock(filters_0, filters_0 // 2, filters_0)

        self.output_block = OutputBlock(filters_0, channels)

    def forward(self, inputs):
        out_0 = self.input_block(inputs)    # Level 0
        out_0 = self.block_0_0(out_0)
        out_0 = self.block_0_1(out_0)

        out_1 = self.down_0(out_0)          # Level 1
        out_1 = self.block_1_0(out_1)
        out_1 = self.block_1_1(out_1)

        out_2 = self.down_1(out_1)          # Level 2
        out_2 = self.block_2_0(out_2)
        out_2 = self.block_2_1(out_2)

        out_3 = self.down_2(out_2)          # Level 3 (Bottleneck)
        out_3 = self.block_3_0(out_3)
        out_3 = self.block_3_1(out_3)

        out_4 = self.up_2([out_3, out_2])   # Level 2
        out_4 = self.block_2_2(out_4)
        out_4 = self.block_2_3(out_4)

        out_5 = self.up_1([out_4, out_1])   # Level 1
        out_5 = self.block_1_2(out_5)
        out_5 = self.block_1_3(out_5)

        out_6 = self.up_0([out_5, out_0])   # Level 0
        out_6 = self.block_0_2(out_6)
        out_6 = self.block_0_3(out_6)

        return self.output_block(out_6) + inputs