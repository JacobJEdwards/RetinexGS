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

class RefinementNet(nn.Module):
    def __init__(self: Self, in_channels: int, out_channels: int, embed_dim: int):
        super(RefinementNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.film_bottleneck = FiLMLayer(embed_dim=embed_dim, feature_channels=128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        e1 = self.relu(self.bn1(self.conv1(x)))
        e1 = self.relu(self.bn1_2(self.conv1_2(e1)))
        e2_pre_mod = self.relu(self.bn2(self.conv2(e1)))
        e2_pre_mod = self.relu(self.bn2_2(self.conv2_2(e2_pre_mod))) 

        e2 = self.film_bottleneck(e2_pre_mod, embedding)

        d1 = self.relu(self.bn3(self.upconv1(e2)))
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d1 = torch.cat([d1, e1], dim=1) 
        d1 = self.relu(self.bn3_2(self.conv3(d1))) 

        output = self.output_layer(d1)
        return output
    
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

        self.relu = nn.LeakyReLU(0.2, inplace=True)
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

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.bn1_2 = nn.BatchNorm2d(64) 
        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=64) 

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) 
        self.bn2_2 = nn.BatchNorm2d(128)


        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) 
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(64)
        self.conv3_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(64)

        self.upconv2 = nn.ConvTranspose2d(64, out_channels, kernel_size=2, stride=2)

        self.output_head_medium = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.output_head_coarse = nn.Conv2d(128, out_channels, kernel_size=3, padding=1) 

        self.combination_layer = nn.Conv2d(
            # in_channels=num_scales * out_channels,
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
        )

        self.use_refinement = use_refinement
        if self.use_refinement:
            self.refinement_net = RefinementNet(out_channels, out_channels, embed_dim)

        self.relu = nn.LeakyReLU(negative_slope=0.01)


        # self.sigmoid = nn.Sigmoid()

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        c1 = self.relu(self.bn1(self.conv1(x)))
        c1 = self.relu(self.bn1_2(self.conv1_2(c1)))
        c1_modulated = self.film1(c1, embedding)
        p1 = self.pool(c1_modulated)
        c2 = self.relu(self.bn2(self.conv2(p1)))
        c2 = self.relu(self.bn2_2(self.conv2_2(c2))) 
        p2 = self.pool(c2)

        up1 = self.upconv1(p2)
        up1 = F.interpolate(
            up1, size=c1_modulated.shape[2:], mode="bilinear", align_corners=False # Interpolate to c1_modulated size
        )
        merged = torch.cat([up1, c1_modulated], dim=1) 
        c3 = self.relu(self.bn3_2(self.conv3(merged)))
        c3 = self.relu(self.bn3_3(self.conv3_2(c3)))

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

        if self.use_refinement:
            illumination_residual = self.refinement_net(final_illumination, embedding)
            final_illumination = final_illumination + illumination_residual
            final_illumination = torch.clamp(final_illumination, 0, 1)

        return final_illumination