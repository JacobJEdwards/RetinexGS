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

        if self.use_refinement:
            # The refinement_net now acts on the initial combined illumination map
            # and returns a residual.
            illumination_residual = self.refinement_net(final_illumination, embedding)
            final_illumination = final_illumination + illumination_residual
            final_illumination = torch.clamp(final_illumination, 0, 1)

        return final_illumination

class DenoisingNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int = 0):
        super(DenoisingNet, self).__init__()

        # --- Improved DenoisingNet Architecture ---
        # Added a simple UNet-like structure for better context and detail preservation
        # Used LeakyReLU for non-linearity

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True) # Changed to LeakyReLU
        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=64) if embed_dim > 0 else None

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2) # Downsample
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True) # Changed to LeakyReLU
        self.film2 = FiLMLayer(embed_dim=embed_dim, feature_channels=128) if embed_dim > 0 else None

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.film3 = FiLMLayer(embed_dim=embed_dim, feature_channels=128) if embed_dim > 0 else None

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1) # Upsample
        self.bn4 = nn.BatchNorm2d(64)
        self.relu4 = nn.LeakyReLU(0.2, inplace=True)
        self.film4 = FiLMLayer(embed_dim=embed_dim, feature_channels=64) if embed_dim > 0 else None

        # Output layer to predict the residual
        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

        self.embed_dim = embed_dim

        # Initialize output_conv weights to near zero for residual learning
        nn.init.constant_(self.output_conv.weight, 0.)
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0.)


    def forward(self, x: Tensor, embedding: Tensor | None = None) -> Tensor:
        identity = x

        # Encoder path
        out = self.conv1(x)
        out = self.bn1(out)
        out1 = self.relu1(out)
        if self.film1 is not None:
            out1 = self.film1(out1, embedding)

        out = self.conv2(out1) # Downsample
        out = self.bn2(out)
        out2 = self.relu2(out)
        if self.film2 is not None:
            out2 = self.film2(out2, embedding)

        # Bottleneck
        out = self.conv3(out2)
        out = self.bn3(out)
        out = self.relu3(out)
        if self.film3 is not None:
            out = self.film3(out, embedding)

        # Decoder path
        out = self.upconv1(out) # Upsample
        out = self.bn4(out)
        out = self.relu4(out)
        if self.film4 is not None:
            out = self.film4(out, embedding)

        # You could add skip connections here if you had a more direct U-Net
        # For simplicity, let's assume the previous `out1` is at the right resolution to add a skip connection
        # if using actual U-Net, you'd concatenate rather than add (e.g., torch.cat([out, out1], dim=1))
        # For this simple example, we'll keep it sequential for now after upsampling.

        residual = self.output_conv(out)

        denoised_output = identity + residual
        return self.sigmoid(denoised_output)

# --- RefinementNet (Moved here from utils.py for direct modification) ---
# Assuming FiLMLayer is imported from utils
class RefinementNet(nn.Module):
    def __init__(self: Self, in_channels: int, out_channels: int, embed_dim: int):
        super(RefinementNet, self).__init__()

        # --- Improved RefinementNet Architecture ---
        # Added more layers and ensured skip connection philosophy (outputting residual)
        # Used LeakyReLU

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=64)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.film2 = FiLMLayer(embed_dim=embed_dim, feature_channels=64)

        # Additional layer
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.film3 = FiLMLayer(embed_dim=embed_dim, feature_channels=64)


        self.output_conv = nn.Conv2d(64, out_channels, kernel_size=3, padding=1) # To output residual

        # Initialize the output convolutional layer to near zero
        nn.init.constant_(self.output_conv.weight, 0.)
        if self.output_conv.bias is not None:
            nn.init.constant_(self.output_conv.bias, 0.)

    def forward(self: Self, x: Tensor, embedding: Tensor) -> Tensor:
        # No identity = x here, as the residual is added externally in MultiScaleRetinexNet
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

        # Return the residual directly. The addition and clamping happens in MultiScaleRetinexNet.
        return residual