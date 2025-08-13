import math

import torch.nn as nn
from torch import Tensor
from torch.nn import init
import torch
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.concat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channel, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False, padding_mode: str = 'replicate'):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
            padding_mode=padding_mode
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class RetinexBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.conv(x)))

class UpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        # self.conv = DepthwiseSeparableConv(in_channels, out_channels * 4, kernel_size=3, padding=1)
        # self.shuffle = nn.PixelShuffle(2)
        # self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        # self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        # return self.act(self.norm(self.shuffle(self.conv(x))))
        x = self.upsample(x)
        x = self.conv(x)
        return x

class ECALayer(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avgpool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class MambaBlock(nn.Module):
    def __init__(self, channels: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.channels = channels
        self.norm = nn.LayerNorm(channels)

        # self.mamba = Mamba(
        #     d_model=channels,
        #     d_state=d_state,
        #     d_conv=d_conv,
        #     expand=expand,
        #     bidirectional=True
        # )

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape

        x_residual = x

        # (B, C, H, W) -> (B, H*W, C)
        x_seq = x.flatten(2).transpose(1, 2)

        x_mamba = self.mamba(self.norm(x_seq))

        # (B, H*W, C) -> (B, C, H, W)
        x_out = x_mamba.transpose(1, 2).reshape(b, c, h, w)

        return x_out + x_residual

class SpatiallyFiLMLayer(nn.Module):
    def __init__(self, feature_channels: int, embed_dim: int):
        super().__init__()
        conditioning_channels = feature_channels + embed_dim
        self.param_predictor = nn.Sequential(
            DepthwiseSeparableConv(conditioning_channels, feature_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=1)
        )
        nn.init.zeros_(self.param_predictor[-1].weight)
        nn.init.zeros_(self.param_predictor[-1].bias)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        b, _, h, w = x.shape
        tiled_embedding = embedding.view(b, -1, 1, 1).expand(b, -1, h, w)
        conditioning_input = torch.cat([x, tiled_embedding], dim=1)

        mod_params = self.param_predictor(conditioning_input)
        gamma, beta = torch.chunk(mod_params, 2, dim=1)

        return (1 + gamma) * x + beta

class FiLMLayer(nn.Module):
    def __init__(self, embed_dim: int, feature_channels: int):
        super(FiLMLayer, self).__init__()
        self.layer = nn.Linear(embed_dim, feature_channels * 2)

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        gamma_beta = self.layer(embedding)
        gamma_beta = gamma_beta.view(gamma_beta.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        return gamma * x + beta

class DenoisingHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.enc1 = RetinexBlock(in_channels, 16, stride=2)
        self.enc2 = RetinexBlock(16, 32, stride=2)
        self.bottle = RetinexBlock(32, 32)

        self.dec2 = UpBlock(32, 16)
        self.dec2_conv = RetinexBlock(32, 16)

        self.dec1 = UpBlock(16, in_channels)
        self.final_conv = RetinexBlock(in_channels * 2, in_channels)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottle(e2)

        d2_up = self.dec2(b)
        if d2_up.shape[2:] != e1.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d2_cat = torch.cat([d2_up, e1], dim=1)
        d2 = self.dec2_conv(d2_cat)

        d1_up = self.dec1(d2)
        if d1_up.shape[2:] != x.shape[2:]:
            d1_up = F.interpolate(d1_up, size=x.shape[2:], mode='bilinear', align_corners=False)
        d1_cat = torch.cat([d1_up, x], dim=1)

        output = self.final_conv(d1_cat)

        return output

class MultiScaleRetinexNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            embed_dim: int = 32,
            enable_dynamic_weights: bool = False,
            predictive_adaptive_curve: bool = False,
            learn_local_exposure: bool = False,
            num_weight_scales: int = 11,
    ) -> None:
        super(MultiScaleRetinexNet, self).__init__()
        self.denoising_head = DenoisingHead(in_channels)
        self.embed_dim = embed_dim

        self.in_conv = RetinexBlock(in_channels, 16)
        self.film1 = SpatiallyFiLMLayer(embed_dim=embed_dim, feature_channels=16)

        self.enc1 = RetinexBlock(16, 32, stride=2)
        self.enc2 = RetinexBlock(32, 64, stride=2)

        self.bottleneck = nn.Sequential(
            RetinexBlock(64, 64),
            # MambaBlock(64),
            CBAM(64)
        )

        self.dec2 = UpBlock(64, 32)
        self.dec2_conv = RetinexBlock(64, 32)
        self.dec2_attn = ECALayer(32)

        self.dec1 = UpBlock(32, 16)
        self.dec1_conv = RetinexBlock(32, 16)
        self.dec1_attn = ECALayer(16)

        self.out_conv = DepthwiseSeparableConv(16, out_channels, kernel_size=3, padding=1)

        self.nested_dec = RetinexBlock(16, 16)

        self.enable_dynamic_weights = enable_dynamic_weights
        self.predictive_adaptive_curve = predictive_adaptive_curve
        self.learn_local_exposure = learn_local_exposure

        if self.predictive_adaptive_curve:
            self.adaptive_curve_head = nn.Sequential(
                DepthwiseSeparableConv(16, 8, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(8, 2, kernel_size=1)
            )
            nn.init.zeros_(self.adaptive_curve_head[-1].weight)
            nn.init.zeros_(self.adaptive_curve_head[-1].bias)

        if self.learn_local_exposure:
            self.predicted_local_mean_head = nn.Sequential(
                DepthwiseSeparableConv(16, 8, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(8, 1, kernel_size=1),
                nn.Sigmoid()
            )

        if self.enable_dynamic_weights:
            self.log_vars = nn.Parameter(torch.zeros(num_weight_scales, dtype=torch.float32))

        self.confidence_head = nn.Sequential(
            DepthwiseSeparableConv(16, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.variance_head = nn.Sequential(
            DepthwiseSeparableConv(16, 8, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(8, 1, kernel_size=1)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.GroupNorm):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)

    def forward(self, x: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None,
    Tensor | None, Tensor | None, Tensor | None]:
        x = self.denoising_head(x)

        b, _ = embedding.shape
        embed_full = embedding.view(b, self.embed_dim, 1, 1).expand(b, self.embed_dim, x.shape[2], x.shape[3])
        embed_half = F.avg_pool2d(embed_full, 2)
        embed_quarter = F.avg_pool2d(embed_half, 2)

        e0 = self.in_conv(x)
        e0_modulated = self.film1(e0, embedding)

        # e1 = self.enc1(e0_modulated)
        # e2 = self.enc2(e1)

        e1 = self.enc1(e0_modulated)
        e1 = SpatiallyFiLMLayer(32, self.embed_dim)(e1, embed_half.view(b, self.embed_dim))

        e2 = self.enc2(e1)
        e2 = SpatiallyFiLMLayer(64, self.embed_dim)(e2, embed_quarter.view(b, self.embed_dim))

        b = self.bottleneck(e2)

        d2_up = self.dec2(b)
        if d2_up.shape[2:] != e1.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d2 = torch.cat([d2_up, e1], dim=1)
        d2 = self.dec2_conv(d2)
        d2 = self.dec2_attn(d2)

        d1_up = self.dec1(d2)
        if d1_up.shape[2:] != e0_modulated.shape[2:]:
            d1_up = F.interpolate(d1_up, size=e0_modulated.shape[2:], mode='bilinear', align_corners=False)

        d1 = torch.cat([d1_up, e0_modulated], dim=1)
        d1 = self.dec1_conv(d1)
        d1 = self.dec1_attn(d1)

        d1_nested = self.nested_dec(torch.cat([d1, e0_modulated], dim=1)) + d1

        final_illumination = self.out_conv(d1_nested)

        if self.predictive_adaptive_curve:
            adaptive_params = self.adaptive_curve_head(d1)
            alpha_map_raw, beta_map_raw = torch.chunk(adaptive_params, 2, dim=1)
            base_alpha, base_beta, scale = 0.4, 0.7, 0.1
            alpha_map = base_alpha + scale * torch.tanh(alpha_map_raw)
            beta_map = base_beta + scale * torch.tanh(beta_map_raw)
        else:
            alpha_map = None
            beta_map = None

        if self.learn_local_exposure:
            predicted_local_mean_map = self.predicted_local_mean_head(d1)
            predicted_local_mean_val = F.adaptive_avg_pool2d(predicted_local_mean_map, output_size=8)
        else:
            predicted_local_mean_val = None

        if self.enable_dynamic_weights:
            dynamic_weights = torch.exp(-self.log_vars)
        else:
            dynamic_weights = None

        confidence_map = self.confidence_head(d1_nested)
        log_variance_map = self.variance_head(d1_nested)

        return final_illumination, alpha_map, beta_map, predicted_local_mean_val, dynamic_weights, confidence_map, log_variance_map
