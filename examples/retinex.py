import math

import torch.nn as nn
from torch import Tensor
from torch.nn import init
import torch
import torch.nn.functional as F


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
        self.conv = DepthwiseSeparableConv(in_channels, out_channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.norm = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act = nn.SiLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.shuffle(self.conv(x))))


class ECALayer(nn.Module):
    def __init__(self, channel: int, gamma: float = 2, b: float = 1):
        super().__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        y = self.avg_pool(x)
        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = DepthwiseSeparableConv(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale


class SimpleSSM(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.fc_skip = nn.Linear(d_model, d_model)
        self.fc_x = nn.Linear(d_model, d_state)
        self.fc_dt = nn.Linear(d_model, d_state)
        self.fc_out = nn.Linear(d_model, d_model)
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, 1))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (b, seq, d_model)
        skip = self.fc_skip(x)
        dt = F.softplus(self.fc_dt(x))  # (b, seq, d_state)
        delta = dt.unsqueeze(-1)  # (b, seq, d_state, 1)
        A_bar = torch.exp(delta * self.A)  # (b, seq, d_state, d_state)
        B_bar = delta * self.B  # (b, seq, d_state, 1)
        x_proj = self.fc_x(x).unsqueeze(-1)  # (b, seq, d_state, 1)
        h = torch.zeros(x.shape[0], self.d_state, 1, device=x.device)  # (b, d_state, 1)
        ys = []
        for t in range(x.shape[1]):
            h = torch.matmul(A_bar[:, t], h) + B_bar[:, t] * x_proj[:, t]
            y_t = torch.matmul(self.C, h).squeeze(-1) + self.D * x[:, t]
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        y = self.fc_out(y)
        y = y + skip
        return y

class SSMBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.ssm = SimpleSSM(channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        b, c, h, w = x.shape
        seq = x.flatten(2).transpose(1, 2)  # (b, h*w, c)
        seq = self.norm(seq)
        out_fwd = self.ssm(seq)
        seq_rev = torch.flip(seq, dims=[1])
        out_rev = self.ssm(seq_rev)
        out_rev = torch.flip(out_rev, dims=[1])
        out = (out_fwd + out_rev) / 2
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return out

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

class CBAM(nn.Module):
    def __init__(self, channel: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = SEBlock(channel, reduction)
        self.spatial_attention = SpatialAttentionModule(kernel_size)

    def forward(self, x: Tensor) -> Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

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


class RefinementNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super(RefinementNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn1_2 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
        self.bn2 = nn.InstanceNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=2, padding_mode='replicate')
        self.bn2_2 = nn.InstanceNorm2d(128)

        self.film_bottleneck = FiLMLayer(embed_dim=embed_dim, feature_channels=128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn3 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn3_2 = nn.InstanceNorm2d(64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, padding_mode='replicate')

        self.relu = nn.SiLU()

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        e1 = self.relu(self.bn1(self.conv1(x)))
        e1 = self.relu(self.bn1_2(self.conv1_2(e1)))
        e2_pre_mod = self.relu(self.bn2(self.conv2(e1)))
        e2_pre_mod = self.relu(self.bn2_2(self.conv2_2(e2_pre_mod)))

        e2 = self.film_bottleneck(e2_pre_mod, embedding)

        d1 = self.relu(self.bn3(self.upconv1(e2)))
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(
                d1, size=e1.shape[2:], mode="bilinear", align_corners=False
            )

        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.relu(self.bn3_2(self.conv3(d1)))

        output = self.output_layer(d1)
        return output


class RetinexNet(nn.Module):
    def __init__(
            self, in_channels: int = 3, out_channels: int = 1, embed_dim: int = 32
    ) -> None:
        super(RetinexNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)

        self.film1 = FiLMLayer(embed_dim=embed_dim, feature_channels=32)

        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        self.relu = nn.SiLU()

    def forward(self, x: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
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

        return final_illumination, None, None, None, None


class MultiScaleRetinexNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            embed_dim: int = 32,
            enable_dynamic_weights: bool = False,
            predictive_adaptive_curve: bool = False,
            learn_local_exposure: bool = True,
            num_weight_scales: int = 11,
    ) -> None:
        super(MultiScaleRetinexNet, self).__init__()
        self.in_conv = RetinexBlock(in_channels, 16)
        self.film1 = SpatiallyFiLMLayer(embed_dim=embed_dim, feature_channels=16)

        self.enc1 = RetinexBlock(16, 32, stride=2)
        self.enc2 = RetinexBlock(32, 64, stride=2)

        self.bottleneck = nn.Sequential(
            RetinexBlock(64, 64),
            SSMBlock(64),
            ECALayer(64)
        )

        self.dec2 = UpBlock(64, 32)
        self.dec2_conv = RetinexBlock(64, 32)

        self.dec1 = UpBlock(32, 16)
        self.dec1_conv = RetinexBlock(32, 16)

        self.out_conv = DepthwiseSeparableConv(16, out_channels, kernel_size=3, padding=1)

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
    Tensor | None]:
        e0 = self.in_conv(x)
        e0_modulated = self.film1(e0, embedding)

        e1 = self.enc1(e0_modulated)
        e2 = self.enc2(e1)

        b = self.bottleneck(e2)

        d2_up = self.dec2(b)
        if d2_up.shape[2:] != e1.shape[2:]:
            d2_up = F.interpolate(d2_up, size=e1.shape[2:], mode='bilinear', align_corners=False)

        d2 = torch.cat([d2_up, e1], dim=1)
        d2 = self.dec2_conv(d2)

        d1_up = self.dec1(d2)
        d1 = torch.cat([d1_up, e0_modulated], dim=1)
        d1 = self.dec1_conv(d1)

        final_illumination = self.out_conv(d1)

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

        return final_illumination, alpha_map, beta_map, predicted_local_mean_val, dynamic_weights
