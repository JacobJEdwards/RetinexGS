import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# --- NEW: ResidualBlock for a more powerful network backbone ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, padding_mode="replicate")
        self.bn1 = nn.InstanceNorm2d(out_channels)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.bn2 = nn.InstanceNorm2d(out_channels)

        # Shortcut connection to match dimensions if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.InstanceNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The residual connection
        out = self.relu(out)
        return out


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        scale = self.sigmoid(self.conv(x_cat))
        return x * scale


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ModulationHead(nn.Module):
    def __init__(self, feature_channels: int):
        super(ModulationHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, kernel_size=3, padding=1),
            nn.PReLU(),
            nn.Conv2d(feature_channels, feature_channels * 2, kernel_size=1)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.head(x)


class SpatiallyFiLMLayer(nn.Module):
    def __init__(self, feature_channels: int):
        super(SpatiallyFiLMLayer, self).__init__()
        self.modulation_head = ModulationHead(feature_channels)

    def forward(self, x: Tensor, conditioning_features: Tensor) -> Tensor:
        modulation_params = self.modulation_head(conditioning_features)
        gamma, beta = torch.chunk(modulation_params, 2, dim=1)

        if gamma.shape[2:] != x.shape[2:]:
            gamma = F.interpolate(gamma, size=x.shape[2:], mode='bilinear', align_corners=False)
        if beta.shape[2:] != x.shape[2:]:
            beta = F.interpolate(beta, size=x.shape[2:], mode='bilinear', align_corners=False)

        gated_gamma = torch.sigmoid(gamma)
        return gated_gamma * x + beta


class FiLMLayer(nn.Module):
    def __init__(self, embed_dim: int, feature_channels: int):
        super(FiLMLayer, self).__init__()
        # --- IMPROVEMENT: Using a small MLP for more expressive power ---
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, feature_channels),
            nn.PReLU(),
            nn.Linear(feature_channels, feature_channels * 2)
        )

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        gamma_beta = self.layer(embedding)
        gamma_beta = gamma_beta.view(gamma_beta.size(0), -1, 1, 1)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)

        return gamma * x + beta


class RefinementNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int):
        super(RefinementNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.InstanceNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.InstanceNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2)
        self.bn2 = nn.InstanceNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.InstanceNorm2d(128)

        self.film_bottleneck = FiLMLayer(embed_dim=embed_dim, feature_channels=128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.bn3 = nn.InstanceNorm2d(64)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn3_2 = nn.InstanceNorm2d(64)

        self.output_layer = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        self.relu = nn.PReLU()

    def forward(self, x: Tensor, embedding: Tensor) -> Tensor:
        e1 = self.relu(self.bn1(self.conv1(x)))
        e1 = self.relu(self.bn1_2(self.conv1_2(e1)))

        e2_pre_mod = self.relu(self.bn2(self.conv2(e1)))
        e2_pre_mod = self.relu(self.bn2_2(self.conv2_2(e2_pre_mod)))

        e2 = self.film_bottleneck(e2_pre_mod, embedding)

        d1 = self.relu(self.bn3(self.upconv1(e2)))
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)

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
        self.relu = nn.PReLU()

    def forward(self, x: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        c1 = self.relu(self.conv1(x))
        c1_modulated = self.film1(c1, embedding)
        p1 = self.pool(c1_modulated)
        c2 = self.relu(self.conv2(p1))
        p2 = self.pool(c2)
        up1 = self.upconv1(p2)
        up1_resized = F.interpolate(up1, size=p1.shape[2:], mode="bilinear", align_corners=False)
        merged = torch.cat([up1_resized, p1], dim=1)
        c3 = self.relu(self.conv3(merged))
        log_illumination = self.upconv2(c3)
        final_illumination = F.interpolate(log_illumination, size=x.shape[2:], mode="bilinear", align_corners=False)
        return final_illumination, None, None, None, None


class MultiScaleRetinexNet(nn.Module):
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            embed_dim: int = 32,
            use_refinement: bool = False,
            use_dilated_convs: bool = False,
            predictive_adaptive_curve: bool = False,
            spatially_film: bool = False,
            use_se_blocks: bool = False,
            enable_dynamic_weights: bool = False,
            use_spatial_attention: bool = False,
            use_pixel_shuffle: bool = False,
            use_stride_conv: bool = False,
            num_weight_scales: int = 11,
    ) -> None:
        super(MultiScaleRetinexNet, self).__init__()

        self.use_pixel_shuffle = use_pixel_shuffle

        # --- IMPROVEMENT: Replacing plain convs with ResidualBlocks ---
        self.layer1 = ResidualBlock(in_channels, 32)
        self.layer2 = ResidualBlock(32, 64, stride=2) # Downsample
        self.bottleneck = ResidualBlock(64, 64) # Bottleneck layer

        self.spatially_film = spatially_film
        if spatially_film:
            self.spatial_conditioning_encoder = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
                nn.PReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
                nn.PReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, padding_mode="replicate"),
                nn.PReLU(),
            )
            self.film_spatial = SpatiallyFiLMLayer(feature_channels=64)
        else:
            self.film_global = FiLMLayer(embed_dim=embed_dim, feature_channels=32)

        self.enable_dynamic_weights = enable_dynamic_weights
        if self.enable_dynamic_weights:
            self.log_vars = nn.Parameter(torch.zeros(num_weight_scales, dtype=torch.float32))

        self.use_dilated_convs = use_dilated_convs
        if self.use_dilated_convs:
            self.dilated_block = ResidualBlock(64, 64)

        self.use_se_blocks = use_se_blocks
        if self.use_se_blocks:
            self.se1 = SEBlock(32)
            self.se2 = SEBlock(64)
            self.se3 = SEBlock(32)

        self.use_spatial_attention = use_spatial_attention
        if self.use_spatial_attention:
            self.spatial_att1 = SpatialAttentionModule()
            self.spatial_att2 = SpatialAttentionModule()
            self.spatial_att3 = SpatialAttentionModule()

        if self.use_pixel_shuffle:
            self.upconv1 = nn.Sequential(
                nn.Conv2d(64, 32 * (2**2), kernel_size=3, padding=1, padding_mode="replicate"),
                nn.PixelShuffle(2),
                nn.Conv2d(32, 32, kernel_size=1, padding_mode="replicate")
            )
            self.upconv2 = nn.Sequential(
                nn.Conv2d(32, out_channels * (2**2), kernel_size=3, padding=1, padding_mode="replicate"),
                nn.PixelShuffle(2),
                nn.Conv2d(out_channels, out_channels, kernel_size=1, padding_mode="replicate")
            )
        else:
            self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
            self.upconv2 = nn.ConvTranspose2d(32, out_channels, kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 32, kernel_size=3, padding=1, padding_mode="replicate")
        self.bn3 = nn.InstanceNorm2d(32)

        self.output_head_medium = nn.Conv2d(32, out_channels, kernel_size=3, padding=1, padding_mode="replicate")
        self.output_head_coarse = nn.Conv2d(64, out_channels, kernel_size=3, padding=1, padding_mode="replicate")

        self.combination_layer = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.use_refinement = use_refinement
        if self.use_refinement:
            self.refinement_net = RefinementNet(out_channels, out_channels, embed_dim)

        self.relu = nn.PReLU()

        self.predictive_adaptive_curve = predictive_adaptive_curve
        if self.predictive_adaptive_curve:
            self.adaptive_curve_head = nn.Sequential(
                nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
                nn.PReLU(),
                nn.Conv2d(32, 2, kernel_size=1),
            )

        self.predictive_local_exposure_mean = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, padding_mode="replicate"),
            nn.PReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: Tensor, embedding: Tensor) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        # --- Encoder with ResidualBlocks ---
        c1 = self.layer1(x) # [B, 32, H, W]
        if not self.spatially_film:
            c1 = self.film_global(c1, embedding)
        if self.use_se_blocks: c1 = self.se1(c1)
        if self.use_spatial_attention: c1 = self.spatial_att1(c1)

        c2 = self.layer2(c1) # [B, 64, H/2, W/2]

        # --- Bottleneck ---
        bottleneck = self.bottleneck(c2) # [B, 64, H/2, W/2]

        if self.spatially_film:
            spatial_cond_features = self.spatial_conditioning_encoder(x)
            if spatial_cond_features.shape[2:] != bottleneck.shape[2:]:
                aligned_cond_features = F.interpolate(spatial_cond_features, size=bottleneck.shape[2:], mode='bilinear', align_corners=False)
            else:
                aligned_cond_features = spatial_cond_features
            bottleneck = self.film_spatial(bottleneck, aligned_cond_features)

        if self.use_dilated_convs:
            bottleneck = self.dilated_block(bottleneck)
        if self.use_se_blocks: bottleneck = self.se2(bottleneck)
        if self.use_spatial_attention: bottleneck = self.spatial_att2(bottleneck)

        p2 = bottleneck # The output of the encoder/bottleneck is now p2 for the decoder

        # --- Decoder ---
        up1 = self.upconv1(p2)
        up1 = F.interpolate(up1, size=c1.shape[2:], mode="bilinear", align_corners=False)
        merged = torch.cat([up1, c1], dim=1) # Skip connection from c1
        c3 = self.relu(self.bn3(self.conv3(merged)))

        if self.use_se_blocks: c3 = self.se3(c3)
        if self.use_spatial_attention: c3 = self.spatial_att3(c3)

        log_illumination_full_res = self.upconv2(c3)
        log_illumination_full_res = F.interpolate(log_illumination_full_res, size=x.shape[2:], mode="bilinear", align_corners=False)

        log_illumination_medium_res = self.output_head_medium(c3)
        log_illumination_medium_res = F.interpolate(log_illumination_medium_res, size=x.shape[2:], mode="bilinear", align_corners=False)

        log_illum_coarse_res = self.output_head_coarse(p2)
        log_illum_coarse_res = F.interpolate(log_illum_coarse_res, size=x.shape[2:], mode="bilinear", align_corners=False)

        concatenated_maps = (log_illum_coarse_res + log_illumination_medium_res + log_illumination_full_res)
        final_illumination = self.combination_layer(concatenated_maps)

        if self.use_refinement:
            final_illumination = self.refinement_net(final_illumination, embedding)

        alpha_map, beta_map = None, None
        if self.predictive_adaptive_curve:
            adaptive_params = self.adaptive_curve_head(c3)
            adaptive_params = F.interpolate(adaptive_params, size=x.shape[2:], mode="bilinear", align_corners=False)
            alpha_map_raw, beta_map_raw = torch.chunk(adaptive_params, 2, dim=1)
            alpha_map = 0.5 + 1.0 * torch.sigmoid(alpha_map_raw)
            beta_map = 0.1 + 0.8 * torch.sigmoid(beta_map_raw)

        dynamic_weights = torch.exp(-self.log_vars) if self.enable_dynamic_weights else None

        predicted_local_mean_val = self.predictive_local_exposure_mean(c3)
        predicted_local_mean_val = F.interpolate(predicted_local_mean_val, size=x.shape[2:], mode="bilinear", align_corners=False)
        predicted_local_mean_val = F.adaptive_avg_pool2d(predicted_local_mean_val, output_size=8)

        return final_illumination, alpha_map, beta_map, predicted_local_mean_val, dynamic_weights