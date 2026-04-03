import os

import cv2
import torch
import numpy as np
import imageio.v2 as imageio
import tyro
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from scipy import stats
import torch.nn.functional as F

from config import Config
from retinex_temp import MultiScaleRetinexNet
from losses import (
    AdaptiveCurveLoss,
    EdgeAwareSmoothingLoss,
    WhitePreservationLoss,
    ExposureLoss,
    SpatialLoss,
    HistogramLoss,
    PerceptualColorLoss,
    ChromaLoss
)

def load_image(image_path: str, device: torch.device, max_size: int = 1024) -> torch.Tensor:
    img = imageio.imread(image_path)[..., :3]
    h, w = img.shape[:2]
    if h > w:
        new_h, new_w = max_size, int(max_size / (w / h))
    else:
        new_h, new_w = max_size, int(max_size / (h / w))

    img = img.astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img).unsqueeze(0).permute(0, 3, 1, 2).to(device)

    img_resized = F.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)

    return img_resized.permute(0, 2, 3, 1)

@torch.no_grad()
def output(cfg: Config, illumination_map: torch.Tensor, reflectance_map: torch.Tensor, step: int) -> None:
    print(f"Saving outputs for step {step}")
    ill_out = illumination_map.squeeze(0).permute(1, 2, 0).cpu().numpy()
    ref_out = reflectance_map.squeeze(0).permute(1, 2, 0).cpu().numpy()

    ill_out_vis = np.clip(ill_out, 0, 1)
    ref_out_vis = np.clip(ref_out, 0, 1)

    imageio.imwrite(os.path.join(cfg.result_dir, f"illumination_{step}.png"), (ill_out_vis * 255).astype(np.uint8))
    imageio.imwrite(os.path.join(cfg.result_dir, f"reflectance_{step}.png"), (ref_out_vis * 255).astype(np.uint8))


def main(cfg: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg.result_dir, exist_ok=True)

    print(f"Loading image from {cfg.data_dir}")
    pixels = load_image(cfg.data_dir, device)  # [B=1, H, W, 3]
    input_image_for_net = pixels.permute(0, 3, 1, 2)  # [B=1, 3, H, W]

    retinex_net = MultiScaleRetinexNet(
        in_channels=3,
        out_channels=3,
        embed_dim=cfg.retinex_embedding_dim,
    ).to(device)

    appearance_embeds = torch.nn.Embedding(1, cfg.retinex_embedding_dim).to(device)

    target_hist = torch.tensor(stats.norm.pdf(
        np.linspace(0, 1, 255), loc=0.5, scale=0.2
    ), dtype=torch.float32, device=device)
    target_histogram_dist = torch.nn.Parameter(target_hist)

    param_groups = [
        {"params": retinex_net.parameters(), "lr": cfg.retinex_opt_lr},
        {"params": appearance_embeds.parameters(), "lr": cfg.appearance_embedding_lr},
        {"params": target_histogram_dist, "lr": 1e-3}
    ]

    loss_adaptive_curve = AdaptiveCurveLoss(learn_lambdas=cfg.learn_adaptive_curve_lambdas).to(device)
    if cfg.learn_adaptive_curve_lambdas:
        param_groups.append({"params": loss_adaptive_curve.parameters(), "lr": 1e-3})

    optimizer = torch.optim.AdamW(param_groups, fused=True)
    scaler = torch.amp.GradScaler(enabled=True)

    loss_edge_aware_smooth = EdgeAwareSmoothingLoss().to(device)
    loss_white_preservation = WhitePreservationLoss(
        luminance_threshold=cfg.luminance_threshold,
        chroma_tolerance=cfg.chroma_tolerance,
        gain=cfg.gain,
    ).to(device)
    loss_exposure = ExposureLoss(patch_size=cfg.exposure_loss_patch_size, mean_val=cfg.exposure_mean_val).to(device)
    loss_spatial = SpatialLoss().to(device)
    histogram_loss = HistogramLoss().to(device)
    loss_chroma = ChromaLoss().to(device)
    loss_perceptual_colour = PerceptualColorLoss().to(device)

    print(f"Training for {cfg.max_steps} steps")
    pbar = tqdm(range(cfg.max_steps))

    for step in pbar:
        optimizer.zero_grad()

        with (torch.amp.autocast(device_type="cuda", enabled=True)):
            embed_ids = torch.zeros(1, dtype=torch.long, device=device)
            retinex_embedding = appearance_embeds(embed_ids)

            log_illumination_map = checkpoint(
                retinex_net,
                input_image_for_net,
                retinex_embedding,
                use_reentrant=False
            )
            illumination_map = torch.sigmoid(log_illumination_map)

            if not cfg.allow_chromatic_illumination:
                illumination_map = torch.mean(illumination_map, dim=1, keepdim=True).repeat(1, 3, 1, 1)

            reflectance_map = input_image_for_net / illumination_map
            reflectance_map = torch.clamp(reflectance_map, 0.0, 1.0).nan_to_num()

            total_loss = 0.0

            if cfg.loss_adaptive_curve:
                total_loss += cfg.lambda_illum_curve * loss_adaptive_curve(reflectance_map)

            if cfg.loss_exposure:
                total_loss += cfg.lambda_illum_exposure * loss_exposure(reflectance_map)

            if cfg.loss_reflectance_spa:
                total_loss += cfg.lambda_reflect * loss_spatial(input_image_for_net, reflectance_map, contrast=0.5)

            if cfg.loss_smooth_edge_aware:
                total_loss += cfg.lambda_edge_aware_smooth * loss_edge_aware_smooth(illumination_map, input_image_for_net)

            if cfg.loss_white_preservation:
                total_loss += cfg.lambda_white_preservation * loss_white_preservation(pixels, reflectance_map.permute(0, 2, 3, 1))

            if cfg.loss_histogram:
                total_loss += cfg.lambda_histogram * histogram_loss(reflectance_map, target_histogram_dist)

            if cfg.loss_perceptual_color:
                total_loss += cfg.lambda_perceptual_color * loss_perceptual_colour(reflectance_map.permute(0, 2, 3, 1), pixels)

            if cfg.loss_variance:
                illum_std = torch.std(illumination_map, dim=[2, 3])
                total_loss += cfg.lambda_illum_variance * (torch.mean(illum_std) + 1e-6)

            if cfg.loss_chroma:
                total_loss += cfg.lambda_chroma * loss_chroma(illumination_map)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 100 == 0:
            pbar.set_description(f"Loss: {total_loss.item():.4f}")

        if step % 500 == 0:
            output(cfg, illumination_map, reflectance_map, step)

    output(cfg, illumination_map, reflectance_map, cfg.max_steps)

if __name__ == "__main__":
    config = tyro.cli(Config)
    main(config)