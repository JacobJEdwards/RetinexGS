import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import imageio
import kornia
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, ChainedScheduler, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from config import Config
from lib_bilagrid import color_correct, BilateralGrid, bi_slice
from utils import CameraResponseNet
from rendering_intrinsics import rasterize_intrinsics
from losses import TotalVariationLoss, GeometryAwareSmoothingLoss, GrayWorldLoss, LogTotalVariationLoss, ChromaticityContinuityLoss
from losses import ExclusionLoss, PatchConsistencyLoss
from utils import IlluminationField, sh_to_rgb, quaternion_to_matrix
from rendering_double import rasterization_dual
from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from utils import (
    knn,
    rgb_to_sh,
    set_random_seed,
)


def create_splats_with_optimizers(
        parser: Parser,
        init_type: str = "sfm",
        init_num_pts: int = 100_000,
        init_extent: float = 3.0,
        init_opacity: float = 0.1,
        init_scale: float = 1.0,
        means_lr: float = 1.6e-4,
        scales_lr: float = 5e-3,
        opacities_lr: float = 5e-2,
        quats_lr: float = 1e-3,
        sh0_lr: float = 2.5e-3,
        shN_lr: float = 2.5e-3 / 20,
        scene_scale: float = 1.0,
        sh_degree: int = 3,
        sparse_grad: bool = False,
        visible_adam: bool = False,
        batch_size: int = 1,
        feature_dim: int | None = None,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
) -> tuple[
    torch.nn.ParameterDict,
    dict[str, torch.optim.Adam | torch.optim.SparseAdam | SelectiveAdam],
]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    dist2_avg = (knn(points)[:, 1:] ** 2).mean(dim=-1)
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)

    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))
    opacities = torch.logit(torch.full((N,), init_opacity))

    params = [
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
    ]

    if feature_dim is None:
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), sh0_lr))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), shN_lr))
    else:
        features = torch.rand(N, feature_dim)
        params.append(("features", torch.nn.Parameter(features), sh0_lr))
        colors = torch.logit(rgbs)
        params.append(("colors", torch.nn.Parameter(colors), sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = batch_size * world_size

    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }

    return splats, optimizers


class Runner:
    def __init__(
            self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        self.start_time = time.time()
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        cfg.result_dir.mkdir(exist_ok=True, parents=True)
        self.ckpt_dir = cfg.result_dir / "ckpts"
        self.ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.stats_dir = cfg.result_dir / "stats"
        self.stats_dir.mkdir(exist_ok=True, parents=True)
        self.render_dir = cfg.result_dir / "renders"
        self.render_dir.mkdir(exist_ok=True, parents=True)
        self.ply_dir = cfg.result_dir / "ply"
        self.ply_dir.mkdir(exist_ok=True, parents=True)

        self.writer = SummaryWriter(log_dir=str(cfg.result_dir / "tb"))

        self.parser = Parser(
            data_dir=str(cfg.data_dir),
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(self.parser, patch_size=cfg.patch_size)
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.illumination_field = IlluminationField(
            self.scene_scale,
            use_view_dirs=cfg.use_view_dirs,
            use_normals=cfg.use_normals,
        ).to(self.device)

        if world_size > 1:
            self.illumination_field = DDP(
                self.illumination_field, device_ids=[local_rank]
            )

        # Removed retinex_net from the optimizer
        self.illum_field_optimizer = torch.optim.Adam(
            self.illumination_field.parameters(), lr=cfg.illumination_field_lr, fused=True
        )

        if cfg.use_bilateral_grid:
            self.bilateral_grid = BilateralGrid(len(self.trainset)).to(self.device)
            self.bilateral_grid_optimizer = torch.optim.Adam(
                self.bilateral_grid.parameters(), lr=2e-3, fused=True
            )
        else:
            self.bilateral_grid = None

        if cfg.use_camera_response_network:
            self.camera_response_net = CameraResponseNet(
                embedding_dim=cfg.appearance_embedding_dim
            ).to(self.device)
            self.camera_response_optimizer = torch.optim.Adam(
                self.camera_response_net.parameters(), lr=cfg.camera_net_lr, fused=True
            )

            num_train_images = len(self.trainset)
            self.appearance_embeds = torch.nn.Embedding(
                num_train_images, cfg.appearance_embedding_dim
            ).to(self.device)
            self.appearance_embeds_optimizer = torch.optim.Adam(
                self.appearance_embeds.parameters(), lr=cfg.appearance_embedding_lr, fused=True
            )

        self.loss_tv = TotalVariationLoss().to(self.device)
        self.loss_geometry_smooth = GeometryAwareSmoothingLoss().to(self.device)

        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            means_lr=cfg.means_lr,
            scales_lr=cfg.scales_lr,
            opacities_lr=cfg.opacities_lr,
            quats_lr=cfg.quats_lr,
            sh0_lr=cfg.sh0_lr,
            shN_lr=cfg.shN_lr,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            batch_size=cfg.batch_size,
            feature_dim=None,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(
                self.device
            )
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(
                self.device
            )

        n_gauss = len(self.splats["means"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers: list[ExponentialLR | ChainedScheduler | CosineAnnealingLR] = [
            ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
            CosineAnnealingLR(self.illum_field_optimizer, T_max=max_steps, eta_min=0),
        ]

        if cfg.use_camera_response_network:
            schedulers.append(
                CosineAnnealingLR(
                    self.camera_response_optimizer, T_max=max_steps, eta_min=0
                )
            )

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=False,
        )
        trainloader_iter = iter(trainloader)
        global_tic = time.time()
        pbar = tqdm.tqdm(range(init_step, max_steps))

        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            sh_degree_to_use = min(
                step // cfg.sh_degree_interval, cfg.sh_degree
            )

            # Initialize with lighting frozen
            if step == 0:
                if cfg.use_camera_response_network:
                    for param in self.camera_response_net.parameters():
                        param.requires_grad = False
                    for param in self.appearance_embeds.parameters():
                        param.requires_grad = False
                for param in self.illumination_field.parameters():
                    param.requires_grad = False

            # Unfreeze lighting networks AFTER Gaussians have formed a base structure
            if step == 1500:
                if cfg.use_camera_response_network:
                    for param in self.camera_response_net.parameters():
                        param.requires_grad = True
                    for param in self.appearance_embeds.parameters():
                        param.requires_grad = True
                for param in self.illumination_field.parameters():
                    param.requires_grad = True

            with torch.autocast(enabled=False, device_type=device):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                height, width = pixels.shape[1:3]
                pixels = torch.clamp(pixels, 0.0, 1.0)

                base_reflectance_sh = torch.cat(
                    [self.splats["sh0"], self.splats["shN"]], 1
                )

                intrinsic_maps, _, info = rasterize_intrinsics(
                    means=self.splats["means"],
                    quats=self.splats["quats"],
                    scales=torch.exp(self.splats["scales"]),
                    opacities=torch.sigmoid(self.splats["opacities"]),
                    base_reflectance_sh=base_reflectance_sh,
                    viewmats=torch.linalg.inv(camtoworlds.float()),
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                )

                reflectance_map = intrinsic_maps["reflectance"]
                world_position_map = intrinsic_maps["world_position"]
                world_normal_map = intrinsic_maps["world_normal"]

                view_dirs_input = None
                if cfg.use_view_dirs:
                    cam_origin = camtoworlds.squeeze(0)[:3, 3]
                    view_dirs_input = world_position_map.view(-1, 3) - cam_origin

                # 1. Local 3D Illumination (Shadows/Bounces)
                illum_A, illum_b = self.illumination_field(
                    world_position_map.view(-1, 3),
                    view_dirs=view_dirs_input,
                    normals=world_normal_map.view(-1, 3),
                )
                illum_A_map = illum_A.view(1, height, width, 3, 3)
                illum_b_map = illum_b.view(1, height, width, 3)

                # Extract diagonal and use softplus to ensure light is strictly positive
                illum_scale = F.softplus(torch.diagonal(illum_A_map, dim1=-2, dim2=-1))

                # Apply local lighting: I = R * L + b
                scene_lit_color_map = (reflectance_map * illum_scale) + illum_b_map

                # 2. Global Camera Response (Exposure / White Balance)
                if cfg.use_camera_response_network:
                    image_ids = data["image_id"].to(device)
                    appearance_embedding = self.appearance_embeds(image_ids)
                    c, d = self.camera_response_net(appearance_embedding)
                    final_color_map = (
                            c[:, None, None, :] * scene_lit_color_map + d[:, None, None, :]
                    )
                else:
                    final_color_map = scene_lit_color_map

                if cfg.use_bilateral_grid:
                    grid_y, grid_x = torch.meshgrid(
                        torch.linspace(0, 1, height, device=device),
                        torch.linspace(0, 1, width, device=device),
                        indexing="ij"
                    )
                    grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(cfg.batch_size, -1, -1, -1)
                    grid_out = bi_slice(self.bilateral_grid, grid_xy, reflectance_map, image_ids.view(-1, 1))
                    colors_low = grid_out["rgb"]
                else:
                    colors_low = torch.clamp(final_color_map, 0.0, 1.0)

                # Illumination map for visualization
                gray_color = torch.full_like(reflectance_map, 0.5)
                illum_color_map = (gray_color * illum_scale) + illum_b_map
                illum_map = torch.clamp(illum_color_map, 0.0, 1.0)

                info["means2d"].retain_grad()

                loss_reconstruct_low = F.l1_loss(colors_low, pixels)
                ssim_loss_low = 1.0 - self.ssim(colors_low.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
                loss = (1.0 - cfg.ssim_lambda) * loss_reconstruct_low + cfg.ssim_lambda * ssim_loss_low

                # --- NEW DISENTANGLEMENT PRIORS ---

                # 1. Achromatic Prior: Penalize RGB variance in the illumination scale.
                # This forces the lighting to primarily affect brightness (shadows) rather than color.
                illum_scale_mean = illum_scale.mean(dim=-1, keepdim=True)
                loss += 0.1 * F.mse_loss(illum_scale, illum_scale_mean.expand_as(illum_scale))

                # 2. Smoothness Prior: Force local lighting to be spatially smooth.
                # Because the MLP is forced to be smooth, sharp textures MUST be learned by the splats.
                loss += 0.05 * self.loss_geometry_smooth(illum_scale.permute(0, 3, 1, 2), world_normal_map.permute(0, 3, 1, 2))

                # 3. Bias Suppression: Force the network to rely on the multiplication scale, not the additive bias
                loss += 0.05 * torch.mean(illum_b_map ** 2)

                # 4. Camera Regularization
                if cfg.use_camera_response_network:
                    loss += 0.01 * torch.mean(appearance_embedding ** 2)
                    loss += 0.1 * (torch.mean((c - 1.0)**2) + torch.mean(d**2))

                if cfg.lambda_shn_reg > 0.0:
                    loss += cfg.lambda_shn_reg * self.splats["shN"].pow(2).mean()

                self.cfg.strategy.step_pre_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info)

            loss.backward()
            pbar.set_description(f"loss={loss.item():.3f} sh_deg={sh_degree_to_use}")

            if world_rank == 0 and step % cfg.tb_every == 0:
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                # Removed retinex_net from payload
                data_save = {"step": step, "splats": self.splats.state_dict(), "illumination_field": self.illumination_field.state_dict()}
                torch.save(data_save, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt")

            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad()
            self.illum_field_optimizer.step()
            self.illum_field_optimizer.zero_grad()
            if cfg.use_camera_response_network:
                self.appearance_embeds_optimizer.step()
                self.appearance_embeds_optimizer.zero_grad()
                self.camera_response_optimizer.step()
                self.camera_response_optimizer.zero_grad()
            if cfg.use_bilateral_grid:
                self.bilateral_grid_optimizer.step()
                self.bilateral_grid_optimizer.zero_grad()

            for scheduler in schedulers: scheduler.step()
            self.cfg.strategy.step_post_backward(params=self.splats, optimizers=self.optimizers, state=self.strategy_state, step=step, info=info)

            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                self.render_traj(step)

    def rasterize_splats(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            masks: Tensor | None = None,
            **kwargs,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        rasterize_mode: Literal["antialiased", "classic"] = "classic"
        colors_enh = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        (render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info) = rasterization_dual(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors_enh, colors_low=colors_enh,
            viewmats=torch.linalg.inv(camtoworlds.float()), Ks=Ks, width=width, height=height, packed=False,
            absgrad=(self.cfg.strategy.absgrad if isinstance(self.cfg.strategy, DefaultStrategy) else False),
            rasterize_mode=rasterize_mode, **kwargs,
        )
        return (render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        print(f"Running evaluation for step {step} on '{stage}' set...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, shuffle=False, num_workers=1
        )
        ellipse_time_total = 0
        metrics = defaultdict(list)
        for i, data in enumerate(tqdm.tqdm(valloader, desc=f"Eval {stage}")):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)

            pixels = data["image"].to(device) / 255.0

            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            base_reflectance_sh = torch.cat(
                [self.splats["sh0"], self.splats["shN"]], 1
            )
            intrinsic_maps, _, info = rasterize_intrinsics(
                means=self.splats["means"],
                quats=self.splats["quats"],
                scales=torch.exp(self.splats["scales"]),
                opacities=torch.sigmoid(self.splats["opacities"]),
                base_reflectance_sh=base_reflectance_sh,
                viewmats=torch.linalg.inv(camtoworlds.float()),
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
            )
            reflectance_map = intrinsic_maps["reflectance"]
            world_position_map = intrinsic_maps["world_position"]
            world_normal_map = intrinsic_maps["world_normal"]
            points_3d_world_flat = world_position_map.view(-1, 3)
            world_normals_flat = world_normal_map.view(-1, 3)

            view_dirs_input = None
            if cfg.use_view_dirs:
                cam_origin = camtoworlds.squeeze(0)[:3, 3]
                view_dirs_input = points_3d_world_flat - cam_origin

            illum_A, illum_b = self.illumination_field(
                points_3d_world_flat,
                view_dirs=view_dirs_input,
                normals=world_normals_flat,
            )
            illum_A_map = illum_A.view(1, height, width, 3, 3)
            illum_b_map = illum_b.view(1, height, width, 3)

            # Use softplus diagonal to maintain physics bounds
            illum_scale = F.softplus(torch.diagonal(illum_A_map, dim1=-2, dim2=-1))
            scene_lit_color_map = (reflectance_map * illum_scale) + illum_b_map

            if cfg.use_camera_response_network:
                image_ids = data["image_id"].to(device)
                embedding = self.appearance_embeds(image_ids)
                c, d = self.camera_response_net(embedding)
                final_color_map = (
                        c[:, None, None, :] * scene_lit_color_map + d[:, None, None, :]
                )
            else:
                final_color_map = scene_lit_color_map

            # Direct sRGB output clamps
            colors_low = torch.clamp(final_color_map, 0.0, 1.0)
            colors_enh = torch.clamp(reflectance_map, 0.0, 1.0)

            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)

            if world_rank == 0:
                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors_low.permute(0, 3, 1, 2)
                colors_enh_p = colors_enh.permute(0, 3, 1, 2)
                orig_name = data["image_name"][0]
                orig_stem = os.path.splitext(orig_name)[0]
                orig_stem = orig_stem.replace("/", "_").replace("\\", "_")

                if cfg.save_images:
                    canvas_list_low = [pixels, colors_low]

                    canvas_eval_low = (
                        torch.cat(canvas_list_low, dim=2).squeeze(0).cpu().numpy()
                    )
                    canvas_eval_low = (canvas_eval_low * 255).astype(np.uint8)

                    canvas_list_enh = [pixels, colors_enh]
                    canvas_eval_enh = (
                        torch.cat(canvas_list_enh, dim=2).squeeze(0).cpu().numpy()
                    )

                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_low_{i:04d}_{orig_stem}.png",
                        canvas_eval_low,
                    )

                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_step{step}_enh_{i:04d}_{orig_stem}.png",
                        (canvas_eval_enh * 255).astype(np.uint8),
                    )

                    colors_low_np = colors_low.squeeze(0).cpu().numpy()
                    colors_enh_np = colors_enh.squeeze(0).cpu().numpy()

                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_low_{i:04d}_{orig_stem}.png",
                        (colors_low_np * 255).astype(np.uint8),
                    )

                    imageio.imwrite(
                        f"{self.render_dir}/{stage}_enh_{i:04d}_{orig_stem}.png",
                        (colors_enh_np * 255).astype(np.uint8),
                    )

                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                # Track unlit enhancements to verify splat quality directly
                metrics["psnr_enh"].append(self.psnr(colors_enh_p, pixels_p))
                metrics["ssim_enh"].append(self.ssim(colors_enh_p, pixels_p))
                metrics["lpips_enh"].append(self.lpips(colors_enh_p, pixels_p))

        if world_rank == 0:
            avg_ellipse_time = (
                ellipse_time_total / len(valloader) if len(valloader) > 0 else 0
            )

            stats_eval = {}
            for k, v_list in metrics.items():
                if v_list:
                    if isinstance(v_list[0], torch.Tensor):
                        stats_eval[k] = torch.stack(v_list).mean().item()
                    else:
                        stats_eval[k] = sum(v_list) / len(v_list)
                else:
                    stats_eval[k] = 0

            stats_eval.update(
                {
                    "ellipse_time": avg_ellipse_time,
                    "num_GS": len(self.splats["means"]),
                    "total_time": time.time() - self.start_time,
                }
            )

            print_parts_eval = [
                f"PSNR: {stats_eval.get('psnr', 0):.3f}",
                f"SSIM: {stats_eval.get('ssim', 0):.4f}",
                f"LPIPS: {stats_eval.get('lpips', 0):.3f}",
                f"PSNR Enh: {stats_eval.get('psnr_enh', 0):.3f}",
                f"SSIM Enh: {stats_eval.get('ssim_enh', 0):.4f}",
                f"LPIPS Enh: {stats_eval.get('lpips_enh', 0):.3f}",
                f"Time: {stats_eval.get('ellipse_time', 0):.3f}s/image",
                f"GS: {stats_eval.get('num_GS', 0)}",
            ]

            print(f"Eval {stage} Step {step}: " + " | ".join(print_parts_eval))

            raw_metrics = {}
            for k, v_list in metrics.items():
                if v_list:
                    if isinstance(v_list[0], torch.Tensor):
                        raw_metrics[k] = [v.item() for v in v_list]
                    else:
                        raw_metrics[k] = v_list
                else:
                    raw_metrics[k] = []

            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats_eval, f)
            for k_stat, v_stat in stats_eval.items():
                self.writer.add_scalar(f"{stage}/{k_stat}", v_stat, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        if self.cfg.disable_video:
            return
        if self.world_rank != 0:
            return

        print(f"Running trajectory rendering for step {step}...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all_np = self.parser.camtoworlds
        if not len(camtoworlds_all_np):
            print("No camera poses found for trajectory rendering. Skipping.")
            return

        if cfg.render_traj_path == "interp":
            camtoworlds_all_np = generate_interpolated_path(camtoworlds_all_np, 1)
        elif cfg.render_traj_path == "ellipse":
            height_mean = camtoworlds_all_np[:, 2, 3].mean()
            camtoworlds_all_np = generate_ellipse_path_z(
                camtoworlds_all_np, height=height_mean
            )
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all_np = generate_spiral_path(
                camtoworlds_all_np,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf.get("spiral_radius_scale", 0.5),
            )
        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all_np = np.concatenate(
            [
                camtoworlds_all_np,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all_np), axis=0
                ),
            ],
            axis=1,
        )

        camtoworlds_all_torch = torch.from_numpy(camtoworlds_all_np).float().to(device)

        first_val_cam_key = (
            list(self.parser.Ks_dict.keys())[0] if self.parser.Ks_dict else None
        )
        if not first_val_cam_key:
            print("No camera intrinsics found for trajectory rendering. Skipping.")
            return

        K_traj = (
            torch.from_numpy(self.parser.Ks_dict[first_val_cam_key]).float().to(device)
        )
        width_traj, height_traj = self.parser.imsize_dict[first_val_cam_key]

        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)

        video_path_color = f"{video_dir}/traj_{step}_color.mp4"
        video_path_depth = f"{video_dir}/traj_{step}_depth.mp4"
        video_writer_color = imageio.get_writer(video_path_color, fps=30)
        video_writer_depth = imageio.get_writer(video_path_depth, fps=30)

        for i in tqdm.trange(len(camtoworlds_all_torch), desc="Rendering trajectory"):
            cam_c2w = camtoworlds_all_torch[i : i + 1]
            cam_K = K_traj[None]

            out = self.rasterize_splats(
                camtoworlds=cam_c2w,
                Ks=cam_K,
                width=width_traj,
                height=height_traj,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )

            renders_traj, _, _, _, _ = out

            # Output is natively sRGB, no linear scaling required
            colors_traj = torch.clamp(renders_traj[..., 0:3], 0.0, 1.0)

            depths_traj = renders_traj[..., 3:4]
            depths_traj_norm = (depths_traj - depths_traj.min()) / (
                    depths_traj.max() - depths_traj.min() + 1e-10
            )

            canvas_color = colors_traj.squeeze(0).cpu().numpy()
            canvas_color_uint8 = (canvas_color * 255).astype(np.uint8)

            canvas_depth = depths_traj_norm.repeat(1, 1, 1, 3).squeeze(0).cpu().numpy()
            canvas_depth_uint8 = (canvas_depth * 255).astype(np.uint8)

            video_writer_color.append_data(canvas_color_uint8)
            video_writer_depth.append_data(canvas_depth_uint8)

        video_writer_color.close()
        video_writer_depth.close()
        print(f"Videos saved to {video_path_color} and {video_path_depth}")

        self.writer.flush()


def objective(trial: optuna.Trial):
    cfg = Config()

    cfg.lambda_illum_smoothness = trial.suggest_float("lambda_illum_smoothness", 1e-5, 1.0, log=True)
    cfg.lambda_tv_loss = trial.suggest_categorical(
        "lambda_tv_loss", [0, 100, 250, 500, 750, 1000, 2000]
    )
    cfg.lambda_shn_reg = trial.suggest_float("lambda_shn_reg", 0.1, 2.0, log=True)
    cfg.lambda_exclusion = trial.suggest_float("lambda_exclusion", 0.0, 0.3)
    cfg.appearance_embedding_lr = trial.suggest_float(
        "appearance_embedding_lr", 1e-5, 6e-3, log=True
    )
    cfg.illumination_field_lr = trial.suggest_float(
        "illumination_field_lr", 1e-6, 1e-2, log=True
    )
    cfg.camera_net_lr = trial.suggest_float(
        "camera_net_lr", 1e-5, 6e-3, log=True
    )
    cfg.appearance_embedding_dim = trial.suggest_categorical(
        "appearance_embedding_dim", [16, 32, 64, 128]
    )
    cfg.uncertainty_weighting = trial.suggest_categorical(
        "uncertainty_weighting", [True, False]
    )

    cfg.max_steps = 3000
    cfg.eval_steps = [3000]

    total_psnr = 0
    total_ssim = 0
    total_lpips = 0

    count = 0
    configs = [
        (Path("/workspace/360_v2/room"), "_multiexposure"),
        (Path("/workspace/360_v2/counter"), "_multiexposure"),
        (Path("/workspace/360_v2/kitchen"), "_contrast"),
        (Path("/workspace/360_v2/stump"), "_contrast"),
        (Path("/workspace/360_v2/counter"), "_variance"),
        (Path("/workspace/360_v2/stump"), "_variance"),
    ]

    for (datadir, postfix) in configs:
        cfg.postfix = postfix
        cfg.data_dir = datadir
        try:
            runner = Runner(0, 0, 1, cfg)
            runner.train()

            with open(f"{runner.stats_dir}/val_step{cfg.max_steps-1:04d}.json") as f:
                stats = json.load(f)

            psnr = stats.get("psnr_enh", 0)
            ssim = stats.get("ssim_enh", 0)
            lpips = stats.get("lpips_enh", 0)

            total_psnr += psnr
            total_ssim += ssim
            total_lpips += lpips

            count += 1

        finally:
            del runner
            torch.cuda.empty_cache()

    return total_psnr / count, total_ssim / count, total_lpips / count


def main(local_rank: int, world_rank, world_size: int, cfg_param: Config):
    if world_size > 1 and not cfg_param.disable_viewer:
        cfg_param.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg_param)

    if cfg_param.ckpt is not None:
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True)
            for file in cfg_param.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])

        runner.illumination_field.load_state_dict(
            {
                k: torch.cat([ckpt["illumination_field"][k] for ckpt in ckpts])
                for k in runner.illumination_field.state_dict().keys()
            }
        )
        runner.illumination_field.eval()

        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()


if __name__ == "__main__":
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(strategy=DefaultStrategy(verbose=True, refine_stop_iter=8000)),
        ),
        "mcmc": (
            "Gaussian splatting training using MCMC.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }

    config = tyro.cli(
        Config,
    )

    config.adjust_steps(config.steps_scaler)
    torch.set_float32_matmul_precision("high")

    cli(main, config, verbose=True)