import gc
import json
import math
import os
import time
from collections import defaultdict
from typing import Any

import imageio
import kornia.color
import numpy as np
import optuna
import torch
import torch.nn.functional as F
import tqdm
import tyro
import yaml
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, ChainedScheduler, CosineAnnealingLR
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from config import Config
from losses import white_preservation_loss, HistogramLoss
from gsplat.distributed import cli
from utils import ContentAwareIlluminationOptModule, IlluminationOptModule
from losses import (
    ColourConsistencyLoss,
    ExposureLoss,
    SpatialLoss,
    AdaptiveCurveLoss,
    LocalExposureLoss,
    ExclusionLoss, EdgeAwareSmoothingLoss
)
from rendering_double import rasterization_dual
from gsplat import export_splats
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import MCMCStrategy, DefaultStrategy
from utils import (
    knn,
    rgb_to_sh,
    set_random_seed,
)
from retinex import MultiScaleRetinexNet


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

    # lumincance
    adjust_k = torch.nn.Parameter(
        torch.ones_like(colors[:, :1, :]), requires_grad=True
    )  # enhance, for multiply
    adjust_b = torch.nn.Parameter(
        torch.zeros_like(colors[:, :1, :]), requires_grad=True
    )  # bias, for add,

    params.append(("adjust_k", adjust_k, sh0_lr))
    params.append(("adjust_b", adjust_b, sh0_lr))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)

    BS = batch_size * world_size

    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.AdamW

    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
            fused=True
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
            postfix=cfg.postfix,
        )

        self.trainset = Dataset(
            self.parser, patch_size=cfg.patch_size
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        if cfg.use_illum_opt:
            if cfg.illum_opt_type == "base":
                self.illum_module = IlluminationOptModule(num_images=len(self.trainset)).to(self.device)
            elif cfg.illum_opt_type == "content_aware":
                self.illum_module = ContentAwareIlluminationOptModule(
                    num_images=len(self.trainset),
                ).to(self.device)

            if world_size > 1:
                self.illum_module = DDP(self.illum_module, device_ids=[local_rank])


            initial_lr = cfg.illum_opt_lr * math.sqrt(cfg.batch_size)

            self.illum_optimizers = [
                torch.optim.AdamW(
                    self.illum_module.parameters(),
                    lr=initial_lr,
                    weight_decay=1e-4,
                    fused=True
                )
            ]

        self.loss_color = ColourConsistencyLoss().to(self.device)
        self.loss_exposure = ExposureLoss(patch_size=32).to(self.device)
        self.loss_spatial = SpatialLoss(
            learn_contrast=cfg.learn_spatial_contrast,
            num_images=len(self.trainset),
        ).to(self.device)
        self.loss_adaptive_curve = AdaptiveCurveLoss(
            learn_lambdas=cfg.learn_adaptive_curve_lambdas
        ).to(self.device)
        self.loss_edge_aware_smooth = EdgeAwareSmoothingLoss(learn_gamma=cfg.learn_edge_aware_gamma).to(self.device)
        self.loss_exposure_local = LocalExposureLoss(
            patch_size=64, patch_grid_size=8
        ).to(self.device)
        self.loss_exclusion = ExclusionLoss().to(self.device)
        self.histogram_loss = HistogramLoss().to(self.device)

        self.target_histogram_dist = nn.Parameter(torch.randn(256).to(self.device).softmax(dim=0))

        retinex_in_channels = 1 if cfg.use_hsv_color_space else 3
        retinex_out_channels = 1 if cfg.use_hsv_color_space else 3

        self.global_mean_val_param = nn.Parameter(
            torch.tensor([0.5], dtype=torch.float32).to(self.device)
        )

        self.retinex_net = MultiScaleRetinexNet(
            in_channels=retinex_in_channels,
            out_channels=retinex_out_channels,
            predictive_adaptive_curve=cfg.predictive_adaptive_curve,
            learn_local_exposure=cfg.learn_local_exposure,
            embed_dim=cfg.retinex_embedding_dim
        ).to(self.device)

        if world_size > 1:
            self.retinex_net = DDP(self.retinex_net, device_ids=[local_rank])

        net_params = list(self.retinex_net.parameters())

        net_params += self.loss_edge_aware_smooth.parameters()
        net_params += self.loss_adaptive_curve.parameters()
        net_params += self.loss_spatial.parameters()
        net_params.append(self.global_mean_val_param)
        net_params.append(self.target_histogram_dist)

        self.retinex_optimizer = torch.optim.AdamW(
            net_params,
            lr=cfg.retinex_opt_lr * math.sqrt(cfg.batch_size),
            weight_decay=1e-4,
            fused=True
        )
        self.retinex_embed_dim = cfg.retinex_embedding_dim
        self.retinex_embeds = nn.Embedding(
            len(self.trainset), self.retinex_embed_dim
        ).to(self.device)

        if world_size > 1:
            self.retinex_embeds = DDP(self.retinex_embeds, device_ids=[local_rank])

        self.retinex_embed_optimizer = torch.optim.AdamW(
            [{"params": self.retinex_embeds.parameters(), "lr": cfg.retinex_embedding_lr}],
            weight_decay=1e-5,
            fused=True
        )

        feature_dim = None
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
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        self.cfg.strategy.check_sanity(self.splats, self.optimizers)
        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

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
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    def rasterize_splats(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            masks: Tensor | None = None,
            **kwargs,
    ) -> (
            tuple[Tensor, Tensor, dict[str, Any]]
            | tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]
    ):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        rasterize_mode: Literal["antialiased", "classic"] = (
           "classic"
        )

        input_images_for_illum = kwargs.pop("input_images_for_illum", None)

        if self.cfg.use_illum_opt:
            if self.cfg.illum_opt_type == "content_aware":
                image_gain, image_gamma, _ = self.illum_module(input_images_for_illum, image_ids)
                colors_low = image_gain * (torch.clamp(colors, 1e-6, 1.0) ** image_gamma)
            elif self.cfg.illum_opt_type == "base":
                image_adjust_k, image_adjust_b = self.illum_module(image_ids)
                colors_low = colors * image_adjust_k + image_adjust_b
            else:
                raise ValueError(f"Unknown illum opt type: {self.cfg.illum_opt_type}")

        else:
            adjust_k = self.splats["adjust_k"]  # 1090, 1, 3
            adjust_b = self.splats["adjust_b"]  # 1090, 1, 3

            colors_low = colors * adjust_k + adjust_b  # least squares: x_enh=a*x+b

        (
            render_colors_enh,
            render_colors_low,
            render_enh_alphas,
            render_low_alphas,
            info,
        ) = rasterization_dual(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            colors_low=colors_low,
            viewmats=torch.linalg.inv(camtoworlds.float()),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=False,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            rasterize_mode=rasterize_mode,
            **kwargs,
        )

        return (
            render_colors_enh,
            render_colors_low,
            render_enh_alphas,
            render_low_alphas,
            info,
        )  # return colors and alphas

    def get_retinex_output(
            self, images_ids: Tensor, pixels: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor | None, Tensor | None, Tensor | None]:
        epsilon = torch.finfo(pixels.dtype).eps

        if self.cfg.use_hsv_color_space:
            pixels_nchw = pixels.permute(0, 3, 1, 2)
            pixels_hsv = kornia.color.rgb_to_hsv(pixels_nchw)
            v_channel = pixels_hsv[:, 2:3, :, :]
            input_image_for_net = v_channel
            log_input_image = torch.log(input_image_for_net + epsilon)
        else:
            pixels_hsv = torch.tensor(0.0, device=self.device)
            input_image_for_net = pixels.permute(0, 3, 1, 2)
            log_input_image = torch.log(input_image_for_net + epsilon)

        retinex_embedding = self.retinex_embeds(images_ids)

        log_illumination_map, alpha, beta, local_exposure = checkpoint(
            self.retinex_net,
            input_image_for_net,
            retinex_embedding,
            use_reentrant=False,
        )
        illumination_map = torch.exp(log_illumination_map)
        illumination_map = torch.clamp(illumination_map, min=1e-5)

        log_reflectance_target = log_input_image - log_illumination_map

        if self.cfg.use_hsv_color_space:
            reflectance_v_target = torch.exp(log_reflectance_target)

            h_channel = pixels_hsv[:, 0:1, :, :]
            s_channel = pixels_hsv[:, 1:2, :, :]
            # s_channel_adjusted = torch.clamp(s_channel / torch.clamp(illumination_map, min=1e-5), 0.0, 1.0)
            reflectance_hsv_target = torch.cat(
                [h_channel, s_channel, reflectance_v_target], dim=1
            )
            reflectance_map = kornia.color.hsv_to_rgb(reflectance_hsv_target)
        else:
            reflectance_map = torch.exp(log_reflectance_target)

        reflectance_map = torch.clamp(reflectance_map, 0, 1)

        return (
            input_image_for_net,
            illumination_map,
            reflectance_map,
            alpha,
            beta,
            local_exposure,
        )

    def retinex_train_step(self, images_ids: Tensor, pixels: Tensor, step: int, is_pretrain: bool = True) -> Tensor:
        cfg = self.cfg
        device = self.device

        (
            input_image_for_net,
            illumination_map,
            reflectance_map,
            alpha,
            beta,
            local_exposure_mean,
        ) = self.get_retinex_output(images_ids=images_ids, pixels=pixels)
        global_mean_val_target = torch.sigmoid(self.global_mean_val_param)

        loss_color_val = (
            self.loss_color(illumination_map)
            if not cfg.use_hsv_color_space
            else torch.tensor(0.0, device=device)
        )
        loss_adaptive_curve = self.loss_adaptive_curve(reflectance_map, alpha, beta)
        if cfg.learn_global_exposure:
            loss_exposure_val = self.loss_exposure(
                reflectance_map, global_mean_val_target
            )
        else:
            loss_exposure_val = self.loss_exposure(reflectance_map)

        con_degree = (0.5 / torch.mean(pixels)).item()
        org_loss_reflectance_spa_map = self.loss_spatial.forward_per_pixel(
            input_image_for_net, reflectance_map, contrast=con_degree, image_id=images_ids
        )

        loss_reflectance_spa = org_loss_reflectance_spa_map.mean()

        loss_smooth_edge_aware = self.loss_edge_aware_smooth(
            illumination_map, input_image_for_net
        )
        if cfg.learn_local_exposure:
            loss_exposure_local = self.loss_exposure_local(
                reflectance_map, local_exposure_mean
            )
        else:
            loss_exposure_local = self.loss_exposure_local(reflectance_map)

        loss_exclusion_val = self.loss_exclusion(reflectance_map, illumination_map)

        loss_white_preservation = white_preservation_loss(
            input_image=pixels, illumination_map=illumination_map.permute(0, 2, 3, 1)
        )

        loss_histogram = self.histogram_loss(reflectance_map, self.target_histogram_dist)


        individual_losses = torch.stack(
            [
                loss_reflectance_spa,  # 0
                loss_color_val,  # 1
                loss_exposure_val,  # 2
                loss_adaptive_curve,  # 4
                loss_smooth_edge_aware,  # 8
                loss_exposure_local,  # 9
                loss_exclusion_val,  # 11
                loss_white_preservation,  # 12
                loss_histogram,  # 13
            ]
        )

        base_lambdas = torch.tensor(
            [
                cfg.lambda_reflect,
                cfg.lambda_illum_color,
                cfg.lambda_illum_exposure,
                cfg.lambda_illum_curve,
                cfg.lambda_edge_aware_smooth,
                cfg.lambda_illum_exposure_local,
                cfg.lambda_illum_exclusion,
                cfg.lambda_white_preservation,
                cfg.lambda_histogram,
            ],
            device=device,
        )

        total_loss = (base_lambdas * individual_losses).sum()

        if step % self.cfg.tb_every == 0:
            self.writer.add_scalar("retinex_net/total_loss", total_loss.item(), step)

            loss_names = [
                "reflect_spa",
                "color_val",
                "exposure_val",
                "adaptive_curve",
                "smooth_edge_aware",
                "exposure_local",
                "exclusion_val",
                "white_preservation",
                "histogram_loss",
            ]

            title = "retinex_net" if is_pretrain else "train"

            for i, name in enumerate(loss_names):
                self.writer.add_scalar(
                    f"{title}/loss_{name}_unweighted", individual_losses[i].item(), step
                )

                self.writer.add_scalar(
                    f"{title}/loss_{name}_fixed_weighted",
                    (individual_losses[i] * base_lambdas[i]).item(),
                    step,
                )

            if cfg.learn_edge_aware_gamma:
                self.writer.add_scalar(
                    f"{title}/edge_aware_gamma_adjustment",
                    self.loss_edge_aware_smooth.gamma_adjustment.item(),
                    step,
                )


            if cfg.learn_global_exposure:
                self.writer.add_scalar(
                    f"{title}/global_mean_val_param",
                    self.global_mean_val_param.item(),
                    step,
                )
            if cfg.learn_local_exposure:
                self.writer.add_scalar(
                    f"{title}/local_mean_val_param",
                    local_exposure_mean.mean().item(),
                    step,
                )

            if cfg.learn_adaptive_curve_lambdas:
                self.writer.add_scalar(
                    f"{title}/learnable_adaptive_curve_lambda1",
                    self.loss_adaptive_curve.lambda1.item(),
                    step,
                )
                self.writer.add_scalar(
                    f"{title}/learnable_adaptive_curve_lambda2",
                    self.loss_adaptive_curve.lambda2.item(),
                    step,
                )
                self.writer.add_scalar(
                    f"{title}/learnable_adaptive_curve_lambda3",
                    self.loss_adaptive_curve.lambda3.item(),
                    step,
                )

            if self.cfg.tb_save_image:
                self.writer.add_images(
                    f"{title}/input_image_for_net",
                    input_image_for_net,
                    step,
                )
                self.writer.add_images(
                    f"{title}/pixels",
                    pixels.permute(0, 3, 1, 2),
                    step,
                )

                self.writer.add_images(
                    f"{title}/illumination_map",
                    illumination_map,
                    step,
                )
                self.writer.add_images(
                    f"{title}/target_reflectance",
                    reflectance_map,
                    step,
                )


        return total_loss

    def pre_train_retinex(self) -> None:
        cfg = self.cfg
        device = self.device

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )

        trainloader_iter = iter(trainloader)

        initial_retinex_lr = self.retinex_optimizer.param_groups[0]["lr"]
        initial_embed_lr = self.retinex_embed_optimizer.param_groups[0]["lr"]
        schedulers = [
            CosineAnnealingLR(
                self.retinex_optimizer,
                T_max=cfg.pretrain_steps + cfg.max_steps,
                eta_min=initial_retinex_lr * 0.01,
            ),
            CosineAnnealingLR(
                self.retinex_embed_optimizer,
                T_max=cfg.pretrain_steps + cfg.max_steps,
                eta_min=initial_embed_lr * 0.01,
            ),
        ]

        pbar = tqdm.tqdm(range(self.cfg.pretrain_steps), desc="Pre-training RetinexNet")
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            with torch.autocast(enabled=False, device_type=device):
                images_ids = data["image_id"].to(device)
                pixels = data["image"].to(device) / 255.0


                loss = self.retinex_train_step(
                    images_ids=images_ids, pixels=pixels, step=step
                )

            loss.backward()

            self.retinex_optimizer.step()
            self.retinex_embed_optimizer.step()

            self.retinex_optimizer.zero_grad()
            self.retinex_embed_optimizer.zero_grad()

            for scheduler in schedulers:
                scheduler.step()

            pbar.set_postfix({"loss": loss.item()})


    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers: list[ExponentialLR | ChainedScheduler | CosineAnnealingLR] = [
            ExponentialLR(self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)),
        ]

        initial_retinex_lr = self.retinex_optimizer.param_groups[0]["lr"]
        schedulers.append(
            CosineAnnealingLR(
                self.retinex_optimizer,
                T_max=max_steps,
                eta_min=initial_retinex_lr * 0.01,
            )
        )

        initial_embed_lr = self.retinex_embed_optimizer.param_groups[0]["lr"]
        schedulers.append(
            CosineAnnealingLR(
                self.retinex_embed_optimizer,
                T_max=max_steps,
                eta_min=initial_embed_lr * 0.01,
            )
        )

        if cfg.pretrain_retinex:
            self.pre_train_retinex()

        trainloader = torch.utils.data.DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
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
            )  # Defined early

            with torch.autocast(enabled=False, device_type=device):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)

                image_ids = data["image_id"].to(device)
                pixels = data["image"].to(device) / 255.0

                masks = data["mask"].to(device) if "mask" in data else None

                height, width = pixels.shape[1:3]

                out = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                    image_ids=image_ids,
                    render_mode="RGB",
                    masks=masks,
                    input_images_for_illum=pixels.permute(0, 3, 1, 2),
                )

                if len(out) == 5:
                    renders_enh, renders_low, alphas_enh, alphas_low, info = out
                else:
                    renders_low, alphas_low, info = out
                    renders_enh, alphas_enh = renders_low, alphas_low

                if renders_low.shape[-1] == 4:
                    colors_low, depths_low = (
                        renders_low[..., 0:3],
                        renders_low[..., 3:4],
                    )
                    colors_enh, depths_enh = (
                        renders_enh[..., 0:3],
                        renders_enh[..., 3:4],
                    )
                else:
                    colors_low, depths_low = renders_low, None
                    colors_enh, depths_enh = renders_enh, None

                colors_low = torch.clamp(colors_low, 0.0, 1.0)
                colors_enh = torch.clamp(colors_enh, 0.0, 1.0)
                pixels = torch.clamp(pixels, 0.0, 1.0)

                info["means2d"].retain_grad()

                with torch.no_grad():
                    (
                        gt_input_for_net,
                        gt_illumination_map,
                        gt_reflectance_target,
                        _, _, _
                    ) = self.get_retinex_output(images_ids=image_ids, pixels=pixels)

                gt_reflectance_target_permuted = gt_reflectance_target.permute(0, 2, 3, 1).detach()

                loss_reconstruct_low = F.l1_loss(colors_low, pixels)

                ssim_loss_low = 1.0 - self.ssim(
                    colors_low.permute(0, 3, 1, 2),
                    pixels.permute(0, 3, 1, 2),
                )
                low_loss = (1.0 - cfg.ssim_lambda) * loss_reconstruct_low + cfg.ssim_lambda * ssim_loss_low


                loss_reconstruct_enh = F.l1_loss(colors_enh, gt_reflectance_target_permuted)
                ssim_loss_enh = 1.0 - self.ssim(
                    colors_enh.permute(0, 3, 1, 2),
                    gt_reflectance_target_permuted.permute(0, 3, 1, 2),
                )
                enh_loss = (1.0 - cfg.ssim_lambda) * loss_reconstruct_enh + cfg.ssim_lambda * ssim_loss_enh

                retinex_loss = self.retinex_train_step(images_ids=image_ids, pixels=pixels, step=step,
                                                       is_pretrain=False)

                reconstructed_from_components = colors_enh.permute(0, 3, 1, 2) * gt_illumination_map.detach()
                loss_bidirectional = F.l1_loss(reconstructed_from_components, pixels.permute(0, 3, 1, 2))

                loss = (
                        cfg.lambda_low * low_loss
                        + (1.0 - cfg.lambda_low) * enh_loss
                        + cfg.lambda_illumination * retinex_loss
                        + cfg.lambda_bidirectional * loss_bidirectional
                )

                self.cfg.strategy.step_pre_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )

                if cfg.opacity_reg > 0.0:
                    loss += (
                            cfg.opacity_reg
                            * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                    )
                if cfg.scale_reg > 0.0:
                    loss += (
                            cfg.scale_reg
                            * torch.abs(torch.exp(self.splats["scales"])).mean()
                    )

            loss.backward()

            desc_parts = [f"loss={loss.item():.3f}", f"retinex_loss={retinex_loss.item():.3f} ",
                          f"sh_deg={sh_degree_to_use}"]
            pbar.set_description("| ".join(desc_parts))

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1low", loss_reconstruct_low.item(), step)
                self.writer.add_scalar("train/ssim_low", ssim_loss_low.item(), step)
                self.writer.add_scalar("train/l1enh", loss_reconstruct_enh.item(), step)
                self.writer.add_scalar("train/ssim_enh", ssim_loss_enh.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                self.writer.add_scalar("train/loss_bidirectional", loss_bidirectional.item(), step)
                if cfg.tb_save_image:
                    with torch.no_grad():
                        self.writer.add_images(
                            "train/render_low", colors_low.permute(0, 3, 1, 2), step
                        )
                        self.writer.add_images(
                            "train/pixels",
                            pixels.permute(0, 3, 1, 2),
                            step,
                        )
                        self.writer.add_images(
                            "train/render_enh",
                            colors_enh.permute(0, 3, 1, 2),
                            step,
                        )
                        self.writer.add_images(
                            "train/illumination_map",
                            gt_illumination_map,
                            step,
                        )
                        self.writer.add_images(
                            "train/reflectance_target",
                            gt_reflectance_target,
                            step,
                        )

                self.writer.flush()

            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats_save = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats_save)
                with open(
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                        "w",
                ) as f:
                    json.dump(stats_save, f)
                data_save = {"step": step, "splats": self.splats.state_dict(), "retinex_net": (
                    self.retinex_net.module.state_dict()
                    if world_size > 1
                    else self.retinex_net.state_dict()),
                    "retinex_embeds": self.retinex_embeds.state_dict(),
                             "illum_module": self.illum_module.state_dict() if self.cfg.use_illum_opt else None,
                }
                torch.save(
                    data_save, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                    step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                sh0_export = self.splats["sh0"]
                shN_export = self.splats["shN"]

                export_splats(
                    means=self.splats["means"],
                    scales=self.splats["scales"],
                    quats=self.splats["quats"],
                    opacities=self.splats["opacities"],
                    sh0=sh0_export,
                    shN=shN_export,
                    save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )


            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad()

            if self.cfg.use_illum_opt:
                for optimizer in self.illum_optimizers:
                    # scaler.step(optimizer)
                    optimizer.step()
                    optimizer.zero_grad()

            # scaler.step(self.retinex_optimizer)
            self.retinex_optimizer.step()
            self.retinex_optimizer.zero_grad()
            # scaler.step(self.retinex_embed_optimizer)
            self.retinex_embed_optimizer.step()
            self.retinex_embed_optimizer.zero_grad()

            for scheduler in schedulers:
                scheduler.step()

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
            if step in [i - 1 for i in cfg.eval_steps]:
                self.render_traj(step)

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
            image_id = data["image_id"].to(device)

            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()

            out = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
                image_ids=image_id,
                input_images_for_illum=pixels.permute(0, 3, 1, 2),
            )

            if len(out) == 5:
                colors_enh, colors_low, alphas_enh, alphas_low, info = out
            else:
                colors_low, alphas_low, info = out
                colors_enh, alphas_enh = colors_low, colors_low

            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)

            colors_low = torch.clamp(colors_low, 0.0, 1.0)
            colors_enh = torch.clamp(colors_enh, 0.0, 1.0)

            if world_rank == 0:
                canvas_list_low = [pixels, colors_low]
                canvas_list_enh = [pixels, colors_enh]

                canvas_eval_low = (
                    torch.cat(canvas_list_low, dim=2).squeeze(0).cpu().numpy()
                )
                canvas_eval_low = (canvas_eval_low * 255).astype(np.uint8)

                canvas_eval_enh = (
                    torch.cat(canvas_list_enh, dim=2).squeeze(0).cpu().numpy()
                )
                canvas_eval_enh = (canvas_eval_enh * 255).astype(np.uint8)

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_low_{i:04d}.png",
                    canvas_eval_low,
                )

                colors_low_np = colors_low.squeeze(0).cpu().numpy()

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_low_{i:04d}.png",
                    (colors_low_np * 255).astype(np.uint8),
                )

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_enh_{i:04d}.png",
                    canvas_eval_enh,
                )
                colors_enh_np = colors_enh.squeeze().cpu().numpy()
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_enh_{i:04d}.png",
                    (colors_enh_np * 255).astype(np.uint8),
                )

                # illumination_map_np = illumination_map.squeeze(0).cpu().numpy()
                # imageio.imwrite(
                #     f"{self.render_dir}/{stage}_illumination_map_{i:04d}.png",
                #     (illumination_map_np * 255).astype(np.uint8),
                # )
                #
                # reflectance_map_np = reflectance_map.squeeze(0).cpu().numpy()
                # imageio.imwrite(
                #     f"{self.render_dir}/{stage}_reflectance_map_{i:04d}.png",
                #     (reflectance_map_np * 255).astype(np.uint8),
                # )

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors_low.permute(0, 3, 1, 2)

                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                with torch.no_grad():
                    colors_enh_p = colors_enh.permute(0, 3, 1, 2)

                    metrics["lpips_enh"].append(self.lpips(colors_enh_p, pixels_p))
                    metrics["ssim_enh"].append(self.ssim(colors_enh_p, pixels_p))
                    metrics["psnr_enh"].append(self.psnr(colors_enh_p, pixels_p))

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
            ]
            print_parts_eval.extend(
                [
                    f"Time: {stats_eval.get('ellipse_time', 0):.3f}s/image",
                    f"GS: {stats_eval.get('num_GS', 0)}",
                ]
            )
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
            with open(f"{self.stats_dir}/{stage}_raw_step{step:04d}.json", "w") as f:
                json.dump(raw_metrics, f)
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
        video_path = f"{video_dir}/traj_{step}.mp4"
        video_writer = imageio.get_writer(video_path, fps=30)

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

            if len(out) == 5:
                renders_traj, _, _, _, _ = out
            else:
                renders_traj, _, _ = out

            colors_traj = torch.clamp(renders_traj[..., 0:3], 0.0, 1.0)
            depths_traj = renders_traj[..., 3:4]
            depths_traj_norm = (depths_traj - depths_traj.min()) / (
                    depths_traj.max() - depths_traj.min() + 1e-10
            )

            canvas_traj_list = [colors_traj, depths_traj_norm.repeat(1, 1, 1, 3)]
            canvas_traj = torch.cat(canvas_traj_list, dim=2).squeeze(0).cpu().numpy()
            canvas_traj_uint8 = (canvas_traj * 255).astype(np.uint8)
            video_writer.append_data(canvas_traj_uint8)
            
        video_writer.close()
        print(f"Video saved to {video_path}")

        self.writer.flush()

    def eval_retinex(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, shuffle=False, num_workers=1
        )
        metrics = defaultdict(list)
        for i, data in enumerate(tqdm.tqdm(valloader, desc="Eval Retinex")):
            pixels = data["image"].to(device) / 255.0
            image_ids = data["image_id"].to(device)

            with torch.no_grad():
                (
                    gt_input_for_net,
                    gt_illumination_map,
                    gt_reflectance_target,
                    _, _, _
                ) = self.get_retinex_output(images_ids=image_ids, pixels=pixels)

                _, reflectance_map, _, _ = self.retinex_net(
                    gt_input_for_net
                )

                pixels_p = pixels.permute(0, 3, 1, 2)
                reflectance_map_p = reflectance_map.permute(0, 3, 1, 2)

                metrics["psnr"].append(self.psnr(reflectance_map_p, pixels_p))
                metrics["ssim"].append(self.ssim(reflectance_map_p, pixels_p))
                metrics["lpips"].append(self.lpips(reflectance_map_p, pixels_p))

        if world_rank == 0:
            stats_eval = {}
            for k, v_list in metrics.items():
                if v_list:
                    if isinstance(v_list[0], torch.Tensor):
                        stats_eval[k] = torch.stack(v_list).mean().item()
                    else:
                        stats_eval[k] = sum(v_list) / len(v_list)
                else:
                    stats_eval[k] = 0

        return stats_eval["psnr"], stats_eval["ssim"], stats_eval["lpips"]

def main(local_rank: int, world_rank, world_size: int, cfg_param: Config):
    if world_size > 1 and not cfg_param.disable_viewer:
        cfg_param.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg_param)

    if cfg_param.ckpt is not None:
        ckpts = [
            torch.load(file, map_location=runner.device, weights_only=True) for file in cfg_param.ckpt
        ]
        for k in runner.splats.keys():
            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])

        runner.retinex_net.load_state_dict(
            {k: torch.cat([ckpt["retinex_net"][k] for ckpt in ckpts]) for k in runner.retinex_net.state_dict().keys()}
        )

        runner.retinex_embeds.load_state_dict(
            {k: torch.cat([ckpt["retinex_embeds"][k] for ckpt in ckpts]) for k in runner.retinex_embeds.state_dict().keys()}
        )

        if runner.cfg.use_illum_opt:
            runner.illum_module.load_state_dict(
                {k: torch.cat([ckpt["illum_module"][k] for ckpt in ckpts]) for k in runner.illum_module.state_dict().keys()}
            )

        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()

def objective(trial: optuna.Trial):
    cfg = Config()

    cfg.lambda_reflect = trial.suggest_float("lambda_reflect", 0.0, 5.0)
    cfg.lambda_illum_curve = trial.suggest_float("lambda_illum_curve", 1.0, 50.0, log=True)
    cfg.lambda_illum_exposure = trial.suggest_float("lambda_illum_exposure", 0.0, 5.0)
    cfg.lambda_edge_aware_smooth = trial.suggest_float("lambda_edge_aware_smooth", 10, 100.0, log=True)
    cfg.lambda_illum_exposure_local = trial.suggest_float("lambda_illum_exposure_local", 0.0, 1.0)
    cfg.lambda_white_preservation = trial.suggest_float("lambda_white_preservation", 1e-3, 10.0, log=True)
    cfg.lambda_histogram = trial.suggest_float("lambda_histogram", 1e-3, 10.0, log=True)

    cfg.lambda_exclusion = trial.suggest_float("lambda_exclusion", 0.0, 2.0)

    cfg.predictive_adaptive_curve = trial.suggest_categorical("predictive_adaptive_curve", [True, False])
    cfg.learn_spatial_contrast = trial.suggest_categorical("learn_spatial_contrast", [True, False])
    cfg.learn_local_exposure = trial.suggest_categorical("learn_local_exposure", [True, False])
    cfg.learn_global_exposure = trial.suggest_categorical("learn_global_exposure", [True, False])
    cfg.learn_edge_aware_gamma = trial.suggest_categorical("learn_edge_aware_gamma", [True, False])

    cfg.retinex_embedding_dim = trial.suggest_categorical("retinex_embedding_dim", [32, 64])

    cfg.max_steps = 3000
    cfg.eval_steps = [3000]
    cfg.pretrain_steps = 2500

    runner = None
    try:
        runner = Runner(0, 0, 1, cfg)
        runner.pre_train_retinex()

        return runner.eval_retinex()

    finally:
        if runner is not None:
            del runner

        gc.collect()
        torch.cuda.empty_cache()

BilateralGrid = None
color_correct = None
slice_func = None
total_variation_loss = None

if __name__ == "__main__":
    # configs = {
    #     "default": (
    #         "Gaussian splatting training using densification heuristics from the original paper.",
    #         Config(strategy=DefaultStrategy(verbose=True, refine_stop_iter=8000)),
    #     ),
    #     "mcmc": (
    #         "Gaussian splatting training using MCMC.",
    #         Config(
    #             init_opa=0.5,
    #             init_scale=0.1,
    #             opacity_reg=0.01,
    #             scale_reg=0.01,
    #             strategy=MCMCStrategy(verbose=True),
    #         ),
    #     ),
    # }
    # # config = tyro.extras.overridable_config_cli(configs)
    # config = tyro.cli(
    #     Config,
    # )
    #
    # config.adjust_steps(config.steps_scaler)
    # torch.set_float32_matmul_precision("high")
    #
    # cli(main, config, verbose=True)

    study = optuna.create_study(directions=["maximize", "maximize", "minimize"])

    study.optimize(objective, n_trials=50)

    print("Study statistics: ")
    print(f"  Number of finished trials: {len(study.trials)}")

    print("Best trials (Pareto front):")
    for i, trial in enumerate(study.best_trials):
        print(f"  Trial {i}:")
        print(f"    Values: PSNR={trial.values[0]:.4f}, SSIM={trial.values[1]:.4f}, LPIPS={trial.values[2]:.4f}")
        print("    Params: ")
        for key, value in trial.params.items():
            print(f"      {key}: {value}")




