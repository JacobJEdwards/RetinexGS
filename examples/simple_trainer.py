import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from fused_ssim import fused_ssim
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed

from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from nerfview import CameraState, RenderTabState, apply_float_colormap

try:
    import piq
    PIQ_AVAILABLE = True
except ImportError:
    PIQ_AVAILABLE = False
    print("Warning: 'piq' library not found. NR-IQA functionalities (CLIPIQA, NIQE) will be disabled.")
    print("Please install it using: pip install piq clip-anytorch")


@dataclass
class Config:
    disable_viewer: bool = False
    ckpt: Optional[List[str]] = None
    compression: Optional[Literal["png"]] = None
    render_traj_path: str = "interp"

    data_dir: str = "data/360_v2/garden"
    data_factor: int = 4
    result_dir: str = "results/garden"
    test_every: int = 8
    patch_size: Optional[int] = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    port: int = 8080

    batch_size: int = 1
    steps_scaler: float = 1.0

    max_steps: int = 30_000
    eval_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    save_ply: bool = False
    ply_steps: List[int] = field(default_factory=lambda: [7_000, 30_000])
    disable_video: bool = False

    init_type: str = "sfm"
    init_num_pts: int = 100_000
    init_extent: float = 3.0
    sh_degree: int = 3
    sh_degree_interval: int = 1000
    init_opa: float = 0.1
    init_scale: float = 1.0
    ssim_lambda: float = 0.2

    near_plane: float = 0.01
    far_plane: float = 1e10

    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = False
    antialiased: bool = False

    random_bkgd: bool = False

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    pose_opt: bool = False
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6

    use_bilateral_grid: bool = False
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    depth_loss: bool = False
    depth_lambda: float = 1e-2

    tb_every: int = 100
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    with_ut: bool = False
    with_eval3d: bool = False

    use_fused_bilagrid: bool = False

    enable_clipiqa_loss: bool = True
    clipiqa_lambda: float = 0.01
    clipiqa_model_type: Literal["clipiqa"] = "clipiqa"

    eval_niqe: bool = True

    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.ply_steps = [int(i * factor) for i in self.ply_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


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
        feature_dim: Optional[int] = None,
        device: str = "cuda",
        world_rank: int = 0,
        world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)
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
    optimizer_class = None
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
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        os.makedirs(cfg.result_dir, exist_ok=True)
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)
        self.ply_dir = f"{cfg.result_dir}/ply"
        os.makedirs(self.ply_dir, exist_ok=True)

        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        feature_dim = 32 if cfg.app_opt else None
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
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
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

        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            global BilateralGrid
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]


        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        self.clipiqa_metric = None
        if cfg.enable_clipiqa_loss and PIQ_AVAILABLE:
            try:
                self.clipiqa_metric = piq.CLIPIQA(data_range=1., model_type=cfg.clipiqa_model_type)
                self.clipiqa_metric = self.clipiqa_metric.to(self.device)
                print(f"CLIPIQA metric ({cfg.clipiqa_model_type}) initialized for training loss.")
            except Exception as e:
                print(f"Error initializing CLIPIQA: {e}. CLIPIQA loss will be skipped.")
                cfg.enable_clipiqa_loss = False
        elif cfg.enable_clipiqa_loss and not PIQ_AVAILABLE:
            print("CLIPIQA loss enabled in config, but 'piq' library is not available. Skipping CLIPIQA.")
            cfg.enable_clipiqa_loss = False


        self.niqe_metric = None
        if cfg.eval_niqe and PIQ_AVAILABLE:
            try:
                self.niqe_metric = piq.NIQE(data_range=1.0, reduction='mean')
                self.niqe_metric = self.niqe_metric.to(self.device)
                print("NIQE metric initialized for evaluation.")
            except Exception as e:
                print(f"Error initializing NIQE: {e}. NIQE evaluation will be skipped.")
                cfg.eval_niqe = False
        elif cfg.eval_niqe and not PIQ_AVAILABLE:
            print("NIQE evaluation enabled in config, but 'piq' library is not available. Skipping NIQE.")
            cfg.eval_niqe = False


        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

    def rasterize_splats(
            self,
            camtoworlds: Tensor,
            Ks: Tensor,
            width: int,
            height: int,
            masks: Optional[Tensor] = None,
            rasterize_mode: Optional[Literal["classic", "antialiased"]] = None,
            camera_model: Optional[Literal["pinhole", "ortho", "fisheye"]] = None,
            **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors_feat = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors_feat = colors_feat + self.splats["colors"]
            colors = torch.sigmoid(colors_feat)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)

        if rasterize_mode is None:
            rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        if camera_model is None:
            camera_model = self.cfg.camera_model
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),
            Ks=Ks,
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            with_ut=self.cfg.with_ut,
            with_eval3d=self.cfg.with_eval3d,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

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

        schedulers = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            global total_variation_loss
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )


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
            if not cfg.disable_viewer:
                while self.viewer.state == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            num_train_rays_per_step = (
                    pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None
            if cfg.depth_loss:
                points = data["points"].to(device)
                depths_gt = data["depths"].to(device)

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)
            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths_map = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths_map = renders, None

            if cfg.use_bilateral_grid:
                global slice, color_correct
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                    )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(
                    self.bil_grids,
                    grid_xy.expand(colors.shape[0], -1, -1, -1),
                    colors,
                    image_ids.unsqueeze(-1),
                )["rgb"]


            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

            clipiqa_loss_value = torch.tensor(0.0, device=device)
            clipiqa_score_value = torch.tensor(0.0, device=device)
            if cfg.enable_clipiqa_loss and self.clipiqa_metric is not None:
                colors_nchw = colors.permute(0, 3, 1, 2).contiguous()
                current_clipiqa_score = self.clipiqa_metric(colors_nchw)
                clipiqa_score_value = current_clipiqa_score.mean()
                clipiqa_loss_value = -clipiqa_score_value
                loss = loss + cfg.clipiqa_lambda * clipiqa_loss_value

            depthloss_value = torch.tensor(0.0, device=device)
            if cfg.depth_loss and depths_map is not None:
                points_norm = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                    dim=-1,
                )
                grid = points_norm.unsqueeze(2)
                depths_sampled = F.grid_sample(
                    depths_map.permute(0, 3, 1, 2), grid, align_corners=True
                )
                depths_sampled = depths_sampled.squeeze(3).squeeze(1)
                disp = torch.where(depths_sampled > 0.0, 1.0 / depths_sampled, torch.zeros_like(depths_sampled))
                disp_gt = 1.0 / depths_gt
                depthloss_value = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss_value * cfg.depth_lambda

            tvloss_value = torch.tensor(0.0, device=device)
            if cfg.use_bilateral_grid:
                tvloss_value = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss_value


            if cfg.opacity_reg > 0.0:
                loss = loss + cfg.opacity_reg * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
            if cfg.scale_reg > 0.0:
                loss = loss + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()

            loss.backward()

            desc_parts = [f"loss={loss.item():.3f}"]
            if cfg.enable_clipiqa_loss and self.clipiqa_metric is not None:
                desc_parts.append(f"clipiqa={clipiqa_score_value.item():.3f}")
            desc_parts.append(f"sh_deg={sh_degree_to_use}")
            if cfg.depth_loss and depths_map is not None:
                desc_parts.append(f"depth_l={depthloss_value.item():.6f}")
            if cfg.use_bilateral_grid:
                desc_parts.append(f"tv_l={tvloss_value.item():.6f}")
            if cfg.pose_opt and cfg.pose_noise:
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc_parts.append(f"pose_err={pose_err.item():.6f}")
            pbar.set_description("| ".join(desc_parts))


            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.enable_clipiqa_loss and self.clipiqa_metric is not None:
                    self.writer.add_scalar("train/clipiqa_score", clipiqa_score_value.item(), step)
                    self.writer.add_scalar("train/clipiqa_loss", clipiqa_loss_value.item(), step)
                if cfg.depth_loss and depths_map is not None:
                    self.writer.add_scalar("train/depthloss", depthloss_value.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss_value.item(), step)
                if cfg.tb_save_image:
                    canvas_tb = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas_tb = canvas_tb.reshape(-1, *canvas_tb.shape[2:])
                    self.writer.add_image("train/render", canvas_tb, step, dataformats='HWC')
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
                        f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json", "w"
                ) as f:
                    json.dump(stats_save, f)
                data_save = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    data_save["pose_adjust"] = self.pose_adjust.module.state_dict() if world_size > 1 else self.pose_adjust.state_dict()
                if cfg.app_opt:
                    data_save["app_module"] = self.app_module.module.state_dict() if world_size > 1 else self.app_module.state_dict()
                torch.save(
                    data_save, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1) and cfg.save_ply:
                rgb_export = None
                if self.cfg.app_opt:
                    rgb_export_feat = self.app_module(
                        features=self.splats["features"],
                        embed_ids=None,
                        dirs=torch.zeros_like(self.splats["means"][None, :, :]),
                        sh_degree=sh_degree_to_use,
                    )
                    rgb_export_feat = rgb_export_feat + self.splats["colors"]
                    rgb_export = torch.sigmoid(rgb_export_feat).squeeze(0).unsqueeze(1)
                    sh0_export = rgb_to_sh(rgb_export)
                    shN_export = torch.empty([sh0_export.shape[0], 0, 3], device=sh0_export.device)
                else:
                    sh0_export = self.splats["sh0"]
                    shN_export = self.splats["shN"]
                export_splats(
                    means=self.splats["means"], scales=self.splats["scales"], quats=self.splats["quats"],
                    opacities=self.splats["opacities"], sh0=sh0_export, shN=shN_export,
                    format="ply", save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                )

            if cfg.sparse_grad:
                assert cfg.packed
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad_val = self.splats[k].grad
                    if grad_val is None or grad_val.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None], values=grad_val[gaussian_ids],
                        size=self.splats[k].size(), is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(self.splats["opacities"], dtype=bool)
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            for optimizer in self.optimizers.values():
                if cfg.visible_adam: optimizer.step(visibility_mask)
                else: optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers: optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers: optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers: optimizer.step(); optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers: scheduler.step()

            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats, optimizers=self.optimizers, state=self.strategy_state,
                    step=step, info=info, packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats, optimizers=self.optimizers, state=self.strategy_state,
                    step=step, info=info, lr=schedulers[0].get_last_lr()[0],
                )
            else: assert_never(self.cfg.strategy)

            if step in [i - 1 for i in cfg.eval_steps]: self.eval(step)
            if step in [i - 1 for i in cfg.eval_steps]: self.render_traj(step)
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = num_train_rays_per_step * num_train_steps_per_sec
                self.viewer.render_tab_state.num_train_rays_per_sec = num_train_rays_per_sec
                self.viewer.update(step, num_train_rays_per_step)

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val"):
        print(f"Running evaluation for step {step} on '{stage}' set...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank

        valloader = torch.utils.data.DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
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
            colors, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds, Ks=Ks, width=width, height=height,
                sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane, masks=masks,
            )
            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)

            if world_rank == 0:
                canvas_list = [pixels, colors]
                canvas_eval = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas_eval = (canvas_eval * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png", canvas_eval,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)

                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

                if cfg.eval_niqe and self.niqe_metric is not None:
                    niqe_score = self.niqe_metric(colors_p.contiguous())
                    metrics["niqe"].append(niqe_score)

                if cfg.use_bilateral_grid:
                    global color_correct
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))
                    metrics["cc_ssim"].append(self.ssim(cc_colors_p, pixels_p))
                    metrics["cc_lpips"].append(self.lpips(cc_colors_p, pixels_p))


        if world_rank == 0:
            avg_ellipse_time = ellipse_time_total / len(valloader) if len(valloader) > 0 else 0

            stats_eval = {}
            for k, v_list in metrics.items():
                if v_list:
                    if isinstance(v_list[0], torch.Tensor):
                        stats_eval[k] = torch.stack(v_list).mean().item()
                    else:
                        stats_eval[k] = sum(v_list) / len(v_list)
                else:
                    stats_eval[k] = 0

            stats_eval.update({
                "ellipse_time": avg_ellipse_time, "num_GS": len(self.splats["means"]),
            })

            print_parts_eval = [
                f"PSNR: {stats_eval.get('psnr', 0):.3f}",
                f"SSIM: {stats_eval.get('ssim', 0):.4f}",
                f"LPIPS: {stats_eval.get('lpips', 0):.3f}",
            ]
            if cfg.eval_niqe and self.niqe_metric is not None and 'niqe' in stats_eval:
                print_parts_eval.append(f"NIQE: {stats_eval.get('niqe',0):.3f} (lower is better)")
            if cfg.use_bilateral_grid:
                print_parts_eval.extend([
                    f"CC_PSNR: {stats_eval.get('cc_psnr',0):.3f}",
                    f"CC_SSIM: {stats_eval.get('cc_ssim',0):.4f}",
                    f"CC_LPIPS: {stats_eval.get('cc_lpips',0):.3f}",
                ])
            print_parts_eval.extend([
                f"Time: {stats_eval.get('ellipse_time',0):.3f}s/image",
                f"GS: {stats_eval.get('num_GS',0)}"
            ])
            print(f"Eval {stage} Step {step}: " + " | ".join(print_parts_eval))

            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats_eval, f)
            for k_stat, v_stat in stats_eval.items():
                self.writer.add_scalar(f"{stage}/{k_stat}", v_stat, step)
            self.writer.flush()

    @torch.no_grad()
    def render_traj(self, step: int):
        if self.cfg.disable_video: return
        if self.world_rank != 0: return

        print(f"Running trajectory rendering for step {step}...")
        cfg = self.cfg
        device = self.device

        camtoworlds_all_np = self.parser.camtoworlds_all
        if not len(camtoworlds_all_np):
            print("No camera poses found for trajectory rendering. Skipping.")
            return

        if cfg.render_traj_path == "interp":
            camtoworlds_all_np = generate_interpolated_path(camtoworlds_all_np, 1)
        elif cfg.render_traj_path == "ellipse":
            height_mean = camtoworlds_all_np[:, 2, 3].mean()
            camtoworlds_all_np = generate_ellipse_path_z(camtoworlds_all_np, height=height_mean)
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all_np = generate_spiral_path(
                camtoworlds_all_np, bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf.get("spiral_radius_scale", 0.5),
            )
        else:
            raise ValueError(f"Render trajectory type not supported: {cfg.render_traj_path}")

        camtoworlds_all_np = np.concatenate([
            camtoworlds_all_np,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all_np), axis=0),
        ], axis=1)

        camtoworlds_all_torch = torch.from_numpy(camtoworlds_all_np).float().to(device)

        first_val_cam_key = list(self.parser.Ks_dict.keys())[0] if self.parser.Ks_dict else None
        if not first_val_cam_key:
            print("No camera intrinsics found for trajectory rendering. Skipping.")
            return

        K_traj = torch.from_numpy(self.parser.Ks_dict[first_val_cam_key]).float().to(device)
        width_traj, height_traj = self.parser.imsize_dict[first_val_cam_key]


        video_dir = f"{cfg.result_dir}/videos"
        os.makedirs(video_dir, exist_ok=True)
        video_path = f"{video_dir}/traj_{step}.mp4"
        video_writer = imageio.get_writer(video_path, fps=30)

        all_frame_niqe_scores = []

        for i in tqdm.trange(len(camtoworlds_all_torch), desc="Rendering trajectory"):
            cam_c2w = camtoworlds_all_torch[i : i + 1]
            cam_K = K_traj[None]

            renders_traj, _, _ = self.rasterize_splats(
                camtoworlds=cam_c2w, Ks=cam_K, width=width_traj, height=height_traj,
                sh_degree=cfg.sh_degree, near_plane=cfg.near_plane, far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )
            colors_traj = torch.clamp(renders_traj[..., 0:3], 0.0, 1.0)
            depths_traj = renders_traj[..., 3:4]
            depths_traj_norm = (depths_traj - depths_traj.min()) / (depths_traj.max() - depths_traj.min() + 1e-10)

            if cfg.eval_niqe and self.niqe_metric is not None:
                colors_traj_nchw = colors_traj.permute(0, 3, 1, 2).contiguous()
                niqe_score_traj = self.niqe_metric(colors_traj_nchw)
                all_frame_niqe_scores.append(niqe_score_traj.item())

            canvas_traj_list = [colors_traj, depths_traj_norm.repeat(1, 1, 1, 3)]
            canvas_traj = torch.cat(canvas_traj_list, dim=2).squeeze(0).cpu().numpy()
            canvas_traj_uint8 = (canvas_traj * 255).astype(np.uint8)
            video_writer.append_data(canvas_traj_uint8)
        video_writer.close()
        print(f"Video saved to {video_path}")

        if cfg.eval_niqe and self.niqe_metric is not None and all_frame_niqe_scores:
            avg_traj_niqe = sum(all_frame_niqe_scores) / len(all_frame_niqe_scores)
            print(f"Average NIQE for trajectory video (step {step}): {avg_traj_niqe:.3f} (Lower is better)")
            self.writer.add_scalar(f"render_traj/avg_niqe_step_{step}", avg_traj_niqe, step)
        self.writer.flush()


    @torch.no_grad()
    def run_compression(self, step: int):
        if self.compression_method is None: return
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)
        splats_c = self.compression_method.decompress(compress_dir)
        for k_splat in splats_c.keys():
            self.splats[k_splat].data = splats_c[k_splat].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
            self, camera_state: CameraState, render_tab_state: RenderTabState
    ):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        width = render_tab_state.render_width if render_tab_state.preview_render else render_tab_state.viewer_width
        height = render_tab_state.render_height if render_tab_state.preview_render else render_tab_state.viewer_height

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K_viewer = torch.from_numpy(camera_state.get_K((width, height))).float().to(self.device)

        RENDER_MODE_MAP = {
            "rgb": "RGB", "depth(accumulated)": "D", "depth(expected)": "ED", "alpha": "RGB",
        }

        render_colors_viewer, render_alphas_viewer, info_viewer = self.rasterize_splats(
            camtoworlds=c2w[None], Ks=K_viewer[None], width=width, height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane, far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip, eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device) / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (info_viewer["radii"] > 0).all(-1).sum().item()

        renders_final = None
        if render_tab_state.render_mode == "rgb":
            renders_final = render_colors_viewer[0, ..., 0:3].clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth_viewer = render_colors_viewer[0, ..., 0:1]
            near_plane_norm = render_tab_state.near_plane if render_tab_state.normalize_nearfar else depth_viewer.min()
            far_plane_norm = render_tab_state.far_plane if render_tab_state.normalize_nearfar else depth_viewer.max()
            depth_norm = torch.clip((depth_viewer - near_plane_norm) / (far_plane_norm - near_plane_norm + 1e-10), 0, 1)
            if render_tab_state.inverse: depth_norm = 1 - depth_norm
            renders_final = apply_float_colormap(depth_norm, render_tab_state.colormap).cpu().numpy()
        elif render_tab_state.render_mode == "alpha":
            alpha_viewer = render_alphas_viewer[0, ..., 0:1]
            if render_tab_state.inverse: alpha_viewer = 1 - alpha_viewer
            renders_final = apply_float_colormap(alpha_viewer, render_tab_state.colormap).cpu().numpy()
        return renders_final


def main(local_rank: int, world_rank, world_size: int, cfg_param: Config):
    if world_size > 1 and not cfg_param.disable_viewer:
        cfg_param.disable_viewer = True
        if world_rank == 0: print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg_param)

    if cfg_param.ckpt is not None:
        ckpts_loaded = [
            torch.load(file, map_location=runner.device)
            for file in cfg_param.ckpt
        ]
        if ckpts_loaded:
            if len(cfg_param.ckpt) > 1 and world_size > 1 :
                for k_splat in runner.splats.keys():
                    runner.splats[k_splat].data = torch.cat([c["splats"][k_splat] for c in ckpts_loaded], dim=0)
            else:
                runner.splats.load_state_dict(ckpts_loaded[0]["splats"])

            step_loaded = ckpts_loaded[0]["step"]
            if cfg_param.pose_opt and "pose_adjust" in ckpts_loaded[0]:
                pose_dict = ckpts_loaded[0]["pose_adjust"]
                if world_size > 1: runner.pose_adjust.module.load_state_dict(pose_dict)
                else: runner.pose_adjust.load_state_dict(pose_dict)
            if cfg_param.app_opt and "app_module" in ckpts_loaded[0]:
                app_dict = ckpts_loaded[0]["app_module"]
                if world_size > 1: runner.app_module.module.load_state_dict(app_dict)
                else: runner.app_module.load_state_dict(app_dict)

            print(f"Resuming from checkpoint step {step_loaded}")
            runner.eval(step=step_loaded)
            runner.render_traj(step=step_loaded)
            if cfg_param.compression is not None:
                runner.run_compression(step=step_loaded)
    else:
        runner.train()

    if not cfg_param.disable_viewer and world_rank == 0:
        runner.viewer.complete()
        print("Viewer running... Ctrl+C to exit.")
        try:
            while True: time.sleep(1.0)
        except KeyboardInterrupt:
            print("Exiting viewer.")


BilateralGrid = None
color_correct = None
slice_func = None
total_variation_loss = None

if __name__ == "__main__":
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(strategy=DefaultStrategy(verbose=True)),
        ),
        "mcmc": (
            "Gaussian splatting training using MCMC.",
            Config(
                init_opa=0.5, init_scale=0.1, opacity_reg=0.01, scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    if cfg.use_bilateral_grid or cfg.use_fused_bilagrid:
        if cfg.use_fused_bilagrid:
            cfg.use_bilateral_grid = True
            try:
                from fused_bilagrid import (
                    BilateralGrid as FusedBilateralGrid,
                    color_correct as fused_color_correct,
                    slice as fused_slice,
                    total_variation_loss as fused_total_variation_loss,
                )
                BilateralGrid = FusedBilateralGrid
                color_correct = fused_color_correct
                slice_func = fused_slice
                total_variation_loss = fused_total_variation_loss
                print("Using Fused Bilateral Grid.")
            except ImportError:
                raise ImportError("Fused Bilateral Grid components not found. Please ensure it's installed.")
        else:
            cfg.use_bilateral_grid = True
            try:
                from lib_bilagrid import (
                    BilateralGrid as LibBilateralGrid,
                    color_correct as lib_color_correct,
                    slice as lib_slice,
                    total_variation_loss as lib_total_variation_loss,
                )
                BilateralGrid = LibBilateralGrid
                color_correct = lib_color_correct
                slice_func = lib_slice
                total_variation_loss = lib_total_variation_loss
                print("Using Standard Bilateral Grid (lib_bilagrid).")
            except ImportError:
                raise ImportError("Standard Bilateral Grid (lib_bilagrid) components not found.")

    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except ImportError:
            raise ImportError(
                "To use PNG compression, you need to install torchpq and plas. "
                "torchpq: https://github.com/DeMoriarty/TorchPQ "
                "plas: pip install git+https://github.com/fraunhoferhhi/PLAS.git"
            )

    if cfg.with_ut:
        assert cfg.with_eval3d, "Training with UT requires setting `with_eval3d` flag."

    cli(main, cfg, verbose=True)