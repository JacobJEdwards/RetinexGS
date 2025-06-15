import json
import math
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Any

import imageio
import numpy as np
import piq
import torch
import torch.nn.functional as F
import tqdm
import tyro
import viser
import yaml
from nerfview import CameraState, RenderTabState, apply_float_colormap
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ExponentialLR, ChainedScheduler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
    generate_novel_views,
)
from losses import ColourConsistencyLoss, ExposureLoss, SpatialLoss
from rendering_double import rasterization_dual
from utils import RetinexNet
from gsplat import export_splats
from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from utils import (
    AppearanceOptModule,
    CameraOptModule,
    knn,
    rgb_to_sh,
    set_random_seed,
    generate_variational_intrinsics,
)


@dataclass
class Config:
    disable_viewer: bool = True
    ckpt: list[str] | None = None
    compression: Literal["png"] | None = None
    render_traj_path: str = "interp"

    data_dir: Path = Path("../../colmap")
    data_factor: int = 1
    result_dir: Path = Path("../../result")
    test_every: int = 8
    patch_size: int | None = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    port: int = 4000

    batch_size: int = 1
    steps_scaler: float = 1.0

    max_steps: int = 30_000
    eval_steps: list[int] = field(default_factory=lambda: [3000, 7_000, 14_000, 30_000])
    save_steps: list[int] = field(default_factory=lambda: [3000, 7_000, 14_000, 30_000])
    save_ply: bool = True
    ply_steps: list[int] = field(default_factory=lambda: [3000, 7_000, 14_000, 30_000])
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

    strategy: DefaultStrategy | MCMCStrategy = field(default_factory=DefaultStrategy)
    packed: bool = False
    sparse_grad: bool = False
    visible_adam: bool = True
    antialiased: bool = False

    random_bkgd: bool = True

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    pose_opt: bool = True
    pose_opt_lr: float = 1e-5
    pose_opt_reg: float = 1e-6
    pose_noise: float = 0.0

    app_opt: bool = False
    app_embed_dim: int = 16
    app_opt_lr: float = 1e-3
    app_opt_reg: float = 1e-6

    use_bilateral_grid: bool = False
    bilateral_grid_shape: tuple[int, int, int] = (16, 16, 8)

    depth_loss: bool = False
    depth_lambda: float = 1e-2

    tb_every: int = 100
    tb_save_image: bool = True

    lpips_net: Literal["vgg", "alex"] = "alex"

    with_ut: bool = False
    with_eval3d: bool = False

    use_fused_bilagrid: bool = False

    enable_clipiqa_loss: bool = False
    clipiqa_lambda: float = 10
    clipiqa_model_type: Literal["clipiqa"] = "clipiqa"
    num_novel_views: int = 100
    num_novel_to_render: int = 4
    novel_view_translation_pertube: float = 0.5
    novel_view_rotation_pertube: float = 20.0
    enable_loss_schedule: bool = True
    clipiqa_lambda_start_factor: float = 0.01
    clipiqa_lambda_warmup_steps: int = 5000
    clipiqa_novel_view_frequency: int = 8

    enable_hard_view_mining: bool = False
    hard_view_mining_pool_size: int = 200
    hard_view_mining_every: int = 1000
    hard_view_mining_batch_size: int = 4

    enable_variational_intrinsics: bool = False
    focal_length_perturb_factor: float = 0.1
    principal_point_perturb_pixel: int = 10


    enable_retinex: bool = True
    lambda_reflect = 0.4
    lambda_smooth = 0.1
    lambda_low = 1.0
    lambda_color = 0.1
    lambda_exposure = 0.1

    eval_niqe: bool = False

    def adjust_steps(self, factor: float) -> None:
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
    adjust_k = torch.nn.Parameter(torch.ones_like(colors[:, :1, :]), requires_grad=True)    # enhance, for multiply
    adjust_b = torch.nn.Parameter(torch.zeros_like(colors[:, :1, :]), requires_grad=True)   # bias, for add

    params.append(("adjust_k", adjust_k, sh0_lr))
    params.append(("adjust_b", adjust_b, sh0_lr))


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
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        cfg.result_dir.mkdir(exist_ok=True)
        self.ckpt_dir = cfg.result_dir / "ckpts"
        self.ckpt_dir.mkdir(exist_ok=True)
        self.stats_dir = cfg.result_dir / "stats"
        self.stats_dir.mkdir(exist_ok=True)
        self.render_dir = cfg.result_dir / "renders"
        self.render_dir.mkdir(exist_ok=True)
        self.ply_dir = cfg.result_dir / "ply"
        self.ply_dir.mkdir(exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(cfg.result_dir / "tb"))

        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
        )
        self.trainset = Dataset(
            self.parser, patch_size=cfg.patch_size, load_depths=cfg.depth_loss
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.retinex_net = RetinexNet().to(self.device)
        self.retinex_optimizer = torch.optim.Adam(
            self.retinex_net.parameters(),
            lr=1e-4 * math.sqrt(cfg.batch_size),
            weight_decay=1e-4,
        )
        self.retinex_embed_dim = 32
        self.retinex_embeds = nn.Embedding(len(self.trainset), self.retinex_embed_dim).to(self.device)
        torch.nn.init.zeros_(self.retinex_embeds.weight)

        self.retinex_embed_optimizer = torch.optim.Adam(
            [{"params": self.retinex_embeds.parameters(), "lr": 1e-3}]
        )

        # self.loss_color = ColourConsistencyLoss().to(self.device)
        self.loss_exposure = ExposureLoss(patch_size=16, mean_val=0.4).to(self.device)

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

        self.hard_view_candidate_poses = None
        self.hard_view_indices = None
        if cfg.enable_clipiqa_loss:
            print("Initializing candidate pool for Online Hard View Mining...")
            candidate_poses_np = generate_interpolated_path(
                self.parser.camtoworlds, n_interp=5
            )
            if candidate_poses_np.shape[1] == 3:
                print("Padding interpolated poses from 3x4 to 4x4...")
                bottom_row = np.array([[[0.0, 0.0, 0.0, 1.0]]])
                repeated_bottom_row = np.repeat(
                    bottom_row, len(candidate_poses_np), axis=0
                )
                candidate_poses_np = np.concatenate(
                    [candidate_poses_np, repeated_bottom_row], axis=1
                )

            perturbed_poses_np = generate_novel_views(
                self.parser.camtoworlds,
                num_novel_views=cfg.hard_view_mining_pool_size // 2,
                translation_perturbation=cfg.novel_view_translation_pertube,
                rotation_perturbation=cfg.novel_view_rotation_pertube,
            )
            all_poses_np = np.concatenate(
                [candidate_poses_np, perturbed_poses_np], axis=0
            )
            np.random.shuffle(all_poses_np)

            self.hard_view_candidate_poses = (
                torch.from_numpy(all_poses_np[: cfg.hard_view_mining_pool_size])
                .float()
                .to(self.device)
            )
            print(
                f"Created a pool of {len(self.hard_view_candidate_poses)} candidate poses for hard mining."
            )

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
            torch.nn.init.zeros_(cast(Tensor, self.app_module.color_head[-1].weight))
            torch.nn.init.zeros_(cast(Tensor, self.app_module.color_head[-1].bias))
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

        if cfg.enable_clipiqa_loss:
            if cfg.clipiqa_model_type == "clipiqa":
                self.clipiqa_model = piq.CLIPIQA(data_range=1.0).to(self.device)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            global BilateralGrid
            assert BilateralGrid is not None
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
                )
            ]

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

        self.niqe_metric = None
        if cfg.eval_niqe:
            try:
                self.niqe_metric = piq.BRISQUELoss(data_range=1.0).to(self.device)
                print("BRISQUE metric initialized for evaluation.")
            except Exception as e:
                print(
                    f"Error initializing BRISQUE: {e}. BRISQUE evaluation will be skipped."
                )
                cfg.eval_niqe = False

        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = GsplatViewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                output_dir=Path(cfg.result_dir),
                mode="training",
            )

        # Running stats for prunning & growing.
        n_gauss = len(self.splats["means"])
        self.running_stats = {
            "grad2d": torch.zeros(n_gauss, device=self.device),  # norm of the gradient
            "count": torch.zeros(n_gauss, device=self.device, dtype=torch.int),
        }

    @torch.no_grad()
    def find_hard_views(self, sh_degree_to_use: int):
        print(
            f"\nPerforming Online Hard View Mining (World Rank: {self.world_rank})..."
        )
        cfg = self.cfg
        device = self.device
        all_scores = []

        if not self.parser.Ks_dict or not self.parser.imsize_dict:
            print(
                "Warning: Ks_dict or imsize_dict is empty. Skipping hard view mining."
            )
            self.hard_view_indices = None
            return

        first_cam_key = list(self.parser.Ks_dict.keys())[0]
        K_hard_view = (
            torch.from_numpy(self.parser.Ks_dict[first_cam_key]).float().to(device)
        )
        width, height = self.parser.imsize_dict[first_cam_key]

        base_c2w_for_novel_views_np = np.copy(self.parser.camtoworlds)

        if (
            base_c2w_for_novel_views_np.shape[1] == 3
            and base_c2w_for_novel_views_np.shape[2] == 4
        ):  # (N, 3, 4)
            print(
                "Padding base_c2w_for_novel_views_np from 3x4 to 4x4 for hard view generation..."
            )
            bottom_row = np.array([[[0.0, 0.0, 0.0, 1.0]]])
            repeated_bottom_row = np.repeat(
                bottom_row, len(base_c2w_for_novel_views_np), axis=0
            )
            base_c2w_for_novel_views_np = np.concatenate(
                [base_c2w_for_novel_views_np, repeated_bottom_row], axis=1
            )
        elif base_c2w_for_novel_views_np.shape[1:] != (4, 4):
            print(
                f"Warning: Unexpected base_c2w_for_novel_views_np shape {base_c2w_for_novel_views_np.shape}. Skipping hard view mining."
            )
            self.hard_view_indices = None
            return

        print("Initializing candidate pool for Online Hard View Mining...")
        candidate_poses_np = generate_interpolated_path(
            base_c2w_for_novel_views_np, n_interp=5
        )

        if candidate_poses_np.shape[1] == 3 and candidate_poses_np.shape[2] == 4:
            print(
                "Padding interpolated poses from 3x4 to 4x4 for hard view mining pool..."
            )
            bottom_row = np.array([[[0.0, 0.0, 0.0, 1.0]]])
            repeated_bottom_row = np.repeat(bottom_row, len(candidate_poses_np), axis=0)
            candidate_poses_np = np.concatenate(
                [candidate_poses_np, repeated_bottom_row], axis=1
            )
        elif (
            candidate_poses_np.shape[1:] != (4, 4) and candidate_poses_np.size > 0
        ):  # if not empty and not 4x4
            print(
                f"Warning: Unexpected candidate_poses_np shape {candidate_poses_np.shape} after interpolation. Skipping hard view mining."
            )
            self.hard_view_indices = None
            return

        perturbed_poses_np = generate_novel_views(
            base_c2w_for_novel_views_np,
            num_novel_views=cfg.hard_view_mining_pool_size // 2,
            translation_perturbation=cfg.novel_view_translation_pertube,
            rotation_perturbation=cfg.novel_view_rotation_pertube,
        )

        if candidate_poses_np.size == 0 and perturbed_poses_np.size == 0:
            print(
                "Warning: No candidate or perturbed poses generated for hard view mining. Skipping."
            )
            self.hard_view_indices = None
            return
        elif candidate_poses_np.size == 0:
            all_poses_np = perturbed_poses_np
        elif perturbed_poses_np.size == 0:
            all_poses_np = candidate_poses_np
        else:
            all_poses_np = np.concatenate(
                [candidate_poses_np, perturbed_poses_np], axis=0
            )

        np.random.shuffle(all_poses_np)

        self.hard_view_candidate_poses = (
            torch.from_numpy(all_poses_np[: cfg.hard_view_mining_pool_size])
            .float()
            .to(self.device)
        )

        if len(self.hard_view_candidate_poses) == 0:
            print(
                "Warning: Hard view candidate pose pool is empty. Skipping hard view mining."
            )
            self.hard_view_indices = None
            return

        print(
            f"Created a pool of {len(self.hard_view_candidate_poses)} candidate poses for hard mining."
        )

        if cfg.enable_variational_intrinsics:
            K_hard_view_batch = generate_variational_intrinsics(
                K_hard_view,
                num_intrinsics=len(self.hard_view_candidate_poses),
                focal_perturb_factor=cfg.focal_length_perturb_factor,
                principal_point_perturb_pixel=cfg.principal_point_perturb_pixel,
            )
        else:
            K_hard_view_batch = K_hard_view.unsqueeze(0).expand(
                len(self.hard_view_candidate_poses), -1, -1
            )

        batch_size = 16
        for i in tqdm.tqdm(
            range(0, len(self.hard_view_candidate_poses), batch_size),
            desc="Mining Hard Views",
        ):
            batch_poses = self.hard_view_candidate_poses[i : i + batch_size]
            batch_Ks = K_hard_view_batch[i : i + batch_size]

            if not cfg.enable_retinex:
                renders, _, _ = self.rasterize_splats(
                    camtoworlds=batch_poses,
                    Ks=batch_Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    render_mode="RGB",
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                )
            else:
                renders, _, _, _, _ = self.rasterize_splats(
                    camtoworlds=batch_poses,
                    Ks=batch_Ks,
                    width=width,
                    height=height,
                    sh_degree=sh_degree_to_use,
                    render_mode="RGB",
                    near_plane=cfg.near_plane,
                    far_plane=cfg.far_plane,
                )
            colors_nchw = renders.permute(0, 3, 1, 2).contiguous()
            colors_nchw = torch.clamp(colors_nchw, 0.0, 1.0)
            scores = self.clipiqa_model(colors_nchw)
            all_scores.append(scores)

        if not all_scores:
            print(
                "Warning: No scores generated during hard view mining. Skipping update to hard_view_indices."
            )
            self.hard_view_indices = None
            return

        all_scores = torch.cat(all_scores)
        sorted_indices = torch.argsort(all_scores)
        self.hard_view_indices = sorted_indices
        print(
            f"Hard View Mining complete. Hardest score: {all_scores.min():.4f}, Easiest score: {all_scores.max():.4f}. Pool size: {len(self.hard_view_indices)}\n"
        )

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Tensor | None = None,
        rasterize_mode: Literal["classic", "antialiased"] | None = None,
        **kwargs,
    ) -> tuple[Tensor, Tensor, dict[str, Any]] | tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
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
            rasterize_mode: Literal["antialiased", "classic"] = (
                "antialiased" if self.cfg.antialiased else "classic"
            )

        if not cfg.enable_retinex:
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

        adjust_k = self.splats["adjust_k"]  # 1090, 1, 3
        adjust_b = self.splats["adjust_b"]  # 1090, 1, 3

        colors_low = colors * adjust_k + adjust_b  # least squares: x_enh=a*x+b

        render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info = rasterization_dual(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            colors_low=colors_low,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
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
            **kwargs,)

        return render_colors_enh, render_colors_low, render_enh_alphas, render_low_alphas, info   # return colors and alphas


    def pre_train_retinex(self):
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

        loss_contrast = SpatialLoss().to(device)

        pbar = tqdm.tqdm(range(3000), desc="Pre-training RetinexNet")
        for step in pbar:
            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            images_ids = data["image_id"].to(device)
            pixels = data["image"].to(device) / 255.0
            input_image_for_net = pixels.permute(0, 3, 1, 2)

            self.retinex_optimizer.zero_grad()
            self.retinex_embed_optimizer.zero_grad()

            retinex_embedding = self.retinex_embeds(images_ids)

            log_illumination_map = self.retinex_net(input_image_for_net, retinex_embedding)  # [B, 3, H, W]
            illumination_map = torch.exp(log_illumination_map)

            loss_spa_val = loss_contrast(input_image_for_net, illumination_map)
            # loss_color_val = self.loss_color(
            #     illumination_map
            # )
            loss_exposure_val = self.loss_exposure(
                illumination_map
            )

            loss = (
                cfg.lambda_reflect * loss_spa_val
                # + cfg.lambda_color * loss_color_val
                + cfg.lambda_exposure * loss_exposure_val
            )

            loss.backward()

            self.retinex_optimizer.step()
            self.retinex_embed_optimizer.step()

            pbar.set_postfix({"loss": loss.item()})

            if step % 100 == 0:
                self.writer.add_scalar("retinex_net/loss", loss.item(), step)
                self.writer.add_scalar(
                    "retinex_net/loss_spatial", loss_spa_val.item(), step
                )
                # self.writer.add_scalar(
                #     "retinex_net/loss_color", loss_color_val.item(), step
                # )
                self.writer.add_scalar(
                    "retinex_net/loss_exposure", loss_exposure_val.item(), step
                )


    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        loss_contrast = SpatialLoss()

        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers: list[ExponentialLR | ChainedScheduler] = [
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
            torch.optim.lr_scheduler.ExponentialLR(
                self.retinex_optimizer, gamma=0.01 ** (1.0 / max_steps)
            )
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

            current_clipiqa_lambda = cfg.clipiqa_lambda
            if cfg.enable_loss_schedule and step < cfg.clipiqa_lambda_warmup_steps:
                warmup_factor = min(1.0, step / cfg.clipiqa_lambda_warmup_steps)
                start_factor = cfg.clipiqa_lambda_start_factor
                current_clipiqa_lambda = cfg.clipiqa_lambda * (
                    start_factor + (1.0 - start_factor) * warmup_factor
                )

            sh_degree_to_use = min(
                step // cfg.sh_degree_interval, cfg.sh_degree
            )  # Defined early
            if (
                cfg.enable_hard_view_mining
                and cfg.enable_clipiqa_loss
                and self.clipiqa_model is not None
                and step > 0
                and step % cfg.hard_view_mining_every == 0
            ):
                self.find_hard_views(sh_degree_to_use=sh_degree_to_use)

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

            retinex_embedding = self.retinex_embeds(image_ids)
            input_image_for_net = pixels.permute(0, 3, 1, 2)
            log_input_image = torch.log(input_image_for_net + 1e-8)  # [1, 3, H, W]
            log_illumination_map = self.retinex_net(input_image_for_net, retinex_embedding)  # [1, 3, H, W]
            log_reflectance_target = log_input_image - log_illumination_map

            reflectance_target = torch.exp(log_reflectance_target)
            reflectance_target = torch.clamp(reflectance_target, 0, 1)

            illumination_map = torch.exp(log_illumination_map)  # [1, 3, H, W]

            if cfg.enable_retinex:
                renders_enh, renders_low, alphas_enh, alphas_low, info = self.rasterize_splats(
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
            else:
                renders_low, alphas_low, info = self.rasterize_splats(
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
                renders_enh, alphas_enh = renders_low, alphas_low

            if renders_low.shape[-1] == 4:
                colors_low, depths_low = renders_low[..., 0:3], renders_low[..., 3:4]
                colors_enh, depths_enh = renders_enh[..., 0:3], renders_enh[..., 3:4]
            else:
                colors_low, depths_low = renders_low, None
                colors_enh, depths_enh = renders_enh, None

            if cfg.use_bilateral_grid:
                global slice, color_correct
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)

                colors_low = slice(
                    self.bil_grids,
                    grid_xy.expand(colors_low.shape[0], -1, -1, -1),
                    colors_low,
                    image_ids.unsqueeze(-1),
                )["rgb"]

                colors_enh = slice(
                    self.bil_grids,
                    grid_xy.expand(colors_enh.shape[0], -1, -1, -1),
                    colors_enh,
                    image_ids.unsqueeze(-1),
                )["rgb"]


            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors_low += bkgd * (1.0 - alphas_low)
                colors_enh += bkgd * (1.0 - alphas_enh)

            info["means2d"].retain_grad()

            reflectance_target_permuted = reflectance_target.permute(0, 2, 3, 1)  # [1, H, W, 3]

            loss_reflectance = F.l1_loss(colors_enh, reflectance_target_permuted)

            loss_reconstruct_low = F.l1_loss(colors_low, pixels)

            ssim_loss_low = 1.0 - self.ssim(
                colors_low.permute(0, 3, 1, 2),
                pixels.permute(0, 3, 1, 2),
            )


            con_degree = (0.5 / torch.mean(pixels).item())

            loss_illum_smooth = loss_contrast(
                input_image_for_net,
                illumination_map,
                contrast=con_degree,
            )

            low_loss = (loss_reconstruct_low * (1.0 - cfg.ssim_lambda)) + (
                    ssim_loss_low * cfg.ssim_lambda
            )

            # some problems here
            # whcih is the issue ?
            # loss_illum_color = self.loss_color(
            #     illumination_map
            # )
            loss_illum_exposure = self.loss_exposure(
                illumination_map
            )

            loss_illumination = (
                cfg.lambda_reflect * loss_illum_smooth
                + cfg.lambda_exposure * loss_illum_exposure
                # + cfg.lambda_color * loss_illum_color
            )

            loss = (
                cfg.lambda_reflect * loss_reflectance
                + low_loss
                + loss_illumination
            )


            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )


            clipiqa_loss_value = torch.tensor(0.0, device=device)
            clipiqa_score_value = torch.tensor(0.0, device=device)

            apply_clipiqa_novel_view_loss = (
                cfg.enable_clipiqa_loss
                and self.clipiqa_model is not None
                and current_clipiqa_lambda > 0
                and step % cfg.clipiqa_novel_view_frequency == 0
            )

            if apply_clipiqa_novel_view_loss:
                if not self.parser.Ks_dict or not self.parser.imsize_dict:
                    if world_rank == 0:
                        print(
                            f"Step {step}: Warning: Ks_dict or imsize_dict is empty. Skipping CLIPIQA novel view loss."
                        )
                else:
                    first_cam_key = list(self.parser.Ks_dict.keys())[0]
                    K_novel_base = (
                        torch.from_numpy(self.parser.Ks_dict[first_cam_key])
                        .float()
                        .to(device)
                    )
                    width_novel, height_novel = self.parser.imsize_dict[first_cam_key]

                    novel_c2w_for_loss = None
                    num_actual_views_for_clipiqa = 0

                    if (
                        cfg.enable_hard_view_mining
                        and self.hard_view_candidate_poses is not None
                        and self.hard_view_indices is not None
                        and len(self.hard_view_indices) > 0
                    ):
                        num_to_sample = min(
                            cfg.hard_view_mining_batch_size, len(self.hard_view_indices)
                        )
                        selected_indices = self.hard_view_indices[:num_to_sample]
                        novel_c2w_for_loss = self.hard_view_candidate_poses[
                            selected_indices
                        ]
                        num_actual_views_for_clipiqa = novel_c2w_for_loss.shape[0]
                        if world_rank == 0 and step % (cfg.tb_every * 10) == 0:
                            print(
                                f"Step {step}: Using {num_actual_views_for_clipiqa} hard views for CLIPIQA loss."
                            )

                    if novel_c2w_for_loss is None or num_actual_views_for_clipiqa == 0:
                        base_poses_np = np.copy(self.parser.camtoworlds)
                        if base_poses_np.shape[0] > 0:
                            if (
                                base_poses_np.shape[1] == 3
                                and base_poses_np.shape[2] == 4
                            ):
                                bottom_row = np.array([[[0.0, 0.0, 0.0, 1.0]]])
                                repeated_bottom_row = np.repeat(
                                    bottom_row, len(base_poses_np), axis=0
                                )
                                base_poses_np = np.concatenate(
                                    [base_poses_np, repeated_bottom_row], axis=1
                                )

                        if base_poses_np.shape[1:] == (4, 4):
                            novel_c2w_np = generate_novel_views(
                                base_poses=base_poses_np,
                                num_novel_views=cfg.num_novel_to_render,
                                translation_perturbation=cfg.novel_view_translation_pertube,
                                rotation_perturbation=cfg.novel_view_rotation_pertube,
                            )
                            if novel_c2w_np.size > 0:
                                novel_c2w_for_loss = (
                                    torch.from_numpy(novel_c2w_np).float().to(device)
                                )
                                num_actual_views_for_clipiqa = novel_c2w_for_loss.shape[
                                    0
                                ]
                                if world_rank == 0 and step % (cfg.tb_every * 10) == 0:
                                    print(
                                        f"Step {step}: Generated {num_actual_views_for_clipiqa} novel views for CLIPIQA loss."
                                    )
                        else:
                            if world_rank == 0:
                                print(
                                    f"Step {step}: Warning: Base poses for novel view generation are not 4x4. Skipping CLIPIQA."
                                )
                    else:
                        if world_rank == 0:
                            print(
                                f"Step {step}: Warning: No base poses available. Skipping CLIPIQA."
                            )

                    if (
                        novel_c2w_for_loss is not None
                        and num_actual_views_for_clipiqa > 0
                    ):
                        if cfg.enable_variational_intrinsics:
                            K_novel_for_loss = generate_variational_intrinsics(
                                K_novel_base,
                                num_intrinsics=num_actual_views_for_clipiqa,
                                focal_perturb_factor=cfg.focal_length_perturb_factor,
                                principal_point_perturb_pixel=cfg.principal_point_perturb_pixel,
                            )
                        else:
                            K_novel_for_loss = K_novel_base.unsqueeze(0).expand(
                                num_actual_views_for_clipiqa, -1, -1
                            )


                        out = self.rasterize_splats(
                            camtoworlds=novel_c2w_for_loss,
                            Ks=K_novel_for_loss,
                            width=width_novel,
                            height=height_novel,
                            sh_degree=sh_degree_to_use,
                            render_mode="RGB",
                            near_plane=cfg.near_plane,
                            far_plane=cfg.far_plane,
                        )

                        if cfg.enable_retinex:
                            novel_renders, _, _, _, _ = out
                        else:
                            novel_renders, _, _ = out

                        colors_nchw = novel_renders.permute(0, 3, 1, 2).contiguous()
                        colors_nchw = torch.clamp(colors_nchw, 0.0, 1.0)

                        current_clipiqa_score = self.clipiqa_model(colors_nchw).mean()
                        clipiqa_score_value = current_clipiqa_score
                        temp_clipiqa_loss_value = -clipiqa_score_value
                        loss = loss + current_clipiqa_lambda * temp_clipiqa_loss_value
                        clipiqa_loss_value = temp_clipiqa_loss_value

            depthloss_value = torch.tensor(0.0, device=device)

            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                        ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths_low = F.grid_sample(
                    depths_low.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths_low = depths_low.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths_low > 0.0, 1.0 / depths_low, torch.zeros_like(depths_low))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
                
                if cfg.enable_retinex:
                    # query depths from depth map
                    points = torch.stack(
                        [
                            points[:, :, 0] / (width - 1) * 2 - 1,
                            points[:, :, 1] / (height - 1) * 2 - 1,
                            ],
                        dim=-1,
                    )  # normalize to [-1, 1]
                    grid = points.unsqueeze(2)  # [1, M, 1, 2]
                    depths_enh = F.grid_sample(
                        depths_enh.permute(0, 3, 1, 2), grid, align_corners=True
                    )  # [1, 1, M, 1]
                    depths_enh = depths_enh.squeeze(3).squeeze(1)  # [1, M]
                    # calculate loss in disparity space
                    disp = torch.where(depths_enh > 0.0, 1.0 / depths_enh, torch.zeros_like(depths_enh))
                    disp_gt = 1.0 / depths_gt  # [1, M]
                    depthloss_enh = F.l1_loss(disp, disp_gt) * self.scene_scale
                    loss += depthloss_enh * cfg.depth_lambda
                    

            tvloss_value: Tensor = torch.tensor(0.0, device=device)
            if cfg.use_bilateral_grid:
                tvloss_value = cast(
                    Tensor, 10 * total_variation_loss(self.bil_grids.grids)
                )
                loss += tvloss_value

            if cfg.opacity_reg > 0.0:
                loss += (
                    cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                loss += (
                    cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                )

            loss.backward()

            desc_parts = [f"loss={loss.item():.3f}"]
            if cfg.enable_clipiqa_loss and clipiqa_score_value.item() != 0.0:
                desc_parts.append(
                    f"clipiqa={clipiqa_score_value.item():.3f} (λ={current_clipiqa_lambda:.2f})"
                )
            if cfg.enable_retinex:
                desc_parts.append(
                    f"retinex_loss={loss_reflectance.item():.3f} "
                    f"illum_smooth={loss_illum_smooth.item():.3f} "
                    # f"illum_color={loss_illum_color.item():.3f} "
                    f"illum_exposure={loss_illum_exposure.item():.3f}"
                )
            desc_parts.append(f"sh_deg={sh_degree_to_use}")
            if cfg.depth_loss:
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
                self.writer.add_scalar("train/l1loss", loss_reflectance.item(), step)
                self.writer.add_scalar("train/ssimloss", ssim_loss_low.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.enable_retinex:
                    self.writer.add_scalar(
                        "train/reflectance_loss", loss_reflectance.item(), step
                    )
                    self.writer.add_scalar(
                        "train/illumination_smooth", loss_illum_smooth.item(), step
                    )
                    # self.writer.add_scalar(
                    #     "train/illumination_color", loss_illum_color.item(), step
                    # )
                    self.writer.add_scalar(
                        "train/illumination_exposure", loss_illum_exposure.item(), step
                    )
                if cfg.enable_clipiqa_loss:
                    self.writer.add_scalar(
                        "train/clipiqa_score", clipiqa_score_value.item(), step
                    )
                    self.writer.add_scalar(
                        "train/clipiqa_loss", clipiqa_loss_value.item(), step
                    )
                    self.writer.add_scalar(
                        "train/clipiqa_lambda", current_clipiqa_lambda, step
                    )
                if cfg.depth_loss:
                    self.writer.add_scalar(
                        "train/depthloss", depthloss_value.item(), step
                    )
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss_value.item(), step)
                if cfg.tb_save_image:
                    canvas_tb = (
                        torch.cat([pixels, colors_low], dim=2).detach().cpu().numpy()
                    )
                    canvas_tb = canvas_tb.reshape(-1, *canvas_tb.shape[2:])
                    self.writer.add_image(
                        "train/render", canvas_tb, step, dataformats="HWC"
                    )

                    canvas_enh = (
                        torch.cat([pixels, colors_enh], dim=2).detach().cpu().numpy()
                    )
                    canvas_enh = canvas_enh.reshape(-1, *canvas_enh.shape[2:])
                    self.writer.add_image(
                        "train/render_enh", canvas_enh, step, dataformats="HWC"
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
                data_save = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    data_save["pose_adjust"] = (
                        self.pose_adjust.module.state_dict()
                        if world_size > 1
                        else self.pose_adjust.state_dict()
                    )
                if cfg.app_opt:
                    data_save["app_module"] = (
                        self.app_module.module.state_dict()
                        if world_size > 1
                        else self.app_module.state_dict()
                    )
                if cfg.use_bilateral_grid:
                    data_save["bil_grids"] = (
                        self.bil_grids.module.state_dict()
                        if world_size > 1
                        else self.bil_grids.state_dict()
                    )

                if cfg.enable_retinex:
                    data_save["retinex_net"] = (
                        self.retinex_net.module.state_dict()
                        if world_size > 1
                        else self.retinex_net.state_dict()
                    )
                torch.save(
                    data_save, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
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
                    shN_export = torch.empty(
                        [sh0_export.shape[0], 0, 3], device=sh0_export.device
                    )
                else:
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

            if cfg.sparse_grad:
                assert cfg.packed
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad_val = self.splats[k].grad
                    if grad_val is None or grad_val.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],
                        values=grad_val[gaussian_ids],
                        size=self.splats[k].size(),
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).all(-1).any(0)

            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad()
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad()
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad()
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad()

            self.retinex_optimizer.step()
            self.retinex_optimizer.zero_grad()
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
                    packed=cfg.packed,
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
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (max(time.time() - tic, 1e-10))
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                self.viewer.render_tab_state.num_train_rays_per_sec = (
                    num_train_rays_per_sec
                )
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

            out = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )

            if cfg.enable_retinex:
                colors, _, alphas, _, info = out
            else:
                colors, alphas, info = out

            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)

            colors = torch.clamp(colors, 0.0, 1.0)

            if world_rank == 0:
                canvas_list = [pixels, colors]
                canvas_eval = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas_eval = (canvas_eval * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas_eval,
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
                }
            )

            print_parts_eval = [
                f"PSNR: {stats_eval.get('psnr', 0):.3f}",
                f"SSIM: {stats_eval.get('ssim', 0):.4f}",
                f"LPIPS: {stats_eval.get('lpips', 0):.3f}",
            ]
            if cfg.eval_niqe and self.niqe_metric is not None and "niqe" in stats_eval:
                print_parts_eval.append(
                    f"BRISQUE: {stats_eval.get('niqe', 0):.3f} (lower is better)"
                )
            if cfg.use_bilateral_grid:
                print_parts_eval.extend(
                    [
                        f"CC_PSNR: {stats_eval.get('cc_psnr', 0):.3f}",
                        f"CC_SSIM: {stats_eval.get('cc_ssim', 0):.4f}",
                        f"CC_LPIPS: {stats_eval.get('cc_lpips', 0):.3f}",
                    ]
                )
            print_parts_eval.extend(
                [
                    f"Time: {stats_eval.get('ellipse_time', 0):.3f}s/image",
                    f"GS: {stats_eval.get('num_GS', 0)}",
                ]
            )
            print(f"Eval {stage} Step {step}: " + " | ".join(print_parts_eval))

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
        video_path = f"{video_dir}/traj_{step}.mp4"
        video_writer = imageio.get_writer(video_path, fps=30)

        all_frame_niqe_scores = []

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

            if cfg.enable_retinex:
                renders_traj, _, _, _, _ = out
            else:
                renders_traj, _, _ = out


            colors_traj = torch.clamp(renders_traj[..., 0:3], 0.0, 1.0)
            depths_traj = renders_traj[..., 3:4]
            depths_traj_norm = (depths_traj - depths_traj.min()) / (
                depths_traj.max() - depths_traj.min() + 1e-10
            )

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
            print(
                f"Average BRISQUE for trajectory video (step {step}): {avg_traj_niqe:.3f} (Lower is better)"
            )
            self.writer.add_scalar(
                f"render_traj/avg_niqe_step_{step}", avg_traj_niqe, step
            )
        self.writer.flush()

    @torch.no_grad()
    def run_compression(self, step: int):
        if self.compression_method is None:
            return
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
        width = (
            render_tab_state.render_width
            if render_tab_state.preview_render
            else render_tab_state.viewer_width
        )
        height = (
            render_tab_state.render_height
            if render_tab_state.preview_render
            else render_tab_state.viewer_height
        )

        c2w = torch.from_numpy(camera_state.c2w).float().to(self.device)
        K_viewer = (
            torch.from_numpy(camera_state.get_K((width, height)))
            .float()
            .to(self.device)
        )

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
        }

        render_colors_viewer, render_alphas_viewer, info_viewer = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K_viewer[None],
            width=width,
            height=height,
            sh_degree=min(render_tab_state.max_sh_degree, self.cfg.sh_degree),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=self.device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
        )
        render_tab_state.total_gs_count = len(self.splats["means"])
        render_tab_state.rendered_gs_count = (
            (info_viewer["radii"] > 0).all(-1).sum().item()
        )

        renders_final = None
        if render_tab_state.render_mode == "rgb":
            renders_final = render_colors_viewer[0, ..., 0:3].clamp(0, 1).cpu().numpy()
        elif render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            depth_viewer = render_colors_viewer[0, ..., 0:1]
            near_plane_norm = (
                render_tab_state.near_plane
                if render_tab_state.normalize_nearfar
                else depth_viewer.min()
            )
            far_plane_norm = (
                render_tab_state.far_plane
                if render_tab_state.normalize_nearfar
                else depth_viewer.max()
            )
            depth_norm = torch.clip(
                (depth_viewer - near_plane_norm)
                / (far_plane_norm - near_plane_norm + 1e-10),
                0,
                1,
            )
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders_final = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha_viewer = render_alphas_viewer[0, ..., 0:1]
            if render_tab_state.inverse:
                alpha_viewer = 1 - alpha_viewer
            renders_final = (
                apply_float_colormap(alpha_viewer, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        return renders_final


def main(local_rank: int, world_rank, world_size: int, cfg_param: Config):
    if world_size > 1 and not cfg_param.disable_viewer:
        cfg_param.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg_param)

    if cfg_param.ckpt is not None:
        ckpts_loaded = [
            torch.load(file, map_location=runner.device) for file in cfg_param.ckpt
        ]
        if ckpts_loaded:
            if len(cfg_param.ckpt) > 1 and world_size > 1:
                for k_splat in runner.splats.keys():
                    runner.splats[k_splat].data = torch.cat(
                        [c["splats"][k_splat] for c in ckpts_loaded], dim=0
                    )
            else:
                runner.splats.load_state_dict(ckpts_loaded[0]["splats"])

            step_loaded = ckpts_loaded[0]["step"]
            if cfg_param.pose_opt and "pose_adjust" in ckpts_loaded[0]:
                pose_dict = ckpts_loaded[0]["pose_adjust"]
                if world_size > 1:
                    runner.pose_adjust.module.load_state_dict(pose_dict)
                else:
                    runner.pose_adjust.load_state_dict(pose_dict)
            if cfg_param.app_opt and "app_module" in ckpts_loaded[0]:
                app_dict = ckpts_loaded[0]["app_module"]
                if world_size > 1:
                    runner.app_module.module.load_state_dict(app_dict)
                else:
                    runner.app_module.load_state_dict(app_dict)

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
            while True:
                time.sleep(1.0)
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
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
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
                raise ImportError(
                    "Fused Bilateral Grid components not found. Please ensure it's installed."
                )
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
                raise ImportError(
                    "Standard Bilateral Grid (lib_bilagrid) components not found."
                )

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
