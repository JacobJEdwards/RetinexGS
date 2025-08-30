from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from gsplat import MCMCStrategy, DefaultStrategy


@dataclass
class Config:
    disable_viewer: bool = True
    ckpt: list[str] | None = None
    compression: Literal["png"] | None = None
    render_traj_path: str = "interp"

    data_dir: Path = Path("/workspace/360_v2/bicycle")
    data_factor: int = 1
    result_dir: Path = Path("/workspace/result")
    test_every: int = 8
    patch_size: int | None = None
    global_scale: float = 1.0
    normalize_world_space: bool = True
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    port: int = 4000

    batch_size: int = 1
    steps_scaler: float = 1.0

    max_steps: int = 10_000
    eval_steps: list[int] = field(default_factory=lambda: [3_000, 10_000])
    save_steps: list[int] = field(default_factory=lambda: [10_000])
    save_ply: bool = False
    ply_steps: list[int] = field(default_factory=lambda: [10_000])
    disable_video: bool = True

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

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    tb_every: int = 100
    tb_save_image: bool = True

    lpips_net: Literal["vgg", "alex"] = "alex"

    lambda_low: float = 0.80
    lambda_illumination: float = 0.3

    lambda_illum_curve: float = 2.0
    lambda_illum_exposure: float = 1.0
    lambda_edge_aware_smooth: float = 30.0
    lambda_white_preservation: float = 3.7
    lambda_perceptual_color: float = 2.5
    lambda_freq: float = 0.005

    lambda_illum_variance: float = 1.0
    lambda_reflect: float = 1.0
    lambda_histogram: float = 1

    luminance_threshold: float = 95.0
    chroma_tolerance: float = 2.7
    gain: float = 2.0

    exposure_loss_patch_size: int = 32
    exposure_mean_val: float = 0.5

    learn_adaptive_curve_lambdas: bool = False
    learn_adaptive_curve_use_embedding: bool = False

    allow_chromatic_illumination: bool = True

    loss_adaptive_curve: bool = True
    loss_exposure: bool = True
    loss_smooth_edge_aware: bool = True
    loss_white_preservation: bool = True
    loss_perceptual_color: bool = True
    loss_frequency: bool = True

    loss_variance: bool = False
    loss_histogram: bool = False
    loss_reflectance_spa: bool = False

    uncertainty_weighting: bool = False
    learnt_weighting: bool = False

    save_images: bool = False

    postfix: str = "_multiexposure"

    retinex_opt_lr: float = 7e-3
    retinex_embedding_lr: float = 5e-4
    loss_opt_lr: float = 1e-4

    retinex_embedding_dim: int = 128

    freeze_step: int = 8000

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