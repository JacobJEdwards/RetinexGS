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
    save_steps: list[int] = field(default_factory=lambda: [3_000, 10_000])
    save_ply: bool = False
    ply_steps: list[int] = field(default_factory=lambda: [3_000, 10_000])
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

    lambda_low: float = 0.2
    lambda_illumination: float = 0.1

    lambda_reflect: float = 1.0
    lambda_illum_curve: float = 1.0
    lambda_illum_exposure: float = 1.0
    lambda_edge_aware_smooth: float = 20.0

    lambda_illum_exposure_local: float = 1.8
    lambda_white_preservation: float = 3.7
    lambda_histogram: float = 1
    lambda_illum_exclusion: float = 0.15
    lambda_perceptual_color: float = 1.0

    luminance_threshold: float = 75.0
    dark_luminance_threshold: float = 10.0
    chroma_tolerance: float = 2.75
    gain: float = 2.0

    lambda_illum_color: float = 1.

    pretrain_retinex: bool = True
    pretrain_steps: int = 3000

    use_lab_color_space: bool = False

    predictive_adaptive_curve: bool = False

    exposure_loss_patch_size: int = 32
    exposure_mean_val: float = 0.48
    exposure_loss_use_embedding: bool = False

    learn_spatial_contrast: bool = True
    learn_adaptive_curve_lambdas: bool = True
    learn_adaptive_curve_thresholds: bool = False
    learn_adaptive_curve_use_embedding: bool = True
    learn_local_exposure: bool = True
    learn_global_exposure: bool = True
    learn_edge_aware_gamma: bool = False
    learn_white_preservation: bool = False
    learn_dark_preservation: bool = False
    use_enhancement_gate: bool = True

    dynamic_weights: bool = False

    loss_adaptive_curve: bool = True
    loss_exposure: bool = True
    loss_reflectance_spa: bool = True
    loss_smooth_edge_aware: bool = True
    loss_exposure_local: bool = False
    loss_exclusion: bool = False
    loss_white_preservation: bool = False
    loss_histogram: bool = False
    loss_perceptual_color: bool = False

    postfix: str = "_multiexposure"

    retinex_opt_lr: float = 1e-3
    retinex_embedding_lr: float = 5e-5

    retinex_embedding_dim: int = 32

    freeze_step: int = 10

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