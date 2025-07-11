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

    data_dir: Path = Path("../../NeRF_360/bicycle")
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

    max_steps: int = 10_000
    eval_steps: list[int] = field(default_factory=lambda: [3_000, 7_000, 10_000])
    save_steps: list[int] = field(default_factory=lambda: [3_000, 7_000, 10_000])
    save_ply: bool = True
    ply_steps: list[int] = field(default_factory=lambda: [3_000, 7_000, 10_000])
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
    bilateral_grid_shape: tuple[int, int, int] = (16, 16, 8)

    depth_loss: bool = False
    depth_lambda: float = 1e-2

    tb_every: int = 100
    tb_save_image: bool = True

    lpips_net: Literal["vgg", "alex"] = "alex"

    with_ut: bool = False
    with_eval3d: bool = False

    use_fused_bilagrid: bool = False

    enable_retinex: bool = True
    multi_scale_retinex: bool = True

    lambda_low: float = 0.25
    lambda_illumination: float = 0.3

    lambda_reflect: float = 6.0
    lambda_smooth: float = 600.0
    lambda_illum_color: float = 0.5
    lambda_illum_exposure: float = 0
    lambda_illum_variance: float = 0.05
    lambda_illum_contrast: float = 0.1
    lambda_illum_curve: float = 1.5
    lambda_illum_exposure_local: float = 0

    lambda_laplacian: float = 0.2
    lambda_gradient: float = 0.01
    lambda_frequency: float = 0
    lambda_edge_aware_smooth: float = 15
    lambda_illum_frequency: float = 0.1
    lambda_exclusion: float = 5.0

    pretrain_retinex: bool = True
    pretrain_steps: int = 5000

    use_hsv_color_space: bool = True
    use_refinement_net: bool = False
    use_denoising_net: bool = False
    use_denoising_embedding: bool = False
    
    predictive_adaptive_curve: bool = True
    spatial_film: bool = False
    use_dilated_convs: bool = True
    use_se_blocks: bool = True
    use_spatial_attention: bool = False
    enable_dynamic_weights: bool = True
    use_pixel_shuffle: bool = True
    use_stride_conv: bool = True
    
    learn_spatial_contrast: bool = True
    learn_adaptive_curve_lambdas: bool = True
    learn_local_exposure: bool = False
    learn_global_exposure: bool = True
    
    use_illum_opt: bool = True
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
