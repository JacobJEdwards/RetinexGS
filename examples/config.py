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

    data_dir: Path = Path("../../360_v2/room")
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
    eval_steps: list[int] = field(default_factory=lambda: [10_000])
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

    save_ckpt: bool = False
    save_images: bool = False

    strategy: DefaultStrategy | MCMCStrategy = field(default_factory=DefaultStrategy)

    means_lr: float = 1.6e-4
    scales_lr: float = 5e-3
    opacities_lr: float = 5e-2
    quats_lr: float = 1e-3
    sh0_lr: float = 2.5e-3
    shN_lr: float = 2.5e-3 / 20

    opacity_reg: float = 0.0
    scale_reg: float = 0.0

    tb_every: int = 1000
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "alex"

    postfix: str = "_contrast"

    appearance_embedding_dim: int = 64

    use_view_dirs: bool = True
    use_normals: bool = True
    use_camera_response_network: bool = True

    lambda_illum_smoothness: float = 0.001
    lambda_exclusion: float = 0.1
    lambda_shn_reg: float = 0.8
    lambda_tv_loss: float = 500.0

    appearance_embedding_lr: float = 6e-3
    camera_net_lr: float = 3e-4
    illumination_field_lr: float = 6e-5
    loss_lr: float = 1e-4

    uncertainty_weighting: bool = True

    learning_steps: int = 2800

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
