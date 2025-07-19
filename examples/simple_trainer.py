import json
import math
import os
import time
from collections import defaultdict
from typing import Any

import imageio
import kornia
import numpy as np
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
from typing_extensions import assert_never

from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_ellipse_path_z,
    generate_interpolated_path,
    generate_spiral_path,
)
from config import Config
from examples.utils import PhysicsAwareIllumination
from rendering_pbr import rasterization_pbr
from losses import ExclusionLoss
from utils import IlluminationField, DecomposedIlluminationField
from gsplat import export_splats
from gsplat.distributed import cli
from gsplat.optimizers import SelectiveAdam
from gsplat.strategy import MCMCStrategy
from default import DefaultStrategy
from utils import (
    knn,
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

        albedo_lr: float = 2.5e-3,
        rough_metal_lr: float = 2.5e-3,

        scene_scale: float = 1.0,
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

    initial_albedo = torch.logit(rgbs.clamp(0.01, 0.99))
    initial_roughness = torch.logit(torch.full((N, 1), 0.7))
    initial_metallic = torch.logit(torch.full((N, 1), 0.1))

    params = [
        ("means", torch.nn.Parameter(points), means_lr * scene_scale),
        ("scales", torch.nn.Parameter(scales), scales_lr),
        ("quats", torch.nn.Parameter(quats), quats_lr),
        ("opacities", torch.nn.Parameter(opacities), opacities_lr),
        ("albedo", torch.nn.Parameter(initial_albedo), albedo_lr),
        ("roughness", torch.nn.Parameter(initial_roughness), rough_metal_lr),
        ("metallic", torch.nn.Parameter(initial_metallic), rough_metal_lr),
    ]

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
            # factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            # is_mip360=True,
        )
        self.trainset = Dataset(
            self.parser, patch_size=cfg.patch_size
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        self.illumination_field = PhysicsAwareIllumination().to(self.device)


        if world_size > 1:
            self.illumination_field = DDP(self.illumination_field, device_ids=[local_rank])

        self.illum_field_optimizer = torch.optim.AdamW(self.illumination_field.parameters())

        self.loss_exclusion = ExclusionLoss().to(self.device)

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
            albedo_lr=cfg.albedo_lr,
            rough_metal_lr=cfg.rough_metal_lr,
            scene_scale=self.scene_scale,
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
    ):
        means = self.splats["means"]
        quats = self.splats["quats"]
        scales = torch.exp(self.splats["scales"])
        opacities = torch.sigmoid(self.splats["opacities"])
        albedo = self.splats["albedo"]
        roughness = self.splats["roughness"]
        metallic = self.splats["metallic"]

        light_color, light_dir = self.illumination_field(means)

        return rasterization_pbr(
            means=means, quats=quats, scales=scales, opacities=opacities,
            albedo=albedo, roughness=roughness, metallic=metallic,
            viewmats=torch.linalg.inv(camtoworlds.float()),
            Ks=Ks,
            width=width,
            height=height,
            light_color=light_color,
            packed=self.cfg.packed,
            light_dir=light_dir,
            backgrounds=kwargs.get("backgrounds")
        )


    @torch.no_grad()
    def visualize_illumination_field(
            self,
            depth_map: Tensor,
            camtoworld: Tensor,
            K: Tensor
    ) -> Tensor:
        H, W = depth_map.shape[1:3]

        grid = kornia.utils.create_meshgrid(H, W, normalized_coordinates=False).to(self.device)
        grid = grid.permute(2, 0, 1).unsqueeze(0)  # [1, 2, H, W]

        points_3d_cam = kornia.geometry.depth.unproject_points(
            grid.permute(0, 2, 3, 1), depth_map.permute(0, 2, 3, 1), K
        )

        B, _, _, C = points_3d_cam.shape
        points_3d_cam = points_3d_cam.reshape(B, -1, C)

        points_3d_world = kornia.geometry.transform_points(camtoworld, points_3d_cam)

        gain, gamma = self.illumination_field(points_3d_world.squeeze(0))

        illum_map_vis = gain.reshape(H, W, 3)

        return illum_map_vis


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
            ExponentialLR(self.illum_field_optimizer, gamma=0.01 ** (1.0 / max_steps)),
        ]

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

            with torch.autocast(enabled=False, device_type=device):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                pixels = data["image"].to(device) / 255.0
                height, width = pixels.shape[1:3]

                (
                    renders,
                    alphas,
                    info,
                ) = self.rasterize_splats(
                    camtoworlds=camtoworlds,
                    Ks=Ks,
                    width=width,
                    height=height,
                    render_mode="RGB+ED"
                )

                colors = torch.clamp(renders[..., :3], 0.0, 1.0)
                pixels = torch.clamp(pixels, 0.0, 1.0)

                loss_reconstruct = F.l1_loss(colors, pixels)
                ssim_loss = 1.0 - self.ssim(colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2))
                loss = (1.0 - cfg.ssim_lambda) * loss_reconstruct + cfg.ssim_lambda * ssim_loss

                info["means2d"].retain_grad()

                # smoothness loss for illumination field
                rand_points = (torch.rand(4096, 3, device=device) * 2 - 1) * self.scene_scale
                rand_points.requires_grad = True

                rand_points = (torch.rand(4096, 3, device=device) * 2 - 1) * self.scene_scale
                rand_points.requires_grad = True

                shadow_field_module = self.illumination_field.shadow_field

                shadow_values = shadow_field_module(rand_points)

                d_shadow = torch.autograd.grad(outputs=shadow_values.sum(), inputs=rand_points, create_graph=True)[0]

                loss_illum_smoothness = d_shadow.norm(2, dim=-1).mean()
                loss += cfg.lambda_illum_smoothness * loss_illum_smoothness


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

            desc_parts = [f"loss={loss.item():.3f}"]
            pbar.set_description("| ".join(desc_parts))

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.tb_save_image:
                    with torch.no_grad():
                        self.writer.add_images(
                            "train/pixels",
                            pixels.permute(0, 3, 1, 2),
                            step,
                        )
                        self.writer.add_images(
                            "train/render_enh",
                            colors.permute(0, 3, 1, 2),
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
                data_save = {"step": step, "splats": self.splats.state_dict(),
                             "illumination_field": self.illumination_field.state_dict(),
                             "stats": stats_save}

                torch.save(
                    data_save, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )
            if (
                    step in [i - 1 for i in cfg.ply_steps] or step == max_steps - 1
            ) and cfg.save_ply:
                # sh0_export = self.splats["sh0"]
                # shN_export = self.splats["shN"]
                albedo_export = self.splats["albedo"]
                metallic_export = self.splats["metallic"]
                roughness_export = self.splats["roughness"]


                # todo: export with pbr
                # export_splats(
                #     means=self.splats["means"],
                #     scales=self.splats["scales"],
                #     quats=self.splats["quats"],
                #     opacities=self.splats["opacities"],
                #     albedo=albedo_export,
                #     metallic=metallic_export,
                #     roughness=roughness_export,
                #     save_to=f"{self.ply_dir}/point_cloud_{step}.ply",
                # )


            for optimizer in self.optimizers.values():
                optimizer.step()
                optimizer.zero_grad()

            self.illum_field_optimizer.step()
            self.illum_field_optimizer.zero_grad()

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

            renders, _, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                masks=masks,
                render_mode="RGB+ED",
            )

            torch.cuda.synchronize()
            ellipse_time_total += max(time.time() - tic, 1e-10)

            colors = torch.clamp(renders[..., :3], 0.0, 1.0)


            if world_rank == 0:
                canvas_list = [pixels, colors]

                canvas_eval_low = (
                    torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                )
                canvas_eval_low = (canvas_eval_low * 255).astype(np.uint8)

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_low_{i:04d}.png",
                    canvas_eval_low,
                )

                colors_low_np = colors.squeeze(0).cpu().numpy()

                imageio.imwrite(
                    f"{self.render_dir}/{stage}_low_{i:04d}.png",
                    (colors_low_np * 255).astype(np.uint8),
                )

                pixels_p = pixels.permute(0, 3, 1, 2)
                colors_p = colors.permute(0, 3, 1, 2)

                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))

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

        step = ckpts[0]["step"]
        runner.eval(step=step)
        runner.render_traj(step=step)
    else:
        runner.train()


BilateralGrid = None
color_correct = None
slice_func = None
total_variation_loss = None

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
    # config = tyro.extras.overridable_config_cli(configs)
    config = tyro.cli(
        Config,
    )

    config.adjust_steps(config.steps_scaler)
    torch.set_float32_matmul_precision("high")

    cli(main, config, verbose=True)