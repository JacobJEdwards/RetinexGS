import math
from typing import Any

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    RollingShutterType,
    fully_fused_projection,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_eval3d,
    spherical_harmonics,
)
from gsplat.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)


def rasterization_dual(
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
        colors_low: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        sh_degree: int | None = None,
        packed: bool = True,
        tile_size: int = 16,
        backgrounds: Tensor | None = None,
        render_mode: Literal["RGB", "D", "ED", "RGB+D", "RGB+ED"] = "RGB",
        sparse_grad: bool = False,
        absgrad: bool = False,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
        channel_chunk: int = 32,
        distributed: bool = False,
        camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole",
        segmented: bool = False,
        covars: Tensor | None = None,
        with_ut: bool = False,
        with_eval3d: bool = False,
        # distortion
        radial_coeffs: Tensor | None = None,  # [..., C, 6] or [..., C, 4]
        tangential_coeffs: Tensor | None = None,  # [..., C, 2]
        thin_prism_coeffs: Tensor | None = None,  # [..., C, 4]
        # rolling shutter
        rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
        viewmats_rs: Tensor | None = None,  # [..., C, 4, 4]
) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Any]]:
    meta = {}
    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device

    assert means.shape == batch_dims + (N, 3), means.shape
    if covars is None:
        assert quats.shape == batch_dims + (N, 4), quats.shape
        assert scales.shape == batch_dims + (N, 3), scales.shape
    else:
        assert covars.shape == batch_dims + (N, 3, 3), covars.shape
        quats, scales = None, None
        tri_indices = ([0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2])
        covars = covars[..., tri_indices[0], tri_indices[1]]
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED"], render_mode

    def _check_color_shape(colors_tensor, name):
        if sh_degree is None:
            assert (
                           colors_tensor.dim() == num_batch_dims + 2
                           and colors_tensor.shape[:-1] == batch_dims + (N,)
                   ) or (
                           colors_tensor.dim() == num_batch_dims + 3
                           and colors_tensor.shape[:-1] == batch_dims + (C, N)
                   ), f"{name} shape: {colors_tensor.shape}"
            if distributed:
                assert (
                        colors_tensor.dim() == num_batch_dims + 2
                ), f"Distributed mode only supports per-Gaussian {name}."
        else:
            assert (
                           colors_tensor.dim() == num_batch_dims + 3
                           and colors_tensor.shape[:-2] == batch_dims + (N,)
                           and colors_tensor.shape[-1] == 3
                   ) or (
                           colors_tensor.dim() == num_batch_dims + 4
                           and colors_tensor.shape[:-2] == batch_dims + (C, N)
                           and colors_tensor.shape[-1] == 3
                   ), f"{name} shape: {colors_tensor.shape}"
            assert (sh_degree + 1) ** 2 <= colors_tensor.shape[-2], f"{name} shape: {colors_tensor.shape}"
            if distributed:
                assert (
                        colors_tensor.dim() == num_batch_dims + 3
                ), f"Distributed mode only supports per-Gaussian {name}."

    _check_color_shape(colors, "colors")
    _check_color_shape(colors_low, "colors_low")

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."
    if radial_coeffs is not None or tangential_coeffs is not None or thin_prism_coeffs is not None or rolling_shutter != RollingShutterType.GLOBAL:
        assert with_ut, "Distortion and rolling shutter require `with_ut=True`."
    if rolling_shutter != RollingShutterType.GLOBAL:
        assert viewmats_rs is not None, "Rolling shutter requires `viewmats_rs`."
    else:
        assert viewmats_rs is None, "`viewmats_rs` should be None for global shutter."
    if with_ut or with_eval3d:
        assert (quats is not None) and (scales is not None), "UT and eval3d require quats and scales."
        assert not packed, "Packed mode is not supported with UT or eval3d."
        assert not sparse_grad, "Sparse grad is not supported with UT or eval3d."

    def reshape_view(C: int, world_view: torch.Tensor, N_world: list) -> torch.Tensor:
        view_list = list(
            map(
                lambda x: x.split(int(x.shape[0] / C), dim=0),
                world_view.split([C * N_i for N_i in N_world], dim=0),
            )
        )
        return torch.stack([torch.cat(l, dim=0) for l in zip(*view_list)], dim=0)

    if distributed:
        assert batch_dims == (), "Distributed mode does not support batch dimensions"
        world_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        N_world = all_gather_int32(world_size, N, device=device)
        C_world = [C] * world_size
        viewmats, Ks = all_gather_tensor_list(world_size, [viewmats, Ks])
        if viewmats_rs is not None:
            (viewmats_rs,) = all_gather_tensor_list(world_size, [viewmats_rs])
        C = len(viewmats)

    if with_ut:
        proj_results = fully_fused_projection_with_ut(
            means, quats, scales, opacities, viewmats, Ks, width, height,
            eps2d=eps2d, near_plane=near_plane, far_plane=far_plane, radius_clip=radius_clip,
            calc_compensations=(rasterize_mode == "antialiased"), camera_model=camera_model,
            radial_coeffs=radial_coeffs, tangential_coeffs=tangential_coeffs,
            thin_prism_coeffs=thin_prism_coeffs, rolling_shutter=rolling_shutter, viewmats_rs=viewmats_rs,
        )
    else:
        proj_results = fully_fused_projection(
            means, covars, quats, scales, viewmats, Ks, width, height,
            eps2d=eps2d, packed=packed, near_plane=near_plane, far_plane=far_plane,
            radius_clip=radius_clip, sparse_grad=sparse_grad,
            calc_compensations=(rasterize_mode == "antialiased"), camera_model=camera_model,
            opacities=opacities,
        )

    if packed:
        (batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, conics, compensations) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, conics, compensations = proj_results
        opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))
        batch_ids, camera_ids, gaussian_ids, image_ids = None, None, None, None

    if compensations is not None:
        opacities *= compensations

    meta.update({
        "batch_ids": batch_ids, "camera_ids": camera_ids, "gaussian_ids": gaussian_ids,
        "radii": radii, "means2d": means2d, "depths": depths, "conics": conics, "opacities": opacities,
    })

    def _process_color(color_tensor: Tensor):
        if sh_degree is None:
            if packed:
                if color_tensor.dim() == num_batch_dims + 2:
                    return color_tensor.view(B, N, -1)[batch_ids, gaussian_ids]
                else:
                    return color_tensor.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
            else:
                if color_tensor.dim() == num_batch_dims + 2:
                    return torch.broadcast_to(color_tensor[..., None, :, :], batch_dims + (C, N, -1))
                else:
                    return color_tensor
        else: 
            campos = torch.inverse(viewmats)[..., :3, 3]
            if viewmats_rs is not None:
                campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
                campos = 0.5 * (campos + campos_rs)

            if packed:
                dirs = means.view(B, N, 3)[batch_ids, gaussian_ids] - campos.view(B, C, 3)[batch_ids, camera_ids]
                masks = (radii > 0).all(dim=-1)
                if color_tensor.dim() == num_batch_dims + 3:
                    shs = color_tensor.view(B, N, -1, 3)[batch_ids, gaussian_ids]
                else:
                    shs = color_tensor.view(B, C, N, -1, 3)[batch_ids, camera_ids, gaussian_ids]
            else: 
                dirs = means[..., None, :, :] - campos[..., None, :]
                masks = (radii > 0).all(dim=-1)
                if color_tensor.dim() == num_batch_dims + 3:
                    shs = torch.broadcast_to(color_tensor[..., None, :, :, :], batch_dims + (C, N, -1, 3))
                else:
                    shs = color_tensor

            processed_colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)
            return torch.clamp_min(processed_colors + 0.5, 0.0)

    colors = _process_color(colors)
    colors_low = _process_color(colors_low)

    if distributed:
        if packed:
            cnts = torch.bincount(camera_ids, minlength=C).split(C_world, dim=0)
            cnts = [c.sum() for c in cnts]
            collected_splits = all_to_all_int32(world_size, cnts, device=device)

            (radii,) = all_to_all_tensor_list(world_size, [radii], cnts, output_splits=collected_splits)
            (means2d, depths, conics, opacities, colors, colors_low) = all_to_all_tensor_list(
                world_size, [means2d, depths, conics, opacities, colors, colors_low],
                cnts, output_splits=collected_splits
            )
            offsets = torch.cumsum(torch.tensor([0] + C_world[:-1], device=device, dtype=torch.long), dim=0)
            camera_ids = camera_ids - offsets.repeat_interleave(torch.stack(cnts))
            offsets = torch.cumsum(torch.tensor([0] + N_world[:-1], device=device, dtype=torch.long), dim=0)
            gaussian_ids = gaussian_ids + offsets.repeat_interleave(torch.stack(cnts))
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size, [camera_ids, gaussian_ids], cnts, output_splits=collected_splits
            )
            C = C_world[world_rank]
        else: 
            C = C_world[world_rank]
            (radii,) = all_to_all_tensor_list(world_size, [radii.flatten(0, 1)], [C_i * N for C_i in C_world], [C * N_i for N_i in N_world])
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors, colors_low) = all_to_all_tensor_list(
                world_size, [t.flatten(0, 1) for t in [means2d, depths, conics, opacities, colors, colors_low]],
                [C_i * N for C_i in C_world], [C * N_i for N_i in N_world]
            )
            means2d, depths, conics, opacities = reshape_view(C, means2d, N_world), reshape_view(C, depths, N_world), reshape_view(C, conics, N_world), reshape_view(C, opacities, N_world)
            colors, colors_low = reshape_view(C, colors, N_world), reshape_view(C, colors_low, N_world)

    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        colors_low = torch.cat((colors_low, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat([backgrounds, torch.zeros(batch_dims + (C, 1), device=device)], dim=-1)
    elif render_mode in ["D", "ED"]:
        colors = colors_low = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(batch_dims + (C, 1), device=device)

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height,
        segmented=segmented, packed=packed, n_images=I, image_ids=image_ids, gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    meta.update({
        "tile_width": tile_width, "tile_height": tile_height, "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids, "flatten_ids": flatten_ids, "isect_offsets": isect_offsets,
        "width": width, "height": height, "tile_size": tile_size, "n_batches": B, "n_cameras": C,
    })

    # Perform dual rasterization
    if colors.shape[-1] > channel_chunk:
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors_list, render_low_colors_list, render_alphas_list = [], [], []

        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            colors_low_chunk = colors_low[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk] if backgrounds is not None else None

            if with_eval3d:
                raise NotImplementedError("Chunked dual rendering for `with_eval3d` is not implemented.")
            else:
                rc, ra = rasterize_to_pixels(
                    means2d, conics, colors_chunk, opacities, width, height, tile_size, isect_offsets,
                    flatten_ids, backgrounds=backgrounds_chunk, packed=packed, absgrad=absgrad
                )
                rlc, _ = rasterize_to_pixels(
                    means2d, conics, colors_low_chunk, opacities, width, height, tile_size, isect_offsets,
                    flatten_ids, backgrounds=backgrounds_chunk, packed=packed, absgrad=absgrad
                )
            render_colors_list.append(rc)
            render_low_colors_list.append(rlc)
            render_alphas_list.append(ra)

        render_colors = torch.cat(render_colors_list, dim=-1)
        render_low_colors = torch.cat(render_low_colors_list, dim=-1)
        render_alphas = render_alphas_list[0]
    else:
        if with_eval3d:
            render_colors, render_alphas = rasterize_to_pixels_eval3d(
                means, quats, scales, colors, opacities, viewmats, Ks, width, height, tile_size,
                isect_offsets, flatten_ids, backgrounds=backgrounds, camera_model=camera_model,
                radial_coeffs=radial_coeffs, tangential_coeffs=tangential_coeffs, thin_prism_coeffs=thin_prism_coeffs,
                rolling_shutter=rolling_shutter, viewmats_rs=viewmats_rs,
            )
            render_low_colors, _ = rasterize_to_pixels_eval3d(
                means, quats, scales, colors_low, opacities, viewmats, Ks, width, height, tile_size,
                isect_offsets, flatten_ids, backgrounds=backgrounds, camera_model=camera_model,
                radial_coeffs=radial_coeffs, tangential_coeffs=tangential_coeffs, thin_prism_coeffs=thin_prism_coeffs,
                rolling_shutter=rolling_shutter, viewmats_rs=viewmats_rs,
            )
        else:
            render_colors, render_alphas = rasterize_to_pixels(
                means2d, conics, colors, opacities, width, height, tile_size,
                isect_offsets, flatten_ids, backgrounds=backgrounds, packed=packed, absgrad=absgrad
            )
            render_low_colors, render_low_alphas = rasterize_to_pixels(
                means2d, conics, colors_low, opacities, width, height, tile_size,
                isect_offsets, flatten_ids, backgrounds=backgrounds, packed=packed, absgrad=absgrad
            )

    if render_mode in ["ED", "RGB+ED"]:
        render_colors = torch.cat([
            render_colors[..., :-1],
            render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ], dim=-1)
        render_low_colors = torch.cat([
            render_low_colors[..., :-1],
            render_low_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ], dim=-1)

    return render_colors, render_low_colors, render_alphas, render_low_alphas, meta