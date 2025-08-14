import math
from typing import Any

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal

from gsplat import fully_fused_projection_with_ut
from gsplat.cuda._wrapper import (
    RollingShutterType,
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)
from gsplat.distributed import all_gather_tensor_list, all_gather_int32


def rasterize_intrinsics(
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
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
) -> tuple[dict[str, Tensor], Tensor, dict[str, Any]]:
    """
    Performs a single-pass rasterization to render intrinsic properties of the scene.

    Instead of rendering final colors, this function renders a multi-channel image
    containing:
    1.  Base Reflectance (3 channels): The intrinsic color of the Gaussians.
    2.  World Position (3 channels): The blended world-space position for each pixel.
    3.  Depth (1 channel): The camera-space depth for each pixel.

    These intrinsic maps can then be used in a subsequent step to apply illumination models.

    Args:
        means (Tensor): The 3D means of the Gaussians.
        colors (Tensor): The Spherical Harmonics coefficients for base reflectance.
        ... (other standard Gaussian Splatting parameters)

    Returns:
        tuple: A tuple containing:
            - intrinsic_maps (Dict[str, Tensor]): A dictionary with keys
              'reflectance', 'world_position', and 'depth'.
            - render_alpha (Tensor): The accumulated alpha map.
            - meta (dict): A dictionary with metadata from the rasterization process.
    """
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
    assert render_mode in ["RGB", "D", "ED", "RGB+D", "RGB+ED", "RGB+I"], render_mode

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
                assert colors_tensor.dim() == num_batch_dims + 2, (
                    f"Distributed mode only supports per-Gaussian {name}."
                )
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
            assert (sh_degree + 1) ** 2 <= colors_tensor.shape[-2], (
                f"{name} shape: {colors_tensor.shape}"
            )
            if distributed:
                assert colors_tensor.dim() == num_batch_dims + 3, (
                    f"Distributed mode only supports per-Gaussian {name}."
                )

    _check_color_shape(colors, "colors")

    if absgrad:
        assert not distributed, "AbsGrad is not supported in distributed mode."
    if (
            radial_coeffs is not None
            or tangential_coeffs is not None
            or thin_prism_coeffs is not None
            or rolling_shutter != RollingShutterType.GLOBAL
    ):
        assert with_ut, "Distortion and rolling shutter require `with_ut=True`."
    if rolling_shutter != RollingShutterType.GLOBAL:
        assert viewmats_rs is not None, "Rolling shutter requires `viewmats_rs`."
    else:
        assert viewmats_rs is None, "`viewmats_rs` should be None for global shutter."
    if with_ut or with_eval3d:
        assert (quats is not None) and (scales is not None), (
            "UT and eval3d require quats and scales."
        )
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
            means,
            quats,
            scales,
            opacities,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            viewmats_rs=viewmats_rs,
        )
    else:
        proj_results = fully_fused_projection(
            means,
            covars,
            quats,
            scales,
            viewmats,
            Ks,
            width,
            height,
            eps2d=eps2d,
            packed=packed,
            near_plane=near_plane,
            far_plane=far_plane,
            radius_clip=radius_clip,
            sparse_grad=sparse_grad,
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model,
            opacities=opacities,
        )

    if packed:
        (
            batch_ids, camera_ids, gaussian_ids, radii,
            means2d, depths, conics, compensations,
        ) = proj_results
        opacities_proj = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, conics, compensations = proj_results
        opacities_proj = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))
        batch_ids, camera_ids, gaussian_ids, image_ids = None, None, None, None

    if compensations is not None:
        opacities_proj *= compensations

    meta.update(
        {
            "batch_ids": batch_ids,
            "camera_ids": camera_ids,
            "gaussian_ids": gaussian_ids,
            "radii": radii,
            "means2d": means2d,
            "depths": depths,
            "conics": conics,
            "opacities": opacities,
        }
    )

    def _process_color(color_tensor: Tensor):
        if sh_degree is None:
            if packed:
                if color_tensor.dim() == num_batch_dims + 2:
                    return color_tensor.view(B, N, -1)[batch_ids, gaussian_ids]
                else:
                    return color_tensor.view(B, C, N, -1)[
                        batch_ids, camera_ids, gaussian_ids
                    ]
            else:
                if color_tensor.dim() == num_batch_dims + 2:
                    return torch.broadcast_to(
                        color_tensor[..., None, :, :], batch_dims + (C, N, -1)
                    )
                else:
                    return color_tensor
        else:
            campos = torch.inverse(viewmats)[..., :3, 3]
            if viewmats_rs is not None:
                campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
                campos = 0.5 * (campos + campos_rs)

            if packed:
                dirs = (
                        means.view(B, N, 3)[batch_ids, gaussian_ids]
                        - campos.view(B, C, 3)[batch_ids, camera_ids]
                )
                masks = (radii > 0).all(dim=-1)
                if color_tensor.dim() == num_batch_dims + 3:
                    shs = color_tensor.view(B, N, -1, 3)[batch_ids, gaussian_ids]
                else:
                    shs = color_tensor.view(B, C, N, -1, 3)[
                        batch_ids, camera_ids, gaussian_ids
                    ]
            else:
                dirs = means[..., None, :, :] - campos[..., None, :]
                masks = (radii > 0).all(dim=-1)
                if color_tensor.dim() == num_batch_dims + 3:
                    shs = torch.broadcast_to(
                        color_tensor[..., None, :, :, :], batch_dims + (C, N, -1, 3)
                    )
                else:
                    shs = color_tensor

            processed_colors = spherical_harmonics(sh_degree, dirs, shs, masks=masks)
            return torch.clamp_min(processed_colors + 0.5, 0.0)

    colors = _process_color(colors)

    campos = torch.inverse(viewmats)[..., :3, 3]
    if packed:
        dirs = means.view(B, N, 3)[batch_ids, gaussian_ids] - campos.view(B, C, 3)[batch_ids, camera_ids]
        masks = (radii > 0).all(dim=-1)
        shs = colors.view(B, N, -1, 3)[batch_ids, gaussian_ids]
    else:
        dirs = means[..., None, :, :] - campos[..., None, :]
        masks = (radii > 0).all(dim=-1)
        shs = torch.broadcast_to(colors[..., None, :, :, :], batch_dims + (C, N, -1, 3))

    base_reflectance_rgb = spherical_harmonics(sh_degree, dirs, shs, masks=masks)
    base_reflectance_rgb = torch.clamp_min(base_reflectance_rgb + 0.5, 0.0)

    if packed:
        world_positions = means.view(B, N, 3)[batch_ids, gaussian_ids]
    else:
        world_positions = torch.broadcast_to(means[..., None, :, :], batch_dims + (C, N, 3))


    properties_to_render = torch.cat(
        [base_reflectance_rgb, world_positions, depths.unsqueeze(-1)],
        dim=-1
    )

    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d, radii, depths, tile_size, tile_width, tile_height,
        packed=packed, n_images=I, image_ids=image_ids
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    rendered_properties, render_alpha = rasterize_to_pixels(
        means2d,
        conics,
        properties_to_render,
        opacities_proj,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
    )

    reflectance_map = rendered_properties[..., :3]
    world_position_map = rendered_properties[..., 3:6]

    depth_map = rendered_properties[..., 6:7]
    expected_depth_map = depth_map / render_alpha.clamp(min=1e-10)

    intrinsic_maps = {
        "reflectance": reflectance_map,
        "world_position": world_position_map,
        "depth": expected_depth_map,
    }

    meta.update({ "flatten_ids": flatten_ids })

    return intrinsic_maps, render_alpha, meta