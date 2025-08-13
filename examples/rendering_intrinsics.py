import math
from typing import Any, Dict

import torch
import torch.distributed
from torch import Tensor
from typing_extensions import Literal

from gsplat.cuda._wrapper import (
    RollingShutterType,
    fully_fused_projection,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    spherical_harmonics,
)
def rasterize_intrinsics(
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        base_reflectance_sh: Tensor,  # [..., (C,) N, K, 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        sh_degree: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        packed: bool = True,
        tile_size: int = 16,
        backgrounds: Tensor | None = None,
        absgrad: bool = False,
        rasterize_mode: Literal["classic", "antialiased"] = "classic",
) -> tuple[Dict[str, Tensor], Tensor, dict[str, Any]]:
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
        base_reflectance_sh (Tensor): The Spherical Harmonics coefficients for base reflectance.
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

    proj_results = fully_fused_projection(
        means=means,
        covars=None,
        quats=quats,
        scales=scales,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        eps2d=eps2d,
        packed=packed,
        near_plane=near_plane,
        far_plane=far_plane,
        radius_clip=radius_clip,
        sparse_grad=False,
        calc_compensations=(rasterize_mode == "antialiased"),
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

    meta.update({ "means2d": means2d, "radii": radii, "depths": depths })


    campos = torch.inverse(viewmats)[..., :3, 3]
    if packed:
        dirs = means.view(B, N, 3)[batch_ids, gaussian_ids] - campos.view(B, C, 3)[batch_ids, camera_ids]
        masks = (radii > 0).all(dim=-1)
        shs = base_reflectance_sh.view(B, N, -1, 3)[batch_ids, gaussian_ids]
    else:
        dirs = means[..., None, :, :] - campos[..., None, :]
        masks = (radii > 0).all(dim=-1)
        shs = torch.broadcast_to(base_reflectance_sh[..., None, :, :, :], batch_dims + (C, N, -1, 3))

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