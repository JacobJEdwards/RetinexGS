import math
from typing import Any, Literal

import torch
import torch.distributed
import torch.nn.functional as F
from torch import Tensor

from gsplat.cuda._wrapper import (
    RollingShutterType,
    fully_fused_projection,
    fully_fused_projection_with_ut,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels,
    rasterize_to_pixels_eval3d,
)
from gsplat.distributed import (
    all_gather_int32,
    all_gather_tensor_list,
    all_to_all_int32,
    all_to_all_tensor_list,
)

EPS = 1e-8

def D_GGX(N_dot_H: Tensor, roughness: Tensor) -> Tensor:
    """ Trowbridge-Reitz GGX Normal Distribution Function """
    a = roughness * roughness
    a2 = a * a
    denom = N_dot_H * N_dot_H * (a2 - 1.0) + 1.0
    return a2 / (math.pi * denom * denom + EPS)

def F_Schlick(cos_theta: Tensor, F0: Tensor) -> Tensor:
    """ Schlick's approximation for Fresnel """
    return F0 + (1.0 - F0) * torch.pow(torch.clamp(1.0 - cos_theta, 0.0, 1.0), 5.0)

def G_Smith(N_dot_V: Tensor, N_dot_L: Tensor, roughness: Tensor) -> Tensor:
    """ Smith's method for Geometry shadowing-masking """
    r = roughness + 1.0
    k = (r * r) / 8.0  # k_direct for GGX

    ggx_v = N_dot_V / (N_dot_V * (1.0 - k) + k + EPS)
    ggx_l = N_dot_L / (N_dot_L * (1.0 - k) + k + EPS)
    return ggx_v * ggx_l

def pbr_shading(
        V: Tensor,          # [..., 3]
        L: Tensor,          # [..., 3]
        N: Tensor,          # [..., 3]
        albedo: Tensor,     # [..., 3]
        roughness: Tensor,  # [..., 1]
        metallic: Tensor,   # [..., 1]
        light_color: Tensor # [..., 3]
) -> Tensor:
    """
    Calculates the final color of a surface point using a PBR model.
    """
    N_dot_V = torch.clamp(torch.sum(N * V, dim=-1, keepdim=True), min=EPS)
    N_dot_L = torch.clamp(torch.sum(N * L, dim=-1, keepdim=True), min=EPS)

    H = torch.nn.functional.normalize(V + L, dim=-1)
    N_dot_H = torch.clamp(torch.sum(N * H, dim=-1, keepdim=True), min=EPS)
    V_dot_H = torch.clamp(torch.sum(V * H, dim=-1, keepdim=True), min=EPS)

    # Specular Term
    NDF = D_GGX(N_dot_H, roughness)
    G = G_Smith(N_dot_V, N_dot_L, roughness)
    F0 = torch.full_like(albedo, 0.04)
    F0 = torch.lerp(F0, albedo, metallic)
    F = F_Schlick(V_dot_H, F0)

    numerator = NDF * G * F
    denominator = 4.0 * N_dot_V * N_dot_L + EPS
    specular = numerator / denominator

    # Diffuse Term (Lambertian)
    kS = F
    kD = 1.0 - kS
    kD *= 1.0 - metallic
    diffuse = (kD * albedo / math.pi)

    # Final combined color
    radiance = (diffuse + specular) * light_color * N_dot_L
    return radiance

def quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """
    Helper function to apply a quaternion rotation to a vector.
    Args:
        q: (..., 4) tensor of quaternions in (w, x, y, z) format.
        v: (..., 3) tensor of 3D vectors.
    Returns:
        (..., 3) tensor of rotated vectors.
    """

    if v.dim() == 1:
        v = v.view((1,) * (q.dim() - 1) + (3,))

    # Ensure quaternion is normalized
    q = F.normalize(q, p=2, dim=-1)
    q_w = q[..., :1]
    q_vec = q[..., 1:]

    # Using the efficient formula: v' = v + 2 * cross(q_vec, cross(q_vec, v) + q_w * v)
    t = 2 * torch.cross(q_vec, v, dim=-1)
    v_rotated = v + q_w * t + torch.cross(q_vec, t, dim=-1)
    return v_rotated

def rasterization_pbr(
        means: Tensor,  # [..., N, 3]
        quats: Tensor,  # [..., N, 4]
        scales: Tensor,  # [..., N, 3]
        opacities: Tensor,  # [..., N]
        albedo: Tensor, # [..., N, 3]
        roughness: Tensor, # [..., N, 1]
        metallic: Tensor, # [..., N, 1]
        light_color: Tensor, # [..., 3]
        viewmats: Tensor,  # [..., C, 4, 4]
        Ks: Tensor,  # [..., C, 3, 3]
        width: int,
        height: int,
        near_plane: float = 0.01,
        far_plane: float = 1e10,
        radius_clip: float = 0.0,
        eps2d: float = 0.3,
        packed: bool = False,
        tile_size: int = 16,
        light_dir: Tensor = torch.tensor([0.5, 0.5, -0.5]), # Added for PBR
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
        radial_coeffs: Tensor | None = None,
        tangential_coeffs: Tensor | None = None,
        thin_prism_coeffs: Tensor | None = None,
        # rolling shutter
        rolling_shutter: RollingShutterType = RollingShutterType.GLOBAL,
        viewmats_rs: Tensor | None = None,
) -> tuple[Tensor, Tensor, dict[str, Any]]:
    """
    Fully-featured PBR rasterization function.
    Combines PBR shading with the advanced rasterization pipeline.
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
        assert quats is not None and scales is not None
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

    assert albedo.shape == batch_dims + (N, 3), f"Albedo shape mismatch: {albedo.shape}"
    assert roughness.shape == batch_dims + (N, 1), f"Roughness shape mismatch: {roughness.shape}"
    assert metallic.shape == batch_dims + (N, 1), f"Metallic shape mismatch: {metallic.shape}"

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

    # Projection
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
            calc_compensations=(rasterize_mode == "antialiased"),
            camera_model=camera_model, opacities=opacities,
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


    # PBR Shading Calculation
    campos = torch.inverse(viewmats)[..., :3, 3]
    if viewmats_rs is not None:
        campos_rs = torch.inverse(viewmats_rs)[..., :3, 3]
        campos = 0.5 * (campos + campos_rs)

    light_dir = F.normalize(light_dir.to(device).float(), dim=0)

    if packed:
        means_packed = means.view(B, N, 3)[batch_ids, gaussian_ids]
        quats_packed = quats.view(B, N, 4)[batch_ids, gaussian_ids]
        albedo_packed = albedo.view(B, N, 3)[batch_ids, gaussian_ids]
        roughness_packed = roughness.view(B, N, 1)[batch_ids, gaussian_ids]
        metallic_packed = metallic.view(B, N, 1)[batch_ids, gaussian_ids]
        light_color_packed = light_color.view(B, N, 3)[batch_ids, gaussian_ids]

        view_dirs = F.normalize(means_packed - campos.view(B, C, 3)[batch_ids, camera_ids])

        base_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        normals = quat_apply(quats_packed, base_normal)

        colors = pbr_shading(
            V=view_dirs, L=light_dir.expand_as(means_packed), N=normals,
            albedo=torch.sigmoid(albedo_packed), roughness=torch.sigmoid(roughness_packed),
            metallic=torch.sigmoid(metallic_packed), light_color=torch.sigmoid(light_color_packed)
        )
    else:
        view_dirs = F.normalize(means[..., None, :, :] - campos[..., None, :])

        base_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=torch.float32)
        quats_exp = torch.broadcast_to(quats[..., None, :, :], batch_dims + (C, N, 4))
        normals = quat_apply(quats_exp, base_normal)

        light_dir_exp = light_dir.expand(*batch_dims, C, N, 3)

        colors = pbr_shading(
            V=view_dirs, L=light_dir_exp, N=normals,
            albedo=torch.sigmoid(torch.broadcast_to(albedo[..., None, :, :], batch_dims + (C, N, 3))),
            roughness=torch.sigmoid(torch.broadcast_to(roughness[..., None, :, :], batch_dims + (C, N, 1))),
            metallic=torch.sigmoid(torch.broadcast_to(metallic[..., None, :, :], batch_dims + (C, N, 1))),
            light_color=torch.sigmoid(torch.broadcast_to(light_color[..., None, :, :], batch_dims + (C, N, 3)))
        )

    if distributed:
        if packed:
            # count how many elements need to be sent to each rank
            cnts = torch.bincount(camera_ids, minlength=C)  # all cameras
            cnts = cnts.split(C_world, dim=0)
            cnts = [cuts.sum() for cuts in cnts]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            collected_splits = all_to_all_int32(world_size, cnts, device=device)
            (radii,) = all_to_all_tensor_list(
                world_size, [radii], cnts, output_splits=collected_splits
            )
            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [means2d, depths, conics, opacities, colors],
                cnts,
                output_splits=collected_splits,
            )

            # before sending the data, we should turn the camera_ids from global to local.
            # i.e. the camera_ids produced by the projection stage are over all cameras world-wide,
            # so we need to turn them into camera_ids that are local to each rank.
            offsets = torch.tensor(
                [0] + C_world[:-1], device=camera_ids.device, dtype=camera_ids.dtype
            )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            camera_ids = camera_ids - offsets

            # and turn gaussian ids from local to global.
            offsets = torch.tensor(
                [0] + N_world[:-1],
                device=gaussian_ids.device,
                dtype=gaussian_ids.dtype,
                )
            offsets = torch.cumsum(offsets, dim=0)
            offsets = offsets.repeat_interleave(torch.stack(cnts))
            gaussian_ids = gaussian_ids + offsets

            # all to all communication across all ranks.
            (camera_ids, gaussian_ids) = all_to_all_tensor_list(
                world_size,
                [camera_ids, gaussian_ids],
                cnts,
                output_splits=collected_splits,
            )

            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

        else:
            # Silently change C from global #Cameras to local #Cameras.
            C = C_world[world_rank]

            # all to all communication across all ranks. After this step, each rank
            # would have all the necessary GSs to render its own images.
            (radii,) = all_to_all_tensor_list(
                world_size,
                [radii.flatten(0, 1)],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            radii = reshape_view(C, radii, N_world)

            (means2d, depths, conics, opacities, colors) = all_to_all_tensor_list(
                world_size,
                [
                    means2d.flatten(0, 1),
                    depths.flatten(0, 1),
                    conics.flatten(0, 1),
                    opacities.flatten(0, 1),
                    colors.flatten(0, 1),
                ],
                splits=[C_i * N for C_i in C_world],
                output_splits=[C * N_i for N_i in N_world],
            )
            means2d = reshape_view(C, means2d, N_world)
            depths = reshape_view(C, depths, N_world)
            conics = reshape_view(C, conics, N_world)
            opacities = reshape_view(C, opacities, N_world)
            colors = reshape_view(C, colors, N_world)

    # Rasterize to pixels
    if render_mode in ["RGB+D", "RGB+ED"]:
        colors = torch.cat((colors, depths[..., None]), dim=-1)
        if backgrounds is not None:
            backgrounds = torch.cat(
                [
                    backgrounds,
                    torch.zeros(batch_dims + (C, 1), device=backgrounds.device),
                ],
                dim=-1,
            )
    elif render_mode in ["D", "ED"]:
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)
    else:  # RGB
        pass

    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        segmented=segmented,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    # print("rank", world_rank, "Before isect_offset_encode")
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    meta.update(
        {
            "tile_width": tile_width,
            "tile_height": tile_height,
            "tiles_per_gauss": tiles_per_gauss,
            "isect_ids": isect_ids,
            "flatten_ids": flatten_ids,
            "isect_offsets": isect_offsets,
            "width": width,
            "height": height,
            "tile_size": tile_size,
            "n_batches": B,
            "n_cameras": C,
        }
    )

    # print("rank", world_rank, "Before rasterize_to_pixels")
    if colors.shape[-1] > channel_chunk:
        # slice into chunks
        n_chunks = (colors.shape[-1] + channel_chunk - 1) // channel_chunk
        render_colors, render_alphas = [], []
        for i in range(n_chunks):
            colors_chunk = colors[..., i * channel_chunk : (i + 1) * channel_chunk]
            backgrounds_chunk = (
                backgrounds[..., i * channel_chunk : (i + 1) * channel_chunk]
                if backgrounds is not None
                else None
            )
            if with_eval3d:
                render_colors_, render_alphas_ = rasterize_to_pixels_eval3d(
                    means,
                    quats,
                    scales,
                    colors_chunk,
                    opacities,
                    viewmats,
                    Ks,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    camera_model=camera_model,
                    radial_coeffs=radial_coeffs,
                    tangential_coeffs=tangential_coeffs,
                    thin_prism_coeffs=thin_prism_coeffs,
                    rolling_shutter=rolling_shutter,
                    viewmats_rs=viewmats_rs,
                )
            else:
                render_colors_, render_alphas_ = rasterize_to_pixels(
                    means2d,
                    conics,
                    colors_chunk,
                    opacities,
                    width,
                    height,
                    tile_size,
                    isect_offsets,
                    flatten_ids,
                    backgrounds=backgrounds_chunk,
                    packed=packed,
                    absgrad=absgrad,
                )
            render_colors.append(render_colors_)
            render_alphas.append(render_alphas_)
        render_colors = torch.cat(render_colors, dim=-1)
        render_alphas = render_alphas[0]  # discard the rest
    else:
        if with_eval3d:
            render_colors, render_alphas = rasterize_to_pixels_eval3d(
                means,
                quats,
                scales,
                colors,
                opacities,
                viewmats,
                Ks,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                camera_model=camera_model,
                radial_coeffs=radial_coeffs,
                tangential_coeffs=tangential_coeffs,
                thin_prism_coeffs=thin_prism_coeffs,
                rolling_shutter=rolling_shutter,
                viewmats_rs=viewmats_rs,
            )
        else:
            render_colors, render_alphas = rasterize_to_pixels(
                means2d,
                conics,
                colors,
                opacities,
                width,
                height,
                tile_size,
                isect_offsets,
                flatten_ids,
                backgrounds=backgrounds,
                packed=packed,
                absgrad=absgrad,
            )
    if render_mode in ["ED", "RGB+ED"]:
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
                ],
            dim=-1,
        )

    return render_colors, render_alphas, meta
