import random

import numpy as np
from sklearn.neighbors import NearestNeighbors
from torch import Tensor
import matplotlib.pyplot as plt
from matplotlib import colormaps
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tinycudann as tcnn


class CrossAttention(nn.Module):
    def __init__(
        self, img_channels: int = 3, hidden_dim: int = 128, out_dim: int = 255
    ) -> None:
        super(CrossAttention, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, kernel_size=3, padding=1),  # (1, 64, H, W)
            nn.SiLU(),
            nn.MaxPool2d(2, 2),  # (1, 64, H/2, W/2)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # (1, 128, H/2, W/2)
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # (1, 128, 4, 4)
        )

        self.query_fc = nn.Linear(4 * 4, hidden_dim)  # embedding -> query
        self.key_fc = nn.Linear(128, hidden_dim)  # image feature -> key
        self.value_fc = nn.Linear(128, hidden_dim)  # image feature -> value

        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, image, embedding):
        img_feat = self.conv(image)

        img_feat = einops.rearrange(img_feat, "b c h w -> b (h w) c")

        embed_flat = embedding.view(1, -1)  # (1, 16)
        query = self.query_fc(embed_flat).unsqueeze(1)  # (1, 1, hidden_dim)

        key = self.key_fc(img_feat)  # (1, 16, hidden_dim)
        value = self.value_fc(img_feat)  # (1, 16, hidden_dim)

        attention_scores = torch.bmm(query, key.transpose(1, 2)) / (
            128**0.5
        )  # (1, 1, 16)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (1, 1, 16)

        context = torch.bmm(attention_weights, value)  # (1, 1, hidden_dim)
        context = context.squeeze(1)  # (1, hidden_dim)

        output = self.out_fc(context)  # (1, 255)
        return output


class CameraOptModule(torch.nn.Module):
    """Camera pose optimization module."""

    def __init__(self, n: int):
        super().__init__()
        # Delta positions (3D) + Delta rotations (6D)
        self.embeds = torch.nn.Embedding(n, 9)
        # Identity rotation in 6D representation
        self.register_buffer("identity", torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))

    def zero_init(self):
        torch.nn.init.zeros_(self.embeds.weight)

    def random_init(self, std: float):
        torch.nn.init.normal_(self.embeds.weight, std=std)

    def forward(self, camtoworlds: Tensor, embed_ids: Tensor) -> Tensor:
        """Adjust camera pose based on deltas.

        Args:
            camtoworlds: (..., 4, 4)
            embed_ids: (...,)

        Returns:
            updated camtoworlds: (..., 4, 4)
        """
        assert camtoworlds.shape[:-2] == embed_ids.shape
        batch_dims = camtoworlds.shape[:-2]
        pose_deltas = self.embeds(embed_ids)  # (..., 9)
        dx, drot = pose_deltas[..., :3], pose_deltas[..., 3:]
        rot = rotation_6d_to_matrix(
            drot + self.identity.expand(*batch_dims, -1)  # pyright: ignore [reportCallIssue]
        )  # (..., 3, 3)
        transform = torch.eye(4, device=pose_deltas.device).repeat((*batch_dims, 1, 1))
        transform[..., :3, :3] = rot
        transform[..., :3, 3] = dx
        return torch.matmul(camtoworlds, transform)


class AppearanceOptModule(torch.nn.Module):
    """Appearance optimization module."""

    def __init__(
        self,
        n: int,
        feature_dim: int,
        embed_dim: int = 16,
        sh_degree: int = 3,
        mlp_width: int = 64,
        mlp_depth: int = 2,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.sh_degree = sh_degree
        self.embeds = torch.nn.Embedding(n, embed_dim)
        layers = [
            torch.nn.Linear(embed_dim + feature_dim + (sh_degree + 1) ** 2, mlp_width),
            torch.nn.SiLU(inplace=True),
        ]
        for _ in range(mlp_depth - 1):
            layers.append(torch.nn.Linear(mlp_width, mlp_width))
            layers.append(torch.nn.SiLU(inplace=True))
        layers.append(torch.nn.Linear(mlp_width, 3))
        self.color_head = torch.nn.Sequential(*layers)

    def forward(
        self, features: Tensor, embed_ids: Tensor, dirs: Tensor, sh_degree: int
    ) -> Tensor:
        """Adjust appearance based on embeddings.

        Args:
            features: (N, feature_dim)
            embed_ids: (C,)
            dirs: (C, N, 3)
            sh_degree: Spherical harmonics degree to use for view directions.

        Returns:
            colors: (C, N, 3)
        """
        from gsplat.cuda._torch_impl import _eval_sh_bases_fast

        C, N = dirs.shape[:2]
        # Camera embeddings
        if embed_ids is None:
            embeds = torch.zeros(C, self.embed_dim, device=features.device)
        else:
            embeds = self.embeds(embed_ids)  # [C, D2]
        embeds = embeds[:, None, :].expand(-1, N, -1)  # [C, N, D2]
        # GS features
        features = features[None, :, :].expand(C, -1, -1)  # [C, N, D1]
        # View directions
        dirs = F.normalize(dirs, dim=-1)  # [C, N, 3]
        num_bases_to_use = (sh_degree + 1) ** 2
        num_bases = (self.sh_degree + 1) ** 2
        sh_bases = torch.zeros(C, N, num_bases, device=features.device)  # [C, N, K]
        sh_bases[:, :, :num_bases_to_use] = _eval_sh_bases_fast(num_bases_to_use, dirs)
        # Get colors
        if self.embed_dim > 0:
            h = torch.cat([embeds, features, sh_bases], dim=-1)  # [C, N, D1 + D2 + K]
        else:
            h = torch.cat([features, sh_bases], dim=-1)
        colors = self.color_head(h)
        return colors


def rotation_6d_to_matrix(d6: Tensor) -> Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1]. Adapted from pytorch3d.
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from https://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def knn(x: Tensor, K: int = 4) -> Tensor:
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


def rgb_to_sh(rgb: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0

def sh_to_rgb(sh: Tensor) -> Tensor:
    C0 = 0.28209479177387814
    return sh * C0 + 0.5

def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ref: https://github.com/hbb1/2d-gaussian-splatting/blob/main/utils/general_utils.py#L163
def colormap(img, cmap="jet"):
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H / dpi, W / dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data).float().permute(2, 0, 1)
    plt.close()
    return img


def apply_float_colormap(img: torch.Tensor, colormap: str = "turbo") -> torch.Tensor:
    """Convert single channel to a color img.

    Args:
        img (torch.Tensor): (..., 1) float32 single channel image.
        colormap (str): Colormap for img.

    Returns:
        (..., 3) colored img with colors in [0, 1].
    """
    img = torch.nan_to_num(img, 0)
    if colormap == "gray":
        return img.repeat(1, 1, 3)
    img_long = (img * 255).long()
    img_long_min = torch.min(img_long)
    img_long_max = torch.max(img_long)
    assert img_long_min >= 0, f"the min value is {img_long_min}"
    assert img_long_max <= 255, f"the max value is {img_long_max}"
    return torch.tensor(
        colormaps[colormap].colors,  # type: ignore
        device=img.device,
    )[img_long[..., 0]]


def apply_depth_colormap(
    depth: torch.Tensor,
    acc: torch.Tensor | None = None,
    near_plane: float | None = None,
    far_plane: float | None = None,
) -> torch.Tensor:
    """Converts a depth image to color for easier analysis.

    Args:
        depth (torch.Tensor): (..., 1) float32 depth.
        acc (torch.Tensor | None): (..., 1) optional accumulation mask.
        near_plane: Closest depth to consider. If None, use min image value.
        far_plane: Furthest depth to consider. If None, use max image value.

    Returns:
        (..., 3) colored depth image with colors in [0, 1].
    """
    near_plane = near_plane or float(torch.min(depth))
    far_plane = far_plane or float(torch.max(depth))
    depth = (depth - near_plane) / (far_plane - near_plane + 1e-10)
    depth = torch.clip(depth, 0.0, 1.0)
    img = apply_float_colormap(depth)
    if acc is not None:
        img = img * acc + (1.0 - acc)
    return img

def quaternion_to_matrix(quaternions: Tensor) -> Tensor:
    quaternions = F.normalize(quaternions, p=2, dim=-1)

    w, x, y, z = torch.unbind(quaternions, -1)

    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = torch.stack([
        1 - 2 * (y2 + z2), 2 * (xy - wz),     2 * (xz + wy),
        2 * (xy + wz),     1 - 2 * (x2 + z2), 2 * (yz - wx),
        2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (x2 + y2)
    ], dim=-1)

    return mat.reshape(quaternions.shape[:-1] + (3, 3))

def generate_variational_intrinsics(
    base_K: Tensor,
    num_intrinsics: int,
    focal_perturb_factor: float,
    principal_point_perturb_pixel: int,
) -> Tensor:
    device = base_K.device
    fx_base, fy_base = base_K[0, 0], base_K[1, 1]
    cx_base, cy_base = base_K[0, 2], base_K[1, 2]

    new_Ks = base_K.unsqueeze(0).repeat(num_intrinsics, 1, 1)

    focal_perturb = (
        1.0
        + (torch.rand(num_intrinsics, 2, device=device) * 2 - 1) * focal_perturb_factor
    )
    new_Ks[:, 0, 0] = fx_base * focal_perturb[:, 0]
    new_Ks[:, 1, 1] = fy_base * focal_perturb[:, 1]

    principal_point_perturb = (
        torch.rand(num_intrinsics, 2, device=device) * 2 - 1
    ) * principal_point_perturb_pixel
    new_Ks[:, 0, 2] = cx_base + principal_point_perturb[:, 0]
    new_Ks[:, 1, 2] = cy_base + principal_point_perturb[:, 1]

    return new_Ks


class PositionalEncoder(nn.Module):
    def __init__(self, num_freqs: int):
        super().__init__()
        self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, num_freqs)

    def forward(self, x: Tensor) -> Tensor:
        # x: [..., 3]
        # returns: [..., 3 * 2 * num_freqs]
        x = x.unsqueeze(-1) # [..., 3, 1]
        projs = x * self.freq_bands.to(x.device) # [..., 3, N_freqs]
        return torch.cat([torch.sin(projs), torch.cos(projs)], dim=-1).flatten(-2)


class IlluminationField(nn.Module):
    def __init__(self,
                 scene_scale: float,
                 num_freqs: int = 4,
                 dir_num_freqs: int = 4,
                 hidden_dim: int = 64,
                 num_layers: int = 2,
                 use_hash_grid: bool = True,
                 use_view_dirs: bool = True,
                 use_appearance_embeds: bool = False,
                 use_normals: bool = True,
                 appearance_embedding_dim: int = 32,
                 ):
        super().__init__()
        self.scene_scale = scene_scale
        self.use_hash_grid = use_hash_grid
        self.use_view_dirs = use_view_dirs
        self.use_appearance_embeds = use_appearance_embeds
        self.use_normals = use_normals

        self.hidden_dim = hidden_dim

        if self.use_hash_grid:
            per_level_scale = 1.4472692012786865
            self.encoder = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid",
                    "n_levels": 16,
                    "n_features_per_level": 2,
                    "log2_hashmap_size": 19,
                    "base_resolution": 16,
                    "per_level_scale": per_level_scale,
                },
                dtype=torch.float32,
            )
            pos_in_dim = self.encoder.n_output_dims
        else:
            self.encoder = PositionalEncoder(num_freqs)
            pos_in_dim = 3 * 2 * num_freqs

        self.dir_in_dim = 0
        if self.use_view_dirs:
            self.dir_encoder = PositionalEncoder(dir_num_freqs)
            self.dir_in_dim += 3 * 2 * dir_num_freqs

        self.normal_in_dim = 0
        if self.use_normals:
            self.normal_encoder = PositionalEncoder(num_freqs)
            self.normal_in_dim += 3 * 2 * num_freqs

        self.embed_in_dim = 0
        if use_appearance_embeds:
            self.embed_in_dim += appearance_embedding_dim

        in_dim = pos_in_dim + self.dir_in_dim + self.embed_in_dim + self.normal_in_dim

        self.mlp_base = tcnn.Network(
            n_input_dims=in_dim,
            n_output_dims=hidden_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        self.mlp_head = nn.Linear(hidden_dim, 12)

        with torch.no_grad():
            self.mlp_head.weight.data.normal_(0.0, 1e-4)
            self.mlp_head.bias.data.zero_()

    def forward(self, x: Tensor, embeds: Tensor | None = None, view_dirs: Tensor | None = None, normals: Tensor |
                                                                                                         None = None)\
            -> tuple[Tensor, Tensor]:
        # x: [N, 3]
        if self.use_hash_grid:
            normalized_x = x / (2.0 * self.scene_scale) + 0.5
            normalized_x = torch.clamp(normalized_x, 0.0, 1.0)
            encoded_x = self.encoder(normalized_x)
        else:
            encoded_x = self.encoder(x)

        mlp_input = [encoded_x]

        if self.use_view_dirs:
            if view_dirs is not None:
                encoded_dirs = self.dir_encoder(F.normalize(view_dirs, dim=-1))
                mlp_input.append(encoded_dirs)
            else:
                zeros = torch.zeros(x.shape[0], self.dir_in_dim, device=x.device)
                mlp_input.append(zeros)

        if self.use_normals:
            if normals is not None:
                encoded_normals = self.normal_encoder(F.normalize(normals, dim=-1))
                mlp_input.append(encoded_normals)
            else:
                zeros = torch.zeros(x.shape[0], self.normal_in_dim, device=x.device)
                mlp_input.append(zeros)

        if self.use_appearance_embeds:
            if embeds is not None:
                num_points = x.shape[0]
                broadcasted_embeds = embeds.expand(num_points, -1)
                mlp_input.append(broadcasted_embeds)
            else:
                zeros = torch.zeros(x.shape[0], self.embed_in_dim, device=x.device)
                mlp_input.append(zeros)

        mlp_input_tensor = torch.cat(mlp_input, dim=-1)

        hidden_features = self.mlp_base(mlp_input_tensor)
        residual = mlp_input_tensor[:, :self.hidden_dim]
        hidden_features += residual
        params = self.mlp_head(hidden_features.float())

        matrix_A_flat = params[..., :9]
        bias_b = params[..., 9:]

        num_points = x.shape[0]
        identity = torch.eye(3, device=x.device).unsqueeze(0).expand(num_points, -1, -1)
        matrix_A = matrix_A_flat.view(num_points, 3, 3) + identity

        return matrix_A, bias_b

class CameraResponseNet(nn.Module):
    def __init__(self, embedding_dim, hidden_dim=32):
        super().__init__()
        self.mlp_base = tcnn.Network(
            n_input_dims=embedding_dim,
            n_output_dims=hidden_dim,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": 2,
            },
        )
        self.mlp_head = nn.Linear(hidden_dim, 6)

        with torch.no_grad():
            self.mlp_head.weight.zero_()
            self.mlp_head.bias.zero_()
            self.mlp_head.bias.data[0:3] = 1.0

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        # embedding: [B, D_embed]
        hidden_features = self.mlp_base(embedding)
        params = self.mlp_head(hidden_features.float()) # [B, 6]

        c, d = params.split(3, dim=-1) # 2 x [B, 3]
        return c, d
