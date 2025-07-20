import math

import torch
from torch import nn, Tensor
import torch.nn.functional as F

class Light(nn.Module):
    """
    Base class for all light sources.
    """
    def __init__(self, initial_color=None):
        super().__init__()
        if initial_color is None:
            initial_color = torch.ones(3)
        self.raw_color = nn.Parameter(torch.log(torch.expm1(initial_color)))

    @property
    def color(self):
        """ The color of the light, ensured to be positive. """
        return F.softplus(self.raw_color)

    def get_contribution(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """

        Calculates the light's contribution at given 3D points.

        Args:
            points (Tensor): A tensor of shape [N, 3] representing points in space.

        Returns:
            A tuple containing:
            - light_colors (Tensor): The color of the light at each point [N, 3].
            - light_dirs (Tensor): The normalized direction from the points to the light [N, 3].
            - attenuation (Tensor): The attenuation factor for each point [N, 1].
        """
        raise NotImplementedError

class DirectionalLight(Light):
    """
    Represents a light source at an infinite distance, defined by a direction.
    """
    def __init__(self, initial_direction=None, initial_color=None):
        super().__init__(initial_color)
        if initial_direction is None:
            initial_direction = torch.tensor([0.0, 1.0, 0.0])
        self.direction = nn.Parameter(F.normalize(initial_direction, dim=-1))

    def get_contribution(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N = points.shape[0]
        light_dir = F.normalize(self.direction, dim=-1)
        # Directional lights have no attenuation
        attenuation = torch.ones(N, 1, device=points.device)
        return self.color.expand(N, -1), light_dir.expand(N, -1), attenuation

class PointLight(Light):
    """
    Represents a light that emits uniformly in all directions from a single point.
    """
    def __init__(self, initial_position=None, initial_color=None):
        super().__init__(initial_color)
        if initial_position is None:
            initial_position = torch.zeros(3)
        self.position = nn.Parameter(initial_position)

    def get_contribution(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N = points.shape[0]
        light_dirs = self.position - points
        distance_sq = torch.sum(light_dirs**2, dim=-1, keepdim=True)
        light_dirs = F.normalize(light_dirs, dim=-1)

        attenuation = 1.0 / (distance_sq + 1e-6)

        return self.color.expand(N, -1), light_dirs, attenuation

class SpotLight(Light):
    """
    Represents a light that emits in a cone shape.
    """
    def __init__(self, initial_position=None, initial_direction=None, initial_color=None,
                 cone_angle_deg: float = 30.0, falloff_factor: float = 2.0):
        super().__init__(initial_color)
        if initial_position is None:
            initial_position = torch.zeros(3)
        if initial_direction is None:
            initial_direction = torch.tensor([0.0, -1.0, 0.0])

        self.position = nn.Parameter(initial_position)
        self.direction = nn.Parameter(F.normalize(initial_direction, dim=-1))

        self.register_buffer('cone_angle', torch.tensor(math.radians(cone_angle_deg)))
        self.register_buffer('falloff_factor', torch.tensor(falloff_factor))

        self.register_buffer('cos_cone_angle', torch.cos(self.cone_angle))

    def get_contribution(self, points: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        N = points.shape[0]
        light_dirs = self.position - points
        distance_sq = torch.sum(light_dirs**2, dim=-1, keepdim=True)
        light_dirs = F.normalize(light_dirs, dim=-1)

        spot_dir = F.normalize(self.direction, dim=-1)
        cos_theta = torch.sum(light_dirs * spot_dir, dim=-1, keepdim=True)

        in_cone_mask = (cos_theta > self.cos_cone_angle).float()

        falloff = torch.clamp((cos_theta - self.cos_cone_angle) / (1.0 - self.cos_cone_angle + 1e-6), 0.0, 1.0)
        spot_attenuation = torch.pow(falloff, self.falloff_factor) * in_cone_mask

        distance_attenuation = 1.0 / (distance_sq + 1e-6)

        total_attenuation = spot_attenuation * distance_attenuation

        return self.color.expand(N, -1), light_dirs, total_attenuation

class AmbientLight(nn.Module):
    def __init__(self, initial_color=None):
        super().__init__()
        if initial_color is None:
            initial_color = torch.tensor([0.1, 0.1, 0.1])
        self.raw_color = nn.Parameter(initial_color)

    @property
    def color(self):
        return F.softplus(self.raw_color)

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


class ShadowField(nn.Module):
    def __init__(self, num_lights: int, num_freqs: int = 6, hidden_dim: int = 64, num_layers: int = 3):
        super().__init__()
        self.encoder = PositionalEncoder(num_freqs)
        in_dim = 3 * 2 * num_freqs
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, num_lights))
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: Tensor) -> Tensor:
        encoded_points = self.encoder(points)
        shadow_params = self.mlp(encoded_points)
        return torch.sigmoid(shadow_params)

class LearnedIlluminationField(nn.Module):
    def __init__(self, num_directional_lights=1, num_point_lights=2):
        super().__init__()
        self.num_directional_lights = num_directional_lights
        self.num_point_lights = num_point_lights

        dir_light_params = self.num_directional_lights * 6
        point_light_params = self.num_point_lights * 6

        total_params = dir_light_params + point_light_params

        self.mlp = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, total_params)
        )

        self.latent_code = nn.Parameter(torch.randn(1, 1))

    def forward(self) -> list[Light]:
        params = self.mlp(self.latent_code).squeeze(0)

        lights = []
        current_idx = 0

        for _ in range(self.num_directional_lights):
            direction = params[current_idx : current_idx+3]
            color = params[current_idx+3 : current_idx+6]
            current_idx += 6

            lights.append(
                DirectionalLight(initial_direction=torch.tanh(direction), initial_color=F.softplus(color))
            )

        for _ in range(self.num_point_lights):
            position = params[current_idx : current_idx+3]
            color = params[current_idx+3 : current_idx+6]
            current_idx += 6

            lights.append(
                PointLight(initial_position=torch.tanh(position) * 3.0, initial_color=F.softplus(color))
            )

        return lights

class PhysicsAwareIllumination(nn.Module):
    def __init__(self, lights: list[Light] | LearnedIlluminationField):
        super().__init__()
        self.ambient_light = AmbientLight()

        if isinstance(lights, LearnedIlluminationField):
            self.learned_illumination = lights
            self.mode = 'learned'
            self.num_dynamic_lights = lights.num_directional_lights + lights.num_point_lights
        else:
            self.static_lights = nn.ModuleList(lights)
            self.mode = 'static'
            self.num_dynamic_lights = len(lights)

        if self.num_dynamic_lights > 0:
            self.shadow_field = ShadowField(num_lights=self.num_dynamic_lights)
        else:
            self.shadow_field = None


    def forward(self, points: Tensor) -> tuple[Tensor, list[tuple[Tensor, Tensor, Tensor]]]:
        N = points.shape[0]
        ambient_color = self.ambient_light.color.expand(N, -1)

        if self.mode == 'learned':
            dynamic_lights = self.learned_illumination()
        else:
            dynamic_lights = self.static_lights

        if self.shadow_field is not None and len(dynamic_lights) > 0:
            shadow_factors = self.shadow_field(points)
        else:
            shadow_factors = torch.ones(N, len(dynamic_lights), device=points.device)

        light_contributions = []
        for i, light in enumerate(dynamic_lights):
            light.to(points.device)
            color, direction, attenuation = light.get_contribution(points)
            shadow = shadow_factors[:, i:i+1]
            light_contributions.append((color, direction, attenuation * shadow))

        return ambient_color, light_contributions



class IrradianceField(nn.Module):
    def __init__(self, num_freqs: int = 4, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        self.pos_encoder = PositionalEncoder(num_freqs)
        # encoded position (2*3*num_freqs) + normal (3)
        in_dim = (3 * 2 * num_freqs) + 3

        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
        layers.append(nn.Linear(hidden_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, points: Tensor, normals: Tensor) -> Tensor:
        encoded_points = self.pos_encoder(points)
        mlp_input = torch.cat([encoded_points, normals], dim=-1)
        raw_irradiance = self.mlp(mlp_input)
        return F.softplus(raw_irradiance)
