import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as K
from torch import finfo


def multi_scale_retinex_loss(input_img: torch.Tensor, sigma_list=None) -> torch.Tensor:
    if sigma_list is None:
        sigma_list = [15, 80, 250]

    img = torch.clamp(input_img, 1e-3, 1.0)
    log_img = torch.log(img)

    retinex_components = []
    for sigma in sigma_list:
        blur = gaussian_blur(img, sigma)
        blur = torch.clamp(blur, min=1e-3)
        log_blur = torch.log(blur)
        retinex = log_img - log_blur
        retinex_components.append(retinex)

    return torch.mean(torch.stack(retinex_components), dim=0)


class RetinexLossMSR(nn.Module):
    def __init__(
        self,
        sigma_list: list[float] | None = None,
        detail_weight: float = 0.0,
        illum_weight: float = 1.0,
    ):
        super().__init__()
        if sigma_list is None:
            sigma_list = [15, 80, 250]

        self.sigma_list = sigma_list
        self.detail_weight = detail_weight
        self.illum_weight = illum_weight

    def forward(
        self, input_img: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        reflectance = multi_scale_retinex_loss(input_img, self.sigma_list)

        grad_x = reflectance[:, :, :, :-1] - reflectance[:, :, :, 1:]
        grad_y = reflectance[:, :, :-1, :] - reflectance[:, :, 1:, :]
        loss_detail = grad_x.abs().mean() + grad_y.abs().mean()

        illumination = torch.log(torch.clamp(input_img, 1e-3)) - reflectance
        illum_grad_x = illumination[:, :, :, :-1] - illumination[:, :, :, 1:]
        illum_grad_y = illumination[:, :, :-1, :] - illumination[:, :, 1:, :]
        loss_smooth_illum = illum_grad_x.abs().mean() + illum_grad_y.abs().mean()

        loss = self.detail_weight * loss_detail + self.illum_weight * loss_smooth_illum
        return loss, {
            "detail_loss": loss_detail,
            "illumination_loss": loss_smooth_illum,
            "reflectance": reflectance,
            "illumination": illumination,
        }


def gradient(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    return grad_x, grad_y


def illumination_smoothness_loss(illum: torch.Tensor) -> torch.Tensor:
    grad_x, grad_y = gradient(illum)
    return torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))


def reflectance_smoothness_loss(reflect: torch.Tensor) -> torch.Tensor:
    grad_x, grad_y = gradient(reflect)
    return torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))


def reconstruction_loss(
    input_image: torch.Tensor, reflect: torch.Tensor, illum: torch.Tensor
) -> torch.Tensor:
    recon = reflect * illum
    return F.l1_loss(recon, input_image)


class RetinexLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, beta: float = 0.1, gamma: float = 0.1):
        super(RetinexLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(
        self, input_image: torch.Tensor, reflect: torch.Tensor, illum: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        loss_recon = reconstruction_loss(input_image, reflect, illum)
        loss_illum_smooth = illumination_smoothness_loss(illum)
        loss_reflect_smooth = reflectance_smoothness_loss(reflect)

        loss = (
            self.alpha * loss_recon
            + self.beta * loss_illum_smooth
            + self.gamma * loss_reflect_smooth
        )

        return loss, {
            "recon": loss_recon,
            "illumination_loss": loss_illum_smooth,
            "detail_loss": loss_reflect_smooth,
            "reflectance": reflect,
            "illumination": illum,
        }


def fixed_retinex_decomposition(
    renders: torch.Tensor, sigma: float = 3
) -> tuple[torch.Tensor, torch.Tensor]:
    s_linear = torch.clamp(renders, min=1e-6, max=1.0)

    l_linear_approx = gaussian_blur(s_linear, sigma)
    l_linear_final = torch.clamp(l_linear_approx, min=1e-6, max=1.0)

    r_linear_final = torch.clamp(s_linear / (l_linear_final + 1e-6), min=0.0, max=1.0)

    return r_linear_final, l_linear_final


@torch.compile
def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(2 * round(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    return K.gaussian_blur2d(img, (kernel_size, kernel_size), (sigma, sigma))


@torch.compile
def retinex_on_v_channel(
    renders_rgb: torch.Tensor,
    sigma_list: list[float] | None = None,
    gain: float = 1.2,
    offset: float = 0.0,
) -> torch.Tensor:
    epsilon = finfo(renders_rgb.dtype).eps

    if sigma_list is None:
        sigma_list = [15, 80, 250]

    renders_hsv = kornia.color.rgb_to_hsv(renders_rgb)

    v_channel = renders_hsv[:, 2:3, :, :]

    s_linear = torch.clamp(v_channel, min=epsilon)
    log_s = torch.log(s_linear)

    retinex_components = []
    for sigma in sigma_list:
        s_blurred = gaussian_blur(s_linear, sigma)
        log_s_blurred = torch.log(torch.clamp(s_blurred, min=epsilon))
        retinex_components.append(log_s - log_s_blurred)

    log_R = torch.mean(torch.stack(retinex_components), dim=0)
    v_new_linear = torch.exp(log_R)

    v_new = gain * v_new_linear + offset
    v_new_clamped = torch.clamp(v_new, 0.0, 1.0)

    h_s_channels = renders_hsv[:, 0:2, :, :]
    final_hsv = torch.cat([h_s_channels, v_new_clamped], dim=1)

    final_rgb = kornia.color.hsv_to_rgb(final_hsv)

    return torch.clamp(final_rgb, 0.0, 1.0)


@torch.compile
def multi_scale_retinex_decomposition(
    renders: torch.Tensor,
    sigma_list: list[float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    epsilon_log = finfo(renders.dtype).eps
    epsilon_div = finfo(renders.dtype).eps

    if sigma_list is None:
        sigma_list = [15, 80, 250]

    s_linear = torch.clamp(renders, min=epsilon_log, max=1.0)
    log_s = torch.log(s_linear)

    retinex_components = []

    for sigma in sigma_list:
        log_illumination_estimate = gaussian_blur(log_s, sigma)
        retinex_components.append(log_s - log_illumination_estimate)

    log_r = torch.mean(torch.stack(retinex_components), dim=0)

    r_linear = torch.exp(log_r)

    r_final = torch.clamp(r_linear, 0.0, 1.0)

    l_final = torch.clamp(s_linear / (r_final + epsilon_div), 0.0, 1.0)

    return r_final, l_final
