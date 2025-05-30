import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_blur(img: torch.Tensor, sigma: float=1.0) -> torch.Tensor:
    channels = img.shape[1]
    size = int(2 * round(3 * sigma) + 1)
    padding = size // 2

    grid = torch.arange(size, dtype=torch.float32, device=img.device) - size // 2
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel /= kernel.sum()

    kernel_x = kernel.view(1, 1, 1, -1).repeat(channels, 1, 1, 1)
    kernel_y = kernel.view(1, 1, -1, 1).repeat(channels, 1, 1, 1)

    blurred = F.pad(img, (padding, padding, padding, padding), mode="reflect")
    blurred = F.conv2d(blurred, kernel_x, groups=channels)
    blurred = F.conv2d(blurred, kernel_y, groups=channels)
    return blurred

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
    def __init__(self, sigma_list: list[float] | None=None, detail_weight: float=1.0, illum_weight: float=0.5):
        super().__init__()
        if sigma_list is None:
            sigma_list = [15, 80, 250]
            
        self.sigma_list = sigma_list
        self.detail_weight = detail_weight
        self.illum_weight = illum_weight

    def forward(self, input_img: torch.Tensor) -> tuple[torch.Tensor, dict]:
        reflectance = multi_scale_retinex_loss(input_img, self.sigma_list)

        grad_x = reflectance[:, :, :, :-1] - reflectance[:, :, :, 1:]
        grad_y = reflectance[:, :, :-1, :] - reflectance[:, :, 1:, :]
        loss_detail = (grad_x.abs().mean() + grad_y.abs().mean())

        illumination = torch.log(torch.clamp(input_img, 1e-3)) - reflectance
        illum_grad_x = illumination[:, :, :, :-1] - illumination[:, :, :, 1:]
        illum_grad_y = illumination[:, :, :-1, :] - illumination[:, :, 1:, :]
        loss_smooth_illum = (illum_grad_x.abs().mean() + illum_grad_y.abs().mean())

        loss = (self.detail_weight * loss_detail + 
                self.illum_weight * loss_smooth_illum)
        return loss, {
            'detail_loss': loss_detail.item(),
            'illumination_loss': loss_smooth_illum.item()
        }
        

def gradient(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    return grad_x, grad_y

def edge_aware_weight(image: torch.Tensor, lambd: int=10) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x, grad_y = gradient(image)
    weight_x = torch.exp(-lambd * torch.abs(grad_x))
    weight_y = torch.exp(-lambd * torch.abs(grad_y))
    return weight_x, weight_y

def illumination_smoothness_loss(illum: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
    grad_illum_x, grad_illum_y = gradient(illum)
    weight_x, weight_y = edge_aware_weight(input_image)
    loss_x = torch.mean(torch.abs(grad_illum_x * weight_x))
    loss_y = torch.mean(torch.abs(grad_illum_y * weight_y))
    return loss_x + loss_y

def reflectance_smoothness_loss(reflect: torch.Tensor) -> torch.Tensor:
    grad_x, grad_y = gradient(reflect)
    return torch.mean(torch.abs(grad_x)) + torch.mean(torch.abs(grad_y))

def reconstruction_loss(input_image: torch.Tensor, reflect: torch.Tensor, illum: torch.Tensor) -> torch.Tensor:
    recon = reflect * illum
    return F.l1_loss(recon, input_image)

class RetinexLoss(nn.Module):
    def __init__(self, alpha: float=1.0, beta: float=0.1, gamma: float=0.1):
        super(RetinexLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta  
        self.gamma = gamma

    def forward(self, input_image: torch.Tensor, reflect: torch.Tensor, illum: torch.Tensor) -> tuple[torch.Tensor, dict]:
        loss_recon = reconstruction_loss(input_image, reflect, illum)
        loss_illum_smooth = illumination_smoothness_loss(illum, input_image)
        loss_reflect_smooth = reflectance_smoothness_loss(reflect)

        loss = (self.alpha * loss_recon +
                self.beta * loss_illum_smooth +
                self.gamma * loss_reflect_smooth)

        return loss, {
            'recon': loss_recon.item(),
            'illumination_loss': loss_illum_smooth.item(),
            'detail_loss': loss_reflect_smooth.item()
        }

def fixed_retinex_decomposition(renders: torch.Tensor, sigma: float = 3) -> tuple[torch.Tensor, torch.Tensor]:
    renders_clamped = torch.clamp(renders, min=1e-3)
    log_renders = torch.log(renders_clamped)

    blurred = gaussian_blur(renders_clamped, sigma)


    # approximately the illumination as the blurred image
    illum = blurred
    reflect = log_renders - illum

    R = torch.exp(reflect)
    L = torch.exp(illum)

    return R.clamp(0, 1), L.clamp(0, 1)