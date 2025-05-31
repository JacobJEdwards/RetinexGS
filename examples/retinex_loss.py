import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.filters as K

def gaussian_blur(img: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = int(2 * round(3 * sigma) + 1)
    return K.gaussian_blur2d(img, (kernel_size, kernel_size), (sigma, sigma))


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
    def __init__(self, sigma_list: list[float] | None=None, detail_weight: float=0.0, illum_weight: float=1.0):
        super().__init__()
        if sigma_list is None:
            sigma_list = [15, 80, 250]
            
        self.sigma_list = sigma_list
        self.detail_weight = detail_weight
        self.illum_weight = illum_weight

    def forward(self, input_img: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
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
            'detail_loss': loss_detail,
            'illumination_loss': loss_smooth_illum,
            'reflectance': reflectance,
            'illumination': illumination
        }
        

def gradient(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grad_x = img[:, :, :, :-1] - img[:, :, :, 1:]
    grad_y = img[:, :, :-1, :] - img[:, :, 1:, :]
    return grad_x, grad_y


def illumination_smoothness_loss(illum: torch.Tensor, input_image: torch.Tensor) -> torch.Tensor:
    grad_x, grad_y = gradient(illum)
    grad_x_input, grad_y_input = gradient(input_image)
    
    return (torch.mean(torch.abs(grad_x)) + 
            torch.mean(torch.abs(grad_y)) + 
            torch.mean(torch.abs(grad_x_input)) + 
            torch.mean(torch.abs(grad_y_input)))

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

    def forward(self, input_image: torch.Tensor, reflect: torch.Tensor, illum: torch.Tensor) -> tuple[torch.Tensor, 
                                                                                                dict[str, torch.Tensor]]:
        loss_recon = reconstruction_loss(input_image, reflect, illum)
        loss_illum_smooth = illumination_smoothness_loss(illum, input_image)
        loss_reflect_smooth = reflectance_smoothness_loss(reflect)

        loss = (self.alpha * loss_recon +
                self.beta * loss_illum_smooth +
                self.gamma * loss_reflect_smooth)

        return loss, {
            'recon': loss_recon,
            'illumination_loss': loss_illum_smooth,
            'detail_loss': loss_reflect_smooth,
            'reflectance': reflect,
            'illumination': illum
        }

def fixed_retinex_decomposition(renders: torch.Tensor, sigma: float = 3) -> tuple[torch.Tensor, torch.Tensor]:
    s_linear = torch.clamp(renders, min=1e-6, max=1.0) 

    l_linear_approx = gaussian_blur(s_linear, sigma)
    l_linear_final = torch.clamp(l_linear_approx, min=1e-6, max=1.0)

    r_linear_final = torch.clamp(s_linear / (l_linear_final + 1e-6), min=0.0, max=1.0)

    return r_linear_final, l_linear_final
