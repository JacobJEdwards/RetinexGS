from typing import Self

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def gamma_curve(x, g):
    # Power Curve, Gamma Correction Curve
    y = torch.clamp(x, 1e-3, 1)
    y = y**g
    return y


def s_curve(x, alpha=0.5, beta=1.0):
    # S-Curve, from "Personalization of image enhancement (CVPR 2010)"
    below_alpha = x <= alpha

    ratio_below_alpha = torch.clamp(x / alpha, 0, 1 - 1e-3)
    s_below_alpha = alpha - alpha * ((1 - ratio_below_alpha) ** beta)

    ratio_above_alpha = torch.clamp((x - alpha) / (1 - alpha), 0, 1 - 1e-3)
    s_above_alpha = alpha + (1 - alpha) * (ratio_above_alpha**beta)

    return torch.where(below_alpha, s_below_alpha, s_above_alpha)


# inverse tone curve MSE loss
def img2mse_tone(x, y):
    eta = 1e-4
    x = torch.clip(x, min=eta, max=1 - eta)
    # the inverse tone curve, pls refer to paper (Eq.13):
    # "https://openaccess.thecvf.com/content/ICCV2021/papers/Cui_Multitask_AET_With_Orthogonal_Tangent_Regularity_for_Dark_Object_Detection_ICCV_2021_paper.pdf"
    f = lambda c: 0.5 - torch.sin(torch.asin(1.0 - 2.0 * c) / 3.0)
    return torch.mean((f(x) - f(y)) ** 2)


def curve_loss(x, y, z):
    dim = x.shape[-1]
    coor = torch.linspace(0, 1, dim).unsqueeze(0).to(x.device)
    der_x = (x[:, 1:] - x[:, :-1]) / (coor[:, 1:] - coor[:, :-1])
    der_y = (y[:, 1:] - y[:, :-1]) / (coor[:, 1:] - coor[:, :-1])
    der_z = (z[:, 1:] - z[:, :-1]) / (coor[:, 1:] - coor[:, :-1])
    # consine similarity
    loss = (
        (1 - torch.mean(F.cosine_similarity(der_x, der_y)))
        + (1 - torch.mean(F.cosine_similarity(der_y, der_z)))
        + (1 - torch.mean(F.cosine_similarity(der_x, der_z)))
    )
    return loss


class HistogramPriorLoss(nn.Module):
    def __init__(self, lambda_smooth=0.1):
        super(HistogramPriorLoss, self).__init__()
        self.lambda_smooth = lambda_smooth
        # self.lambda_reg = lambda_reg

    @staticmethod
    def compute_histogram_equalization(inp):
        # Resize Images
        inp = torch.mean(
            nn.functional.interpolate(inp.permute(0, 3, 1, 2), scale_factor=0.25),
            dim=1,
        )
        flat = inp.flatten()
        hist = torch.histc(flat, bins=255, min=0.0, max=1.0)
        cdf = torch.cumsum(hist, dim=0)
        cdf /= cdf[-1]
        return cdf.unsqueeze(0)

    def forward(self, output, inp, psedo_curve, step):
        hist_eq_prior = HistogramPriorLoss.compute_histogram_equalization(inp)

        cl = torch.mean((output - hist_eq_prior) ** 2)

        psedo_curve_loss = torch.mean((psedo_curve - output) ** 2) + 0.01 * torch.mean(
            (psedo_curve - hist_eq_prior) ** 2
        )

        smooth_loss = torch.mean((output[:, 1:] - output[:, :-1]) ** 2)

        total_loss = cl + self.lambda_smooth * smooth_loss + 0.5 * psedo_curve_loss

        if step >= 3000:
            total_loss = (
                0.1 * cl + self.lambda_smooth * smooth_loss + 0.5 * psedo_curve_loss
            )

        return total_loss


# Adaptive Curve Loss
class AdaptiveCurveLoss(nn.Module):
    def __init__(
        self: Self,
        alpha: float=0.3,
        beta: float=0.7,
        low_thresh: float=0.3,
        high_thresh: float=0.7,
        lambda1: float=1.0,
        lambda2: float=1.0,
        lambda3: float=1.0,
    ):
        """
        Custom loss function for controlling curve enhancement and compression.
        :param alpha: Controls the enhancement strength in dark regions (>1 enhances low light)
        :param beta: Controls the suppression strength in highlight regions (<1 compresses highlights)
        :param low_thresh: Low-light threshold for segment control in dark regions (default 0.3)
        :param high_thresh: Highlight threshold for segment control in bright regions (default 0.7)
        :param lambda1: Weight of the loss for dark region enhancement
        :param lambda2: Weight of the loss for highlight suppression
        :param lambda3: Weight of the smoothness loss
        """
        super(AdaptiveCurveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self: Self, output: Tensor) -> Tensor:
        low_mask = (output < self.low_thresh).float()
        low_light_loss = torch.mean(low_mask * torch.abs(output - self.alpha))

        high_mask = (output > self.high_thresh).float()
        high_light_loss = torch.mean(high_mask * torch.abs(output - self.beta))
        
        grad_y = (output[:, :, 1:, :] - output[:, :, :-1, :]) ** 2
        grad_x = (output[:, :, :, 1:] - output[:, :, :, :-1]) ** 2
        smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)

        total_loss = (
                self.lambda1 * low_light_loss
                + self.lambda2 * high_light_loss
                + self.lambda3 * smooth_loss
        )

        return total_loss

class ColourConsistencyLoss(nn.Module):
    def __init__(self: Self) -> None:
        super(ColourConsistencyLoss, self).__init__()

    def forward(self: Self, x: Tensor) -> Tensor:
        eps = torch.finfo(x.dtype).eps

        mean_rgb = torch.mean(x, [2, 3], keepdim=True)
        mr, mg, mb = torch.chunk(mean_rgb, 3, dim=1)

        Drg = (mr - mg) ** 2
        Drb = (mr - mb) ** 2
        Dgb = (mg - mb) ** 2

        k = torch.sqrt(Drg + Drb + Dgb + eps)

        return k.mean()


# Exposure Loss, control the generated image exposure
class ExposureLoss(nn.Module):
    def __init__(self: Self, patch_size: int, mean_val: float = 0.5) -> None:
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.register_buffer('mean_val_tensor', torch.tensor([mean_val]))

    def forward(self: Self, x: Tensor) -> Tensor:
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - self.mean_val_tensor, 2))
        return d

class SpatialLoss(nn.Module):
    weight_left: Tensor
    weight_right: Tensor
    weight_up: Tensor
    weight_down: Tensor
    
    def __init__(self: Self) -> None:
        super(SpatialLoss, self).__init__()
        kernel_left = torch.tensor([[0,0,0],[-1,1,0],[0,0,0]], dtype=torch.float32).view(1,1,3,3)
        kernel_right = torch.tensor([[0,0,0],[0,1,-1],[0,0,0]], dtype=torch.float32).view(1,1,3,3)
        kernel_up = torch.tensor([[0,-1,0],[0,1,0],[0,0,0]], dtype=torch.float32).view(1,1,3,3)
        kernel_down = torch.tensor([[0,0,0],[0,1,0],[0,-1,0]], dtype=torch.float32).view(1,1,3,3)

        self.register_buffer('weight_left', kernel_left)
        self.register_buffer('weight_right', kernel_right)
        self.register_buffer('weight_up', kernel_up)
        self.register_buffer('weight_down', kernel_down)

        self.pool = nn.AvgPool2d(4)
        
    def forward(self: Self, org: Tensor, enhance: Tensor, contrast: int = 8):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        def p(pool: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            org_letf = F.conv2d(pool, self.weight_left, padding=1)
            org_right = F.conv2d(pool, self.weight_right, padding=1)
            org_up = F.conv2d(pool, self.weight_up, padding=1)
            org_down = F.conv2d(pool, self.weight_down, padding=1)

            return org_letf, org_right, org_up, org_down

        D_org_letf, D_org_right, D_org_up, D_org_down = p(org_pool)
        D_enhance_letf, D_enhance_right, D_enhance_up, D_enhance_down = p(enhance_pool)

        D_left = torch.pow(D_org_letf * contrast - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right * contrast - D_enhance_right, 2)
        D_up = torch.pow(D_org_up * contrast - D_enhance_up, 2)
        D_down = torch.pow(D_org_down * contrast - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return E.mean()

class LaplacianLoss(nn.Module):
    kernel: Tensor
    
    def __init__(self: Self) -> None:
        super(LaplacianLoss, self).__init__()
        kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kernel', kernel)

    def forward(self: Self, x: Tensor) -> Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True)
        laplacian = F.conv2d(x_gray, self.kernel, padding=1)
        return torch.mean(torch.abs(laplacian))

class TotalVariationLoss(nn.Module):
    def __init__(self: Self, weight: float = 1.0) -> None:
        super(TotalVariationLoss, self).__init__()
        self.weight = weight
        
    @staticmethod
    def _tensor_size(t: Tensor) -> int:
        return t.size()[1]*t.size()[2]*t.size()[3]

    def forward(self: Self, x: Tensor) -> Tensor:
        diff_h = x[:, :, 1:, :] - x[:, :, :-1, :]
        diff_w = x[:, :, :, 1:] - x[:, :, :, :-1]

        tv_h = torch.pow(diff_h, 2).sum()
        tv_w = torch.pow(diff_w, 2).sum()

        loss = (tv_h + tv_w) / x.numel()

        return self.weight * loss

class SmoothingLoss(nn.Module):
    weight_left: Tensor
    weight_right: Tensor
    weight_up: Tensor
    weight_down: Tensor
    
    def __init__(self: Self) -> None:
        super(SmoothingLoss, self).__init__()
        kernel_left = torch.tensor([[0,0,0],[-1,1,0],[0,0,0]], dtype=torch.float32).view(1,1,3,3)
        kernel_right = torch.tensor([[0,0,0],[0,1,-1],[0,0,0]], dtype=torch.float32).view(1,1,3,3)
        kernel_up = torch.tensor([[0,-1,0],[0,1,0],[0,0,0]], dtype=torch.float32).view(1,1,3,3)
        kernel_down = torch.tensor([[0,0,0],[0,1,0],[0,-1,0]], dtype=torch.float32).view(1,1,3,3)

        self.register_buffer('weight_left', kernel_left)
        self.register_buffer('weight_right', kernel_right)
        self.register_buffer('weight_up', kernel_up)
        self.register_buffer('weight_down', kernel_down)

        self.pool = nn.AvgPool2d(4)

    def forward(self: Self, enhance: Tensor): 
        enhance_mean = torch.mean(enhance, 1, keepdim=True)
        enhance_pool = self.pool(enhance_mean)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_enhance_letf, 2)
        D_right = torch.pow(D_enhance_right, 2)
        D_up = torch.pow(D_enhance_up, 2)
        D_down = torch.pow(D_enhance_down, 2)

        E = torch.mean(D_left + D_right + D_up + D_down)

        return E

class GradientLoss(nn.Module):
    kernel_x: Tensor
    kernel_y: Tensor

    def __init__(self: Self) -> None:
        super(GradientLoss, self).__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('kernel_x', kernel_x)
        self.register_buffer('kernel_y', kernel_y)

    def forward(self: Self, x: Tensor, y: Tensor) -> Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True)
        y_gray = torch.mean(y, dim=1, keepdim=True)

        grad_x_x = F.conv2d(x_gray, self.kernel_x, padding=1)
        grad_y_x = F.conv2d(y_gray, self.kernel_x, padding=1)

        grad_x_y = F.conv2d(x_gray, self.kernel_y, padding=1)
        grad_y_y = F.conv2d(y_gray, self.kernel_y, padding=1)

        loss_x = torch.mean(torch.abs(grad_x_x - grad_y_x))
        loss_y = torch.mean(torch.abs(grad_x_y - grad_y_y))

        return loss_x + loss_y

class FrequencyLoss(nn.Module):
    def __init__(self: Self) -> None:
        super(FrequencyLoss, self).__init__()

    def forward(self: Self, x: Tensor, y: Tensor) -> Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True).squeeze(1) 
        y_gray = torch.mean(y, dim=1, keepdim=True).squeeze(1)

        fft_x = torch.fft.fftshift(torch.fft.rfft2(x_gray, norm="ortho"))
        fft_y = torch.fft.fftshift(torch.fft.rfft2(y_gray, norm="ortho"))

        magnitude_x = torch.log(torch.abs(fft_x) + 1e-8)
        magnitude_y = torch.log(torch.abs(fft_y) + 1e-8)

        loss = F.l1_loss(magnitude_x, magnitude_y)
        return loss

class EdgeAwareSmoothingLoss(nn.Module):
    def __init__(self: Self, gamma: float = 0.1) -> None:
        super(EdgeAwareSmoothingLoss, self).__init__()
        self.gamma = gamma

    def forward(self: Self, img: Tensor, guide_img: Tensor) -> Tensor:
        if img.shape[1] > 1: 
            img_gray = torch.mean(img, dim=1, keepdim=True)
        else:
            img_gray = img

        if guide_img.shape[1] > 1:
            guide_img_gray = torch.mean(guide_img, dim=1, keepdim=True)
        else:
            guide_img_gray = guide_img

        dx_img = img_gray[:, :, :, 1:] - img_gray[:, :, :, :-1]
        dy_img = img_gray[:, :, 1:, :] - img_gray[:, :, :-1, :]

        dx_guide = guide_img_gray[:, :, :, 1:] - guide_img_gray[:, :, :, :-1]
        dy_guide = guide_img_gray[:, :, 1:, :] - guide_img_gray[:, :, :-1, :]

        weights_x = torch.exp(-torch.abs(dx_guide) / self.gamma)
        weights_y = torch.exp(-torch.abs(dy_guide) / self.gamma)

        loss_x = torch.mean(weights_x * torch.abs(dx_img))
        loss_y = torch.mean(weights_y * torch.abs(dy_img))

        return loss_x + loss_y

if __name__ == "__main__":
    x_in_low = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_in_enh = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_gt = torch.rand(1, 3, 399, 499)  # GT low-light

    curve_1 = torch.linspace(0, 1, 255).unsqueeze(0)
    curve_2 = gamma_curve(curve_1, 1.0)
    curve_3 = s_curve(curve_2, alpha=1.0)


