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
        self,
        alpha=0.2,
        beta=0.6,
        low_thresh=0.2,
        high_thresh=0.6,
        lambda1=1.0,
        lambda2=1.0,
        lambda3=0.1,
    ):
        """
        自定义损失函数，用于控制曲线的增强和压缩。
        :param alpha: 控制暗部区域的提升力度 (>1 会增强低光)
        :param beta: 控制高光区域的抑制力度 (<1 会压缩高光)
        :param low_thresh: 低光阈值，用于暗部区域分段控制 (默认 0.3)
        :param high_thresh: 高光阈值，用于高光区域分段控制 (默认 0.7)
        :param lambda1: 暗部增强损失的权重
        :param lambda2: 高光抑制损失的权重
        :param lambda3: 平滑损失的权重
        """
        super(AdaptiveCurveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3

    def forward(self, output):
        low_mask = (output < self.low_thresh).float()
        low_light_loss = torch.mean(low_mask * torch.abs(output - self.alpha))

        high_mask = (output > self.high_thresh).float()
        high_light_loss = torch.mean(high_mask * torch.abs(output - self.beta))

        smooth_loss = torch.mean((output[:, 1:] - output[:, :-1]) ** 2)

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
        self.mean_val = mean_val

    def forward(self: Self, x: Tensor) -> Tensor:
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)

        d = torch.mean(
            torch.pow(mean - torch.FloatTensor([self.mean_val]).to(x.device), 2)
        )
        return d


# Spatial Loss, control the spatial consistency of the generated image
class SpatialLoss(nn.Module):
    def __init__(self: Self) -> None:
        super(SpatialLoss, self).__init__()

        kernel_left = (
            torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_right = (
            torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_up = (
            torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        kernel_down = (
            torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]])
            .cuda()
            .unsqueeze(0)
            .unsqueeze(0)
        )
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self: Self, org: Tensor, enhance: Tensor, contrast: int = 8):
        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        def p(pool: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
            org_letf = F.conv2d(pool, self.weight_left.to(pool.device), padding=1)
            org_right = F.conv2d(pool, self.weight_right.to(pool.device), padding=1)
            org_up = F.conv2d(pool, self.weight_up.to(pool.device), padding=1)
            org_down = F.conv2d(pool, self.weight_down.to(pool.device), padding=1)

            return org_letf, org_right, org_up, org_down

        D_org_letf, D_org_right, D_org_up, D_org_down = p(org_pool)
        D_enhance_letf, D_enhance_right, D_enhance_up, D_enhance_down = p(enhance_pool)

        D_left = torch.pow(D_org_letf * contrast - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right * contrast - D_enhance_right, 2)
        D_up = torch.pow(D_org_up * contrast - D_enhance_up, 2)
        D_down = torch.pow(D_org_down * contrast - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return torch.mean(E)


class LaplacianLoss(nn.Module):
    def __init__(self: Self) -> None:
        super(LaplacianLoss, self).__init__()
        self.kernel = (
            torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32)
            .view(1, 1, 3, 3)
            .cuda()
        )

    def forward(self: Self, x: Tensor) -> Tensor:
        laplacian = F.conv2d(x, self.kernel, padding=1)
        return torch.mean(torch.abs(laplacian))


if __name__ == "__main__":
    x_in_low = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_in_enh = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_gt = torch.rand(1, 3, 399, 499)  # GT low-light

    curve_1 = torch.linspace(0, 1, 255).unsqueeze(0)
    curve_2 = gamma_curve(curve_1, 1.0)
    curve_3 = s_curve(curve_2, alpha=1.0, beta=1.0)
