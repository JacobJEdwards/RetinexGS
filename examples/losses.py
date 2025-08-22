import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision import models

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
    lambda1: Tensor
    lambda2: Tensor
    lambda3: Tensor

    def __init__(
            self,
            alpha: float = 0.3,
            beta: float = 0.7,
            initial_low_thresh: float = 0.3,
            initial_high_thresh: float = 0.7,
            lambda1: float = 1.0,
            lambda2: float = 1.0,
            lambda3: float = 1.0,
            learn_lambdas: bool = False,
            learn_thresholds: bool = False,
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

        self.learn_thresholds = learn_thresholds
        if self.learn_thresholds:
            self.low_thresh = nn.Parameter(torch.tensor([torch.logit(torch.tensor(initial_low_thresh, dtype=torch.float32))]))
            self.high_thresh = nn.Parameter(torch.tensor([torch.logit(torch.tensor(initial_high_thresh, dtype=torch.float32))]))
        else:
            self.low_thresh = initial_low_thresh
            self.high_thresh = initial_high_thresh

        self.learn_lambdas = learn_lambdas
        if self.learn_lambdas:
            self.lambda1 = nn.Parameter(torch.tensor([lambda1], dtype=torch.float32))
            self.lambda2 = nn.Parameter(torch.tensor([lambda2], dtype=torch.float32))
            self.lambda3 = nn.Parameter(torch.tensor([lambda3], dtype=torch.float32))
        else:
            self.register_buffer("lambda1", torch.tensor([lambda1], dtype=torch.float32))
            self.register_buffer("lambda2", torch.tensor([lambda2], dtype=torch.float32))
            self.register_buffer("lambda3", torch.tensor([lambda3], dtype=torch.float32))

    def forward_with_maps(
            self,
            output: Tensor,
            alpha_map: Tensor,
            beta_map: Tensor,
    ) -> Tensor:
        if alpha_map.shape[2:] != output.shape[2:]:
            alpha_map = F.interpolate(alpha_map, size=output.shape[2:], mode='bilinear', align_corners=False)
        if beta_map.shape[2:] != output.shape[2:]:
            beta_map = F.interpolate(beta_map, size=output.shape[2:], mode='bilinear', align_corners=False)

        if self.learn_thresholds:
            low_thresh_val = torch.sigmoid(self.low_thresh)
            high_thresh_val = torch.sigmoid(self.high_thresh)
        else:
            low_thresh_val = self.low_thresh
            high_thresh_val = self.high_thresh

        low_mask = (output < low_thresh_val).float()
        low_light_loss = torch.mean(low_mask * torch.abs(output - alpha_map))

        high_mask = (output > high_thresh_val).float()
        high_light_loss = torch.mean(high_mask * torch.abs(output - beta_map))

        grad_y = (output[:, :, 1:, :] - output[:, :, :-1, :]) ** 2
        grad_x = (output[:, :, :, 1:] - output[:, :, :, :-1]) ** 2
        smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)

        lambda1_val = F.softplus(self.lambda1) if self.learn_lambdas else self.lambda1
        lambda2_val = F.softplus(self.lambda2) if self.learn_lambdas else self.lambda2
        lambda3_val = F.softplus(self.lambda3) if self.learn_lambdas else self.lambda3

        total_loss = (
                lambda1_val * low_light_loss
                + lambda2_val * high_light_loss
                + lambda3_val * smooth_loss
        )

        return total_loss.squeeze()

    def forward(self, output: Tensor, alpha_map: Tensor | None = None, beta_map: Tensor | None = None) -> Tensor:
        if alpha_map is not None and beta_map is not None:
            return self.forward_with_maps(output, alpha_map, beta_map)

        if self.learn_thresholds:
            low_thresh_val = torch.sigmoid(self.low_thresh)
            high_thresh_val = torch.sigmoid(self.high_thresh)
        else:
            low_thresh_val = self.low_thresh
            high_thresh_val = self.high_thresh

        low_mask = (output < low_thresh_val).float()
        low_light_loss = torch.mean(low_mask * torch.abs(output - self.alpha))

        high_mask = (output > high_thresh_val).float()
        high_light_loss = torch.mean(high_mask * torch.abs(output - self.beta))

        grad_y = (output[:, :, 1:, :] - output[:, :, :-1, :]) ** 2
        grad_x = (output[:, :, :, 1:] - output[:, :, :, :-1]) ** 2
        smooth_loss = torch.mean(grad_x) + torch.mean(grad_y)

        lambda1_val = F.softplus(self.lambda1) if self.learn_lambdas else self.lambda1
        lambda2_val = F.softplus(self.lambda2) if self.learn_lambdas else self.lambda2
        lambda3_val = F.softplus(self.lambda3) if self.learn_lambdas else self.lambda3

        total_loss = (
                lambda1_val * low_light_loss
                + lambda2_val * high_light_loss
                + lambda3_val * smooth_loss
        )
        return total_loss.squeeze()


class ColourConsistencyLoss(nn.Module):
    def __init__(self) -> None:
        super(ColourConsistencyLoss, self).__init__()

    @staticmethod
    def forward(x: Tensor) -> Tensor:
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
    def __init__(self, patch_size: int, mean_val: float = 0.4) -> None:
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.register_buffer("mean_val_tensor", torch.tensor([mean_val]))

    def forward(self, x: Tensor, exposure: Tensor | None = None) -> Tensor:
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        target = exposure if exposure is not None else self.mean_val_tensor
        d = torch.mean(torch.pow(mean - target, 2))
        return d


class SpatialLoss(nn.Module):
    weight_left: Tensor
    weight_right: Tensor
    weight_up: Tensor
    weight_down: Tensor

    def __init__(self, learn_contrast: bool = False, initial_contrast: float = 8.0, num_images: int | None = None) -> \
            None:
        super(SpatialLoss, self).__init__()
        kernel_left = torch.tensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_right = torch.tensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_up = torch.tensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_down = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)

        self.pool = nn.AvgPool2d(4)

        self.learn_contrast = learn_contrast
        if self.learn_contrast:
            self.learnable_contrast = nn.Embedding(num_images, 1) if num_images is not None else nn.Parameter(torch.tensor([initial_contrast], dtype=torch.float32))
            self.learnable_contrast.weight.data.fill_(initial_contrast)
        else:
            self.register_buffer("learnable_contrast", torch.tensor([initial_contrast], dtype=torch.float32))


    def forward_per_pixel(self, org: Tensor, enhance: Tensor, contrast: int = 8, image_id: Tensor | None = None):
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

        if self.learn_contrast:
            if image_id is None:
                current_contrast = F.softplus(self.learnable_contrast) if self.learn_contrast else (contrast if contrast is not None else self.learnable_contrast)
            else:
                current_contrast = F.softplus(self.learnable_contrast(image_id))
                current_contrast = current_contrast.view(-1, 1, 1, 1)
        else:
            current_contrast = contrast if contrast is not None else self.learnable_contrast


        D_left = torch.pow(D_org_letf * current_contrast - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right * current_contrast - D_enhance_right, 2)
        D_up = torch.pow(D_org_up * current_contrast - D_enhance_up, 2)
        D_down = torch.pow(D_org_down * current_contrast - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return E


class LaplacianLoss(nn.Module):
    kernel: Tensor

    def __init__(self) -> None:
        super(LaplacianLoss, self).__init__()
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel", kernel)

    def forward(self, x: Tensor) -> Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True)
        laplacian = F.conv2d(x_gray, self.kernel, padding=1)
        return torch.mean(torch.abs(laplacian))


class TotalVariationLoss(nn.Module):
    def __init__(self, weight: float = 1.0) -> None:
        super(TotalVariationLoss, self).__init__()
        self.weight = weight

    @staticmethod
    def _tensor_size(t: Tensor) -> int:
        return t.size()[1] * t.size()[2] * t.size()[3]

    def forward(self, x: Tensor) -> Tensor:
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

    def __init__(self) -> None:
        super(SmoothingLoss, self).__init__()
        kernel_left = torch.tensor(
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_right = torch.tensor(
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_up = torch.tensor(
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_down = torch.tensor(
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]], dtype=torch.float32
        ).view(1, 1, 3, 3)

        self.register_buffer("weight_left", kernel_left)
        self.register_buffer("weight_right", kernel_right)
        self.register_buffer("weight_up", kernel_up)
        self.register_buffer("weight_down", kernel_down)

        self.pool = nn.AvgPool2d(4)

    def forward(self, enhance: Tensor):
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

    def __init__(self) -> None:
        super(GradientLoss, self).__init__()
        kernel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        kernel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("kernel_x", kernel_x)
        self.register_buffer("kernel_y", kernel_y)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
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
    def __init__(self) -> None:
        super(FrequencyLoss, self).__init__()

    @staticmethod
    def forward(x: Tensor, y: Tensor) -> Tensor:
        x_gray = torch.mean(x, dim=1, keepdim=True).squeeze(1)
        y_gray = torch.mean(y, dim=1, keepdim=True).squeeze(1)

        fft_x = torch.fft.fftshift(torch.fft.rfft2(x_gray, norm="ortho"))
        fft_y = torch.fft.fftshift(torch.fft.rfft2(y_gray, norm="ortho"))

        magnitude_x = torch.log(torch.abs(fft_x) + 1e-8)
        magnitude_y = torch.log(torch.abs(fft_y) + 1e-8)

        loss = F.l1_loss(magnitude_x, magnitude_y)
        return loss

class IlluminationFrequencyLoss(nn.Module):
    def __init__(self):
        super(IlluminationFrequencyLoss, self).__init__()


    @staticmethod
    def forward(illumination_map: Tensor) -> Tensor:
        if illumination_map.shape[1] > 1:
            illum_gray = torch.mean(illumination_map, dim=1, keepdim=True).squeeze(1)
        else:
            illum_gray = illumination_map.squeeze(1)

        fft_illum = torch.fft.fftshift(torch.fft.rfft2(illum_gray, norm="ortho"))
        magnitude_illum = torch.log(torch.abs(fft_illum) + 1e-8)

        H, W = illum_gray.shape[1:]
        center_h, center_w = H // 2, W // 2

        freq_y = torch.linspace(-center_h, H - center_h - 1, H, device=illum_gray.device) if H % 2 == 0 else torch.linspace(-center_h, H - center_h, H, device=illum_gray.device)
        freq_x = torch.linspace(0, W // 2, W // 2 + 1, device=illum_gray.device)

        mesh_x, mesh_y = torch.meshgrid(freq_x, freq_y, indexing='xy')

        radius = torch.sqrt(mesh_x**2 + mesh_y**2)
        max_radius = torch.sqrt(torch.tensor((center_h**2 + (W//2)**2), dtype=torch.float32, device=illum_gray.device))
        normalized_radius = radius / max_radius

        freq_penalty_mask = normalized_radius

        loss = torch.mean(magnitude_illum * freq_penalty_mask)
        return loss

class EdgeAwareSmoothingLoss(nn.Module):
    initial_gamma: Tensor

    def __init__(self, initial_gamma: float = 0.2, learn_gamma: bool = True, num_images: int | None = None) -> None:
        super(EdgeAwareSmoothingLoss, self).__init__()
        self.learn_gamma = learn_gamma
        self.use_embedding = num_images is not None

        self.register_buffer("initial_gamma", torch.tensor(initial_gamma, dtype=torch.float32))

        if self.learn_gamma:
            if self.use_embedding:
                self.gamma_adjustments_embedding = nn.Embedding(num_images, 1)
                self.gamma_adjustments_embedding.weight.data.fill_(0.0)
            else:
                self.gamma_adjustment = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))



    def forward(self, img: Tensor, guide_img: Tensor, image_id: Tensor | None = None) -> Tensor:
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

        if self.learn_gamma:
            if self.use_embedding:
                if image_id is None:
                    raise ValueError("image_id must be provided when using embedding for gamma.")
                adjustment = self.gamma_adjustments_embedding(image_id).view(-1, 1, 1, 1)
            else:
                adjustment = self.gamma_adjustment
            effective_gamma = self.initial_gamma + 0.1 * torch.tanh(adjustment)
        else:
            effective_gamma = self.initial_gamma

        effective_gamma += 1e-8

        weights_x = torch.exp(-torch.abs(dx_guide) / effective_gamma)
        weights_y = torch.exp(-torch.abs(dy_guide) / effective_gamma)

        loss_x = torch.mean(weights_x * torch.abs(dx_img))
        loss_y = torch.mean(weights_y * torch.abs(dy_img))

        return loss_x + loss_y

class LocalExposureLoss(nn.Module):
    def __init__(self, patch_size: int, mean_val: float = 0.5, patch_grid_size: int | tuple[int, int] | None = None) -> None:
        super(LocalExposureLoss, self).__init__()
        self.patch_size = patch_size
        self.register_buffer("mean_val_tensor", torch.tensor([mean_val]))
        self.patch_grid_size = patch_grid_size

        if self.patch_grid_size is not None:
            if isinstance(self.patch_grid_size, int):
                self.patch_grid_size = (self.patch_grid_size, self.patch_grid_size)
            self.patch_pool = nn.AdaptiveAvgPool2d(self.patch_grid_size)
        else:
            self.global_pool = nn.AvgPool2d(patch_size)


    def forward(self, x: Tensor, mean_val: Tensor | None = None) -> Tensor:
        if x.shape[1] > 1:
            x = torch.mean(x, 1, keepdim=True)

        mean_val = self.mean_val_tensor if mean_val is None else mean_val
        if self.patch_grid_size is not None:
            mean_patches = self.patch_pool(x)
            d = torch.mean(torch.pow(mean_patches - mean_val, 2))
        else:
            mean = self.global_pool(x)
            d = torch.mean(torch.pow(mean - mean_val, 2))

        return d

class ExclusionLoss(nn.Module):

    def __init__(self, level=3):
        """
        Loss on the gradient. based on:
        http://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Single_Image_Reflection_CVPR_2018_paper.pdf
        """
        super(ExclusionLoss, self).__init__()
        self.level = level
        self.avg_pool = torch.nn.AvgPool2d(2, stride=2)
        self.sigmoid = nn.Sigmoid()

    def get_gradients(self, img1, img2):
        img2 = img2.to(img1.device)
        self.avg_pool = self.avg_pool.to(img1.device)
        self.sigmoid = self.sigmoid.to(img1.device)

        gradx_loss = []
        grady_loss = []

        n_channels = img1.shape[1]

        for l in range(self.level):
            if img1.shape[2:] != img2.shape[2:]:
                raise ValueError(
                    f"ExclusionLoss: Input images must have the same spatial dimensions. "
                    f"Got {img1.shape} and {img2.shape}"
                )

            gradx1, grady1 = self.compute_gradient(img1)
            gradx2, grady2 = self.compute_gradient(img2)

            alphax = 1.0
            alphay = 1.0

            gradx1_s = (self.sigmoid(gradx1) * 2) - 1
            grady1_s = (self.sigmoid(grady1) * 2) - 1
            gradx2_s = (self.sigmoid(gradx2 * alphax) * 2) - 1
            grady2_s = (self.sigmoid(grady2 * alphay) * 2) - 1

            gradx_loss += self._all_comb(gradx1_s, gradx2_s)
            grady_loss += self._all_comb(grady1_s, grady2_s)

            img1 = self.avg_pool(img1)
            img2 = self.avg_pool(img2)

        return gradx_loss, grady_loss, (n_channels**2)

    @staticmethod
    def _all_comb(grad1_s, grad2_s):
        B, C1, H, W = grad1_s.shape
        B, C2, H, W = grad2_s.shape

        v = []
        for i in range(C2):
            for j in range(C1):
                v.append(torch.mean((grad1_s[:, j, :, :] ** 2) * (grad2_s[:, i, :, :] ** 2)) ** 0.25)
        return v

    def forward(self, img1, img2):
        gradx_loss, grady_loss, n_combinations = self.get_gradients(img1, img2)

        if not n_combinations > 0:
            return torch.tensor(0.0, device=img1.device)

        loss_gradxy = sum(gradx_loss) / (self.level * n_combinations) + sum(grady_loss) / (self.level * n_combinations)

        return loss_gradxy / 2.0

    @staticmethod
    def compute_gradient(img: Tensor) -> tuple[Tensor, Tensor]:
        gradx = img[:, :, 1:, :] - img[:, :, :-1, :]
        grady = img[:, :, :, 1:] - img[:, :, :, :-1]
        return gradx, grady

class PatchConsistencyLoss(nn.Module):
    """
    Enforces that the rendered reflectance is consistent across different views.
    It works by warping the reflectance map from a source view to a target view
    and computing a photometric loss.
    """
    def __init__(self):
        super(PatchConsistencyLoss, self).__init__()

    def forward(self, reflectance_maps: Tensor, depth_maps: Tensor,
                camtoworlds: Tensor, Ks: Tensor) -> Tensor:

        if reflectance_maps.shape[0] <= 1:
            return torch.tensor(0.0, device=reflectance_maps.device)

        target_reflectance = reflectance_maps[0:1] # (1, C, H, W)
        source_reflectance = reflectance_maps[1:2] # (1, C, H, W)

        target_depth = depth_maps[0:1] # (1, 1, H, W)

        T_target_to_world = camtoworlds[0:1] # (1, 4, 4)
        T_source_to_world = camtoworlds[1:2] # (1, 4, 4)
        K_target = Ks[0:1] # (1, 3, 3)
        K_source = Ks[1:2] # (1, 3, 3)

        T_world_to_target = torch.inverse(T_target_to_world)
        T_world_to_source = torch.inverse(T_source_to_world)

        T_target_to_source = T_world_to_source @ T_target_to_world

        try:
            warped_source_reflectance = kornia.geometry.warp_frame_depth(
                image_src=source_reflectance,
                depth_dst=target_depth,
                src_trans_dst=T_target_to_source,
                camera_matrix=K_source,
            )

            valid_mask = (target_depth > 0).float()

            photometric_loss = F.l1_loss(
                target_reflectance * valid_mask,
                warped_source_reflectance * valid_mask
            )
        except Exception as e:
            print(f"Kornia warping failed with error: {e}. Skipping patch consistency loss for this batch.")
            return torch.tensor(0.0, device=reflectance_maps.device)

        return photometric_loss

def white_preservation_loss(
        input_image: Tensor,
        reflectance_map: Tensor,
        luminance_threshold: float = 95.0,
        chroma_tolerance: float = 5.0,
        gain: float = 10.0
) -> Tensor:
    input_lab = kornia.color.rgb_to_lab(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    L = input_lab[..., 0]
    a = input_lab[..., 1]
    b = input_lab[..., 2]

    luminance_mask = torch.sigmoid((L - luminance_threshold) * gain)
    chroma_mask = torch.exp(-(a.pow(2) + b.pow(2)) / (2 * chroma_tolerance**2))

    soft_white_mask = luminance_mask * chroma_mask
    soft_white_mask = soft_white_mask.unsqueeze(-1)

    loss = torch.mean(torch.abs(reflectance_map - input_image) * soft_white_mask)

    return loss

def interp(x: Tensor, xp: Tensor, fp: Tensor) -> Tensor:
    right_indices = torch.searchsorted(xp, x, right=True)

    right_indices = torch.clamp(right_indices, 1, len(xp) - 1)
    left_indices = right_indices - 1

    xp_left, xp_right = xp[left_indices], xp[right_indices]
    fp_left, fp_right = fp[left_indices], fp[right_indices]

    t = (x - xp_left) / (xp_right - xp_left + 1e-8)

    return fp_left + t * (fp_right - fp_left)

class HistogramLoss(nn.Module):
    def __init__(self):
        super(HistogramLoss, self).__init__()

    @staticmethod
    def forward(reflectance_map: Tensor, target_dist: Tensor) -> Tensor:
        reflectance_flat = reflectance_map.mean(dim=1, keepdim=True).view(-1)

        reflectance_sorted, _ = torch.sort(reflectance_flat)
        reflectance_cdf = torch.linspace(0.0, 1.0, steps=len(reflectance_sorted), device=reflectance_map.device)

        target_cdf = torch.cumsum(target_dist, dim=0)

        target_quantiles = interp(
            reflectance_cdf,
            target_cdf,
            torch.linspace(0.0, 1.0, steps=len(target_dist), device=reflectance_map.device)
        )

        loss = torch.abs(reflectance_sorted - target_quantiles).mean()
        return loss


if __name__ == "__main__":
    x_in_low = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_in_enh = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_gt = torch.rand(1, 3, 399, 499)  # GT low-light

    curve_1 = torch.linspace(0, 1, 255).unsqueeze(0)
    curve_2 = gamma_curve(curve_1, 1.0)
    curve_3 = s_curve(curve_2, alpha=1.0)