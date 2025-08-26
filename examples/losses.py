import kornia
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import kornia.filters as KF


# Adaptive Curve Loss
class AdaptiveCurveLoss(nn.Module):
    lambda1: Tensor
    lambda2: Tensor
    lambda3: Tensor

    def __init__(
            self,
            alpha: float = 0.2,
            beta: float = 0.8,
            initial_low_thresh: float = 0.3,
            initial_high_thresh: float = 0.7,
            lambda1: float = 1.0,
            lambda2: float = 1.0,
            lambda3: float = 1.0,
            learn_lambdas: bool = False,
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

    def forward(self, output: Tensor) -> Tensor:
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
    def __init__(self, patch_size: int, mean_val: float = 0.5) -> None:
        super(ExposureLoss, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.register_buffer("mean_val_tensor", torch.tensor([mean_val]))

    def forward(self, x: Tensor) -> Tensor:
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        mean_val = self.mean_val_tensor

        d = torch.mean(torch.pow(mean - mean_val, 2))
        return d



class SpatialLoss(nn.Module):
    weight_left: Tensor
    weight_right: Tensor
    weight_up: Tensor
    weight_down: Tensor

    def __init__(self) -> \
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


    def forward(self, org: Tensor, enhance: Tensor, contrast: int = 8):
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

        current_contrast = contrast

        D_left = torch.pow(D_org_letf * current_contrast - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right * current_contrast - D_enhance_right, 2)
        D_up = torch.pow(D_org_up * current_contrast - D_enhance_up, 2)
        D_down = torch.pow(D_org_down * current_contrast - D_enhance_down, 2)
        E = D_left + D_right + D_up + D_down

        return E.mean()

class EdgeAwareSmoothingLoss(nn.Module):
    initial_gamma: Tensor

    def __init__(self, initial_gamma: float = 0.2) -> None:
        super(EdgeAwareSmoothingLoss, self).__init__()
        self.register_buffer("initial_gamma", torch.tensor(initial_gamma, dtype=torch.float32))

    def forward(self, img: Tensor, guide_img: Tensor) -> Tensor:
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

        effective_gamma = self.initial_gamma

        effective_gamma += 1e-8

        weights_x = torch.exp(-torch.abs(dx_guide) / effective_gamma)
        weights_y = torch.exp(-torch.abs(dy_guide) / effective_gamma)

        loss_x = torch.mean(weights_x * torch.abs(dx_img))
        loss_y = torch.mean(weights_y * torch.abs(dy_img))

        total_loss = loss_x + loss_y

        return total_loss

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

    def get_gradients(self, img1: Tensor, img2: Tensor) -> tuple[Tensor, Tensor, int]:
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

class WhitePreservationLoss(nn.Module):
    def __init__(self, luminance_threshold: float = 95.0, chroma_tolerance: float = 5.0, gain: float = 10.0):
        super(WhitePreservationLoss, self).__init__()
        self.register_buffer("luminance_threshold", torch.tensor(luminance_threshold))
        self.register_buffer("chroma_tolerance", torch.tensor(chroma_tolerance))
        self.register_buffer("gain", torch.tensor(gain))

    def forward(self, input_image: Tensor, reflectance_map: Tensor) -> Tensor:
        input_lab = kornia.color.rgb_to_lab(input_image.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

        L = input_lab[..., 0]
        a = input_lab[..., 1]
        b = input_lab[..., 2]

        luminance_threshold = self.luminance_threshold
        chroma_tolerance = self.chroma_tolerance
        gain = self.gain

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

class PerceptualColorLoss(nn.Module):
    def __init__(self):
        super(PerceptualColorLoss, self).__init__()

    @staticmethod
    def forward(img1_rgb: Tensor, img2_rgb: Tensor) -> Tensor:
        if img1_rgb.shape[1] != 3 or img2_rgb.shape[1] != 3:
            return torch.tensor(0.0, device=img1_rgb.device)

        img1_lab = kornia.color.rgb_to_lab(img1_rgb)
        img2_lab = kornia.color.rgb_to_lab(img2_rgb)

        img1_ab = img1_lab[:, 1:3, :, :]
        img2_ab = img2_lab[:, 1:3, :, :]

        return F.l1_loss(img1_ab, img2_ab)

if __name__ == "__main__":
    x_in_low = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_in_enh = torch.rand(1, 3, 399, 499)  # Pred normal-light
    x_gt = torch.rand(1, 3, 399, 499)  # GT low-light

    curve_1 = torch.linspace(0, 1, 255).unsqueeze(0)
    curve_2 = gamma_curve(curve_1, 1.0)
    curve_3 = s_curve(curve_2, alpha=1.0)