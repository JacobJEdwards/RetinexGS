import math
from pathlib import Path

import numpy as np
import cv2
import os
import torch.nn.functional as F

import torch
from pycolmap import SceneManager

# Assume you have your rasterization function available, e.g., from a library
# from gsplat.cuda_rasterizer import rasterization # This is a placeholder, replace with your actual import
# Since the header is provided, let's assume it's in a module called `gsplat_renderer`
from typing import Optional, Tuple, Dict, Literal
from torch import Tensor

COLMAP_MODEL_PATH = Path("../new_images/cheng_video/model/colmap/dense/sparse")
OUTPUT_RENDER_DIR = Path("colmap_renders")
CHECKPOINT_PATH = Path("colmap_checkpoints")
RENDER_IMAGE_EXT = ".png"

manager = SceneManager(str(COLMAP_MODEL_PATH))
manager.load_cameras()
manager.load_images()
manager.load_points3D()

print(f"Loaded {len(manager.cameras)} cameras and {len(manager.images)} images from COLMAP.")

colmap_cameras = manager.cameras
colmap_images = manager.images

first_cam = next(iter(colmap_cameras.values()))
RENDER_WIDTH = first_cam.width
RENDER_HEIGHT = first_cam.height
print(f"Using render dimensions: {RENDER_WIDTH}x{RENDER_HEIGHT} from COLMAP camera.")

os.makedirs(OUTPUT_RENDER_DIR, exist_ok=True)

print(f"Starting rendering for {len(colmap_images)} views...")

image_ids_to_render = sorted(colmap_images.keys())

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

means, quats, scales, opacities, sh0, shN = [], [], [], [], [], []

# Correct the CHECKPOINT_PATH to point to an actual file, e.g., a .pt file
# For demonstration, let's assume `colmap_checkpoints/splat_model.pt` exists
# or create a dummy one if running standalone for testing.
# For example:
# CHECKPOINT_PATH = Path("colmap_checkpoints/splat_model.pt")
# if not CHECKPOINT_PATH.exists():
#     print(f"Creating a dummy checkpoint file at {CHECKPOINT_PATH} for demonstration.")
#     dummy_ckpt = {
#         "splats": {
#             "means": torch.randn(100, 3),
#             "quats": F.normalize(torch.randn(100, 4), p=2, dim=-1),
#             "scales": torch.randn(100, 3),
#             "opacities": torch.randn(100, 1),
#             "sh0": torch.randn(100, 3),
#             "shN": torch.randn(100, 45) # 15 * 3 for degree 3 SH
#         }
#     }
#     os.makedirs(CHECKPOINT_PATH.parent, exist_ok=True)
#     torch.save(dummy_ckpt, CHECKPOINT_PATH)

try:
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)["splats"]
    means.append(ckpt["means"])
    quats.append(F.normalize(ckpt["quats"], p=2, dim=-1))
    scales.append(torch.exp(ckpt["scales"]))
    opacities.append(torch.sigmoid(ckpt["opacities"]))
    sh0.append(ckpt["sh0"] if "sh0" in ckpt else torch.zeros_like(ckpt["means"])[:, :3])  # Ensure sh0 is present
    shN.append(ckpt["shN"] if "shN" in ckpt else torch.zeros_like(ckpt["means"])[:, :15 * 3])  # Ensure shN is present

    means = torch.cat(means).to(device)
    quats = torch.cat(quats).to(device)
    scales = torch.cat(scales).to(device)
    opacities = torch.cat(opacities).to(device)
    sh0 = torch.cat(sh0).to(device)
    shN = torch.cat(shN).to(device)
    colors = torch.cat([sh0, shN], dim=-2).to(device) # Ensure colors are on the correct device
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print("Number of Gaussians:", len(means))
except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {CHECKPOINT_PATH}. Please ensure the path is correct and the file exists.")
    print("For demonstration purposes, creating dummy Gaussian data.")
    num_gaussians = 10000
    means = torch.randn(num_gaussians, 3, device=device)
    quats = F.normalize(torch.randn(num_gaussians, 4, device=device), p=2, dim=-1)
    scales = torch.exp(torch.randn(num_gaussians, 3, device=device) * 0.5) # Example scales
    opacities = torch.sigmoid(torch.randn(num_gaussians, 1, device=device)) # Example opacities
    sh0 = torch.randn(num_gaussians, 3, device=device)
    shN = torch.randn(num_gaussians, 15 * 3, device=device) # Example for SH degree 3
    colors = torch.cat([sh0, shN], dim=-2).to(device)
    sh_degree = 3 # Assuming SH degree 3 for dummy data
    print("Number of Gaussians (dummy):", len(means))


for img_id in image_ids_to_render:
    img_data = colmap_images[img_id]
    camera_data = colmap_cameras[img_data.camera_id]

    image_filename = img_data.name
    output_filepath = os.path.join(OUTPUT_RENDER_DIR, os.path.splitext(image_filename)[0] + RENDER_IMAGE_EXT)

    R_wc = img_data.R()  # World to Camera Rotation (3x3 numpy array)
    t_wc = img_data.tvec.reshape(3, 1)  # World to Camera Translation (3x1 numpy array)

    # COLMAP's R and tvec define the transformation from world to camera (W2C).
    # The rasterization function expects `viewmats` as the world-to-camera matrix.
    # So, colmap_W2C_pose is directly your `viewmat`.
    colmap_W2C_pose = np.eye(4, dtype=np.float32)
    colmap_W2C_pose[:3, :3] = R_wc
    colmap_W2C_pose[:3, 3] = t_wc.flatten()

    # Convert to a torch tensor and add batch/camera dimension
    viewmat = torch.from_numpy(colmap_W2C_pose).float().to(device)
    viewmats_tensor = viewmat.unsqueeze(0) # [1, 4, 4] for a single camera

    # Construct the intrinsic camera matrix (K)
    # COLMAP camera models and their parameters:
    # SIMPLE_PINHOLE: f, cx, cy
    # PINHOLE: fx, fy, cx, cy
    # SIMPLE_RADIAL: f, cx, cy, k1
    # RADIAL: f, cx, cy, k1, k2
    # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
    # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

    K = np.eye(3, dtype=np.float32)
    camera_model = camera_data.model_name
    params = camera_data.params

    if camera_model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
    elif camera_model == "PINHOLE":
        fx, fy, cx, cy = params
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
    elif camera_model == "SIMPLE_RADIAL" or camera_model == "RADIAL":
        # For these models, the K matrix is the same as pinhole,
        # but distortion coefficients are also present.
        # The `rasterization` function can take radial_coeffs, etc.
        if camera_model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = params
            radial_coeffs_np = np.array([k1, 0, 0, 0, 0, 0], dtype=np.float32)
        else: # RADIAL
            f, cx, cy, k1, k2 = params
            radial_coeffs_np = np.array([k1, k2, 0, 0, 0, 0], dtype=np.float32)

        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = cx
        K[1, 2] = cy
        radial_coeffs_tensor = torch.from_numpy(radial_coeffs_np).float().to(device).unsqueeze(0)
    elif camera_model == "OPENCV" or camera_model == "FULL_OPENCV":
        # Full set of distortion parameters for OpenCV models
        fx, fy, cx, cy = params[0:4]
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        # Extract distortion coefficients
        k1, k2, p1, p2 = params[4:8]
        radial_coeffs_np = np.array([k1, k2, 0, 0, 0, 0], dtype=np.float32) # k1, k2 are radial
        tangential_coeffs_np = np.array([p1, p2], dtype=np.float32) # p1, p2 are tangential

        if camera_model == "FULL_OPENCV" and len(params) > 8:
            k3, k4, k5, k6 = params[8:12]
            # Adjust radial_coeffs to include k3-k6 if the rasterizer supports it
            radial_coeffs_np = np.array([k1, k2, k3, k4, k5, k6], dtype=np.float32)

        radial_coeffs_tensor = torch.from_numpy(radial_coeffs_np).float().to(device).unsqueeze(0)
        tangential_coeffs_tensor = torch.from_numpy(tangential_coeffs_np).float().to(device).unsqueeze(0)

    else:
        print(f"Warning: Unsupported camera model: {camera_model}. Proceeding without distortion parameters.")
        radial_coeffs_tensor = None
        tangential_coeffs_tensor = None
        # You might want to handle other camera models like FISHEYE if your rasterizer supports them.
        # For now, let's assume PINHOLE or RADIAL variants.

    Ks_tensor = torch.from_numpy(K).float().to(device).unsqueeze(0) # [1, 3, 3] for a single camera

    # Prepare other parameters for rasterization
    # Reshape means, quats, scales, opacities, colors to be [N, X]
    # The `rasterization` function expects [..., N, X] for batching.
    # Here, we have a single set of Gaussians, so they are already [N, X].
    # We pass them directly.

    # Call the rasterization function
    rendered_image, rendered_alpha, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities.squeeze(-1), # opacities should be [N] not [N, 1]
        colors=colors,
        viewmats=viewmats_tensor,
        Ks=Ks_tensor,
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
        sh_degree=sh_degree,
        packed=True, # Assuming packed mode is preferred
        tile_size=16, # Default
        backgrounds=torch.zeros((1, 3), device=device) if render_mode == "RGB" else None, # Black background
        render_mode="RGB", # Rendering an RGB image
        camera_model=camera_model.lower().replace("_", ""), # Convert COLMAP model name to the rasterizer's expected format (e.g., "pinhole", "simplepinhole")
        radial_coeffs=radial_coeffs_tensor if 'radial_coeffs_tensor' in locals() else None,
        tangential_coeffs=tangential_coeffs_tensor if 'tangential_coeffs_tensor' in locals() else None,
        # thin_prism_coeffs=None, # If applicable for your camera model
        # rolling_shutter=RollingShutterType.GLOBAL, # Assuming global shutter
        # viewmats_rs=None, # Only needed for rolling shutter
    )

    # Post-processing and saving the image
    rendered_image_np = (rendered_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    # The rendered image will be [H, W, 3] from the rasterizer, if not specified otherwise
    # OpenCV expects BGR, so convert if needed, or save as RGB directly if your image saving function supports it.
    cv2.imwrite(output_filepath, cv2.cvtColor(rendered_image_np, cv2.COLOR_RGB2BGR))
    print(f"Rendered image saved to {output_filepath}")

print("Rendering complete.")