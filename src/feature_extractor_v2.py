"""

src/feature_extractor_v2.py -- 35-D Geometry & Profile Harvester

 Pranaya Shrestha
Project:  PS-1 Class-Agnostic Primitive Analysis

Features Extracted (Exact 35-Dimensional Profile):
  1. Inner-Mask Depth Statistics (3x3 morphological erosion).
  2. Depth Percentiles & Inter-Quantiles (p10, p25, p50, p75, p90, IQR75_25, IQR90_10).
  3. Spatial Depth Asymmetries (D_LR, D_TB).
  4. 5-Slice Vertical Width & Depth Profiles (Taper Ratios: W1/W3, W5/W3, W1/W5).
  5. 3D Surface Normal Curvatures & Absolute Statistics (std_Nx, std_Ny, std_Nz, mean_Nz, mean_abs_Nx, mean_abs_Ny, local_normal_variation, planar_surface_ratio, curved_surface_ratio).
  6. Silhouette & Contour Descriptors (Extent, Solidity, Convexity, Eccentricity, Equivalent Diameter, 2 Log Hu Moments, Aspect Ratio).
  7. Composite Interaction Ratios (curvature_thickness_ratio, box_compactness_score, smoothness_texture_ratio).
  8. Dynamic Depth Reliability Score (R_d in [0.1, 1.0]).

"""

import os
import cv2
import numpy as np
import scipy.stats as stats
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
_CANDIDATE_DEPTH_PATHS = [
    _SRC_DIR.parent / "models" / "depth_anything_v2",
    _SRC_DIR.parent.parent / "retail_geometry_project" / "models" / "depth_anything_v2",
    Path("./models/depth_anything_v2").resolve(),
]
LOCAL_DEPTH_PATH = next((str(p) for p in _CANDIDATE_DEPTH_PATHS if p.is_dir()), "depth-anything/Depth-Anything-V2-Small-hf")
HF_DEPTH_ID      = "depth-anything/Depth-Anything-V2-Small-hf"

# Ordered list of all 35 Geometric Feature Names
GEOMETRIC_FEATURE_NAMES_V2 = [
    # 1. Surface Normal Dynamics & Curvatures (9)
    "std_Nx", "std_Ny", "std_Nz",
    "mean_Nz", "mean_abs_Nx", "mean_abs_Ny",
    "local_normal_variation", "planar_surface_ratio", "curved_surface_ratio",

    # 2. Depth Distribution & Percentiles (10)
    "depth_std", "depth_range", "depth_iqr_75_25", "depth_iqr_90_10",
    "depth_p10", "depth_p25", "depth_p50", "depth_p75", "depth_p90",
    "depth_skewness",

    # 3. Asymmetry & Slices (5)
    "depth_asymmetry_lr", "depth_asymmetry_tb",
    "slice_width_ratio_top_mid", "slice_width_ratio_bot_mid", "slice_width_ratio_top_bot",

    # 4. Silhouette, Moments & Contour (8)
    "aspect_ratio", "extent", "solidity", "convexity",
    "eccentricity", "equivalent_diameter",
    "hu_moment_1", "hu_moment_2",

    # 5. Composite Interaction Ratios (3)
    "curvature_thickness_ratio", "box_compactness_score", "smoothness_texture_ratio"
]


class GeometricFeatureExtractorV2:
    """
    Extracts 35-dimensional mask-aware geometric features from RGB crops,
    estimated metric depth maps, and object masks.
    """
    def __init__(self, device=None, depth_source=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if depth_source is None:
            depth_source = LOCAL_DEPTH_PATH if os.path.isdir(LOCAL_DEPTH_PATH) else HF_DEPTH_ID

        self.depth_source = depth_source
        self.image_processor = None
        self.depth_model = None

    def _lazy_load_depth_model(self):
        if self.depth_model is None:
            self.image_processor = AutoImageProcessor.from_pretrained(self.depth_source)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_source).to(self.device)
            self.depth_model.eval()

    def predict_metric_depth(self, rgb_crop):
        """Generates raw monocular depth map via Depth Anything V2."""
        self._lazy_load_depth_model()
        h, w = rgb_crop.shape[:2]
        if h < 4 or w < 4:
            # Pad tiny crops to minimum 8x8
            rgb_crop = cv2.resize(rgb_crop, (max(8, w), max(8, h)), interpolation=cv2.INTER_LINEAR)
            orig_h, orig_w = h, w
            h, w = rgb_crop.shape[:2]
        else:
            orig_h, orig_w = h, w

        pil_img = Image.fromarray(rgb_crop).convert("RGB")
        inputs = self.image_processor(images=pil_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.depth_model(**inputs)
        depth_pred = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1), size=(orig_h, orig_w),
            mode="bicubic", align_corners=False
        )
        return depth_pred[0, 0].cpu().numpy().astype(np.float32)

    def compute_surface_normals(self, depth_norm):
        """Computes unit 3D surface normal vector components (Nx, Ny, Nz)."""
        dz_dx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
        Nx = -dz_dx
        Ny = -dz_dy
        Nz = np.ones_like(depth_norm, dtype=np.float32)
        norm_len = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-6
        return Nx / norm_len, Ny / norm_len, Nz / norm_len

    def compute_local_normal_variation(self, Nx, Ny, Nz, mask):
        """Computes local normal curvature via 3x3 Laplacian filtering."""
        lap_x = np.abs(cv2.Laplacian(Nx, cv2.CV_32F, ksize=3))
        lap_y = np.abs(cv2.Laplacian(Ny, cv2.CV_32F, ksize=3))
        lap_z = np.abs(cv2.Laplacian(Nz, cv2.CV_32F, ksize=3))
        lap_mag = np.sqrt(lap_x**2 + lap_y**2 + lap_z**2)
        return float(np.mean(lap_mag[mask])) if np.sum(mask) > 0 else 0.0

    def compute_silhouette_descriptors(self, mask):
        """Extracts shape contour metrics, convexity, eccentricity, and Hu moments."""
        h, w = mask.shape[:2]
        n_mask = int(np.sum(mask))
        if n_mask < 10:
            return {
                "extent": 0.0, "solidity": 0.0, "convexity": 0.0,
                "eccentricity": 0.0, "equivalent_diameter": 0.0,
                "hu_moment_1": 0.0, "hu_moment_2": 0.0
            }

        mask_u8 = (mask.astype(np.uint8) * 255)
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return {
                "extent": 0.0, "solidity": 0.0, "convexity": 0.0,
                "eccentricity": 0.0, "equivalent_diameter": 0.0,
                "hu_moment_1": 0.0, "hu_moment_2": 0.0
            }

        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        peri = float(cv2.arcLength(cnt, True))
        extent = float(n_mask / (w * h + 1e-5))

        # Convex Hull & Convexity
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        hull_peri = float(cv2.arcLength(hull, True))
        solidity = float(area / (hull_area + 1e-5))
        convexity = float(hull_peri / (peri + 1e-5))

        # Equivalent Diameter normalized by max bounding dimension
        eq_diam = float(np.sqrt(4.0 * area / (np.pi + 1e-5)) / float(max(h, w)))

        # Eccentricity
        eccentricity = 0.0
        if len(cnt) >= 5:
            try:
                (cx, cy), (ma, MA), angle = cv2.fitEllipse(cnt)
                a = max(ma, MA) / 2.0
                b = min(ma, MA) / 2.0
                if a > 0:
                    eccentricity = float(np.sqrt(max(0.0, 1.0 - (b / a)**2)))
            except Exception:
                eccentricity = 0.0

        # Hu Moments
        moments = cv2.moments(cnt)
        hu = cv2.HuMoments(moments).flatten()
        hu_log = [-np.sign(val) * np.log10(np.abs(val) + 1e-10) for val in hu]

        return {
            "extent": extent,
            "solidity": solidity,
            "convexity": convexity,
            "eccentricity": eccentricity,
            "equivalent_diameter": eq_diam,
            "hu_moment_1": float(hu_log[0]),
            "hu_moment_2": float(hu_log[1])
        }

    def compute_5slice_profiles(self, mask, depth_norm):
        """Divides object mask into 5 vertical slices and computes width taper ratios."""
        h, w = mask.shape[:2]
        slice_h = max(1, h // 5)
        widths = []
        slice_depth_means = []

        for s in range(5):
            y_start = s * slice_h
            y_end = (s + 1) * slice_h if s < 4 else h
            slice_mask = mask[y_start:y_end, :]
            slice_pixels = np.sum(slice_mask)
            if slice_pixels > 0:
                cols_with_mask = np.where(np.sum(slice_mask, axis=0) > 0)[0]
                slice_w = (cols_with_mask[-1] - cols_with_mask[0] + 1) if len(cols_with_mask) > 0 else 0
                widths.append(float(slice_w))
                slice_depth_means.append(float(np.mean(depth_norm[y_start:y_end, :][slice_mask])))
            else:
                widths.append(0.0)
                slice_depth_means.append(0.0)

        mid_w = widths[2] + 1e-5
        bot_w = widths[4] + 1e-5
        top_w_ratio = float(widths[0] / mid_w)
        bot_w_ratio = float(widths[4] / mid_w)
        top_bot_ratio = float(widths[0] / bot_w)

        return top_w_ratio, bot_w_ratio, top_bot_ratio, slice_depth_means

    def extract_features_and_reliability(self, rgb_crop, raw_depth=None, raw_mask=None):
        """
        Extracts full 35-dimensional geometric feature dictionary along with
        adaptive depth reliability score Rd in [0.1, 1.0].
        """
        h, w = rgb_crop.shape[:2]

        # 1. Depth Map Estimation & Guided Bilateral Filtering
        if raw_depth is None:
            raw_depth = self.predict_metric_depth(rgb_crop)
        elif raw_depth.shape[:2] != (h, w):
            raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        guide_bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
        depth_u8 = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filter_rad = max(1, min(8, min(h, w) // 4))
        clean_depth = cv2.ximgproc.guidedFilter(guide=guide_bgr, src=depth_u8, radius=filter_rad, eps=50).astype(np.float32)

        # 2. Object Mask Handling & 3x3 Morphological Erosion (Eliminating boundary bleed)
        if raw_mask is None:
            raw_mask = np.zeros((h, w), dtype=bool)
            raw_mask[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)] = True

        mask_u8 = raw_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
        eroded_mask = (eroded_mask_u8 > 0)

        if np.sum(eroded_mask) < 10:
            eroded_mask = raw_mask.copy()
        if np.sum(eroded_mask) < 1:
            eroded_mask = np.ones((h, w), dtype=bool)

        n_raw = max(1, int(np.sum(raw_mask)))
        n_eroded = int(np.sum(eroded_mask))
        erosion_ratio = float(n_eroded / (n_raw + 1e-5))

        # 3. Object-Normalized Depth (Zn)
        depth_in_mask = clean_depth[eroded_mask]
        if len(depth_in_mask) == 0:
            depth_in_mask = clean_depth.flatten()

        d_min, d_max = float(np.min(depth_in_mask)), float(np.max(depth_in_mask))
        depth_norm = np.zeros_like(clean_depth)
        if d_max > d_min:
            depth_norm[eroded_mask] = (clean_depth[eroded_mask] - d_min) / (d_max - d_min + 1e-5)

        depth_masked = depth_norm[eroded_mask]
        if len(depth_masked) == 0:
            depth_masked = depth_norm.flatten()

        # 4. Depth Distribution & Percentiles
        p10, p25, p50, p75, p90 = np.percentile(depth_masked, [10, 25, 50, 75, 90])
        depth_std_val = float(np.std(depth_masked))
        depth_range_val = float(d_max - d_min)
        depth_iqr_75_25 = float(p75 - p25)
        depth_iqr_90_10 = float(p90 - p10)
        depth_skew = float(stats.skew(depth_masked)) if len(depth_masked) > 2 else 0.0

        # 5. Left-Right & Top-Bottom Asymmetry
        mid_x = w // 2
        mid_y = h // 2
        mask_left = eroded_mask.copy()
        mask_left[:, mid_x:] = False
        mask_right = eroded_mask.copy()
        mask_right[:, :mid_x] = False

        mask_top = eroded_mask.copy()
        mask_top[mid_y:, :] = False
        mask_bot = eroded_mask.copy()
        mask_bot[:mid_y, :] = False

        mean_l = float(np.mean(depth_norm[mask_left])) if np.sum(mask_left) > 0 else 0.0
        mean_r = float(np.mean(depth_norm[mask_right])) if np.sum(mask_right) > 0 else 0.0
        mean_t = float(np.mean(depth_norm[mask_top])) if np.sum(mask_top) > 0 else 0.0
        mean_b = float(np.mean(depth_norm[mask_bot])) if np.sum(mask_bot) > 0 else 0.0

        depth_asym_lr = float(abs(mean_l - mean_r))
        depth_asym_tb = float(abs(mean_t - mean_b))

        # 6. 5-Slice Width & Depth Taper Profiles
        top_w_ratio, bot_w_ratio, top_bot_ratio, slice_depth_means = self.compute_5slice_profiles(eroded_mask, depth_norm)

        # 7. 3D Surface Normals & Curvature
        Nx, Ny, Nz = self.compute_surface_normals(depth_norm)
        std_Nx = float(np.std(Nx[eroded_mask]))
        std_Ny = float(np.std(Ny[eroded_mask]))
        std_Nz = float(np.std(Nz[eroded_mask]))
        mean_Nz = float(np.mean(Nz[eroded_mask]))
        mean_abs_Nx = float(np.mean(np.abs(Nx[eroded_mask])))
        mean_abs_Ny = float(np.mean(np.abs(Ny[eroded_mask])))

        # Local Normal Laplacian Curvature Variation
        local_normal_var = self.compute_local_normal_variation(Nx, Ny, Nz, eroded_mask)

        # Planar vs. Curved Surface Ratios
        norm_xy_mag = np.sqrt(Nx**2 + Ny**2)
        planar_pixels = np.sum((norm_xy_mag[eroded_mask] < 0.15))
        curved_pixels = np.sum((norm_xy_mag[eroded_mask] >= 0.15))
        planar_ratio = float(planar_pixels / (n_eroded + 1e-5))
        curved_ratio = float(curved_pixels / (n_eroded + 1e-5))

        # 8. Silhouette & Contour Descriptors
        sil_desc = self.compute_silhouette_descriptors(raw_mask)
        aspect_ratio_val = float(w / float(h + 1e-5))

        # 9. Composite Interaction Ratios
        curvature_thickness_ratio = float(std_Nx / (depth_std_val + 1e-5))
        box_compactness_score = float(sil_desc["solidity"] * aspect_ratio_val)
        smoothness_texture_ratio = float(mean_abs_Nx / (depth_iqr_75_25 + 1e-5))

        # 10. Dynamic Depth Reliability Score Rd in [0.1, 1.0]
        grad_smoothness_penalty = min(1.0, local_normal_var * 2.5)
        valid_pixel_score = min(1.0, n_eroded / 100.0)
        depth_reliability = float(np.clip(
            0.4 * erosion_ratio + 0.3 * (1.0 - grad_smoothness_penalty) + 0.3 * valid_pixel_score,
            0.10, 1.0
        ))

        # Assemble Exact 35-Feature Dictionary
        feature_dict = {
            # 1. Surface Normal Dynamics & Curvatures (9)
            "std_Nx": std_Nx,
            "std_Ny": std_Ny,
            "std_Nz": std_Nz,
            "mean_Nz": mean_Nz,
            "mean_abs_Nx": mean_abs_Nx,
            "mean_abs_Ny": mean_abs_Ny,
            "local_normal_variation": local_normal_var,
            "planar_surface_ratio": planar_ratio,
            "curved_surface_ratio": curved_ratio,

            # 2. Depth Distribution & Percentiles (10)
            "depth_std": depth_std_val,
            "depth_range": depth_range_val,
            "depth_iqr_75_25": depth_iqr_75_25,
            "depth_iqr_90_10": depth_iqr_90_10,
            "depth_p10": float(p10),
            "depth_p25": float(p25),
            "depth_p50": float(p50),
            "depth_p75": float(p75),
            "depth_p90": float(p90),
            "depth_skewness": depth_skew,

            # 3. Asymmetry & Slices (5)
            "depth_asymmetry_lr": depth_asym_lr,
            "depth_asymmetry_tb": depth_asym_tb,
            "slice_width_ratio_top_mid": top_w_ratio,
            "slice_width_ratio_bot_mid": bot_w_ratio,
            "slice_width_ratio_top_bot": top_bot_ratio,

            # 4. Silhouette & Moments (8)
            "aspect_ratio": aspect_ratio_val,
            "extent": sil_desc["extent"],
            "solidity": sil_desc["solidity"],
            "convexity": sil_desc["convexity"],
            "eccentricity": sil_desc["eccentricity"],
            "equivalent_diameter": sil_desc["equivalent_diameter"],
            "hu_moment_1": sil_desc["hu_moment_1"],
            "hu_moment_2": sil_desc["hu_moment_2"],

            # 5. Composite Interaction Ratios (3)
            "curvature_thickness_ratio": curvature_thickness_ratio,
            "box_compactness_score": box_compactness_score,
            "smoothness_texture_ratio": smoothness_texture_ratio
        }

        # Convert to ordered 35-D feature vector
        feature_vector = np.array([feature_dict[k] for k in GEOMETRIC_FEATURE_NAMES_V2], dtype=np.float32)

        return feature_dict, feature_vector, depth_reliability, clean_depth, eroded_mask


_GLOBAL_EXTRACTOR_V2 = None

def extract_geometry_features_v2(rgb_crop, raw_depth=None, raw_mask=None):
    global _GLOBAL_EXTRACTOR_V2
    if _GLOBAL_EXTRACTOR_V2 is None:
        _GLOBAL_EXTRACTOR_V2 = GeometricFeatureExtractorV2()
    return _GLOBAL_EXTRACTOR_V2.extract_features_and_reliability(rgb_crop, raw_depth, raw_mask)
