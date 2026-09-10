"""
========================================================================================
src/feature_extractor_v2.py — 35-Dimensional Mask-Aware Geometric Feature Harvester
========================================================================================
Engineer: Pranaya Shrestha (Lead Engineer — Primitive Geometry Classifier & Sim-to-Real)
Project:  PS-1 Class-Agnostic Geometric Primitive Analysis

----------------------------------------------------------------------------------------
THEORY & MATHEMATICAL FOUNDATIONS: 3D GEOMETRY FROM MONOCULAR DEPTH
----------------------------------------------------------------------------------------
[Basic Concept: Why Not Just Use Color Images?]:
  Deep learning models (like ResNet or Vision Transformers) trained only on standard RGB
  pixels often rely on superficial visual shortcuts: brand logos, printed text, bright packaging
  colors, or barcode textures. When a new, unseen product appears on a supermarket shelf,
  an RGB-only model easily fails.
  However, physical objects obey strict three-dimensional geometric laws:
    - A beverage can is always a cylinder, whether it is Coca-Cola, beer, or tomato paste.
    - A cereal box is always a cuboid with flat perpendicular planes.
    - A chocolate bar is a flat rectangular slab.
    - A potato chip bag is an irregular, non-rigid deformable pouch.
  By extracting 3D physical and morphological features directly from metric depth maps, our
  classifier becomes truly class-agnostic and robust to retail distribution shifts.

[Monocular Depth & The Boundary Bleed Problem]:
  We use the state-of-the-art foundation model Depth Anything V2 (Small-hf) to estimate
  metric relative depth from RGB crops.
  However, monocular depth networks suffer from "Boundary Bleed" (depth bleeding): at the
  silhouette boundary between the object and the background shelf, depth values are averaged,
  creating artificial slope artifacts.
  To eliminate this, we apply two defensive preprocessing steps:
    1. Edge-Preserving Guided Bilateral Filtering (`cv2.ximgproc.guidedFilter`): Uses the original
       high-resolution RGB image as a spatial structural guide to sharpen depth boundaries.
    2. Morphological Mask Erosion: A 3x3 structuring element erodes 1-2 pixels inward from
       the object mask boundary, guaranteeing that all geometric statistics are computed
       strictly on the internal physical surface of the product.

----------------------------------------------------------------------------------------
EXACT 35-DIMENSIONAL FEATURE TAXONOMY BREAKDOWN:
----------------------------------------------------------------------------------------
Group 1: Surface Normal Dynamics & Curvatures (9 Features)
  - Surface normal vector: N = (-dZ/dx, -dZ/dy, 1) / ||N||
  - `std_Nx`, `std_Ny`, `std_Nz`: Variance of normal orientations. Cylinders exhibit high
    curvature variance along their curved axis and near-zero variance along their straight axis.
    Cuboids exhibit low variance on flat planes with sharp gradient spikes at edges.
  - `mean_Nz`: Direct alignment with the camera optical axis. Flat slabs facing the camera
    exhibit mean_Nz close to 1.0.
  - `mean_abs_Nx`, `mean_abs_Ny`: Mean absolute surface slope in horizontal and vertical directions.
  - `local_normal_variation`: Computed via a 3x3 Laplacian filter on normal channels, measuring
    high-frequency surface roughness and micro-facets.
  - `planar_surface_ratio`: Fraction of mask pixels where ||(Nx, Ny)|| < 0.15 (planar facets).
  - `curved_surface_ratio`: Fraction of mask pixels where ||(Nx, Ny)|| >= 0.15 (curved facets).

Group 2: Depth Distribution & Quantile Spread (10 Features)
  - `depth_std`, `depth_range`: Overall front-to-back depth span.
  - `depth_iqr_75_25`, `depth_iqr_90_10`: Robust inter-quantile ranges insensitive to outlier pixels.
  - `depth_p10`, `depth_p25`, `depth_p50`, `depth_p75`, `depth_p90`: Depth percentiles capturing
    the cumulative volumetric profile of the packaging.
  - `depth_skewness`: Fisher-Pearson coefficient of skewness. Symmetric objects (cans, boxes) have
    near-zero skewness; pouches with bulging centers have pronounced positive or negative skew.

Group 3: Spatial Asymmetry & 5-Slice Width Profiles (5 Features)
  - `depth_asymmetry_lr`: |mean(Depth_Left) - mean(Depth_Right)|. Measures lateral tilt/warp.
  - `depth_asymmetry_tb`: |mean(Depth_Top) - mean(Depth_Bottom)|. Measures vertical pitch.
  - 5-Slice Vertical Width Taper Ratios: Object is sliced into 5 horizontal strips from top to bottom:
    `slice_width_ratio_top_mid` (W1/W3), `slice_width_ratio_bot_mid` (W5/W3), `slice_width_ratio_top_bot` (W1/W5).
    Bottles and jars have narrow tops (W1/W3 < 0.7); cylindrical cans maintain W1/W3 ≈ 1.0;
    deformable pouches vary widely.

Group 4: 2D Silhouette, Moments & Contour Geometry (8 Features)
  - `aspect_ratio`: Width / Height of bounding box.
  - `extent`: Area(Mask) / Area(Bounding Box).
  - `solidity`: Area(Mask) / Area(Convex Hull). Physical pouch guardrail: non-rigid bags have
    dented boundaries with lower solidity (< 0.85); rigid cans and boxes have solidity > 0.95.
  - `convexity`: Perimeter(Convex Hull) / Perimeter(Mask).
  - `eccentricity`: sqrt(1 - (b/a)^2) of fitted ellipse.
  - `equivalent_diameter`: Normalized circular diameter.
  - `hu_moment_1`, `hu_moment_2`: Log-transformed scale/rotation invariant Hu moments.

Group 5: Composite Interaction Ratios (3 Features)
  - `curvature_thickness_ratio`: std_Nx / (depth_std + 1e-5) (identifies thin curved vs thick planar objects).
  - `box_compactness_score`: solidity * aspect_ratio.
  - `smoothness_texture_ratio`: mean_abs_Nx / (depth_iqr_75_25 + 1e-5).

Group 6: Dynamic Depth Reliability Score R_d in [0.10, 1.0]
  - Dynamically weights geometric confidence based on mask erosion ratio, gradient smoothness,
    and valid pixel count. Used in adaptive downstream multi-modal fusion.
========================================================================================
"""

import os
import cv2
import numpy as np
import scipy.stats as stats
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from pathlib import Path

# --------------------------------------------------------------------------------------
# LOCAL MODEL PATH RESOLUTION & FALLBACK CONFIGURATION
# --------------------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent
_CANDIDATE_DEPTH_PATHS = [
    _SRC_DIR.parent / "models" / "depth_anything_v2",
    _SRC_DIR.parent.parent / "retail_geometry_project" / "models" / "depth_anything_v2",
    Path("./models/depth_anything_v2").resolve(),
]
# Prefer local pre-downloaded weights for offline execution, fallback to HuggingFace hub
LOCAL_DEPTH_PATH = next((str(p) for p in _CANDIDATE_DEPTH_PATHS if p.is_dir()), "depth-anything/Depth-Anything-V2-Small-hf")
HF_DEPTH_ID      = "depth-anything/Depth-Anything-V2-Small-hf"

# Canonical ordered list of all 35 geometric feature names
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
    State-of-the-art 35-Dimensional Mask-Aware Geometric Feature Harvester.

    Integrates Depth Anything V2 monocular depth estimation with edge-preserving guided
    filtering, morphological mask erosion, 3D surface normal computation, and contour
    morphometry to produce a comprehensive geometric signature for retail crops.
    """

    def __init__(self, device=None, depth_source=None):
        """
        Initializes the feature extractor with lazy model loading.

        Args:
            device: torch.device ('cuda' or 'cpu'). If None, automatically detects CUDA.
            depth_source: Path to local model directory or HuggingFace model repo ID.
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if depth_source is None:
            depth_source = LOCAL_DEPTH_PATH if os.path.isdir(LOCAL_DEPTH_PATH) else HF_DEPTH_ID

        self.depth_source = depth_source
        self.image_processor = None
        self.depth_model = None

    def _lazy_load_depth_model(self):
        """
        Loads the Depth Anything V2 weights into memory only when the first depth
        prediction is actually requested, minimizing startup latency and GPU VRAM footprint.
        """
        if self.depth_model is None:
            self.image_processor = AutoImageProcessor.from_pretrained(self.depth_source)
            self.depth_model = AutoModelForDepthEstimation.from_pretrained(self.depth_source).to(self.device)
            self.depth_model.eval()

    def predict_metric_depth(self, rgb_crop: np.ndarray) -> np.ndarray:
        """
        Runs Depth Anything V2 inference on a cropped RGB patch.

        Handles edge cases such as extremely small crops by padding to at least 8x8 pixels,
        performs bicubic interpolation back to the original crop resolution, and returns
        a single-channel float32 depth map.

        Args:
            rgb_crop: RGB uint8 numpy array of shape (H, W, 3).

        Returns:
            depth_pred: Float32 numpy array of shape (H, W) with relative depth values.
        """
        self._lazy_load_depth_model()
        h, w = rgb_crop.shape[:2]

        # Guardrail against tiny crops that could break convolutional stride operations
        if h < 4 or w < 4:
            rgb_crop = cv2.resize(rgb_crop, (max(8, w), max(8, h)), interpolation=cv2.INTER_LINEAR)
            orig_h, orig_w = h, w
            h, w = rgb_crop.shape[:2]
        else:
            orig_h, orig_w = h, w

        pil_img = Image.fromarray(rgb_crop).convert("RGB")
        inputs = self.image_processor(images=pil_img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.depth_model(**inputs)

        # Upsample estimated depth map back to exact original crop dimensions
        depth_pred = torch.nn.functional.interpolate(
            outputs.predicted_depth.unsqueeze(1),
            size=(orig_h, orig_w),
            mode="bicubic",
            align_corners=False
        )
        return depth_pred[0, 0].cpu().numpy().astype(np.float32)

    def compute_surface_normals(self, depth_norm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Computes the unit 3D surface normal vector field (Nx, Ny, Nz) from a depth map.

        Mathematical Formulation:
          Using Sobel gradient operators with a 3x3 kernel:
            dz_dx = Sobel(Z, dx=1, dy=0)
            dz_dy = Sobel(Z, dx=0, dy=1)
          The tangent vectors on the surface are:
            Tx = (1, 0, dz_dx)
            Ty = (0, 1, dz_dy)
          Their cross product yields the unnormalized normal vector:
            N_raw = Tx × Ty = (-dz_dx, -dz_dy, 1)
          Normalizing to unit length:
            N = N_raw / sqrt(dz_dx^2 + dz_dy^2 + 1)

        Args:
            depth_norm: Normalized 2D float32 depth map in [0, 1].

        Returns:
            Nx, Ny, Nz: Normalized unit surface normal components, each of shape (H, W).
        """
        dz_dx = cv2.Sobel(depth_norm, cv2.CV_32F, 1, 0, ksize=3)
        dz_dy = cv2.Sobel(depth_norm, cv2.CV_32F, 0, 1, ksize=3)
        Nx = -dz_dx
        Ny = -dz_dy
        Nz = np.ones_like(depth_norm, dtype=np.float32)
        norm_len = np.sqrt(Nx**2 + Ny**2 + Nz**2) + 1e-6
        return Nx / norm_len, Ny / norm_len, Nz / norm_len

    def compute_local_normal_variation(
        self,
        Nx: np.ndarray,
        Ny: np.ndarray,
        Nz: np.ndarray,
        mask: np.ndarray
    ) -> float:
        """
        Computes high-frequency surface normal curvature via 3x3 Laplacian filtering.

        Why Laplacian?
          The Laplacian operator computes the divergence of the gradient (second derivative),
          measuring how rapidly the surface orientation changes from pixel to neighboring pixel.
          Smooth rigid surfaces (cans, boxes) have very low normal variation, whereas crinkled
          pouches, shrink wrap, and textured packaging exhibit high local normal variation.

        Args:
            Nx, Ny, Nz: Unit surface normal channels.
            mask: Boolean array indicating valid object pixels.

        Returns:
            Mean Laplacian gradient magnitude within the masked object area.
        """
        lap_x = np.abs(cv2.Laplacian(Nx, cv2.CV_32F, ksize=3))
        lap_y = np.abs(cv2.Laplacian(Ny, cv2.CV_32F, ksize=3))
        lap_z = np.abs(cv2.Laplacian(Nz, cv2.CV_32F, ksize=3))
        lap_mag = np.sqrt(lap_x**2 + lap_y**2 + lap_z**2)
        return float(np.mean(lap_mag[mask])) if np.sum(mask) > 0 else 0.0

    def compute_silhouette_descriptors(self, mask: np.ndarray) -> dict[str, float]:
        """
        Extracts 2D contour shape descriptors, convexity, eccentricity, and Hu moments.

        Features Computed:
          - extent: Ratio of contour pixels to the total bounding box area.
          - solidity: Ratio of contour area to its convex hull area (A_contour / A_hull).
          - convexity: Ratio of convex hull perimeter to contour perimeter (P_hull / P_contour).
          - eccentricity: Geometric eccentricity of the best-fitting ellipse (0 = circle, 1 = line).
          - equivalent_diameter: Diameter of a circle with the same area, normalized by max(H, W).
          - hu_moment_1, hu_moment_2: Log-transformed first two Hu moments (scale/rotation invariant).

        Args:
            mask: Binary 2D mask of the product proposal.

        Returns:
            Dictionary containing 8 contour and moment features.
        """
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

        # Select the dominant external contour
        cnt = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(cnt))
        peri = float(cv2.arcLength(cnt, True))
        extent = float(n_mask / (w * h + 1e-5))

        # Convex Hull calculation for Solidity and Convexity
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        hull_peri = float(cv2.arcLength(hull, True))
        solidity = float(area / (hull_area + 1e-5))
        convexity = float(hull_peri / (peri + 1e-5))

        # Equivalent circular diameter normalized by largest dimension
        eq_diam = float(np.sqrt(4.0 * area / (np.pi + 1e-5)) / float(max(h, w)))

        # Elliptical eccentricity: e = sqrt(1 - (b/a)^2)
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

        # Log Hu Moments for scale-invariant shape characterization
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

    def compute_5slice_profiles(
        self,
        mask: np.ndarray,
        depth_norm: np.ndarray
    ) -> tuple[float, float, float, list[float]]:
        """
        Divides the object mask into 5 equal vertical slices and computes width taper ratios.

        Why Vertical Slicing?
          In retail packaging:
            - Cylindrical cans have constant width from top to bottom: W_top ≈ W_mid ≈ W_bot (ratios ≈ 1.0).
            - Bottles and flasks taper near the top: W_top / W_mid << 1.0.
            - Irregular deformable bags taper unpredictably.
          By analyzing the ratio of top slice width to middle slice width (W1/W3), bottom to middle (W5/W3),
          and top to bottom (W1/W5), the model obtains a direct silhouette taper descriptor.

        Args:
            mask: Binary 2D mask of the object.
            depth_norm: Normalized 2D float32 depth map.

        Returns:
            top_w_ratio:   Ratio of top slice width to middle slice width (W1 / W3).
            bot_w_ratio:   Ratio of bottom slice width to middle slice width (W5 / W3).
            top_bot_ratio: Ratio of top slice width to bottom slice width (W1 / W5).
            slice_depth_means: List of mean depth values for each of the 5 slices.
        """
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

    def extract_features_and_reliability(
        self,
        rgb_crop: np.ndarray,
        raw_depth: np.ndarray | None = None,
        raw_mask: np.ndarray | None = None
    ) -> tuple[dict[str, float], np.ndarray, float, np.ndarray, np.ndarray]:
        """
        Extracts the full 35-dimensional geometric feature vector along with the adaptive
        depth reliability score R_d in [0.10, 1.0].

        Execution Flow:
          1. Metric Depth Estimation & Guided Bilateral Filtering.
          2. Morphological 3x3 Mask Erosion (suppressing boundary depth artifacts).
          3. Within-Mask Normalization: Zn = (Z - Z_min) / (Z_max - Z_min).
          4. Depth Distribution & Inter-Quantile Dispersion (10 features).
          5. Lateral & Vertical Depth Asymmetries (2 features).
          6. 5-Slice Width & Depth Taper Profiles (3 features).
          7. 3D Surface Normals, Curvatures & Planar/Curved Ratios (9 features).
          8. 2D Silhouette, Moments & Contour Geometry (8 features).
          9. Composite Interaction Ratios (3 features).
          10. Dynamic Depth Reliability Estimation (R_d).

        Args:
            rgb_crop:  RGB image crop of shape (H, W, 3).
            raw_depth: Optional precomputed depth map. If None, runs Depth Anything V2.
            raw_mask:  Optional binary foreground mask. If None, assumes center 90% bounding box.

        Returns:
            feature_dict:     Dictionary mapping each of the 35 feature names to its float value.
            feature_vector:   Ordered 35-dimensional float32 numpy array.
            depth_reliability: Adaptive scalar R_d in [0.10, 1.0].
            clean_depth:      Filtered float32 depth map.
            eroded_mask:      Boolean mask used for internal feature aggregation.
        """
        h, w = rgb_crop.shape[:2]

        # Step 1: Depth Map Estimation & Guided Bilateral Filtering
        if raw_depth is None:
            raw_depth = self.predict_metric_depth(rgb_crop)
        elif raw_depth.shape[:2] != (h, w):
            raw_depth = cv2.resize(raw_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        guide_bgr = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR)
        depth_u8 = cv2.normalize(raw_depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        filter_rad = max(1, min(8, min(h, w) // 4))
        clean_depth = cv2.ximgproc.guidedFilter(
            guide=guide_bgr,
            src=depth_u8,
            radius=filter_rad,
            eps=50
        ).astype(np.float32)

        # Step 2: Object Mask Handling & 3x3 Morphological Erosion
        if raw_mask is None:
            raw_mask = np.zeros((h, w), dtype=bool)
            raw_mask[int(h * 0.05):int(h * 0.95), int(w * 0.05):int(w * 0.95)] = True

        mask_u8 = raw_mask.astype(np.uint8)
        kernel = np.ones((3, 3), np.uint8)
        eroded_mask_u8 = cv2.erode(mask_u8, kernel, iterations=1)
        eroded_mask = (eroded_mask_u8 > 0)

        # Fallback guardrails if erosion removes too many pixels
        if np.sum(eroded_mask) < 10:
            eroded_mask = raw_mask.copy()
        if np.sum(eroded_mask) < 1:
            eroded_mask = np.ones((h, w), dtype=bool)

        n_raw = max(1, int(np.sum(raw_mask)))
        n_eroded = int(np.sum(eroded_mask))
        erosion_ratio = float(n_eroded / (n_raw + 1e-5))

        # Step 3: Object-Normalized Relative Depth (Zn)
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

        # Step 4: Depth Distribution & Percentiles
        p10, p25, p50, p75, p90 = np.percentile(depth_masked, [10, 25, 50, 75, 90])
        depth_std_val = float(np.std(depth_masked))
        depth_range_val = float(d_max - d_min)
        depth_iqr_75_25 = float(p75 - p25)
        depth_iqr_90_10 = float(p90 - p10)
        depth_skew = float(stats.skew(depth_masked)) if len(depth_masked) > 2 else 0.0

        # Step 5: Left-Right & Top-Bottom Spatial Depth Asymmetry
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

        # Step 6: 5-Slice Width & Depth Taper Profiles
        top_w_ratio, bot_w_ratio, top_bot_ratio, slice_depth_means = self.compute_5slice_profiles(eroded_mask, depth_norm)

        # Step 7: 3D Surface Normals & Curvatures
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

        # Step 8: 2D Silhouette, Moments & Contour Geometry
        sil_desc = self.compute_silhouette_descriptors(raw_mask)
        aspect_ratio_val = float(w / float(h + 1e-5))

        # Step 9: Composite Interaction Ratios
        curvature_thickness_ratio = float(std_Nx / (depth_std_val + 1e-5))
        box_compactness_score = float(sil_desc["solidity"] * aspect_ratio_val)
        smoothness_texture_ratio = float(mean_abs_Nx / (depth_iqr_75_25 + 1e-5))

        # Step 10: Dynamic Depth Reliability Score R_d in [0.10, 1.0]
        grad_smoothness_penalty = min(1.0, local_normal_var * 2.5)
        valid_pixel_score = min(1.0, n_eroded / 100.0)
        depth_reliability = float(np.clip(
            0.4 * erosion_ratio + 0.3 * (1.0 - grad_smoothness_penalty) + 0.3 * valid_pixel_score,
            0.10, 1.0
        ))

        # Assemble the 35 features into the standard feature dictionary
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

        # Convert to ordered 35-D numpy array matching GEOMETRIC_FEATURE_NAMES_V2
        feature_vector = np.array([feature_dict[k] for k in GEOMETRIC_FEATURE_NAMES_V2], dtype=np.float32)

        return feature_dict, feature_vector, depth_reliability, clean_depth, eroded_mask


# Global singleton instance for high-throughput batch processing
_GLOBAL_EXTRACTOR_V2 = None

def extract_geometry_features_v2(rgb_crop, raw_depth=None, raw_mask=None):
    """
    Convenience function providing a global reusable GeometricFeatureExtractorV2 instance.
    Avoids re-instantiating the extractor and reloading model weights across function calls.
    """
    global _GLOBAL_EXTRACTOR_V2
    if _GLOBAL_EXTRACTOR_V2 is None:
        _GLOBAL_EXTRACTOR_V2 = GeometricFeatureExtractorV2()
    return _GLOBAL_EXTRACTOR_V2.extract_features_and_reliability(rgb_crop, raw_depth, raw_mask)
