"""
Connected component extraction and state estimation from detector output maps.
Paper 1 Section III-B: extract CCs from region/affinity, estimate orientation and scale.
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class TextComponent:
    """A detected text connected component with geometric state."""
    id: int
    center: Tuple[float, float]   # (x, y)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    area: int
    orientation_deg: float  # Dominant orientation in degrees [0, 180)
    orientation_bin: int    # Discrete bin index
    scale: float            # Estimated interline distance in pixels
    scale_bin: int          # Discrete bin index
    score: float            # Average region score
    mask: Optional[np.ndarray] = field(default=None, repr=False)


def extract_components(region_map: np.ndarray, affinity_map: np.ndarray,
                       orientation_logits: np.ndarray, scale_logits: np.ndarray,
                       region_thresh: float = 0.30, affinity_thresh: float = 0.15,
                       min_area: int = 8, min_score: float = 0.20,
                       num_orient_bins: int = 32, num_scale_bins: int = 10,
                       scale_bin_edges: Optional[List[float]] = None,
                       stride: int = 2) -> List[TextComponent]:
    """
    Extract text connected components from detector predictions.

    Args:
        region_map: (H, W) region heatmap in [0, 1].
        affinity_map: (H, W) affinity heatmap in [0, 1].
        orientation_logits: (num_orient, H, W) orientation class logits.
        scale_logits: (num_scale, H, W) scale class logits.
        region_thresh: Threshold for binarizing region map.
        affinity_thresh: Threshold for binarizing affinity map.
        min_area: Minimum CC area in pixels.
        min_score: Minimum average region score for a CC.
        stride: Output stride of detector (to map back to image coords).

    Returns:
        List of TextComponent objects.
    """
    # Combine region and affinity heatmaps (CRAFT formulation)
    binary = ((region_map >= region_thresh) | (affinity_map >= affinity_thresh)).astype(np.uint8) * 255

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Argmax maps for orientation and scale
    orient_map = np.argmax(orientation_logits, axis=0)  # (H, W)
    scale_map = np.argmax(scale_logits, axis=0)         # (H, W)

    # Scale bin midpoints for converting bin to pixel value
    if scale_bin_edges is None:
        scale_bin_edges = [0, 8, 12, 18, 26, 36, 48, 62, 78, 96, 1e6]
    scale_bin_edges = [float(x) for x in scale_bin_edges]
    scale_midpoints = []
    for i in range(len(scale_bin_edges) - 1):
        mid = (scale_bin_edges[i] + min(scale_bin_edges[i + 1], 150.0)) / 2
        scale_midpoints.append(mid)

    components = []
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cc_mask = (labels == label_id)

        # Robust score: max region score within CC (checks character presence)
        max_score = float(region_map[cc_mask].max())
        avg_score = float(region_map[cc_mask].mean())
        if max_score < min_score:
            continue

        cx, cy = float(centroids[label_id][0]), float(centroids[label_id][1])
        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])

        # Text-pixel-weighted orientation estimation
        cc_orient_vals = orient_map[cc_mask]
        if len(cc_orient_vals) > 0:
            # Weighted by region score
            weights = region_map[cc_mask]
            orient_bin = int(np.round(np.average(cc_orient_vals, weights=weights + 1e-8)))
            orient_bin = max(0, min(num_orient_bins - 1, orient_bin))
        else:
            orient_bin = 0
        orient_deg = (orient_bin + 0.5) * (180.0 / num_orient_bins)

        # Text-pixel-weighted scale estimation
        cc_scale_vals = scale_map[cc_mask]
        if len(cc_scale_vals) > 0:
            weights = region_map[cc_mask]
            scale_bin = int(np.round(np.average(cc_scale_vals, weights=weights + 1e-8)))
            scale_bin = max(0, min(num_scale_bins - 1, scale_bin))
        else:
            scale_bin = min(num_scale_bins - 1, max(0, int(h * 1.5 / 10)))
        scale_val = scale_midpoints[scale_bin] if scale_bin < len(scale_midpoints) else h * 1.5

        # Bounding box coordinates mapped from detector stride to original resolution
        bx = max(0, x * stride - 1)
        by = max(0, y * stride - 1)
        bw = max(4, w * stride)
        bh = max(6, h * stride + 1)

        comp = TextComponent(
            id=label_id,
            center=(cx * stride, cy * stride),
            bbox=(bx, by, bw, bh),
            area=area,
            orientation_deg=orient_deg,
            orientation_bin=orient_bin,
            scale=scale_val,
            scale_bin=scale_bin,
            score=avg_score,
            mask=cc_mask,
        )
        components.append(comp)

    return components
