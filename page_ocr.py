"""
Page-Level Document Recognition Framework for Camera-Captured Document Images
Based on IEEE Access 2025:
"Development of OCR Service for Page-Level Recognition for Camera-Captured Document Images"
Authors: Junyoung Park, Wonjun Kang, Seonji Park, Keuntek Lee, Hyung Il Koo, Nam Ik Cho.

Key Modules Implemented:
1. FourTensorNet: Deep Neural Network predicting (A)ffinity, (R)egion, (O)rientation (32 levels), and (S)cale (10 levels).
2. FourTensorLoss: Loss formulation combining MSE and pixel-wise masked cross-entropy.
3. Connected Component (CC) Extraction & State Estimation.
4. Delaunay Triangulation & Scale-Adaptive Edge Pruning for Text-Block Segmentation.
5. Skew Correction & Curvilinear Polynomial Text-Line Fitting (order k <= 4, condition RMSE <= s_bar / 4).
6. Topological Reading Order Detection (Intra-line x' sort, Inter-line f_i(x_avg) sort).
7. WordNinja Unigram Splitting & Bleed-Through Confidence Thresholding.
"""

import math
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import Delaunay
import wordninja
from typing import List, Dict, Tuple, Optional, Any

# ==============================================================================
# 1. Four-Tensor Deep Neural Network Architecture
# ==============================================================================

class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class FourTensorNet(nn.Module):
    """
    Deep Neural Network yielding 4 spatial prediction tensors (Paper Section III-A, Fig. 1 & 3):
    1. Affinity score: 1 channel, [0, 1]
    2. Region score: 1 channel, [0, 1]
    3. Orientation: N_D = 32 discrete classes (180 degrees / 32)
    4. Scale: N_S = 10 discrete classes (interline distance levels)
    """
    def __init__(self, num_orientation=32, num_scale=10):
        super(FourTensorNet, self).__init__()
        self.num_orientation = num_orientation
        self.num_scale = num_scale

        # Encoder (U-Net / CRAFT feature extractor style)
        self.inc = DoubleConv(3, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Decoder with Skip Connections (3 upsampling stages to match input H x W)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)

        # Output Heads
        # 1. Affinity Score (A)
        self.head_affinity = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 2. Region Score (R)
        self.head_region = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        # 3. Orientation (O) - 32 discrete levels
        self.head_orientation = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_orientation, kernel_size=1)
        )
        # 4. Scale (S) - 10 discrete levels
        self.head_scale = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, self.num_scale, kernel_size=1)
        )

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        u1 = self.up1(x4)
        u1 = torch.cat([u1, x3], dim=1)
        u1 = self.conv_up1(u1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv_up2(u2)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, x1], dim=1)
        feats = self.conv_up3(u3)

        affinity = self.head_affinity(feats)
        region = self.head_region(feats)
        orientation_logits = self.head_orientation(feats)
        scale_logits = self.head_scale(feats)

        return {
            'affinity': affinity,
            'region': region,
            'orientation': orientation_logits,
            'scale': scale_logits
        }


class FourTensorLoss(nn.Module):
    """
    Combined Loss Function (Paper Section III-C, Eq. 1, 2, 3):
    L = L_r + L_a + lambda_s * L_s + lambda_o * L_o
    where lambda_s = 0.1, lambda_o = 0.3.
    """
    def __init__(self, lambda_s=0.1, lambda_o=0.3):
        super(FourTensorLoss, self).__init__()
        self.lambda_s = lambda_s
        self.lambda_o = lambda_o

    def forward(self, preds, targets):
        # preds: dict with 'affinity', 'region', 'orientation', 'scale'
        # targets: dict with 'affinity_gt', 'region_gt', 'orientation_gt', 'scale_gt'
        pred_region = preds['region']
        pred_affinity = preds['affinity']
        pred_orientation = preds['orientation']
        pred_scale = preds['scale']

        gt_region = targets['region_gt']
        gt_affinity = targets['affinity_gt']
        gt_orientation = targets['orientation_gt']
        gt_scale = targets['scale_gt']

        # Region and Affinity MSE loss
        loss_r = F.mse_loss(pred_region, gt_region)
        loss_a = F.mse_loss(pred_affinity, gt_affinity)

        # Scale and Orientation masked Cross-Entropy Loss (Eq. 1 & Eq. 2)
        # Masked by soft region score map S_r*(z)
        ce_scale = F.cross_entropy(pred_scale, gt_scale, reduction='none') # B x H x W
        ce_orientation = F.cross_entropy(pred_orientation, gt_orientation, reduction='none') # B x H x W

        mask = gt_region.squeeze(1) # B x H x W
        loss_s = (mask * ce_scale).sum() / (mask.sum() + 1e-6)
        loss_o = (mask * ce_orientation).sum() / (mask.sum() + 1e-6)

        total_loss = loss_r + loss_a + self.lambda_s * loss_s + self.lambda_o * loss_o
        return {
            'total_loss': total_loss,
            'loss_r': loss_r,
            'loss_a': loss_a,
            'loss_s': loss_s,
            'loss_o': loss_o
        }


# ==============================================================================
# 2. Geometric Analysis: CC Extraction, Delaunay Triangulation & Curvilinear Lines
# ==============================================================================

class ConnectedComponent:
    """Represents a text Connected Component (CC) c_p."""
    def __init__(self, cc_id: int, center: Tuple[float, float], bbox: Tuple[int, int, int, int],
                 scale: float, orientation: float, area: int, mask: np.ndarray):
        self.id = cc_id
        self.x = center[0]
        self.y = center[1]
        self.bbox = bbox # x, y, w, h
        self.scale = scale # Estimated interline / text scale s_p
        self.orientation = orientation # In radians theta_p
        self.area = area
        self.mask = mask
        # Rotated coordinates (set during block skew correction)
        self.x_rot = self.x
        self.y_rot = self.y


def extract_connected_components(region_map: np.ndarray,
                                 scale_map: np.ndarray,
                                 orientation_map: np.ndarray,
                                 region_thresh: float = 0.35,
                                 min_area: int = 15) -> List[ConnectedComponent]:
    """
    Extracts Connected Components (CCs) by thresholding the region score map (Paper Section III-B).
    Unlike MSER, region score map guarantees CCs are text pixels without non-text filtering.
    """
    binary_map = (region_map > region_thresh).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)

    ccs = []
    # Base scale levels calibration (pixel distance per scale class, 10 levels)
    scale_levels_px = [12.0, 16.0, 22.0, 30.0, 40.0, 52.0, 68.0, 88.0, 112.0, 144.0]
    num_orientation_bins = 32

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        cx, cy = centroids[label_id]
        x = stats[label_id, cv2.CC_STAT_LEFT]
        y = stats[label_id, cv2.CC_STAT_TOP]
        w = stats[label_id, cv2.CC_STAT_WIDTH]
        h = stats[label_id, cv2.CC_STAT_HEIGHT]

        cc_mask = (labels == label_id)
        
        # Scale estimation: average scale level within CC mask
        cc_scale_indices = scale_map[cc_mask]
        if len(cc_scale_indices) > 0:
            avg_scale_idx = int(np.round(np.mean(cc_scale_indices)))
            avg_scale_idx = max(0, min(len(scale_levels_px) - 1, avg_scale_idx))
            scale_val = scale_levels_px[avg_scale_idx]
        else:
            scale_val = max(16.0, float(h) * 1.5)

        # Orientation estimation: dominant orientation angle within CC mask
        cc_orient_indices = orientation_map[cc_mask]
        if len(cc_orient_indices) > 0:
            dominant_bin = int(np.bincount(cc_orient_indices).argmax())
            # Convert bin [0, 31] to radians [0, pi)
            orient_rad = (dominant_bin / float(num_orientation_bins)) * math.pi
        else:
            orient_rad = 0.0

        cc = ConnectedComponent(
            cc_id=label_id,
            center=(float(cx), float(cy)),
            bbox=(x, y, w, h),
            scale=scale_val,
            orientation=orient_rad,
            area=area,
            mask=cc_mask
        )
        ccs.append(cc)

    return ccs


def segment_text_blocks(ccs: List[ConnectedComponent],
                        epsilon: float = 2.0,
                        scale_diff_thresh: float = 25.0) -> List[List[ConnectedComponent]]:
    """
    Text-Block Segmentation using Delaunay Triangulation & Scale-Adaptive Edge Pruning
    (Paper Section III-D & Section VIII-A, Eq. 4):
    Criterion for edge removal: d_pq >= epsilon * min(s_p, s_q) or |s_p - s_q| > scale_diff_thresh
    """
    if len(ccs) == 0:
        return []
    if len(ccs) <= 3:
        return [ccs]

    points = np.array([[cc.x, cc.y] for cc in ccs], dtype=np.float32)

    # Add small jitter to avoid degenerate Delaunay coplanar points error
    jitter = np.random.normal(0, 1e-4, points.shape)
    tri = Delaunay(points + jitter)

    # Build adjacency graph
    n = len(ccs)
    adj = {i: set() for i in range(n)}

    # Collect edges from Delaunay simplices
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            u, v = simplex[i], simplex[(i + 1) % 3]
            edge = tuple(sorted((u, v)))
            edges.add(edge)

    # Filter edges according to Eq. 4: d_pq < epsilon * min(s_p, s_q)
    for u, v in edges:
        p1, p2 = points[u], points[v]
        dist = np.linalg.norm(p1 - p2)
        s_u = ccs[u].scale
        s_v = ccs[v].scale

        # Paper Eq. 4 + Section VIII-A scale difference handling
        max_allowed_dist = epsilon * min(s_u, s_v)
        scale_diff = abs(s_u - s_v)

        if dist < max_allowed_dist and scale_diff < scale_diff_thresh:
            adj[u].add(v)
            adj[v].add(u)

    # Connected subgraphs correspond to text blocks
    visited = [False] * n
    blocks = []

    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            block_indices = [i]

            while queue:
                curr = queue.pop(0)
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
                        block_indices.append(neighbor)

            block_ccs = [ccs[idx] for idx in block_indices]
            blocks.append(block_ccs)

    return blocks


class TextLine:
    """Represents an extracted curvilinear text-line."""
    def __init__(self, line_id: int, ccs: List[ConnectedComponent], poly_coeffs: np.ndarray,
                 block_id: int = 0):
        self.line_id = line_id
        self.ccs = ccs
        self.poly_coeffs = poly_coeffs # Polynomial coefficients f(x)
        self.block_id = block_id
        self.text = ""
        self.confidence = 0.0
        self.reading_order = 0
        self.bbox = self._calculate_bbox()

    def _calculate_bbox(self):
        if not self.ccs:
            return (0, 0, 0, 0)
        x_min = min(cc.bbox[0] for cc in self.ccs)
        y_min = min(cc.bbox[1] for cc in self.ccs)
        x_max = max(cc.bbox[0] + cc.bbox[2] for cc in self.ccs)
        y_max = max(cc.bbox[1] + cc.bbox[3] for cc in self.ccs)
        return (x_min, y_min, x_max - x_min, y_max - y_min)


def detect_curvilinear_lines(block_ccs: List[ConnectedComponent],
                             poly_order: int = 4,
                             block_id: int = 0) -> List[TextLine]:
    """
    Curvilinear Text-Line Detection within a segmented text-block (Paper Section III-E, Eq. 5).
    Corrects skew, bottom-up groups CCs, and enforces curvilinear condition:
    sqrt( (1/|T|) * sum( (y'_p - f(x'_p))^2 ) ) <= s_bar / 4
    """
    if len(block_ccs) == 0:
        return []

    # 1. Correct skew using dominant angle of CCs in the block (Section III-E)
    angles = [cc.orientation for cc in block_ccs]
    # Use median / dominant angle
    dominant_angle = float(np.median(angles))

    # Compute block center
    center_x = float(np.mean([cc.x for cc in block_ccs]))
    center_y = float(np.mean([cc.y for cc in block_ccs]))

    cos_a = math.cos(-dominant_angle)
    sin_a = math.sin(-dominant_angle)

    for cc in block_ccs:
        dx = cc.x - center_x
        dy = cc.y - center_y
        cc.x_rot = cos_a * dx - sin_a * dy
        cc.y_rot = sin_a * dx + cos_a * dy

    # 2. Bottom-up grouping with polynomial fit verification
    # Sort CCs primarily along x_rot
    sorted_ccs = sorted(block_ccs, key=lambda c: (c.x_rot, c.y_rot))
    lines_ccs = []
    used = [False] * len(sorted_ccs)

    for i in range(len(sorted_ccs)):
        if used[i]:
            continue
        seed_cc = sorted_ccs[i]
        used[i] = True
        curr_line = [seed_cc]

        # Greedily search along x_rot direction for adjacent CCs
        while True:
            best_candidate_idx = None
            best_dist = float('inf')

            for j in range(len(sorted_ccs)):
                if used[j]:
                    continue
                cand_cc = sorted_ccs[j]
                s_bar = float(np.mean([c.scale for c in curr_line] + [cand_cc.scale]))

                # Horizontal distance from current line endpoints
                dx = cand_cc.x_rot - curr_line[-1].x_rot
                if dx < -5.0 or dx > s_bar * 4.0:
                    continue

                # Predict y based on current line polynomial or last point
                if len(curr_line) >= 2:
                    xs_cur = np.array([c.x_rot for c in curr_line])
                    ys_cur = np.array([c.y_rot for c in curr_line])
                    deg = min(poly_order, len(curr_line) - 1)
                    poly = np.polyfit(xs_cur, ys_cur, deg=deg)
                    y_pred = float(np.polyval(poly, cand_cc.x_rot))
                else:
                    y_pred = curr_line[-1].y_rot

                y_dist = abs(cand_cc.y_rot - y_pred)
                if y_dist <= s_bar * 0.75:
                    # Check curvilinear condition on candidate set (Eq. 5)
                    test_set = curr_line + [cand_cc]
                    xs_test = np.array([c.x_rot for c in test_set])
                    ys_test = np.array([c.y_rot for c in test_set])
                    k = min(poly_order, len(test_set) - 1)
                    coeffs = np.polyfit(xs_test, ys_test, deg=k)
                    rmse = float(np.sqrt(np.mean((ys_test - np.polyval(coeffs, xs_test)) ** 2)))

                    if rmse <= (s_bar / 4.0) + 1e-3:
                        tot_dist = dx + y_dist
                        if tot_dist < best_dist:
                            best_dist = tot_dist
                            best_candidate_idx = j

            if best_candidate_idx is not None:
                used[best_candidate_idx] = True
                curr_line.append(sorted_ccs[best_candidate_idx])
                # Ensure curr_line is ordered by x_rot
                curr_line.sort(key=lambda c: c.x_rot)
            else:
                break

        lines_ccs.append(curr_line)

    # 3. Fit final polynomial for each line
    result_lines = []
    for l_id, l_ccs in enumerate(lines_ccs):
        if len(l_ccs) >= 2:
            xs = np.array([c.x_rot for c in l_ccs])
            ys = np.array([c.y_rot for c in l_ccs])
            k = min(poly_order, len(l_ccs) - 1)
            coeffs = np.polyfit(xs, ys, deg=k)
        else:
            coeffs = np.array([l_ccs[0].y_rot])

        line_obj = TextLine(line_id=l_id, ccs=l_ccs, poly_coeffs=coeffs, block_id=block_id)
        result_lines.append(line_obj)

    return result_lines


def determine_reading_order(lines: List[TextLine], block_ccs: List[ConnectedComponent]) -> List[TextLine]:
    """
    Topological Reading Order Detection (Paper Section V-A, Eq. 10):
    1. Intra-line reading order: sort CCs within text-line T_i by x'_p ascending.
    2. Inter-line reading order: compute x_avg^i = (1 / |C_i|) * sum(x'_p) for the block.
       Evaluate line polynomial at x_avg^i: f_i(x_avg^i).
       Sort lines by ascending f_i(x_avg^i).
    """
    if len(lines) == 0:
        return []

    # 1. Intra-line: sort CCs within line by rotated x_rot ascending
    for line in lines:
        line.ccs.sort(key=lambda c: c.x_rot)

    # 2. Inter-line: compute block-level average x_rot (Eq. 10)
    x_avg = float(np.mean([c.x_rot for c in block_ccs]))

    # Evaluate polynomial at x_avg for each line
    def get_line_y_at_xavg(line: TextLine) -> float:
        if len(line.poly_coeffs) > 0:
            return float(np.polyval(line.poly_coeffs, x_avg))
        return float(np.mean([c.y_rot for c in line.ccs]))

    # Sort lines in ascending order of f_i(x_avg)
    lines.sort(key=get_line_y_at_xavg)

    for order_idx, line in enumerate(lines):
        line.reading_order = order_idx + 1

    return lines


# ==============================================================================
# 3. Post-Processing: WordNinja Splitting & Bleed-Through Confidence Filter
# ==============================================================================

def postprocess_line_text(raw_text: str,
                          confidence: float,
                          confidence_thresh: float = 0.15,
                          apply_wordninja: bool = True) -> Optional[str]:
    """
    Text Post-Processing (Paper Section V-C & Section VIII-B):
    1. Bleed-through rejection: drops text where recognition confidence < confidence_thresh.
    2. WordNinja unigram frequency splitting: corrects mistakenly concatenated words.
    """
    if confidence < confidence_thresh:
        # Bleed-through or noisy ghost text rejected
        return None

    cleaned_words = []
    tokens = raw_text.split()
    for tok in tokens:
        # If token is excessively long without spaces and purely alphabetical, split via WordNinja
        if apply_wordninja and len(tok) > 12 and tok.isalpha():
            split_words = wordninja.split(tok)
            cleaned_words.extend(split_words)
        else:
            cleaned_words.append(tok)

    return " ".join(cleaned_words)


# ==============================================================================
# 4. High-Level Page Recognition Engine
# ==============================================================================

class PageOCREngine:
    """
    Full Page-Level OCR Engine combining 4-Tensor Detection, Delaunay Triangulation,
    Curvilinear Polynomial Line Extraction, Reading Order Detection, and Post-Processing.
    """
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FourTensorNet().to(self.device)
        self.model.eval()

    def detect_and_parse(self, image_np: np.ndarray,
                         region_thresh: float = 0.35,
                         epsilon: float = 2.0) -> Dict[str, Any]:
        """
        Processes a full page document image.
        Returns structured blocks, lines in reading order, and visualization data.
        """
        orig_h, orig_w = image_np.shape[:2]

        # Resize to standard detection resolution (768 x 768 per paper Section VII-C)
        det_h, det_w = 768, 768
        img_resized = cv2.resize(image_np, (det_w, det_h))
        img_tensor = torch.tensor(img_resized, dtype=torch.float32, device=self.device).permute(2, 0, 1).unsqueeze(0) / 255.0

        with torch.no_grad():
            outputs = self.model(img_tensor)

        region_map = outputs['region'].squeeze().cpu().numpy()
        affinity_map = outputs['affinity'].squeeze().cpu().numpy()
        orient_logits = outputs['orientation'].squeeze().cpu().numpy()
        scale_logits = outputs['scale'].squeeze().cpu().numpy()

        # Argmax for orientation and scale class maps
        orient_map = np.argmax(orient_logits, axis=0) # 0 to 31
        scale_map = np.argmax(scale_logits, axis=0)   # 0 to 9

        # Upsample maps back to original image dimensions
        region_map_full = cv2.resize(region_map, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        scale_map_full = cv2.resize(scale_map.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)
        orient_map_full = cv2.resize(orient_map.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST).astype(np.int32)

        # 1. Connected Component Extraction
        ccs = extract_connected_components(region_map_full, scale_map_full, orient_map_full, region_thresh=region_thresh)

        # Fallback heuristic CC generator if synthetic random init yields sparse components
        if len(ccs) < 2:
            # Generate pseudo-CCs using standard morphological gradient / MSER on document
            gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY) if len(image_np.shape) == 3 else image_np
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 3)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            for i in range(1, min(num_labels, 150)):
                area = stats[i, cv2.CC_STAT_AREA]
                if 20 < area < (orig_h * orig_w * 0.1):
                    cx, cy = centroids[i]
                    x, y, w, h = stats[i, :4]
                    ccs.append(ConnectedComponent(
                        cc_id=i, center=(float(cx), float(cy)), bbox=(x, y, w, h),
                        scale=max(16.0, float(h) * 1.5), orientation=0.0, area=area, mask=(labels == i)
                    ))

        # 2. Text-Block Segmentation via Delaunay Triangulation
        blocks_ccs = segment_text_blocks(ccs, epsilon=epsilon)

        # Sort blocks in reading order (top-to-bottom, left-to-right)
        blocks_ccs.sort(key=lambda b: (min(c.y for c in b), min(c.x for c in b)))

        # 3. For each block: Skew correction, Curvilinear polynomial lines, and Reading order
        parsed_blocks = []
        all_lines = []
        global_line_counter = 1

        for b_id, b_ccs in enumerate(blocks_ccs):
            lines = detect_curvilinear_lines(b_ccs, poly_order=4, block_id=b_id)
            lines = determine_reading_order(lines, b_ccs)

            block_lines_data = []
            for line in lines:
                line.global_id = global_line_counter
                global_line_counter += 1
                all_lines.append(line)
                block_lines_data.append(line)

            parsed_blocks.append({
                'block_id': b_id + 1,
                'ccs_count': len(b_ccs),
                'lines': block_lines_data
            })

        return {
            'blocks': parsed_blocks,
            'lines': all_lines,
            'ccs': ccs,
            'maps': {
                'region': region_map,
                'affinity': affinity_map,
                'orientation': orient_map,
                'scale': scale_map
            }
        }


def render_four_tensor_vis(maps: Dict[str, np.ndarray]) -> np.ndarray:
    """
    Renders a 2x2 multi-panel visualization of the 4 prediction tensors:
    - Top-Left: Region Score Map (Heatmap)
    - Top-Right: Affinity Score Map (Heatmap)
    - Bottom-Left: 32-Level Orientation Map (HSV colored)
    - Bottom-Right: 10-Level Scale Map (Color map)
    """
    reg = (np.clip(maps['region'], 0, 1) * 255).astype(np.uint8)
    aff = (np.clip(maps['affinity'], 0, 1) * 255).astype(np.uint8)
    orient = maps['orientation'].astype(np.float32)
    scale = maps['scale'].astype(np.float32)

    h, w = reg.shape[:2]

    # Panel 1: Region Map (Inferno/Jet)
    p1 = cv2.applyColorMap(reg, cv2.COLORMAP_JET)
    cv2.putText(p1, "(R)egion Score Map", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 2: Affinity Map (Viridis/Ocean)
    p2 = cv2.applyColorMap(aff, cv2.COLORMAP_OCEAN)
    cv2.putText(p2, "(A)ffinity Score Map", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 3: Orientation Map (32 levels mapped to 0-180 degrees)
    orient_norm = ((orient / 31.0) * 255).astype(np.uint8)
    p3 = cv2.applyColorMap(orient_norm, cv2.COLORMAP_HSV)
    cv2.putText(p3, "(O)rientation Field [32 Bins]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Panel 4: Scale Map (10 levels mapped to colors)
    scale_norm = ((scale / 9.0) * 255).astype(np.uint8)
    p4 = cv2.applyColorMap(scale_norm, cv2.COLORMAP_TURBO)
    cv2.putText(p4, "(S)cale Level Map [10 Bins]", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])
    return grid

