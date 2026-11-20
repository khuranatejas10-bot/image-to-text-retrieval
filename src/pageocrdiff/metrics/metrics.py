"""
Comprehensive evaluation metrics for PageOCRDiff.
Metrics implemented (all paper-referenced):
  - CER  (Character Error Rate)
  - WER  (Word Error Rate)
  - Edit Distance (Levenshtein)
  - Precision / Recall / F1 for line detection
  - IoU for bounding boxes
  - TBR-L (Text-Block ROUGE-L) for block order fidelity
  - PSNR / SSIM for SR quality
"""
import re
import math
import numpy as np
from typing import List, Tuple, Dict, Sequence, Optional


# ─────────────────────────────── Levenshtein ──────────────────────────────

def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance (character-level)."""
    m, n = len(s1), len(s2)
    if m == 0:
        return n
    if n == 0:
        return m
    # dp[j] = edit distance for s1[:i] vs s2[:j]
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr
    return prev[n]


def cer(hypothesis: str, reference: str, normalise: bool = True) -> float:
    """Character Error Rate = edit_distance(hyp, ref) / len(ref)."""
    ref = reference.strip()
    hyp = hypothesis.strip()
    dist = edit_distance(hyp, ref)
    if normalise and len(ref) > 0:
        return dist / len(ref)
    return float(dist)


def wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate (whitespace-tokenised)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    dist = edit_distance(" ".join(hyp_words), " ".join(ref_words))
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else float(len(hyp_words))
    return edit_distance_word_level(hyp_words, ref_words) / len(ref_words)


def edit_distance_word_level(hyp: List[str], ref: List[str]) -> int:
    """Word-level edit distance."""
    m, n = len(hyp), len(ref)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j - 1], prev[j], curr[j - 1])
        prev = curr
    return prev[n]


# ─────────────────────────────── IoU ──────────────────────────────────────

def iou_box(box1: Tuple[int, int, int, int],
            box2: Tuple[int, int, int, int]) -> float:
    """
    Compute IoU between two (x, y, w, h) bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    ix1 = max(x1, x2)
    iy1 = max(y1, y2)
    ix2 = min(x1 + w1, x2 + w2)
    iy2 = min(y1 + h1, y2 + h2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / union if union > 0 else 0.0


def mean_iou(pred_boxes: List[Tuple], gt_boxes: List[Tuple],
             iou_threshold: float = 0.5) -> float:
    """Compute mean IoU across matched predicted/GT box pairs."""
    if not pred_boxes or not gt_boxes:
        return 0.0
    ious = []
    used = set()
    for pb in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if j in used:
                continue
            v = iou_box(pb, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            ious.append(best_iou)
            used.add(best_j)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────── Detection P/R/F1 ─────────────────────────

def detection_prf(
    pred_boxes: List[Tuple],
    gt_boxes: List[Tuple],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute precision, recall, F1 for object detection.
    Each predicted box is matched to at most one GT box by greedy IoU.
    """
    if not gt_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": len(pred_boxes), "fn": 0}
    if not pred_boxes:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": len(gt_boxes)}

    gt_matched = [False] * len(gt_boxes)
    tp = 0
    for pb in pred_boxes:
        best_iou = 0.0
        best_j = -1
        for j, gb in enumerate(gt_boxes):
            if gt_matched[j]:
                continue
            v = iou_box(pb, gb)
            if v > best_iou:
                best_iou = v
                best_j = j
        if best_j >= 0 and best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_j] = True

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ─────────────────────────────── TBR-L ────────────────────────────────────

def _lcs_length(a: List, b: List) -> int:
    """Length of longest common subsequence."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    prev = [0] * (n + 1)
    for i in range(m):
        curr = [0] * (n + 1)
        for j in range(n):
            if a[i] == b[j]:
                curr[j + 1] = prev[j] + 1
            else:
                curr[j + 1] = max(prev[j + 1], curr[j])
        prev = curr
    return prev[n]


def tbrl(predicted_blocks: List[str], reference_blocks: List[str]) -> float:
    """
    Text-Block ROUGE-L (TBR-L): measures block-level reading order fidelity.
    TBR-L = (2 * P_lcs * R_lcs) / (P_lcs + R_lcs)
    where P_lcs = LCS(pred, ref) / |pred| and R_lcs = LCS(pred, ref) / |ref|.

    Args:
        predicted_blocks: list of block-level text strings in predicted order.
        reference_blocks: list of block-level text strings in GT order.
    """
    if not predicted_blocks or not reference_blocks:
        return 0.0
    lcs = _lcs_length(predicted_blocks, reference_blocks)
    p_lcs = lcs / len(predicted_blocks)
    r_lcs = lcs / len(reference_blocks)
    if p_lcs + r_lcs == 0:
        return 0.0
    return 2 * p_lcs * r_lcs / (p_lcs + r_lcs)


# ─────────────────────────────── SR Quality ───────────────────────────────

def psnr(img1: np.ndarray, img2: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio. Inputs should be float in [0, max_val]."""
    mse = float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))
    if mse < 1e-12:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse)


def ssim(img1: np.ndarray, img2: np.ndarray,
         max_val: float = 1.0, win_size: int = 7) -> float:
    """
    Simplified SSIM. Inputs: (H, W) or (H, W, C) float arrays in [0, max_val].
    Returns mean SSIM over channels.
    """
    if img1.ndim == 3:
        return float(np.mean([ssim(img1[..., c], img2[..., c], max_val, win_size)
                               for c in range(img1.shape[2])]))
    k1, k2 = 0.01, 0.03
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    pad = win_size // 2

    from scipy.ndimage import uniform_filter
    mu1 = uniform_filter(img1.astype(np.float64), win_size)
    mu2 = uniform_filter(img2.astype(np.float64), win_size)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 ** 2, mu2 ** 2, mu1 * mu2
    s1 = uniform_filter(img1 ** 2, win_size) - mu1_sq
    s2 = uniform_filter(img2 ** 2, win_size) - mu2_sq
    s12 = uniform_filter(img1 * img2, win_size) - mu1_mu2
    num = (2 * mu1_mu2 + c1) * (2 * s12 + c2)
    den = (mu1_sq + mu2_sq + c1) * (s1 + s2 + c2)
    ssim_map = np.where(den > 0, num / den, 1.0)
    return float(ssim_map[pad:-pad, pad:-pad].mean())


# ─────────────────────────────── Aggregate ────────────────────────────────

def evaluate_page(
    pred_text: str,
    gt_text: str,
    pred_word_boxes: Optional[List[Tuple]] = None,
    gt_word_boxes: Optional[List[Tuple]] = None,
    pred_blocks: Optional[List[str]] = None,
    gt_blocks: Optional[List[str]] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all metrics for one page.
    Returns a flat dict of metric_name → value.
    """
    metrics: Dict[str, float] = {}

    metrics["cer"] = cer(pred_text, gt_text)
    metrics["wer"] = wer(pred_text, gt_text)
    metrics["edit_distance"] = float(edit_distance(pred_text.strip(), gt_text.strip()))

    if pred_word_boxes is not None and gt_word_boxes is not None:
        det = detection_prf(pred_word_boxes, gt_word_boxes, iou_threshold)
        metrics.update({f"word_{k}": float(v) for k, v in det.items()})
        metrics["mean_iou"] = mean_iou(pred_word_boxes, gt_word_boxes, iou_threshold)

    if pred_blocks is not None and gt_blocks is not None:
        metrics["tbrl"] = tbrl(pred_blocks, gt_blocks)

    return metrics
