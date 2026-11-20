"""
Image pre-processing for camera-captured documents.
Paper 1 Section III-A: binarization, deskew, and normalization.
"""
import cv2
import numpy as np
from typing import Tuple, Optional


def preprocess_page(
    image: np.ndarray,
    target_size: Optional[int] = 768,
    do_deskew: bool = True,
    do_denoise: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Full page pre-processing pipeline.

    Args:
        image: (H, W, 3) BGR or RGB uint8 image.
        target_size: If set, resize longest side to this value.
        do_deskew: Detect and correct small skew angles.
        do_denoise: Apply mild denoising.

    Returns:
        (processed_image, scale_factor) where scale_factor maps back to original coords.
    """
    img = image.copy()
    original_h, original_w = img.shape[:2]

    # ── 1. Convert to float for processing ──
    img_f = img.astype(np.float32) / 255.0

    # ── 2. Mild denoising (bilateral) ──
    if do_denoise and img.shape[2] == 3:
        img_uint8 = (img_f * 255).astype(np.uint8)
        img_uint8 = cv2.bilateralFilter(img_uint8, d=5, sigmaColor=25, sigmaSpace=25)
        img_f = img_uint8.astype(np.float32) / 255.0

    # ── 3. Adaptive contrast normalisation (CLAHE on L channel) ──
    img_lab = cv2.cvtColor((img_f * 255).astype(np.uint8), cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(img_lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    img_lab = cv2.merge([l_ch, a_ch, b_ch])
    img = cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)

    # ── 4. Deskew ──
    if do_deskew:
        img, _ = correct_skew(img, max_angle_deg=10.0)

    # ── 5. Resize to target_size (longest side) only if larger ──
    h, w = img.shape[:2]
    if target_size and max(h, w) > target_size:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0

    actual_scale = img.shape[1] / original_w

    # ── 6. Pad to be divisible by 32 (required by detector backbone) ──
    ph = img.shape[0]
    pw = img.shape[1]
    pad_h = (32 - ph % 32) % 32
    pad_w = (32 - pw % 32) % 32
    if pad_h > 0 or pad_w > 0:
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w,
                                 cv2.BORDER_CONSTANT, value=(255, 255, 255))

    return img, actual_scale


def correct_skew(image: np.ndarray, max_angle_deg: float = 10.0) -> Tuple[np.ndarray, float]:
    """
    Estimate and correct small skew using Hough line voting.

    Returns:
        (deskewed_image, detected_angle_degrees)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Binarize
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # Find non-zero coords and compute PCA-based angle
    coords = np.column_stack(np.where(binary > 0))
    if len(coords) < 100:
        return image, 0.0

    # Use minAreaRect on contour points for angle
    try:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle += 90.0
        if abs(angle) > max_angle_deg:
            angle = 0.0  # Don't over-correct
    except Exception:
        angle = 0.0

    if abs(angle) < 0.3:
        return image, angle

    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT,
                              borderValue=(255, 255, 255))
    return rotated, angle


def normalize_for_detector(
    image: np.ndarray,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Normalize BGR image to ImageNet stats, return float32 (H, W, 3)."""
    img = image[:, :, ::-1].astype(np.float32) / 255.0  # BGR → RGB
    img = (img - np.array(mean)) / np.array(std)
    return img


def normalize_word_crop(crop: np.ndarray, target_h: int = 32) -> np.ndarray:
    """Resize word crop to fixed height, convert to grayscale, normalize to [-1, 1]."""
    h, w = crop.shape[:2]
    if h < 4:
        h = 4
    new_w = max(4, int(w * target_h / h))
    resized = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 3:
        resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return resized.astype(np.float32) / 127.5 - 1.0


def extract_word_crop(image: np.ndarray, bbox: Tuple[int, int, int, int],
                      pad: int = 0) -> np.ndarray:
    """Crop a word region from the page image with optional padding."""
    h, w = image.shape[:2]
    x, y, bw, bh = bbox
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + bw + pad)
    y2 = min(h, y + bh + pad)
    return image[y1:y2, x1:x2]
