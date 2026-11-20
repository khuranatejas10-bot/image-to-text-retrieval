"""
End-to-end page OCR inference pipeline.
Orchestrates: preprocessing → detection → CC extraction → layout → recognise → postprocess.
"""
import os
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from .preprocessing import preprocess_page, normalize_for_detector, normalize_word_crop, extract_word_crop
from .components import extract_components, TextComponent
from .grouping import build_layout, TextBlock, TextLine
from .reading_order import determine_reading_order, extract_ordered_text
from .postprocess import postprocess_page_text, correct_word_lexicon


class PageOCRResult:
    """Holds full page OCR output."""
    def __init__(self):
        self.text: str = ""
        self.lines: List[str] = []
        self.blocks: List[TextBlock] = []
        self.word_boxes: List[Dict] = []
        self.confidence: float = 0.0
        self.timing: Dict[str, float] = {}

    def __repr__(self):
        return f"PageOCRResult(lines={len(self.lines)}, confidence={self.confidence:.3f})"


class PageOCRPipeline:
    """
    Integrated page OCR pipeline (Paper 1 full system).

    Stages:
      1. Preprocess image
      2. Run FourHeadDetector
      3. Extract CCs with orientation/scale
      4. Layout: block grouping (Delaunay, epsilon=2) + line grouping (polynomial)
      5. Reading order (XY-cut + poly baseline sort)
      6. Optional OCR-Diff enhancement on low-confidence crops
      7. Run CTC recognizer on word crops
      8. Postprocess (WordNinja + confidence filter)
    """

    def __init__(
        self,
        detector,
        recognizer,
        vocab,
        diffusion_sr=None,
        config=None,
        device: str = "cpu",
        lexicon: Optional[List[str]] = None,
    ):
        self.detector = detector.to(device).eval()
        self.recognizer = recognizer.to(device).eval()
        self.vocab = vocab
        self.diffusion_sr = diffusion_sr.to(device).eval() if diffusion_sr else None
        self.config = config
        self.device = device

        # Config shortcuts
        cfg_d = config.detector if config else None
        self.region_thresh = cfg_d.region_threshold if cfg_d else 0.4
        self.affinity_thresh = cfg_d.affinity_threshold if cfg_d else 0.4
        self.min_comp_area = cfg_d.min_component_area if cfg_d else 10
        self.min_comp_score = cfg_d.min_component_score if cfg_d else 0.3
        self.epsilon = cfg_d.delaunay_epsilon if cfg_d else 2.0
        self.max_poly_degree = cfg_d.max_poly_degree if cfg_d else 4
        self.dev_factor = cfg_d.line_deviation_scale_factor if cfg_d else 0.25
        self.scale_bin_edges = cfg_d.scale_bin_edges if cfg_d else None

        cfg_r = config.recognizer if config else None
        self.img_height = cfg_r.img_height if cfg_r else 32
        self.img_width = cfg_r.img_width if cfg_r else 128

        cfg_sr = config.diffusion_sr if config else None
        self.sr_conf_thresh = cfg_sr.enhancement_confidence_threshold if cfg_sr else 0.5
        self.sr_ddim_steps = cfg_sr.ddim_steps if cfg_sr else 20
        self.hr_size = (cfg_sr.hr_height, cfg_sr.hr_width) if cfg_sr else (32, 128)

        # Lexicon for vocabulary prior correction (Paper 1 Section III-F)
        self.lexicon = lexicon if 'lexicon' in locals() and lexicon else None
        if self.lexicon is None:
            try:
                from ..data.synthetic_page_generator import _WORDS
                self.lexicon = _WORDS
            except Exception:
                self.lexicon = None

    @torch.no_grad()
    def run(self, image: np.ndarray) -> PageOCRResult:
        """
        Run end-to-end OCR on a BGR or RGB image.
        Returns PageOCRResult with full text, structured blocks/lines, and timing.
        """
        result = PageOCRResult()
        t0 = time.time()

        # ── Stage 1: Preprocess ──────────────────────────────────────────────
        img_proc, scale = preprocess_page(image)
        img_norm = normalize_for_detector(img_proc)  # (H, W, 3) float32
        H, W = img_proc.shape[:2]
        result.timing["preprocess"] = time.time() - t0

        # ── Stage 2: Detect ──────────────────────────────────────────────────
        t1 = time.time()
        img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        det_out = self.detector(img_tensor)
        region_map = det_out["region"][0, 0].cpu().numpy()
        affinity_map = det_out["affinity"][0, 0].cpu().numpy()
        orient_logits = det_out["orientation"][0].cpu().numpy()  # (32, H/2, W/2)
        scale_logits = det_out["scale"][0].cpu().numpy()         # (10, H/2, W/2)
        result.timing["detection"] = time.time() - t1

        # ── Stage 3: Extract CCs ─────────────────────────────────────────────
        t2 = time.time()
        components = extract_components(
            region_map, affinity_map, orient_logits, scale_logits,
            region_thresh=self.region_thresh, affinity_thresh=self.affinity_thresh,
            min_area=self.min_comp_area, min_score=self.min_comp_score,
            scale_bin_edges=self.scale_bin_edges,
        )
        result.timing["cc_extraction"] = time.time() - t2


        if not components:
            result.text = ""
            return result

        # ── Stage 4 & 5: Layout + Reading Order ──────────────────────────────
        t3 = time.time()
        blocks = build_layout(components, epsilon=self.epsilon,
                              max_poly_degree=self.max_poly_degree,
                              deviation_scale_factor=self.dev_factor)
        blocks = determine_reading_order(blocks, page_width=W)
        result.blocks = blocks
        result.timing["layout"] = time.time() - t3

        # ── Stage 6 & 7: Recognize words ─────────────────────────────────────
        t4 = time.time()
        all_line_texts = []
        all_line_confs = []
        word_boxes_out = []

        for block in blocks:
            for line in block.lines:
                line_words = []
                line_conf_vals = []
                for comp in line.components:
                    bx, by, bw, bh = comp.bbox
                    best_text = ""
                    best_conf = 0.0
                    best_in_lex = False

                    # Multi-margin crop exploration (Paper 1 Section III-F)
                    offsets = [(0, 0), (0, -2), (0, 2), (-1, -1), (1, -1)]
                    for dx, dh in offsets:
                        cx = max(0, bx + dx)
                        cy = max(0, by)
                        cw_b = max(4, bw)
                        ch_b = max(6, bh + dh)
                        crop = extract_word_crop(img_proc, (cx, cy, cw_b, ch_b), pad=2)
                        if crop.size == 0 or crop.shape[0] < 4 or crop.shape[1] < 4:
                            continue

                        # Optional: OCR-Diff SR enhancement (Paper 2 Section III-D)
                        crop_rgb = crop[:, :, ::-1] if crop.ndim == 3 else crop
                        if self.diffusion_sr and comp.score < self.sr_conf_thresh and self.device != "cpu":
                            crop_rgb = self._enhance_crop(crop_rgb)

                        crop_norm = normalize_word_crop(
                            crop_rgb,
                            target_h=self.img_height
                        )
                        # Pad to max width
                        cw = crop_norm.shape[1]
                        if cw > self.img_width:
                            crop_norm = crop_norm[:, :self.img_width]
                            cw = self.img_width
                        pad_w = self.img_width - cw
                        if pad_w > 0:
                            crop_norm = np.pad(crop_norm, ((0, 0), (0, pad_w)),
                                              mode="constant", constant_values=0.0)

                        crop_tensor = torch.from_numpy(crop_norm).unsqueeze(0).unsqueeze(0).float().to(self.device)
                        logits = self.recognizer(crop_tensor)
                        if hasattr(self.recognizer, "decode_beam_search"):
                            decoded = self.recognizer.decode_beam_search(logits, beam_width=10)
                        else:
                            decoded = self.recognizer.decode_greedy(logits)
                        raw_text = self.vocab.decode(decoded[0]) if decoded else ""

                        # Confidence: mean max-softmax over T
                        probs = torch.softmax(logits[:, 0, :], dim=-1)
                        conf = float(probs.max(dim=-1).values.mean())

                        if not raw_text.strip():
                            continue

                        # Lexicon prior matching
                        in_lex = bool(self.lexicon and (raw_text in self.lexicon or raw_text.capitalize() in self.lexicon or raw_text.lower() in self.lexicon))
                        corr_text = correct_word_lexicon(raw_text, self.lexicon) if self.lexicon else raw_text

                        if in_lex and conf > 0.88:
                            best_text = corr_text
                            best_conf = conf
                            best_in_lex = True
                            break
                        if (not best_in_lex and in_lex) or (conf > best_conf and not best_in_lex):
                            best_text = corr_text
                            best_conf = conf
                            best_in_lex = in_lex

                    word_text = best_text
                    conf = best_conf
                    if word_text.strip() and any(c.isalnum() for c in word_text):
                        line_words.append(word_text)
                        line_conf_vals.append(conf)
                        scale_factor = scale if scale > 0 else 1.0
                        word_boxes_out.append({
                            "text": word_text, "confidence": conf,
                            "bbox": [int(round(bx / scale_factor)), int(round(by / scale_factor)),
                                     int(round(bw / scale_factor)), int(round(bh / scale_factor))],
                            "block_id": block.id,
                        })

                if line_words:
                    all_line_texts.append(" ".join(line_words))
                    all_line_confs.append(float(np.mean(line_conf_vals)) if line_conf_vals else 0.0)

        result.timing["recognition"] = time.time() - t4

        # ── Stage 8: Postprocess ──────────────────────────────────────────────
        full_text, filtered_lines = postprocess_page_text(
            all_line_texts, all_line_confs, lexicon=self.lexicon
        )
        result.text = full_text
        result.lines = filtered_lines
        result.word_boxes = word_boxes_out
        result.confidence = float(np.mean(all_line_confs)) if all_line_confs else 0.0
        result.timing["total"] = time.time() - t0

        return result

    # Alias for flexibility
    process_page = run

    def _enhance_crop(self, crop_rgb: np.ndarray) -> np.ndarray:
        """Run OCR-Diff SR on a single low-confidence / small text crop."""
        if self.diffusion_sr is None:
            return crop_rgb
        if crop_rgb.ndim == 2:
            crop_rgb = cv2.cvtColor(crop_rgb, cv2.COLOR_GRAY2RGB)
        crop_tensor = torch.from_numpy(
            crop_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
        ).unsqueeze(0).to(self.device)
        steps = min(5, self.sr_ddim_steps) if self.device == "cpu" else self.sr_ddim_steps
        enhanced = self.diffusion_sr.enhance(crop_tensor, self.hr_size, steps)
        enhanced_np = (enhanced[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return enhanced_np
