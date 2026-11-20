#!/usr/bin/env python
"""
scripts/evaluate.py
Stage 5: Evaluate the full pipeline on a test set.
Computes CER, WER, TBR-L, detection P/R/F1, IoU per page and aggregates.
Usage:
    python scripts/evaluate.py --config configs/smoke_test.yaml \
        --det_ckpt runs/detector/ckpt_final.pt \
        --rec_ckpt runs/recognizer/ckpt_final.pt \
        --manifest data/synthetic/test_manifest.json
"""
import argparse
import os
import sys
import json

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import cv2
import torch
import numpy as np

from src.pageocrdiff.config import Config
from src.pageocrdiff.models import FourHeadDetector, CTCRecognizer, Vocabulary
from src.pageocrdiff.pipeline import PageOCRPipeline
from src.pageocrdiff.metrics import evaluate_page
from src.pageocrdiff.utils import set_seed, get_logger, save_metrics

logger = get_logger("evaluate")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate full OCR pipeline")
    p.add_argument("--config", default="configs/smoke_test.yaml")
    p.add_argument("--det_ckpt", default=None)
    p.add_argument("--rec_ckpt", default=None)
    p.add_argument("--sr_ckpt", default=None)
    p.add_argument("--manifest", default=None)
    p.add_argument("--data_root", default=None)
    p.add_argument("--output", default="reports/evaluation_results.json")
    p.add_argument("--iou_threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = Config.from_yaml(args.config) if os.path.exists(args.config) else Config()

    device = cfg.resolve_device()
    set_seed(cfg.seed)

    data_root = args.data_root or cfg.data_root
    manifest_path = args.manifest or os.path.join(data_root, "test_manifest.json")
    if not os.path.exists(manifest_path):
        manifest_path = os.path.join(data_root, "val_manifest.json")
    if not os.path.exists(manifest_path):
        logger.error(f"No manifest found at {manifest_path}. Run generate_synthetic_data.py first.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)
    samples = manifest.get("samples", [])
    logger.info(f"Evaluating {len(samples)} samples from {manifest_path}")

    # ── Load models ────────────────────────────────────────────────────────
    detector = FourHeadDetector(
        backbone=cfg.detector.backbone, pretrained=False,
        num_orient=cfg.detector.num_orientation_bins,
        num_scale=cfg.detector.num_scale_bins,
    )
    if args.det_ckpt and os.path.exists(args.det_ckpt):
        state = torch.load(args.det_ckpt, map_location="cpu", weights_only=False)
        detector.load_state_dict(state["model"])
        logger.info(f"Loaded detector: {args.det_ckpt}")
    else:
        logger.warning("Using randomly initialized detector (no checkpoint provided)")

    vocab = Vocabulary(cfg.recognizer.charset)
    use_bilstm = getattr(cfg.recognizer, "use_bilstm", False)
    bilstm_hidden = getattr(cfg.recognizer, "bilstm_hidden", 256)
    rec_state = None
    if args.rec_ckpt and os.path.exists(args.rec_ckpt):
        rec_state = torch.load(args.rec_ckpt, map_location="cpu", weights_only=False)
        m_state = rec_state.get("model", rec_state)
        if any("bilstm" in k for k in m_state.keys()):
            use_bilstm = True
            if "classifier.weight" in m_state:
                bilstm_hidden = m_state["classifier.weight"].shape[1] // 2
        else:
            use_bilstm = False

    recognizer = CTCRecognizer(
        num_classes=vocab.num_classes,
        use_bilstm=use_bilstm,
        bilstm_hidden=bilstm_hidden,
    )
    if rec_state is not None:
        recognizer.load_state_dict(rec_state["model"])
        logger.info(f"Loaded recognizer: {args.rec_ckpt} (use_bilstm={use_bilstm}, hidden={bilstm_hidden})")
    else:
        logger.warning("Using randomly initialized recognizer (no checkpoint provided)")

    sr_model = None
    if args.sr_ckpt and os.path.exists(args.sr_ckpt):
        from src.pageocrdiff.models import OCRDiffSR
        sr_model = OCRDiffSR(base_ch=cfg.diffusion_sr.base_channels,
                              time_dim=cfg.diffusion_sr.time_emb_dim,
                              T=cfg.diffusion_sr.T)
        state = torch.load(args.sr_ckpt, map_location="cpu", weights_only=False)
        sr_model.load_state_dict(state["model"])
        logger.info(f"Loaded SR model: {args.sr_ckpt}")

    pipeline = PageOCRPipeline(detector, recognizer, vocab, diffusion_sr=sr_model,
                                config=cfg, device=device)

    # ── Evaluation loop ────────────────────────────────────────────────────
    all_metrics = []
    agg = {"cer": [], "wer": [], "tbrl": []}

    for s in samples:
        img_path = os.path.join(data_root, s["image"])
        anno_path = os.path.join(data_root, s["annotation"])

        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Cannot read image: {img_path}")
            continue

        with open(anno_path) as f:
            anno = json.load(f)
        gt_text = anno.get("transcription", "")
        gt_word_boxes = [(int(w["x"]), int(w["y"]), int(w["w"]), int(w["h"]))
                         for w in anno.get("words", [])]

        result = pipeline.run(img)

        pred_word_boxes = [(int(w["bbox"][0]), int(w["bbox"][1]),
                            int(w["bbox"][2]), int(w["bbox"][3]))
                           for w in result.word_boxes]
        pred_blocks = [b for b in result.text.split("\n\n") if b.strip()]
        gt_blocks = [b for b in gt_text.split("\n") if b.strip()]

        page_metrics = evaluate_page(
            pred_text=result.text,
            gt_text=gt_text,
            pred_word_boxes=pred_word_boxes,
            gt_word_boxes=gt_word_boxes,
            pred_blocks=pred_blocks,
            gt_blocks=gt_blocks,
            iou_threshold=args.iou_threshold,
        )
        page_metrics["id"] = s["id"]
        all_metrics.append(page_metrics)

        for k in ["cer", "wer", "tbrl", "word_f1", "mean_iou"]:
            if k in page_metrics:
                agg.setdefault(k, []).append(page_metrics[k])

    # ── Aggregate ──────────────────────────────────────────────────────────
    summary = {k: float(np.mean(v)) for k, v in agg.items() if v}
    logger.info("\n── Evaluation Summary ───────────────────")
    for k, v in summary.items():
        logger.info(f"  {k.upper():>12}: {v:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"summary": summary, "per_page": all_metrics}, f, indent=2)
    logger.info(f"\nResults saved: {args.output}")


if __name__ == "__main__":
    main()
