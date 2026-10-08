"""
Command-Line Interface (CLI) for Unified Page-Level OCR-Diff System
Seamlessly integrates IEEE IoTJ 2024 and IEEE Access 2025.

Example usage:
python cli.py --image path/to/document.jpg --ocr-diff --output results.json --save-vis vis.jpg
"""

import argparse
import json
import os
import cv2
from unified_ocr import get_unified_ocr_engine

def main():
    parser = argparse.ArgumentParser(
        description="Unified Page-Level Document OCR with Generative Diffusion (OCR-Diff + Page-Level Access 2025)"
    )
    parser.add_argument("--image", required=True, help="Path to input document image")
    parser.add_argument("--output", default=None, help="Path to output JSON file or text file")
    parser.add_argument("--no-diffusion", action="store_true", help="Disable OCR-Diff diffusion super-resolution")
    parser.add_argument("--steps", type=int, default=5, help="Number of reverse diffusion steps (default: 5)")
    parser.add_argument("--thresh", type=float, default=0.15, help="Bleed-through confidence threshold (default: 0.15)")
    parser.add_argument("--no-wordninja", action="store_true", help="Disable WordNinja unigram splitting")
    parser.add_argument("--gt", default=None, help="Path to ground truth text file for TBR-L and CER evaluation")
    parser.add_argument("--save-vis", default=None, help="Path to save visual overlay image")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return 1

    gt_text = None
    if args.gt and os.path.exists(args.gt):
        with open(args.gt, 'r', encoding='utf-8') as f:
            gt_text = f.read()

    print(f"[CLI] Loading image: {args.image}")
    use_ocr_diff = not args.no_diffusion
    apply_wordninja = not args.no_wordninja

    engine = get_unified_ocr_engine()
    results = engine.process_document(
        image_path_or_np=args.image,
        use_ocr_diff=use_ocr_diff,
        ocr_diff_steps=args.steps,
        confidence_thresh=args.thresh,
        apply_wordninja=apply_wordninja,
        gt_text=gt_text
    )

    print("\n" + "=" * 60)
    print("EXTRACTED DOCUMENT TEXT (IN TOPOLOGICAL READING ORDER):")
    print("=" * 60)
    print(results['full_text'])
    print("=" * 60)

    print(f"\nStats: {results['stats']}")
    if results['metrics']:
        print(f"Evaluation Metrics: {results['metrics']}")

    if args.save_vis:
        cv2.imwrite(args.save_vis, results['visualization_image'])
        print(f"[CLI] Visualization saved to: {args.save_vis}")

    if args.output:
        if args.output.endswith('.json'):
            # Save JSON excluding numpy arrays
            serializable_results = {
                'full_text': results['full_text'],
                'blocks': results['blocks'],
                'metrics': results['metrics'],
                'stats': results['stats']
            }
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)
            print(f"[CLI] Structured JSON output saved to: {args.output}")
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(results['full_text'])
            print(f"[CLI] Text output saved to: {args.output}")

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main() or 0)
