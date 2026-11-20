"""
Generate all figures for the IEEE research paper.
Outputs to paper/figures/
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ── Colour Palette ──
C_BLUE   = "#2563EB"
C_ORANGE = "#EA580C"
C_GREEN  = "#16A34A"
C_RED    = "#DC2626"
C_PURPLE = "#7C3AED"
C_GRAY   = "#6B7280"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})

# ═══════════════════════════════════════════════════════
# Fig 1 – Detector Training Loss Convergence
# ═══════════════════════════════════════════════════════
def fig_detector_loss():
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "kaggle_gpu", "detector", "train_metrics.jsonl")
    steps, total, region, affinity = [], [], [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(d["step"])
            total.append(d["loss"])
            region.append(d["region"])
            affinity.append(d["affinity"])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(steps, total, color=C_BLUE, linewidth=1.5, label="Total Loss")
    ax.plot(steps, region, color=C_ORANGE, linewidth=1.2, linestyle="--", label="Region MSE")
    ax.plot(steps, affinity, color=C_GREEN, linewidth=1.2, linestyle="-.", label="Affinity MSE")
    ax.axhline(y=0.35, color=C_RED, linewidth=0.8, linestyle=":", label="Paper 1 Target (<0.35)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("Four-Head Detector Training Convergence", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper right")
    ax.set_yscale("log")
    ax.set_xlim(0, 3000)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "detector_loss.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "detector_loss.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ detector_loss")

# ═══════════════════════════════════════════════════════
# Fig 2 – Recognizer CTC Loss Convergence
# ═══════════════════════════════════════════════════════
def fig_recognizer_loss():
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "kaggle_gpu", "recognizer", "train_metrics.jsonl")
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(d["step"])
            losses.append(d["ctc_loss"])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(steps, losses, color=C_PURPLE, linewidth=1.2)
    ax.axhline(y=0.20, color=C_RED, linewidth=0.8, linestyle=":", label="Paper 1 CTC Target (~0.20)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("CTC Loss")
    ax.set_title("CTC Recognizer Training Convergence", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_yscale("log")
    ax.set_xlim(0, 10000)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "recognizer_loss.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "recognizer_loss.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ recognizer_loss")

# ═══════════════════════════════════════════════════════
# Fig 3 – Diffusion SR MSE Loss Convergence
# ═══════════════════════════════════════════════════════
def fig_diffusion_loss():
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "kaggle_gpu", "diffusion_sr", "pretrain_metrics.jsonl")
    steps, losses = [], []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            steps.append(d["step"])
            losses.append(d["mse_loss"])

    fig, ax = plt.subplots(figsize=(4.5, 3))
    ax.plot(steps, losses, color=C_GREEN, linewidth=1.5)
    ax.axhline(y=0.05, color=C_RED, linewidth=0.8, linestyle=":", label="Paper 2 Target (<0.05)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE Loss")
    ax.set_title("OCR-Diff Diffusion SR Training Convergence", fontsize=10, fontweight="bold")
    ax.legend(fontsize=7)
    ax.set_xlim(0, 3000)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "diffusion_loss.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "diffusion_loss.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ diffusion_loss")

# ═══════════════════════════════════════════════════════
# Fig 4 – Per-Page CER/WER Distribution
# ═══════════════════════════════════════════════════════
def fig_per_page_metrics():
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "kaggle_gpu", "evaluation_results.json")
    with open(path) as f:
        data = json.load(f)
    pages = data["per_page"]
    ids = [p["id"].replace("test_", "") for p in pages]
    cers = [p["cer"] for p in pages]
    wers = [p["wer"] for p in pages]

    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3))
    bars1 = ax.bar(x - width/2, cers, width, label="CER", color=C_BLUE, alpha=0.85)
    bars2 = ax.bar(x + width/2, wers, width, label="WER", color=C_ORANGE, alpha=0.85)
    ax.axhline(y=0.05, color=C_RED, linewidth=1, linestyle=":", label="Paper 1 CER Target (0.05)")
    ax.axhline(y=0.08, color=C_RED, linewidth=1, linestyle="--", alpha=0.6, label="Paper 1 WER Target (0.08)")
    ax.set_xlabel("Test Page Index")
    ax.set_ylabel("Error Rate")
    ax.set_title("Per-Page CER and WER Distribution (20 Test Documents)", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=6, rotation=45)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(0, max(max(cers), max(wers)) * 1.4 + 0.01)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "per_page_cer_wer.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "per_page_cer_wer.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ per_page_cer_wer")

# ═══════════════════════════════════════════════════════
# Fig 5 – Benchmark Comparison Bar Chart (Ours vs Papers)
# ═══════════════════════════════════════════════════════
def fig_benchmark_comparison():
    metrics = ["CER", "WER", "Box F1", "IoU", "SSIM", "TBR-L"]
    ours    = [0.0286, 0.0456, 0.9575, 0.8659, 0.9460, 0.9850]
    paper1  = [0.050,  0.080,  0.8500, 0.7000, None,   0.800]
    paper2  = [None,   None,   None,   None,   0.880,  None]

    # For visualization, combine paper baselines (take whichever is defined)
    paper_best = []
    for p1, p2 in zip(paper1, paper2):
        if p1 is not None:
            paper_best.append(p1)
        elif p2 is not None:
            paper_best.append(p2)
        else:
            paper_best.append(0)

    x = np.arange(len(metrics))
    width = 0.3

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(x - width/2, ours, width, label="PageOCRDiff (Ours)", color=C_BLUE, alpha=0.9, edgecolor="white")
    ax.bar(x + width/2, paper_best, width, label="IEEE Paper Benchmark", color=C_GRAY, alpha=0.6, edgecolor="white")

    ax.set_ylabel("Score / Rate")
    ax.set_title("Comprehensive Benchmark: Ours vs. IEEE Papers", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.grid(True, axis="y", alpha=0.3)

    # Annotate values
    for i, v in enumerate(ours):
        ax.text(i - width/2, v + 0.02, f"{v:.4f}" if v < 0.1 else f"{v:.3f}", ha="center", fontsize=6, fontweight="bold", color=C_BLUE)
    for i, v in enumerate(paper_best):
        if v > 0:
            ax.text(i + width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=6, color=C_GRAY)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "benchmark_comparison.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "benchmark_comparison.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ benchmark_comparison")

# ═══════════════════════════════════════════════════════
# Fig 6 – Per-Page Box F1 & IoU
# ═══════════════════════════════════════════════════════
def fig_per_page_detection():
    path = os.path.join(os.path.dirname(__file__), "..", "runs", "kaggle_gpu", "evaluation_results.json")
    with open(path) as f:
        data = json.load(f)
    pages = data["per_page"]
    ids = [p["id"].replace("test_", "") for p in pages]
    f1s = [p["word_f1"] for p in pages]
    ious = [p["mean_iou"] for p in pages]

    x = np.arange(len(ids))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(x - width/2, f1s, width, label="Word F1", color=C_GREEN, alpha=0.85)
    ax.bar(x + width/2, ious, width, label="Mean IoU", color=C_PURPLE, alpha=0.75)
    ax.axhline(y=0.85, color=C_RED, linewidth=1, linestyle=":", label="Paper 1 F1 Target (>0.85)")
    ax.axhline(y=0.70, color=C_ORANGE, linewidth=1, linestyle="--", alpha=0.6, label="Paper 1 IoU Target (>0.70)")
    ax.set_xlabel("Test Page Index")
    ax.set_ylabel("Score")
    ax.set_title("Per-Page Detection Box F1 and Mean IoU", fontsize=10, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(ids, fontsize=6, rotation=45)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_ylim(0.6, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "per_page_detection.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "per_page_detection.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ per_page_detection")

# ═══════════════════════════════════════════════════════
# Fig 7 – Radar / Spider Chart: Multi-Metric Summary
# ═══════════════════════════════════════════════════════
def fig_radar():
    categories = ["CER\n(1−err)", "WER\n(1−err)", "Box F1", "IoU", "SSIM", "TBR-L"]
    ours = [1-0.0286, 1-0.0456, 0.9575, 0.8659, 0.9460, 0.9850]
    paper = [1-0.050, 1-0.080, 0.8500, 0.7000, 0.880, 0.800]

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    ours += ours[:1]
    paper += paper[:1]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    ax.plot(angles, ours, 'o-', color=C_BLUE, linewidth=2, markersize=5, label="PageOCRDiff (Ours)")
    ax.fill(angles, ours, alpha=0.15, color=C_BLUE)
    ax.plot(angles, paper, 's--', color=C_RED, linewidth=1.5, markersize=4, label="IEEE Paper Baseline")
    ax.fill(angles, paper, alpha=0.1, color=C_RED)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=8)
    ax.set_ylim(0.6, 1.02)
    ax.set_title("Multi-Metric Performance Radar", fontsize=10, fontweight="bold", pad=20)
    ax.legend(fontsize=7, loc="lower right", bbox_to_anchor=(1.3, 0))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "radar_chart.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "radar_chart.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ radar_chart")

# ═══════════════════════════════════════════════════════
# Fig 8 – Architecture Block Diagram (using matplotlib patches)
# ═══════════════════════════════════════════════════════
def fig_architecture():
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')

    box_style = dict(boxstyle="round,pad=0.4", edgecolor="#374151", linewidth=1.5)
    colors = {
        'input': '#DBEAFE', 'detector': '#BBF7D0', 'layout': '#FDE68A',
        'diffusion': '#E9D5FF', 'recog': '#FECACA', 'output': '#D1FAE5'
    }

    boxes = [
        (5, 6.3, "Camera-Captured\nDocument Image", colors['input']),
        (5, 5.2, "Four-Head Detector\n(Region, Affinity, Orientation, Scale)\nResNet-18 + U-Net Decoder", colors['detector']),
        (2.5, 3.9, "CC Extraction &\nDelaunay Block\nSegmentation (ε=2.0)", colors['layout']),
        (7.5, 3.9, "Curvilinear Polynomial\nLine Fitting (k≤4)\nXY-Cut Reading Order", colors['layout']),
        (2.5, 2.5, "OCR-Diff Diffusion SR\nConditional U-Net\nLinear Attention O(N)\n10-Step DDIM", colors['diffusion']),
        (7.5, 2.5, "BiLSTM + CTC\nRecognizer\nBeam Search Decoding", colors['recog']),
        (5, 1.2, "WordNinja Post-Processing\nLexicon Correction & Confidence Filter", colors['recog']),
        (5, 0.2, "Structured Text Output\n(Reading Order Preserved)", colors['output']),
    ]

    for x, y, text, color in boxes:
        bbox = dict(box_style, facecolor=color)
        ax.text(x, y, text, ha='center', va='center', fontsize=7,
                fontweight='bold', bbox=bbox, family='serif')

    # Arrows
    arrows = [
        (5, 5.95, 5, 5.65), (5, 4.85, 2.5, 4.35), (5, 4.85, 7.5, 4.35),
        (2.5, 3.5, 2.5, 3.0), (7.5, 3.5, 7.5, 3.0),
        (2.5, 2.0, 5, 1.55), (7.5, 2.0, 5, 1.55),
        (5, 0.85, 5, 0.52),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#374151', lw=1.5))

    ax.set_title("PageOCRDiff System Architecture", fontsize=12, fontweight='bold', pad=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "architecture.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG_DIR, "architecture.png"), bbox_inches="tight")
    plt.close(fig)
    print("  ✓ architecture")


if __name__ == "__main__":
    print("Generating figures...")
    fig_detector_loss()
    fig_recognizer_loss()
    fig_diffusion_loss()
    fig_per_page_metrics()
    fig_benchmark_comparison()
    fig_per_page_detection()
    fig_radar()
    fig_architecture()
    print("\nAll figures generated in:", FIG_DIR)
