# Page-Level OCR-Diff: Unified Generative Diffusion & Document Layout Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![IEEE IoTJ 2024](https://img.shields.io/badge/IEEE%20IoTJ-2024-green.svg)](https://doi.org/10.1109/JIOT.2024.3390700)
[![IEEE Access 2025](https://img.shields.io/badge/IEEE%20Access-2025-green.svg)](https://doi.org/10.1109/ACCESS.2025.3572001)

A state-of-the-art Optical Character Recognition (OCR) and layout reconstruction framework combining two cutting-edge research paradigms:

1. **OCR-Diff (IEEE Internet of Things Journal, 2024)**:  
   *“OCR-Diff: A Two-Stage Deep Learning Framework for Optical Character Recognition Using Diffusion Model in Industrial Internet of Things”*  
   *(Chae-Won Park, Vikas Palakonda, Sangseok Yun, Il-Min Kim, and Jae-Mo Kang)*
2. **Page-Level Document Recognition (IEEE Access, 2025)**:  
   *“Development of OCR Service for Page-Level Recognition for Camera-Captured Document Images”*  
   *(Junyoung Park, Wonjun Kang, Seonji Park, Keuntek Lee, Hyung Il Koo, and Nam Ik Cho)*

---

## 🌟 Key Architecture & Contributions

```
                    [Camera-Captured / Low-Resolution / Warped Document]
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │       FourTensorNet (IEEE Access 2025 Backbone)       │
                 │   Outputs: Affinity (A), Region (R),                   │
                 │            Orientation (32 Bins), Scale (10 Bins)      │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                        [Connected Component (CC) Extraction]
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │        Text-Block Segmentation via Delaunay Graph      │
                 │   Pruning condition: d_pq >= eps * min(s_p, s_q)       │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │        Curvilinear Polynomial Text-Line Detection      │
                 │   k <= 4 polynomial fitting: RMSE <= s_bar / 4         │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │         Topological Reading Order Resolution           │
                 │   Intra-line: x' sort | Inter-line: f_i(x_avg) sort    │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │          OCR-Diff Enhancement (IEEE IoTJ 2024)         │
                 │   Conditional U-Net with Linear Attention O(N)         │
                 │   Reverse Diffusion Residual Learning: X_hat = X_0 + x │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                 ┌────────────────────────────────────────────────────────┐
                 │        Recognition & Linguistic Post-Processing        │
                 │   - CTC / Attention Recognition                        │
                 │   - WordNinja unigram frequency word splitting         │
                 │   - Bleed-through confidence threshold rejection       │
                 └───────────────────────────┬────────────────────────────┘
                                             │
                                             ▼
                    [LLM-Ready Reading Order Text, Markdown & JSON]
```

### 1. OCR-Diff: Generative Diffusion Text Super-Resolution
- **Feature Extractor**: 5 stacked residual blocks with global skip connection yielding text representation $x_f \in \mathbb{R}^{W \times H \times D}$.
- **Customized Conditional U-Net**:
  - Incorporates **Linear Attention** ($O(N)$ spatial complexity) at the bottleneck to capture long-range stroke dependencies.
  - Time embedding vector $\tau_t \in \mathbb{R}^{2K}$ ($K=64$) through a 2-layer MLP.
  - Forward corruption with cosine beta schedule: $X_t = \sqrt{\bar{\alpha}_t}X + \sqrt{1 - \bar{\alpha}_t}E$.
  - Reverse sampling residual learning: $\hat{X} = \hat{X}_0 + x_{up}$.
  - Two-stage training: MSE pre-training loss $\to$ Recognizer-guided fine-tuning loss.

### 2. Page-Level Recognition for Camera-Captured Images
- **4-Tensor Deep Network**: Predicts Region score ($R$), Affinity score ($A$), 32-bin Orientation ($O$), and 10-bin Scale ($S$).
- **Combined Loss**:
  $$\mathcal{L} = \mathcal{L}_r + \mathcal{L}_a + \lambda_s \mathcal{L}_s + \lambda_o \mathcal{L}_o$$
  with $\lambda_s = 0.1, \lambda_o = 0.3$.
- **Delaunay Triangulation Block Segmentation**:
  $$d_{pq} \ge \epsilon \times \min(s_p, s_q) \quad (\epsilon = 2)$$
- **Curvilinear Polynomial Text-Line Fitting**:
  $$\sqrt{\min_f \frac{1}{|T|} \sum_{c_p \in T} (y'_p - f(x'_p))^2} \le \frac{\bar{s}}{4}$$
- **Topological Reading Order**: Evaluates polynomial height at block center $x^i_{avg} = \frac{1}{|C_i|} \sum x'_p$, sorting lines by $f_i(x^i_{avg})$.
- **WordNinja Post-Processing & Bleed-Through Rejection**: Corrects glued words and discards reverse-side ghost print.
- **TBR-L (Text-Block ROUGE-L) Metric**:
  $$\text{TBR-L} = \max_{\pi \in \mathcal{P}} \frac{\text{LCS}(F, \pi(B_1, \dots, B_n))}{|\pi(B_1, \dots, B_n)|}$$

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/khuranatejas10-bot/page-level-ocr-diff.git
cd page-level-ocr-diff

# Install dependencies
pip install -r requirements.txt
```

### CLI Document Extraction

Extract structured text from any camera-captured image, receipt, or book page:

```bash
# Basic extraction with reading order preserved
python cli.py --image path/to/document.jpg --output results.json --save-vis vis.jpg

# Enable high-fidelity diffusion super-resolution (e.g. 5 steps)
python cli.py --image path/to/document.jpg --steps 5 --save-vis vis.jpg

# Evaluate with Ground-Truth text for TBR-L, CER, and Edit Distance
python cli.py --image path/to/document.jpg --gt ground_truth.txt
```

### Interactive Web Dashboard

Launch the Flask web server with real-time 4-tensor visualization, Delaunay layout graphs, and diffusion before/after comparisons:

```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 🧪 Verification & Test Suite

Run the complete unit and mathematical verification test suite:

```bash
python test_unified_system.py
```

All 10 unit tests verify:
- Cosine schedule, forward diffusion corruption, and reverse residual sampling.
- Linear attention $O(N)$ projection invariance.
- 4-tensor network forward pass and masked cross-entropy loss.
- Delaunay triangulation scale-adaptive edge pruning.
- Curvilinear 4th-order polynomial line fitting and topological reading order sorting.
- WordNinja splitting and bleed-through rejection thresholding.
- Levenshtein Edit Distance, CER, and Text-Block ROUGE-L (TBR-L) calculations.

---

## 📁 Repository Structure

```
├── ocr_diff.py              # IEEE IoTJ 2024: OCR-Diff Conditional U-Net, Linear Attention & Diffusion
├── page_ocr.py              # IEEE Access 2025: 4-Tensor Net, Delaunay Triangulation, Curvilinear Lines
├── metrics.py               # TBR-L, Character Error Rate (CER), Levenshtein Edit Distance
├── unified_ocr.py           # Unified Master Engine integrating both research frameworks
├── cli.py                   # Command-line interface with JSON/Vis export & metrics
├── app.py                   # Web dashboard & REST API server
├── test_unified_system.py   # Unit test suite verifying mathematical formulations
├── templates/               # Web application templates
├── static/                  # Styles, scripts, and UI assets
└── requirements.txt         # Project dependencies
```

---

## 📖 Citation

If you use this codebase or methodology, please cite the underlying research papers:

```bibtex
@article{park2024ocrdiff,
  title={OCR-Diff: A Two-Stage Deep Learning Framework for Optical Character Recognition Using Diffusion Model in Industrial Internet of Things},
  author={Park, Chae-Won and Palakonda, Vikas and Yun, Sangseok and Kim, Il-Min and Kang, Jae-Mo},
  journal={IEEE Internet of Things Journal},
  volume={11},
  number={15},
  pages={25997--26000},
  year={2024},
  publisher={IEEE}
}

@article{park2025pagelevel,
  title={Development of OCR Service for Page-Level Recognition for Camera-Captured Document Images},
  author={Park, Junyoung and Kang, Wonjun and Park, Seonji and Lee, Keuntek and Koo, Hyung Il and Cho, Nam Ik},
  journal={IEEE Access},
  volume={13},
  pages={91263--91275},
  year={2025},
  publisher={IEEE}
}
```

---

## 📄 License
This project is open-source under the MIT License.
