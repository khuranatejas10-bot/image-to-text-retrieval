# Keyword-Based Image Retrieval System with OCR-Diff Super-Resolution

A lightweight, robust, and highly efficient system for keyword-based image retrieval utilizing **OCR-Diff Generative Diffusion Super-Resolution**, OpenCV-based image preprocessing, deep-learning-based OCR (EasyOCR), rule-based post-correction, and Levenshtein-distance fuzzy string matching.

This project integrates the system architecture described in:
1. **"OCR-Diff: A Two-Stage Deep Learning Framework for Optical Character Recognition Using Diffusion Model in Industrial Internet of Things"** by Chae-Won Park, Vikas Palakonda, Sangseok Yun, Il-Min Kim, and Jae-Mo Kang (*IEEE Internet of Things Journal*, Vol. 11, No. 15, August 2024).
2. **"A Lightweight and Robust System for Keyword-Based Image Retrieval Using OCR and Fuzzy Matching"** by Devishree Naidu, Siddhi Kothekar, Mrunal Labhe, and Adiba Ali (2026).

---

## 🌟 Key Features

*   **OCR-Diff Generative Diffusion Super-Resolution:** Implements a PyTorch-backed two-stage diffusion framework with customized Conditional U-Net, Linear Attention, and Feature Extractor for text image super-resolution ($\hat{X} = \hat{X}_0 + x_{up}$).
*   **Geometric Deskewing (Algorithm 2):** Automatically estimates text orientation using minimum-area bounding rectangles and rotates images via affine transformation.
*   **Horizontal Line Removal (Algorithm 3):** Uses morphological operations (horizontal kernels, erosion, and dilation) to remove underlines, rulings, or borders without damaging text details.
*   **Adaptive Binarization (Algorithm 1):** Applies Gaussian adaptive thresholding to maximize text-to-background contrast under uneven lighting conditions.
*   **Deep Learning OCR (Algorithm 4):** Integrates EasyOCR (built on CRNN + CTC loss) for accurate multilingual word-level bounding box and text extraction.
*   **Rule-Based Post-Correction (Algorithm 5):** Fixes common OCR confusion errors (e.g., mistaken substitutions like `l`, `i`, `—` for `1` or `o`, `O` for `0`).
*   **SHA-256 Deduplication (Algorithm 6):** Computes unique binary hashes for uploaded files, skipping reprocessing of identical documents to save up to 30% computing time.
*   **Fuzzy Keyword Search (Algorithm 7):** Performs word-level tokenization and uses Levenshtein-distance partial matching to retrieve images, effectively handling spelling variations and OCR noise.
*   **Interactive Visual Dashboard:** A high-end dark-mode frontend featuring drag-and-drop batch uploads, OCR-Diff toggle switches, step-by-step visual inspectors for preprocessing stages (OCR-Diff, Deskewed, Line Removed, Binarized), and dynamic canvas bounding box overlays highlighting matched keywords.

---

## 🔬 Mathematical Formulation of OCR-Diff

### 1. Stage 1: Forward Diffusion & Noise Estimation Pretraining
Given ground truth HR text image $X$, upsampled LR text image $x_{up}$, and time step $t$:
$$X_t = \sqrt{\bar{\alpha}_t} X + \sqrt{1 - \bar{\alpha}_t} E, \quad E \sim \mathcal{N}(0, \mathbf{I})$$
Features are extracted via $x_f = f_\phi(x_{up})$ (5 residual blocks with skip connection).
Pretraining loss minimizes mean squared error:
$$\mathcal{L}_{pre} = \| E - E_\theta(X_t, x_f, \tau_t) \|^2$$

Where time embedding $\tau_t \in \mathbb{R}^{2K}$ ($K=64$) is computed via:
$$\tau_t = \left[ \left\{ \sin\left(\frac{t}{10000^{k/(K-1)}}\right) \right\}_{k=1}^K, \left\{ \cos\left(\frac{t}{10000^{k/(K-1)}}\right) \right\}_{k=1}^K \right]$$

### 2. Linear Attention Mechanism
In the middle bottleneck layer of the U-Net, Linear Attention computes:
$$Q_{softmax} = \text{Softmax}(Q), \quad K_{softmax} = \text{Softmax}(K)$$
$$\text{Context} = K_{softmax}^T V, \quad \text{Output} = \gamma (\text{Context} \cdot Q_{softmax}) + X$$
This reduces computational complexity from $\mathcal{O}(N^2)$ to $\mathcal{O}(N)$.

### 3. Stage 2: Reverse Diffusion & Fine-Tuning
Reverse diffusion iteratively reconstructs residual image $\hat{X}_0$:
$$\hat{X}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( \hat{X}_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} E_\theta(\hat{X}_t, x_f, \tau_t) \right) + \sigma_t Z$$
HR text image is reconstructed via residual addition:
$$\hat{X} = \hat{X}_0 + x_{up}$$
Fine-tuning loss combines MSE and frozen recognizer cross-entropy:
$$\mathcal{L}_{fine} = \| X - \hat{X} \|^2 - \lambda \sum_{i=1}^N \omega_i \langle y_i, \log \hat{y}_i \rangle$$

---

## 🛠️ Technology Stack

*   **Backend:** Python 3, Flask (Web Framework)
*   **Deep Learning & Super-Resolution:** PyTorch, torchvision, OCR-Diff Pipeline
*   **Image Processing:** OpenCV, NumPy
*   **Optical Character Recognition:** EasyOCR (PyTorch-backed)
*   **Fuzzy Matching Engine:** RapidFuzz (C-optimized Levenshtein calculations)
*   **Database:** SQLite 3

---

## 📁 Repository Structure

```text
image-to-text-retrieval/
│
├── static/
│   ├── css/
│   │   └── styles.css          # Premium Dark-Mode Glassmorphism Styling
│   └── js/
│       └── main.js            # Drag & drop upload, OCR-Diff toggle, canvas drawing, search, and viewports logic
│
├── templates/
│   └── index.html             # Main Frontend Dashboard UI Template with OCR-Diff Inspector
│
├── uploads/                   # Folder holding original and processed pipeline stage images
│
├── app.py                     # Flask Server containing REST API Endpoints & OCR-Diff status API
├── ocr_diff.py                # PyTorch OCR-Diff (Linear Attention, Conditional U-Net, Feature Extractor, Diffusion Sampler)
├── database.py                # Database connection, schemas, hashing, and search matching
├── pipeline.py                # OpenCV Preprocessing, OCR-Diff Integration, and EasyOCR
├── requirements.txt           # Python Project Dependencies
├── test_pipeline.py           # Automated unit tests covering OCR-Diff, pipeline, and database functions
└── README.md                  # Comprehensive Project Documentation
```

---

## 🚀 Getting Started

### 📋 Prerequisites

*   Python 3.8 or higher installed on your system.
*   GPU acceleration (CUDA) is optional but supported for faster OCR-Diff diffusion sampling and EasyOCR text extraction. The engine falls back automatically to standard CPU execution.

### 🔧 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/khuranatejas10-bot/image-to-text-retrieval.git
    cd image-to-text-retrieval
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    
    # Windows Command Prompt:
    venv\Scripts\activate
    
    # Git Bash / Linux / macOS:
    source venv/bin/activate
    
    # PowerShell:
    venv\Scripts\Activate.ps1
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

---

## 🖥️ Running the Application

1.  **Start the Flask development server:**
    ```bash
    python app.py
    ```

2.  **Access the web portal:**
    Open your browser and navigate to:
    ```text
    http://localhost:5000
    ```

3.  **Basic Workflow:**
    *   Go to **Upload Images** and drag and drop document scans, signs, or low-resolution labels.
    *   Toggle **Enable OCR-Diff Diffusion** to enable generative super-resolution text restoration.
    *   Switch to **Search Index**, input a search term (e.g. "report"), select a fuzzy threshold, and press Enter.
    *   Click on any search card to open the detail modal and view preprocessed steps side-by-side (OCR-Diff SR, Deskewed, Line Removed, Gaussian Binary) or inspect the keyword highlight overlay!

---

## 🧪 Running Unit Tests

Automated tests check OCR-Diff PyTorch modules (Linear Attention, Feature Extractor, U-Net, Diffusion Sampler), binarization, line removal, database metadata saving, SHA-256 cache hits, and Levenshtein fuzzy searches.

Run unit tests via the standard Python unittest runner:
```bash
python -m unittest test_pipeline.py
```

---

## 📖 Citation & References

This implementation is modeled after the algorithms, architectures, and experiments documented in:

> Chae-Won Park, Vikas Palakonda, Sangseok Yun, Il-Min Kim, Jae-Mo Kang. **"OCR-Diff: A Two-Stage Deep Learning Framework for Optical Character Recognition Using Diffusion Model in Industrial Internet of Things"**. *IEEE Internet of Things Journal*, Vol. 11, No. 15, pp. 25997–26000, 1 August 2024.

> Devishree Naidu, Siddhi Kothekar, Mrunal Labhe, Adiba Ali. **"A Lightweight and Robust System for Keyword-Based Image Retrieval Using OCR and Fuzzy Matching"**. *International Conference on Emerging Trends and Innovations in ICT (ICEI)*, 2026.

Key algorithms adapted:
1.  **OCR-Diff Diffusion Model:** Two-stage residual diffusion model for text image super-resolution ($\hat{X} = \hat{X}_0 + x_{up}$).
2.  **Linear Attention Module:** $\mathcal{O}(N)$ complexity attention mechanism embedded in U-Net bottleneck.
3.  **Algorithm 1:** Image Preprocessing (Adaptive Thresholding)
4.  **Algorithm 2:** Deskewing Algorithm (Affine text line rotation)
5.  **Algorithm 3:** Line Removal (Horizontal structuring element morphology)
6.  **Algorithm 4:** OCR using EasyOCR (CRNN + CTC neural architecture)
7.  **Algorithm 5:** OCR Error Correction (Rule-based normalization dictionary)
8.  **Algorithm 6:** SHA-256 Hash for Image Deduplication (Binary file hashing)
9.  **Algorithm 7:** Keyword Search with Fuzzy Matching (Partial Levenshtein ratio matching)

