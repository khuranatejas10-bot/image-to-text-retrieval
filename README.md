# Keyword-Based Image Retrieval System

A lightweight, robust, and highly efficient system for keyword-based image retrieval utilizing OpenCV-based image preprocessing, deep-learning-based OCR (EasyOCR), rule-based post-correction, and Levenshtein-distance fuzzy string matching.

This project is a complete, production-ready implementation of the system architecture described in the research paper: **"A Lightweight and Robust System for Keyword-Based Image Retrieval Using OCR and Fuzzy Matching"** by Devishree Naidu, Siddhi Kothekar, Mrunal Labhe, and Adiba Ali (2026).

---

## 🌟 Key Features

*   **Geometric Deskewing (Algorithm 2):** Automatically estimates text orientation using minimum-area bounding rectangles and rotates images via affine transformation.
*   **Horizontal Line Removal (Algorithm 3):** Uses morphological operations (horizontal kernels, erosion, and dilation) to remove underlines, rulings, or borders without damaging text details.
*   **Adaptive Binarization (Algorithm 1):** Applies Gaussian adaptive thresholding to maximize text-to-background contrast under uneven lighting conditions.
*   **Deep Learning OCR (Algorithm 4):** Integrates EasyOCR (built on CRNN + CTC loss) for accurate multilingual word-level bounding box and text extraction.
*   **Rule-Based Post-Correction (Algorithm 5):** Fixes common OCR confusion errors (e.g., mistaken substitutions like `l`, `i`, `—` for `1` or `o`, `O` for `0`).
*   **SHA-256 Deduplication (Algorithm 6):** Computes unique binary hashes for uploaded files, skipping reprocessing of identical documents to save up to 30% computing time.
*   **Fuzzy Keyword Search (Algorithm 7):** Performs word-level tokenization and uses Levenshtein-distance partial matching to retrieve images, effectively handling spelling variations and OCR noise.
*   **Interactive Visual Dashboard:** A high-end dark-mode frontend featuring drag-and-drop batch uploads, fuzzy similarity sliders, step-by-step visual inspectors for preprocessing stages, and dynamic canvas bounding box overlays highlighting matched keywords.

---

## 🛠️ Technology Stack

*   **Backend:** Python 3, Flask (Web Framework)
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
│       └── main.js            # Drag & drop upload, canvas drawing, search, and viewports logic
│
├── templates/
│   └── index.html             # Main Frontend Dashboard UI Template
│
├── uploads/                   # Folder holding original and processed pipeline stage images
│
├── app.py                     # Flask Server containing REST API Endpoints
├── database.py                # Database connection, schemas, hashing, and search matching
├── pipeline.py                # OpenCV Preprocessing and EasyOCR Integration
├── requirements.txt           # Python Project Dependencies
├── test_pipeline.py           # Automated unit tests covering pipeline and database functions
└── README.md                  # Comprehensive Project Documentation
```

---

## 🚀 Getting Started

### 📋 Prerequisites

*   Python 3.8 or higher installed on your system.
*   GPU acceleration (CUDA) is optional but recommended to speed up EasyOCR text extraction. The engine will fallback automatically to standard CPU execution.

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
    *   Go to **Upload Images** and drag and drop document scans, signs, or handwritten labels.
    *   Wait for the processing queue to complete (EasyOCR models will download on their first run).
    *   Switch to **Search Index**, input a search term (e.g. "report"), select a fuzzy threshold, and press Enter.
    *   Click on any search card to open the detail modal and view preprocessed steps side-by-side or inspect the keyword highlight overlay!

---

## 🧪 Running Unit Tests

Automated tests check binarization, line removal, database metadata saving, SHA-256 cache hits, and Levenshtein fuzzy searches.

Run unit tests via the standard Python unittest runner:
```bash
python -m unittest test_pipeline.py
```

---

## 📖 Citation & References

This implementation is modeled after the pipeline, algorithms, and experiments documented in:

> Devishree Naidu, Siddhi Kothekar, Mrunal Labhe, Adiba Ali. "A Lightweight and Robust System for Keyword-Based Image Retrieval Using OCR and Fuzzy Matching". *International Conference on Emerging Trends and Innovations in ICT (ICEI)*, 2026.

Key algorithms adapted:
1.  **Algorithm 1:** Image Preprocessing (Adaptive Thresholding)
2.  **Algorithm 2:** Deskewing Algorithm (Affine text line rotation)
3.  **Algorithm 3:** Line Removal (Horizontal structuring element morphology)
4.  **Algorithm 4:** OCR using EasyOCR (CRNN + CTC neural architecture)
5.  **Algorithm 5:** OCR Error Correction (Rule-based normalization dictionary)
6.  **Algorithm 6:** SHA-256 Hash for Image Deduplication (Binary file hashing)
7.  **Algorithm 7:** Keyword Search with Fuzzy Matching (Partial Levenshtein ratio matching)
