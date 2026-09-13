import cv2
import numpy as np
import easyocr
import os
from ocr_diff import get_ocr_diff_pipeline

def deskew(image):
    """
    Algorithm 2: Deskewing Algorithm
    Rotates the image to align text horizontally.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # coords <- all pixel coordinates where pixel < 255
    y, x = np.where(gray < 255)
    coords = np.column_stack((x, y))
    
    if len(coords) == 0:
        return image
    
    # angle <- minimum area rectangle angle from coords
    # cv2.minAreaRect returns: (center(x, y), size(w, h), angle)
    rect = cv2.minAreaRect(coords.astype(np.float32))
    angle = rect[-1]
    
    # Algorithm 2 angle correction logic:
    # Under older OpenCV versions, angle is in [-90, 0)
    # Under newer OpenCV versions, angle is in [0, 90]
    # Let's adjust standard angle behavior to align with the paper's logic
    if angle < -45:
        angle = -(90 + angle)
    elif angle > 45:
        angle = 90 - angle
    else:
        angle = -angle
        
    # Prevent rotation if the angle is negligible (e.g., < 0.5 degrees)
    if abs(angle) < 0.5:
        return image
        
    (h, w) = image.shape[:2]
    # M <- rotation matrix with center (w/2, h/2) and angle
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    # return rotated image using M
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def remove_lines(image):
    """
    Algorithm 3: Line Removal
    Removes horizontal lines from the image.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
        
    # thresh <- binary inverse threshold on gray with value 200
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # kernel <- horizontal kernel of size (1, 15)
    # In OpenCV, cv2.getStructuringElement uses (width, height), so (15, 1) represents a width of 15 and height of 1.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
    
    # horizontal <- erode thresh using kernel
    horizontal = cv2.erode(thresh, kernel, iterations=1)
    # horizontal <- dilate horizontal using kernel
    horizontal = cv2.dilate(horizontal, kernel, iterations=1)
    
    # Invert horizontal line mask
    # return bitwise AND of original image with inverse of horizontal as mask
    # To properly erase the lines (fill with white background instead of black),
    # we replace pixels where the horizontal mask is active with white (255).
    cleaned = image.copy()
    if len(image.shape) == 3:
        cleaned[horizontal > 0] = [255, 255, 255]
    else:
        cleaned[horizontal > 0] = 255
        
    return cleaned

def preprocess_image(image_path, use_ocr_diff=True):
    """
    Algorithm 1: Image Preprocessing with OCR-Diff Generative Diffusion Integration
    Applies optional OCR-Diff super-resolution, deskewing, line removal, and adaptive thresholding.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
        
    # 1. OCR-Diff Generative Diffusion Super Resolution (IEEE IoTJ 2024 Paper)
    if use_ocr_diff:
        ocr_diff_pipe = get_ocr_diff_pipeline()
        ocr_diff_enhanced = ocr_diff_pipe.enhance_image_np(image, num_steps=10)
    else:
        ocr_diff_enhanced = image.copy()
        
    # 2: image <- DESKEW(image)
    deskewed = deskew(ocr_diff_enhanced)
    
    # 3: image <- REMOVE_LINES(image)
    no_lines = remove_lines(deskewed)
    
    # 4: gray <- convert image to grayscale
    if len(no_lines.shape) == 3:
        gray = cv2.cvtColor(no_lines, cv2.COLOR_BGR2GRAY)
    else:
        gray = no_lines.copy()
        
    # 5: return adaptive threshold of gray image using Gaussian method
    preprocessed = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    return preprocessed, deskewed, no_lines, ocr_diff_enhanced

def correct_ocr_text(text):
    """
    Algorithm 5: OCR Error Correction (Rule-based)
    Corrects common OCR character confusions.
    """
    mapping = {
        '1': ['l', 'i', '—'],
        '0': ['O', 'o'],
        '8': ['B'],
        '5': ['S'],
        '2': ['Z']
    }
    
    corrected_text = text
    for correct, wrongs in mapping.items():
        for wrong in wrongs:
            corrected_text = corrected_text.replace(wrong, correct)
    return corrected_text

class OCRExtractor:
    """
    Algorithm 4: OCR using EasyOCR
    Extracts text and bounding box locations from preprocessed images.
    """
    def __init__(self):
        # Initialize reader with English language
        # EasyOCR will automatically detect and download PyTorch and models if needed.
        self.reader = easyocr.Reader(['en'], gpu=False, verbose=False) # Fallback to CPU by default

    def extract_text(self, processed_image):
        """
        Extracts words, bounding boxes, and confidence levels.
        """
        # EasyOCR readtext accepts numpy arrays directly
        # Returns list of tuples: (bbox, text, confidence)
        results = self.reader.readtext(processed_image)
        
        extracted_segments = []
        full_text_list = []
        
        for bbox, text, confidence in results:
            # bbox is list of 4 points: [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
            # convert coordinates to float/int lists
            box_coords = [[int(pt[0]), int(pt[1])] for pt in bbox]
            
            corrected_text = correct_ocr_text(text)
            
            extracted_segments.append({
                'word': text,
                'corrected_word': corrected_text,
                'box': box_coords,
                'confidence': float(confidence)
            })
            full_text_list.append(corrected_text)
            
        full_text = " ".join(full_text_list)
        return full_text, extracted_segments
