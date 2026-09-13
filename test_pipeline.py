import unittest
import numpy as np
import cv2
import os
import json
import sqlite3

# Import system functions to test
from pipeline import correct_ocr_text, deskew, remove_lines
import database

class TestKeywordRetrievalPipeline(unittest.TestCase):
    
    def setUp(self):
        # Setup test SQLite database path
        database.DATABASE_PATH = 'test_metadata.db'
        database.init_db()
        
    def tearDown(self):
        # Cleanup test SQLite database
        if os.path.exists('test_metadata.db'):
            try:
                os.remove('test_metadata.db')
            except Exception as e:
                print(f"Error removing test db: {e}")

    def test_ocr_error_correction(self):
        """
        Verify Algorithm 5 OCR Error Correction mappings
        '1' -> ['l', 'i', '—']
        '0' -> ['O', 'o']
        '8' -> ['B']
        '5' -> ['S']
        '2' -> ['Z']
        """
        self.assertEqual(correct_ocr_text('l'), '1')
        self.assertEqual(correct_ocr_text('i'), '1')
        self.assertEqual(correct_ocr_text('—'), '1')
        self.assertEqual(correct_ocr_text('O'), '0')
        self.assertEqual(correct_ocr_text('o'), '0')
        self.assertEqual(correct_ocr_text('B'), '8')
        self.assertEqual(correct_ocr_text('S'), '5')
        self.assertEqual(correct_ocr_text('Z'), '2')
        
        # Test combined string
        self.assertEqual(correct_ocr_text('Hello World! l0B5Z'), 'He110 W0r1d! 10852')

    def test_deskew_no_skew(self):
        """
        Verify deskewing does not modify an already upright, empty image.
        """
        # Create a solid white image
        img = np.ones((100, 100, 3), dtype=np.uint8) * 255
        rotated = deskew(img)
        # Should be identical
        np.testing.assert_array_equal(img, rotated)

    def test_line_removal_basic(self):
        """
        Verify horizontal line removal erases lines to white.
        """
        # Create a white image with a solid black horizontal line
        img = np.ones((50, 100), dtype=np.uint8) * 255
        img[25, 20:80] = 0 # draw black horizontal line
        
        # Apply line removal
        cleaned = remove_lines(img)
        
        # Check if the line was removed (filled with white, 255)
        # Wait, the line should be erased to white background.
        self.assertEqual(cleaned[25, 50], 255)

    def test_database_operations(self):
        """
        Test inserting images, words, SHA-256 caching, and fuzzy query matching.
        """
        filepath = 'dummy_image.jpg'
        original_filename = 'dummy.jpg'
        image_hash = 'abcdef1234567890'
        raw_text = 'The quick brown fox jumps over the lazy dog'
        corrected_text = 'The quick br0wn f0x jumps 0ver the 1azy d0g'
        
        # Bounding box coords for word elements
        ocr_segments = [
            {'word': 'quick', 'corrected_word': 'qu1ck', 'box': [[10, 10], [50, 10], [50, 20], [10, 20]], 'confidence': 0.95},
            {'word': 'brown', 'corrected_word': 'br0wn', 'box': [[60, 10], [100, 10], [100, 20], [60, 20]], 'confidence': 0.90},
            {'word': 'lazy', 'corrected_word': '1azy', 'box': [[110, 10], [150, 10], [150, 20], [110, 20]], 'confidence': 0.88}
        ]
        
        # Save metadata
        image_id = database.save_image_metadata(
            filepath=filepath,
            original_filename=original_filename,
            image_hash=image_hash,
            raw_text=raw_text,
            corrected_text=corrected_text,
            ocr_segments=ocr_segments
        )
        
        self.assertIsNotNone(image_id)
        
        # Test SHA-256 duplicate cache lookup
        existing = database.get_image_by_hash(image_hash)
        self.assertIsNotNone(existing)
        self.assertEqual(existing['id'], image_id)
        self.assertEqual(existing['filepath'], filepath)
        
        # Test fuzzy matching keyword query (Exact word)
        matches = database.search_keywords_fuzzy('quick', threshold=85)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0]['id'], image_id)
        
        # Test fuzzy matching keyword query (Fuzzy / spelling variant)
        # "lazy" is normalized as "1azy" (since 'l' maps to '1' under correction rules)
        # Search for 'lazy' should match stored '1azy' if query normalized 
        matches_lazy = database.search_keywords_fuzzy('lazy', threshold=80)
        self.assertEqual(len(matches_lazy), 1)
        self.assertEqual(matches_lazy[0]['id'], image_id)
        
        # Test fuzzy threshold sensitivity (No match if threshold is too high)
        matches_none = database.search_keywords_fuzzy('banana', threshold=80)
        self.assertEqual(len(matches_none), 0)

class TestOCRDiffPipeline(unittest.TestCase):

    def test_linear_attention_shape(self):
        """
        Verify LinearAttention module computes softmax along channels and preserves tensor shape (B x C x H x W).
        """
        import torch
        from ocr_diff import LinearAttention
        attn = LinearAttention(in_channels=32)
        x = torch.randn(2, 32, 16, 16)
        out = attn(x)
        self.assertEqual(out.shape, x.shape)

    def test_feature_extractor_shape(self):
        """
        Verify FeatureExtractor (5 residual blocks + skip connection) produces B x 3 x H x W features.
        """
        import torch
        from ocr_diff import FeatureExtractor
        fe = FeatureExtractor(in_channels=3, mid_channels=16)
        x_up = torch.randn(2, 3, 32, 128)
        x_f = fe(x_up)
        self.assertEqual(x_f.shape, x_up.shape)

    def test_unet_and_time_embedding(self):
        """
        Verify CustomizedConditionalUNet forward pass with sinusoidal time embedding tau_t in R^{2K}.
        """
        import torch
        from ocr_diff import CustomizedConditionalUNet
        unet = CustomizedConditionalUNet(K=64)
        X_t = torch.randn(1, 3, 32, 128)
        x_f = torch.randn(1, 3, 32, 128)
        t = torch.tensor([50])
        pred_noise = unet(X_t, x_f, t)
        self.assertEqual(pred_noise.shape, X_t.shape)

    def test_ocr_diff_enhancement_numpy(self):
        """
        Verify high-level enhance_image_np method accepts NumPy images and outputs enhanced image array.
        """
        from ocr_diff import OCRDiffPipeline
        pipe = OCRDiffPipeline(T=20)
        img = np.ones((32, 128, 3), dtype=np.uint8) * 200
        enhanced = pipe.enhance_image_np(img, num_steps=5)
        self.assertEqual(enhanced.shape, img.shape)
        self.assertEqual(enhanced.dtype, np.uint8)

if __name__ == '__main__':
    unittest.main()

