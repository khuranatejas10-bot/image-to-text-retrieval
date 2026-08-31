import sqlite3
import hashlib
import json
import os
from rapidfuzz import fuzz
from pipeline import correct_ocr_text

DATABASE_PATH = 'metadata.db'

def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the SQLite database with required tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create images table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            hash TEXT UNIQUE NOT NULL,
            raw_text TEXT,
            corrected_text TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create ocr_words table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS ocr_words (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id INTEGER NOT NULL,
            word TEXT NOT NULL,
            corrected_word TEXT NOT NULL,
            box_json TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (image_id) REFERENCES images (id) ON DELETE CASCADE
        )
    ''')
    
    conn.commit()
    conn.close()

def calculate_sha256(filepath):
    """
    Algorithm 6: SHA-256 Hash for Image Deduplication
    Computes SHA-256 hash of an image.
    """
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        # Read in blocks to handle large files efficiently
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_image_by_hash(image_hash):
    """
    Check if an image hash already exists in the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images WHERE hash = ?", (image_hash,))
    row = cursor.fetchone()
    conn.close()
    return row

def save_image_metadata(filepath, original_filename, image_hash, raw_text, corrected_text, ocr_segments):
    """
    Saves extracted OCR text and segment coordinates to SQLite database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Insert main image record
        cursor.execute(
            "INSERT INTO images (filepath, original_filename, hash, raw_text, corrected_text) VALUES (?, ?, ?, ?, ?)",
            (filepath, original_filename, image_hash, raw_text, corrected_text)
        )
        image_id = cursor.lastrowid
        
        # Insert individual word records
        for segment in ocr_segments:
            cursor.execute(
                "INSERT INTO ocr_words (image_id, word, corrected_word, box_json, confidence) VALUES (?, ?, ?, ?, ?)",
                (
                    image_id,
                    segment['word'],
                    segment['corrected_word'],
                    json.dumps(segment['box']),
                    segment['confidence']
                )
            )
            
        conn.commit()
        return image_id
    except sqlite3.IntegrityError:
        conn.rollback()
        # In case of hash race condition, fetch the existing record
        cursor.execute("SELECT id FROM images WHERE hash = ?", (image_hash,))
        row = cursor.fetchone()
        return row['id'] if row else None
    finally:
        conn.close()

def search_keywords_fuzzy(keyword, threshold=80.0):
    """
    Algorithm 7: Keyword Search with Fuzzy Matching
    Searches the SQLite database using partial Levenshtein distance matching.
    """
    # Apply OCR error correction rules to the search keyword so that it is normalized 
    # identically to the stored database indices
    normalized_keyword = correct_ocr_text(keyword.strip().lower())
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Retrieve all indexed images
    cursor.execute("SELECT id, filepath, original_filename, raw_text, corrected_text FROM images")
    images = cursor.fetchall()
    
    matched_results = []
    
    for img in images:
        image_id = img['id']
        
        # Fetch all individual words/segments for this image
        cursor.execute("SELECT word, corrected_word, box_json, confidence FROM ocr_words WHERE image_id = ?", (image_id,))
        words_rows = cursor.fetchall()
        
        max_score = 0.0
        matched_bounding_boxes = []
        
        for w_row in words_rows:
            # We match using lowercase corrected text to make the search robust 
            # and leverage the rule-based error corrections
            stored_word = w_row['corrected_word'].lower()
            
            # Use partial_ratio to check if the keyword matches part of the word
            score = fuzz.partial_ratio(normalized_keyword, stored_word)
            
            if score >= threshold:
                if score > max_score:
                    max_score = score
                
                # Retrieve word box and convert back to Python list
                box = json.loads(w_row['box_json'])
                matched_bounding_boxes.append({
                    'word': w_row['word'],
                    'corrected_word': w_row['corrected_word'],
                    'box': box,
                    'confidence': w_row['confidence'],
                    'score': score
                })
        
        if max_score >= threshold:
            matched_results.append({
                'id': image_id,
                'filepath': img['filepath'],
                'original_filename': img['original_filename'],
                'raw_text': img['raw_text'],
                'corrected_text': img['corrected_text'],
                'score': round(max_score, 2),
                'matched_words': matched_bounding_boxes
            })
            
    conn.close()
    
    # Sort results by similarity score descending
    matched_results.sort(key=lambda x: x['score'], reverse=True)
    return matched_results

def get_all_images():
    """
    Retrieves all images from the database.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM images ORDER BY created_at DESC")
    rows = cursor.fetchall()
    
    images = []
    for row in rows:
        # Get count of words indexed for this image
        cursor.execute("SELECT COUNT(*) as count FROM ocr_words WHERE image_id = ?", (row['id'],))
        word_count = cursor.fetchone()['count']
        
        images.append({
            'id': row['id'],
            'filepath': row['filepath'],
            'original_filename': row['original_filename'],
            'hash': row['hash'],
            'raw_text': row['raw_text'],
            'corrected_text': row['corrected_text'],
            'created_at': row['created_at'],
            'word_count': word_count
        })
    conn.close()
    return images

def delete_image_by_id(image_id):
    """
    Deletes an image and its associated files and database records.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Fetch filepath to delete local file
    cursor.execute("SELECT filepath FROM images WHERE id = ?", (image_id,))
    row = cursor.fetchone()
    if row:
        filepath = row['filepath']
        # Also check preprocessed/deskewed paths if saved
        base, ext = os.path.splitext(filepath)
        preprocessed_path = f"{base}_preprocessed{ext}"
        deskewed_path = f"{base}_deskewed{ext}"
        no_lines_path = f"{base}_nolines{ext}"
        
        for path in [filepath, preprocessed_path, deskewed_path, no_lines_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    print(f"Error removing file {path}: {e}")
                    
        cursor.execute("DELETE FROM images WHERE id = ?", (image_id,))
        conn.commit()
        success = True
    else:
        success = False
        
    conn.close()
    return success
