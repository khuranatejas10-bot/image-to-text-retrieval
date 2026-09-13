import os
import json
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
import cv2

from database import (
    init_db,
    calculate_sha256,
    get_image_by_hash,
    save_image_metadata,
    search_keywords_fuzzy,
    get_all_images,
    delete_image_by_id
)
from pipeline import preprocess_image, OCRExtractor

# Initialize Flask app
app = Flask(__name__)

# Configure Upload Folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'css'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static', 'js'), exist_ok=True)

# Initialize database
init_db()

# Lazy initialization of EasyOCR extractor to ensure instant web server startup
_ocr_extractor = None

def get_ocr_extractor():
    global _ocr_extractor
    if _ocr_extractor is None:
        _ocr_extractor = OCRExtractor()
    return _ocr_extractor

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'bmp', 'tiff', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """
    Renders the homepage.
    """
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_images():
    """
    Uploads and processes one or more images.
    """
    if 'files' not in request.files and 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
        
    uploaded_files = request.files.getlist('files') or request.files.getlist('file')
    results = []
    
    for file in uploaded_files:
        if file.filename == '':
            continue
            
        if not allowed_file(file.filename):
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': 'File extension not allowed'
            })
            continue
            
        try:
            # 1. Save file temporarily to compute hash
            original_filename = secure_filename(file.filename)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_{original_filename}")
            file.save(temp_path)
            
            # 2. Compute SHA-256 hash (Algorithm 6)
            img_hash = calculate_sha256(temp_path)
            
            # 3. Check for duplicates
            existing_image = get_image_by_hash(img_hash)
            if existing_image:
                # Remove temporary file
                os.remove(temp_path)
                
                # Retrieve word count
                conn = sqlite3_conn = database_conn = sqlite3_connect = None
                # Fetch word details for highlighting
                db_results = search_keywords_fuzzy("", threshold=0)  # threshold 0 returns all, but let's just use search fuzzy or fetch
                # Wait, we can just load the duplicate metadata
                base, ext = os.path.splitext(existing_image['filepath'])
                rel_path = os.path.relpath(existing_image['filepath'], app.root_path).replace('\\', '/')
                
                results.append({
                    'id': existing_image['id'],
                    'filename': original_filename,
                    'status': 'cached',
                    'message': 'Duplicate detected; loaded from cache.',
                    'filepath': rel_path,
                    'hash': img_hash,
                    'raw_text': existing_image['raw_text'],
                    'corrected_text': existing_image['corrected_text']
                })
                continue
                
            # Retrieve OCR-Diff flag from request form (default True)
            use_ocr_diff_str = request.form.get('use_ocr_diff', 'true')
            use_ocr_diff = use_ocr_diff_str.lower() in ['true', '1', 'yes', 'on']

            # Rename temp file to permanent hash-based name to avoid collisions
            _, ext = os.path.splitext(original_filename)
            permanent_filename = f"{img_hash}{ext}"
            permanent_filepath = os.path.join(app.config['UPLOAD_FOLDER'], permanent_filename)
            os.rename(temp_path, permanent_filepath)
            
            # 4. Preprocess image step-by-step (Algorithm 1 + OCR-Diff Super Resolution)
            # Returns binarized preprocessed, deskewed (Alg 2), no_lines (Alg 3), and ocr_diff_enhanced
            preprocessed, deskewed, no_lines, ocr_diff_enhanced = preprocess_image(permanent_filepath, use_ocr_diff=use_ocr_diff)
            
            # Save intermediate steps for UI visualization
            ocr_diff_filename = f"{img_hash}_ocr_diff{ext}"
            deskewed_filename = f"{img_hash}_deskewed{ext}"
            no_lines_filename = f"{img_hash}_nolines{ext}"
            preprocessed_filename = f"{img_hash}_preprocessed{ext}"
            
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], ocr_diff_filename), ocr_diff_enhanced)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], deskewed_filename), deskewed)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], no_lines_filename), no_lines)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], preprocessed_filename), preprocessed)
            
            # 5. OCR text extraction (Algorithm 4) and Correction (Algorithm 5)
            full_text, ocr_segments = get_ocr_extractor().extract_text(preprocessed)
            
            # 6. Save metadata to DB
            image_id = save_image_metadata(
                filepath=permanent_filepath,
                original_filename=original_filename,
                image_hash=img_hash,
                raw_text=" ".join([seg['word'] for seg in ocr_segments]),
                corrected_text=full_text,
                ocr_segments=ocr_segments
            )
            
            rel_path = os.path.relpath(permanent_filepath, app.root_path).replace('\\', '/')
            ocr_diff_rel_path = f"uploads/{ocr_diff_filename}"
            
            results.append({
                'id': image_id,
                'filename': original_filename,
                'status': 'success',
                'message': 'Processed successfully',
                'filepath': rel_path,
                'ocr_diff_path': ocr_diff_rel_path,
                'hash': img_hash,
                'raw_text': " ".join([seg['word'] for seg in ocr_segments]),
                'corrected_text': full_text
            })
            
        except Exception as e:
            # Cleanup temp file if exists
            if 'temp_path' in locals() and os.path.exists(temp_path):
                os.remove(temp_path)
            results.append({
                'filename': file.filename,
                'status': 'error',
                'message': str(e)
            })
            
    return jsonify({'results': results})

@app.route('/api/ocr-diff/config', methods=['GET'])
def ocr_diff_config():
    """
    Returns OCR-Diff model architecture & diffusion configuration parameters.
    """
    from ocr_diff import get_ocr_diff_pipeline
    pipe = get_ocr_diff_pipeline()
    return jsonify({
        'framework': 'OCR-Diff (IEEE IoTJ 2024)',
        'two_stage_training': True,
        'linear_attention': True,
        'diffusion_steps': pipe.T,
        'time_embedding_dim_K': pipe.K,
        'device': str(pipe.device),
        'residual_learning': 'X_hat = X_hat_0 + x_up'
    })

@app.route('/search', methods=['POST'])
def search_images():
    """
    Endpoint for fuzzy search queries.
    """
    data = request.get_json() or {}
    keyword = data.get('keyword', '')
    threshold = float(data.get('threshold', 80.0))
    
    if not keyword:
        return jsonify({'error': 'Keyword parameter is required'}), 400
        
    try:
        matches = search_keywords_fuzzy(keyword, threshold)
        # Adapt filepaths to web-servable relative paths
        for match in matches:
            match['filepath'] = os.path.relpath(match['filepath'], app.root_path).replace('\\', '/')
        return jsonify({'matches': matches})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images', methods=['GET'])
def list_images():
    """
    Lists all indexed images.
    """
    try:
        images = get_all_images()
        for img in images:
            img['filepath'] = os.path.relpath(img['filepath'], app.root_path).replace('\\', '/')
        return jsonify({'images': images})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/images/<int:image_id>', methods=['DELETE'])
def delete_image(image_id):
    """
    Deletes an image from index and disk.
    """
    try:
        success = delete_image_by_id(image_id)
        if success:
            return jsonify({'success': True, 'message': 'Image deleted successfully'})
        else:
            return jsonify({'success': False, 'error': 'Image not found'}), 404
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """
    Serves uploaded files and intermediate processing images.
    """
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
