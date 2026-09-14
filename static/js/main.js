// State management
let activeImagesList = [];
let uploadQueueFiles = [];
let activeModalImage = null;
let currentSearchMatches = [];

// Initialize Page
document.addEventListener('DOMContentLoaded', () => {
    loadGallery();
    initTabs();
    initModalTabs();
});

// Notifications
function showNotification(title, message, type = 'info') {
    const container = document.getElementById('notification-container');
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    let icon = 'fa-circle-info';
    if (type === 'success') icon = 'fa-circle-check';
    if (type === 'error') icon = 'fa-circle-exclamation';

    notification.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <div class="notification-content">
            <h5>${title}</h5>
            <p>${message}</p>
        </div>
    `;
    
    container.appendChild(notification);
    
    // Animate in
    setTimeout(() => notification.classList.add('show'), 10);
    
    // Remove after 4s
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => notification.remove(), 300);
    }, 4000);
}

// Control Panel Tabs
function initTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const tabId = btn.getAttribute('data-tab');
            document.getElementById(tabId).classList.add('active');
        });
    });
}

// Modal Viewport Tabs
function initModalTabs() {
    const viewTabs = document.querySelectorAll('.view-tab');
    viewTabs.forEach(btn => {
        btn.addEventListener('click', () => {
            viewTabs.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.view-container').forEach(c => c.classList.remove('active'));
            
            btn.classList.add('active');
            const viewId = btn.getAttribute('data-view');
            document.getElementById(viewId).classList.add('active');
        });
    });
}

// Slider Label Update
function updateThresholdLabel(value) {
    document.getElementById('threshold-value').innerText = `${value}%`;
}

// Drag & Drop / File Upload Actions
function triggerFileInput() {
    document.getElementById('file-input').click();
}

function handleDragOver(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('dragover');
    if (e.dataTransfer.files.length > 0) {
        addFilesToQueue(e.dataTransfer.files);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        addFilesToQueue(e.target.files);
    }
}

function addFilesToQueue(files) {
    for (let i = 0; i < files.length; i++) {
        const file = files[i];
        if (!file.type.match('image.*')) {
            showNotification('Invalid File', `${file.name} is not an image.`, 'error');
            continue;
        }
        // Avoid duplicate queue entries
        if (!uploadQueueFiles.some(f => f.name === file.name && f.size === file.size)) {
            uploadQueueFiles.push(file);
        }
    }
    renderQueue();
}

function renderQueue() {
    const queueContainer = document.getElementById('upload-queue');
    const queueItems = document.getElementById('queue-items');
    queueItems.innerHTML = '';
    
    if (uploadQueueFiles.length === 0) {
        queueContainer.style.display = 'none';
        return;
    }
    
    queueContainer.style.display = 'block';
    
    uploadQueueFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'queue-item';
        item.innerHTML = `
            <span>${file.name} (${(file.size / 1024 / 1024).toFixed(2)} MB)</span>
            <button onclick="removeQueueItem(${index})"><i class="fa-solid fa-trash"></i></button>
        `;
        queueItems.appendChild(item);
    });
}

function removeQueueItem(index) {
    uploadQueueFiles.splice(index, 1);
    renderQueue();
}

function clearQueue() {
    uploadQueueFiles = [];
    renderQueue();
    document.getElementById('file-input').value = '';
}

// Upload & Indexing Process
function uploadQueue() {
    if (uploadQueueFiles.length === 0) return;
    
    const formData = new FormData();
    uploadQueueFiles.forEach(file => {
        formData.append('files', file);
    });
    
    const progressContainer = document.getElementById('progress-container');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const progressStatus = document.getElementById('progress-status');
    const progressPercent = document.getElementById('progress-percent');
    
    progressContainer.style.display = 'block';
    progressBarFill.style.width = '0%';
    progressPercent.innerText = '0%';
    progressStatus.innerText = 'Uploading images...';
    
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/upload', true);
    
    // Track upload progress
    xhr.upload.onprogress = (e) => {
        if (e.lengthComputable) {
            const percent = Math.round((e.loaded / e.total) * 100);
            progressBarFill.style.width = `${percent}%`;
            progressPercent.innerText = `${percent}%`;
            if (percent === 100) {
                progressStatus.innerText = 'Processing image & extracting text...';
            }
        }
    };
    
    xhr.onload = () => {
        progressContainer.style.display = 'none';
        if (xhr.status === 200) {
            const data = JSON.parse(xhr.responseText);
            let successes = 0;
            let duplicates = 0;
            let errors = 0;
            
            data.results.forEach(res => {
                if (res.status === 'success') successes++;
                else if (res.status === 'cached') duplicates++;
                else errors++;
            });
            
            if (successes > 0 || duplicates > 0) {
                showNotification(
                    'Indexing Complete', 
                    `Successfully indexed ${successes} new images. ${duplicates} duplicate(s) loaded from cache.`, 
                    'success'
                );
            }
            if (errors > 0) {
                showNotification('Upload Warnings', `Failed to process ${errors} files.`, 'error');
            }
            
            clearQueue();
            loadGallery();
        } else {
            showNotification('Server Error', 'Failed to upload images. Check console logs.', 'error');
            console.error(xhr.responseText);
        }
    };
    
    xhr.onerror = () => {
        progressContainer.style.display = 'none';
        showNotification('Connection Error', 'Network error while uploading.', 'error');
    };
    
    xhr.send(formData);
}

// Load Gallery items
function loadGallery() {
    const loader = document.getElementById('gallery-loader');
    const grid = document.getElementById('gallery-grid');
    const emptyState = document.getElementById('empty-state');
    const countBadge = document.getElementById('gallery-count');
    
    loader.style.display = 'flex';
    grid.style.display = 'none';
    emptyState.style.display = 'none';
    
    fetch('/images')
        .then(res => res.json())
        .then(data => {
            loader.style.display = 'none';
            if (data.images && data.images.length > 0) {
                activeImagesList = data.images;
                countBadge.innerText = `${data.images.length} Images`;
                grid.style.display = 'grid';
                renderGalleryGrid(data.images);
            } else {
                activeImagesList = [];
                countBadge.innerText = '0 Images';
                emptyState.style.display = 'flex';
            }
        })
        .catch(err => {
            loader.style.display = 'none';
            showNotification('Error', 'Failed to load gallery images.', 'error');
            console.error(err);
        });
}

function renderGalleryGrid(images, scoresMap = null) {
    const grid = document.getElementById('gallery-grid');
    grid.innerHTML = '';
    
    images.forEach(img => {
        const card = document.createElement('div');
        card.className = 'gallery-card';
        card.onclick = () => openModal(img.id);
        
        let scoreHTML = '';
        if (scoresMap && scoresMap[img.id] !== undefined) {
            scoreHTML = `<div class="score-badge"><i class="fa-solid fa-bullseye"></i> Match: ${scoresMap[img.id]}%</div>`;
        }
        
        // Use preprocessed image as preview if available, otherwise original
        const ext = img.filepath.split('.').pop();
        const base = img.filepath.substring(0, img.filepath.lastIndexOf('.'));
        const preprocessedUrl = `/${base}_preprocessed.${ext}`;
        
        // Format Date
        const dateStr = new Date(img.created_at).toLocaleDateString(undefined, { 
            month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' 
        });
        
        card.innerHTML = `
            <div class="card-img-wrapper">
                <img src="/${img.filepath}" alt="${img.original_filename}" loading="lazy">
                ${scoreHTML}
            </div>
            <div class="card-info">
                <h4>${img.original_filename}</h4>
                <p class="card-preview-text">${img.corrected_text || 'No text extracted.'}</p>
            </div>
            <div class="card-footer">
                <span><i class="fa-solid fa-spell-check"></i> ${img.word_count || 0} tokens</span>
                <span><i class="fa-regular fa-clock"></i> ${dateStr}</span>
            </div>
        `;
        grid.appendChild(card);
    });
}

// Search Functionality
function handleSearch(e) {
    e.preventDefault();
    const keyword = document.getElementById('search-keyword').value.trim();
    const threshold = document.getElementById('threshold-slider').value;
    
    if (!keyword) return;
    
    const loader = document.getElementById('gallery-loader');
    const grid = document.getElementById('gallery-grid');
    const emptyState = document.getElementById('empty-state');
    const countBadge = document.getElementById('gallery-count');
    const title = document.getElementById('gallery-title');
    const resetBtn = document.getElementById('reset-search-btn');
    
    loader.style.display = 'flex';
    grid.style.display = 'none';
    emptyState.style.display = 'none';
    resetBtn.style.display = 'inline-flex';
    
    fetch('/search', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ keyword, threshold })
    })
    .then(res => res.json())
    .then(data => {
        loader.style.display = 'none';
        title.innerText = `Search Matches for "${keyword}"`;
        
        if (data.matches && data.matches.length > 0) {
            currentSearchMatches = data.matches;
            
            // Map scores to easily show badges
            const scoresMap = {};
            const matchedImages = data.matches.map(m => {
                scoresMap[m.id] = m.score;
                // Normalize keys to align with what get_all_images returns
                return {
                    id: m.id,
                    filepath: m.filepath,
                    original_filename: m.original_filename,
                    raw_text: m.raw_text,
                    corrected_text: m.corrected_text,
                    word_count: m.matched_words.length,
                    created_at: new Date().toISOString() // Fallback
                };
            });
            
            countBadge.className = 'badge success-badge';
            countBadge.innerText = `${data.matches.length} Matches Found`;
            grid.style.display = 'grid';
            renderGalleryGrid(matchedImages, scoresMap);
            showNotification('Search Finished', `Found ${data.matches.length} matching image(s).`, 'success');
        } else {
            currentSearchMatches = [];
            countBadge.className = 'badge';
            countBadge.innerText = '0 Matches';
            grid.style.display = 'none';
            emptyState.innerHTML = `
                <i class="fa-solid fa-magnifying-glass-minus"></i>
                <p>No matches found.</p>
                <span>Try reducing the similarity threshold or refining the keyword.</span>
            `;
            emptyState.style.display = 'flex';
            showNotification('No Matches', 'No images matched your query keyword.', 'info');
        }
    })
    .catch(err => {
        loader.style.display = 'none';
        showNotification('Search Error', 'Failed to perform keyword search.', 'error');
        console.error(err);
    });
}

function resetSearch() {
    document.getElementById('search-keyword').value = '';
    document.getElementById('gallery-title').innerText = 'All Indexed Images';
    document.getElementById('gallery-count').className = 'badge';
    document.getElementById('reset-search-btn').style.display = 'none';
    currentSearchMatches = [];
    loadGallery();
}

// Modal Viewport & Canvas Highlighting
function openModal(imageId) {
    // Check if image is in currentSearchMatches first (which has details about matching word bounding boxes)
    let matchedImg = currentSearchMatches.find(m => m.id === imageId);
    let baseImg = activeImagesList.find(img => img.id === imageId);
    
    if (!baseImg && matchedImg) {
        baseImg = matchedImg;
    }
    
    if (!baseImg) {
        showNotification('Error', 'Image data could not be retrieved.', 'error');
        return;
    }
    
    activeModalImage = baseImg;
    
    const ext = baseImg.filepath.split('.').pop();
    const base = baseImg.filepath.substring(0, baseImg.filepath.lastIndexOf('.'));
    
    // Set viewport images
    document.getElementById('modal-img-original').src = `/${baseImg.filepath}`;
    document.getElementById('modal-img-ocrdiff').src = `/${base}_ocr_diff.${ext}`;
    document.getElementById('modal-img-preprocessed').src = `/${base}_preprocessed.${ext}`;
    document.getElementById('modal-img-deskewed').src = `/${base}_deskewed.${ext}`;
    document.getElementById('modal-img-nolines').src = `/${base}_nolines.${ext}`;
    
    // Set metadata fields
    document.getElementById('modal-title').innerText = baseImg.original_filename;
    document.getElementById('modal-filename').innerText = baseImg.original_filename;
    document.getElementById('modal-hash').innerText = `SHA-256: ${baseImg.hash || 'N/A'}`;
    document.getElementById('modal-raw-text').innerText = baseImg.raw_text || 'No text extracted.';
    document.getElementById('modal-corrected-text').innerText = baseImg.corrected_text || 'No text indexed.';
    
    // Reset viewports to display original
    document.querySelectorAll('.view-tab').forEach(b => b.classList.remove('active'));
    document.querySelector('.view-tab[data-view="original-view"]').classList.add('active');
    document.querySelectorAll('.view-container').forEach(c => c.classList.remove('active'));
    document.getElementById('original-view').classList.add('active');
    
    // Handle fuzzy similarity fields
    const scoreField = document.querySelector('.search-score-field');
    if (matchedImg) {
        scoreField.style.display = 'flex';
        document.getElementById('modal-score').innerText = `Score: ${matchedImg.score}%`;
    } else {
        scoreField.style.display = 'none';
    }
    
    const modal = document.getElementById('detail-modal');
    modal.classList.add('open');
    
    // Draw bounding boxes on load/resize
    const origImg = document.getElementById('modal-img-original');
    
    if (origImg.complete) {
        drawBoundingBoxes(matchedImg);
    } else {
        origImg.onload = () => drawBoundingBoxes(matchedImg);
    }
    
    // Handle window resize dynamically to adjust canvas scaling
    window.onresize = () => drawBoundingBoxes(matchedImg);
}

function drawBoundingBoxes(matchedImg) {
    const canvas = document.getElementById('highlight-canvas');
    const img = document.getElementById('modal-img-original');
    
    if (!canvas || !img) return;
    
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (!matchedImg || !matchedImg.matched_words || matchedImg.matched_words.length === 0) {
        canvas.style.display = 'none';
        return;
    }
    
    canvas.style.display = 'block';
    
    // Set logical dimensions of canvas to match image dimensions
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    
    // Draw each matching word boundary box
    matchedImg.matched_words.forEach(match => {
        const box = match.box; // [[x0, y0], [x1, y1], [x2, y2], [x3, y3]]
        
        ctx.strokeStyle = '#ef4444'; // Red outline
        ctx.lineWidth = Math.max(2, img.naturalWidth / 400); // Scale line width with image size
        ctx.fillStyle = 'rgba(239, 68, 68, 0.25)'; // Highlight fill
        
        ctx.beginPath();
        ctx.moveTo(box[0][0], box[0][1]);
        ctx.lineTo(box[1][0], box[1][1]);
        ctx.lineTo(box[2][0], box[2][1]);
        ctx.lineTo(box[3][0], box[3][1]);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();
        
        // Add a small label showing match percentage
        ctx.fillStyle = '#ef4444';
        ctx.font = `bold ${Math.max(12, img.naturalWidth / 80)}px sans-serif`;
        // Position label above top-left corner of the box
        ctx.fillText(`${match.score.toFixed(0)}%`, box[0][0], box[0][1] - 5);
    });
}

function closeModal() {
    document.getElementById('detail-modal').classList.remove('open');
    window.onresize = null; // Clear event listener
    activeModalImage = null;
}

// Delete Active Image
function deleteActiveImage() {
    if (!activeModalImage) return;
    
    if (confirm(`Are you sure you want to delete "${activeModalImage.original_filename}" from the database and disk?`)) {
        fetch(`/images/${activeModalImage.id}`, {
            method: 'DELETE'
        })
        .then(res => res.json())
        .then(data => {
            if (data.success) {
                showNotification('Deleted', 'Image deleted successfully.', 'success');
                closeModal();
                // If we are currently searching, repeat the search. Else, refresh all images.
                const keyword = document.getElementById('search-keyword').value.trim();
                if (keyword && currentSearchMatches.length > 0) {
                    // Update search list
                    currentSearchMatches = currentSearchMatches.filter(m => m.id !== activeModalImage.id);
                    // Re-trigger search form submit programmatically
                    document.getElementById('search-form').dispatchEvent(new Event('submit'));
                } else {
                    loadGallery();
                }
            } else {
                showNotification('Error', data.error || 'Failed to delete image.', 'error');
            }
        })
        .catch(err => {
            showNotification('Error', 'Network error while deleting image.', 'error');
            console.error(err);
        });
    }
}
