"""
FastAPI application for Skin Cancer Detection.
Provides REST endpoints + a browser-friendly HTML UI.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from app.inference import CLASS_NAMES, load_model, predict_from_bytes

logger = logging.getLogger(__name__)

# ── lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the model on startup."""
    logger.info("Loading model on startup …")
    try:
        load_model()
        logger.info("Model ready.")
    except Exception as exc:
        logger.warning("Model could not be loaded at startup: %s", exc)
    yield


# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Skin Cancer Detector API",
    description=(
        "Upload a dermoscopic image and get a binary malignant / benign prediction "
        "powered by a fine-tuned ResNet-50 model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── HTML UI ────────────────────────────────────────────────────────────────────
_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>DermaScan AI — Deep Learning Skin Lesion Classifier</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
  :root {
    --bg-base: #07090e;
    --bg-card: rgba(18, 22, 34, 0.75);
    --bg-card-border: rgba(255, 255, 255, 0.08);
    --bg-subtle: rgba(255, 255, 255, 0.03);
    
    --primary: #38bdf8;
    --primary-glow: rgba(56, 189, 248, 0.25);
    --secondary: #818cf8;
    
    --success: #10b981;
    --success-glow: rgba(16, 185, 129, 0.2);
    --danger: #f43f5e;
    --danger-glow: rgba(244, 63, 94, 0.2);
    
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --text-dim: #64748b;
    
    --radius-xl: 24px;
    --radius-lg: 16px;
    --radius-md: 12px;
  }

  *, *::before, *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }

  body {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    background-color: var(--bg-base);
    background-image: 
      radial-gradient(at 0% 0%, rgba(56, 189, 248, 0.12) 0px, transparent 50%),
      radial-gradient(at 100% 0%, rgba(129, 140, 248, 0.12) 0px, transparent 50%),
      radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.8) 0px, transparent 80%);
    background-attachment: fixed;
    color: var(--text-main);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 3rem 1.5rem;
    overflow-x: hidden;
  }

  /* ── Header ── */
  header {
    text-align: center;
    max-width: 680px;
    margin-bottom: 2.5rem;
    animation: fadeInDown 0.7s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .nav-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.35rem 0.9rem;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    color: var(--primary);
    text-transform: uppercase;
    margin-bottom: 1.25rem;
    box-shadow: 0 0 20px var(--primary-glow);
  }
  .pulse-dot {
    width: 7px;
    height: 7px;
    background: var(--primary);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--primary);
    animation: blink 2s infinite ease-in-out;
  }

  h1 {
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1.15;
    margin-bottom: 0.85rem;
    background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }

  .subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    line-height: 1.6;
    font-weight: 400;
  }

  /* ── Layout ── */
  .container {
    width: 100%;
    max-width: 620px;
    animation: fadeInUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--bg-card-border);
    border-radius: var(--radius-xl);
    padding: 2.25rem;
    box-shadow: 
      0 20px 40px -15px rgba(0, 0, 0, 0.6),
      0 0 0 1px rgba(255, 255, 255, 0.04);
  }

  /* ── Presets / Quick Test ── */
  .sample-row {
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    flex-wrap: wrap;
  }
  .sample-title {
    font-size: 0.8rem;
    color: var(--text-dim);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    width: 100%;
    margin-bottom: 0.2rem;
  }
  .sample-chip {
    background: var(--bg-subtle);
    border: 1px solid var(--bg-card-border);
    color: var(--text-muted);
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s ease;
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
  }
  .sample-chip:hover {
    background: rgba(56, 189, 248, 0.1);
    border-color: rgba(56, 189, 248, 0.3);
    color: var(--primary);
    transform: translateY(-1px);
  }

  /* ── Drop Zone ── */
  .drop-zone {
    border: 2px dashed rgba(255, 255, 255, 0.12);
    border-radius: var(--radius-lg);
    padding: 2.75rem 1.5rem;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    background: rgba(255, 255, 255, 0.015);
    position: relative;
    overflow: hidden;
  }
  .drop-zone:hover, .drop-zone.dragover {
    border-color: var(--primary);
    background: rgba(56, 189, 248, 0.04);
    box-shadow: 0 0 30px var(--primary-glow);
    transform: scale(0.995);
  }
  .drop-zone input[type=file] {
    position: absolute;
    inset: 0;
    opacity: 0;
    cursor: pointer;
    z-index: 10;
  }
  .drop-icon-container {
    width: 64px;
    height: 64px;
    margin: 0 auto 1.25rem;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.8rem;
    color: var(--primary);
    transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
  }
  .drop-zone:hover .drop-icon-container {
    transform: scale(1.1) rotate(6deg);
  }
  .drop-label {
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-main);
    margin-bottom: 0.35rem;
  }
  .drop-sub {
    font-size: 0.85rem;
    color: var(--text-dim);
  }

  /* ── Image Preview ── */
  #preview-wrap {
    display: none;
    margin-top: 1.5rem;
    border-radius: var(--radius-lg);
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.1);
    position: relative;
    box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    background: #000;
  }
  #preview-wrap img {
    width: 100%;
    height: 280px;
    object-fit: contain;
    display: block;
    background: #090d16;
  }
  #preview-clear {
    position: absolute;
    top: 0.75rem;
    right: 0.75rem;
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.15);
    color: var(--text-muted);
    border-radius: 50%;
    width: 34px;
    height: 34px;
    font-size: 1rem;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.2s;
    z-index: 20;
  }
  #preview-clear:hover {
    color: #fff;
    background: var(--danger);
    border-color: var(--danger);
    transform: rotate(90deg);
  }

  /* ── Primary Action Button ── */
  .btn-submit {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.65rem;
    width: 100%;
    margin-top: 1.5rem;
    padding: 1rem 1.5rem;
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: #030712;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: all 0.25s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 4px 20px var(--primary-glow);
  }
  .btn-submit:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(56, 189, 248, 0.4);
    filter: brightness(1.08);
  }
  .btn-submit:active:not(:disabled) {
    transform: translateY(0);
  }
  .btn-submit:disabled {
    opacity: 0.4;
    cursor: not-allowed;
    box-shadow: none;
    filter: grayscale(0.5);
  }

  /* ── Loading Spinner ── */
  .spinner {
    width: 20px;
    height: 20px;
    border: 2.5px solid rgba(3, 7, 18, 0.25);
    border-top-color: #030712;
    border-radius: 50%;
    animation: spin 0.75s linear infinite;
    display: none;
  }

  /* ── Result Card ── */
  #result {
    margin-top: 2rem;
    display: none;
    animation: zoomIn 0.45s cubic-bezier(0.16, 1, 0.3, 1);
  }

  .result-banner {
    border-radius: var(--radius-lg);
    padding: 1.75rem;
    border: 1px solid;
    position: relative;
    overflow: hidden;
  }
  .result-banner.benign {
    background: linear-gradient(145deg, rgba(16, 185, 129, 0.12) 0%, rgba(18, 22, 34, 0.6) 100%);
    border-color: rgba(16, 185, 129, 0.35);
    box-shadow: 0 10px 30px var(--success-glow);
  }
  .result-banner.malignant {
    background: linear-gradient(145deg, rgba(244, 63, 94, 0.15) 0%, rgba(18, 22, 34, 0.6) 100%);
    border-color: rgba(244, 63, 94, 0.35);
    box-shadow: 0 10px 30px var(--danger-glow);
  }

  .result-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1.5rem;
  }
  .result-meta-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 700;
    color: var(--text-dim);
    margin-bottom: 0.35rem;
  }
  .result-badge {
    font-size: 1.6rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: -0.01em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
  }
  .result-banner.benign .result-badge { color: var(--success); }
  .result-banner.malignant .result-badge { color: var(--danger); }

  .confidence-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.4rem 0.8rem;
    border-radius: 8px;
    background: rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(255, 255, 255, 0.08);
  }

  /* ── Interactive Progress Metrics ── */
  .prob-metric {
    margin-top: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.9rem;
  }
  .prob-header {
    display: flex;
    justify-content: space-between;
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-muted);
  }
  .prob-percent {
    font-family: 'JetBrains Mono', monospace;
    color: var(--text-main);
  }
  .bar-rail {
    height: 8px;
    background: rgba(255, 255, 255, 0.06);
    border-radius: 99px;
    overflow: hidden;
    position: relative;
  }
  .bar-fill {
    height: 100%;
    border-radius: 99px;
    width: 0%;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
  }
  .bar-fill.benign { background: linear-gradient(90deg, #059669, #10b981); }
  .bar-fill.malignant { background: linear-gradient(90deg, #e11d48, #f43f5e); }

  /* ── Clinical Advisory ── */
  .clinical-note {
    margin-top: 1.5rem;
    padding: 1rem 1.25rem;
    background: rgba(0, 0, 0, 0.25);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: var(--radius-md);
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.55;
    display: flex;
    gap: 0.75rem;
  }
  .clinical-note svg {
    flex-shrink: 0;
    margin-top: 2px;
    stroke: var(--text-dim);
  }

  /* ── Status / Error Toast ── */
  #error-msg {
    display: none;
    margin-top: 1rem;
    padding: 0.85rem 1rem;
    background: var(--danger-glow);
    border: 1px solid rgba(244, 63, 94, 0.3);
    border-radius: var(--radius-md);
    color: #fda4af;
    font-size: 0.875rem;
    animation: shake 0.3s ease;
  }

  /* ── Footer ── */
  footer {
    margin-top: 3.5rem;
    color: var(--text-dim);
    font-size: 0.85rem;
    text-align: center;
  }
  footer a {
    color: var(--text-muted);
    text-decoration: none;
    transition: color 0.2s;
  }
  footer a:hover {
    color: var(--primary);
  }

  /* ── Animations ── */
  @keyframes fadeInDown {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
  }
  @keyframes zoomIn {
    from { opacity: 0; transform: scale(0.96); }
    to { opacity: 1; transform: scale(1); }
  }
  @keyframes spin {
    to { transform: rotate(360deg); }
  }
  @keyframes blink {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.85); }
  }
  @keyframes shake {
    0%, 100% { transform: translateX(0); }
    25% { transform: translateX(-4px); }
    75% { transform: translateX(4px); }
  }
</style>
</head>
<body>

<header>
  <div class="nav-pill">
    <span class="pulse-dot"></span> ResNet-50 Classifier Active
  </div>
  <h1>Skin Cancer Detector</h1>
  <p class="subtitle">
    Medical computer vision pipeline triaging dermoscopic lesions into 
    <strong>benign</strong> or <strong>malignant</strong> classifications.
  </p>
</header>

<div class="container">
  <div class="glass-card">
    
    <!-- Quick sample tester chips -->
    <div class="sample-row">
      <span class="sample-title">⚡ Quick Test Synthetics</span>
      <button class="sample-chip" id="sample-benign-btn" type="button">
        <span>🟢</span> Normal Nevus Pattern
      </button>
      <button class="sample-chip" id="sample-malignant-btn" type="button">
        <span>🔴</span> Irregular Pigment Pattern
      </button>
    </div>

    <!-- Drag & Drop container -->
    <div class="drop-zone" id="dropZone">
      <input type="file" id="fileInput" accept="image/jpeg,image/png,image/webp" />
      <div class="drop-icon-container">
        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7"></path>
          <line x1="16" y1="5" x2="22" y2="5"></line>
          <line x1="19" y1="2" x2="19" y2="8"></line>
          <circle cx="9" cy="9" r="2"></circle>
          <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"></path>
        </svg>
      </div>
      <p class="drop-label">Drop lesion image here or click to browse</p>
      <p class="drop-sub">Dermoscopic JPEG, PNG, or WEBP (Standard 300 &times; 225 resolution recommended)</p>
    </div>

    <!-- Image preview wrap -->
    <div id="preview-wrap">
      <img id="preview" src="" alt="Dermoscopic view preview" />
      <button id="preview-clear" title="Remove image" aria-label="Remove image">&times;</button>
    </div>

    <!-- Error message display -->
    <div id="error-msg"></div>

    <!-- Submit Analysis button -->
    <button class="btn-submit" id="analyzeBtn" disabled>
      <div class="spinner" id="spinner"></div>
      <span id="btn-text">Run Neural Inference</span>
    </button>

    <!-- Results display container -->
    <div id="result">
      <div class="result-banner" id="result-banner">
        <div class="result-top">
          <div>
            <div class="result-meta-title">Triage Classification</div>
            <div class="result-badge" id="result-badge">
              <span id="result-icon"></span>
              <span id="result-class-name">Benign</span>
            </div>
          </div>
          <div class="confidence-pill" id="result-conf">0.0% Conf</div>
        </div>

        <!-- Class probabilities breakdown -->
        <div class="prob-metric">
          <div>
            <div class="prob-header">
              <span>Benign Probability</span>
              <span class="prob-percent" id="pb-val">0.0%</span>
            </div>
            <div class="bar-rail">
              <div class="bar-fill benign" id="pb-fill"></div>
            </div>
          </div>

          <div>
            <div class="prob-header">
              <span>Malignant (Melanoma / Carcinoma)</span>
              <span class="prob-percent" id="pm-val">0.0%</span>
            </div>
            <div class="bar-rail">
              <div class="bar-fill malignant" id="pm-fill"></div>
            </div>
          </div>
        </div>

        <div class="clinical-note">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="8" x2="12" y2="12"></line>
            <line x1="12" y1="16" x2="12.01" y2="16"></line>
          </svg>
          <div>
            <strong>Clinical Note:</strong> Neural inference is probabilistic and trained on the ISIC / HAM10000 dataset archive. Suspicious lesions exhibiting ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolving) must always be evaluated via dermoscopy and histopathology by a board-certified dermatologist.
          </div>
        </div>

      </div>
    </div>

  </div>
</div>

<footer>
  Fine-Tuned ResNet-50 &nbsp;&bull;&nbsp;
  <a href="/docs" target="_blank">Swagger OpenAPI Docs</a> &nbsp;&bull;&nbsp;
  <a href="/health" target="_blank">Health Status</a> &nbsp;&bull;&nbsp;
  <a href="https://github.com/vishalvemula1/skin-cancer" target="_blank">Source Code</a>
</footer>

<script>
const dropZone     = document.getElementById('dropZone');
const fileInput    = document.getElementById('fileInput');
const previewWrap  = document.getElementById('preview-wrap');
const previewImg   = document.getElementById('preview');
const clearBtn     = document.getElementById('preview-clear');
const analyzeBtn   = document.getElementById('analyzeBtn');
const spinner      = document.getElementById('spinner');
const btnText      = document.getElementById('btn-text');
const resultDiv    = document.getElementById('result');
const resultBanner = document.getElementById('result-banner');
const errorDiv     = document.getElementById('error-msg');

let selectedFile = null;

function setFile(file) {
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  previewWrap.style.display = 'block';
  analyzeBtn.disabled = false;
  resultDiv.style.display = 'none';
  errorDiv.style.display = 'none';
}

function clearPreview() {
  selectedFile = null;
  previewWrap.style.display = 'none';
  previewImg.src = '';
  fileInput.value = '';
  analyzeBtn.disabled = true;
  resultDiv.style.display = 'none';
  errorDiv.style.display = 'none';
}

fileInput.addEventListener('change', e => {
  if (e.target.files && e.target.files[0]) {
    setFile(e.target.files[0]);
  }
});

clearBtn.addEventListener('click', clearPreview);

dropZone.addEventListener('dragover', e => {
  e.preventDefault();
  dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) {
    setFile(file);
  }
});

// Quick canvas generators for easy 1-click test demos
function createSyntheticPattern(type) {
  const canvas = document.createElement('canvas');
  canvas.width = 300;
  canvas.height = 225;
  const ctx = canvas.getContext('2d');
  
  // Skin tone base
  ctx.fillStyle = '#dfaf91';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  
  // Lesion center
  const cx = 150, cy = 112;
  const grad = ctx.createRadialGradient(cx, cy, 10, cx, cy, type === 'benign' ? 45 : 70);
  
  if (type === 'benign') {
    grad.addColorStop(0, '#532c1c');
    grad.addColorStop(0.7, '#884c30');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.ellipse(cx, cy, 45, 38, 0, 0, 2 * Math.PI);
    ctx.fill();
  } else {
    grad.addColorStop(0, '#1a0c09');
    grad.addColorStop(0.4, '#481914');
    grad.addColorStop(0.8, '#8c3127');
    grad.addColorStop(1, 'transparent');
    ctx.fillStyle = grad;
    ctx.beginPath();
    ctx.moveTo(cx - 50, cy - 30);
    ctx.bezierCurveTo(cx - 20, cy - 60, cx + 45, cy - 40, cx + 65, cy);
    ctx.bezierCurveTo(cx + 80, cy + 45, cx + 20, cy + 60, cx - 15, cy + 50);
    ctx.bezierCurveTo(cx - 65, cy + 35, cx - 80, cy - 10, cx - 50, cy - 30);
    ctx.fill();
  }

  canvas.toBlob(blob => {
    const syntheticFile = new File([blob], `${type}_test_sample.png`, { type: 'image/png' });
    setFile(syntheticFile);
  }, 'image/png');
}

document.getElementById('sample-benign-btn').addEventListener('click', () => createSyntheticPattern('benign'));
document.getElementById('sample-malignant-btn').addEventListener('click', () => createSyntheticPattern('malignant'));

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  analyzeBtn.disabled = true;
  spinner.style.display = 'block';
  btnText.textContent = 'Analyzing Deep Representations…';
  resultDiv.style.display = 'none';
  errorDiv.style.display = 'none';

  try {
    const form = new FormData();
    form.append('file', selectedFile);

    const resp = await fetch('/predict', { method: 'POST', body: form });
    if (!resp.ok) {
      const err = await resp.json();
      throw new Error(err.detail || 'Inference call failed');
    }
    const data = await resp.json();
    displayResult(data);
  } catch(err) {
    errorDiv.textContent = 'Error: ' + err.message;
    errorDiv.style.display = 'block';
  } finally {
    analyzeBtn.disabled = false;
    spinner.style.display = 'none';
    btnText.textContent = 'Run Neural Inference';
  }
});

function displayResult(data) {
  const isMal = data.predicted_class === 'malignant';
  
  resultBanner.className = 'result-banner ' + data.predicted_class;
  document.getElementById('result-class-name').textContent = data.predicted_class;
  document.getElementById('result-icon').textContent = isMal ? '⚠️' : '🛡️';
  document.getElementById('result-conf').textContent = (data.confidence * 100).toFixed(1) + '% Confidence';

  const pb = data.probabilities['benign'] || 0;
  const pm = data.probabilities['malignant'] || 0;

  document.getElementById('pb-val').textContent = (pb * 100).toFixed(1) + '%';
  document.getElementById('pm-val').textContent = (pm * 100).toFixed(1) + '%';

  resultDiv.style.display = 'block';

  // Smooth bar transitions
  requestAnimationFrame(() => {
    document.getElementById('pb-fill').style.width = (pb * 100) + '%';
    document.getElementById('pm-fill').style.width = (pm * 100) + '%';
  });
}
</script>
</body>
</html>
"""


# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index():
    """Serve the browser-friendly UI."""
    return HTMLResponse(content=_HTML)


@app.get("/health", tags=["Meta"])
async def health():
    """Quick liveness check."""
    return {"status": "ok", "classes": CLASS_NAMES}


@app.post("/predict", tags=["Prediction"])
async def predict(file: UploadFile = File(...)):
    """
    Classify a dermoscopic image as **benign** or **malignant**.

    - **file**: A JPEG / PNG / WEBP dermoscopic lesion image.

    Returns `predicted_class`, `confidence`, and per-class `probabilities`.
    """
    # validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail="Only image files are accepted (JPEG, PNG, WEBP, …).",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = predict_from_bytes(image_bytes)
    except Exception as exc:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc

    return JSONResponse(content=result)
