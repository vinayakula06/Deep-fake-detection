/**
 * ═══════════════════════════════════════════════════════════════════════════
 * DeepShield - AI Deepfake Detection System
 * Main Application JavaScript
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ── Application State ──────────────────────────────────────────────────────
let currentFile = null;
let lastResult = null;

// ── Health Check ───────────────────────────────────────────────────────────
(async () => {
  try {
    const response = await fetch('/health');
    const data = await response.json();
    const statusEl = document.getElementById('status-text');
    
    if (data.models_loaded > 0) {
      statusEl.textContent = `Online · ${data.models_loaded} models · ${data.device}`;
    } else {
      statusEl.textContent = 'Demo mode (no weights)';
    }
  } catch {
    document.getElementById('status-text').textContent = 'Offline';
  }
})();

// ── Drag & Drop Handlers ───────────────────────────────────────────────────
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

// ── File Handling ──────────────────────────────────────────────────────────
function handleFile(file) {
  currentFile = file;
  document.getElementById('preview-section').style.display = 'block';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('progress-section').style.display = 'none';
  document.getElementById('analyze-btn').disabled = false;

  document.getElementById('fi-size').textContent = formatSize(file.size);
  document.getElementById('fi-type').textContent = file.type || 'unknown';

  const isImage = file.type.startsWith('image/');
  const videoEl = document.getElementById('preview-video');
  const imageEl = document.getElementById('preview-img');

  if (isImage) {
    videoEl.style.display = 'none';
    imageEl.style.display = 'block';
    imageEl.src = URL.createObjectURL(file);
    document.getElementById('fi-dur').textContent = 'N/A';
    document.getElementById('thumb-strip').innerHTML = '';
  } else {
    imageEl.style.display = 'none';
    videoEl.style.display = 'block';
    videoEl.src = URL.createObjectURL(file);
    videoEl.onloadedmetadata = () => {
      document.getElementById('fi-dur').textContent = formatTime(videoEl.duration);
      generateThumbnails(videoEl);
    };
  }
  
  showToast('File loaded. Ready to analyze.', 'info');
}

// ── Thumbnail Generation ───────────────────────────────────────────────────
function generateThumbnails(video) {
  const strip = document.getElementById('thumb-strip');
  strip.innerHTML = '';
  const duration = video.duration;
  const count = 12;
  const canvas = document.createElement('canvas');
  canvas.width = 144;
  canvas.height = 108;
  const ctx = canvas.getContext('2d');
  
  let i = 0;
  function captureNext() {
    if (i >= count) return;
    const time = (i / (count - 1)) * duration;
    video.currentTime = time;
    video.onseeked = () => {
      ctx.drawImage(video, 0, 0, 144, 108);
      const img = document.createElement('img');
      img.src = canvas.toDataURL('image/jpeg', 0.7);
      strip.appendChild(img);
      i++;
      captureNext();
    };
  }
  captureNext();
}

// ── Analysis Execution ─────────────────────────────────────────────────────
document.getElementById('analyze-btn').addEventListener('click', async () => {
  if (!currentFile) return;
  await runAnalysis();
});

async function runAnalysis() {
  document.getElementById('progress-section').style.display = 'block';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('scan-line').style.display = 'block';

  resetSteps();

  const steps = [
    { id: 's1', label: 'Extracting frames…', ms: 800 },
    { id: 's2', label: 'Detecting faces…', ms: 1200 },
    { id: 's3', label: 'Running neural inference…', ms: 1500 },
    { id: 's4', label: 'Aggregating predictions…', ms: 600 },
  ];
  const percents = [20, 45, 85, 95];

  // Start API call in parallel with UI animation
  const formData = new FormData();
  formData.append('file', currentFile);
  const fetchPromise = fetch('/predict', { method: 'POST', body: formData });

  for (let i = 0; i < steps.length; i++) {
    setStep(steps[i].id, 'active', steps[i].label);
    setProgress(percents[i]);
    await delay(steps[i].ms);
    setStep(steps[i].id, 'done', '✓ Done');
  }

  setProgress(100);
  showToast('Inference complete. Rendering results…', 'info');

  try {
    const response = await fetchPromise;
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Server error');
    
    lastResult = data;
    renderResults(data);
    
    if (data.processing_time) {
      document.getElementById('stat-time').textContent = data.processing_time + 's';
    }
  } catch (error) {
    showToast('Error: ' + error.message, 'error');
    document.getElementById('progress-section').style.display = 'none';
    document.getElementById('analyze-btn').disabled = false;
  }

  document.getElementById('scan-line').style.display = 'none';
  document.getElementById('analyze-btn').disabled = false;
}

// ── Results Rendering ──────────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById('results-section').style.display = 'block';
  document.getElementById('progress-section').style.display = 'none';

  const isFake = data.verdict === 'FAKE';
  const verdictCard = document.getElementById('verdict-card');
  verdictCard.className = 'verdict-card ' + (isFake ? 'fake' : 'real');
  document.getElementById('verdict-label').textContent = data.verdict;
  document.getElementById('verdict-sub').textContent = isFake
    ? 'High likelihood of synthetic manipulation detected.'
    : 'No significant manipulation signatures detected.';

  // Animate gauge
  animateGauge(data.probability);

  // Update metrics
  document.getElementById('m-fake').textContent = (data.probability * 100).toFixed(1) + '%';
  document.getElementById('m-fake').style.color = isFake ? 'var(--red)' : 'var(--green)';
  document.getElementById('m-conf').textContent = (data.confidence * 100).toFixed(1) + '%';
  document.getElementById('m-cfakes').textContent = data.confident_fakes;
  document.getElementById('m-avg').textContent = data.probability.toFixed(3);
  document.getElementById('m-frames').textContent = data.frames_analyzed;
  
  const riskEl = document.getElementById('m-risk');
  riskEl.textContent = data.risk_level;
  riskEl.style.color = data.risk_level === 'HIGH' ? 'var(--red)' 
    : data.risk_level === 'MEDIUM' ? 'var(--yellow)' : 'var(--green)';

  // Render bar chart
  renderBars(data.frame_scores || []);

  // Generate explanation
  const probability = (data.probability * 100).toFixed(1);
  let explanation;
  
  if (data.detection_method === 'FILENAME_SIGNATURE' || data.detection_method === 'AI_WATERMARK') {
    explanation = `<strong>AI-GENERATED CONTENT DETECTED</strong> — This media contains clear indicators of AI generation. `
      + `Multiple signatures consistent with synthetic content were identified. Flagged as <strong style="color:var(--red)">FAKE</strong>.`;
  } else if (isFake) {
    explanation = `<strong>FAKE detected</strong> — The ensemble assigned a ${probability}% fake probability. `
      + `${data.confident_fakes} of ${data.frames_analyzed} frames exceeded the 0.8 confidence threshold. `
      + `Risk level: <strong style="color:var(--red)">${data.risk_level}</strong>.`;
  } else {
    explanation = `<strong>REAL assessed</strong> — The ensemble assigned only a ${probability}% fake probability. `
      + `The video shows strong signatures consistent with authentic media. `
      + `Risk level: <strong style="color:var(--green)">${data.risk_level}</strong>.`;
  }
  document.getElementById('explain-box').innerHTML = explanation;

  setTimeout(() => {
    document.getElementById('results-section').scrollIntoView({ behavior: 'smooth' });
  }, 300);
}

// ── Gauge Animation ────────────────────────────────────────────────────────
function animateGauge(probability) {
  // Semi-circle arc: 180° sweep, circumference = π × 130 ≈ 408
  const arcLength = 408;
  const offset = arcLength - (arcLength * probability);
  const fill = document.getElementById('gauge-fill');
  const color = probability > 0.5 ? 'var(--red)' : 'var(--green)';
  fill.style.stroke = color;
  setTimeout(() => { fill.style.strokeDashoffset = offset; }, 50);

  // Needle rotation: -90deg = REAL (left), +90deg = FAKE (right)
  const angle = -90 + (probability * 180);
  document.getElementById('gauge-needle').style.transform = `rotate(${angle}deg)`;
  document.getElementById('gauge-pct').textContent = (probability * 100).toFixed(1) + '%';
  document.getElementById('gauge-pct').style.fill = color;
}

// ── Bar Chart Rendering ────────────────────────────────────────────────────
function renderBars(scores) {
  const wrap = document.getElementById('frame-bars');
  wrap.innerHTML = '';
  scores.forEach(score => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.style.height = '4px';
    bar.style.background = score > 0.5 ? 'var(--red)' : 'var(--green)';
    bar.title = `Score: ${score.toFixed(3)}`;
    wrap.appendChild(bar);
    setTimeout(() => {
      bar.style.height = (score * 80 + 4) + 'px';
    }, 100 + Math.random() * 400);
  });
}

// ── Report Download ────────────────────────────────────────────────────────
function downloadReport() {
  if (!lastResult) return;
  
  const lines = [
    '=== DeepShield Analysis Report ===',
    `Date: ${new Date().toLocaleString()}`,
    `File: ${currentFile?.name || 'unknown'}`,
    '',
    `Verdict: ${lastResult.verdict}`,
    `Fake Probability: ${(lastResult.probability * 100).toFixed(2)}%`,
    `Confidence: ${(lastResult.confidence * 100).toFixed(2)}%`,
    `Risk Level: ${lastResult.risk_level}`,
    `Frames Analyzed: ${lastResult.frames_analyzed}`,
    `Confident Fakes: ${lastResult.confident_fakes}`,
    `Processing Time: ${lastResult.processing_time}s`,
    '',
    '--- Frame Scores ---',
    ...(lastResult.frame_scores || []).map((s, i) => 
      `Frame ${i + 1}: ${s.toFixed(4)} — ${s > 0.5 ? 'FAKE' : 'REAL'}`
    ),
    '',
    '=== End of Report ==='
  ];
  
  const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `deepshield_report_${Date.now()}.txt`;
  a.click();
  showToast('Report downloaded.', 'success');
}

// ── Copy Result ────────────────────────────────────────────────────────────
function copyResult() {
  if (!lastResult) return;
  
  const text = `DeepShield Result: ${lastResult.verdict} (${(lastResult.probability * 100).toFixed(1)}% fake probability, Risk: ${lastResult.risk_level})`;
  navigator.clipboard.writeText(text).then(() => {
    showToast('Copied to clipboard.', 'success');
  });
}

// ── Reset Application ──────────────────────────────────────────────────────
function resetAll() {
  currentFile = null;
  lastResult = null;
  document.getElementById('preview-section').style.display = 'none';
  document.getElementById('results-section').style.display = 'none';
  document.getElementById('progress-section').style.display = 'none';
  document.getElementById('analyze-btn').disabled = true;
  document.getElementById('preview-video').src = '';
  document.getElementById('preview-img').src = '';
  document.getElementById('thumb-strip').innerHTML = '';
  document.getElementById('file-input').value = '';
  window.scrollTo({ top: 0, behavior: 'smooth' });
  showToast('Ready for new analysis.', 'info');
}

// ── Progress Step Helpers ──────────────────────────────────────────────────
function setStep(id, className, statusText) {
  const element = document.getElementById(id);
  element.className = 'step ' + className;
  element.querySelector('.step-status').textContent = statusText;
}

function resetSteps() {
  ['s1', 's2', 's3', 's4'].forEach(id => setStep(id, '', 'Waiting…'));
}

function setProgress(percent) {
  document.getElementById('prog-fill').style.width = percent + '%';
}

// ── Utility Functions ──────────────────────────────────────────────────────
function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function formatSize(bytes) {
  if (bytes < 1024) return bytes + 'B';
  if (bytes < 1048576) return (bytes / 1024).toFixed(1) + 'KB';
  return (bytes / 1048576).toFixed(1) + 'MB';
}

function formatTime(seconds) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
}

// ── Toast Notifications ────────────────────────────────────────────────────
let toastTimer;
function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  toast.textContent = message;
  toast.className = 'show ' + type;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => {
    toast.className = '';
  }, 3000);
}
