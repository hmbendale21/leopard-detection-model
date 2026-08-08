/* ─── LeopardEye Main JS ─── */

/* ══════════════════
   PARTICLE SYSTEM
══════════════════ */
(function () {
  const canvas = document.getElementById('particle-canvas');
  const ctx    = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  function randomParticle() {
    return {
      x: Math.random() * W,
      y: Math.random() * H,
      r: Math.random() * 1.5 + 0.3,
      dx: (Math.random() - 0.5) * 0.3,
      dy: (Math.random() - 0.5) * 0.3,
      alpha: Math.random() * 0.6 + 0.1,
      color: Math.random() > 0.5 ? '#e8a020' : '#00e87a'
    };
  }

  for (let i = 0; i < 120; i++) particles.push(randomParticle());

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle   = p.color;
      ctx.fill();
      p.x += p.dx;
      p.y += p.dy;
      if (p.x < 0 || p.x > W) p.dx *= -1;
      if (p.y < 0 || p.y > H) p.dy *= -1;
    });
    ctx.globalAlpha = 1;
    requestAnimationFrame(draw);
  }
  draw();
})();

/* ══════════════════
   NAVBAR SCROLL
══════════════════ */
window.addEventListener('scroll', () => {
  const nav = document.getElementById('navbar');
  if (window.scrollY > 40) {
    nav.style.background = 'rgba(8,12,10,0.95)';
  } else {
    nav.style.background = 'rgba(8,12,10,0.72)';
  }
});

/* ══════════════════
   COUNTER ANIMATION
══════════════════ */
function animateCounters() {
  document.querySelectorAll('.stat-num').forEach(el => {
    const target = parseFloat(el.dataset.target);
    const isFloat = target % 1 !== 0;
    let current   = 0;
    const step    = target / 60;
    const timer   = setInterval(() => {
      current += step;
      if (current >= target) { current = target; clearInterval(timer); }
      el.textContent = isFloat ? current.toFixed(1) : Math.floor(current);
    }, 20);
  });
}

/* Intersection observer for stats */
const statsSection = document.getElementById('stats-bar');
const statsObserver = new IntersectionObserver(entries => {
  if (entries[0].isIntersecting) {
    animateCounters();
    statsObserver.disconnect();
  }
}, { threshold: 0.4 });
statsObserver.observe(statsSection);

/* ══════════════════
   DETECTION STATE
══════════════════ */
let timerInterval = null;
let startTime     = null;
let isRunning     = false;
let modelsReady   = false;

let mediaStream     = null;
let captureTimer    = null;
let inFlight         = false;   // avoid overlapping requests to the server
const CAPTURE_INTERVAL_MS = 600;
let lastBeepTime     = 0;
const BEEP_COOLDOWN_MS = 2000;

/* Simple beep using the Web Audio API (no audio file needed) */
function playBeep() {
  const now = Date.now();
  if (now - lastBeepTime < BEEP_COOLDOWN_MS) return;
  lastBeepTime = now;
  try {
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.3);
    osc.connect(gain).connect(ctx.destination);
    osc.start();
    osc.stop(ctx.currentTime + 0.3);
  } catch (_) { /* ignore if audio isn't available */ }
}

// Disable detect buttons until models are ready
function setModelsReady(ready) {
  modelsReady = ready;
  document.querySelectorAll('#detect-btn, #cta-detect-btn').forEach(btn => {
    if (ready) {
      btn.disabled = false;
      btn.querySelector('.btn-label').textContent = 'Start Detection';
      btn.style.opacity = '1';
      btn.style.cursor  = 'pointer';
    } else {
      btn.disabled = true;
      btn.querySelector('.btn-label').textContent = 'Loading AI Models...';
      btn.style.opacity = '0.6';
      btn.style.cursor  = 'not-allowed';
    }
  });
}

function updateTimer() {
  const elapsed = Math.floor((Date.now() - startTime) / 1000);
  const mm      = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss      = String(elapsed % 60).padStart(2, '0');
  const el = document.getElementById('timer-val');
  if (el) el.textContent = `${mm}:${ss}`;
}

function setDetectionUI(running) {
  isRunning = running;
  const detectBtns = document.querySelectorAll('#detect-btn, #cta-detect-btn');
  const stopBtn    = document.getElementById('stop-btn');
  const badge      = document.getElementById('badge-text');
  const statusIcon = document.getElementById('status-icon');
  const statusMsg  = document.getElementById('status-msg');
  const timerDisp  = document.getElementById('timer-display');
  const detLabel   = document.getElementById('det-label');

  if (running) {
    document.body.classList.add('detecting');
    detectBtns.forEach(b => { b.style.display = 'none'; });
    if (stopBtn) stopBtn.style.display = 'inline-flex';
    if (badge)      badge.textContent   = 'Detection Active';
    if (statusIcon) statusIcon.textContent = '🟡';
    if (statusMsg)  statusMsg.textContent  = 'Scanning your camera feed...';
    if (timerDisp)  timerDisp.style.display = 'block';
    if (detLabel)   detLabel.style.display  = 'block';
    startTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);
  } else {
    document.body.classList.remove('detecting');
    detectBtns.forEach(b => { b.style.display = 'inline-flex'; });
    if (stopBtn)    stopBtn.style.display  = 'none';
    if (badge)      badge.textContent      = 'System Ready';
    if (statusIcon) statusIcon.textContent = '🟢';
    if (statusMsg)  statusMsg.textContent  = 'Idle — Press Start Detection';
    if (timerDisp)  timerDisp.style.display = 'none';
    if (detLabel)   detLabel.style.display  = 'none';
    clearInterval(timerInterval);
  }
}

function showToast(msg, icon = '✅') {
  const toast = document.getElementById('toast');
  document.getElementById('toast-msg').textContent  = msg;
  document.getElementById('toast-icon').textContent = icon;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), 3500);
}

/* ══════════════════
   CAMERA + DETECTION LOOP
   (browser owns the webcam; server only scores frames)
══════════════════ */
const videoEl  = document.getElementById('webcam-video');
const overlayEl = document.getElementById('overlay-canvas');
const heroImgEl = document.getElementById('hero-leo-img');
const captureCanvas = document.createElement('canvas'); // off-screen, for sending frames

async function startDetection() {
  if (isRunning || !modelsReady) return;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showToast('This browser cannot access a camera. Try Chrome/Edge/Safari over HTTPS.', '❌');
    return;
  }

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 }, audio: false });
  } catch (err) {
    showToast('Camera permission denied or unavailable.', '❌');
    console.error(err);
    return;
  }

  videoEl.srcObject = mediaStream;
  heroImgEl.style.display = 'none';
  videoEl.style.display = 'block';
  overlayEl.style.display = 'block';

  await videoEl.play();
  overlayEl.width  = videoEl.videoWidth  || 640;
  overlayEl.height = videoEl.videoHeight || 480;

  setDetectionUI(true);
  showToast('Camera live! Scanning for leopards...', '🐆');
  captureLoop();
}

function stopDetection() {
  clearTimeout(captureTimer);
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  videoEl.style.display = 'none';
  overlayEl.style.display = 'none';
  heroImgEl.style.display = 'block';
  const ctx = overlayEl.getContext('2d');
  ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);

  setDetectionUI(false);
  showToast('Detection stopped.', '⏹️');
}

/* Grab a frame, send it to the server, draw the result, repeat */
function captureLoop() {
  if (!isRunning) return;
  captureTimer = setTimeout(async () => {
    await captureAndSendFrame();
    captureLoop();
  }, CAPTURE_INTERVAL_MS);
}

async function captureAndSendFrame() {
  if (inFlight || !videoEl.videoWidth) return;
  inFlight = true;

  captureCanvas.width  = videoEl.videoWidth;
  captureCanvas.height = videoEl.videoHeight;
  const cctx = captureCanvas.getContext('2d');
  cctx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);
  const dataUrl = captureCanvas.toDataURL('image/jpeg', 0.7);

  try {
    const res  = await fetch('/api/detect/frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl }),
    });
    const data = await res.json();
    if (data.success) drawDetection(data);
  } catch (err) {
    // transient network hiccup — just skip this frame
  } finally {
    inFlight = false;
  }
}

function drawDetection(result) {
  const ctx = overlayEl.getContext('2d');
  ctx.clearRect(0, 0, overlayEl.width, overlayEl.height);

  const detLabel  = document.getElementById('det-label');
  const statusMsg = document.getElementById('status-msg');

  if (result.leopard_found && result.box) {
    const [x1, y1, x2, y2] = result.box;
    ctx.strokeStyle = '#00e87a';
    ctx.lineWidth = 3;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

    const label = `LEOPARD  ${(result.confidence * 100).toFixed(0)}%`;
    ctx.font = 'bold 16px Outfit, sans-serif';
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fillRect(x1, y1 - 26, textW + 12, 22);
    ctx.fillStyle = '#00e87a';
    ctx.fillText(label, x1 + 6, y1 - 9);

    if (detLabel) detLabel.style.display = 'block';
    if (statusMsg) statusMsg.textContent = `Leopard detected — ${(result.confidence * 100).toFixed(0)}% confidence`;
    playBeep();
  } else {
    if (detLabel) detLabel.style.display = 'none';
    if (statusMsg) statusMsg.textContent = 'Scanning your camera feed...';
  }
}

/* Initial status check on load — polls until models ready */
async function pollModelsReady() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    if (data.models_ready) {
      setModelsReady(true);
      const el = document.getElementById('status-msg');
      if (el && !isRunning) el.textContent = 'Models ready — click Start Detection!';
      return;
    }
  } catch (_) {}
  setTimeout(pollModelsReady, 1000);
}

window.addEventListener('DOMContentLoaded', () => {
  setModelsReady(false); // start disabled
  pollModelsReady();
});

/* ══════════════════
   SMOOTH SECTION REVEAL
══════════════════ */
const revealEls = document.querySelectorAll('.feature-card, .pipe-step, .gallery-item, .tech-stack');
const revealObserver = new IntersectionObserver(entries => {
  entries.forEach(e => {
    if (e.isIntersecting) {
      e.target.style.opacity = '1';
      e.target.style.transform = 'translateY(0)';
    }
  });
}, { threshold: 0.1 });

revealEls.forEach(el => {
  el.style.opacity   = '0';
  el.style.transform = 'translateY(28px)';
  el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
  revealObserver.observe(el);
});
