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
let timerInterval   = null;
let startTime       = null;
let isRunning        = false;
let modelsReady      = false;
let mediaStream       = null;
let captureLoopId    = null;
let inFlight         = false;
let lastBeepTime     = 0;
const BEEP_COOLDOWN_MS = 2000;
const CAPTURE_INTERVAL_MS = 600;   // ~1.6 fps — tuned for CPU inference on a free-tier server

let audioCtx = null;
function beep() {
  try {
    audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)();
    const osc  = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = 'sine';
    osc.frequency.value = 880;
    gain.gain.setValueAtTime(0.15, audioCtx.currentTime);
    osc.connect(gain).connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.25);
  } catch (_) {}
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
    if (statusMsg)  statusMsg.textContent  = 'Detection running — scanning your camera feed';
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
   WEBCAM + LIVE DETECTION (runs entirely via browser camera + server inference)
══════════════════ */
async function startDetection() {
  if (isRunning || !modelsReady) return;

  const video   = document.getElementById('webcam-video');
  const canvas  = document.getElementById('overlay-canvas');
  const heroImg = document.getElementById('hero-leo-img');

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'environment' },
      audio: false
    });
  } catch (err) {
    showToast('Camera access denied or unavailable.', '❌');
    console.error(err);
    return;
  }

  video.srcObject = mediaStream;
  await video.play();

  video.style.display  = 'block';
  canvas.style.display = 'block';
  if (heroImg) heroImg.style.display = 'none';

  canvas.width  = video.videoWidth  || 640;
  canvas.height = video.videoHeight || 480;

  setDetectionUI(true);
  showToast('Camera active! Scanning for leopards...', '🐆');
  runCaptureLoop(video, canvas);
}

function stopDetection() {
  if (captureLoopId) { clearTimeout(captureLoopId); captureLoopId = null; }
  if (mediaStream) {
    mediaStream.getTracks().forEach(t => t.stop());
    mediaStream = null;
  }
  const video   = document.getElementById('webcam-video');
  const canvas  = document.getElementById('overlay-canvas');
  const heroImg = document.getElementById('hero-leo-img');
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  video.style.display  = 'none';
  canvas.style.display = 'none';
  if (heroImg) heroImg.style.display = 'block';

  setDetectionUI(false);
  showToast('Detection stopped.', '⏹️');
}

// Offscreen canvas used purely to grab JPEG bytes from the video element
const grabCanvas = document.createElement('canvas');
const grabCtx     = grabCanvas.getContext('2d');

async function captureAndSend(video, overlayCanvas) {
  if (inFlight) return;
  inFlight = true;
  try {
    grabCanvas.width  = video.videoWidth;
    grabCanvas.height = video.videoHeight;
    grabCtx.drawImage(video, 0, 0, grabCanvas.width, grabCanvas.height);
    const dataUrl = grabCanvas.toDataURL('image/jpeg', 0.7);

    const res  = await fetch('/api/detect_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });
    const data = await res.json();

    if (data.success) {
      drawDetection(overlayCanvas, data);
    }
  } catch (err) {
    // transient network hiccup — keep looping
  } finally {
    inFlight = false;
  }
}

function drawDetection(overlayCanvas, data) {
  const ctx = overlayCanvas.getContext('2d');
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  const detLabel = document.getElementById('det-label');

  if (data.leopard && data.box) {
    const [x1, y1, x2, y2] = data.box;
    const scaleX = overlayCanvas.width  / data.frame_width;
    const scaleY = overlayCanvas.height / data.frame_height;

    ctx.strokeStyle = '#00e87a';
    ctx.lineWidth   = 3;
    ctx.strokeRect(x1 * scaleX, y1 * scaleY, (x2 - x1) * scaleX, (y2 - y1) * scaleY);

    const label = `LEOPARD ${(data.confidence * 100).toFixed(0)}%`;
    ctx.font = 'bold 16px Outfit, sans-serif';
    const textW = ctx.measureText(label).width;
    ctx.fillStyle = 'rgba(0,0,0,0.6)';
    ctx.fillRect(x1 * scaleX - 2, y1 * scaleY - 24, textW + 12, 22);
    ctx.fillStyle = '#00e87a';
    // Un-mirror the text so it reads correctly against the mirrored canvas
    ctx.save();
    ctx.translate(x1 * scaleX + textW + 8, y1 * scaleY - 8);
    ctx.scale(-1, 1);
    ctx.fillText(label, 0, 0);
    ctx.restore();

    if (detLabel) detLabel.style.display = 'block';

    const now = Date.now();
    if (now - lastBeepTime > BEEP_COOLDOWN_MS) {
      lastBeepTime = now;
      beep();
    }
  } else {
    if (detLabel) detLabel.style.display = 'none';
  }
}

function runCaptureLoop(video, overlayCanvas) {
  if (!isRunning) return;
  captureAndSend(video, overlayCanvas);
  captureLoopId = setTimeout(() => runCaptureLoop(video, overlayCanvas), CAPTURE_INTERVAL_MS);
}

/* Poll until models are ready on the server */
async function pollModelsReady() {

  try {

      const res = await fetch("/api/status");

      const data = await res.json();

      console.log(data);

      if (data.models_ready) {

          setModelsReady(true);

          const el = document.getElementById("status-msg");

          if (el) {

              el.textContent = "AI Models Ready";

          }

          return;

      }

      if (data.error) {

          console.error(data.error);

          const el = document.getElementById("status-msg");

          if (el) {

              el.textContent = data.error;

          }

          return;

      }

  }

  catch(err){

      console.error(err);

  }

  setTimeout(pollModelsReady,3000);

}

window.addEventListener("DOMContentLoaded", () => {

  setModelsReady(false);

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
