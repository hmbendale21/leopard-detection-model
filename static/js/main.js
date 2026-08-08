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
    if (statusMsg)  statusMsg.textContent  = 'Detection running — camera window is open';
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
   API CALLS
══════════════════ */
async function startDetection() {
  if (isRunning || !modelsReady) return;

  showToast('Opening camera instantly...', '🐆');

  try {
    const res  = await fetch('/api/detect/start', { method: 'POST' });
    const data = await res.json();

    if (data.success) {
      setDetectionUI(true);
      showToast('Camera opened! Detection running.', '✅');
      pollStatus();
    } else {
      showToast(data.message || 'Failed to start.', '❌');
    }
  } catch (err) {
    showToast('Server error — is app.py running?', '❌');
    console.error(err);
  }
}

async function stopDetection() {
  try {
    const res  = await fetch('/api/detect/stop', { method: 'POST' });
    const data = await res.json();
    setDetectionUI(false);
    showToast(data.message || 'Detection stopped.', '⏹️');
  } catch (err) {
    showToast('Could not stop detection.', '❌');
  }
}

/* Poll the server every 3 s to sync state */
function pollStatus() {
  if (!isRunning) return;
  setTimeout(async () => {
    try {
      const res  = await fetch('/api/status');
      const data = await res.json();
      if (!data.running && isRunning) {
        setDetectionUI(false);
        showToast('Detection finished or camera closed.', 'ℹ️');
        return;
      }
    } catch (_) {}
    pollStatus();
  }, 3000);
}

/* Initial status check on load — also polls until models ready */
async function pollModelsReady() {
  try {
    const res  = await fetch('/api/status');
    const data = await res.json();
    if (data.models_ready) {
      setModelsReady(true);
      const el = document.getElementById('status-msg');
      if (el && !isRunning) el.textContent = 'Models ready — click Detect for instant camera!';
      if (data.running) { setDetectionUI(true); pollStatus(); }
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
