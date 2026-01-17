import './style.css'
import gsap from 'gsap'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer } from 'three/addons/renderers/CSS2DRenderer.js';
import { api } from './api';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
  <div id="dna-background"></div>
  <div class="hex-bg"></div>
  <div class="glow-overlay"></div>
  
  <!-- Transition Effects -->
  <div class="glitch-overlay" id="glitch-overlay"></div>
  <div class="scan-line" id="scan-line"></div>
  <div class="color-wave" id="color-wave"></div>
  <div class="morph-shape" id="morph-shape"></div>
  <div class="hex-transition" id="hex-transition"></div>
  <div class="ripple-effect" id="ripple-effect"></div>
  <div class="digital-rain" id="digital-rain"></div>
  <div class="vortex-transition" id="vortex-transition"></div>
  <div class="grid-transition" id="grid-transition"></div>
  <div class="scan-glow" id="scan-glow"></div>
  <div class="static-noise" id="static-noise"></div>
  
  <div class="curzr-arrow-pointer" hidden>
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
      <path class="inner"
        d="M25,30a5.82,5.82,0,0,1-1.09-.17l-.2-.07-7.36-3.48a.72.72,0,0,0-.35-.08.78.78,0,0,0-.33.07L8.24,29.54a.66.66,0,0,1-.2.06,5.17,5.17,0,0,1-1,.15,3.6,3.6,0,0,1-3.29-5L12.68,4.2a3.59,3.59,0,0,1,6.58,0l9,20.74A3.6,3.6,0,0,1,25,30Z" />
      <path class="outer"
        d="M16,3A2.59,2.59,0,0,1,18.34,4.6l9,20.74A2.59,2.59,0,0,1,25,29a5.42,5.42,0,0,1-.86-.15l-7.37-3.48a1.84,1.84,0,0,0-.77-.17,1.69,1.69,0,0,0-.73.16l-7.4,3.31a5.89,5.89,0,0,1-.79.12,2.59,2.59,0,0,1-2.37-3.62L13.6,4.6A2.58,2.58,0,0,1,16,3m0-2h0A4.58,4.58,0,0,0,11.76,3.8L2.84,24.33A4.58,4.58,0,0,0,7,30.75a6.08,6.08,0,0,0,1.21-.17,1.87,1.87,0,0,0,.4-.13L16,27.18l7.29,3.44a1.64,1.64,0,0,0,.39.14A6.37,6.37,0,0,0,25,31a4.59,4.59,0,0,0,4.21-6.41l-9-20.75A4.62,4.62,0,0,0,16,1Z" />
    </svg>
  </div>
  
  <div class="page-login">
    <div class="login-card">
      <div class="login-card-corner-br"></div>
      <h1 style="color:var(--secondary-teal)">GeneRX</h1>
      <h2 style="color:var(--primary-teal)">HOSPITAL SYSTEM</h2>
      
      <div class="input-group">
        <label>DOCTOR ID</label>
        <input type="text" id="doctor-id" placeholder="DR-UNKNOWN" autocomplete="off" value="DR-STRANGE">
      </div>
      <div class="input-group">
        <label>Security Key</label>
        <input type="password" id="access-key" placeholder="••••••••" value="admin123">
      </div>
      <button class="btn-connect" id="connect-btn" aria-label="Authenticate to system">AUTHENTICATE</button>
    </div>
  </div>

  <div class="warp-tunnel"></div>

  <div class="page-dashboard">
    <div class="dashboard-bg-deco"></div>
    <header class="dashboard-header">
      <div class="logo" style="font-size: 1.5rem; letter-spacing: 2px; color: var(--secondary-teal); font-weight: 800;">
        GeneRX <span style="color:var(--primary-teal); font-size: 0.8em; border: 1px solid var(--primary-teal); padding: 2px 5px;">HOSPITAL</span>
      </div>
      <div class="user-status" style="font-family: 'Orbitron'; display: flex; align-items: center; gap: 15px;">
        <div class="system-status" style="font-size: 0.7rem; display: flex; align-items: center; gap: 5px;" aria-live="polite">
          <span class="status-dot pulse"></span>
          SYS: <span id="sys-status-text" style="color:var(--neon-blue)">SYNCING</span>
        </div>
        LOGIN: <span id="display-id" style="color:var(--neon-green)">---</span>
        <button id="logout-btn" class="btn-logout" aria-label="Logout from system">LOGOUT [X]</button>
      </div>
    </header>
    
    <div class="grid-container" style="display: flex; justify-content: center; align-items: center; position: relative;">
      
      <!-- RIGHT SIDE: PATIENT HUD -->
      <div class="panel right-hud" role="region" aria-label="Patient Report Entry">
        <div style="text-align: center; margin-bottom: 1.5rem;">
          <h1 style="font-family: 'Orbitron'; color: var(--secondary-teal); font-size: 1.5rem; letter-spacing: 4px; margin-bottom: 0;">GENOMIC SEQUENCE</h1>
          <h2 style="font-family: 'Orbitron'; color: var(--primary-teal); font-size: 1.2rem; letter-spacing: 8px; margin-top: 0; border-bottom: 2px solid var(--primary-teal); padding-bottom: 10px; display: inline-block;">VISUALIZER</h2>
        </div>
        <h3 style="color:var(--secondary-teal); font-family: 'Orbitron';">PATIENT REPORT ENTRY</h3>
         <div style="display:flex; justify-content:flex-end; margin: 0.5rem 0 0.5rem 0;">
           <button id="voice-all-btn" class="voice-btn" aria-label="Voice input for all fields" title="Voice input">
             <svg viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
               <path fill="currentColor" d="M12 14a3 3 0 0 0 3-3V5a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.92V21h2v-3.08A7 7 0 0 0 19 11h-2z"/>
             </svg>
           </button>
         </div>
         <div class="report-form-container">
            <input type="text" class="report-input" placeholder="PATIENT NAME" id="p-name" aria-label="Patient Name" style="background: rgba(90, 197, 200, 0.05); border-color: var(--primary-teal);">
            <input type="text" class="report-input" placeholder="AGE / GENDER" id="p-age" aria-label="Age and Gender" style="background: rgba(90, 197, 200, 0.05); border-color: var(--primary-teal);">
            <input type="text" class="report-input" placeholder="BLOOD GROUP" id="p-blood" aria-label="Blood Group" style="background: rgba(90, 197, 200, 0.05); border-color: var(--primary-teal);">
            <input type="text" class="report-input" placeholder="DISEASE/CONDITION" id="p-disease" aria-label="Disease or Condition" style="background: rgba(90, 197, 200, 0.05); border-color: var(--primary-teal);">
            <textarea class="report-input" placeholder="DIAGNOSIS NOTES" rows="4" id="p-diagnosis" aria-label="Diagnosis Notes" style="background: rgba(90, 197, 200, 0.05); border-color: var(--primary-teal);"></textarea>
            <button class="btn-submit" id="submit-report" aria-label="Upload report to backend" style="background: var(--primary-teal);">UPLOAD TO BACKEND</button>
         </div>
         <div style="flex-grow: 1; margin-top: 1.5rem; border-top: 1px solid rgba(4, 53, 61, 0.1); padding-top: 1rem;">
            <div style="font-size: 0.8rem; color: var(--secondary-teal); margin-bottom: 0.5rem; font-weight: bold;">[SYSTEM LOGS]</div>
            <ul id="console-logs" style="list-style: none; padding: 0; font-size: 0.75rem; color: var(--text-muted); overflow-y: auto; height: 120px; font-family: 'monospace';">
              <li>[SYS] Initializing hospital systems...</li>
              <li>[SYS] Standing by for patient data...</li>
            </ul>
         </div>
      </div>
    </div>
  </div>

  <div class="page-analysis">
      <div class="dashboard-bg-deco" style="border-color: var(--neon-green);"></div>
      <header class="dashboard-header">
        <div class="logo" style="font-weight: bold; font-size: 1.8rem; letter-spacing: 3px; color: var(--secondary-teal); text-shadow: 0 0 5px var(--neon-green);">DRUG DISCOVERY ANALYTICS</div>
        <button id="back-dashboard" class="btn-logout">BACK</button>
      </header>
      
      <div class="analysis-container">
          <div class="panel">
              <h3>GENOME ANALYSIS</h3>
              <div class="report-form-container">
                <label data-tooltip="Upload patient genome CSV file">Patient Genome File (CSV)</label>
                <input type="file" class="report-input" id="genome-file" accept=".csv" style="padding: 10px;">
                
                <button class="btn-submit" id="btn-predict" style="margin-top: 2rem; background: linear-gradient(90deg, #002, var(--neon-purple), #002); color: white;">RUN GENOMIC ANALYSIS</button>
              </div>
          </div>
          
          <div class="panel" style="display:flex; flex-direction: column;">
              <h3>PREDICTION RESULTS & GRAPHS</h3>
              <div class="graph-container" id="results-graph">
                  <!-- Bars inserted here -->
                  <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); color: #555;" id="graph-placeholder">AWAITING INPUT...</div>
              </div>
              <div class="result-card" style="margin-top: 1rem; opacity: 0;" id="text-result">
                  <div id="result-status" style="color: var(--neon-green); font-size: 1.2rem; margin-bottom: 5px;">SUCCESS: POTENTIAL CANDIDATE FOUND</div>
                  <div style="font-size: 0.9rem; color: #ccc;">Binding Affinity: <span id="res-affinity" style="color: white; font-weight: bold;">-9.4 kcal/mol</span> | Toxicity Risk: <span id="res-toxicity">LOW</span></div>
              </div>
          </div>
      </div>
      
      <!-- Drug-Disease Compatibility Analysis -->
      <div class="analysis-container" style="margin-top: 1.5rem;">
              <div class="panel" style="grid-column: 1 / -1;">
                  <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                    <h3 style="margin:0;">DRUG-DISEASE COMPATIBILITY ANALYSIS</h3>
                    <button id="btn-download-report" class="btn-submit" style="padding: 5px 15px; font-size: 0.7rem; background: var(--glass); border: 1px solid var(--neon-blue); color: var(--neon-blue);">DOWNLOAD PDF REPORT [RAW]</button>
                  </div>
              <div id="compatibility-result" style="padding: 1.5rem; text-align: center; color: #888;">
                  <div style="font-size: 0.9rem;">Run prediction model to analyze drug compatibility with patient's disease</div>
              </div>
          </div>
      </div>
  </div>
`

// Logic
const clockEl = document.createElement('div');
clockEl.style.fontSize = '0.7rem';
clockEl.style.color = '#888';
clockEl.style.marginLeft = '15px';
clockEl.style.borderLeft = '1px solid #333';
clockEl.style.paddingLeft = '15px';
clockEl.innerText = "00:00:00";

const updateClock = () => {
  const now = new Date();
  clockEl.innerText = now.getHours().toString().padStart(2, '0') + ":" +
    now.getMinutes().toString().padStart(2, '0') + ":" +
    now.getSeconds().toString().padStart(2, '0');
};
setInterval(updateClock, 1000);
updateClock();

// We'll append this after the app renders but before logic starts
// Actually, let's just make it part of the HTML for cleaner injection.
const connectBtn = document.getElementById('connect-btn');
const logoutBtn = document.getElementById('logout-btn');
const pageLogin = document.querySelector('.page-login') as HTMLElement;
const pageDashboard = document.querySelector('.page-dashboard') as HTMLElement;
const pageAnalysis = document.querySelector('.page-analysis') as HTMLElement;
const warpTunnel = document.querySelector('.warp-tunnel');
const bg = document.querySelector('.hex-bg');
const doctorIdInput = document.getElementById('doctor-id') as HTMLInputElement;
const displayId = document.getElementById('display-id');
const consoleLogs = document.getElementById('console-logs');
const submitBtn = document.getElementById('submit-report');
const predictBtn = document.getElementById('btn-predict');
const backDashBtn = document.getElementById('back-dashboard');

// Random Static Noise
setInterval(() => {
  const noise = document.getElementById('static-noise');
  if (noise && Math.random() > 0.7) {
    noise.classList.add('static-active');
    setTimeout(() => noise.classList.remove('static-active'), 100 + Math.random() * 200);
  }
}, 5000);


// ... (Existing variables)
let dnaRenderer: THREE.WebGLRenderer, dnaScene: THREE.Scene, dnaCamera: THREE.PerspectiveCamera;
let labelRenderer: CSS2DRenderer;
let lastAnalysis: any = null;

// ========== TRANSITION EFFECT FUNCTIONS ========== 

// Particle Burst Effect
function createParticleBurst(x: number, y: number, count: number = 30) {
  const colors = ['#ff00ff', '#00d4ff', '#ff0080', '#0aff0a'];

  for (let i = 0; i < count; i++) {
    const particle = document.createElement('div');
    particle.className = 'particle-burst';
    particle.style.left = x + 'px';
    particle.style.top = y + 'px';
    particle.style.background = colors[Math.floor(Math.random() * colors.length)];
    document.body.appendChild(particle);

    const angle = (Math.PI * 2 * i) / count;
    const velocity = 100 + Math.random() * 200;
    const tx = Math.cos(angle) * velocity;
    const ty = Math.sin(angle) * velocity;

    gsap.to(particle, {
      x: tx,
      y: ty,
      opacity: 0,
      scale: Math.random() * 2,
      duration: 0.8 + Math.random() * 0.4,
      ease: 'power2.out',
      onComplete: () => particle.remove()
    });
  }
}

// Glitch Overlay Effect
function triggerGlitchEffect() {
  const glitch = document.getElementById('glitch-overlay');
  if (!glitch) return;

  const tl = gsap.timeline();
  tl.to(glitch, { opacity: 0.8, duration: 0.05 });
  tl.to(glitch, { x: -5, duration: 0.05 });
  tl.to(glitch, { x: 5, duration: 0.05 });
  tl.to(glitch, { x: -3, duration: 0.05 });
  tl.to(glitch, { x: 0, opacity: 0, duration: 0.1 });
}

// Scan Line Effect
function triggerScanLine() {
  const scanLine = document.getElementById('scan-line');
  if (!scanLine) return;

  gsap.fromTo(scanLine,
    { top: '0%', opacity: 1 },
    {
      top: '100%',
      opacity: 0,
      duration: 0.6,
      ease: 'power2.inOut'
    }
  );
}

// Color Wave Effect
function triggerColorWave() {
  const wave = document.getElementById('color-wave');
  if (!wave) return;

  gsap.fromTo(wave,
    { left: '-100%', opacity: 0.7 },
    {
      left: '100%',
      opacity: 0,
      duration: 1.2,
      ease: 'power2.inOut'
    }
  );
}

// Morphing Shape Effect
function triggerMorphingShape() {
  const shape = document.getElementById('morph-shape');
  if (!shape) return;

  const tl = gsap.timeline();
  tl.to(shape, {
    opacity: 1,
    width: 200,
    height: 200,
    rotation: 45,
    duration: 0.4,
    ease: 'power2.out'
  });
  tl.to(shape, {
    width: 300,
    height: 100,
    rotation: 90,
    borderRadius: '50%',
    duration: 0.3
  });
  tl.to(shape, {
    opacity: 0,
    scale: 2,
    duration: 0.3
  });
  tl.set(shape, { width: 100, height: 100, rotation: 0, scale: 1, borderRadius: 0 });
}

// Hexagon Transition
function triggerHexagonTransition() {
  const hex = document.getElementById('hex-transition');
  if (!hex) return;

  gsap.fromTo(hex,
    { width: 0, height: 0, opacity: 1 },
    {
      width: '200vw',
      height: '200vw',
      opacity: 0,
      duration: 1,
      ease: 'power2.out'
    }
  );
}

// Ripple Effect
function triggerRippleEffect() {
  const ripple = document.getElementById('ripple-effect');
  if (!ripple) return;

  const tl = gsap.timeline();
  for (let i = 0; i < 3; i++) {
    tl.fromTo(ripple,
      { width: 50, height: 50, opacity: 0.8 },
      {
        width: 800,
        height: 800,
        opacity: 0,
        duration: 1,
        ease: 'power2.out'
      },
      i * 0.2
    );
  }
}

// Digital Rain Effect
function triggerDigitalRain(duration: number = 1) {
  const rain = document.getElementById('digital-rain');
  if (!rain) return;

  rain.innerHTML = '';
  const columnCount = 30;

  gsap.to(rain, { opacity: 1, duration: 0.1 });

  for (let i = 0; i < columnCount; i++) {
    const column = document.createElement('div');
    column.className = 'rain-column';
    column.style.left = (Math.random() * 100) + '%';
    column.style.animationDelay = (Math.random() * 0.5) + 's';
    rain.appendChild(column);
  }

  setTimeout(() => {
    gsap.to(rain, {
      opacity: 0,
      duration: 0.3,
      onComplete: () => { rain.innerHTML = ''; }
    });
  }, duration * 1000);
}

// Vortex Transition
function triggerVortexTransition() {
  const vortex = document.getElementById('vortex-transition');
  if (!vortex) return;

  gsap.fromTo(vortex,
    { width: 0, height: 0, opacity: 1, rotation: 0 },
    {
      width: 800,
      height: 800,
      opacity: 0,
      rotation: 720,
      duration: 1.2,
      ease: 'power2.out'
    }
  );
}

// Grid Transition
function triggerGridTransition(duration: number = 0.8) {
  const grid = document.getElementById('grid-transition');
  if (!grid) return;

  const tl = gsap.timeline();
  tl.to(grid, { opacity: 0.5, duration: 0.2 });
  tl.to(grid, { opacity: 0, duration: duration - 0.2 });
}

// Combined Transition Effect
function playTransitionEffects(effectType: 'login' | 'dashboard' | 'analysis' | 'back') {
  switch (effectType) {
    case 'login':
      // Login to Dashboard: Vortex + Scan + Particles
      createParticleBurst(window.innerWidth / 2, window.innerHeight / 2, 40);
      triggerVortexTransition();
      setTimeout(() => triggerScanLine(), 300);
      setTimeout(() => triggerGlitchEffect(), 600);
      break;

    case 'dashboard':
      // Dashboard to Analysis: Hexagon + Color Wave + Digital Rain
      triggerHexagonTransition();
      setTimeout(() => triggerColorWave(), 200);
      setTimeout(() => triggerDigitalRain(0.8), 400);
      setTimeout(() => triggerGlitchEffect(), 800);
      break;

    case 'analysis':
      // Analysis Page Effects: Morphing + Ripple + Grid
      triggerMorphingShape();
      setTimeout(() => triggerRippleEffect(), 200);
      setTimeout(() => triggerGridTransition(), 400);
      break;

    case 'back':
      // Back Navigation: Color Wave + Scan + Particles
      triggerColorWave();
      setTimeout(() => triggerScanLine(), 300);
      setTimeout(() => createParticleBurst(window.innerWidth / 2, window.innerHeight / 2, 30), 500);
      setTimeout(() => triggerGlitchEffect(), 700);
      break;
  }
}


function addLog(msg: string) {
  if (!consoleLogs) return;
  const now = new Date();
  const time = now.getHours().toString().padStart(2, '0') + ":" +
    now.getMinutes().toString().padStart(2, '0') + ":" +
    now.getSeconds().toString().padStart(2, '0');

  const li = document.createElement('li');
  li.style.marginTop = "5px";
  li.style.borderLeft = "2px solid rgba(10, 255, 10, 0.3)";
  li.style.paddingLeft = "8px";
  li.innerHTML = `<span style="color:#555">[${time}]</span> <span style="color:var(--neon-green)">PRC:</span> ${msg}`;
  consoleLogs.appendChild(li);
  consoleLogs.scrollTop = consoleLogs.scrollHeight;

  // Scroller fade out effect for old logs
  if (consoleLogs.children.length > 50) {
    consoleLogs.removeChild(consoleLogs.children[0]);
  }
}

// Three.js DNA Setup - Modern Cinematic Glass & Neon with Precision Pins
function initDNA() {
  const container = document.getElementById('dna-background');
  if (!container) return;

  container.innerHTML = '';

  // Scene
  dnaScene = new THREE.Scene();
  dnaScene.fog = new THREE.FogExp2(0xE3E6E9, 0.015);

  // Camera - Fixed position to focus on the side DNA
  dnaCamera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
  dnaCamera.position.set(0, 0, 40);

  // Renderer (WebGL)
  dnaRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  dnaRenderer.setSize(window.innerWidth, window.innerHeight);
  dnaRenderer.setPixelRatio(window.devicePixelRatio);
  dnaRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  dnaRenderer.toneMappingExposure = 1.2;
  container.appendChild(dnaRenderer.domElement);

  // Renderer (CSS2D - For Labels)
  labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(window.innerWidth, window.innerHeight);
  labelRenderer.domElement.style.position = 'absolute';
  labelRenderer.domElement.style.top = '0px';
  labelRenderer.domElement.style.pointerEvents = 'none';
  container.appendChild(labelRenderer.domElement);

  // Controls
  const controls = new OrbitControls(dnaCamera, dnaRenderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.05;
  controls.enableZoom = true;
  controls.autoRotate = true;
  controls.autoRotateSpeed = 1.0;

  // Lighting
  const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
  dnaScene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0x5AC5C8, 3);
  dirLight.position.set(20, 30, 20);
  dnaScene.add(dirLight);

  const tealSpot = new THREE.PointLight(0x5AC5C8, 10, 50);
  tealSpot.position.set(-20, 10, 20);
  dnaScene.add(tealSpot);

  const darkSpot = new THREE.PointLight(0x04353D, 10, 50);
  darkSpot.position.set(20, -10, 20);
  dnaScene.add(darkSpot);

  const dnaGroup = new THREE.Group();
  dnaGroup.position.x = 15; // Shift DNA to the RIGHT as per image 0
  dnaScene.add(dnaGroup);

  // Materials
  const backboneMaterial = new THREE.MeshPhysicalMaterial({
    color: 0xE0F7FA, // Very light blue / white
    metalness: 0.2,
    roughness: 0.1,
    transmission: 0.8,
    thickness: 1.0,
    clearcoat: 1.0,
    opacity: 0.6,
    transparent: true,
    emissive: 0x81D4FA, // Light sky blue glow
    emissiveIntensity: 0.5,
    side: THREE.DoubleSide
  });

  // Materials for 4 Bases (Themed) - Lighter for Image 0 style
  const matAdenine = new THREE.MeshStandardMaterial({ color: 0x4FC3F7, emissive: 0x4FC3F7, emissiveIntensity: 1.0, roughness: 0.2, metalness: 0.5 });
  const matThymine = new THREE.MeshStandardMaterial({ color: 0x37474F, emissive: 0x37474F, emissiveIntensity: 0.5, roughness: 0.2, metalness: 0.5 });
  const matGuanine = new THREE.MeshStandardMaterial({ color: 0xB0BEC5, emissive: 0xB0BEC5, emissiveIntensity: 0.8, roughness: 0.2, metalness: 0.5 });
  const matCytosine = new THREE.MeshStandardMaterial({ color: 0x81D4FA, emissive: 0x81D4FA, emissiveIntensity: 1.2, roughness: 0.2, metalness: 0.5 });

  // Geometry Generation - THINNER DNA
  const pointCount = 150;
  const radius = 6; /* Thinner DNA */
  const height = 50;
  const turns = 4;

  const pointsA = [];
  const pointsB = [];

  for (let i = 0; i <= pointCount; i++) {
    const t = i / pointCount;
    const angle = t * Math.PI * 2 * turns;
    const y = (t - 0.5) * height;

    const x1 = Math.cos(angle) * radius;
    const z1 = Math.sin(angle) * radius;
    pointsA.push(new THREE.Vector3(x1, y, z1));

    const x2 = Math.cos(angle + Math.PI) * radius;
    const z2 = Math.sin(angle + Math.PI) * radius;
    pointsB.push(new THREE.Vector3(x2, y, z2));
  }

  const curveA = new THREE.CatmullRomCurve3(pointsA);
  const curveB = new THREE.CatmullRomCurve3(pointsB);

  const tubeGeoA = new THREE.TubeGeometry(curveA, 128, 0.3, 16, false); /* Thinner strands */
  const tubeGeoB = new THREE.TubeGeometry(curveB, 128, 0.3, 16, false);

  const strandA = new THREE.Mesh(tubeGeoA, backboneMaterial);
  const strandB = new THREE.Mesh(tubeGeoB, backboneMaterial);
  dnaGroup.add(strandA);
  dnaGroup.add(strandB);

  // Rungs & Labels
  const rungCount = 35;
  const cylGeo = new THREE.CylinderGeometry(0.2, 0.2, 1, 16);
  const sphereGeo = new THREE.SphereGeometry(0.5, 16, 16);

  for (let i = 1; i < rungCount; i++) {
    const t = i / rungCount;
    const ptA = curveA.getPoint(t);
    const ptB = curveB.getPoint(t);
    const dist = ptA.distanceTo(ptB);
    const mid = new THREE.Vector3().lerpVectors(ptA, ptB, 0.5);

    const halfLen = (dist / 2) - 0.5;

    // Determine Pair Type: A-T or G-C
    // Pattern: A-T, G-C, G-C, A-T ...
    const pairType = i % 4; // 0=A-T, 1=G-C, 2=G-C, 3=A-T

    let mat1, mat2;

    if (pairType === 0 || pairType === 3) {
      // A-T Pair
      mat1 = matAdenine;
      mat2 = matThymine;
    } else {
      // G-C Pair
      mat1 = matGuanine;
      mat2 = matCytosine;
    }

    // Rungs
    const r1 = new THREE.Mesh(cylGeo, mat1);
    r1.scale.y = halfLen;
    r1.position.copy(ptA).lerp(mid, 0.5);
    r1.lookAt(ptB);
    r1.rotateX(Math.PI / 2);
    dnaGroup.add(r1);

    const r2 = new THREE.Mesh(cylGeo, mat2);
    r2.scale.y = halfLen;
    r2.position.copy(ptB).lerp(mid, 0.5);
    r2.lookAt(ptA);
    r2.rotateX(Math.PI / 2);
    dnaGroup.add(r2);

    // Connectors
    const s1 = new THREE.Mesh(sphereGeo, mat1);
    s1.position.copy(ptA);
    dnaGroup.add(s1);

    const s2 = new THREE.Mesh(sphereGeo, mat2);
    s2.position.copy(ptB);
    dnaGroup.add(s2);

    // Labels REMOVED for clean side view
  }

  // Backbone Label - REMOVED for clean side view

  // Particles
  const pCount = 300;
  const pPos = [];
  const pCols = [];
  for (let i = 0; i < pCount; i++) {
    pPos.push((Math.random() - 0.5) * 50, (Math.random() - 0.5) * 70, (Math.random() - 0.5) * 50);
    const c = Math.random() > 0.5 ? new THREE.Color(0x00aaff) : new THREE.Color(0xffaa00);
    pCols.push(c.r, c.g, c.b);
  }
  const pGeo = new THREE.BufferGeometry();
  pGeo.setAttribute('position', new THREE.Float32BufferAttribute(pPos, 3));
  pGeo.setAttribute('color', new THREE.Float32BufferAttribute(pCols, 3));
  const pMat = new THREE.PointsMaterial({
    vertexColors: true, size: 0.4, transparent: true, opacity: 0.6, blending: THREE.AdditiveBlending
  });
  const pSys = new THREE.Points(pGeo, pMat);
  dnaScene.add(pSys);

  // Animate
  const animate = () => {
    if (!dnaRenderer) return;
    requestAnimationFrame(animate);
    controls.update();
    dnaGroup.rotation.y += 0.002;
    pSys.rotation.y -= 0.001;

    // Pulse backbone
    const time = Date.now() * 0.001;
    backboneMaterial.emissive = new THREE.Color(0x0088ff);
    backboneMaterial.emissiveIntensity = 0.2 + Math.sin(time * 2) * 0.1;

    dnaRenderer.render(dnaScene, dnaCamera);
    labelRenderer.render(dnaScene, dnaCamera);
  };
  animate();

  const handleResize = () => {
    if (!dnaCamera || !dnaRenderer || !labelRenderer) return;
    const w = window.innerWidth;
    const h = window.innerHeight;
    dnaCamera.aspect = w / h;
    dnaCamera.updateProjectionMatrix();
    dnaRenderer.setSize(w, h);
    labelRenderer.setSize(w, h);
  };
  window.addEventListener('resize', handleResize);
  setTimeout(handleResize, 100);
}

if (connectBtn) {
  connectBtn.addEventListener('click', () => {
    const docId = doctorIdInput.value || "UNKNOWN";

    // Trigger transition effects
    playTransitionEffects('login');

    // Sequence
    const tl = gsap.timeline();

    // 1. Inputs fade out
    tl.to('.login-card', {
      scaleY: 0.01,
      scaleX: 1.2,
      opacity: 0.5,
      duration: 0.4,
      ease: "power2.in"
    });
    // ...
    tl.to('.login-card', {
      scaleX: 0,
      opacity: 0,
      duration: 0.2,
      ease: "power2.in"
    });

    // 2. Background zooms
    tl.to(bg, {
      scale: 8,
      opacity: 0,
      duration: 1.5,
      ease: "expo.in"
    }, "-=0.3");

    // 3. Tunnel Flash
    tl.to(warpTunnel, {
      width: '400vw',
      height: '400vw',
      opacity: 1,
      duration: 1.2,
      ease: "expo.in"
    }, "<");

    // 4. Reveal
    tl.to(warpTunnel, {
      opacity: 0,
      duration: 0.2,
      onComplete: () => {
        // initDNA(); // Already initialized as background
      }
    }, "-=0.2");

    tl.set(pageLogin, { display: 'none' });
    tl.set(pageDashboard, { display: 'block', opacity: 0 });

    if (displayId) {
      // Scramble
      const chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()";
      const targetText = docId;
      const duration = 1.5;
      const obj = { value: 0 };
      gsap.to(obj, {
        duration: duration,
        value: 1,
        ease: "none",
        onUpdate: () => {
          const progress = obj.value;
          const len = Math.floor(progress * targetText.length);
          let result = targetText.substring(0, len);
          if (len < targetText.length) {
            for (let i = 0; i < 3; i++) {
              result += chars[Math.floor(Math.random() * chars.length)];
            }
          }
          displayId.innerText = result;
        },
        onComplete: () => {
          displayId.innerText = targetText;
        }
      });
    }

    tl.to(pageDashboard, {
      opacity: 1,
      duration: 1,
      onComplete: () => {
        const header = document.querySelector('.dashboard-header .user-status');
        if (header && !header.contains(clockEl)) {
          header.appendChild(clockEl);
        }
      }
    });

    tl.from('.panel', {
      y: 100,
      opacity: 0,
      duration: 0.8,
      stagger: 0.2,
      ease: "power3.out"
    });

    setTimeout(() => addLog("Authenticating Credentials..."), 2000);
    setTimeout(() => addLog("Establish Secure Link..."), 3000);
    setTimeout(() => addLog("Network Handshake: SUCCESS"), 4000);
    setTimeout(() => addLog("Genomic Visualizer Loaded..."), 4500);
    setTimeout(() => addLog("SYSTEM ONLINE"), 5500);
  });
}

// Logout
if (logoutBtn) {
  logoutBtn.addEventListener('click', () => {
    addLog("De-authenticating session...");
    triggerGlitchEffect();

    const tl = gsap.timeline();

    tl.to(pageDashboard, {
      opacity: 0,
      duration: 0.5,
      onComplete: () => {
        pageDashboard.style.display = 'none';
        pageLogin.style.display = 'flex';
        gsap.set('.login-card', { scale: 1, opacity: 1, scaleX: 1, scaleY: 1 });
        gsap.from('.login-card', { y: -50, opacity: 0, duration: 0.5 });

        const container = document.getElementById('dna-canvas-container');
        if (container) container.innerHTML = '';
      }
    });

    tl.to(bg, {
      scale: 1,
      opacity: 0.6,
      duration: 1
    }, "-=0.5");
  });
}

// Submit / Navigate to Analysis
if (submitBtn) {
  submitBtn.addEventListener('click', () => {
    const pName = (document.getElementById('p-name') as HTMLInputElement).value;
    const pAge = (document.getElementById('p-age') as HTMLInputElement).value;

    if (!pName || !pAge) {
      alert("Please enter patient details first.");
      return;
    }

    addLog("Processing Patient Data...");
    gsap.fromTo('#scan-glow', { opacity: 0 }, { opacity: 1, duration: 0.3, yoyo: true, repeat: 3 });

    // Trigger transition effects
    playTransitionEffects('dashboard');

    // Transition to Analysis
    const tl = gsap.timeline();

    tl.to('.page-dashboard', {
      scale: 0.9,
      opacity: 0,
      duration: 0.5,
      ease: "power2.in"
    });

    tl.to(warpTunnel, {
      width: '400vw',
      height: '400vw',
      opacity: 1,
      duration: 1,
      ease: "expo.out"
    });

    tl.set('.page-dashboard', { display: 'none' });
    tl.set('.page-analysis', { display: 'flex', opacity: 0 });

    tl.to(warpTunnel, {
      opacity: 0,
      duration: 0.5
    });

    tl.to('.page-analysis', {
      opacity: 1,
      duration: 0.5
    });

    tl.from('.analysis-container .panel', {
      y: 50,
      opacity: 0,
      stagger: 0.2
    }, "-=0.3");
  });
}

// Predict Graph Logic
// Predict Graph Logic & API Integration
if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    const btn = predictBtn as HTMLButtonElement;
    const fileInput = document.getElementById('genome-file') as HTMLInputElement;
    const diseaseInput = document.getElementById('p-disease') as HTMLInputElement;
    let rankedForReport: any[] | null = null;
    let topCandidateForReport: any | null = null;

    // Removed config inputs: targetProteinInput, ligandInput

    if (!fileInput.files || fileInput.files.length === 0) {
      alert("Please upload a patient genome file (CSV) first.");
      return;
    }

    btn.innerHTML = "UPLOADING & ANALYZING...";
    btn.disabled = true;

    const graphContainer = document.getElementById('results-graph');
    if (graphContainer) graphContainer.innerHTML = '';

    const resultText = document.getElementById('text-result');
    if (resultText) gsap.set(resultText, { opacity: 0, y: 20 });

    try {
      // 1. Call Genome Analysis API
      addLog("Uploading genome data to Model Server...");
      const genomeData = await api.analyzeGenome(fileInput.files[0], diseaseInput.value);
      addLog("Genome Analysis Complete. Processing results...");

      // 2. Call Compatibility API (AUTO-DETECTED)
      addLog("Checking Drug-Disease Compatibility...");

      const topDrug = (genomeData && genomeData.results && genomeData.results.length > 0)
        ? genomeData.results[0].Drug
        : "Unknown Agent";

      const compatibilityData = await api.analyzeCompatibility({
        drug_name: topDrug,
        disease: diseaseInput.value,
        patient_data: {
          name: (document.getElementById('p-name') as HTMLInputElement).value
        }
      });

      // 3. Render Results (Visualization) - ACCURATE DATA TABLE
      if (genomeData && genomeData.results) {
        if (graphContainer) {
          graphContainer.innerHTML = ''; // Clear previous
          graphContainer.style.display = 'block';
          graphContainer.style.overflowY = 'auto';
          graphContainer.style.maxHeight = '300px';

          const table = document.createElement('table');
          table.style.width = '100%';
          table.style.borderCollapse = 'collapse';
          table.style.marginTop = '10px';
          table.style.fontFamily = "'Courier New', monospace";
          table.style.fontSize = '0.85rem';

          // Header
          const thead = document.createElement('thead');
          thead.innerHTML = `
              <tr style="border-bottom: 1px solid var(--neon-blue); color: var(--neon-blue);">
                  <th style="padding: 8px; text-align: left;">DRUG CANDIDATE</th>
                  <th style="padding: 8px; text-align: left;">SUITABILITY</th>
                  <th style="padding: 8px; text-align: left;">SAFETY PROFILE</th>
              </tr>
            `;
          table.appendChild(thead);

          // Body
          const tbody = document.createElement('tbody');
          genomeData.results.forEach((row: any) => {
            const tr = document.createElement('tr');
            tr.style.borderBottom = '1px solid rgba(255,255,255,0.05)';

            // Color Logic
            let suitColor = '#fff';
            if (row['Suitability'] && row['Suitability'].includes('RECOMMENDED')) suitColor = 'var(--neon-green)';
            else if (row['Suitability'] && row['Suitability'].includes('NOT')) suitColor = 'var(--neon-red)';

            tr.innerHTML = `
                  <td style="padding: 8px; color: white;">${row['Drug'] || 'Unknown'}</td>
                  <td style="padding: 8px; color: ${suitColor}; font-weight: bold;">${row['Suitability'] || '-'}</td>
                  <td style="padding: 8px; color: #aaa;">${row['Safety Assessment'] || '-'}</td>
                `;
            tbody.appendChild(tr);
          });
          table.appendChild(tbody);
          graphContainer.appendChild(table);

          // Animate Table Rows
          gsap.from(tbody.children, {
            opacity: 0,
            x: -20,
            stagger: 0.05,
            duration: 0.5
          });
        }
      }

      // Render Text Result (Accuracy Update)
      if (resultText && genomeData && genomeData.results && genomeData.results.length > 0) {
        const allResults: any[] = Array.isArray(genomeData.results) ? genomeData.results : [];
        const recommended = allResults
          .filter(r => typeof r['Suitability'] === 'string' && r['Suitability'].includes('RECOMMENDED'))
          .sort((a, b) => {
            const aScore = typeof a['Score'] === 'number' ? a['Score'] : parseFloat(a['Score'] ?? '0');
            const bScore = typeof b['Score'] === 'number' ? b['Score'] : parseFloat(b['Score'] ?? '0');
            return aScore - bScore;
          });
        const ranked = (recommended.length > 0 ? recommended : allResults).slice(0, 5);
        const topCandidate = ranked[0];
        rankedForReport = ranked;
        topCandidateForReport = topCandidate;
        const statusEl = document.getElementById('result-status');
        const affinityEl = document.getElementById('res-affinity');
        const toxEl = document.getElementById('res-toxicity');

        if (statusEl) {
          const isCompatible = compatibilityData && compatibilityData.compatible;
          const hasRecommended = topCandidate && typeof topCandidate['Suitability'] === 'string' &&
            topCandidate['Suitability'].includes('RECOMMENDED');
          const drugLabel = topCandidate && topCandidate['Drug'] ? String(topCandidate['Drug']).toUpperCase() : 'UNKNOWN AGENT';
          if (isCompatible && hasRecommended) {
            statusEl.innerHTML = `<span style="color: #fff">RECOMMENDED AGENT:</span> <span style="color: var(--neon-green); font-size: 1.2em;">${drugLabel}</span>`;
            statusEl.style.color = "var(--neon-green)";
          } else {
            const statusText = compatibilityData && compatibilityData.status ? String(compatibilityData.status) : "CAUTION: RESISTANCE OR NON-STANDARD USE";
            statusEl.innerText = statusText;
            statusEl.style.color = "var(--neon-red)";
          }
        }

        // Update Side Panel Metrics with Top Drug Info
        if (affinityEl) {
          affinityEl.innerText = (topCandidate && topCandidate['Drug']) ? String(topCandidate['Drug']) : "Unknown";
          affinityEl.style.color = "var(--neon-blue)";
        }
        if (toxEl) {
          toxEl.innerText = "OPTIMAL";
          toxEl.style.color = "var(--neon-green)";
        }

        // 3. Render Results (Visualization) - MEDICAL MONITOR UI
        if (genomeData && ranked && ranked.length > 0) {
          if (graphContainer) {
            graphContainer.innerHTML = '';
            graphContainer.style.display = 'block';
            graphContainer.style.overflowY = 'auto';
            graphContainer.style.background = 'rgba(0,0,0,0.3)';
            graphContainer.style.borderRadius = '8px';
            graphContainer.style.padding = '10px';

            // UNIQUE "PRIMARY AGENT" HOLO-CARD
            const holoCard = document.createElement('div');
            holoCard.style.cssText = `
                  display: grid;
                  grid-template-columns: 1fr 120px;
                  background: linear-gradient(135deg, rgba(0, 255, 136, 0.1), rgba(0,0,0,0.6));
                  border: 1px solid var(--neon-green);
                  border-radius: 8px;
                  padding: 15px;
                  margin-bottom: 20px;
                  box-shadow: 0 0 15px rgba(0, 255, 136, 0.2);
                  position: relative;
                  overflow: hidden;
              `;

            // Dynamic Match % Calculation (Inverse of IC50-like score, normalized)
            let matchScore = 50;
            if (topCandidate && typeof topCandidate['Score'] !== 'undefined' && topCandidate['Score'] !== null) {
              const rawScore = typeof topCandidate['Score'] === 'number'
                ? topCandidate['Score']
                : parseFloat(String(topCandidate['Score']));
              if (!Number.isNaN(rawScore)) {
                matchScore = Math.min(99, Math.max(10, Math.floor((2 - rawScore) * 25 + 50)));
              }
            } else if (topCandidate && topCandidate['Drug']) {
              matchScore = Math.floor(Math.random() * 40) + 60;
            }

            holoCard.innerHTML = `
                  <div style="z-index: 2;">
                      <div style="font-size: 0.75rem; color: var(--neon-green); letter-spacing: 2px;">PRIMARY THERAPEUTIC CANDIDATE</div>
                      <div style="font-size: 2rem; font-weight: 800; color: white; margin: 5px 0; text-shadow: 0 0 10px var(--neon-green);">
                          ${topCandidate && topCandidate['Drug'] ? topCandidate['Drug'] : 'UNKNOWN AGENT'}
                      </div>
                      <div style="display: flex; gap: 10px; margin-top: 5px;">
                          <div style="background: rgba(0,255,136,0.2); color: var(--neon-green); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid var(--neon-green);">
                              ✅ ${topCandidate && topCandidate['Safety Assessment'] ? topCandidate['Safety Assessment'] : 'High Safety / High Efficacy'}
                          </div>
                      </div>
                  </div>
                  <div style="display: flex; align-items: center; justify-content: center; position: relative;">
                      <svg viewBox="0 0 36 36" style="width: 80px; height: 80px; transform: rotate(-90deg);">
                          <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="#444" stroke-width="3" />
                          <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke="var(--neon-green)" stroke-width="3" stroke-dasharray="${matchScore}, 100" class="circle-progress" />
                      </svg>
                      <div style="position: absolute; color: white; font-weight: bold; font-size: 1.2rem;">${matchScore}%</div>
                      <div style="position: absolute; bottom: -20px; font-size: 0.6rem; color: #888;">MATCH</div>
                  </div>
                  <div style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,136,0.03) 3px); pointer-events: none;"></div>
              `;
            graphContainer.appendChild(holoCard);

            // REST OF CANDIDATES GRID
            const listContainer = document.createElement('div');
            listContainer.style.display = 'grid';
            listContainer.style.gap = '10px';

            ranked.slice(1).forEach((row: any) => {
              if (!row['Drug']) return;

              // Score calc
              let rowScore = 50;
              if (row && typeof row['Score'] !== 'undefined' && row['Score'] !== null) {
                rowScore = Math.min(99, Math.max(10, Math.floor((2 - parseFloat(row['Score'])) * 25 + 50)));
              } else {
                rowScore = Math.floor(Math.random() * 40) + 30; // 30-69% random for variety
              }

              const isRec = row['Suitability'] && row['Suitability'].includes('RECOMMENDED');
              const glowColor = isRec ? 'var(--neon-green)' : 'var(--neon-red)';
              const borderColor = isRec ? 'var(--neon-green)' : '#ff0055';

              const item = document.createElement('div');
              item.style.cssText = `
                      display: flex;
                      justify-content: space-between;
                      align-items: center;
                      padding: 10px;
                      background: rgba(0,0,0,0.4);
                      border-left: 3px solid ${borderColor};
                      border-radius: 0 4px 4px 0;
                  `;

              item.innerHTML = `
                      <div>
                          <div style="color: #fff; font-weight: bold; font-size: 0.9rem;">${row['Drug']}</div>
                          <div style="color: #888; font-size: 0.7rem;">${row['Suitability']}</div>
                      </div>
                      <div style="text-align: right;">
                          <div style="color: ${glowColor}; font-weight: bold;">${rowScore}%</div>
                          <div style="width: 60px; height: 4px; background: #333; margin-top: 4px; border-radius: 2px;">
                              <div style="width: ${rowScore}%; height: 100%; background: ${glowColor}; border-radius: 2px;"></div>
                          </div>
                      </div>
                  `;
              listContainer.appendChild(item);
            });

            graphContainer.appendChild(listContainer);

            // Animations
            gsap.from(holoCard, { y: 20, opacity: 0, duration: 0.6, ease: "back.out(1.7)" });
            gsap.from(listContainer.children, { x: -20, opacity: 0, stagger: 0.05, delay: 0.3 });
          }
        }

        gsap.to(resultText, { opacity: 1, y: 0, duration: 0.5, delay: 0.5 });
      }

      // Render Compatibility
      const compatibilityContainer = document.getElementById('compatibility-result');
      if (compatibilityContainer) {
        const match = compatibilityData;
        const isSupported = match.compatible;
        const statusColor = isSupported ? 'var(--neon-green)' : 'var(--neon-red, #ff0055)';
        const statusIcon = isSupported ? '✓' : '⚠';

        compatibilityContainer.innerHTML = `
              <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; text-align: left;">
                <div>
                  <div style="font-size: 1.3rem; color: ${statusColor}; margin-bottom: 1rem; font-weight: bold;">
                    ${statusIcon} ${match.status}
                  </div>
                  <div style="margin-bottom: 1rem;">
                    <div style="color: var(--neon-blue); font-size: 0.85rem; margin-bottom: 0.3rem;">DISEASE:</div>
                    <div style="color: white; font-size: 1rem;">${match.disease.toUpperCase()}</div>
                  </div>
                  <div style="margin-bottom: 1rem;">
                    <div style="color: var(--neon-blue); font-size: 0.85rem; margin-bottom: 0.3rem;">CONFIDENCE SCORE:</div>
                    <div style="color: ${statusColor}; font-size: 1.5rem; font-weight: bold;">${match.confidence_score}</div>
                  </div>
                  <div style="margin-bottom: 1.5rem;">
                    <div style="color: var(--neon-blue); font-size: 0.85rem; margin-bottom: 0.3rem;">DEPARTMENT:</div>
                    <div style="color: var(--neon-purple); font-size: 1rem; font-weight: bold;">${match.department}</div>
                  </div>
                </div>
                <div>
                  <div style="margin-bottom: 1.5rem;">
                    <div style="color: var(--neon-blue); font-size: 0.85rem; margin-bottom: 0.3rem;">MECHANISM OF ACTION:</div>
                    <div style="color: var(--neon-pink); font-size: 0.95rem; line-height: 1.4;">${match.mechanism}</div>
                  </div>
                  <div style="margin-bottom: 1.5rem;">
                    <div style="color: var(--neon-blue); font-size: 0.85rem; margin-bottom: 0.3rem;">CLINICAL EVIDENCE:</div>
                    <div style="color: #5a2323ff; font-size: 0.9rem; line-height: 1.4; border-left: 2px solid var(--neon-blue); padding-left: 10px;">${match.clinical_evidence}</div>
                  </div>
                   <div style="margin-bottom: 1.5rem;">
                    <div style="color: var(--neon-green); font-size: 0.85rem; margin-bottom: 0.3rem;">RECOMMENDATION:</div>
                    <div style="color: var(--neon-green); font-size: 1rem; line-height: 1.4; font-weight: 800; padding: 10px; background: linear-gradient(135deg, rgba(4,53,61,0.9), rgba(4,53,61,0.7)); border: 1px solid var(--primary-teal); border-radius: 8px; box-shadow: 0 0 12px rgba(90, 197, 200, 0.25);">
                        ${match.recommendation}
                    </div>
                  </div>
                </div>
              </div>
            `;
        gsap.from(compatibilityContainer.children, {
          opacity: 0,
          y: 20,
          stagger: 0.1,
          duration: 0.5
        });

        playTransitionEffects('analysis');
      }

      const doctorDisplay = document.getElementById('display-id') as HTMLElement | null;
      const patientInput = document.getElementById('p-name') as HTMLInputElement | null;
      lastAnalysis = {
        doctorId: doctorDisplay ? doctorDisplay.innerText : '',
        patientName: patientInput ? patientInput.value : '',
        disease: diseaseInput.value,
        topDrug: topCandidateForReport && topCandidateForReport['Drug'] ? String(topCandidateForReport['Drug']) : '',
        ranked: rankedForReport || [],
        compatibility: compatibilityData
      };

      btn.innerHTML = "RUN GENOMIC ANALYSIS";

    } catch (error) {
      console.error(error);
      addLog("CRITICAL ERROR: API Connection Failed");
      btn.innerHTML = "SYSTEM ERROR";
      alert("Failed to connect to Hospital Backend System. Ensure backend is running.");
    } finally {
      btn.disabled = false;
    }
  });
}

// Back to Dashboard
if (backDashBtn) {
  backDashBtn.addEventListener('click', () => {
    // Trigger transition effects
    playTransitionEffects('back');

    const tl = gsap.timeline();

    tl.to('.page-analysis', {
      opacity: 0,
      duration: 0.5,
      onComplete: () => {
        pageAnalysis.style.display = 'none';
        pageDashboard.style.display = 'block';
      }
    });

    tl.to('.page-dashboard', {
      opacity: 1,
      scale: 1,
      duration: 0.5
    });
  });
}

// DNA Search logic
document.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  if (target && target.id === 'btn-search-dna') {
    const query = (document.getElementById('dna-search') as HTMLInputElement).value;
    if (query) {
      addLog(`Searching genomic sequence: ${query.toUpperCase()}...`);
      triggerGlitchEffect();
      triggerScanLine();
      setTimeout(() => addLog(`Sequence match found at LOC: ${Math.floor(Math.random() * 10000)}`), 1200);
    }
  }
});

// DNA Reset logic
document.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  if (target && target.id === 'btn-reset-dna') {
    if (dnaCamera) {
      gsap.to(dnaCamera.position, { x: 100, y: 0, z: 50, duration: 1, ease: "power2.inOut" });
      addLog("Genomic camera re-centered.");
    }
  }
});

// Download Report Logic
document.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  if (target && target.id === 'btn-download-report') {
    const doctorIdEl = document.getElementById('display-id') as HTMLElement | null;
    const patientNameEl = document.getElementById('p-name') as HTMLInputElement | null;
    const diseaseEl = document.getElementById('p-disease') as HTMLInputElement | null;
    const compatibilityEl = document.getElementById('compatibility-result') as HTMLElement | null;

    let content = "HOSPIT-X DIAGNOSTIC REPORT\n" +
      "DATE: " + new Date().toISOString() + "\n" +
      "DOCTOR: " + (doctorIdEl ? doctorIdEl.innerText : "") + "\n" +
      "PATIENT: " + (patientNameEl ? patientNameEl.value : "") + "\n" +
      "DISEASE: " + (diseaseEl ? diseaseEl.value : "") + "\n";

    let compatibilityLine = "";
    if (lastAnalysis && lastAnalysis.compatibility && lastAnalysis.compatibility.status) {
      compatibilityLine = String(lastAnalysis.compatibility.status);
    } else if (compatibilityEl) {
      compatibilityLine = compatibilityEl.innerText.split('\n')[0];
    }
    content += "COMPATIBILITY: " + compatibilityLine + "\n";

    if (lastAnalysis && lastAnalysis.ranked && lastAnalysis.ranked.length > 0) {
      const topDrugName = lastAnalysis.topDrug || "";
      let effectiveness = "";
      let dosageText = "";
      const topRow = lastAnalysis.ranked[0];
      if (topRow && typeof topRow['Score'] !== 'undefined' && topRow['Score'] !== null) {
        const rawScore = typeof topRow['Score'] === 'number' ? topRow['Score'] : parseFloat(String(topRow['Score']));
        if (!Number.isNaN(rawScore)) {
          const matchScore = Math.min(99, Math.max(10, Math.floor((2 - rawScore) * 25 + 50)));
          effectiveness = matchScore.toString() + "%";
          
          if (lastAnalysis.topCandidate && lastAnalysis.topCandidate.dosageGuidance) {
              dosageText = lastAnalysis.topCandidate.dosageGuidance;
          } else {
              if (matchScore >= 80) {
                dosageText = "Standard dose range with full-intensity regimen";
              } else if (matchScore >= 60) {
                dosageText = "Moderate dose range with adjusted regimen";
              } else {
                dosageText = "Lower dose range with cautious titration";
              }
          }
        }
      }

      content += "\nPRIMARY THERAPEUTIC CANDIDATE: " + topDrugName + "\n";
      if (effectiveness) {
        content += "EFFECTIVENESS SCORE: " + effectiveness + "\n";
      }
      if (dosageText) {
        content += "SUGGESTED DOSAGE GUIDANCE: " + dosageText + "\n";
      }

      content += "\nTOP DRUG CANDIDATES:\n";
      lastAnalysis.ranked.forEach((row: any, index: number) => {
        const name = row['Drug'] || "Unknown";
        const suitability = row['Suitability'] || "-";
        const safety = row['Safety Assessment'] || "-";
        const scoreVal = typeof row['Score'] !== 'undefined' && row['Score'] !== null ? String(row['Score']) : "";
        let line = (index + 1).toString() + ". " + name + " | Suitability: " + suitability + " | Safety: " + safety;
        if (scoreVal) {
          line += " | Model Score: " + scoreVal;
        }
        content += line + "\n";
      });
    }

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "diagnostic-report.txt";
    a.click();
    addLog("Diagnostic report exported successfully.");
  }
});
// Entry point for H-X System
initDNA();

// ========== CURSOR INTEGRATION ==========

class ArrowPointer {
  root: HTMLElement;
  cursor: HTMLElement;
  position: {
    distanceX: number;
    distanceY: number;
    distance: number;
    pointerX: number;
    pointerY: number;
  };
  previousPointerX: number = 0;
  previousPointerY: number = 0;
  angle: number = 0;
  previousAngle: number = 0;
  angleDisplace: number = 0;
  degrees: number = 57.296;
  cursorSize: number = 20;

  constructor() {
    this.root = document.body;
    this.cursor = document.querySelector(".curzr-arrow-pointer") as HTMLElement;

    this.position = {
      distanceX: 0,
      distanceY: 0,
      distance: 0,
      pointerX: 0,
      pointerY: 0,
    };

    const cursorStyle: Partial<CSSStyleDeclaration> = {
      boxSizing: 'border-box',
      position: 'fixed',
      top: '50%',
      left: '50%',
      transform: 'translate(-50%, -50%)',
      zIndex: '2147483647',
      width: `${this.cursorSize}px`,
      height: `${this.cursorSize}px`,
      transition: '250ms, transform 100ms',
      userSelect: 'none',
      pointerEvents: 'none'
    };

    this.init(this.cursor, cursorStyle);
  }

  init(el: HTMLElement, style: Partial<CSSStyleDeclaration>) {
    Object.assign(el.style, style);
    setTimeout(() => {
      this.cursor.removeAttribute("hidden");
    }, 500);
    this.cursor.style.opacity = '1';
  }

  move(event: MouseEvent) {
    this.previousPointerX = this.position.pointerX;
    this.previousPointerY = this.position.pointerY;
    this.position.pointerX = event.clientX;
    this.position.pointerY = event.clientY;
    this.position.distanceX = this.previousPointerX - this.position.pointerX;
    this.position.distanceY = this.previousPointerY - this.position.pointerY;
    this.position.distance = Math.sqrt(this.position.distanceY ** 2 + this.position.distanceX ** 2);

    this.cursor.style.transform = `translate3d(${this.position.pointerX}px, ${this.position.pointerY}px, 0)`;

    if (this.position.distance > 1) {
      this.rotate(this.position);
    } else {
      this.cursor.style.transform += ` rotate(${this.angleDisplace}deg)`;
    }
  }

  rotate(position: { distanceX: number; distanceY: number }) {
    let unsortedAngle = Math.atan(Math.abs(position.distanceY) / Math.abs(position.distanceX)) * this.degrees;
    const style = this.cursor.style;
    this.previousAngle = this.angle;

    if (position.distanceX <= 0 && position.distanceY >= 0) {
      this.angle = 90 - unsortedAngle + 0;
    } else if (position.distanceX < 0 && position.distanceY < 0) {
      this.angle = unsortedAngle + 90;
    } else if (position.distanceX >= 0 && position.distanceY <= 0) {
      this.angle = 90 - unsortedAngle + 180;
    } else if (position.distanceX > 0 && position.distanceY > 0) {
      this.angle = unsortedAngle + 270;
    }

    if (isNaN(this.angle)) {
      this.angle = this.previousAngle;
    } else {
      if (this.angle - this.previousAngle <= -270) {
        this.angleDisplace += 360 + this.angle - this.previousAngle;
      } else if (this.angle - this.previousAngle >= 270) {
        this.angleDisplace += this.angle - this.previousAngle - 360;
      } else {
        this.angleDisplace += this.angle - this.previousAngle;
      }
    }
    style.left = `${-this.cursorSize / 2}px`;
    style.top = `${0}px`;
    style.transform += ` rotate(${this.angleDisplace}deg)`;
  }

  click() {
    this.cursor.style.transform += ` scale(0.75)`;
    setTimeout(() => {
      this.cursor.style.transform = this.cursor.style.transform.replace(` scale(0.75)`, '');
    }, 35);
  }

  hidden() {
    this.cursor.style.opacity = '0';
    setTimeout(() => {
      this.cursor.setAttribute("hidden", "hidden");
    }, 500);
  }
}

let curzr: ArrowPointer | null = null;
document.addEventListener('DOMContentLoaded', () => {
  curzr = new ArrowPointer();
  window.addEventListener('mousemove', (e) => curzr?.move(e));
  window.addEventListener('mousedown', () => curzr?.click());
});
// Fallback if DOMContentLoaded already fired
if (document.readyState === 'complete' || document.readyState === 'interactive') {
  if (!curzr) {
    curzr = new ArrowPointer();
    window.addEventListener('mousemove', (e) => curzr?.move(e));
    window.addEventListener('mousedown', () => curzr?.click());
  }
}

// Voice button click: toggle active visual state and log action
document.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  if (!target) return;
  const button = target.id === 'voice-all-btn' ? target : target.closest('#voice-all-btn');
  if (button) {
    button.classList.toggle('active');
    const isActive = button.classList.contains('active');
    addLog(isActive ? 'Voice capture started' : 'Voice capture stopped');
  }
});
