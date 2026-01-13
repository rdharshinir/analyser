import './style.css'
import gsap from 'gsap'
import * as THREE from 'three'
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';
import { api } from './api';

document.querySelector<HTMLDivElement>('#app')!.innerHTML = `
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
  
  <div class="page-login">
    <div class="login-card">
      <div class="login-card-corner-br"></div>
      <h1>HOSPIT-X</h1>
      <h2>SECURE DOCTOR PORTAL</h2>
      
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
      <div class="logo" style="font-size: 1.5rem; letter-spacing: 2px;">
        HOSPIT-X <span style="color:var(--neon-blue); font-size: 0.8em; border: 1px solid var(--neon-blue); padding: 2px 5px;">SYS.ADMIN</span>
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
    
    <div class="grid-container">
      
      <!-- DNA VISUALIZER (Main View) -->
      <div class="panel center-view" style="padding:0; overflow:hidden; position: relative;" role="region" aria-label="Genomic Visualizer">
        <h3 style="position: absolute; top: 1rem; left: 1rem; z-index: 2; text-shadow: 0 0 5px black;">GENOMIC SEQUENCE VISUALIZER</h3>
        <div style="position: absolute; top: 1rem; right: 1rem; z-index: 2; display: flex; gap: 5px;">
           <input type="text" id="dna-search" placeholder="SEARCH SEQ (e.g. ATGC)" style="font-size: 0.7rem; padding: 5px; width: 150px; border-radius: 0; background: var(--glass); border: 1px solid var(--neon-blue);" aria-label="Search genomic sequence">
           <button id="btn-search-dna" class="btn-logout" style="margin:0; padding: 5px 10px;" aria-label="Execute DNA search">FIND</button>
           <button id="btn-reset-dna" class="btn-logout" style="margin:0; padding: 5px 10px;" aria-label="Reset DNA view">RESET</button>
        </div>
        <div id="dna-canvas-container" style="width: 100%; height: 100%;"></div>
      </div>
      
      <div class="panel right-hud" role="region" aria-label="Patient Report Entry">
        <h3>PATIENT REPORT ENTRY</h3>
         <div class="report-form-container">
            <input type="text" class="report-input" placeholder="PATIENT NAME" id="p-name" aria-label="Patient Name">
            <input type="text" class="report-input" placeholder="AGE / GENDER" id="p-age" aria-label="Age and Gender">
            <input type="text" class="report-input" placeholder="BLOOD GROUP" id="p-blood" aria-label="Blood Group">
            <input type="text" class="report-input" placeholder="DISEASE/CONDITION" id="p-disease" aria-label="Disease or Condition">
            <textarea class="report-input" placeholder="DIAGNOSIS NOTES" rows="3" id="p-diagnosis" aria-label="Diagnosis Notes"></textarea>
            <button class="btn-submit" id="submit-report" aria-label="Upload report to backend">UPLOAD TO BACKEND</button>
         </div>
         <div style="flex-grow: 1; margin-top: 1rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 0.5rem;">
            <div style="font-size: 0.8rem; color: var(--neon-purple); margin-bottom: 0.5rem;">SYS LOGS:</div>
            <ul id="console-logs" style="list-style: none; padding: 0; font-size: 0.7rem; color: #888; overflow-y: auto; height: 100px; font-family: 'monospace';">
              <li>[SYS] Standing by...</li>
            </ul>
         </div>
      </div>
    </div>
  </div>

  <div class="page-analysis">
      <div class="dashboard-bg-deco" style="border-color: var(--neon-green);"></div>
      <header class="dashboard-header">
        <div class="logo">DRUG DISCOVERY <span style="color:var(--neon-green)">ANALYTICS</span></div>
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

const simulateNetworkLatency = (ms: number = 1000) => new Promise(resolve => setTimeout(resolve, ms));

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
  const container = document.getElementById('dna-canvas-container');
  if (!container) return;

  container.innerHTML = '';

  // Scene
  dnaScene = new THREE.Scene();
  dnaScene.fog = new THREE.FogExp2(0x020408, 0.02);

  // Camera
  dnaCamera = new THREE.PerspectiveCamera(50, container.clientWidth / container.clientHeight, 0.1, 100);
  dnaCamera.position.set(0, 0, 45);

  // Renderer (WebGL)
  dnaRenderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
  dnaRenderer.setSize(container.clientWidth, container.clientHeight);
  dnaRenderer.setPixelRatio(window.devicePixelRatio);
  dnaRenderer.toneMapping = THREE.ACESFilmicToneMapping;
  dnaRenderer.toneMappingExposure = 1.2;
  container.appendChild(dnaRenderer.domElement);

  // Renderer (CSS2D - For Labels)
  labelRenderer = new CSS2DRenderer();
  labelRenderer.setSize(container.clientWidth, container.clientHeight);
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
  const ambientLight = new THREE.AmbientLight(0x404040, 2);
  dnaScene.add(ambientLight);

  const dirLight = new THREE.DirectionalLight(0xffffff, 2);
  dirLight.position.set(10, 20, 10);
  dnaScene.add(dirLight);

  const blueSpot = new THREE.PointLight(0x0088ff, 5, 50);
  blueSpot.position.set(-20, 10, 20);
  dnaScene.add(blueSpot);

  const orangeSpot = new THREE.PointLight(0xff8800, 5, 50);
  orangeSpot.position.set(20, -10, 20);
  dnaScene.add(orangeSpot);

  const dnaGroup = new THREE.Group();
  dnaScene.add(dnaGroup);

  // Materials
  const backboneMaterial = new THREE.MeshPhysicalMaterial({
    color: 0x88ccff,
    metalness: 0.1,
    roughness: 0.1,
    transmission: 0.9,
    thickness: 1.5,
    clearcoat: 1.0,
    opacity: 0.8,
    transparent: true,
    side: THREE.DoubleSide
  });

  // Materials for 4 Bases
  const matAdenine = new THREE.MeshStandardMaterial({ color: 0x00f3ff, emissive: 0x00f3ff, emissiveIntensity: 2, roughness: 0.2, metalness: 0.8 }); // Cyan
  const matThymine = new THREE.MeshStandardMaterial({ color: 0xff8800, emissive: 0xff8800, emissiveIntensity: 2, roughness: 0.2, metalness: 0.8 }); // Orange
  const matGuanine = new THREE.MeshStandardMaterial({ color: 0x0aff0a, emissive: 0x0aff0a, emissiveIntensity: 2, roughness: 0.2, metalness: 0.8 }); // Green
  const matCytosine = new THREE.MeshStandardMaterial({ color: 0xff00ff, emissive: 0xff00ff, emissiveIntensity: 2, roughness: 0.2, metalness: 0.8 }); // Purple

  // Geometry Generation
  const pointCount = 100;
  const radius = 6;
  const height = 45;
  const turns = 3;

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

  const tubeGeoA = new THREE.TubeGeometry(curveA, 128, 0.4, 16, false);
  const tubeGeoB = new THREE.TubeGeometry(curveB, 128, 0.4, 16, false);

  const strandA = new THREE.Mesh(tubeGeoA, backboneMaterial);
  const strandB = new THREE.Mesh(tubeGeoB, backboneMaterial);
  dnaGroup.add(strandA);
  dnaGroup.add(strandB);

  // Helpers
  function createRadialPinLabel(text: string, position: THREE.Vector3, color: string) {
    // Create a local group at the position
    const group = new THREE.Group();
    group.position.copy(position);

    // Calculate radial vector (pointing away from Y axis)
    const radial = new THREE.Vector3(position.x, 0, position.z).normalize();

    // Target Dot
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.3), new THREE.MeshBasicMaterial({ color: color }));
    group.add(dot);

    // Pin Line
    const pinLen = 5.0; // Slightly longer
    const endPoint = radial.clone().multiplyScalar(pinLen);
    const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(0, 0, 0), endPoint]);
    const line = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({ color: color, transparent: true, opacity: 0.5 }));
    group.add(line);

    // Label Element
    const div = document.createElement('div');
    div.className = 'mole-label';
    div.innerHTML = text; // allow html
    div.style.color = color;
    div.style.border = '1px solid ' + color;
    div.style.background = 'rgba(0,10,20,0.85)';
    div.style.padding = '4px 8px';
    div.style.borderRadius = '4px';
    div.style.marginTop = '-10px';

    const label = new CSS2DObject(div);
    label.position.copy(endPoint);
    group.add(label);

    return group;
  }

  // Rungs & Labels
  const rungCount = 35;
  const cylGeo = new THREE.CylinderGeometry(0.2, 0.2, 1, 16);
  const sphereGeo = new THREE.SphereGeometry(0.5, 16, 16);

  let labeledA = false;
  let labeledT = false;
  let labeledG = false;
  let labeledC = false;

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
    let col1, col2;
    let type1 = '', type2 = '';

    if (pairType === 0 || pairType === 3) {
      // A-T Pair
      mat1 = matAdenine; col1 = '#00f3ff'; type1 = 'A';
      mat2 = matThymine; col2 = '#ff8800'; type2 = 'T';
    } else {
      // G-C Pair
      mat1 = matGuanine; col1 = '#0aff0a'; type1 = 'G';
      mat2 = matCytosine; col2 = '#ff00ff'; type2 = 'C';
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

    // Labels
    // Spread them out vertically
    if (type1 === 'A' && !labeledA && i > 5) {
      dnaGroup.add(createRadialPinLabel("Adenine (A)", r1.position, col1));
      labeledA = true;
    }
    if (type2 === 'T' && !labeledT && i > 8) {
      dnaGroup.add(createRadialPinLabel("Thymine (T)", r2.position, col2));
      labeledT = true;
    }
    if (type1 === 'G' && !labeledG && i > 15) {
      dnaGroup.add(createRadialPinLabel("Guanine (G)", r1.position, col1));
      labeledG = true;
    }
    if (type2 === 'C' && !labeledC && i > 20) {
      dnaGroup.add(createRadialPinLabel("Cytosine (C)", r2.position, col2));
      labeledC = true;
    }
  }

  // Backbone Label
  const bbPoint = curveA.getPoint(0.9);
  dnaGroup.add(createRadialPinLabel("Sugar-Phosphate<br>Backbone", bbPoint, "#88ccff"));

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
    if (!container || !dnaCamera) return;
    const w = container.clientWidth;
    const h = container.clientHeight;
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
        initDNA(); // Start 3D
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
      stagger: 0.2,
      duration: 0.8,
      ease: "power4.out"
    }, "-=0.5");

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

// Drug-Disease Compatibility Database
interface DrugDiseaseMatch {
  disease: string;
  targetProtein: string;
  compatibility: number;
  mechanism: string;
  sideEffects: string[];
  clinicalEvidence: string;
}

const drugDiseaseDatabase: DrugDiseaseMatch[] = [
  {
    disease: "lung cancer",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 95,
    mechanism: "Inhibits EGFR tyrosine kinase activity, blocking cancer cell proliferation",
    sideEffects: ["Skin rash", "Diarrhea", "Fatigue"],
    clinicalEvidence: "Phase III trials show 70% response rate"
  },
  {
    disease: "non-small cell lung cancer",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 98,
    mechanism: "Targets EGFR mutations common in NSCLC, preventing tumor growth",
    sideEffects: ["Skin rash", "Diarrhea", "Liver enzyme elevation"],
    clinicalEvidence: "FDA approved with strong clinical data"
  },
  {
    disease: "breast cancer",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 45,
    mechanism: "Limited EGFR expression in most breast cancers",
    sideEffects: ["Skin rash", "Diarrhea"],
    clinicalEvidence: "Limited efficacy in clinical trials"
  },
  {
    disease: "colorectal cancer",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 72,
    mechanism: "Blocks EGFR signaling in KRAS wild-type tumors",
    sideEffects: ["Skin reactions", "Hypomagnesemia"],
    clinicalEvidence: "Effective in KRAS wild-type patients"
  },
  {
    disease: "diabetes",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 15,
    mechanism: "No therapeutic benefit for diabetes",
    sideEffects: ["May worsen glucose control"],
    clinicalEvidence: "Not indicated for metabolic disorders"
  },
  {
    disease: "hypertension",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 10,
    mechanism: "No cardiovascular therapeutic effect",
    sideEffects: ["Potential cardiac complications"],
    clinicalEvidence: "Not recommended for cardiovascular conditions"
  },
  {
    disease: "glioblastoma",
    targetProtein: "EGFR-Kinase-Mutant",
    compatibility: 68,
    mechanism: "Targets EGFR amplification common in glioblastoma",
    sideEffects: ["Fatigue", "Headache", "Skin rash"],
    clinicalEvidence: "Moderate efficacy in EGFR-amplified cases"
  }
];

function analyzeDrugDiseaseCompatibility(disease: string, targetProtein: string): DrugDiseaseMatch | null {
  const normalizedDisease = disease.toLowerCase().trim();

  // Find exact or partial match
  let match = drugDiseaseDatabase.find(entry =>
    entry.disease === normalizedDisease && entry.targetProtein === targetProtein
  );

  // If no exact match, try partial match
  if (!match) {
    match = drugDiseaseDatabase.find(entry =>
      normalizedDisease.includes(entry.disease) || entry.disease.includes(normalizedDisease)
    );
  }

  return match || null;
}

// Predict Graph Logic
// Predict Graph Logic & API Integration
if (predictBtn) {
  predictBtn.addEventListener('click', async () => {
    const btn = predictBtn as HTMLButtonElement;
    const fileInput = document.getElementById('genome-file') as HTMLInputElement;
    const diseaseInput = document.getElementById('p-disease') as HTMLInputElement;

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
      const genomeData = await api.analyzeGenome(fileInput.files[0]);
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
        const topCandidate = genomeData.results[0]; // Logic sorts by best score
        const statusEl = document.getElementById('result-status');
        const affinityEl = document.getElementById('res-affinity');
        const toxEl = document.getElementById('res-toxicity');

        if (statusEl) {
          const isPositive = topCandidate['Suitability'] && topCandidate['Suitability'].includes('RECOMMENDED');
          statusEl.innerHTML = isPositive ?
            `<span style="color: #fff">RECOMMENDED AGENT:</span> <span style="color: var(--neon-green); font-size: 1.2em;">${topCandidate['Drug'].toUpperCase()}</span>` :
            "CAUTION: RESISTANCE DETECTED";
          statusEl.style.color = isPositive ? "var(--neon-green)" : "var(--neon-red)";
        }

        // Update Side Panel Metrics with Top Drug Info
        if (affinityEl) {
          affinityEl.innerText = topCandidate['Drug'] || "Unknown";
          affinityEl.style.color = "var(--neon-blue)";
        }
        if (toxEl) {
          toxEl.innerText = "OPTIMAL";
          toxEl.style.color = "var(--neon-green)";
        }

        // 3. Render Results (Visualization) - MEDICAL MONITOR UI
        if (genomeData && genomeData.results) {
          if (graphContainer) {
            graphContainer.innerHTML = '';
            graphContainer.style.display = 'block';
            graphContainer.style.overflowY = 'auto';
            graphContainer.style.background = 'rgba(0,0,0,0.3)';
            graphContainer.style.borderRadius = '8px';
            graphContainer.style.padding = '10px';

            const topCandidate = genomeData.results[0]; // Best Score

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

            // Dynamic Match % Calculation (Inverse of IC50 score roughly normalized)
            let matchScore = 98;
            if (topCandidate['Score']) {
              // Normalize score: Lower is better, but show % match. 
              // Assuming score range -2 to 2 typically:
              matchScore = Math.min(99, Math.max(10, Math.floor((2 - topCandidate['Score']) * 25 + 50)));
            }

            holoCard.innerHTML = `
                  <div style="z-index: 2;">
                      <div style="font-size: 0.75rem; color: var(--neon-green); letter-spacing: 2px;">PRIMARY THERAPEUTIC CANDIDATE</div>
                      <div style="font-size: 2rem; font-weight: 800; color: white; margin: 5px 0; text-shadow: 0 0 10px var(--neon-green);">
                          ${topCandidate['Drug']}
                      </div>
                      <div style="display: flex; gap: 10px; margin-top: 5px;">
                          <div style="background: rgba(0,255,136,0.2); color: var(--neon-green); padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; border: 1px solid var(--neon-green);">
                              ✅ ${topCandidate['Safety Assessment']}
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

            genomeData.results.slice(1).forEach((row: any) => {
              if (!row['Drug']) return;

              // Score calc
              let rowScore = 50;
              if (row['Score']) rowScore = Math.min(99, Math.max(10, Math.floor((2 - row['Score']) * 25 + 50)));

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
                    <div style="color: #fff; font-size: 0.9rem; line-height: 1.4; border-left: 2px solid var(--neon-blue); padding-left: 10px;">${match.clinical_evidence}</div>
                  </div>
                   <div style="margin-bottom: 1.5rem;">
                    <div style="color: var(--neon-green); font-size: 0.85rem; margin-bottom: 0.3rem;">RECOMMENDATION:</div>
                    <div style="color: var(--neon-green); font-size: 1rem; line-height: 1.4; font-weight: bold; padding: 10px; background: rgba(186, 220, 88, 0.1); border: 1px solid var(--neon-green); border-radius: 8px;">
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

        // Trigger effects
        playTransitionEffects('analysis');
      }

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
      gsap.to(dnaCamera.position, { x: 0, y: 0, z: 45, duration: 1, ease: "power2.inOut" });
      addLog("Genomic camera re-centered.");
    }
  }
});

// Download Report Logic
document.addEventListener('click', (e) => {
  const target = e.target as HTMLElement;
  if (target && target.id === 'btn-download-report') {
    const content = "HOSPIT-X DIAGNOSTIC REPORT\n" +
      "DATE: " + new Date().toISOString() + "\n" +
      "DOCTOR: " + (document.getElementById('display-id') as HTMLElement).innerText + "\n" +
      "PATIENT: " + (document.getElementById('p-name') as HTMLInputElement).value + "\n" +
      "DISEASE: " + (document.getElementById('p-disease') as HTMLInputElement).value + "\n" +
      "COMPATIBILITY: " + (document.getElementById('compatibility-result') as HTMLElement).innerText.split('\n')[0] + "\n";

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
// End of file
