// Interactive maps of the scapholunate interval.
//
// Every measurement is the radial-ulnar gap across one coronal image row: from
// the scaphoid's ulnar-most pixel to the lunate's radial-most pixel. It is drawn
// as a block filling that gap, one pixel tall and one slice deep, so the solid
// between the two bones is the measured interval, coloured by its own width.
// Only rows within 1 mm of a slice's narrowest gap, and no wider than 7 mm, are
// kept, so the solid covers the apposed part of the joint rather than all of it.
//
// Bone surfaces come from the same manual segmentations the measurements were
// taken from, placed in the same millimetre frame.
//
// Axes: x radial(-)/ulnar(+), y proximal(-)/distal(+), z palmar(-)/dorsal(+);
// origin at the scaphoid's ulnar-most point on the slice of maximal scaphoid
// area. All wrists are shown in one standardised orientation.

import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);

function fail(msg) {
  const el = $('loading');
  if (el) { el.textContent = msg; el.style.maxWidth = '32em'; el.style.textAlign = 'center'; }
}
addEventListener('error', () => fail(
  'This figure could not start. It needs a browser with WebGL and ES modules.'));

// ---------------------------------------------------------------- colour ---
const VIRIDIS = [
  [0.267,0.005,0.329],[0.283,0.141,0.458],[0.254,0.265,0.530],[0.207,0.372,0.553],
  [0.164,0.471,0.558],[0.128,0.567,0.551],[0.135,0.659,0.518],[0.267,0.749,0.441],
  [0.478,0.821,0.318],[0.741,0.873,0.150],[0.993,0.906,0.144],
];
function viridis(t) {
  t = Math.min(1, Math.max(0, t)) * (VIRIDIS.length - 1);
  const i = Math.min(VIRIDIS.length - 2, Math.floor(t)), f = t - i;
  const a = VIRIDIS[i], b = VIRIDIS[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
const css = (c) => `rgb(${c.map((v) => Math.round(v * 255)).join(',')})`;

// ------------------------------------------------------------------ data ---
const res = await fetch('./data.json').catch(() => null);
if (!res || !res.ok) {
  fail('Could not load the measurement data. Reload the page, or use the repository linked from the paper.');
  throw new Error('data.json unavailable');
}
const data = await res.json();
const GRID = data.meta.grid;
const D_LO = 1.0, D_HI = 7.0;   // 7.0 is the pipeline cap; 1.0 is this cohort's minimum
const norm = (d) => (d - D_LO) / (D_HI - D_LO);
let slices = null, meshes = null;

// ----------------------------------------------------------------- scene ---
const BG = 0xf7f6f3;
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) {
  fail('This figure needs WebGL, which this browser did not provide.');
  throw e;
}
renderer.outputColorSpace = THREE.SRGBColorSpace;
$('stage').appendChild(renderer.domElement);
let running = true;
renderer.domElement.addEventListener('webglcontextlost', (e) => {
  e.preventDefault();
  running = false;
  const el = document.createElement('div');
  el.id = 'loading';
  el.textContent = 'The graphics context was lost. Reload the page.';
  document.body.appendChild(el);
});

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);

const camera = new THREE.PerspectiveCamera(36, 1, 0.1, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.85;

scene.add(new THREE.HemisphereLight(0xffffff, 0xb9b3a6, 2.0));
scene.add(new THREE.AmbientLight(0xffffff, 0.5));
const key = new THREE.DirectionalLight(0xffffff, 1.05); scene.add(key);
const rim = new THREE.DirectionalLight(0xffffff, 0.45); scene.add(rim);

// --------------------------------------------------------------- builders ---
function gapSolid(p) {
  const n = p.n, hy = p.ps / 2 * 0.94, hz = p.zs / 2 * 0.94;
  const pos = new Float32Array(n * 72), col = new Float32Array(n * 72);
  const nor = new Float32Array(n * 72), idx = new Uint32Array(n * 36);
  const F = [
    [[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1],   [-1,0,0]],
    [[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1],   [1,0,0]],
    [[0,-1,-1],[0,-1,1],[1,-1,1],[1,-1,-1], [0,-1,0]],
    [[0,1,-1],[1,1,-1],[1,1,1],[0,1,1],     [0,1,0]],
    [[0,-1,-1],[1,-1,-1],[1,1,-1],[0,1,-1], [0,0,-1]],
    [[0,-1,1],[0,1,1],[1,1,1],[1,-1,1],     [0,0,1]],
  ];
  let v = 0, o = 0;
  for (let k = 0; k < n; k++) {
    const x0 = p.sx[k], x1 = p.lx[k], y = p.y[k], z = p.z[k];
    const c = viridis(norm(p.d[k]));
    for (const f of F) {
      const base = v;
      for (let q = 0; q < 4; q++) {
        const s = f[q];
        pos[v*3] = s[0] ? x1 : x0; pos[v*3+1] = y + s[1]*hy; pos[v*3+2] = z + s[2]*hz;
        nor[v*3] = f[4][0]; nor[v*3+1] = f[4][1]; nor[v*3+2] = f[4][2];
        col[v*3] = c[0]; col[v*3+1] = c[1]; col[v*3+2] = c[2];
        v++;
      }
      idx[o++] = base; idx[o++] = base+2; idx[o++] = base+1;
      idx[o++] = base; idx[o++] = base+3; idx[o++] = base+2;
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(nor, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  g.computeBoundingBox();
  return g;
}

// Cohort sheet: a surface through the middle of the joint over the 7x7 (y,z)
// grid, coloured by the mean interval width there. Position is anatomy and
// colour is width, the same convention as the per-wrist view.
function meanSheet() {
  const f = data.cohort.field, mid = data.cohort.mid, m = GRID.length;
  const tri = [], cols = [];
  const at = (zi, yi) => [mid[zi][yi], GRID[yi], GRID[zi], f[zi][yi]];
  for (let zi = 0; zi < m - 1; zi++) for (let yi = 0; yi < m - 1; yi++) {
    const q = [at(zi, yi), at(zi, yi+1), at(zi+1, yi+1), at(zi+1, yi)];
    for (const t of [[0,1,2],[0,2,3]]) for (const k of t) {
      tri.push(q[k]); cols.push(viridis(norm(q[k][3])));
    }
  }
  const pos = new Float32Array(tri.length*3), col = new Float32Array(tri.length*3);
  tri.forEach((t, i) => { pos[i*3]=t[0]; pos[i*3+1]=t[1]; pos[i*3+2]=t[2];
                          col[i*3]=cols[i][0]; col[i*3+1]=cols[i][1]; col[i*3+2]=cols[i][2]; });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals(); g.computeBoundingBox();
  return g;
}

function cohortCloud() {
  const pos = [], col = [];
  for (const p of data.patients) for (let k = 0; k < p.n; k++) {
    if (Math.abs(p.y[k]) > data.meta.win || Math.abs(p.z[k]) > data.meta.win) continue;
    const c = viridis(norm(p.d[k]));
    pos.push((p.sx[k] + p.lx[k]) / 2, p.y[k], p.z[k]);
    col.push(c[0], c[1], c[2]);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.computeBoundingBox();
  return g;
}

// ----------------------------------------------------------------- build ---
const MAT_GAP = new THREE.MeshStandardMaterial({
  vertexColors: true, roughness: 0.55, metalness: 0.0, side: THREE.FrontSide });
const MAT_BONE = new THREE.MeshStandardMaterial({
  color: 0xdfcfb2, roughness: 0.92, metalness: 0.0,
  transparent: true, opacity: 0.46, depthWrite: false, side: THREE.DoubleSide });

const solids = data.patients.map((p) => {
  const m = new THREE.Mesh(gapSolid(p), MAT_GAP);
  m.visible = false; scene.add(m); return m;
});
const boneGroups = data.patients.map(() => {
  const g = new THREE.Group(); g.visible = false; g.renderOrder = 5; scene.add(g); return g;
});

const sheetGeom = meanSheet();
const sheet = new THREE.Mesh(sheetGeom, new THREE.MeshStandardMaterial({
  vertexColors: true, roughness: 0.5, metalness: 0.0, side: THREE.DoubleSide,
  transparent: true, opacity: 0.95 }));
sheet.visible = false; scene.add(sheet);

const cloudGeom = cohortCloud();
const cloud = new THREE.Points(cloudGeom, new THREE.PointsMaterial({
  size: 0.2, vertexColors: true, transparent: true, opacity: 0.75,
  sizeAttenuation: true, depthWrite: false }));
cloud.visible = false; scene.add(cloud);

// outline of the coronal slice currently shown in the inset
const planeGeom = new THREE.BufferGeometry();
planeGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(15), 3));
const plane = new THREE.Line(planeGeom, new THREE.LineBasicMaterial({
  color: 0xa8a399, transparent: true, opacity: 0.55 }));
plane.visible = false; scene.add(plane);

const MARK = new THREE.Mesh(new THREE.OctahedronGeometry(1),
  new THREE.MeshBasicMaterial({ color: 0x0f7f9b, depthTest: false, depthWrite: false }));
MARK.renderOrder = 20; scene.add(MARK);

// axis gizmo, drawn into its own corner viewport so nothing can occlude it
const AXES = [
  { dir: [1,0,0], hex: 0xc2352f, name: 'ulnar' },
  { dir: [0,1,0], hex: 0x2c7a3f, name: 'distal' },
  { dir: [0,0,1], hex: 0x1a5f9e, name: 'dorsal' },
];
const GIZMO_PX = 108;
const gizmoScene = new THREE.Scene();
const gizmoCam = new THREE.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, 0.01, 20);
gizmoScene.add(new THREE.HemisphereLight(0xffffff, 0x999999, 2.4));
for (const a of AXES) {
  gizmoScene.add(new THREE.ArrowHelper(
    new THREE.Vector3(...a.dir), new THREE.Vector3(), 1, a.hex, 0.3, 0.17));
}

// ---------------------------------------------------------------- labels ---
const tags = [];
function tag(cls, html) {
  const el = document.createElement('div');
  el.className = 'tag ' + cls; el.innerHTML = html;
  document.body.appendChild(el);
  const t = { el, at: new THREE.Vector3(), on: false, gizmo: false };
  tags.push(t); return t;
}
AXES.forEach((a) => {
  const t = tag('axis', a.name);
  t.el.style.color = '#' + a.hex.toString(16).padStart(6, '0');
  t.at.set(a.dir[0]*1.34, a.dir[1]*1.34, a.dir[2]*1.34);
  t.gizmo = t.on = true;
});
const matchTag = tag('match', '');

// ------------------------------------------------------------------ view ---
const HOME = new THREE.Vector3(0.5, 0.3, 1).normalize();
let scale = 1, lastBox = null;

function frameOn(box, keep) {
  const c = box.getCenter(new THREE.Vector3());
  const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 2);
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
  const dist = (r / Math.sin(Math.min(vFov, hFov) / 2)) * 1.08;
  const dir = keep && camera.position.distanceTo(controls.target) > 1e-3
    ? camera.position.clone().sub(controls.target).normalize() : HOME.clone();
  controls.target.copy(c);
  camera.position.copy(c).addScaledVector(dir, dist);
  camera.up.set(0, 1, 0);
  camera.near = dist / 500; camera.far = dist * 20;
  controls.minDistance = r * 0.12; controls.maxDistance = r * 9;
  camera.updateProjectionMatrix(); controls.update();
  lastBox = box;
  scale = Math.min(Math.max(r / 9, 0.3), 1.7);
  MARK.scale.setScalar(scale * 0.42);
}

// ------------------------------------------------------------- 2D inset ---
const SVGNS = 'http://www.w3.org/2000/svg';
function el(name, attrs) {
  const n = document.createElementNS(SVGNS, name);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  return n;
}

function drawSlice() {
  const svg = $('sliceSvg');
  svg.textContent = '';
  const p = data.patients[current];
  const list = slices ? slices[p.id] : null;
  if (!list || !list.length) { $('sliceZ').textContent = ''; return; }
  const s = list[Math.min(sliceIdx, list.length - 1)];

  // one frame for the whole wrist so the outline does not jump between slices
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const sl of list) for (const poly of [sl.a, sl.b]) for (const q of poly) {
    x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]);
    y0 = Math.min(y0, q[1]); y1 = Math.max(y1, q[1]);
  }
  const w = svg.clientWidth || 244, h = svg.clientHeight || 158, pad = 8;
  const k = Math.min((w - 2*pad) / (x1 - x0), (h - 2*pad) / (y1 - y0));
  const ox = (w - (x1 - x0) * k) / 2, oy = (h - (y1 - y0) * k) / 2;
  const X = (mm) => ox + (mm - x0) * k;
  const Y = (mm) => h - (oy + (mm - y0) * k);   // distal upwards

  for (const [poly, name] of [[s.a, 'scaphoid'], [s.b, 'lunate']]) {
    svg.appendChild(el('polygon', {
      points: poly.map((q) => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(' '),
      fill: '#ece3d2', stroke: '#b3a78e', 'stroke-width': 1, 'stroke-linejoin': 'round' }));
    const cx = poly.reduce((a, q) => a + q[0], 0) / poly.length;
    const cy = poly.reduce((a, q) => a + q[1], 0) / poly.length;
    const t = el('text', { x: X(cx).toFixed(1), y: Y(cy).toFixed(1), fill: '#8d8378',
      'font-size': 9, 'text-anchor': 'middle', 'dominant-baseline': 'middle' });
    t.textContent = name;
    svg.appendChild(t);
  }
  // the measured rows on this slice, each coloured by its own width
  const tol = p.zs / 2 + 1e-6;
  for (let i = 0; i < p.n; i++) {
    if (Math.abs(p.z[i] - s.z) > tol) continue;
    svg.appendChild(el('line', {
      x1: X(p.sx[i]).toFixed(1), x2: X(p.lx[i]).toFixed(1),
      y1: Y(p.y[i]).toFixed(1), y2: Y(p.y[i]).toFixed(1),
      stroke: css(viridis(norm(p.d[i]))), 'stroke-width': Math.max(1, p.ps * k) }));
  }
  $('sliceZ').textContent = s.z === 0 ? 'centre'
    : `${s.z.toFixed(1)} mm ${s.z > 0 ? 'dorsal' : 'palmar'}`;

  // slice outline in the 3D scene
  const b = solids[current].geometry.boundingBox;
  const a = planeGeom.attributes.position.array;
  const pts = [[b.min.x, b.min.y], [b.max.x, b.min.y], [b.max.x, b.max.y],
               [b.min.x, b.max.y], [b.min.x, b.min.y]];
  pts.forEach((q, i) => { a[i*3] = q[0]; a[i*3+1] = q[1]; a[i*3+2] = s.z; });
  planeGeom.attributes.position.needsUpdate = true;
  planeGeom.computeBoundingSphere();
  plane.visible = mode === 'patient';
}

// ------------------------------------------------------------------- ui ----
let mode = 'patient', current = 0, sliceIdx = 0;

data.patients.forEach((p, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = p.id;
  $('pick').appendChild(o);
});

const _box = new THREE.Box3();
function patientBox(i) {
  const b = solids[i].geometry.boundingBox.clone();
  boneGroups[i].children.forEach((m) => {
    m.geometry.computeBoundingBox();
    b.union(_box.copy(m.geometry.boundingBox));
  });
  return b;
}

// open on the slice nearest the joint centre, not the far palmar edge
function centralSlice() {
  const l = slices ? slices[data.patients[current].id] : null;
  if (!l || !l.length) return 0;
  let bi = 0, bd = Infinity;
  l.forEach((s, i) => { const d = Math.abs(s.z); if (d < bd) { bd = d; bi = i; } });
  return bi;
}

function sliceCount() {
  const l = slices ? slices[data.patients[current].id] : null;
  return l ? l.length : 0;
}

function apply(refit) {
  const patient = mode === 'patient';
  const p = data.patients[current];

  solids.forEach((m, i) => m.visible = patient && i === current);
  boneGroups.forEach((g, i) => g.visible = patient && i === current && $('lBones').checked);
  cloud.visible = !patient;
  sheet.visible = !patient;
  plane.visible = patient && sliceCount() > 0;

  $('prev').disabled = !patient || current === 0;
  $('next').disabled = !patient || current === data.patients.length - 1;
  $('pick').disabled = !patient;
  $('who').textContent = patient ? `${p.id} of ${data.patients.length}` : 'all';
  $('mPatient').setAttribute('aria-pressed', patient);
  $('mCohort').setAttribute('aria-pressed', !patient);
  $('lBones').disabled = !patient || !meshes;
  $('slicePanel').style.visibility = patient ? 'visible' : 'hidden';

  if (refit) frameOn(patient ? patientBox(current)
                             : cloudGeom.boundingBox.clone().union(sheetGeom.boundingBox),
                     refit === 'keep');

  if (patient) {
    MARK.visible = matchTag.on = true;
    let bi = 0, bd = Infinity;
    for (let k = 0; k < p.n; k++) {
      const dd = (p.y[k] - p.best.y) ** 2 + (p.z[k] - p.best.z) ** 2;
      if (dd < bd) { bd = dd; bi = k; }
    }
    const mx = (p.sx[bi] + p.lx[bi]) / 2;
    MARK.position.set(mx, p.best.y, p.best.z);
    matchTag.at.set(mx, p.best.y - scale * 1.35, p.best.z);
    const diff = Math.abs(p.best.d - p.manual);
    matchTag.el.innerHTML = `closest to the radiograph: <b>${p.best.d.toFixed(2)} mm</b>` +
      ` (radiograph ${p.manual.toFixed(2)}, off by ${diff.toFixed(2)})`;
    $('subtitle').textContent =
      `${p.n} measurements over ${sliceCount() || '?'} slices, radiograph ${p.manual.toFixed(2)} mm`;
  } else {
    MARK.visible = false; matchTag.on = false;
    $('subtitle').textContent =
      `${data.meta.inwin} of ${data.meta.total} measurements, the part inside the sampled window`;
  }
  const max = Math.max(0, sliceCount() - 1);
  $('sliceRange').max = max;
  sliceIdx = Math.min(sliceIdx, max);
  $('sliceRange').value = sliceIdx;
  $('sPrev').disabled = sliceIdx === 0;
  $('sNext').disabled = sliceIdx >= max;
  drawSlice();
}

function goto(i, refit) {
  current = Math.min(Math.max(i, 0), data.patients.length - 1);
  $('pick').value = current;
  sliceIdx = centralSlice();
  history.replaceState(null, '', '#' + data.patients[current].id);
  apply(refit || 'keep');
}
function setSlice(i) {
  sliceIdx = Math.min(Math.max(i, 0), Math.max(0, sliceCount() - 1));
  $('sliceRange').value = sliceIdx;
  $('sPrev').disabled = sliceIdx === 0;
  $('sNext').disabled = sliceIdx >= sliceCount() - 1;
  drawSlice();
}

$('prev').onclick = () => goto(current - 1);
$('next').onclick = () => goto(current + 1);
$('pick').onchange = (e) => goto(+e.target.value);
$('mPatient').onclick = () => { if (mode !== 'patient') { mode = 'patient'; apply('keep'); } };
$('mCohort').onclick  = () => { if (mode !== 'cohort')  { mode = 'cohort';  apply('keep'); } };
$('lBones').onchange = () => apply(false);
$('reset').onclick = () => apply('home');
$('sPrev').onclick = () => setSlice(sliceIdx - 1);
$('sNext').onclick = () => setSlice(sliceIdx + 1);
$('sliceRange').oninput = (e) => setSlice(+e.target.value);
addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, button')) return;
  if (e.key === 'ArrowLeft')  { goto(current - 1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { goto(current + 1); e.preventDefault(); }
  if (e.key === 'ArrowUp')    { setSlice(sliceIdx + 1); e.preventDefault(); }
  if (e.key === 'ArrowDown')  { setSlice(sliceIdx - 1); e.preventDefault(); }
});

{
  const stops = [];
  for (let i = 0; i <= 10; i++) stops.push(`${css(viridis(i / 10))} ${i * 10}%`);
  $('ramp').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  $('lo').textContent = D_LO.toFixed(1);
  $('hi').textContent = '\u2265 ' + D_HI.toFixed(1) + ' mm';
  $('note').textContent =
    'Each block is the radial-ulnar gap across one coronal row, the same horizontal ' +
    'gap a PA radiograph reads. All wrists are shown in one standardised orientation, ' +
    'so a wrist of the opposite side appears mirrored.';
}

function resize() {
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
  if (lastBox) frameOn(lastBox, true);
  drawSlice();
}
let resizeQueued = false;
addEventListener('resize', () => {
  if (resizeQueued) return;
  resizeQueued = true;
  requestAnimationFrame(() => { resizeQueued = false; resize(); });
});

const NICE = [1, 2, 5, 10, 20, 50];
let sbShown = null;
function updateScaleBar() {
  const d = camera.position.distanceTo(controls.target);
  const perPx = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * d / innerHeight;
  let mm = NICE[0];
  for (const c of NICE) { if (c / perPx <= 150) mm = c; }
  const px = Math.round(mm / perPx);
  if (mm === sbShown && Math.abs(px - parseFloat($('sbLine').style.width || 0)) < 1) return;
  sbShown = mm;
  $('sbLine').style.width = px + 'px';
  $('sbText').textContent = mm + ' mm';
}

const v = new THREE.Vector3();
function loop() {
  if (!running) return;
  controls.update();
  updateScaleBar();
  key.position.copy(camera.position);
  rim.position.copy(camera.position).negate();
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.render(scene, camera);

  const gx = 26, gy = Math.max(8, innerHeight - 92 - GIZMO_PX);
  gizmoCam.position.copy(camera.position).sub(controls.target).setLength(6);
  gizmoCam.up.copy(camera.up); gizmoCam.lookAt(0, 0, 0);
  renderer.autoClear = false; renderer.clearDepth();
  renderer.setViewport(gx, gy, GIZMO_PX, GIZMO_PX);
  renderer.render(gizmoScene, gizmoCam);
  renderer.autoClear = true;

  for (const t of tags) {
    if (!t.on) { t.el.style.opacity = 0; continue; }
    v.copy(t.at).project(t.gizmo ? gizmoCam : camera);
    const vis = v.z > -1 && v.z < 1;
    t.el.style.opacity = vis ? 1 : 0;
    if (!vis) continue;
    if (t.gizmo) {
      t.el.style.left = (gx + (v.x*0.5 + 0.5) * GIZMO_PX) + 'px';
      t.el.style.top = (innerHeight - gy - (v.y*0.5 + 0.5) * GIZMO_PX) + 'px';
    } else {
      t.el.style.left = ((v.x*0.5 + 0.5) * innerWidth) + 'px';
      t.el.style.top = ((-v.y*0.5 + 0.5) * innerHeight) + 'px';
    }
  }
  requestAnimationFrame(loop);
}

{
  const want = data.patients.findIndex((p) => p.id === location.hash.slice(1).toUpperCase());
  if (want >= 0) { current = want; $('pick').value = current; }
}
resize(); apply('home');
$('loading').remove();
loop();

// ------------------------------------------------- progressive extras -----
// Slice outlines and bone surfaces are the bulk of the payload, so they load
// after the first frame rather than holding it up.
fetch('./slices.json').then((r) => r.json()).then((j) => {
  slices = j; sliceIdx = centralSlice(); apply(false);
}).catch(() => {});

fetch('./meshes.json').then((r) => r.json()).then(async (index) => {
  const buf = await (await fetch('./meshes.bin')).arrayBuffer();
  meshes = index;
  data.patients.forEach((p, i) => {
    const rec = index.patients[p.id];
    if (!rec) return;
    for (const bone of ['scaphoid', 'lunate']) {
      const m = rec[bone];
      if (!m) continue;
      const q = new Int16Array(buf, m.vertOffset, m.vertCount * 3);
      const pos = new Float32Array(m.vertCount * 3);
      for (let k = 0; k < m.vertCount; k++) for (let c = 0; c < 3; c++)
        pos[k*3+c] = q[k*3+c] * m.scale[c] + m.offset[c];
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      g.setIndex(new THREE.BufferAttribute(new Uint16Array(buf, m.idxOffset, m.idxCount), 1));
      g.computeVertexNormals();
      boneGroups[i].add(new THREE.Mesh(g, MAT_BONE));
    }
  });
  apply('keep');
}).catch(() => { $('lBones').disabled = true; });
