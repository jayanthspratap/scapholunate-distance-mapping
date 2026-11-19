// Interactive maps of the scapholunate interval on CT.
//
// Every measurement is the radial-ulnar gap across one coronal image row: from
// the scaphoid's ulnar-most pixel to the lunate's radial-most pixel. It is drawn
// as a block filling that gap, one pixel tall and one slice deep, so the solid
// between the two bones is the measured interval, coloured by its own width.
// Only rows within 1 mm of a slice's narrowest gap, and no wider than 7 mm, are
// kept, so the solid covers the apposed part of the joint rather than all of it.
// Position is always anatomy and colour is always width; nothing is interpolated.
//
// Bone surfaces come from the same manual segmentations the measurements were
// taken from, placed in the same millimetre frame.
//
// Axes: x radial(-)/ulnar(+), y proximal(-)/distal(+), z palmar(-)/dorsal(+);
// origin at the scaphoid's ulnar-most point on the slice of maximal scaphoid
// area. All wrists are shown in one standardised orientation. Slice positions
// use each scan's true slice spacing.

import * as THREE from './vendor/three.module.min.js';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const COARSE = matchMedia('(pointer: coarse)').matches;

function fail(msg) {
  const el = $('loading');
  if (el) { el.textContent = msg; el.style.maxWidth = '32em'; }
}
addEventListener('error', () => fail('This figure could not start. It needs a browser with WebGL.'));

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
// vertex colours are read as linear, so convert from the sRGB legend values
const _c = new THREE.Color();
const lin = (c) => { _c.setRGB(c[0], c[1], c[2], THREE.SRGBColorSpace); return [_c.r, _c.g, _c.b]; };
const D_LO = 1.0, D_HI = 7.0;   // 7.0 is the pipeline cap; the cohort's narrowest gap is just above 1.0
const norm = (d) => (d - D_LO) / (D_HI - D_LO);
const mm1 = (v) => v.toFixed(1) + ' mm';

// Words for a position from the origin, e.g. "1.0 mm distal, 2.0 mm dorsal".
function where(y, z) {
  if (Math.abs(y) < 0.05 && Math.abs(z) < 0.05) return 'at the grid centre';
  const part = (v, pos, neg) => `${Math.abs(v).toFixed(1)} mm ${v >= 0.05 ? pos : v <= -0.05 ? neg : pos}`;
  return `${part(y, 'distal', 'proximal')}, ${part(z, 'dorsal', 'palmar')}`;
}

// ------------------------------------------------------------------ data ---
// The page loads this file as app.js?v=<release>; the same stamp goes on every data file,
// so a cached copy from an earlier release is never mixed with this one.
const V = new URL(import.meta.url).search;
const res = await fetch('./data.json' + V).catch(() => null);
if (!res || !res.ok) {
  fail('Could not load the measurement data. Reload the page, or use the repository linked from the paper.');
  throw new Error('data.json unavailable');
}
const data = await res.json();
const GRID = data.meta.grid;
let slices = null, meshesReady = false;

// ----------------------------------------------------------------- scene ---
const BG = 0xf6f5f1;
const stage = $('stage');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true });
} catch (e) {
  fail('This figure needs WebGL, which this browser did not provide.');
  throw e;
}
renderer.outputColorSpace = THREE.SRGBColorSpace;
stage.appendChild(renderer.domElement);
let running = true;
renderer.domElement.addEventListener('webglcontextlost', (e) => {
  e.preventDefault(); running = false;
  const el = document.createElement('div');
  el.id = 'loading'; el.textContent = 'The graphics context was lost. Reload the page.';
  stage.parentElement.appendChild(el);
});

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG);
const camera = new THREE.PerspectiveCamera(22, 1, 0.1, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.09;
controls.rotateSpeed = 0.8;
controls.enablePan = true;

// Lambert shading with ambient + head-light summing to exactly 1 (three's diffuse BRDF is 1/pi),
// so a face turned to the viewer shows its legend colour; side faces are darker, for shape.
scene.add(new THREE.AmbientLight(0xffffff, 0.5 * Math.PI));
const key = new THREE.DirectionalLight(0xffffff, 0.5 * Math.PI); scene.add(key);

// --------------------------------------------------------------- builders ---
const FACES = [
  [[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1],   [-1,0,0]],
  [[1,-1,-1],[1,-1,1],[1,1,1],[1,1,-1],   [1,0,0]],
  [[0,-1,-1],[0,-1,1],[1,-1,1],[1,-1,-1], [0,-1,0]],
  [[0,1,-1],[1,1,-1],[1,1,1],[0,1,1],     [0,1,0]],
  [[0,-1,-1],[1,-1,-1],[1,1,-1],[0,1,-1], [0,0,-1]],
  [[0,-1,1],[0,1,1],[1,1,1],[1,-1,1],     [0,0,1]],
];
const TRI_PER_BLOCK = 12;

function gapSolid(p) {
  const n = p.n, hy = p.ps / 2 * 0.94, hz = p.zs / 2 * 0.94;
  const pos = new Float32Array(n * 72), col = new Float32Array(n * 72);
  const nor = new Float32Array(n * 72), idx = new Uint32Array(n * 36);
  let v = 0, o = 0;
  for (let k = 0; k < n; k++) {
    const x0 = p.sx[k], x1 = p.lx[k], y = p.y[k], z = p.z[k];
    const c = lin(viridis(norm(p.d[k])));
    for (const f of FACES) {
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
  g.computeBoundingBox(); g.computeBoundingSphere();
  return g;
}

// Cohort: one flat tile per point of the 7x7 (y, z) grid, at the mean
// radial-ulnar position of the gap there, coloured by the mean width.
// Tiles are separate, so nothing is interpolated between grid points.
const TILE = 0.9;
function cohortTiles() {
  const f = data.cohort.field, mid = data.cohort.mid, m = GRID.length;
  const pos = [], col = [], cells = [];
  for (let zi = 0; zi < m; zi++) for (let yi = 0; yi < m; yi++) {
    const x = mid[zi][yi], y = GRID[yi], z = GRID[zi], c = lin(viridis(norm(f[zi][yi])));
    const h = TILE / 2;
    const q = [[x, y-h, z-h], [x, y+h, z-h], [x, y+h, z+h], [x, y-h, z+h]];
    for (const t of [0, 1, 2, 0, 2, 3]) { pos.push(...q[t]); col.push(...c); }
    cells.push({ y, z, d: f[zi][yi] });
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.computeBoundingBox(); g.computeBoundingSphere();
  return { g, cells };
}

// ----------------------------------------------------------------- build ---
// The interval is drawn after the see-through bones, so its colours stay true to the legend
// wherever it lies behind bone, as on a radiograph.
const MAT_GAP = new THREE.MeshLambertMaterial({ vertexColors: true,
  transparent: true, opacity: 1, depthWrite: true });
const MAT_BONE = new THREE.MeshLambertMaterial({
  color: 0xe2d5bd,
  transparent: true, opacity: 0.38, depthWrite: false, side: THREE.FrontSide });

const solids = data.patients.map((p) => {
  const m = new THREE.Mesh(gapSolid(p), MAT_GAP);
  m.renderOrder = 10; m.visible = false; scene.add(m); return m;
});
const boneGroups = data.patients.map(() => {
  const g = new THREE.Group(); g.visible = false; scene.add(g); return g;   // meshes carry renderOrder 5
});
const boneBoxes = data.patients.map(() => ({}));

const tiles = cohortTiles();
const tileMesh = new THREE.Mesh(tiles.g, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide }));
tileMesh.visible = false; scene.add(tileMesh);
// a hairline frame around the grid so the tiles read as one map
const gridFrame = (() => {
  const h = TILE / 2, lo = GRID[0] - h, hi = GRID[GRID.length - 1] + h;
  const xs = data.cohort.mid.flat(), x = xs.reduce((a, b) => a + b, 0) / xs.length;
  const g = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(x, lo, lo), new THREE.Vector3(x, hi, lo), new THREE.Vector3(x, hi, hi),
    new THREE.Vector3(x, lo, hi), new THREE.Vector3(x, lo, lo)]);
  const l = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xbdb7ab }));
  l.visible = false; scene.add(l); return l;
})();

// the coronal slice shown in the panel, outlined in the scene
const planeGeom = new THREE.BufferGeometry();
planeGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(15), 3));
const plane = new THREE.Line(planeGeom, new THREE.LineBasicMaterial({ color: 0x8f8a81, transparent: true, opacity: 0.7 }));
plane.visible = false; scene.add(plane);

// best match to the radiograph: a dark diamond with a white rim, visible through bone
const MARK = new THREE.Group();
{
  const rimM = new THREE.Mesh(new THREE.OctahedronGeometry(1.32),
    new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, depthWrite: false, side: THREE.BackSide, transparent: true }));
  const core = new THREE.Mesh(new THREE.OctahedronGeometry(1),
    new THREE.MeshBasicMaterial({ color: 0x1c1b19, depthTest: false, depthWrite: false, transparent: true }));
  rimM.renderOrder = 20; core.renderOrder = 21; MARK.add(rimM, core);
}
scene.add(MARK);

// hovered block outline
const HOVER = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(1, 1, 1)),
  new THREE.LineBasicMaterial({ color: 0x1c1b19, depthTest: false, transparent: true, opacity: 0.9 }));
HOVER.renderOrder = 22; HOVER.visible = false; scene.add(HOVER);

// outlines of the blocks on the slice shown in the panel
const SLICE_EDGES = new THREE.LineSegments(new THREE.BufferGeometry(),
  new THREE.LineBasicMaterial({ color: 0x1c1b19, depthTest: false, transparent: true, opacity: 0.55 }));
SLICE_EDGES.renderOrder = 15; SLICE_EDGES.visible = false; scene.add(SLICE_EDGES);
const BOX_EDGES = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
function sliceEdges(p, z) {
  const hy = p.ps / 2 * 0.94, hz = p.zs / 2 * 0.94, tol = p.zs / 2 + 1e-6, pos = [];
  let n = 0;
  for (let k = 0; k < p.n; k++) {
    if (Math.abs(p.z[k] - z) > tol) continue;
    n++;
    const x0 = p.sx[k], x1 = p.lx[k], y = p.y[k], zz = p.z[k];
    const c = [[x0,y-hy,zz-hz],[x1,y-hy,zz-hz],[x1,y+hy,zz-hz],[x0,y+hy,zz-hz],
               [x0,y-hy,zz+hz],[x1,y-hy,zz+hz],[x1,y+hy,zz+hz],[x0,y+hy,zz+hz]];
    for (const [a, b] of BOX_EDGES) pos.push(...c[a], ...c[b]);
  }
  SLICE_EDGES.geometry.dispose();
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  SLICE_EDGES.geometry = g;
  return n;
}

// cohort: each wrist's best match to its radiograph, as a small diamond
function nearestMid(p) {
  let bi = 0, bd = Infinity;
  for (let k = 0; k < p.n; k++) {
    const dd = (p.y[k] - p.best.y) ** 2 + (p.z[k] - p.best.z) ** 2;
    if (dd < bd) { bd = dd; bi = k; }
  }
  return (p.sx[bi] + p.lx[bi]) / 2;
}
const DIAMONDS = new THREE.Group(); DIAMONDS.visible = false; scene.add(DIAMONDS);
const DIA_RIM = new THREE.MeshBasicMaterial({ color: 0xffffff, depthTest: false, depthWrite: false, side: THREE.BackSide, transparent: true });
const DIA_CORE = new THREE.MeshBasicMaterial({ color: 0x1c1b19, depthTest: false, depthWrite: false, transparent: true });
const OCT = new THREE.OctahedronGeometry(1), OCT_RIM = new THREE.OctahedronGeometry(1.35);
data.patients.forEach((p, i) => {
  const g = new THREE.Group();
  const rim = new THREE.Mesh(OCT_RIM, DIA_RIM), core = new THREE.Mesh(OCT, DIA_CORE);
  rim.renderOrder = 20; core.renderOrder = 21; core.userData.wrist = i; g.add(rim, core);
  g.position.set(nearestMid(p), p.best.y, p.best.z); g.scale.setScalar(0.17);
  DIAMONDS.add(g);
});

// ---------------------------------------------------------------- labels ---
const tags = [];
function tag(cls, text) {
  const el = document.createElement('div');
  el.className = 'tag ' + cls; el.textContent = text;
  stage.parentElement.appendChild(el);
  const t = { el, at: new THREE.Vector3(), on: false, axis: null, op: 1 };
  tags.push(t); return t;
}
const boneTag = { scaphoid: tag('bone', 'Scaphoid'), lunate: tag('bone', 'Lunate') };
const DIRS = [
  ['Distal', [0, 1, 0]], ['Proximal', [0, -1, 0]],
  ['Ulnar', [1, 0, 0]], ['Radial', [-1, 0, 0]],
  ['Dorsal', [0, 0, 1]], ['Palmar', [0, 0, -1]],
].map(([name, a]) => { const t = tag('dir', name); t.axis = new THREE.Vector3(...a); return t; });

// ------------------------------------------------------------------ view ---
const VIEWS = {
  oblique: new THREE.Vector3(-0.42, 0.34, 1).normalize(),
  dorsal: new THREE.Vector3(0, 0, 1),
  radial: new THREE.Vector3(-1, 0, 0),
  distal: new THREE.Vector3(0, 1, -0.02).normalize(),
};
let lastBox = null, scale = 1, viewName = 'oblique';
let fly = null;

function fitDistance(box) {
  const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 2);
  const vFov = THREE.MathUtils.degToRad(camera.fov);
  const hFov = 2 * Math.atan(Math.tan(vFov / 2) * camera.aspect);
  return { r, dist: (r / Math.sin(Math.min(vFov, hFov) / 2)) * 0.98 };
}

function labelPad(box) { const s = box.getSize(new THREE.Vector3()); return Math.max(s.x, s.y, s.z) * 0.1 + 1.8; }
function frameOn(box, dirName, animate) {
  const c = box.getCenter(new THREE.Vector3());
  // fit the model plus the ring of direction labels around it
  const { r, dist } = fitDistance(box.clone().expandByScalar(labelPad(box) * 0.45));
  const dir = dirName ? VIEWS[dirName].clone()
    : camera.position.clone().sub(controls.target).normalize();
  const toPos = c.clone().addScaledVector(dir, dist);
  camera.near = dist / 200; camera.far = dist * 20; camera.updateProjectionMatrix();
  controls.minDistance = r * 0.15; controls.maxDistance = r * 8;
  lastBox = box;
  scale = Math.min(Math.max(r / 9, 0.3), 1.6);
  MARK.scale.setScalar(scale * 0.36);
  if (animate) {
    fly = { t0: performance.now(), dur: 520,
            fromT: controls.target.clone(), toT: c,
            fromD: camera.position.clone().sub(controls.target), toD: toPos.clone().sub(c) };
  } else {
    fly = null; controls.target.copy(c); camera.position.copy(toPos); controls.update();
  }
}
function stepFly(now) {
  if (!fly) return;
  let t = Math.min(1, (now - fly.t0) / fly.dur);
  const e = t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2;
  controls.target.lerpVectors(fly.fromT, fly.toT, e);
  const a = fly.fromD.clone().normalize(), b = fly.toD.clone().normalize();
  const len = THREE.MathUtils.lerp(fly.fromD.length(), fly.toD.length(), e);
  const q = new THREE.Quaternion().setFromUnitVectors(a, b);
  const dir = a.clone().applyQuaternion(new THREE.Quaternion().slerp(q, e));
  camera.position.copy(controls.target).addScaledVector(dir, len);
  if (t >= 1) fly = null;
}

function setView(name) {
  if (mode === 'cohort' && (name === 'dorsal' || name === 'distal')) name = 'radial';
  viewName = name;
  document.querySelectorAll('.views button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.view === name));
  frameOn(currentBox(), name, true);
}

// ------------------------------------------------------------- 2D slice ---
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
  if (mode !== 'wrist' || !list || !list.length) {
    $('sliceZ').textContent = ''; $('sliceN').textContent = ''; $('sliceEmpty').hidden = true;
    plane.visible = false; SLICE_EDGES.visible = false; return;
  }
  const si = Math.min(sliceIdx, list.length - 1), s = list[si];
  $('sliceN').textContent = `${si + 1} of ${list.length}`;
  const nOn = sliceEdges(p, s.z);
  SLICE_EDGES.visible = nOn > 0;
  $('sliceEmpty').hidden = nOn > 0;

  // one frame for the whole wrist so the outlines do not jump between slices
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const sl of list) for (const poly of [sl.a, sl.b]) for (const q of poly) {
    x0 = Math.min(x0, q[0]); x1 = Math.max(x1, q[0]);
    y0 = Math.min(y0, q[1]); y1 = Math.max(y1, q[1]);
  }
  const w = svg.clientWidth || 280, h = svg.clientHeight || 150, pad = 6;
  const k = Math.min((w - 2*pad) / (x1 - x0), (h - 2*pad) / (y1 - y0));
  const ox = (w - (x1 - x0) * k) / 2, oy = (h - (y1 - y0) * k) / 2;
  const X = (mm) => ox + (mm - x0) * k;
  const Y = (mm) => h - (oy + (mm - y0) * k);   // distal upwards

  for (const [poly, name] of [[s.a, 'scaphoid'], [s.b, 'lunate']]) {
    if (!poly.length) continue;
    svg.appendChild(el('polygon', {
      points: poly.map((q) => `${X(q[0]).toFixed(1)},${Y(q[1]).toFixed(1)}`).join(' '),
      fill: '#efe7d8', stroke: '#b9ab90', 'stroke-width': 1, 'stroke-linejoin': 'round' }));
    const cx = poly.reduce((a, q) => a + q[0], 0) / poly.length;
    const cy = poly.reduce((a, q) => a + q[1], 0) / poly.length;
    const t = el('text', { x: X(cx).toFixed(1), y: Y(cy).toFixed(1), fill: '#8a7f6c',
      'font-size': 10, 'text-anchor': 'middle', 'dominant-baseline': 'middle' });
    t.textContent = name;
    svg.appendChild(t);
  }
  const tol = p.zs / 2 + 1e-6;
  for (let i = 0; i < p.n; i++) {
    if (Math.abs(p.z[i] - s.z) > tol) continue;
    svg.appendChild(el('line', {
      x1: X(p.sx[i]).toFixed(1), x2: X(p.lx[i]).toFixed(1),
      y1: Y(p.y[i]).toFixed(1), y2: Y(p.y[i]).toFixed(1),
      stroke: css(viridis(norm(p.d[i]))), 'stroke-width': Math.max(1, p.ps * k) }));
  }
  $('sliceZ').textContent = Math.abs(s.z) < 0.05 ? 'centre slice'
    : `${Math.abs(s.z).toFixed(1)} mm ${s.z > 0 ? 'dorsal' : 'palmar'}`;

  const b = solids[current].geometry.boundingBox;
  const a = planeGeom.attributes.position.array;
  const m = 0.6;
  [[b.min.x-m, b.min.y-m], [b.max.x+m, b.min.y-m], [b.max.x+m, b.max.y+m], [b.min.x-m, b.max.y+m], [b.min.x-m, b.min.y-m]]
    .forEach((q, i) => { a[i*3] = q[0]; a[i*3+1] = q[1]; a[i*3+2] = s.z; });
  planeGeom.attributes.position.needsUpdate = true;
  planeGeom.computeBoundingSphere();
  plane.visible = true;
}

// ------------------------------------------------------------------- ui ----
let mode = 'wrist', current = 0, sliceIdx = 0;

data.patients.forEach((p, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = `${p.id}   (PA radiograph ${p.manual.toFixed(1)} mm)`;
  $('pick').appendChild(o);
});

const _box = new THREE.Box3();
function currentBox() {
  if (mode === 'cohort') {
    const b = tiles.g.boundingBox.clone();
    DIAMONDS.children.forEach((g) => b.expandByPoint(g.position));
    return b;
  }
  const b = solids[current].geometry.boundingBox.clone();
  if ($('lBones').checked) boneGroups[current].children.forEach((m) => b.union(_box.copy(m.geometry.boundingBox)));
  return b;
}
function sliceList() { return slices ? slices[data.patients[current].id] || [] : []; }
function centralSlice() {
  const l = sliceList();
  let bi = 0, bd = Infinity;
  l.forEach((s, i) => { const d = Math.abs(s.z); if (d < bd) { bd = d; bi = i; } });
  return bi;
}

function updateLegend() {
  const p = data.patients[current];
  const wrist = mode === 'wrist';
  $('radio').hidden = !wrist;
  $('scale').style.paddingTop = wrist ? '' : '0';
  $('keyWrist').hidden = !wrist;
  $('keyCohort').hidden = wrist;
  if (wrist) {
    const f = Math.min(1, Math.max(0, norm(p.manual))), r = $('radio');
    r.style.left = (f * 100) + '%';
    // keep the label inside the panel near either end of the scale
    r.style.alignItems = f > 0.82 ? 'flex-end' : f < 0.18 ? 'flex-start' : 'center';
    r.style.transform = f > 0.82 ? 'translateX(calc(-100% + 5px))' : f < 0.18 ? 'translateX(-5px)' : 'translateX(-50%)';
    $('radioText').textContent = `PA radiograph ${p.manual.toFixed(1)}`;
    $('bestD').textContent = mm1(p.best.d);
    $('manD').textContent = mm1(p.manual);
    $('farNote').hidden = Math.abs(p.best.d - p.manual) <= 0.5;
  }
}

function apply(refit) {
  const wrist = mode === 'wrist';
  const p = data.patients[current];
  solids.forEach((m, i) => m.visible = wrist && i === current);
  boneGroups.forEach((g, i) => g.visible = wrist && i === current && $('lBones').checked);
  tileMesh.visible = gridFrame.visible = DIAMONDS.visible = !wrist;
  document.querySelectorAll('.views button').forEach((b) => {
    b.hidden = !wrist && (b.dataset.view === 'dorsal' || b.dataset.view === 'distal');
  });
  $('keysHint').hidden = COARSE || !wrist;
  $('wristRow').hidden = !wrist;
  $('sliceSection').hidden = !wrist;
  $('bonesToggle').hidden = !wrist;
  $('noteWrist').hidden = !wrist;
  $('noteCohort').hidden = wrist;
  if (!hinted) $('hint').textContent = `${COARSE ? 'Tap' : 'Point at'} a ${wrist ? 'block' : 'tile'} to read its SLD`;
  $('mWrist').setAttribute('aria-pressed', wrist);
  $('mCohort').setAttribute('aria-pressed', !wrist);
  $('prev').disabled = current === 0;
  $('next').disabled = current === data.patients.length - 1;
  $('who').textContent = p.id;
  $('whoOf').textContent = `${current + 1} of ${data.patients.length}`;
  $('pick').value = current;

  MARK.visible = wrist;
  if (wrist) {
    MARK.position.set(nearestMid(p), p.best.y, p.best.z);
  }
  for (const b of ['scaphoid', 'lunate']) {
    const box = boneBoxes[current][b];
    boneTag[b].on = wrist && $('lBones').checked && !!box;
    if (box) {
      const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
      // near the outer (radial or ulnar) end of each bone, clear of the interval between them
      boneTag[b].at.set(c.x + (b === 'scaphoid' ? -0.3 : 0.3) * s.x, c.y + 0.25 * s.y, c.z);
    }
  }
  setHover(null);
  updateLegend();
  const max = Math.max(0, sliceList().length - 1);
  $('sliceRange').max = max;
  sliceIdx = Math.min(sliceIdx, max);
  $('sliceRange').value = sliceIdx;
  drawSlice();
  if (refit === 'view') frameOn(currentBox(), viewName, true);
  else if (refit === 'home') { viewName = mode === 'cohort' ? 'radial' : 'oblique'; markView(); frameOn(currentBox(), viewName, false); }
  else if (refit) frameOn(currentBox(), null, false);
}
function markView() {
  document.querySelectorAll('.views button').forEach((b) => b.setAttribute('aria-pressed', b.dataset.view === viewName));
}

function goto(i) {
  current = Math.min(Math.max(i, 0), data.patients.length - 1);
  sliceIdx = centralSlice();
  history.replaceState(null, '', '#' + data.patients[current].id);
  apply('keep');
}
function setSlice(i) {
  sliceIdx = Math.min(Math.max(i, 0), Math.max(0, sliceList().length - 1));
  $('sliceRange').value = sliceIdx;
  drawSlice();
}
function setMode(m) {
  if (mode === m) return;
  mode = m;
  history.replaceState(null, '', '#' + (m === 'cohort' ? 'mean' : data.patients[current].id));
  viewName = m === 'cohort' ? 'radial' : 'oblique';
  markView();
  apply(false);
  frameOn(currentBox(), viewName, true);
}

$('prev').onclick = () => goto(current - 1);
$('next').onclick = () => goto(current + 1);
$('pick').onchange = (e) => goto(+e.target.value);
$('mWrist').onclick = () => setMode('wrist');
$('mCohort').onclick = () => setMode('cohort');
$('lBones').onchange = () => { apply(false); frameOn(currentBox(), viewName, true); };
document.querySelectorAll('.views button').forEach((b) => b.onclick = () => setView(b.dataset.view));
$('sliceRange').oninput = (e) => setSlice(+e.target.value);
addEventListener('keydown', (e) => {
  if (e.target.matches('input[type=range], select')) return;
  if (mode === 'wrist' && e.key === 'ArrowLeft')  { goto(current - 1); e.preventDefault(); }
  if (mode === 'wrist' && e.key === 'ArrowRight') { goto(current + 1); e.preventDefault(); }
  if (mode === 'wrist' && e.key === 'ArrowUp')    { setSlice(sliceIdx + 1); e.preventDefault(); }
  if (mode === 'wrist' && e.key === 'ArrowDown')  { setSlice(sliceIdx - 1); e.preventDefault(); }
});
controls.addEventListener('start', () => {
  fly = null;
  if (viewName !== null) { viewName = null; markView(); }
});

// legend ramp and ticks
{
  const stops = [];
  for (let i = 0; i <= 20; i++) stops.push(`${css(viridis(i / 20))} ${i * 5}%`);
  $('ramp').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  $('miniRamp').style.background = $('ramp').style.background;
  for (let v = 1; v <= 7; v++) {
    const s = document.createElement('span');
    s.style.left = (norm(v) * 100) + '%';
    s.textContent = String(v);
    $('ticks').appendChild(s);
  }
}

// ---------------------------------------------------------------- hover ----
const ray = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let hoverAt = null, hinted = false;
function pick(clientX, clientY) {
  const r = renderer.domElement.getBoundingClientRect();
  ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  if (mode === 'cohort') {
    const dh = ray.intersectObject(DIAMONDS, true).find((h) => h.object.userData.wrist !== undefined);
    if (dh) {
      const i = dh.object.userData.wrist, p = data.patients[i];
      return { wristPick: i, text: `<b>${p.id}</b>: closest match ${mm1(p.best.d)}<br><span>PA radiograph ${mm1(p.manual)}</span>`,
               sx: clientX - r.left, sy: clientY - r.top };
    }
  }
  const target = mode === 'wrist' ? solids[current] : tileMesh;
  const hit = ray.intersectObject(target, false)[0];
  if (!hit) return null;
  if (mode === 'wrist') {
    const p = data.patients[current], k = Math.floor(hit.faceIndex / TRI_PER_BLOCK);
    return { d: p.d[k], y: p.y[k], z: p.z[k], x0: p.sx[k], x1: p.lx[k], hy: p.ps, hz: p.zs, k,
             sx: clientX - r.left, sy: clientY - r.top };
  }
  const c = tiles.cells[Math.floor(hit.faceIndex / 2)];
  return { ...c, cohort: true, sx: clientX - r.left, sy: clientY - r.top };
}
function setHover(h) {
  hoverAt = h;
  const tip = $('tip');
  if (!h) { tip.style.opacity = 0; HOVER.visible = false; return; }
  tip.innerHTML = h.wristPick !== undefined ? h.text : h.cohort
    ? `<b>${mm1(h.d)}</b> mean SLD of 25<br><span>${where(h.y, h.z)}</span>`
    : `<b>${mm1(h.d)}</b> SLD<br><span>${where(h.y, h.z)}</span>`;
  const r = renderer.domElement.getBoundingClientRect();
  const flip = h.sx > r.width - 220;
  tip.style.transform = flip ? 'translate(calc(-100% - 14px), -50%)' : 'translate(14px, -50%)';
  tip.style.left = h.sx + 'px'; tip.style.top = h.sy + 'px'; tip.style.opacity = 1;
  if (h.wristPick !== undefined) {
    HOVER.visible = false;
  } else if (!h.cohort) {
    HOVER.scale.set(h.x1 - h.x0, h.hy * 0.94, h.hz * 0.94);
    HOVER.position.set((h.x0 + h.x1) / 2, h.y, h.z);
    HOVER.visible = true;
  } else {
    const xs = data.cohort.mid[GRID.indexOf(h.z)][GRID.indexOf(h.y)];
    HOVER.scale.set(0.05, TILE, TILE); HOVER.position.set(xs, h.y, h.z); HOVER.visible = true;
  }
  if (!hinted) { hinted = true; $('hint').style.opacity = 0; }
}
let down = null;
renderer.domElement.addEventListener('pointerdown', (e) => { down = { x: e.clientX, y: e.clientY }; });
renderer.domElement.addEventListener('pointermove', (e) => {
  if (e.pointerType === 'mouse' && e.buttons === 0) {
    const h = pick(e.clientX, e.clientY); setHover(h);
    renderer.domElement.style.cursor = h && h.wristPick !== undefined ? 'pointer' : '';
  }
  else if (down && Math.hypot(e.clientX - down.x, e.clientY - down.y) > 4) setHover(null);
});
renderer.domElement.addEventListener('pointerleave', () => { if (!COARSE) setHover(null); });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!down) return;
  const still = Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5;
  down = null;
  if (!still) return;
  const h = pick(e.clientX, e.clientY);
  setHover(h);
  // a click on a diamond opens that wrist
  if (h && h.wristPick !== undefined) {
    current = h.wristPick; sliceIdx = centralSlice(); setHover(null); setMode('wrist'); return;
  }
  // a click on a block also moves the slice view to that block's slice
  if (h && !h.cohort) {
    const l = sliceList();
    let bi = sliceIdx, bd = Infinity;
    l.forEach((s, i) => { const d = Math.abs(s.z - h.z); if (d < bd) { bd = d; bi = i; } });
    setSlice(bi);
  }
});

// --------------------------------------------------------------- render ----
function size() {
  const w = stage.clientWidth, h = stage.clientHeight;
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setSize(w, h, false);
  renderer.domElement.style.width = w + 'px'; renderer.domElement.style.height = h + 'px';
  camera.aspect = w / Math.max(1, h); camera.updateProjectionMatrix();
}
function resize() {
  size();
  if (lastBox) frameOn(lastBox, viewName, false);
  drawSlice();
}
new ResizeObserver(() => requestAnimationFrame(resize)).observe(stage);

const NICE = [1, 2, 5, 10, 20];
let sbShown = null;
function updateScaleBar() {
  const d = camera.position.distanceTo(controls.target);
  const perPx = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2) * d / Math.max(1, stage.clientHeight);
  let mm = NICE[0];
  for (const c of NICE) if (c / perPx <= 120) mm = c;
  const px = Math.round(mm / perPx);
  if (mm === sbShown && $('sbLine').style.width === px + 'px') return;
  sbShown = mm;
  $('sbLine').style.width = px + 'px';
  $('sbText').textContent = mm + ' mm';
}

const v = new THREE.Vector3(), viewDir = new THREE.Vector3();
function placeTags() {
  const W = stage.clientWidth, H = stage.clientHeight;
  viewDir.copy(camera.position).sub(controls.target).normalize();
  if (lastBox) {
    const c = lastBox.getCenter(new THREE.Vector3()), s = lastBox.getSize(new THREE.Vector3());
    const pad = labelPad(lastBox);
    for (const t of DIRS) {
      const a = t.axis;
      t.at.set(c.x + a.x * (s.x / 2 + pad), c.y + a.y * (s.y / 2 + pad), c.z + a.z * (s.z / 2 + pad));
      const along = Math.abs(a.dot(viewDir));
      t.op = 1 - THREE.MathUtils.smoothstep(along, 0.55, 0.85);
      t.on = t.op > 0.02;
    }
  }
  for (const t of tags) {
    if (!t.on) { t.el.style.opacity = 0; continue; }
    v.copy(t.at).project(camera);
    const vis = v.z > -1 && v.z < 1;
    const x = (v.x * 0.5 + 0.5) * W, y = (-v.y * 0.5 + 0.5) * H;
    const half = t.el.offsetWidth / 2 + 4;
    const inside = x > half && x < W - half && y > 10 && y < H - 10;
    t.el.style.opacity = vis && inside ? (t.axis ? t.op : 1) : 0;
    t.el.style.left = x + 'px'; t.el.style.top = y + 'px';
    t.px = x; t.py = y;
  }
  // keep the two bone names from sitting on each other
  const A = boneTag.scaphoid, B = boneTag.lunate;
  if (A.on && B.on) {
    const wa = A.el.offsetWidth, wb = B.el.offsetWidth, hh = A.el.offsetHeight;
    if (Math.abs(A.px - B.px) < (wa + wb) / 2 + 6 && Math.abs(A.py - B.py) < hh + 2) {
      B.el.style.top = (A.py + hh + 4) + 'px';
    }
  }
}

function loop(now) {
  if (!running) return;
  stepFly(now);
  controls.update();
  updateScaleBar();
  key.position.copy(camera.position);
  renderer.render(scene, camera);
  placeTags();
  requestAnimationFrame(loop);
}

{
  const h = location.hash.slice(1).toUpperCase();
  if (h === 'MEAN') mode = 'cohort';
  const want = data.patients.findIndex((p) => p.id === h);
  if (want >= 0) current = want;
}
size(); apply('home');
$('loading').remove();
requestAnimationFrame(loop);

// ------------------------------------------------- progressive extras -----
// Slice outlines and bone surfaces are the bulk of the payload, so they load
// after the first frame rather than holding it up.
fetch('./slices.json' + V).then((r) => r.json()).then((j) => {
  slices = j; sliceIdx = centralSlice(); apply(false);
}).catch(() => { $('sliceSection').hidden = true; });

fetch('./meshes.json' + V).then((r) => r.json()).then(async (index) => {
  const buf = await (await fetch('./meshes.bin' + V)).arrayBuffer();
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
      g.computeVertexNormals(); g.computeBoundingBox();
      boneBoxes[i][bone] = g.boundingBox.clone();
      const mesh = new THREE.Mesh(g, MAT_BONE); mesh.renderOrder = 5;
      boneGroups[i].add(mesh);
    }
  });
  meshesReady = true;
  apply(false);
  if (mode === 'wrist' && viewName) frameOn(currentBox(), viewName, false);
}).catch(() => { $('lBones').disabled = true; });
