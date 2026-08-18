// Interactive 3D scapholunate distance maps.
//
// Every measurement is the radial-ulnar gap across one coronal image row:
// from the scaphoid's ulnar-most pixel to the lunate's radial-most pixel.
// It is drawn as a block filling that gap, one pixel tall and one slice deep,
// so the assembled solid is the measured joint interval itself, coloured by
// its own local width. Nothing here is interpolated or smoothed.
//
// Axes follow the manuscript: x radial(-)/ulnar(+), y proximal(-)/distal(+),
// z volar(-)/dorsal(+); millimetres, origin at the scaphoid's ulnar-most point
// on the slice of maximal scaphoid area.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- colour ---
// viridis, sampled at 11 stops and interpolated in sRGB.
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

// ------------------------------------------------------------------ data ---
const data = await (await fetch('./data.json')).json();
const GRID = data.meta.grid;
const D_LO = 1.0, D_HI = 7.0;               // retention bounds from the pipeline
const norm = (d) => (d - D_LO) / (D_HI - D_LO);

// ----------------------------------------------------------------- scene ---
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
$('stage').appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0c0e);

const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.rotateSpeed = 0.85;

scene.add(new THREE.HemisphereLight(0xdae2ee, 0x3a3f47, 1.35));
scene.add(new THREE.AmbientLight(0xffffff, 0.55));
const top = new THREE.DirectionalLight(0xffffff, 0.7); top.position.set(0.2, 1, 0.35);
scene.add(top);
const key = new THREE.DirectionalLight(0xffffff, 1.25); scene.add(key);
const fill = new THREE.DirectionalLight(0xffffff, 0.45); scene.add(fill);

// --------------------------------------------------------------- builders ---

// One block per measurement: scaphoid -> lunate in x, one pixel in y,
// one slice in z. Six faces, own normals, flat colour by width.
function gapSolid(p) {
  const n = p.n, hy = p.ps / 2, hz = p.st / 2;
  const pos = new Float32Array(n * 72), col = new Float32Array(n * 72);
  const nor = new Float32Array(n * 72), idx = new Uint32Array(n * 36);
  const F = [ // 4 corners as [useFarX, ySign, zSign], then the face normal
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
        pos[v * 3]     = s[0] ? x1 : x0;
        pos[v * 3 + 1] = y + s[1] * hy;
        pos[v * 3 + 2] = z + s[2] * hz;
        nor[v * 3] = f[4][0]; nor[v * 3 + 1] = f[4][1]; nor[v * 3 + 2] = f[4][2];
        col[v * 3] = c[0]; col[v * 3 + 1] = c[1]; col[v * 3 + 2] = c[2];
        v++;
      }
      idx[o++] = base; idx[o++] = base + 1; idx[o++] = base + 2;
      idx[o++] = base; idx[o++] = base + 2; idx[o++] = base + 3;
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

// Cohort mean sheet: a surface through the middle of the joint across the
// 7x7 (y,z) grid, coloured by the mean interval width there. Position is
// anatomy and colour is distance, exactly as in the per-patient view.
function meanField() {
  const f = data.cohort.field, mid = data.cohort.mid, m = GRID.length;
  const tri = [], cols = [];
  const at = (zi, yi) => [mid[zi][yi], GRID[yi], GRID[zi], f[zi][yi]];
  for (let zi = 0; zi < m - 1; zi++) for (let yi = 0; yi < m - 1; yi++) {
    const q = [at(zi, yi), at(zi, yi + 1), at(zi + 1, yi + 1), at(zi + 1, yi)];
    for (const t of [[0, 1, 2], [0, 2, 3]]) for (const k of t) {
      tri.push(q[k]); cols.push(viridis(norm(q[k][3])));
    }
  }
  const pos = new Float32Array(tri.length * 3), col = new Float32Array(tri.length * 3);
  tri.forEach((t, i) => { pos[i*3] = t[0]; pos[i*3+1] = t[1]; pos[i*3+2] = t[2];
                          col[i*3] = cols[i][0]; col[i*3+1] = cols[i][1]; col[i*3+2] = cols[i][2]; });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('color', new THREE.BufferAttribute(col, 3));
  g.computeVertexNormals();

  const seg = [];
  for (let zi = 0; zi < m; zi++) for (let yi = 0; yi < m; yi++) {
    if (yi < m - 1) seg.push(at(zi, yi), at(zi, yi + 1));
    if (zi < m - 1) seg.push(at(zi, yi), at(zi + 1, yi));
  }
  const sp = new Float32Array(seg.length * 3);
  seg.forEach((t, i) => { sp[i*3] = t[0]; sp[i*3+1] = t[1]; sp[i*3+2] = t[2]; });
  const gl = new THREE.BufferGeometry();
  gl.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  return { surface: g, lines: gl };
}

// Every patient's measurements inside the sampled window, as points.
function cohortCloud() {
  const pos = [], col = [];
  for (const p of data.patients) for (let k = 0; k < p.n; k++) {
    if (Math.abs(p.y[k]) > 3 || Math.abs(p.z[k]) > 3) continue;
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
  vertexColors: true, roughness: 0.6, metalness: 0.0, side: THREE.DoubleSide });

const solids = data.patients.map((p) => {
  const m = new THREE.Mesh(gapSolid(p), MAT_GAP);
  m.visible = false; scene.add(m); return m;
});

const mf = meanField();
const fieldGroup = new THREE.Group();
fieldGroup.add(new THREE.Mesh(mf.surface, new THREE.MeshStandardMaterial({
  vertexColors: true, roughness: 0.45, metalness: 0.0, side: THREE.DoubleSide,
  transparent: true, opacity: 0.9 })));
fieldGroup.add(new THREE.LineSegments(mf.lines, new THREE.LineBasicMaterial({
  color: 0x0b0c0e, transparent: true, opacity: 0.5 })));
fieldGroup.visible = false;
scene.add(fieldGroup);

const cloudGeom = cohortCloud();
const cloud = new THREE.Points(cloudGeom, new THREE.PointsMaterial({
  size: 0.22, vertexColors: true, transparent: true, opacity: 0.7,
  sizeAttenuation: true, depthWrite: false }));
cloud.visible = false;
scene.add(cloud);

// The manuscript samples a +/-3 mm window in (y,z) at whatever x the joint sits
// at, so this is a flat square on the mid-joint plane -- not a box. Extruding it
// along the gap axis would imply a sampled volume that does not exist.
const winGroup = new THREE.Group();
{
  const mid = data.cohort.mid.flat();
  const x = mid.reduce((a, b) => a + b, 0) / mid.length;
  const seg = [], c = [[-3,-3],[3,-3],[3,3],[-3,3]];
  for (let i = 0; i < 4; i++) {
    const a = c[i], b = c[(i + 1) % 4];
    seg.push([x,a[0],a[1]],[x,b[0],b[1]]);
  }
  const sp = new Float32Array(seg.length * 3);
  seg.forEach((t, i) => { sp[i*3] = t[0]; sp[i*3+1] = t[1]; sp[i*3+2] = t[2]; });
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(sp, 3));
  winGroup.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({
    color: 0x6c7079, transparent: true, opacity: 0.55 })));
  winGroup.visible = false;
  scene.add(winGroup);
}

// axis triad, coloured as in the manuscript's coordinate figure
const AXES = [
  { dir: [1, 0, 0], hex: 0xe03131, name: 'ulnar' },
  { dir: [0, 1, 0], hex: 0x2f9e44, name: 'distal' },
  { dir: [0, 0, 1], hex: 0x1c7ed6, name: 'dorsal' },
];
const GIZMO_PX = 116;
const gizmoScene = new THREE.Scene();
const gizmoCam = new THREE.OrthographicCamera(-1.5, 1.5, 1.5, -1.5, 0.01, 20);
gizmoScene.add(new THREE.HemisphereLight(0xffffff, 0x404040, 2.2));
for (const a of AXES) {
  gizmoScene.add(new THREE.ArrowHelper(
    new THREE.Vector3(...a.dir), new THREE.Vector3(), 1, a.hex, 0.3, 0.17));
}

function marker(hex) {
  const m = new THREE.Mesh(new THREE.OctahedronGeometry(1),
    new THREE.MeshBasicMaterial({ color: hex, depthTest: false, depthWrite: false }));
  m.renderOrder = 10; scene.add(m); return m;
}
const MARK_PEAK = marker(0xffd166), MARK_MATCH = marker(0x7fd7ff);

// ---------------------------------------------------------------- labels ---
const tags = [];
function tag(cls, html) {
  const el = document.createElement('div');
  el.className = 'tag ' + cls; el.innerHTML = html;
  document.body.appendChild(el);
  const t = { el, at: new THREE.Vector3(), on: false };
  tags.push(t); return t;
}
const axisTags = AXES.map((a) => {
  const t = tag('axis', a.name);
  t.el.style.color = '#' + a.hex.toString(16).padStart(6, '0');
  t.at.set(a.dir[0] * 1.32, a.dir[1] * 1.32, a.dir[2] * 1.32);
  t.gizmo = t.on = true; return t;
});
const peakTag = tag('peak', `peak agreement &nbsp;<b>r = ${data.meta.peak.r.toFixed(2)}</b>`);
const matchTag = tag('match', '');

// nearest measurement to a (y,z) node, at the middle of its gap
function nearest(p, y0, z0) {
  let bi = 0, bd = Infinity;
  for (let k = 0; k < p.n; k++) {
    const d = (p.y[k] - y0) ** 2 + (p.z[k] - z0) ** 2;
    if (d < bd) { bd = d; bi = k; }
  }
  return { x: (p.sx[bi] + p.lx[bi]) / 2, y: p.y[bi], z: p.z[bi], off: Math.sqrt(bd) };
}

// ------------------------------------------------------------------ view ---
const HOME = new THREE.Vector3(0.55, 0.34, 1).normalize();
let scale = 1;

function frameOn(geom, keep) {
  const box = geom.boundingBox.clone();
  const c = box.getCenter(new THREE.Vector3());
  const r = Math.max(box.getSize(new THREE.Vector3()).length() / 2, 2);
  const dist = (r / Math.sin(THREE.MathUtils.degToRad(camera.fov) / 2)) * 1.05;
  const dir = keep && camera.position.distanceTo(controls.target) > 1e-3
    ? camera.position.clone().sub(controls.target).normalize()
    : HOME.clone();
  controls.target.copy(c);
  camera.position.copy(c).addScaledVector(dir, dist);
  camera.up.set(0, 1, 0);
  camera.near = dist / 500; camera.far = dist * 12;
  camera.updateProjectionMatrix();
  controls.update();

  // markers stay legible whatever the wrist's extent
  scale = Math.min(Math.max(r / 9, 0.3), 1.7);
  MARK_PEAK.scale.setScalar(scale * 0.45);
  MARK_MATCH.scale.setScalar(scale * 0.45);
}

// ------------------------------------------------------------------- ui ----
let mode = 'patient', current = 0;

const pick = $('pick');
data.patients.forEach((p, i) => {
  const o = document.createElement('option');
  o.value = i; o.textContent = `${p.id} · ${p.lat}`;
  pick.appendChild(o);
});

function apply(refit) {
  const patient = mode === 'patient';
  const p = data.patients[current];

  solids.forEach((m, i) => m.visible = patient && i === current);
  cloud.visible = !patient;
  fieldGroup.visible = !patient || $('lSurface').checked;
  winGroup.visible = $('lWindow').checked;

  $('lSurface').disabled = !patient;
  pick.disabled = !patient;
  $('mPatient').setAttribute('aria-pressed', patient);
  $('mCohort').setAttribute('aria-pressed', !patient);

  if (refit) frameOn(patient ? solids[current].geometry : cloudGeom, refit === 'keep');

  if (patient) {
    // only claim the peak coordinate when this wrist actually has data near it
    const pk = nearest(p, data.meta.peak.y, data.meta.peak.z);
    const near = pk.off <= 2.5;
    MARK_PEAK.visible = peakTag.on = near;
    if (near) { MARK_PEAK.position.set(pk.x, pk.y, pk.z);
                peakTag.at.set(pk.x, pk.y + scale * 1.3, pk.z); }

    const bm = nearest(p, p.best.y, p.best.z);
    MARK_MATCH.visible = matchTag.on = true;
    MARK_MATCH.position.set(bm.x, p.best.y, p.best.z);
    matchTag.at.set(bm.x, p.best.y - scale * 1.3, p.best.z);
    matchTag.el.innerHTML = `radiograph &nbsp;<b>${p.manual.toFixed(2)} mm</b>&nbsp; lands here`;

    $('subtitle').textContent =
      `${p.id} · ${p.lat} wrist · ${p.n} measurements · radiograph ${p.manual.toFixed(2)} mm`;
  } else {
    const yi = GRID.indexOf(data.meta.peak.y), zi = GRID.indexOf(data.meta.peak.z);
    const x = data.cohort.mid[zi][yi];
    MARK_PEAK.visible = peakTag.on = true;
    MARK_PEAK.position.set(x, data.meta.peak.y, data.meta.peak.z);
    peakTag.at.set(x, data.meta.peak.y + scale * 1.3, data.meta.peak.z);
    MARK_MATCH.visible = false; matchTag.on = false;
    $('subtitle').textContent =
      `all ${data.meta.n} wrists · measurements within the sampled window`;
  }
}

$('mPatient').onclick = () => { if (mode !== 'patient') { mode = 'patient'; apply('keep'); } };
$('mCohort').onclick  = () => { if (mode !== 'cohort')  { mode = 'cohort';  apply('keep'); } };
pick.onchange = (e) => { current = +e.target.value; apply('keep'); };
for (const id of ['lSurface', 'lWindow']) $(id).onchange = () => apply(false);
$('reset').onclick = () => apply('home');

// colour ramp
{
  const stops = [];
  for (let i = 0; i <= 10; i++) {
    const c = viridis(i / 10);
    stops.push(`rgb(${c.map((v) => Math.round(v * 255)).join(',')}) ${i * 10}%`);
  }
  $('ramp').style.background = `linear-gradient(90deg, ${stops.join(',')})`;
  $('lo').textContent = D_LO.toFixed(1);
  $('hi').textContent = D_HI.toFixed(1) + ' mm';
}

function resize() {
  camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
}
addEventListener('resize', resize);

const v = new THREE.Vector3();
function loop() {
  controls.update();
  key.position.copy(camera.position);
  fill.position.copy(camera.position).negate();
  renderer.setViewport(0, 0, innerWidth, innerHeight);
  renderer.render(scene, camera);

  const gx = innerWidth - GIZMO_PX - 26, gy = 122;
  gizmoCam.position.copy(camera.position).sub(controls.target).setLength(6);
  gizmoCam.up.copy(camera.up); gizmoCam.lookAt(0, 0, 0);
  renderer.autoClear = false;
  renderer.clearDepth();
  renderer.setViewport(gx, gy, GIZMO_PX, GIZMO_PX);
  renderer.render(gizmoScene, gizmoCam);
  renderer.autoClear = true;

  for (const t of tags) {
    if (!t.on) { t.el.style.opacity = 0; continue; }
    v.copy(t.at).project(t.gizmo ? gizmoCam : camera);
    const vis = v.z < 1;
    t.el.style.opacity = vis ? 1 : 0;
    if (!vis) continue;
    if (t.gizmo) {
      t.el.style.left = (gx + (v.x * 0.5 + 0.5) * GIZMO_PX) + 'px';
      t.el.style.top = (innerHeight - gy - (v.y * 0.5 + 0.5) * GIZMO_PX) + 'px';
    } else {
      t.el.style.left = ((v.x * 0.5 + 0.5) * innerWidth) + 'px';
      t.el.style.top = ((-v.y * 0.5 + 0.5) * innerHeight) + 'px';
    }
  }
  requestAnimationFrame(loop);
}

resize(); apply('home');
$('loading').remove();
loop();
