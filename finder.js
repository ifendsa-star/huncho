/* Huncho Haircut Finder
   - Face detection + measurements: MediaPipe Face Landmarker, runs entirely in the browser (files in ./mp)
   - Face analysis + ranking: Claude (artifact "sample" capability) when available; geometry-based estimate otherwise
   - 3D: the customer's own face mesh, textured with their front photo, rendered with WebGL
   - Photo try-on: needs an image-editing service behind a server (see CONFIG.tryOnEndpoint). Not faked.
*/
import {buildHeadModel, createHeadViewer} from './head3d.js';
const CONFIG = {
  tryOnEndpoint: '',            // e.g. 'https://huncho-api.vercel.app/api/tryon' once a backend exists
  shopPhone: '0431474740',
  shopPhonePretty: '0431 474 740'
};

const STYLES = [
  {id:'low-fade', name:'Low fade', maint:2, diff:1, blurb:'Fade starts just above the ear and neckline. Clean but subtle.'},
  {id:'mid-fade', name:'Mid fade', maint:2, diff:1, blurb:'Blends from the temple down. The all-rounder.'},
  {id:'high-fade', name:'High fade', maint:3, diff:1, blurb:'Sides taken short up high for a sharp, bold shape.'},
  {id:'skin-fade', name:'Skin fade', maint:3, diff:2, blurb:'Down to the skin and blended up seamlessly.'},
  {id:'taper-fade', name:'Taper fade', maint:1, diff:1, blurb:'Tidied at the sideburns and neckline, length kept elsewhere.'},
  {id:'buzz-cut', name:'Buzz cut', maint:2, diff:1, blurb:'One short length all over with a sharp hairline.'},
  {id:'textured-crop', name:'Textured crop', maint:2, diff:2, blurb:'Choppy, textured top worn forward with a short fringe.'},
  {id:'french-crop', name:'French crop', maint:2, diff:1, blurb:'Short top pushed forward to a straight, blunt fringe.'},
  {id:'fringe', name:'Fringe', maint:2, diff:2, blurb:'Longer, soft fringe falling over the forehead.'},
  {id:'slick-back', name:'Slick back', maint:3, diff:3, blurb:'Longer top combed straight back with faded sides.'},
  {id:'side-part', name:'Side part', maint:2, diff:2, blurb:'Classic part with the top combed across.'},
  {id:'crew-cut', name:'Crew cut', maint:1, diff:1, blurb:'Short and classic, a touch longer at the front.'},
  {id:'modern-mullet', name:'Modern mullet', maint:3, diff:3, blurb:'Textured top, faded sides, length left at the back.'}
];
const SHAPES = ['oval','round','square','rectangle','heart','diamond'];
// Barbering rules of thumb used when Claude is not available (and given to Claude as context).
const FIT = {
  oval:      {'textured-crop':92,'side-part':90,'mid-fade':88,'slick-back':86,'modern-mullet':84,'low-fade':84,'taper-fade':84,'french-crop':82,'skin-fade':82,'crew-cut':80,'fringe':80,'high-fade':78,'buzz-cut':74},
  round:     {'high-fade':90,'skin-fade':88,'textured-crop':86,'slick-back':84,'side-part':82,'mid-fade':80,'modern-mullet':72,'crew-cut':70,'taper-fade':66,'low-fade':64,'french-crop':58,'fringe':52,'buzz-cut':50},
  square:    {'crew-cut':90,'buzz-cut':88,'textured-crop':86,'side-part':84,'taper-fade':82,'low-fade':82,'mid-fade':80,'french-crop':78,'slick-back':76,'skin-fade':76,'fringe':70,'high-fade':70,'modern-mullet':70},
  rectangle: {'fringe':90,'french-crop':88,'crew-cut':84,'side-part':82,'taper-fade':82,'low-fade':80,'textured-crop':78,'mid-fade':72,'buzz-cut':66,'modern-mullet':66,'skin-fade':62,'high-fade':56,'slick-back':54},
  heart:     {'fringe':90,'textured-crop':88,'side-part':84,'taper-fade':84,'low-fade':82,'french-crop':80,'modern-mullet':78,'crew-cut':76,'mid-fade':74,'skin-fade':64,'buzz-cut':60,'high-fade':58,'slick-back':58},
  diamond:   {'fringe':90,'textured-crop':88,'french-crop':86,'side-part':82,'taper-fade':82,'modern-mullet':80,'low-fade':78,'crew-cut':74,'mid-fade':72,'slick-back':66,'skin-fade':60,'buzz-cut':58,'high-fade':56}
};
const WHY = {
  oval: 'Balanced proportions suit most shapes, so this comes down to taste and hair type.',
  round: 'Height on top and shorter sides add length and definition to a softer outline.',
  square: 'Works with a strong jaw, either keeping it sharp or softening the corners with texture.',
  rectangle: 'Keeps height down and adds width or a fringe so the face doesn\'t look longer.',
  heart: 'Adds weight lower down and softens a broader forehead.',
  diamond: 'Fills out the forehead and keeps some weight at the sides to balance the cheekbones.'
};

const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const S = {step: 0, angle: 'front', photos: {front: null, left: null, right: null}, analysis: null, ranked: null, pick: null, consent: false, busy: false};
let FL = null, FLmod = null, flPromise = null, FS = null, BUF = null, FLV = null, flvPromise = null;

/* ---------------- MediaPipe ---------------- */
function loadLandmarker() {
  if (flPromise) return flPromise;
  flPromise = (async () => {
    FLmod = await import('./mp/vision_bundle.mjs');
    const base = new URL('mp/', location.href).href;
    FS = {wasmLoaderPath: base + 'vision_wasm_internal.js', wasmBinaryPath: base + 'vision_wasm_internal.wasm'};
    BUF = new Uint8Array(await (await fetch(base + 'face_landmarker.wasm')).arrayBuffer());
    FL = await FLmod.FaceLandmarker.createFromOptions(FS, {baseOptions: {modelAssetBuffer: BUF.slice(), delegate: 'CPU'}, runningMode: 'IMAGE', numFaces: 1, outputFacialTransformationMatrixes: true});
    return FL;
  })();
  flPromise.catch(() => { flPromise = null; });
  return flPromise;
}

function loadVideoLandmarker() {
  if (flvPromise) return flvPromise;
  flvPromise = (async () => {
    await loadLandmarker();
    FLV = await FLmod.FaceLandmarker.createFromOptions(FS, {baseOptions: {modelAssetBuffer: BUF.slice(), delegate: 'CPU'}, runningMode: 'VIDEO', numFaces: 1});
    return FLV;
  })();
  flvPromise.catch(() => { flvPromise = null; });
  return flvPromise;
}

/* ---------------- photos ---------------- */
async function fileToPhoto(file) {
  const url0 = URL.createObjectURL(file);
  const img0 = new Image(); img0.src = url0; await img0.decode();
  const max = 1280, k = Math.min(1, max / Math.max(img0.naturalWidth, img0.naturalHeight));
  const c = document.createElement('canvas'); c.width = Math.round(img0.naturalWidth * k); c.height = Math.round(img0.naturalHeight * k);
  c.getContext('2d').drawImage(img0, 0, 0, c.width, c.height);
  URL.revokeObjectURL(url0);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.9));
  const url = URL.createObjectURL(blob);
  const img = new Image(); img.src = url; await img.decode();
  return {blob, url, img, w: c.width, h: c.height, lm: null, yaw: null};
}
function angleProblem(a, yaw) {
  const off = yaw - 0.5;
  if (a === 'front') return Math.abs(off) > 0.17 ? 'Look straight at the camera.' : '';
  if (Math.abs(off) < 0.09) return 'Turn your head a bit more to the side.';
  const other = S.photos[a === 'left' ? 'right' : 'left'];
  if (other && Math.sign(other.yaw - 0.5) === Math.sign(off)) return 'Turn the other way for this one.';
  return '';
}
function yawOf(lm) { const a = lm[234].x, b = lm[454].x; return (lm[1].x - a) / (b - a); } // ~0.5 facing camera

/* ---------------- geometry ---------------- */
function measure(p) {
  const L = p.lm, P = i => ({x: L[i].x * p.w, y: L[i].y * p.h});
  const d = (a, b) => Math.hypot(P(a).x - P(b).x, P(a).y - P(b).y);
  const m = {forehead: d(54, 284), cheek: d(234, 454), jaw: d(172, 397), chinW: d(149, 378), length: d(10, 152)};
  m.lenToCheek = m.length / m.cheek; m.foreToCheek = m.forehead / m.cheek; m.jawToCheek = m.jaw / m.cheek; m.foreToJaw = m.forehead / m.jaw;
  return m;
}
function shapeFromGeometry(m) {
  let s;
  if (m.foreToJaw > 1.12 && m.jawToCheek < 0.86) s = 'heart';
  else if (m.foreToCheek < 0.84 && m.jawToCheek < 0.86) s = 'diamond';
  else if (m.lenToCheek > 1.42) s = m.jawToCheek > 0.88 ? 'rectangle' : 'oval';
  else if (m.lenToCheek < 1.26) s = m.jawToCheek > 0.9 ? 'square' : 'round';
  else s = m.jawToCheek > 0.92 ? 'square' : 'oval';
  return s;
}
function describe(m) {
  const f = m.foreToCheek > 0.95 ? 'Broad, close to your cheekbone width.' : m.foreToCheek > 0.86 ? 'In proportion with the rest of your face.' : 'Narrower than your cheekbones.';
  const j = m.jawToCheek > 0.92 ? 'Wide and defined, with strong angles.' : m.jawToCheek > 0.84 ? 'Medium width with a balanced taper to the chin.' : 'Narrows towards a softer, more pointed chin.';
  const c = (m.cheek > m.forehead * 1.08 && m.cheek > m.jaw * 1.12) ? 'The widest point of your face, and prominent.' : 'Fairly even with your forehead and jaw.';
  const p = m.lenToCheek > 1.42 ? 'Longer than it is wide.' : m.lenToCheek < 1.26 ? 'Close to as wide as it is long.' : 'Balanced length to width.';
  return {forehead: f, jawline: j, cheekbones: c, proportions: p};
}

/* ---------------- UI shell ---------------- */
const hf = $('#hf');
const STEPS = ['Before we start', 'Face scan', 'Analysis', 'Your cuts', '3D & try-on', 'Send to barber'];
function go(n) {
  S.step = n;
  $$('.hf-pane').forEach((p, i) => p.classList.toggle('on', i === n));
  $$('.hf-steps li').forEach((li, i) => { li.classList.toggle('done', i < n); li.classList.toggle('now', i === n); });
  $('.hf-body').scrollTop = 0;
  if (n === 1) renderScan(); else stopCamera();
  if (n === 4) start3D(); else stop3D();
  if (n === 5) renderTicket();
}
function openFinder() {
  hf.classList.add('open'); document.body.style.overflow = 'hidden';
  if (location.hash !== '#finder') history.pushState(null, '', '#finder');
  go(S.consent ? 1 : 0);
  if (S.consent) loadLandmarker().catch(() => {});
}
function closeFinder() {
  hf.classList.remove('open'); document.body.style.overflow = ''; stop3D(); stopCamera();
  if (location.hash === '#finder') history.pushState(null, '', location.pathname + location.search);
}
function toast(t) { const el = $('#hfToast'); el.textContent = t; el.classList.add('on'); clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove('on'), 2400); }

$$('[data-open-finder]').forEach(b => b.addEventListener('click', e => { e.preventDefault(); openFinder(); }));
$('#hfClose').addEventListener('click', closeFinder);
window.addEventListener('hashchange', () => { if (location.hash === '#finder') { if (!hf.classList.contains('open')) openFinder(); } else if (hf.classList.contains('open')) closeFinder(); });
document.addEventListener('keydown', e => { if (e.key === 'Escape' && hf.classList.contains('open')) closeFinder(); });
if (location.hash === '#finder') setTimeout(openFinder, 400);

/* step 0: consent */
$('#hfAgree').addEventListener('change', e => { $('#hfStart').disabled = !e.target.checked; });
$('#hfStart').addEventListener('click', () => { S.consent = true; loadLandmarker().catch(() => {}); go(1); });

/* delete */
function deleteAll() {
  stopCamera();
  for (const k of ['front', 'left', 'right']) { if (S.photos[k]) URL.revokeObjectURL(S.photos[k].url); S.photos[k] = null; }
  S.analysis = null; S.ranked = null; S.pick = null; S.angle = 'front'; S.scanFrames = []; stopSweep(); tearDown3D();
  $('#hfNotes').value = '';
  if (S.step > 0) go(1);
  toast('Your photos have been deleted');
}
$('#hfDelete').addEventListener('click', deleteAll);

/* ---------------- step 1: scan ---------------- */
const ANGLES = {
  front: {title: 'Look straight ahead', tips: ['Face the camera with your hair off your forehead', 'Even light on your face, no hat or glasses', 'Keep a neutral expression']},
  left: {title: 'Turn to one side', tips: ['Turn your head about a quarter to the side', 'Keep your ear and jawline in view', 'Optional, but it helps your barber']},
  sweep: {title: 'Now roll your head in a slow circle', tips: ['Keep your face inside the ring', 'Move slowly, like setting up Face ID', 'Every tick turns green when that angle is captured']},
  right: {title: 'Now the other side', tips: ['Turn the other way, about a quarter turn', 'Keep the same light and distance', 'Optional, but it helps your barber']}
};
const fileCam = $('#hfCam'), fileUp = $('#hfUpload');
$('#hfTake').addEventListener('click', () => fileCam.click());
$('#hfUp').addEventListener('click', () => fileUp.click());
fileCam.addEventListener('change', e => handleFile(e.target.files[0], e.target));
fileUp.addEventListener('change', e => handleFile(e.target.files[0], e.target));
$$('.shot .th').forEach(b => b.addEventListener('click', () => { S.angle = b.closest('.shot').dataset.a; renderScan(); }));
$$('.shot .rm').forEach(b => b.addEventListener('click', () => { const a = b.closest('.shot').dataset.a; if (S.photos[a]) URL.revokeObjectURL(S.photos[a].url); S.photos[a] = null; S.angle = a; S.analysis = null; renderScan(); }));
$('#hfSkipSides').addEventListener('click', () => { go(2); runAnalysis(); });
$('#hfToAnalysis').addEventListener('click', () => { go(2); runAnalysis(); });

function setBadge(t, cls) { const b = $('#hfBadge'); b.textContent = t; b.className = 'badge' + (cls ? ' ' + cls : ''); b.hidden = !t; }

async function handleFile(file, input) {
  input.value = '';
  if (!file) return;
  await processPhoto(file);
}
async function processPhoto(file) {
  const a = S.angle, scope = $('#hfScope');
  scope.classList.add('busy'); setBadge('Scanning…');
  try {
    const p = await fileToPhoto(file);
    drawLens(p, 0);
    await loadLandmarker();
    const r = FL.detect(p.img);
    const lm = r.faceLandmarks && r.faceLandmarks[0];
    const fail = t => { scope.classList.remove('busy'); URL.revokeObjectURL(p.url); setBadge(t, 'err'); drawEmptyLens(); return false; };
    if (!lm) return fail('No face found. Try brighter light and face the camera.');
    p.lm = lm; p.yaw = yawOf(lm); p.mat = r.facialTransformationMatrixes && r.facialTransformationMatrixes[0] || null;
    const prob = angleProblem(a, p.yaw);
    if (prob) return fail(prob);
    if (S.photos[a]) URL.revokeObjectURL(S.photos[a].url);
    S.photos[a] = p; S.analysis = null; tearDown3D();
    await animateMesh(p);
    scope.classList.remove('busy');
    setBadge('Face captured', 'ok');
    const next = ['front', 'left', 'right'].find(k => !S.photos[k]);
    if (CAM.on && a === 'front') { startSweep(p); setTimeout(() => { S.angle = 'sweep'; renderScan(true); CAM.pauseUntil = performance.now() + 700; }, 700); return true; }
    if (!next) stopCamera();
    setTimeout(() => { if (next) S.angle = next; renderScan(true); if (CAM.on) CAM.pauseUntil = performance.now() + 900; }, 900);
    return true;
  } catch (err) {
    console.error(err); scope.classList.remove('busy');
    setBadge('Face scanner couldn\'t start in this browser. Try Chrome or Safari.', 'err');
    return false;
  }
}

/* ---------------- live camera ---------------- */
function cameraAllowedHere() {
  try { if (document.featurePolicy && !document.featurePolicy.allowsFeature('camera')) return false; } catch (e) {}
  try { if (document.permissionsPolicy && !document.permissionsPolicy.allowsFeature('camera')) return false; } catch (e) {}
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}
const CAM_OK = cameraAllowedHere();
const CAM = {on: false, stream: null, raf: 0, hold: 0, last: 0, lastDetect: 0, lm: null, msg: '', capturing: false, pauseUntil: 0};
const HOLD_MS = 900;
function camHelpHTML(reason) {
  const ua = navigator.userAgent, ios = /iPhone|iPad|iPod/.test(ua) || (/Macintosh/.test(ua) && navigator.maxTouchPoints > 1),
    android = /Android/.test(ua), mac = /Macintosh/.test(ua) && !ios, inApp = /Instagram|FBAN|FBAV|FB_IAB|Line\/|TikTok|Snapchat/i.test(ua),
    chrome = /Chrome|CriOS/.test(ua) && !/Edg/.test(ua), safari = /Safari/.test(ua) && !/Chrome|CriOS|FxiOS|Edg/.test(ua), firefox = /Firefox|FxiOS/.test(ua);
  const steps = [];
  if (inApp) steps.push('You\'re inside an app\'s built-in browser, which blocks the camera. Tap the <b>⋯</b> menu and choose <b>Open in browser</b> (Safari or Chrome), then try again.');
  else if (reason === 'busy') steps.push('Another app (FaceTime, Zoom, Teams, Photo Booth) is using the camera. Close it, then press <b>Turn on camera</b> again.');
  else if (reason === 'none') steps.push('No camera was found on this device. Use <b>Upload a photo</b> instead.');
  else {
    if (ios && safari) steps.push('On iPhone: open <b>Settings → Apps → Safari → Camera</b> and choose <b>Allow</b> (or <b>Ask</b>). Then reload this page.');
    else if (ios) steps.push('On iPhone: open <b>Settings</b>, find your browser app, and switch <b>Camera</b> on. Then reload this page.');
    else if (android) steps.push('Tap the <b>lock icon</b> next to the web address, then <b>Permissions → Camera → Allow</b>. Then reload this page.');
    else {
      if (chrome) steps.push('Click the <b>camera or lock icon</b> at the left of the address bar, set <b>Camera</b> to <b>Allow</b>, then reload the page.');
      else if (safari) steps.push('In the menu bar choose <b>Safari → Settings for this website</b> and set <b>Camera</b> to <b>Allow</b>, then reload the page.');
      else if (firefox) steps.push('Click the <b>camera icon</b> in the address bar, remove the block, then reload and choose <b>Allow</b>.');
      else steps.push('Allow camera access for this site in your browser\'s site settings, then reload the page.');
      if (mac) steps.push('Still blocked? On your Mac open <b>System Settings → Privacy &amp; Security → Camera</b> and switch your browser on.');
    }
  }
  return `<b class="hh">Turn the camera on</b><ol>${steps.map(x => `<li>${x}</li>`).join('')}</ol><p>Or use <b>Upload a photo</b> instead: front, then each side.</p>`;
}
function showCamHelp(reason) { const h = $('#hfCamHelp'); if (!h) return; h.innerHTML = camHelpHTML(reason); h.hidden = false; }
async function startCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) { setBadge('This browser can\'t open the camera.', 'err'); showCamHelp('blocked'); return; }
  $('#hfCamHelp').hidden = true;
  setBadge('Asking for camera access…');
  $('#hfCamBtn').disabled = true;
  try {
    try { CAM.stream = await navigator.mediaDevices.getUserMedia({video: {facingMode: 'user', width: {ideal: 1280}, height: {ideal: 960}}, audio: false}); }
    catch (e1) { if (e1 && (e1.name === 'OverconstrainedError' || e1.name === 'NotReadableError' || e1.name === 'AbortError')) CAM.stream = await navigator.mediaDevices.getUserMedia({video: true, audio: false}); else throw e1; }
  } catch (e) {
    $('#hfCamBtn').disabled = false; CAM.stream = null;
    const n = e && e.name;
    if (!CAM_OK) { $('#hfLensPrompt').hidden = true; $('#hfNoCam').hidden = false; setBadge('Camera isn\'t available on this page.', 'err'); return; }
    const reason = n === 'NotReadableError' || n === 'AbortError' ? 'busy' : n === 'NotFoundError' || n === 'OverconstrainedError' ? 'none' : 'blocked';
    setBadge(reason === 'busy' ? 'Your camera is busy in another app.' : reason === 'none' ? 'No camera found.' : 'Camera access was blocked.', 'err');
    showCamHelp(reason);
    return;
  }
  const v = $('#hfVideo'); v.srcObject = CAM.stream;
  try { await v.play(); } catch (e) {}
  CAM.on = true; CAM.hold = 0; CAM.lm = null; CAM.pauseUntil = 0; $('#hfCamBtn').disabled = false;
  renderScan(true); setBadge('Starting face scanner…');
  try { await loadVideoLandmarker(); } catch (e) { setBadge('Face scanner couldn\'t start in this browser. Try Chrome or Safari.', 'err'); stopCamera(); return; }
  if (!CAM.on) return;
  CAM.last = performance.now(); camLoop();
}
function stopCamera() {
  if (SW.on) { stopSweep(); if (S.angle === 'sweep') S.angle = S.photos.left && S.photos.right ? 'front' : (S.photos.left ? 'right' : 'left'); }
  cancelAnimationFrame(CAM.raf); CAM.raf = 0;
  if (CAM.stream) CAM.stream.getTracks().forEach(t => t.stop());
  CAM.stream = null; CAM.on = false; CAM.hold = 0; CAM.lm = null;
  const v = $('#hfVideo'); if (v) v.srcObject = null;
  setArcProgress(null);
  if ($('#hfCamBtn')) updateCamButtons();
}
function setArcProgress(k) {
  const i = ['front', 'left', 'right'].indexOf(S.angle), arc = $('#arc' + i);
  $$('.scope .arc').forEach(x => { if (!x.classList.contains('done')) x.setAttribute('stroke-dasharray', '94 206'); x.classList.remove('prog'); });
  if (k == null || k < 0.02 || !arc || S.angle === 'sweep' || S.photos[S.angle]) return;
  arc.classList.add('prog'); arc.setAttribute('stroke-dasharray', `${Math.max(0.5, 94 * k)} ${300 - Math.max(0.5, 94 * k)}`);
}
function camLoop() {
  const v = $('#hfVideo'), cv = lensCanvas(), g = cv.getContext('2d');
  const tick = t => {
    if (!CAM.on) return;
    const dt = t - CAM.last; CAM.last = t;
    const vw = v.videoWidth, vh = v.videoHeight;
    if (vw && vh) {
      // mirrored, cover-cropped preview
      const k = Math.max(cv.width / vw, cv.height / vh), dw = vw * k, dh = vh * k, ox = (cv.width - dw) / 2, oy = (cv.height - dh) / 2;
      g.save(); g.clearRect(0, 0, cv.width, cv.height); g.translate(cv.width, 0); g.scale(-1, 1); g.drawImage(v, cv.width - ox - dw, oy, dw, dh); g.restore();
      $('#hfGhost').hidden = true;
      if (t - CAM.lastDetect > 60 && FLV && !CAM.capturing) {
        CAM.lastDetect = t;
        try { const r = FLV.detectForVideo(v, t); CAM.lm = r.faceLandmarks && r.faceLandmarks[0] || null; } catch (e) { CAM.lm = null; }
      }
      const L = CAM.lm, a = S.angle;
      let msg = '', ok = false;
      if (t < CAM.pauseUntil) msg = a === 'front' ? 'Look straight ahead' : 'Now turn your head';
      else if (!L) msg = 'Move your face into the circle';
      else {
        let x0 = 1, x1 = 0; for (const q of L) { x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); }
        const prob = a === 'sweep' ? '' : angleProblem(a, yawOf(L));
        if ((x1 - x0) < 0.14) msg = 'Come a little closer';
        else if (a === 'sweep' && SW.on) { msg = sweepStep(L, t); ok = false; }
        else if (prob) msg = prob.replace(/\.$/, '');
        else { msg = 'Hold still'; ok = true; }
        g.fillStyle = ok || (a === 'sweep') ? 'rgba(228,80,42,.9)' : 'rgba(255,255,255,.55)';
        const rr = Math.max(1.1, cv.width / 300);
        for (let i = 0; i < L.length; i += ok || a === 'sweep' ? 1 : 2) { const q = L[i]; g.beginPath(); g.arc(ox + (1 - q.x) * dw, oy + q.y * dh, rr, 0, 7); g.fill(); }
      }
      CAM.hold = ok ? CAM.hold + dt : Math.max(0, CAM.hold - dt * 2);
      setArcProgress(Math.min(1, CAM.hold / HOLD_MS));
      if (msg !== CAM.msg) { CAM.msg = msg; setBadge(msg, ok ? 'ok' : ''); }
      if (CAM.hold >= HOLD_MS && !CAM.capturing) { capture(); }
    }
    CAM.raf = requestAnimationFrame(tick);
  };
  CAM.raf = requestAnimationFrame(tick);
}
async function capture() {
  const v = $('#hfVideo'); if (!v.videoWidth) return;
  CAM.capturing = true; CAM.hold = 0; setArcProgress(null);
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  const sc = $('#hfScope'); sc.classList.add('flash'); setTimeout(() => sc.classList.remove('flash'), 350);
  const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
  const wasOn = CAM.on; cancelAnimationFrame(CAM.raf); CAM.raf = 0;
  const ok = await processPhoto(blob);
  CAM.capturing = false; CAM.msg = '';
  if (wasOn && CAM.on) { CAM.pauseUntil = performance.now() + (ok ? 1800 : 600); CAM.last = performance.now(); camLoop(); }
}

/* ---------- Face ID-style sweep ---------- */
const TICKS = 24;
const SW = {on: false, base: null, lit: new Array(TICKS).fill(0), sectors: {}, busy: false, last: null};
function pitchOf(lm) { const e = (lm[33].y + lm[263].y) / 2; return (lm[1].y - e) / (lm[152].y - e); }
function buildTicks() {
  const g = $('#hfTicks'); if (!g || g.childElementCount) return;
  let h = '';
  for (let i = 0; i < TICKS; i++) { const a = i / TICKS * Math.PI * 2; const x1 = 50 + Math.cos(a) * 41.5, y1 = 50 + Math.sin(a) * 41.5, x2 = 50 + Math.cos(a) * 45, y2 = 50 + Math.sin(a) * 45; h += `<line class="tick" data-i="${i}" x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"/>`; }
  g.innerHTML = h;
}
function startSweep(front) {
  SW.on = true; SW.base = {yaw: front.yaw, pitch: pitchOf(front.lm)}; SW.lit.fill(0); SW.sectors = {}; SW.busy = false;
  S.scanFrames = [];
  $('#hfScope').classList.add('sweeping'); buildTicks(); $$('#hfTicks .tick').forEach(t => t.classList.remove('on'));
  updateSweepCount();
}
function stopSweep() { SW.on = false; const sc = $('#hfScope'); if (sc) sc.classList.remove('sweeping'); }
function updateSweepCount() {
  const n = Object.keys(SW.sectors).length + (S.photos.front ? 1 : 0), lit = SW.lit.filter(Boolean).length;
  const el = $('#hfSweepCount'); if (el) { el.hidden = !SW.on && !n; el.textContent = `${n} of 9 angles captured`; }
  $('#hfFinishScan').hidden = !(SW.on && S.photos.left && S.photos.right);
  return {n, lit};
}
function sweepStep(L, t) {
  const yo = yawOf(L) - SW.base.yaw, po = pitchOf(L) - SW.base.pitch;
  const vx = -yo / 0.13, vy = po / 0.085, m = Math.hypot(vx, vy);
  if (m < 0.55) return 'Roll your head slowly in a circle';
  const ang = Math.atan2(vy, vx);
  for (let i = 0; i < TICKS; i++) {
    const a = i / TICKS * Math.PI * 2; let d = Math.abs(((ang - a + Math.PI * 3) % (Math.PI * 2)) - Math.PI);
    if (d < 0.36 && !SW.lit[i] && m > 0.75) { SW.lit[i] = 1; const el = $(`#hfTicks .tick[data-i="${i}"]`); if (el) el.classList.add('on'); }
  }
  const sector = ((Math.round(ang / (Math.PI / 4)) % 8) + 8) % 8;
  if (m > 1 && !SW.sectors[sector] && !SW.busy) grabSector(sector, yo, po);
  const {lit} = updateSweepCount();
  if (lit >= TICKS - 2 && Object.keys(SW.sectors).length >= 6 && S.photos.left && S.photos.right) finishSweep();
  return m > 3.6 ? 'Ease back a little' : 'Keep going';
}
async function grabSector(sector, yo, po) {
  const v = $('#hfVideo'); if (!v.videoWidth) return;
  SW.busy = true; SW.sectors[sector] = 'pending';
  try {
    const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight; c.getContext('2d').drawImage(v, 0, 0);
    const blob = await new Promise(r => c.toBlob(r, 'image/jpeg', 0.92));
    const p = await fileToPhoto(blob);
    const r = FL.detect(p.img), lm = r.faceLandmarks && r.faceLandmarks[0];
    if (!lm) { delete SW.sectors[sector]; URL.revokeObjectURL(p.url); SW.busy = false; return; }
    p.lm = lm; p.yaw = yawOf(lm); p.mat = r.facialTransformationMatrixes && r.facialTransformationMatrixes[0] || null; p.sector = sector;
    SW.sectors[sector] = p; S.scanFrames.push(p); S.analysis = null; tearDown3D();
    // strongest left/right turns become the side photos used for texture and analysis
    const side = sector === 0 || sector === 1 || sector === 7 ? 'right' : (sector >= 3 && sector <= 5 ? 'left' : null);
    if (side) { const cur = S.photos[side]; if (!cur || Math.abs(p.yaw - SW.base.yaw) > Math.abs(cur.yaw - SW.base.yaw) || (sector === 0 || sector === 4)) { S.photos[side] = p; } }
    const sc = $('#hfScope'); sc.classList.add('tickflash'); setTimeout(() => sc.classList.remove('tickflash'), 250);
    renderShots();
  } catch (e) { delete SW.sectors[sector]; }
  SW.busy = false;
}
function finishSweep() {
  if (!SW.on) return;
  stopSweep(); SW.lit.fill(1); $$('#hfTicks .tick').forEach(t => t.classList.add('on'));
  stopCamera(); setBadge(`Scan complete: ${Object.keys(SW.sectors).length + 1} angles`, 'ok');
  S.angle = 'front'; renderScan(true); drawLens(S.photos.front, 0.25);
  $('#hfSweepCount').hidden = false; $('#hfSweepCount').textContent = `${Object.keys(SW.sectors).length + 1} angles captured`;
}
function renderShots() {
  ['front', 'left', 'right'].forEach((k, i) => {
    const sh = $(`.shot[data-a="${k}"]`); sh.classList.toggle('has', !!S.photos[k]);
    sh.querySelector('.th').innerHTML = S.photos[k] ? `<img src="${S.photos[k].url}" alt="${k} photo">` : '+';
    sh.querySelector('.rm').hidden = !S.photos[k];
    $('#arc' + i).classList.toggle('done', !!S.photos[k]);
  });
  const hasFront = !!S.photos.front; $('#hfToAnalysis').disabled = !hasFront;
  $('#hfSkipSides').hidden = !hasFront || !!(S.photos.left && S.photos.right);
}

function updateCamButtons() {
  const on = CAM.on, touch = matchMedia('(pointer: coarse)').matches;
  $('#hfCamBtn').hidden = on || !CAM_OK; $('#hfNoCam').hidden = CAM_OK;
  if (!CAM_OK) { $('#hfUp').classList.replace('dark', 'primary'); } $('#hfCamOff').hidden = !on; $('#hfSnap').hidden = !on || S.angle === 'sweep';
  $('#hfTake').hidden = on || !touch;
  $('#hfLensPrompt').hidden = on || !CAM_OK || !!S.photos[S.angle];
}
$('#hfCamBtn').addEventListener('click', startCamera);
$('#hfLensCam').addEventListener('click', startCamera);
$('#hfCamOff').addEventListener('click', () => { stopCamera(); renderScan(); });
$('#hfFinishScan').addEventListener('click', finishSweep);
$('#hfSnap').addEventListener('click', () => { if (CAM.lm) capture(); else setBadge('No face in view yet', 'err'); });

function lensGeom(p, cv) {
  const L = p.lm; let x0 = 1, x1 = 0, y0 = 1, y1 = 0;
  if (L) for (const q of L) { x0 = Math.min(x0, q.x); x1 = Math.max(x1, q.x); y0 = Math.min(y0, q.y); y1 = Math.max(y1, q.y); }
  const cx = L ? (x0 + x1) / 2 * p.w : p.w / 2, cy = L ? (y0 + y1) / 2 * p.h : p.h / 2;
  const span = L ? Math.max((x1 - x0) * p.w, (y1 - y0) * p.h) * 1.7 : Math.min(p.w, p.h);
  const k = cv.width / span;
  return {k, ox: cv.width / 2 - cx * k, oy: cv.height / 2 - cy * k};
}
function lensCanvas() { const cv = $('#hfLens'); const s = cv.getBoundingClientRect().width * (devicePixelRatio || 1); if (cv.width !== Math.round(s)) { cv.width = cv.height = Math.round(s); } return cv; }
function drawEmptyLens() { const cv = lensCanvas(); cv.getContext('2d').clearRect(0, 0, cv.width, cv.height); $('#hfGhost').hidden = false; }
function drawLens(p, meshAlpha, upto) {
  const cv = lensCanvas(), g = cv.getContext('2d'), G = lensGeom(p, cv);
  $('#hfGhost').hidden = true;
  g.clearRect(0, 0, cv.width, cv.height);
  g.drawImage(p.img, G.ox, G.oy, p.w * G.k, p.h * G.k);
  if (!p.lm || !meshAlpha) return;
  const L = p.lm, n = upto == null ? L.length : upto;
  g.fillStyle = `rgba(228,80,42,${meshAlpha})`;
  const r = Math.max(1.2, cv.width / 260);
  for (let i = 0; i < n; i++) { const q = L[i]; g.beginPath(); g.arc(G.ox + q.x * p.w * G.k, G.oy + q.y * p.h * G.k, r, 0, 7); g.fill(); }
}
function animateMesh(p) {
  return new Promise(res => {
    const t0 = performance.now(), D = 1100;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    (function f(t) {
      const k = reduce ? 1 : Math.min(1, (t - t0) / D);
      drawLens(p, 0.9, Math.floor(p.lm.length * k));
      if (k < 1) requestAnimationFrame(f); else { setTimeout(() => { fadeMesh(p); res(); }, 250); }
    })(t0);
  });
}
function fadeMesh(p) { const t0 = performance.now(); (function f(t) { const k = Math.min(1, (t - t0) / 600); drawLens(p, 0.9 * (1 - k) + 0.25 * k); if (k < 1) requestAnimationFrame(f); })(t0); }

function renderScan(keepBadge) {
  const a = S.angle, A = ANGLES[a];
  $('#hfAngleTitle').textContent = A.title;
  $('#hfTips').innerHTML = A.tips.map(t => `<li>${t}</li>`).join('');
  const order = ['front', 'left', 'right'];
  order.forEach((k, i) => {
    const arc = $('#arc' + i); arc.classList.toggle('done', !!S.photos[k]); arc.classList.toggle('active', (k === a || (a === 'sweep' && k !== 'front')) && !S.photos[k]);
    const sh = $(`.shot[data-a="${k}"]`); sh.classList.toggle('has', !!S.photos[k]); sh.classList.toggle('cur', k === a);
    sh.querySelector('.th').innerHTML = S.photos[k] ? `<img src="${S.photos[k].url}" alt="${k} photo">` : '+';
    sh.querySelector('.rm').hidden = !S.photos[k];
  });
  const p = a === 'sweep' ? null : S.photos[a];
  if (!CAM.on) { if (p) drawLens(p, 0.25); else drawEmptyLens(); }
  if (!keepBadge && !CAM.on) setBadge(p ? 'Face captured' : '', p ? 'ok' : '');
  $('#hfTake').textContent = p ? 'Retake photo' : 'Take photo';
  updateCamButtons();
  const hasFront = !!S.photos.front, hasAll = order.every(k => S.photos[k]);
  $('#hfToAnalysis').disabled = !hasFront;
  $('#hfToAnalysis').textContent = hasAll ? 'Analyse my face' : 'Analyse my face';
  $('#hfSkipSides').hidden = !hasFront || hasAll;
}

/* ---------------- step 2: analysis ---------------- */
let sampleNS;
async function getSample() {
  if (sampleNS !== undefined) return sampleNS;
  try { sampleNS = (window.claude && window.claude.use) ? await window.claude.use('sample') : null; } catch (e) { sampleNS = null; }
  return sampleNS;
}

function drawMeasure(p, m, k) {
  const img = $('#hfAnaImg'); img.src = p.url;
  const cv = $('#hfAnaCv'); cv.width = p.w; cv.height = p.h;
  const g = cv.getContext('2d'); g.clearRect(0, 0, p.w, p.h);
  const L = p.lm, P = i => [L[i].x * p.w, L[i].y * p.h];
  const lw = Math.max(2, p.w / 360);
  g.fillStyle = 'rgba(255,255,255,.5)';
  for (const q of L) { g.beginPath(); g.arc(q.x * p.w, q.y * p.h, lw * 0.55, 0, 7); g.fill(); }
  const lines = [[54, 284, 'Forehead'], [234, 454, 'Cheekbones'], [172, 397, 'Jaw'], [10, 152, 'Length']];
  g.font = `600 ${Math.round(p.w / 34)}px Archivo, sans-serif`;
  lines.forEach(([a, b, t], i) => {
    const kk = Math.max(0, Math.min(1, k * 4 - i));
    if (!kk) return;
    const [x1, y1] = P(a), [x2, y2] = P(b);
    g.strokeStyle = '#e4502a'; g.lineWidth = lw * 1.4; g.setLineDash([]);
    g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 + (x2 - x1) * kk, y1 + (y2 - y1) * kk); g.stroke();
    if (kk === 1) {
      g.fillStyle = '#e4502a';
      [[x1, y1], [x2, y2]].forEach(([x, y]) => { g.beginPath(); g.arc(x, y, lw * 2.2, 0, 7); g.fill(); });
      g.fillStyle = '#fff'; g.strokeStyle = 'rgba(0,0,0,.75)'; g.lineWidth = lw * 2.5;
      const tx = t === 'Length' ? x1 + lw * 6 : Math.max(x1, x2) + lw * 6, ty = t === 'Length' ? (y1 + y2) / 2 : (y1 + y2) / 2 + lw * 3;
      g.strokeText(t, tx, ty); g.fillText(t, tx, ty);
    }
  });
}

async function runAnalysis() {
  if (S.busy) return;
  const p = S.photos.front; if (!p) { go(1); return; }
  if (S.analysis) { renderAnalysis(); return; }
  S.busy = true;
  $('#hfAnaResult').hidden = true; $('#hfAnaWorking').hidden = false; $('#hfToCuts').disabled = true;
  const meter = $('#hfMeter'), line = $('#hfAnaLine');
  meter.style.width = '0%';
  const m = measure(p);
  const geoShape = shapeFromGeometry(m);
  // animate measurement overlay
  await new Promise(res => { const t0 = performance.now(); (function f(t) { const k = Math.min(1, (t - t0) / 1600); drawMeasure(p, m, k); if (k < 1) requestAnimationFrame(f); else res(); })(t0); });
  line.textContent = 'Measured 478 points across your face'; meter.style.width = '35%';

  let result = null, source = 'geometry';
  const sample = await getSample();
  if (sample) {
    line.textContent = 'Claude is reviewing your photos…'; meter.style.width = '60%';
    try {
      const lim = sample.limits ? await sample.limits() : {images: false};
      const imgs = lim && lim.images ? ['front', 'left', 'right'].filter(k => S.photos[k]).map(k => S.photos[k].blob) : [];
      const prompt = buildPrompt(m, geoShape, imgs.length);
      const opts = {modelTier: 'default'}; if (imgs.length) opts.images = imgs;
      const v = await sample.json(prompt, opts);
      result = normalise(v, m, geoShape);
      source = imgs.length ? 'claude' : 'claude-text';
    } catch (e) {
      console.warn('Claude analysis unavailable', e);
      line.textContent = e && e.code === 'not_granted' ? 'Claude access was not allowed, using face measurements' : 'Claude was unavailable, using face measurements';
    }
  }
  if (!result) result = normalise(null, m, geoShape);
  result.source = source; result.m = m;
  meter.style.width = '100%';
  S.analysis = result; S.ranked = null;
  await new Promise(r => setTimeout(r, 450));
  S.busy = false;
  renderAnalysis();
}

function buildPrompt(m, geoShape, nImg) {
  const r = x => Math.round(x * 100) / 100;
  return `You are an experienced barber helping a customer at Huncho Barbers choose a haircut. ${nImg ? `You are given ${nImg} photo(s) of the customer: the first faces the camera${nImg > 1 ? ', the others are side views' : ''}.` : 'No photos are attached; work from the measurements.'}
Landmark measurements from the front photo (ratios): face length / cheekbone width = ${r(m.lenToCheek)}; forehead / cheekbone width = ${r(m.foreToCheek)}; jaw / cheekbone width = ${r(m.jawToCheek)}; forehead / jaw = ${r(m.foreToJaw)}. Note: the length is measured from the upper forehead, not the hairline, so true face length is a little longer. A simple geometric rule suggests "${geoShape}".
Estimate the face shape (one of: ${SHAPES.join(', ')}). Treat it as an estimate and say how confident you are. Describe the forehead, jawline, cheekbones and overall proportions in one short plain sentence each, written to the customer ("your ..."). If hair is visible, note its texture and density briefly; otherwise say it's not clear.
Then score every one of these haircuts from 0 to 100 for how well it may suit this customer, considering face shape, features and visible hair type, with a one-sentence reason written to the customer (no hype, no guarantees): ${STYLES.map(s => s.id).join(', ')}.
If a photo is unusable (no face, heavy filter, face covered), set photoOk false and explain in photoIssue.
Do not comment on attractiveness, age, ethnicity, or anything other than face shape, proportions and hair.
Return only JSON: {"faceShape":"...","confidence":"low|medium|high","summary":"one sentence","features":{"forehead":"...","jawline":"...","cheekbones":"...","proportions":"..."},"hair":"...","photoOk":true,"photoIssue":"","recommendations":[{"id":"...","score":0,"why":"..."}]}`;
}

function normalise(v, m, geoShape) {
  const desc = describe(m);
  const shape = v && SHAPES.includes(String(v.faceShape).toLowerCase()) ? String(v.faceShape).toLowerCase() : geoShape;
  const recMap = {};
  if (v && Array.isArray(v.recommendations)) for (const r of v.recommendations) if (r && STYLES.find(s => s.id === r.id)) recMap[r.id] = {score: Math.max(0, Math.min(100, Math.round(+r.score || 0))), why: String(r.why || '')};
  const recs = STYLES.map(s => recMap[s.id] || {score: FIT[shape][s.id], why: WHY[shape]}).map((r, i) => ({...r, id: STYLES[i].id}));
  return {
    shape, confidence: v && ['low', 'medium', 'high'].includes(v.confidence) ? v.confidence : 'medium',
    summary: v && v.summary ? String(v.summary) : `Your proportions point to a${/^[aeiou]/.test(shape) ? 'n' : ''} ${shape} face shape.`,
    features: Object.assign({}, desc, v && v.features ? Object.fromEntries(Object.entries(v.features).filter(([k, x]) => desc[k] && typeof x === 'string' && x)) : {}),
    hair: v && v.hair ? String(v.hair) : '',
    photoOk: !(v && v.photoOk === false), photoIssue: v && v.photoIssue ? String(v.photoIssue) : '',
    recs: recs.sort((a, b) => b.score - a.score)
  };
}

function esc(s) { return String(s).replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c])); }
function renderAnalysis() {
  const A = S.analysis; if (!A) return;
  drawMeasure(S.photos.front, A.m, 1);
  $('#hfAnaWorking').hidden = true; $('#hfAnaResult').hidden = false; $('#hfToCuts').disabled = false;
  $('#hfShape').textContent = A.shape;
  $('#hfConf').textContent = `Estimated, ${A.confidence} confidence`;
  $('#hfSummary').textContent = A.summary + (A.photoOk ? '' : ' ' + A.photoIssue);
  const f = A.features;
  $('#hfFeat').innerHTML = [['Forehead', f.forehead], ['Jawline', f.jawline], ['Cheekbones', f.cheekbones], ['Proportions', f.proportions]].concat(A.hair ? [['Hair', A.hair]] : [])
    .map(([k, v]) => `<div><dt>${k}</dt><dd>${esc(v)}</dd></div>`).join('');
  $('#hfSrc').textContent = A.source === 'claude' ? 'Analysed by Claude from your photos and face measurements'
    : A.source === 'claude-text' ? 'Analysed by Claude from your face measurements'
    : 'Estimated from your face measurements';
}
$('#hfToCuts').addEventListener('click', () => { renderCuts(); go(3); });
$('#hfReanalyse').addEventListener('click', () => { S.analysis = null; runAnalysis(); });

/* ---------------- step 3: cuts ---------------- */
function dots(n) { return `<span class="dots">${[1, 2, 3].map(i => `<i class="${i <= n ? 'f' : ''}"></i>`).join('')}</span>`; }
const LV = ['', 'Low', 'Medium', 'High'], DF = ['', 'Easy', 'Medium', 'Advanced'];
function cutCard(r, i) {
  const s = STYLES.find(x => x.id === r.id);
  return `<button class="cut${S.pick === s.id ? ' sel' : ''}" data-id="${s.id}" aria-pressed="${S.pick === s.id}">
    <div class="pic"><img src="styles/${s.id}.jpg" alt="${s.name} reference photo" loading="lazy"></div>
    ${i < 3 ? `<span class="rank">${i === 0 ? 'Top match' : 'Match ' + (i + 1)}</span>` : ''}
    <span class="score">${r.score}% fit</span>
    <div class="info"><h4>${s.name}</h4><p>${esc(r.why || s.blurb)}</p>
      <div class="lv"><span>Upkeep ${dots(s.maint)}</span><span>Styling ${dots(s.diff)}</span></div>
      <span class="vh-text" style="position:absolute;left:-9999px">Upkeep ${LV[s.maint]}, styling ${DF[s.diff]}</span></div></button>`;
}
function renderCuts() {
  const R = S.analysis.recs;
  if (!S.pick) S.pick = R[0].id;
  $('#hfCutsShape').textContent = S.analysis.shape;
  $('#hfTop').innerHTML = R.slice(0, 3).map(cutCard).join('');
  $('#hfRest').innerHTML = R.slice(3).map((r, i) => cutCard(r, i + 3)).join('');
  $$('#hfTop .cut, #hfRest .cut').forEach(b => b.addEventListener('click', () => { S.pick = b.dataset.id; $$('.cut').forEach(c => { c.classList.toggle('sel', c.dataset.id === S.pick); c.setAttribute('aria-pressed', c.dataset.id === S.pick); }); updatePick(); }));
  updatePick();
}
function updatePick() { const s = STYLES.find(x => x.id === S.pick); $('#hfChosen').innerHTML = `Your pick: <b>${s.name}</b>`; }
$('#hfTo3D').addEventListener('click', () => go(4));

/* ---------------- step 4: 3D + try-on ---------------- */
let GLX = null, raf = 0;
function tearDown3D() { stop3D(); GLX = null; if (HEAD) HEAD.stop(); HEAD = null; headPick = null; }
function stop3D() { cancelAnimationFrame(raf); raf = 0; if (HEAD) HEAD.stop(); }
let HEAD = null, headPick = null;
function start3D() {
  const s0 = STYLES.find(x => x.id === S.pick);
  $('#hfRefImg').src = `styles/${s0.id}.jpg`; $('#hfRefCap').textContent = `${s0.name} reference (not you)`;
  $('#hfYouImg').src = S.photos.front.url;
  $('#hfTryName').textContent = s0.name;
  const connected = !!CONFIG.tryOnEndpoint;
  $('#hfTryState').hidden = connected; $('#hfTryGo').hidden = !connected;
  renderStyleChips();
  try {
    if (!HEAD) {
      $('#hfStageMsg').textContent = 'Building your head…';
      const model = buildHeadModel(S.photos.front, [S.photos.left, S.photos.right], FLmod.FaceLandmarker.FACE_LANDMARKS_TESSELATION);
      HEAD = createHeadViewer($('#hfGL'), model);
      HEAD.onManual(() => $$('#hfViews .chip').forEach(x => x.classList.remove('on')));
      const nat = $('#hfTones .tone[data-c="natural"]'); nat.style.setProperty('--tc', rgbCss(HEAD.naturalHair));
      $$('#hfTones .tone').forEach(t => { if (t.dataset.c !== 'natural') t.style.setProperty('--tc', rgbCss(t.dataset.c.split(',').map(Number))); });
      $$('#hfViews .chip').forEach(x => x.classList.toggle('on', x.dataset.v === 'spin'));
      $$('#hfTones .tone').forEach(x => x.classList.toggle('on', x.dataset.c === 'natural'));
      $('#hfStageMsg').textContent = model.views.length > 1 ? `Built from ${model.views.length} of your photos. Drag to turn, scroll to zoom.` : 'Built from your front photo. Add side photos for a better back and sides.';
    }
    if (headPick !== S.pick) { HEAD.setStyle(S.pick); headPick = S.pick; }
    $('#hf3dTitle').textContent = s0.name;
    HEAD.start();
  } catch (e) { console.error(e); $('#hfStageMsg').textContent = '3D view isn\'t supported in this browser.'; }
}
function renderStyleChips() {
  const R = S.analysis ? S.analysis.recs : STYLES.map(x => ({id: x.id}));
  $('#hfStyleChips').innerHTML = R.map(r => { const st = STYLES.find(x => x.id === r.id); return `<button type="button" class="chip${r.id === S.pick ? ' on' : ''}" data-id="${r.id}">${st.name}</button>`; }).join('');
  $$('#hfStyleChips .chip').forEach(c => c.addEventListener('click', () => { S.pick = c.dataset.id; start3D(); if ($('#hfChosen')) updatePick(); $$('.cut').forEach(x => x.classList.toggle('sel', x.dataset.id === S.pick)); }));
}
function build3D(cv, p) {
  const gl = cv.getContext('webgl', {antialias: true, premultipliedAlpha: false});
  if (!gl) throw new Error('no webgl');
  const L = p.lm, N = L.length;
  // positions in pixel space, centred and scaled
  let cx = 0, cy = 0, cz = 0;
  const P = L.map(q => [q.x * p.w, q.y * p.h, q.z * p.w]);
  P.forEach(v => { cx += v[0]; cy += v[1]; cz += v[2]; }); cx /= N; cy /= N; cz /= N;
  let span = 0; P.forEach(v => { span = Math.max(span, Math.abs(v[1] - cy)); });
  const pos = new Float32Array(N * 3), uv = new Float32Array(N * 2);
  P.forEach((v, i) => { pos[i * 3] = (v[0] - cx) / span; pos[i * 3 + 1] = -(v[1] - cy) / span; pos[i * 3 + 2] = -(v[2] - cz) / span; uv[i * 2] = L[i].x; uv[i * 2 + 1] = L[i].y; });
  // triangles from the tessellation edges (3-cliques)
  const E = FLmod.FaceLandmarker.FACE_LANDMARKS_TESSELATION;
  const adj = Array.from({length: N}, () => new Set());
  E.forEach(e => { adj[e.start].add(e.end); adj[e.end].add(e.start); });
  const tris = [];
  for (let a = 0; a < N; a++) for (const b of adj[a]) if (b > a) for (const c of adj[b]) if (c > b && adj[a].has(c)) tris.push(a, b, c);
  // close the eye and mouth openings the tessellation leaves empty
  const rings = [
    [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246],
    [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466]
  ];
  const mouth = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191];
  for (const ring of rings) {
    const rx = ring.reduce((a, i) => a + pos[i * 3], 0) / ring.length;
    let c = -1, best = 1e9;
    for (const k of [468, 473]) if (k < N && Math.abs(pos[k * 3] - rx) < best) { best = Math.abs(pos[k * 3] - rx); c = k; }
    if (c < 0) c = ring[0];
    for (let i = 0; i < ring.length; i++) tris.push(c, ring[i], ring[(i + 1) % ring.length]);
  }
  for (let i = 1; i < mouth.length - 1; i++) tris.push(mouth[0], mouth[i], mouth[i + 1]);
  // normals, oriented towards the viewer
  const nor = new Float32Array(N * 3);
  for (let t = 0; t < tris.length; t += 3) {
    const [a, b, c] = [tris[t], tris[t + 1], tris[t + 2]];
    const ux = pos[b * 3] - pos[a * 3], uy = pos[b * 3 + 1] - pos[a * 3 + 1], uz = pos[b * 3 + 2] - pos[a * 3 + 2];
    const vx = pos[c * 3] - pos[a * 3], vy = pos[c * 3 + 1] - pos[a * 3 + 1], vz = pos[c * 3 + 2] - pos[a * 3 + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    if (nz < 0) { nx = -nx; ny = -ny; nz = -nz; }
    for (const i of [a, b, c]) { nor[i * 3] += nx; nor[i * 3 + 1] += ny; nor[i * 3 + 2] += nz; }
  }
  const lines = []; E.forEach(e => lines.push(e.start, e.end));

  const vs = `attribute vec3 p;attribute vec2 t;attribute vec3 n;uniform mat4 M;uniform mat4 PR;varying vec2 vt;varying vec3 vn;
    void main(){vt=t;vn=normalize(mat3(M)*n);gl_Position=PR*M*vec4(p,1.);gl_PointSize=2.5;}`;
  const fs = `precision mediump float;uniform sampler2D tex;uniform int mode;varying vec2 vt;varying vec3 vn;
    void main(){ if(mode==1){gl_FragColor=vec4(.98,.35,0.,.85);return;}
      vec3 c=texture2D(tex,vt).rgb; float d=max(dot(normalize(vn),normalize(vec3(-.35,.25,1.))),0.);
      float rim=pow(1.-max(vn.z,0.),3.)*.35;
      gl_FragColor=vec4(c*(.45+.65*d)+vec3(.98,.35,0.)*rim,1.);}`;
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(pr); gl.useProgram(pr);
  const buf = (data, loc, size) => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW); const l = gl.getAttribLocation(pr, loc); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0); };
  buf(pos, 'p', 3); buf(uv, 't', 2); buf(nor, 'n', 3);
  const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(tris), gl.STATIC_DRAW);
  const lb = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lb); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(lines), gl.STATIC_DRAW);
  const tx = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tx);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, p.img);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.enable(gl.DEPTH_TEST);
  const U = n => gl.getUniformLocation(pr, n);
  const st = {gl, U, ib, lb, nt: tris.length, nl: lines.length, yaw: 0, pitch: 0, vy: 0, auto: true, mode: 'photo', drag: null, t0: performance.now()};
  // interaction
  cv.onpointerdown = e => { cv.setPointerCapture(e.pointerId); st.drag = {x: e.clientX, y: e.clientY, yaw: st.yaw, pitch: st.pitch}; st.auto = false; };
  cv.onpointermove = e => { if (!st.drag) return; st.yaw = Math.max(-1.25, Math.min(1.25, st.drag.yaw + (e.clientX - st.drag.x) / 160)); st.pitch = Math.max(-.6, Math.min(.6, st.drag.pitch + (e.clientY - st.drag.y) / 200)); };
  cv.onpointerup = cv.onpointercancel = () => { st.drag = null; };
  return st;
}
function mul(a, b) { const r = new Float32Array(16); for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + j] * b[i * 4 + k]; r[i * 4 + j] = s; } return r; }
function loop3D() {
  if (raf) return;
  const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const f = t => {
    const st = GLX; if (!st) return;
    const {gl, U} = st, cv = gl.canvas, dpr = devicePixelRatio || 1;
    const w = Math.round(cv.clientWidth * dpr), h = Math.round(cv.clientHeight * dpr);
    if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    if (st.auto && !reduce) { const k = (t - st.t0) / 1000; st.yaw = Math.sin(k * 0.7) * 0.75; st.pitch = Math.sin(k * 0.45) * 0.12; }
    const cy = Math.cos(st.yaw), sy = Math.sin(st.yaw), cp = Math.cos(st.pitch), sp = Math.sin(st.pitch);
    const Ry = [cy, 0, -sy, 0, 0, 1, 0, 0, sy, 0, cy, 0, 0, 0, 0, 1], Rx = [1, 0, 0, 0, 0, cp, sp, 0, 0, -sp, cp, 0, 0, 0, 0, 1];
    const T = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0.02, -4.1, 1];
    const M = mul(T, mul(Rx, Ry));
    const fov = 0.62, a = w / h, ft = 1 / Math.tan(fov / 2), nr = 0.1, fr = 20;
    const PR = [ft / a, 0, 0, 0, 0, ft, 0, 0, 0, 0, (fr + nr) / (nr - fr), -1, 0, 0, 2 * fr * nr / (nr - fr), 0];
    gl.uniformMatrix4fv(U('M'), false, M); gl.uniformMatrix4fv(U('PR'), false, new Float32Array(PR));
    if (st.mode !== 'mesh') { gl.uniform1i(U('mode'), 0); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, st.ib); gl.drawElements(gl.TRIANGLES, st.nt, gl.UNSIGNED_SHORT, 0); }
    if (st.mode !== 'photo') { gl.uniform1i(U('mode'), 1); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, st.lb); gl.drawElements(gl.LINES, st.nl, gl.UNSIGNED_SHORT, 0); }
    raf = requestAnimationFrame(f);
  };
  raf = requestAnimationFrame(f);
}
function rgbCss(c) { return `rgb(${c.map(v => Math.round(Math.min(1, v * 1.35) * 255)).join(',')})`; }
$$('#hfViews .chip').forEach(c => c.addEventListener('click', () => { if (!HEAD) return; const v = c.dataset.v; if (v === 'spin') HEAD.setSpin(true); else HEAD.setView(v); $$('#hfViews .chip').forEach(x => x.classList.toggle('on', x === c)); }));
$$('#hfTones .tone').forEach(t => t.addEventListener('click', () => { if (!HEAD) return; HEAD.setHairColor(t.dataset.c === 'natural' ? null : t.dataset.c.split(',').map(Number)); $$('#hfTones .tone').forEach(x => x.classList.toggle('on', x === t)); }));
$$('#hfModes .chip').forEach(c => c.addEventListener('click', () => { if (!HEAD) return; const m = c.dataset.m; HEAD.setMode(m === 'scan' ? 1 : m === 'clay' ? 2 : 0); HEAD.setHair(m !== 'nohair'); $$('#hfModes .chip').forEach(x => x.classList.toggle('on', x === c)); }));
$('#hfTryGo').addEventListener('click', async () => {
  // Only runs when a backend is configured. Expects {image: dataURL, style} -> {image: dataURL|url}
  const s = STYLES.find(x => x.id === S.pick), btn = $('#hfTryGo');
  btn.disabled = true; btn.textContent = 'Creating preview…';
  try {
    const dataUrl = await new Promise(r => { const fr = new FileReader(); fr.onload = () => r(fr.result); fr.readAsDataURL(S.photos.front.blob); });
    const res = await fetch(CONFIG.tryOnEndpoint, {method: 'POST', headers: {'content-type': 'application/json'}, body: JSON.stringify({image: dataUrl, style: s.name})});
    const j = await res.json(); if (!j.image) throw new Error('no image');
    $('#hfYouImg').src = j.image; $('#hfYouCap').textContent = `You with a ${s.name.toLowerCase()} (AI preview)`;
  } catch (e) { toast('Preview failed. Try again in a moment.'); }
  btn.disabled = false; btn.textContent = `Preview ${s.name.toLowerCase()} on me`;
});
$('#hfToSend').addEventListener('click', () => go(5));

/* ---------------- step 5: send ---------------- */
function summaryText() {
  const s = STYLES.find(x => x.id === S.pick), A = S.analysis, notes = $('#hfNotes').value.trim();
  return `Hi Huncho, I'd like a ${s.name.toLowerCase()}.` + (A ? ` Face shape estimate: ${A.shape}.` : '') + (notes ? `\nNotes for my barber: ${notes}` : '') + `\n(From the Haircut Finder on your website)`;
}
function renderTicket() {
  const s = STYLES.find(x => x.id === S.pick); if (!s) return;
  $('#hfTicketImg').src = `styles/${s.id}.jpg`; $('#hfTicketImg').alt = `${s.name} reference photo`;
  $('#hfTicketName').textContent = s.name;
  $('#hfTicketMeta').textContent = `Upkeep ${LV[s.maint].toLowerCase()}, styling ${DF[s.diff].toLowerCase()}` + (S.analysis ? `. Estimated ${S.analysis.shape} face shape` : '');
  $('#hfTicketWhy').textContent = (S.analysis && S.analysis.recs.find(r => r.id === s.id) || {}).why || s.blurb;
  updateSms();
}
function updateSms() { $('#hfSms').href = `sms:${CONFIG.shopPhone}?&body=${encodeURIComponent(summaryText())}`; }
$('#hfNotes').addEventListener('input', updateSms);
$('#hfCopy').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText(summaryText()); toast('Copied. Paste it in a text or DM to the shop'); }
  catch (e) { const t = $('#hfCopyFallback'); t.hidden = false; t.value = summaryText(); t.select(); toast('Select and copy the text below'); }
});
$('#hfChange').addEventListener('click', () => go(3));

/* back buttons */
$$('[data-back]').forEach(b => b.addEventListener('click', () => go(S.step - 1)));
