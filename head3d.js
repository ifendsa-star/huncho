/* Huncho 3D head builder.
   Builds a full head from MediaPipe face landmarks (front photo, plus optional side photos),
   textures it by projecting the customer's own photos, and grows a 3D haircut on the scalp
   using shell-rendered hair. Pure WebGL 1, no libraries. */

const OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109];
const EYE_L = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246];
const EYE_R = [263,249,390,373,374,380,381,382,362,398,384,385,386,387,388,466];
const MOUTH = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191];

/* Haircut recipes. Lengths are in face-height units (forehead to chin = 1).
   fade: height (0 = hairline, 1 = crown) where the sides reach full length; low: length at the very bottom. */
export const HAIR = {
  'low-fade':      {top: .085, side: .035, fade: .28, low: .004, comb: 'forward', curl: .25},
  'mid-fade':      {top: .095, side: .03,  fade: .46, low: .003, comb: 'up',      curl: .3},
  'high-fade':     {top: .08,  side: .02,  fade: .68, low: .002, comb: 'up',      curl: .2},
  'skin-fade':     {top: .1,   side: .025, fade: .5,  low: 0,    comb: 'up',      curl: .55},
  'taper-fade':    {top: .09,  side: .055, fade: .12, low: .008, comb: 'forward', curl: .35},
  'buzz-cut':      {top: .012, side: .011, fade: 0,   low: .009, comb: 'up',      curl: 0},
  'textured-crop': {top: .075, side: .025, fade: .45, low: .003, comb: 'forward', curl: .7, fringe: .035},
  'french-crop':   {top: .065, side: .03,  fade: .3,  low: .004, comb: 'forward', curl: .15, fringe: .055},
  'fringe':        {top: .13,  side: .065, fade: .2,  low: .01,  comb: 'forward', curl: .4, fringe: .13},
  'slick-back':    {top: .14,  side: .025, fade: .45, low: .003, comb: 'back',    curl: .05},
  'side-part':     {top: .105, side: .035, fade: .25, low: .006, comb: 'side',    curl: .1},
  'crew-cut':      {top: .05,  side: .02,  fade: .3,  low: .005, comb: 'forward', curl: .1, crew: true},
  'modern-mullet': {top: .085, side: .025, fade: .45, low: .003, comb: 'forward', curl: .6, fringe: .03, nape: .2}
};

const lerp = (a, b, t) => a + (b - a) * t;
const sstep = (a, b, x) => { const t = Math.max(0, Math.min(1, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

function rotOf(mat) { // 3x3 rotation (row-major) from MediaPipe column-major 4x4
  if (!mat) return null;
  const d = mat.data || mat, sx = Math.hypot(d[0], d[1], d[2]), sy = Math.hypot(d[4], d[5], d[6]), sz = Math.hypot(d[8], d[9], d[10]);
  return [d[0] / sx, d[4] / sy, d[8] / sz, d[1] / sx, d[5] / sy, d[9] / sz, d[2] / sx, d[6] / sy, d[10] / sz];
}
const mul3 = (a, b) => { const r = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]; return r; };
const tr3 = a => [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]];
const app3 = (R, v) => [R[0] * v[0] + R[1] * v[1] + R[2] * v[2], R[3] * v[0] + R[4] * v[1] + R[5] * v[2], R[6] * v[0] + R[7] * v[1] + R[8] * v[2]];
const rotY = a => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
const rotX = a => [1, 0, 0, 0, Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a)];

function sampleColor(img, pts) {
  const c = document.createElement('canvas'); c.width = img.naturalWidth || img.width; c.height = img.naturalHeight || img.height;
  const g = c.getContext('2d', {willReadFrequently: true}); g.drawImage(img, 0, 0);
  let r = 0, gg = 0, b = 0, n = 0;
  for (const [x, y] of pts) {
    const X = Math.max(0, Math.min(c.width - 3, Math.round(x))), Y = Math.max(0, Math.min(c.height - 3, Math.round(y)));
    const d = g.getImageData(X, Y, 3, 3).data;
    for (let i = 0; i < d.length; i += 4) { r += d[i]; gg += d[i + 1]; b += d[i + 2]; n++; }
  }
  return n ? [r / n / 255, gg / n / 255, b / n / 255] : [.5, .4, .35];
}

export function buildHeadModel(front, sides, tess) {
  const W = front.w, H = front.h, L = front.lm, N = L.length;
  const px = i => [L[i].x * W, L[i].y * H];
  const s = Math.hypot(px(10)[0] - px(152)[0], px(10)[1] - px(152)[1]);
  const cx = (px(234)[0] + px(454)[0]) / 2, cy = px(168)[1];
  // face landmarks in head space (face height = 1, +z towards the camera)
  const F = L.map(q => [(q.x * W - cx) / s, -(q.y * H - cy) / s, -(q.z * W) / s]);
  const Y10 = F[10][1], Ych = F[152][1], Z10 = F[10][2];
  const rx = (F[454][0] - F[234][0]) / 2 * 1.07;
  const top = Y10 + 0.37, cyH = (top + Ych) / 2 + 0.04, ry = (top - Ych) / 2 + 0.02;
  const rz = rx * 1.24, cz = Z10 - 0.03 - rz;
  const pivot = [0, cyH, cz];
  const eyeY = (F[33][1] + F[263][1]) / 2, mouthY = F[13][1];

  // face triangles (tessellation cliques + eye/mouth fills)
  const adj = Array.from({length: N}, () => new Set());
  tess.forEach(e => { adj[e.start].add(e.end); adj[e.end].add(e.start); });
  const faceTris = [];
  for (let a = 0; a < N; a++) for (const b of adj[a]) if (b > a) for (const c of adj[b]) if (c > b && adj[a].has(c)) faceTris.push(a, b, c);
  for (const ring of [EYE_L, EYE_R]) { let c = ring[0]; const rx0 = ring.reduce((t, i) => t + F[i][0], 0) / ring.length; let best = 9; for (const k of [468, 473]) if (k < N && Math.abs(F[k][0] - rx0) < best) { best = Math.abs(F[k][0] - rx0); c = k; } for (let i = 0; i < ring.length; i++) faceTris.push(c, ring[i], ring[(i + 1) % ring.length]); }
  for (let i = 1; i < MOUTH.length - 1; i++) faceTris.push(MOUTH[0], MOUTH[i], MOUTH[i + 1]);

  // face polygon + depth lookup
  const poly = OVAL.map(i => F[i]);
  const inPoly = (x, y) => { let c = false; for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) { const [xi, yi] = poly[i], [xj, yj] = poly[j]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
  const faceZ = (x, y) => {
    for (let t = 0; t < faceTris.length; t += 3) {
      const A = F[faceTris[t]], B = F[faceTris[t + 1]], C = F[faceTris[t + 2]];
      const d = (B[1] - C[1]) * (A[0] - C[0]) + (C[0] - B[0]) * (A[1] - C[1]); if (Math.abs(d) < 1e-9) continue;
      const u = ((B[1] - C[1]) * (x - C[0]) + (C[0] - B[0]) * (y - C[1])) / d, v = ((C[1] - A[1]) * (x - C[0]) + (A[0] - C[0]) * (y - C[1])) / d, w = 1 - u - v;
      if (u >= -1e-4 && v >= -1e-4 && w >= -1e-4) return u * A[2] + v * B[2] + w * C[2];
    }
    return null;
  };
  const nearestOval = (x, y) => { let bd = 9, bz = 0; for (let i = 0; i < poly.length; i++) { const [ax, ay, az] = poly[i], [bx, by, bz2] = poly[(i + 1) % poly.length]; const vx = bx - ax, vy = by - ay; const t = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy))); const d = Math.hypot(ax + vx * t - x, ay + vy * t - y); if (d < bd) { bd = d; bz = az + (bz2 - az) * t; } } return [bd, bz]; };

  // ---- head grid ----
  const NU = 128, NV = 96;
  const hp = [], hTheta = [], hPhi = [];
  for (let j = 0; j <= NV; j++) {
    const phi = -Math.PI / 2 + j / NV * Math.PI;
    for (let i = 0; i <= NU; i++) {
      const th = i / NU * Math.PI * 2 - Math.PI;
      const low = Math.max(0, -Math.sin(phi));
      const jaw = lerp(1, 0.8, low) * lerp(1, 1.04, Math.max(0, Math.sin(phi)) * 0.5);
      let x = rx * Math.cos(phi) * Math.sin(th) * jaw, y = cyH + ry * Math.sin(phi), z = cz + rz * Math.cos(phi) * Math.cos(th) * lerp(1, 0.92, low);
      if (Math.cos(th) > 0) {
        if (inPoly(x, y)) { const fz = faceZ(x, y); if (fz !== null) z = fz - 0.012; }
        else { const [d, bz] = nearestOval(x, y); const w = 1 - sstep(0, 0.16, d); z = lerp(z, Math.max(z, bz - 0.01), w * Math.cos(th)); }
      }
      hp.push([x, y, z]); hTheta.push(th); hPhi.push(phi);
    }
  }
  const hIdx = [];
  for (let j = 0; j < NV; j++) for (let i = 0; i < NU; i++) { const a = j * (NU + 1) + i, b = a + 1, c = a + NU + 1, d = c + 1; hIdx.push(a, c, b, b, c, d); }

  // ---- neck ----
  const np = [], nIdx = [], NNU = 48, NNV = 10;
  for (let j = 0; j <= NNV; j++) { const y = lerp(cyH - ry * 0.45, Ych - 0.75, j / NNV); const sc = lerp(1, 1.08, j / NNV); for (let i = 0; i <= NNU; i++) { const th = i / NNU * Math.PI * 2 - Math.PI; np.push([rx * 0.56 * sc * Math.sin(th), y, cz + rz * 0.04 + rz * 0.52 * sc * Math.cos(th)]); } }
  for (let j = 0; j < NNV; j++) for (let i = 0; i < NNU; i++) { const a = j * (NNU + 1) + i, b = a + 1, c = a + NNU + 1, d = c + 1; nIdx.push(a, c, b, b, c, d); }

  // ---- ears ----
  const ep = [], eIdx = [];
  const earY = (eyeY + F[2][1]) / 2 + 0.02, earZ = cz + rz * 0.02;
  for (const side of [-1, 1]) {
    const base = ep.length, EU = 16, EV = 12;
    for (let j = 0; j <= EV; j++) { const ph = -Math.PI / 2 + j / EV * Math.PI; for (let i = 0; i <= EU; i++) { const t = i / EU * Math.PI * 2; ep.push([side * (rx * 0.97 + 0.035 * Math.cos(ph) * (0.6 + 0.4 * Math.cos(t))), earY + 0.14 * Math.sin(ph), earZ + 0.075 * Math.cos(ph) * Math.sin(t)]); } }
    for (let j = 0; j < EV; j++) for (let i = 0; i < EU; i++) { const a = base + j * (EU + 1) + i, b = a + 1, c = a + EU + 1, d = c + 1; eIdx.push(a, c, b, b, c, d); }
  }

  // normals
  const normals = (P, I, outward) => { const n = P.map(() => [0, 0, 0]); for (let t = 0; t < I.length; t += 3) { const [a, b, c] = [I[t], I[t + 1], I[t + 2]]; const u = [P[b][0] - P[a][0], P[b][1] - P[a][1], P[b][2] - P[a][2]], v = [P[c][0] - P[a][0], P[c][1] - P[a][1], P[c][2] - P[a][2]]; const f = [u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0]]; for (const k of [a, b, c]) { n[k][0] += f[0]; n[k][1] += f[1]; n[k][2] += f[2]; } } return n.map((v, k) => { let l = Math.hypot(...v) || 1; let r = [v[0] / l, v[1] / l, v[2] / l]; const o = outward(P[k]); if (r[0] * o[0] + r[1] * o[1] + r[2] * o[2] < 0) r = r.map(x => -x); return r; }); };
  const fromCentre = p => [p[0], (p[1] - cyH) * 0.6, p[2] - cz];
  const hN = normals(hp, hIdx, fromCentre), nN = normals(np, nIdx, p => [p[0], 0, p[2] - cz - rz * 0.04]), eN = normals(ep, eIdx, p => [p[0], 0, 0]);
  const fN = normals(F, faceTris, () => [0, 0, 1]);

  // ---- photo projections ----
  const views = [{photo: front, R: [1, 0, 0, 0, 1, 0, 0, 0, 1]}];
  const Rf = rotOf(front.mat);
  for (const sp of sides) {
    if (!sp || !sp.lm) continue;
    let cands = [];
    const Rs = rotOf(sp.mat);
    if (Rf && Rs) { const R = mul3(Rs, tr3(Rf)); cands.push(R, tr3(R)); }
    const a = Math.asin(Math.max(-1, Math.min(1, (sp.yaw - front.yaw) * 2.3)));
    for (const g of [a, -a]) cands.push(rotY(g));
    // pick the rotation that best explains the side photo's landmarks
    let best = null;
    for (const R of cands) {
      const fit = fitView(R, sp); if (!best || fit.err < best.err) best = {R, ...fit};
    }
    if (best && best.err < 0.08) views.push({photo: sp, R: best.R, a: best.a, tx: best.tx, ty: best.ty});
  }
  function fitView(R, sp) {
    const m = F.map(p => { const q = app3(R, [p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]]); return [q[0] + pivot[0], q[1] + pivot[1]]; });
    const q = sp.lm.map(l => [l.x * sp.w, l.y * sp.h]);
    let mx = 0, my = 0, qx = 0, qy = 0; const n = Math.min(m.length, q.length);
    for (let i = 0; i < n; i++) { mx += m[i][0]; my += m[i][1]; qx += q[i][0]; qy += q[i][1]; } mx /= n; my /= n; qx /= n; qy /= n;
    let num = 0, den = 0; for (let i = 0; i < n; i++) { num += (m[i][0] - mx) * (q[i][0] - qx) - (m[i][1] - my) * (q[i][1] - qy); den += (m[i][0] - mx) ** 2 + (m[i][1] - my) ** 2; }
    const a = num / den, tx = qx - a * mx, ty = qy + a * my;
    let err = 0; for (let i = 0; i < n; i++) err += Math.hypot(a * m[i][0] + tx - q[i][0], -a * m[i][1] + ty - q[i][1]); err /= n * Math.abs(a);
    return {a, tx, ty, err};
  }
  const inPoly2 = (pl, x, y) => { let c = false; for (let i = 0, j = pl.length - 1; i < pl.length; j = i++) { const [xi, yi] = pl[i], [xj, yj] = pl[j]; if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c; } return c; };
  views.forEach((v, k) => {
    const lm = v.photo.lm, pts = OVAL.map(i => [lm[i].x, lm[i].y]);
    const mx = pts.reduce((a, q) => a + q[0], 0) / pts.length, my = pts.reduce((a, q) => a + q[1], 0) / pts.length;
    const e = k === 0 ? 1.03 : 1.1;
    v.mask = pts.map(([x, y]) => [mx + (x - mx) * e, my + (y - my) * e]);
    v.inner = pts.map(([x, y]) => [mx + (x - mx) * 0.9, my + (y - my) * 0.9]);
    const xs = pts.map(q => q[0]); v.neck = {x0: Math.min(...xs), x1: Math.max(...xs), y: lm[152].y - 0.02};
  });
  const sdist = (pl, x, y) => { let md = 9; for (let i = 0, j = pl.length - 1; i < pl.length; j = i++) { const [ax, ay] = pl[j], [bx, by] = pl[i]; const vx = bx - ax, vy = by - ay; const t = Math.max(0, Math.min(1, ((x - ax) * vx + (y - ay) * vy) / (vx * vx + vy * vy))); md = Math.min(md, Math.hypot(ax + vx * t - x, ay + vy * t - y)); } return inPoly2(pl, x, y) ? md : -md; };
  views.forEach(v => { const xs = v.mask.map(q => q[0]); v.fw = Math.max(...xs) - Math.min(...xs); });
  const visible = (v, u, w) => inPoly2(v.mask, u, w) || (w > v.neck.y && u > v.neck.x0 + (v.neck.x1 - v.neck.x0) * 0.12 && u < v.neck.x1 - (v.neck.x1 - v.neck.x0) * 0.12);
  const project = (p, nrm) => views.map((v, k) => {
    const r = project0(p, nrm, v, k);
    const d = sdist(v.mask, r[0], r[1]) / v.fw;
    let w = sstep(-0.02, k ? 0.18 : 0.1, d);
    const nk = v.neck, nx = (r[0] - nk.x0) / (nk.x1 - nk.x0);
    if (r[1] > nk.y && nx > 0.15 && nx < 0.85) w = Math.max(w, sstep(0.15, 0.3, Math.min(nx, 1 - nx)));
    r[2] *= w; return r;
  });
  const project0 = (p, nrm, v, k) => {
    if (k === 0) return [(p[0] * s + cx) / W, (-p[1] * s + cy) / H, Math.pow(Math.max(0, nrm[2]), 1.6)];
    const q = app3(v.R, [p[0] - pivot[0], p[1] - pivot[1], p[2] - pivot[2]]), n2 = app3(v.R, nrm);
    return [(v.a * (q[0] + pivot[0]) + v.tx) / v.photo.w, (-v.a * (q[1] + pivot[1]) + v.ty) / v.photo.h, Math.pow(Math.max(0, n2[2]), 2.2) * 1.15];
  };

  const skin = sampleColor(front.img, [50, 280, 101, 330, 205, 425].map(i => px(i)));
  let hairC = sampleColor(front.img, [[px(10)[0], px(10)[1] - s * 0.1], [px(109)[0], px(109)[1] - s * 0.08], [px(338)[0], px(338)[1] - s * 0.08]]);
  const lum = c => .3 * c[0] + .59 * c[1] + .11 * c[2];
  if (Math.abs(lum(hairC) - lum(skin)) < 0.12 || lum(hairC) > 0.5) hairC = [0.13, 0.09, 0.07];
  hairC = hairC.map(c => Math.min(1, c * 0.78));

  // hairline per direction
  const Yf = Y10 + 0.075, Yt = eyeY + 0.24, Ysb = eyeY - 0.07, Yn = mouthY - 0.02;
  const hairline = th => { const a = Math.abs(th); if (a < .75) return lerp(Yf, Yt, sstep(0, .75, a)); if (a < 1.45) return lerp(Yt, Ysb, sstep(.75, 1.45, a)); if (a < 1.9) return Ysb; return lerp(Ysb, Yn, sstep(1.9, Math.PI, a)); };
  const headInfo = {top, Y10, eyeY, cyH, cz, rx, ry, rz, inPoly, hairline};

  return {F, faceTris, fN, hp, hIdx, hN, hTheta, hPhi, np, nIdx, nN, ep, eIdx, eN, views, project, skin, hairC, headInfo, s};
}

export function hairField(model, id) {
  const st = HAIR[id] || HAIR['mid-fade'];
  const {hp, hTheta, hN, headInfo: I} = model;
  const L = new Float32Array(hp.length), C = new Float32Array(hp.length * 3);
  const partX = -I.rx * 0.38;
  for (let k = 0; k < hp.length; k++) {
    const [x, y, z] = hp[k], th = hTheta[k];
    const yl = I.hairline(th);
    const mask = Math.cos(th) > 0.2 && y < I.Y10 + 0.03 && I.inPoly(x, y) ? 0 : sstep(yl - 0.005, yl + 0.025, y);
    if (!mask) { L[k] = 0; continue; }
    const hv = Math.max(0, Math.min(1, (y - yl) / (I.top - yl)));
    const sideW = Math.abs(Math.sin(th)), backW = Math.max(0, -Math.cos(th)), frontW = Math.max(0, Math.cos(th));
    let len;
    if (st.fade === 0) len = st.top;
    else {
      const side = lerp(st.low, st.side, sstep(0, Math.max(.05, st.fade * .55), hv));
      len = lerp(side, st.top, sstep(st.fade * .75, Math.min(1, st.fade + .18), hv));
    }
    if (st.crew) len *= lerp(.6, 1.15, frontW);
    if (st.nape) len += st.nape * backW * (1 - sstep(.1, .55, hv)) * sstep(.6, 1, backW);
    if (st.fringe) len += st.fringe * Math.pow(frontW, 3) * sstep(.55, .95, hv);
    const vol = 0.85 + 0.3 * Math.sin(x * 23.1 + z * 17.3) * Math.sin(y * 19.7 - x * 11.2);
    L[k] = len * mask * (len > 0.04 ? vol : 1) * lerp(0.55, 1, sstep(0, 0.08, hv));
    // comb direction (tangent-ish), mixed later with the normal
    let c = [0, 0, 0];
    if (st.comb === 'forward') c = [0, -0.35, 1];
    else if (st.comb === 'back') c = [0, 0.15, -1];
    else if (st.comb === 'side') c = [x > partX ? 1 : -1, -0.2, 0.15];
    else c = [0, 0.4, 0];
    // only comb where hair is long (top), fades stand out from the scalp
    const cw = sstep(.35, .8, hv);
    C[k * 3] = c[0] * cw; C[k * 3 + 1] = c[1] * cw; C[k * 3 + 2] = c[2] * cw;
  }
  return {L, C, curl: st.curl, droop: st.nape ? 1 : (id === 'fringe' ? .6 : .15)};
}

/* ---------------- renderer ---------------- */
export function createHeadViewer(canvas, model) {
  const gl = canvas.getContext('webgl', {antialias: true, premultipliedAlpha: false, preserveDrawingBuffer: true});
  if (!gl) throw new Error('no webgl');
  const NV = Math.min(3, model.views.length);
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
  const prog = (vs, fs) => { const p = gl.createProgram(); gl.attachShader(p, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(p); if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p)); return p; };

  const skinVS = `attribute vec3 p;attribute vec3 n;attribute vec3 t0;attribute vec3 t1;attribute vec3 t2;attribute float hl;
    uniform mat4 M;uniform mat4 PR;varying vec3 vt0;varying vec3 vt1;varying vec3 vt2;varying vec3 vn;varying float vhl;varying vec3 vp;
    void main(){vt0=t0;vt1=t1;vt2=t2;vhl=hl;vn=mat3(M)*n;vp=p;gl_Position=PR*M*vec4(p,1.);}`;
  const skinFS = `precision mediump float;uniform sampler2D s0;uniform sampler2D s1;uniform sampler2D s2;uniform vec3 skin;uniform vec3 hairC;uniform int mode;uniform float hasHair;
    varying vec3 vt0;varying vec3 vt1;varying vec3 vt2;varying vec3 vn;varying float vhl;varying vec3 vp;
    void main(){vec3 N=normalize(vn);
      float dif=max(dot(N,normalize(vec3(-.4,.5,.9))),0.)*.75+max(dot(N,normalize(vec3(.8,.2,-.3))),0.)*.25;
      if(mode==2){vec3 c=vec3(.72,.7,.68)*(.3+.85*dif);gl_FragColor=vec4(c,1.);return;}
      vec3 acc=vec3(0.);float w=0.;
      if(vt0.z>0.001&&vt0.x>0.&&vt0.x<1.&&vt0.y>0.&&vt0.y<1.){acc+=texture2D(s0,vt0.xy).rgb*vt0.z;w+=vt0.z;}
      if(vt1.z>0.001&&vt1.x>0.&&vt1.x<1.&&vt1.y>0.&&vt1.y<1.){acc+=texture2D(s1,vt1.xy).rgb*vt1.z;w+=vt1.z;}
      if(vt2.z>0.001&&vt2.x>0.&&vt2.x<1.&&vt2.y>0.&&vt2.y<1.){acc+=texture2D(s2,vt2.xy).rgb*vt2.z;w+=vt2.z;}
      vec3 tex=w>0.?acc/w:skin; float k=smoothstep(.0,.3,w); vec3 c=mix(skin,tex,k)*(.72+.38*dif);
      float scalp=smoothstep(.006,.035,vhl)*hasHair; c=mix(c,hairC*.42*(.6+.5*dif),scalp*.94);
      float stub=smoothstep(.0,.008,vhl)*(1.-scalp)*hasHair;
      float g=fract(sin(dot(floor(vp.xy*900.+vp.z*400.),vec2(12.9898,78.233)))*43758.5453);
      c=mix(c,mix(c*.62,hairC*.55,.5)+(g-.5)*.05,stub*.85);
      gl_FragColor=vec4(c,1.);}`;
  const hairVS = `attribute vec3 p;attribute vec3 n;attribute float hl;attribute vec3 cd;attribute vec2 uv;
    uniform mat4 M;uniform mat4 PR;uniform float t;uniform float curl;uniform float droop;varying vec2 vuv;varying vec3 vn;varying float vt;varying float vhl;
    float h(vec3 q){return fract(sin(dot(q,vec3(12.9898,78.233,37.719)))*43758.5453);}
    float vn3(vec3 q){vec3 i=floor(q),f=fract(q);f=f*f*(3.-2.*f);
      return mix(mix(mix(h(i),h(i+vec3(1,0,0)),f.x),mix(h(i+vec3(0,1,0)),h(i+vec3(1,1,0)),f.x),f.y),mix(mix(h(i+vec3(0,0,1)),h(i+vec3(1,0,1)),f.x),mix(h(i+vec3(0,1,1)),h(i+vec3(1,1,1)),f.x),f.y),f.z);}
    void main(){vt=t;vhl=hl;vuv=uv;
      vec3 cl=vec3(vn3(p*14.),vn3(p*14.+7.3),vn3(p*14.+3.1))-.5;
      vec3 cl2=vec3(vn3(p*6.),vn3(p*6.+2.7),vn3(p*6.+5.9))-.5;
      vec3 dir=normalize(n+cd*1.6*t+cl*curl*1.8*t+cl2*.6*t);
      vec3 q=p+dir*hl*t+vec3(0.,-1.,0.)*hl*t*t*droop*.8;
      vn=mat3(M)*normalize(n+cd*.6);gl_Position=PR*M*vec4(q,1.);}`;
  const hairFS = `precision mediump float;uniform vec3 hairC;uniform float t;varying vec2 vuv;varying vec3 vn;varying float vt;varying float vhl;
    float h(vec2 q){return fract(sin(dot(q,vec2(12.9898,78.233)))*43758.5453);}
    void main(){ if(vhl<.0015)discard;
      vec2 g=vuv*vec2(1100.,620.);vec2 c=floor(g);vec2 f=fract(g)-.5;float r=h(c);
      if(pow(r,.7)<vt*.95)discard; if(length(f)>.5*(1.-vt*.7))discard;
      vec3 N=normalize(vn);float dif=max(dot(N,normalize(vec3(-.4,.6,.8))),0.);
      vec3 col=hairC*(.22+1.05*vt)*(.7+.5*h(c+3.1));col*=.4+.8*dif;
      col+=vec3(1.,.92,.82)*pow(max(dot(N,normalize(vec3(-.2,.9,.5))),0.),18.)*.14*vt;
      gl_FragColor=vec4(col,1.);}`;
  const lineVS = `attribute vec3 p;uniform mat4 M;uniform mat4 PR;void main(){gl_Position=PR*M*vec4(p,1.);}`;
  const lineFS = `precision mediump float;void main(){gl_FragColor=vec4(.98,.36,0.,.9);}`;
  const P1 = prog(skinVS, skinFS), P2 = prog(hairVS, hairFS), P3 = prog(lineVS, lineFS);

  // textures
  const texs = [0, 1, 2].map(k => {
    const tx = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tx);
    const v = model.views[k];
    if (v) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, v.photo.img);
    else gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([128, 100, 90, 255]));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tx;
  });

  function meshBuffers(P, Nn, I, hl) {
    const n = P.length, pos = new Float32Array(n * 3), nor = new Float32Array(n * 3), t = [0, 1, 2].map(() => new Float32Array(n * 3)), h = new Float32Array(n);
    for (let k = 0; k < n; k++) {
      pos.set(P[k], k * 3); nor.set(Nn[k], k * 3);
      const pr = model.project(P[k], Nn[k]);
      for (let v = 0; v < 3; v++) if (pr[v]) t[v].set(pr[v], k * 3);
      h[k] = hl ? hl[k] : 0;
    }
    const mk = d => { const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, d, gl.STATIC_DRAW); return b; };
    const ib = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(I), gl.STATIC_DRAW);
    return {pos: mk(pos), nor: mk(nor), t: t.map(mk), h: mk(h), hRaw: h, ib, count: I.length, n};
  }
  const head = meshBuffers(model.hp, model.hN, model.hIdx);
  const neck = meshBuffers(model.np, model.nN, model.nIdx);
  const ears = meshBuffers(model.ep, model.eN, model.eIdx);
  const face = meshBuffers(model.F, model.fN, model.faceTris);
  const lines = []; for (let t = 0; t < model.faceTris.length; t += 3) { const [a, b, c] = [model.faceTris[t], model.faceTris[t + 1], model.faceTris[t + 2]]; lines.push(a, b, b, c, c, a); }
  const lineIB = gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineIB); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(lines), gl.STATIC_DRAW);

  // hair buffers
  const uv = new Float32Array(model.hp.length * 2);
  model.hp.forEach((p, k) => { uv[k * 2] = (model.hTheta[k] + Math.PI) / (Math.PI * 2); uv[k * 2 + 1] = (model.hPhi[k] + Math.PI / 2) / Math.PI; });
  const uvB = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, uvB); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
  const hlB = gl.createBuffer(), cdB = gl.createBuffer(), hairIB = gl.createBuffer();
  let hairCount = 0, hairParams = {curl: 0, droop: 0}, hasHair = 1;
  function setStyle(id) {
    const f = hairField(model, id);
    gl.bindBuffer(gl.ARRAY_BUFFER, hlB); gl.bufferData(gl.ARRAY_BUFFER, f.L, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, cdB); gl.bufferData(gl.ARRAY_BUFFER, f.C, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, head.h); gl.bufferData(gl.ARRAY_BUFFER, f.L, gl.STATIC_DRAW);
    const I = [], src = model.hIdx; for (let t = 0; t < src.length; t += 3) if (f.L[src[t]] > 0.0008 || f.L[src[t + 1]] > 0.0008 || f.L[src[t + 2]] > 0.0008) I.push(src[t], src[t + 1], src[t + 2]);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, hairIB); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(I), gl.STATIC_DRAW);
    hairCount = I.length; hairParams = {curl: f.curl, droop: f.droop};
  }

  const attr = (p, name, buf, size) => { const l = gl.getAttribLocation(p, name); if (l < 0) return; gl.bindBuffer(gl.ARRAY_BUFFER, buf); gl.enableVertexAttribArray(l); gl.vertexAttribPointer(l, size, gl.FLOAT, false, 0, 0); };
  const off = p => { const n = gl.getProgramParameter(p, gl.ACTIVE_ATTRIBUTES); for (let i = 0; i < 8; i++) gl.disableVertexAttribArray(i); };

  const st = {yaw: 0.5, pitch: 0.05, auto: true, mode: 0, hair: true, drag: null, t0: performance.now(), raf: 0, zoom: 1};
  canvas.onpointerdown = e => { canvas.setPointerCapture(e.pointerId); st.drag = {x: e.clientX, y: e.clientY, yaw: st.yaw, pitch: st.pitch}; st.auto = false; };
  canvas.onpointermove = e => { if (!st.drag) return; st.yaw = st.drag.yaw + (e.clientX - st.drag.x) / 140; st.pitch = Math.max(-.7, Math.min(.7, st.drag.pitch + (e.clientY - st.drag.y) / 220)); };
  canvas.onpointerup = canvas.onpointercancel = () => { st.drag = null; };
  canvas.onwheel = e => { e.preventDefault(); st.zoom = Math.max(.7, Math.min(1.6, st.zoom * (e.deltaY > 0 ? .94 : 1.06))); };

  const m4 = (R, T) => [R[0], R[3], R[6], 0, R[1], R[4], R[7], 0, R[2], R[5], R[8], 0, T[0], T[1], T[2], 1];
  function frame(time) {
    const dpr = devicePixelRatio || 1, w = Math.round(canvas.clientWidth * dpr), h = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
    gl.viewport(0, 0, w, h); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
    if (st.auto && !matchMedia('(prefers-reduced-motion: reduce)').matches) st.yaw = 0.5 + (time - st.t0) / 1000 * 0.45;
    const I = model.headInfo;
    const R = mul3(rotX(st.pitch), rotY(st.yaw));
    const c = app3(R, [0, -I.cyH + 0.06, -I.cz]);
    const dist = 3.4 / st.zoom;
    const M = new Float32Array(m4(R, [c[0], c[1], c[2] - dist]));
    const fov = 0.6, a = w / h, ft = 1 / Math.tan(fov / 2), nr = .1, fr = 30;
    const PR = new Float32Array([ft / a, 0, 0, 0, 0, ft, 0, 0, 0, 0, (fr + nr) / (nr - fr), -1, 0, 0, 2 * fr * nr / (nr - fr), 0]);
    // skin pass
    gl.useProgram(P1);
    gl.uniformMatrix4fv(gl.getUniformLocation(P1, 'M'), false, M); gl.uniformMatrix4fv(gl.getUniformLocation(P1, 'PR'), false, PR);
    gl.uniform3fv(gl.getUniformLocation(P1, 'skin'), model.skin); gl.uniform3fv(gl.getUniformLocation(P1, 'hairC'), model.hairC);
    gl.uniform1i(gl.getUniformLocation(P1, 'mode'), st.mode); gl.uniform1f(gl.getUniformLocation(P1, 'hasHair'), st.hair ? 1 : 0);
    [0, 1, 2].forEach(k => { gl.activeTexture(gl.TEXTURE0 + k); gl.bindTexture(gl.TEXTURE_2D, texs[k]); gl.uniform1i(gl.getUniformLocation(P1, 's' + k), k); });
    for (const m of [head, neck, ears]) {
      off(P1); attr(P1, 'p', m.pos, 3); attr(P1, 'n', m.nor, 3); attr(P1, 't0', m.t[0], 3); attr(P1, 't1', m.t[1], 3); attr(P1, 't2', m.t[2], 3); attr(P1, 'hl', m.h, 1);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, m.ib); gl.drawElements(gl.TRIANGLES, m.count, gl.UNSIGNED_SHORT, 0);
    }
    gl.enable(gl.POLYGON_OFFSET_FILL); gl.polygonOffset(-1, -1);
    off(P1); attr(P1, 'p', face.pos, 3); attr(P1, 'n', face.nor, 3); attr(P1, 't0', face.t[0], 3); attr(P1, 't1', face.t[1], 3); attr(P1, 't2', face.t[2], 3); attr(P1, 'hl', face.h, 1);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, face.ib); gl.drawElements(gl.TRIANGLES, face.count, gl.UNSIGNED_SHORT, 0);
    gl.disable(gl.POLYGON_OFFSET_FILL);
    // hair shells
    if (st.hair && hairCount) {
      gl.useProgram(P2);
      gl.uniformMatrix4fv(gl.getUniformLocation(P2, 'M'), false, M); gl.uniformMatrix4fv(gl.getUniformLocation(P2, 'PR'), false, PR);
      gl.uniform3fv(gl.getUniformLocation(P2, 'hairC'), model.hairC);
      gl.uniform1f(gl.getUniformLocation(P2, 'curl'), hairParams.curl); gl.uniform1f(gl.getUniformLocation(P2, 'droop'), hairParams.droop);
      off(P2); attr(P2, 'p', head.pos, 3); attr(P2, 'n', head.nor, 3); attr(P2, 'hl', hlB, 1); attr(P2, 'cd', cdB, 3); attr(P2, 'uv', uvB, 2);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, hairIB);
      const NS = 36, tl = gl.getUniformLocation(P2, 't');
      for (let i = 1; i <= NS; i++) { gl.uniform1f(tl, i / NS); gl.drawElements(gl.TRIANGLES, hairCount, gl.UNSIGNED_SHORT, 0); }
    }
    if (st.mode === 1) {
      gl.useProgram(P3); gl.uniformMatrix4fv(gl.getUniformLocation(P3, 'M'), false, M); gl.uniformMatrix4fv(gl.getUniformLocation(P3, 'PR'), false, PR);
      off(P3); attr(P3, 'p', face.pos, 3); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, lineIB); gl.disable(gl.DEPTH_TEST); gl.drawElements(gl.LINES, lines.length, gl.UNSIGNED_SHORT, 0);
    }
    st.raf = requestAnimationFrame(frame);
  }
  return {
    setStyle,
    setMode(m) { st.mode = m; },
    setHair(b) { st.hair = b; },
    start() { if (!st.raf) st.raf = requestAnimationFrame(frame); },
    stop() { cancelAnimationFrame(st.raf); st.raf = 0; },
    resetView() { st.auto = true; st.t0 = performance.now(); },
    views: model.views.length
  };
}
