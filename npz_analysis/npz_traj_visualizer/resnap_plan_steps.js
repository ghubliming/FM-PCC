#!/usr/bin/env node
/*
 * resnap_plan_steps.js — STOPGAP for existing scene.json/viewer HTML produced BEFORE the
 * fan->executed off-by-one fix (npz_traj_export.py::_anchor_index, fix_2).
 *
 * The Docker container has no Python, so we cannot re-run the exporter here. This mirrors the
 * exporter's _anchor_index in JS: it snaps every fan's plan_step to the executed index where the
 * fan's h=0 waypoint actually sits (search ±2 around the stored nominal step, argmin distance to
 * an executed sample). That corrects the avoiding/imf eval's post-step-obs recording offset
 * (fan drawn one dot ahead of its executed dot) with no effect on already-aligned uav scenes.
 *
 * It rewrites ONLY plan_steps (x-alignment). The per-step analytic VALUES (track_err in particular)
 * keep their pre-fix magnitude — for fully consistent analytics, re-export via the fixed
 * npz_traj_export.py on the cluster. Use this only to eyeball the fan-connection fix now.
 *
 *   node resnap_plan_steps.js <scene.json | viewer_*.html> [more...]
 */
const fs = require('fs');

function dist(a, b) { let s = 0; for (let d = 0; d < a.length; d++) { const x = (a[d] ?? 0) - (b[d] ?? 0); s += x * x; } return Math.sqrt(s); }
function median(xs) { if (!xs.length) return NaN; const s = xs.slice().sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; }

const CLEAN_TOL = 1e-2;   // ‖h0 - executed‖ below this ⇒ the fan's h=0 lies ON the executed path

// mean-of-candidates h=0 of one fan
function fanH0(fan) { const D = fan[0][0].length; const h0 = new Array(D).fill(0); for (const c of fan) for (let d = 0; d < D; d++) h0[d] += c[0][d]; for (let d = 0; d < D; d++) h0[d] /= fan.length; return h0; }

// nearest executed sample to h0 within ±2 of guess; ties (to 6dp) break toward guess. -> [idx, residual]
function nearestExecuted(h0, exec, guess) {
  const lo = Math.max(0, guess - 2), hi = Math.min(exec.length - 1, guess + 2);
  let best = guess, bestKey = [Infinity, Infinity];
  for (let j = lo; j <= hi; j++) { const p = exec[j]; if (!p) continue; const key = [Math.round(dist(p, h0) * 1e6) / 1e6, Math.abs(j - guess)]; if (key[0] < bestKey[0] || (key[0] === bestKey[0] && key[1] < bestKey[1])) { bestKey = key; best = j; } }
  return [best, exec[best] ? dist(exec[best], h0) : Infinity];
}

// mirror of npz_traj_export.py::_recording_offset — one structural offset per trial from clean fans.
function recordingOffset(guesses, exec, h0s) {
  const clean = []; let offPath = 0;
  for (let i = 0; i < guesses.length; i++) {
    const [idx, rd] = nearestExecuted(h0s[i], exec, guesses[i]);
    if (rd <= CLEAN_TOL) clean.push(guesses[i] - idx); else offPath++;
  }
  const frac = offPath / Math.max(1, guesses.length);
  if (!clean.length) return [0, frac];
  const cnt = {}; for (const o of clean) cnt[o] = (cnt[o] || 0) + 1;
  const mode = +Object.keys(cnt).reduce((a, b) => cnt[b] > cnt[a] ? b : a);
  return [mode, frac];
}

function resnapScene(scene) {
  const report = [];
  for (const vname of Object.keys(scene.variants || {})) {
    for (const tr of scene.variants[vname].trials || []) {
      const exec = tr.executed || [];
      const before = (tr.plan_steps || []).slice();
      const guesses = tr.plan_steps.slice();
      const h0s = tr.plans.map(fan => (fan && fan.length) ? fanH0(fan) : null);
      const valid = guesses.map((g, i) => [g, h0s[i]]).filter(x => x[1]);
      const [offset, frac] = recordingOffset(valid.map(x => x[0]), exec, valid.map(x => x[1]));
      tr.plan_steps = guesses.map(g => Math.max(0, g - offset));
      report.push({ v: vname, t: tr.trial, offset, offPathFrac: frac, before: before.slice(0, 6), after: tr.plan_steps.slice(0, 6) });
    }
  }
  return report;
}

for (const path of process.argv.slice(2)) {
  const isHtml = path.endsWith('.html');
  let raw = fs.readFileSync(path, 'utf8');
  let scene, prefix = '', suffix = '';
  if (isHtml) {
    const m = raw.match(/window\.SCENE_DATA=(\{[\s\S]*?\});\/\*__END__\*\//);
    if (!m) { console.error(`  ! ${path}: no embedded SCENE_DATA marker — skipped`); continue; }
    scene = JSON.parse(m[1]);
  } else {
    scene = JSON.parse(raw);
  }
  const report = resnapScene(scene);
  const payload = JSON.stringify(scene);
  if (isHtml) {
    raw = raw.replace(/window\.SCENE_DATA=\{[\s\S]*?\};\/\*__END__\*\//, 'window.SCENE_DATA=' + payload + ';/*__END__*/');
    fs.writeFileSync(path, raw);
  } else {
    fs.writeFileSync(path, payload);
  }
  console.log(`\n${path}`);
  for (const r of report) console.log(`  ${r.v}[t${r.t}] offset=${r.offset} off_path_frac=${r.offPathFrac.toFixed(2)} plan_steps ${JSON.stringify(r.before)}.. -> ${JSON.stringify(r.after)}..`);
}
