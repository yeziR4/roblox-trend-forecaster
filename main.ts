// main.ts — roblox-trending-forecast-01: empirical rank-kernel forecaster.
// Learns how the Top Trending chart's places move per hour from consecutive reads of the
// history (including exits and the places new games enter at), tilts each game's transition
// toward its recent rank velocity, evolves the chart over the scored window's hours, and
// averages the place distributions over the horizon. Standard library, deterministic.
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

interface Snapshot {
  at: string;
  ok: boolean;
  top_trending: Array<{ rank: number; universe_id: number; name: string; playing: number | null }> | null;
}

interface Task {
  history_until: string;
  window: { from: string; until: string; slots: number };
  places: number;
  output: string;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as Task;
const PLACES = task.places;

const snaps: Snapshot[] = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8')
  .split('\n')
  .filter((l) => l !== '')
  .map((l) => JSON.parse(l) as Snapshot)
  .filter((s) => s.ok && s.top_trending !== null && s.top_trending.length > 0)
  .sort((a, b) => a.at.localeCompare(b.at));

const at = (s: string): number => Date.parse(s);
const HOUR = 3_600_000;

// Per-hour transition kernel over places 1..PLACES plus an exit state (index PLACES).
const K: number[][] = Array.from({ length: PLACES }, () => new Array<number>(PLACES + 1).fill(0));
const entryRate: number[] = new Array<number>(PLACES + 1).fill(0); // index PLACES unused
let pairHours = 0;

function chartOf(s: Snapshot): number[] {
  const ids: number[] = [];
  for (const g of s.top_trending!) {
    if (g.rank >= 1 && g.rank <= PLACES) ids[g.rank - 1] = g.universe_id;
  }
  for (let r = 0; r < PLACES; r++) if (ids[r] === undefined) ids[r] = -1;
  return ids;
}

for (let i = 1; i < snaps.length; i++) {
  const a = chartOf(snaps[i - 1]!);
  const b = chartOf(snaps[i]!);
  const g = (at(snaps[i]!.at) - at(snaps[i - 1]!.at)) / HOUR;
  if (!(g > 0) || g > 48) continue; // ignore absurd gaps
  const w = 1 / Math.max(g, 1);
  const posB = new Map<number, number>();
  for (let r = 0; r < PLACES; r++) if (b[r] !== -1) posB.set(b[r]!, r + 1);
  for (let r = 0; r < PLACES; r++) {
    if (a[r] === -1) continue;
    const to = posB.get(a[r]!);
    if (to === undefined) K[r]![PLACES]! += w; // exit
    else K[r]![to - 1]! += w;
  }
  for (let r = 0; r < PLACES; r++) {
    if (b[r] !== -1 && !a.includes(b[r]!)) entryRate[r]! += w; // a fresh entry at this place
  }
  pairHours += w;
}

// Smooth and normalize the kernel rows, shrinking each toward a sticky prior in proportion to
// how much data the row has (a place with few observed transitions keeps the prior).
const Ks: number[][] = [];
for (let r = 0; r < PLACES; r++) {
  const n = K[r]!.reduce((x, y) => x + y, 0);
  const c = n / (n + 12); // confidence in this row's data
  const row: number[] = [];
  for (let r2 = 0; r2 < PLACES + 1; r2++) {
    const prior = r2 === r ? 0.7 : r2 === PLACES ? 0.03 : Math.abs(r2 - r) === 1 ? 0.12 : Math.abs(r2 - r) === 2 ? 0.015 : 0.0;
    const smoothed = (K[r]![r2]! + 0.15 * prior * 10) / (n + 1.5);
    row.push(c * smoothed + (1 - c) * prior);
  }
  const s = row.reduce((x, y) => x + y, 0);
  Ks.push(row.map((v) => v / s));
}
const entryRateN = (() => {
  const s = entryRate.reduce((x, y) => x + y, 0);
  if (s <= 0) return entryRate.map(() => 0);
  return entryRate.map((v) => v / s);
})();

// The last read's chart and each game's recent velocity (places moved up between the read
// before last and the last; positive = climbing).
const last = snaps[snaps.length - 1]!;
const lastChart = chartOf(last);
const prev = snaps.length >= 2 ? chartOf(snaps[snaps.length - 2]!) : lastChart.slice();
const velocity = new Map<number, number>();
for (let r = 0; r < PLACES; r++) {
  if (lastChart[r] === -1) continue;
  const pi = prev.indexOf(lastChart[r]!);
  velocity.set(lastChart[r]!, pi === -1 ? 0 : pi - r);
}

// Start distributions: the game at place r starts as a point mass at r.
type Dist = number[]; // length PLACES+1, last = exit
const dists = new Map<number, Dist>();
for (let r = 0; r < PLACES; r++) {
  if (lastChart[r] === -1) continue;
  const d = new Array<number>(PLACES + 1).fill(0);
  d[r] = 1;
  dists.set(lastChart[r]!, d);
}

// Momentum tilt: blend a climbing game's distribution with the distribution one seat higher
// (a falling one with one seat lower), in proportion to |velocity| capped, before evolving.
const TILT = Number(process.env.ROBLOX_TILT ?? 0.45);
const blended = new Map<number, Dist>();
for (const [id, d] of dists) {
  const v = velocity.get(id) ?? 0;
  const myPlace = lastChart.indexOf(id) + 1;
  const donorPlace = Math.min(PLACES, Math.max(1, myPlace - Math.sign(v) * Math.min(Math.abs(v), 3)));
  const donorId = lastChart[donorPlace - 1]!;
  if (donorId === -1 || donorId === id) {
    blended.set(id, d);
    continue;
  }
  const donor = new Array<number>(PLACES + 1).fill(0);
  donor[donorPlace - 1] = 1;
  const w = Math.min(0.6, TILT * (Math.abs(v) / 2));
  blended.set(id, d.map((x, i) => (1 - w) * x + w * donor[i]!));
}

// Evolve the chain over the window's slots and average each game's place distribution.
const HORIZON = task.window.slots;
const gameAcc = new Map<number, number[]>();
let state = new Map<number, Dist>();
for (const [id, d] of blended) state.set(id, d.slice());
for (let h = 1; h <= HORIZON; h++) {
  const next = new Map<number, Dist>();
  for (const [id, d] of state) {
    const nd = new Array<number>(PLACES + 1).fill(0);
    for (let r = 0; r < PLACES + 1; r++) {
      const m = d[r]!;
      if (m <= 0) continue;
      if (r === PLACES) nd[PLACES]! += m;
      else for (let r2 = 0; r2 < PLACES + 1; r2++) nd[r2]! += m * Ks[r]![r2]!;
    }
    next.set(id, nd);
  }
  state = next;
  for (const [id, d] of state) {
    let a = gameAcc.get(id);
    if (a === undefined) {
      a = new Array<number>(PLACES).fill(0);
      gameAcc.set(id, a);
    }
    for (let r = 0; r < PLACES; r++) a[r]! += d[r]! / HORIZON;
  }
}

// Sticky prior: the scored reads arrive every hour and the chart is slow. Sharpen each game's
// horizon-averaged distribution on its peaks (a sharpening exponent), then blend it back toward
// the game's current place.
const STICKY = Number(process.env.ROBLOX_STICKY ?? 0.65);
const GAMMA = Number(process.env.ROBLOX_GAMMA ?? 2.0);
for (const [id, a] of gameAcc) {
  const myPlace = lastChart.indexOf(id);
  if (myPlace === -1) continue;
  let s = 0;
  for (let r = 0; r < PLACES; r++) s += Math.pow(Math.max(a[r]!, 0), GAMMA);
  if (s <= 0) continue;
  for (let r = 0; r < PLACES; r++) a[r] = Math.pow(Math.max(a[r]!, 0), GAMMA) / s;
  a[myPlace] = (1 - STICKY) * a[myPlace]! + STICKY;
}

// Rows: per place, the games with the largest probability.
const rows: Array<{ universe_id: number; place: number; p: number }> = [];
const perPlace: Array<Array<{ id: number; p: number }>> = Array.from({ length: PLACES }, () => []);
for (const [id, a] of gameAcc) {
  for (let r = 0; r < PLACES; r++) {
    if (a[r]! > 0.0001) perPlace[r]!.push({ id, p: a[r]! });
  }
}
for (let r = 0; r < PLACES; r++) {
  perPlace[r]!.sort((a, b) => b.p - a.p || a.id - b.id);
  perPlace[r] = perPlace[r]!.slice(0, 20);
}
// Keep every place's named mass strictly under 1 (a game holds one place at a time; the unnamed
// share is what the scorer spreads). The cap leaves room for floating-point wobble in the sum.
for (let r = 0; r < PLACES; r++) {
  const s = perPlace[r]!.reduce((x, y) => x + y.p, 0);
  const scale = s > 0.999 ? 0.999 / s : 1;
  for (const item of perPlace[r]!) {
    const p = Math.min(0.999, Math.max(0.0002, item.p * scale));
    rows.push({ universe_id: item.id, place: r + 1, p });
  }
}

// top10: greedy by place probability, a game used once.
const top10: number[] = [];
const used = new Set<number>();
for (let r = 0; r < PLACES; r++) {
  let best = -1;
  let bestP = -1;
  for (const item of perPlace[r]!) {
    if (used.has(item.id)) continue;
    if (item.p > bestP) {
      bestP = item.p;
      best = item.id;
    }
  }
  if (best >= 0) {
    top10.push(best);
    used.add(best);
  }
}
// Fill any remaining seats from the last chart's order.
for (let r = 0; r < PLACES && top10.length < PLACES; r++) {
  const id = lastChart[r]!;
  if (id !== -1 && !used.has(id)) {
    top10.push(id);
    used.add(id);
  }
}

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities: rows, top10 })}\n`);
