// Perlin noise, for pushing the slicing surface off its plane. This is Ken Perlin's improved
// noise (2002): a smooth random function of three coordinates, zero at every point of a
// whole-number lattice and swelling up or down in between.

import { LruCache } from "./lru";
import { OptionError, refuseNonFinite, type Sweep, type TX } from "./planner";
import { copysign, floorMod, pyRepr, roundHalfEven } from "./pymath";
import type { NoiseGrid } from "./types";

// No value of noise3 is further from 0 than this. Each lattice corner adds g.d, the dot
// product of its gradient g (one of 12 vectors like (1, 1, 0)) with the offset d from that
// corner, which is at most the two largest components of |d| added together. Weighting those
// by how much each corner counts and maximising over the cell gives 1.03635.
export const NOISE_BOUND = 1.0364;

// Working the noise out at every pixel is slow, and it changes little from one pixel to the
// next, so it's worked out at nodes NOISE_STEPS to every `size` pixels and blended smoothly in
// between (Catmull-Rom). That comes within about half a percent of the amplitude of the noise
// itself. Nodes under 2 pixels apart would save nothing, so then there's one at every pixel,
// which gives the noise exactly.
export const NOISE_STEPS = 8;

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10);
}

function lerp(t: number, a: number, b: number): number {
  return a + t * (b - a);
}

/**
 * The dot product of (x, y, z) with one of the 12 vectors from a cube's centre to the middles
 * of its edges, picked by the hash h.
 */
function grad(h: number, x: number, y: number, z: number): number {
  h &= 15;
  const u = h < 8 ? x : y;
  const v = h < 4 ? y : h === 12 || h === 14 ? x : z;
  return ((h & 1) === 0 ? u : -u) + ((h & 2) === 0 ? v : -v);
}

/**
 * The noise at (x, y, z) for a permutation table. With a period (a whole number of cells, or
 * 0 for none) the lattice wraps round along z, so the noise repeats every `period` in z.
 */
export function noise3(perm: ArrayLike<number>, x: number, y: number, z: number,
                       period: number): number {
  const fx = Math.floor(x), fy = Math.floor(y), fz = Math.floor(z);
  const X = fx & 255, Y = fy & 255;  // & works mod 2^32, so on negatives too
  // The lattice planes either side in z.
  let Z0 = fz, Z1 = fz + 1;
  if (period) {
    Z0 = floorMod(Z0, period);
    Z1 = floorMod(Z1, period);
  }
  Z0 &= 255;
  Z1 &= 255;
  x -= fx;
  y -= fy;
  z -= fz;
  const u = fade(x), v = fade(y), w = fade(z);
  // Hash the 8 corners of the lattice cell around the point.
  const a = perm[X] + Y, b = perm[X + 1] + Y;
  const aa = perm[a], ab = perm[a + 1], ba = perm[b], bb = perm[b + 1];
  return lerp(w, lerp(v, lerp(u, grad(perm[aa + Z0], x, y, z),
                                  grad(perm[ba + Z0], x - 1, y, z)),
                         lerp(u, grad(perm[ab + Z0], x, y - 1, z),
                                  grad(perm[bb + Z0], x - 1, y - 1, z))),
                 lerp(v, lerp(u, grad(perm[aa + Z1], x, y, z - 1),
                                  grad(perm[ba + Z1], x - 1, y, z - 1)),
                         lerp(u, grad(perm[ab + Z1], x, y - 1, z - 1),
                                  grad(perm[bb + Z1], x - 1, y - 1, z - 1))));
}

// The lattice hash is numpy's default_rng(seed).permutation(256). default_rng spreads the seed
// through a SeedSequence into the 128-bit state and increment of a PCG64 generator, and the
// shuffle swaps each place from the last down with one drawn by random_interval.

const INIT_A = 0x43b0d7e5, MULT_A = 0x931e8875, INIT_B = 0x8b51f9dd, MULT_B = 0x58f38ded;
const MIX_MULT_L = 0xca01f9dd, MIX_MULT_R = 0x4973f715;

/** SeedSequence(seed).generate_state(4, np.uint64): four 64-bit words. */
function seedState(seed: number): bigint[] {
  const entropy: number[] = [];  // the seed in 32-bit words, lowest first
  let rest = BigInt(seed);
  do {
    entropy.push(Number(rest & 0xffffffffn));
    rest >>= 32n;
  } while (rest > 0n);

  let hashConst = INIT_A;
  const hashmix = (value: number): number => {
    value = (value ^ hashConst) >>> 0;
    hashConst = Math.imul(hashConst, MULT_A) >>> 0;
    value = Math.imul(value, hashConst) >>> 0;
    return (value ^ (value >>> 16)) >>> 0;
  };
  const mix = (x: number, y: number): number => {
    const result = (Math.imul(MIX_MULT_L, x) - Math.imul(MIX_MULT_R, y)) >>> 0;
    return (result ^ (result >>> 16)) >>> 0;
  };
  const pool = [0, 1, 2, 3].map((i) => hashmix(i < entropy.length ? entropy[i] : 0));
  for (let src = 0; src < 4; src++) {
    for (let dst = 0; dst < 4; dst++) {
      if (src !== dst) pool[dst] = mix(pool[dst], hashmix(pool[src]));
    }
  }
  for (let src = 4; src < entropy.length; src++) {
    for (let dst = 0; dst < 4; dst++) pool[dst] = mix(pool[dst], hashmix(entropy[src]));
  }

  const words: number[] = [];
  hashConst = INIT_B;
  for (let i = 0; i < 8; i++) {
    let value = (pool[i % 4] ^ hashConst) >>> 0;
    hashConst = Math.imul(hashConst, MULT_B) >>> 0;
    value = Math.imul(value, hashConst) >>> 0;
    words.push((value ^ (value >>> 16)) >>> 0);
  }
  // Pairs of 32-bit words, little-endian.
  return [0, 1, 2, 3].map((i) => BigInt(words[2 * i]) | (BigInt(words[2 * i + 1]) << 32n));
}

const MASK64 = (1n << 64n) - 1n;
const MASK128 = (1n << 128n) - 1n;
const PCG_MULTIPLIER = (2549297995355413924n << 64n) | 4865540595714422341n;

/** numpy's PCG64 (XSL-RR 128/64), with its 32-bit draws taken a half of a 64-bit one at a time. */
class Pcg64 {
  private state = 0n;
  private readonly inc: bigint;
  private spare: number | null = null;  // the high half of the last 64-bit draw, not yet used

  constructor(initState: bigint, initSeq: bigint) {
    this.inc = ((initSeq << 1n) | 1n) & MASK128;
    this.step();
    this.state = (this.state + initState) & MASK128;
    this.step();
  }

  private step(): void {
    this.state = (this.state * PCG_MULTIPLIER + this.inc) & MASK128;
  }

  next64(): bigint {
    this.step();
    const high = this.state >> 64n;
    const value = high ^ (this.state & MASK64);
    const rot = high >> 58n;
    return ((value >> rot) | (value << ((64n - rot) & 63n))) & MASK64;
  }

  next32(): number {
    if (this.spare !== null) {
      const spare = this.spare;
      this.spare = null;
      return spare;
    }
    const next = this.next64();
    this.spare = Number(next >> 32n);
    return Number(next & 0xffffffffn);
  }
}

/** numpy's random_interval: a draw from 0 to max (at most 2^32 - 1), masked and retried. */
function randomInterval(rng: Pcg64, max: number): number {
  if (max === 0) return 0;
  let mask = max;
  for (const shift of [1, 2, 4, 8, 16]) mask = (mask | (mask >>> shift)) >>> 0;
  let value: number;
  do value = (rng.next32() & mask) >>> 0; while (value > max);
  return value;
}

const permutations = new LruCache<number, Int32Array>(16);

/**
 * The shuffled lattice hash for a seed, written out twice so that indexing past 255 wraps
 * around. A new array each call, from a table kept for the 16 seeds used last.
 */
export function permutation(seed: number): Int32Array {
  const cached = permutations.get(seed);
  if (cached) return cached.slice();
  if (!Number.isInteger(seed)) throw new TypeError("seed must be integer");
  if (seed < 0) throw new OptionError("expected non-negative integer");
  const [s0, s1, i0, i1] = seedState(seed);
  const rng = new Pcg64((s0 << 64n) | s1, (i0 << 64n) | i1);
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i >= 1; i--) {
    const j = randomInterval(rng, i);
    [p[i], p[j]] = [p[j], p[i]];
  }
  const table = new Int32Array(512);
  table.set(p);
  table.set(p, 256);
  permutations.set(seed, table);
  return table.slice();
}

interface Nodes { coords: Float64Array; first: Int32Array; weights: Float64Array }

const nodeCache = new LruCache<string, Nodes>(16);  // each frame of the live view asks again

/**
 * Where to work out the noise along an axis of n pixels, and how to blend it back: the nodes'
 * noise coordinates, and for each pixel the first of the 4 nodes around it and their weights
 * (Catmull-Rom, n × 4). Shared between calls, so not to be changed.
 */
function nodes(n: number, size: number): Nodes {
  const key = `${n},${size}`;
  const cached = nodeCache.get(key);
  if (cached) return cached;
  let step = size / NOISE_STEPS;
  if (step < 2) step = 1.0;
  const first = new Int32Array(n);
  const weights = new Float64Array(n * 4);
  for (let i = 0; i < n; i++) {
    const pos = i / step;
    first[i] = Math.floor(pos);
    const t = pos - first[i], t2 = t * t, t3 = t * t * t;
    const w = i * 4;
    weights[w] = (-t3 + 2 * t2 - t) / 2;
    weights[w + 1] = (3 * t3 - 5 * t2 + 2) / 2;
    weights[w + 2] = (-3 * t3 + 4 * t2 + t) / 2;
    weights[w + 3] = (t3 - t2) / 2;
  }
  // One node before the first pixel and two past the last, for the blend.
  const coords = Float64Array.from({ length: first[n - 1] + 4 }, (_, k) => (k - 1) * step / size);
  const layout = { coords, first, weights };
  nodeCache.set(key, layout);
  return layout;
}

export type NoiseDirection = "time" | "perpendicular";

export interface NoiseOptions {
  size?: number;
  speed?: number;
  direction?: NoiseDirection;
  seed?: number;
  period?: number;
}

/**
 * Perlin noise that pushes each point of a sweep's frame off its plane.
 *
 * Points move up to `amplitude` frames (one frame = one pixel), either through "time" or
 * "perpendicular" to the plane, in the x-t plane. `size` is roughly how far apart the bumps
 * are, in pixels. The bumps change as the sweep goes on, `speed` pixels' worth per output
 * frame; 0 keeps one fixed bumpy surface. `seed` picks the pattern. With a `period`, the bumps
 * come back to the same shape every `period` output frames, the speed rounded to the nearest
 * that does so.
 */
export class Noise {
  readonly amplitude: number;
  readonly size: number;
  readonly speed: number;
  readonly direction: NoiseDirection;
  readonly seed: number;
  readonly period: number;

  constructor(amplitude: number, { size = 64.0, speed = 1.0, direction = "time", seed = 0,
                                   period = 0.0 }: NoiseOptions = {}) {
    this.amplitude = amplitude;
    this.size = size;
    this.speed = speed;
    this.direction = direction;
    this.seed = seed;
    this.period = period;
  }

  /**
   * The (t, x) move of a point where the noise is strongest, on a frame whose plane has this
   * (t, x) unit normal.
   */
  push(normal: Readonly<TX>): TX {
    if (this.direction === "time") return [this.amplitude, 0.0];
    if (this.direction === "perpendicular") {
      return [this.amplitude * normal[0], this.amplitude * normal[1]];
    }
    throw new OptionError(`unknown noise direction ${pyRepr(this.direction)}`);
  }

  /**
   * The same noise on a video shrunk by factor in x, y and t. speed is pixels per output
   * frame, and both shrink together, so it stays.
   */
  scaled(factor: number): Noise {
    return this.with({ amplitude: this.amplitude * factor, size: this.size * factor,
                       period: this.period * factor });
  }

  /** The same noise with some fields changed. */
  with(changes: NoiseOptions & { amplitude?: number }): Noise {
    const { amplitude, size, speed, direction, seed, period } = { ...this, ...changes };
    return new Noise(amplitude, { size, speed, direction, seed, period });
  }

  /**
   * The noise at the nodes for output frame f, from -1 to 1, and how to blend it back to
   * each column and row.
   */
  grid(width: number, height: number, f: number): NoiseGrid {
    if (this.size === 0) throw new RangeError("float division by zero");  // as Python's / does
    const across = nodes(width, this.size), down = nodes(height, this.size);
    let z = f * this.speed / this.size, cells = 0;
    if (this.period !== 0 && this.speed !== 0) {
      // A whole number of lattice cells per period, for the noise to wrap round in. At least
      // 2: in a ring of 1 the planes either side of a cell are the same, so the bumps barely
      // change.
      cells = Math.max(
        2, roundHalfEven(refuseNonFinite(this.period * Math.abs(this.speed) / this.size)));
      z = copysign(f * cells / this.period, this.speed);
    }
    const perm = permutation(this.seed);
    const nodeRows = down.coords.length, nodeCols = across.coords.length;
    const values = new Float64Array(nodeRows * nodeCols);
    for (let i = 0; i < nodeRows; i++) {
      for (let j = 0; j < nodeCols; j++) {
        const value = noise3(perm, across.coords[j], down.coords[i], z, cells);
        values[i * nodeCols + j] = value / NOISE_BOUND;
      }
    }
    // The blend arrays are copied, so that a caller can't change the cached ones.
    return { nodes: values, nodeRows, nodeCols, cols: across.first.slice(),
             colW: across.weights.slice(), rows: down.first.slice(), rowW: down.weights.slice() };
  }

  /** The noise over output frame f, from -1 to 1, row-major height × width. */
  field(width: number, height: number, f: number): Float64Array {
    const { nodes: values, nodeCols, cols, colW, rows, rowW } = this.grid(width, height, f);
    const out = new Float64Array(width * height);
    const across = new Float64Array(nodeCols);
    for (let row = 0; row < height; row++) {
      // First the 4 rows of nodes around the row into one, then each pixel from the 4 nodes
      // around it, clamped to -1 to 1, which a blend could otherwise overshoot by a hair.
      const r = rows[row] * nodeCols, w = row * 4;
      for (let m = 0; m < nodeCols; m++) {
        across[m] = rowW[w] * values[r + m] + rowW[w + 1] * values[r + nodeCols + m]
          + rowW[w + 2] * values[r + 2 * nodeCols + m] + rowW[w + 3] * values[r + 3 * nodeCols + m];
      }
      for (let j = 0; j < width; j++) {
        const c = cols[j], k = j * 4;
        const v = colW[k] * across[c] + colW[k + 1] * across[c + 1]
          + colW[k + 2] * across[c + 2] + colW[k + 3] * across[c + 3];
        out[row * width + j] = Math.min(1.0, Math.max(-1.0, v));
      }
    }
    return out;
  }
}

/** The noise as the sweep uses it: a loop's comes back round with it. */
export function looping(noise: Noise, sweep: Sweep): Noise {
  return sweep.loop ? noise.with({ period: sweep.frames }) : noise;
}
