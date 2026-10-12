// Perlin noise, for pushing the slicing surface off its plane. This is Ken Perlin's improved
// noise (2002): a smooth random function of three coordinates, zero at every point of a
// whole-number lattice and swelling up or down in between.

import { LruCache } from "./lru";
import { floorMod } from "./numbers";
import type { Sweep, TX } from "./planner";
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

export const MAX_SEED = 2 ** 32 - 1;

/**
 * The lattice hash for a seed from 0 to MAX_SEED: 0 to 255 shuffled, written out twice so that
 * indexing past 255 wraps round. The shuffle is Fisher-Yates, drawing from mulberry32.
 */
export function permutation(seed: number): Int32Array {
  let state = seed >>> 0;
  const random = (): number => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
  const p = Array.from({ length: 256 }, (_, i) => i);
  for (let i = 255; i >= 1; i--) {
    const j = Math.floor(random() * (i + 1));
    [p[i], p[j]] = [p[j], p[i]];
  }
  const table = new Int32Array(512);
  table.set(p);
  table.set(p, 256);
  return table;
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
    return [this.amplitude * normal[0], this.amplitude * normal[1]];
  }

  /**
   * The same noise on a video shrunk by factor in x, y and t. speed is pixels per output
   * frame, and both shrink together, so it stays.
   */
  scaled(factor: number): Noise {
    return this.with({ amplitude: this.amplitude * factor, size: this.size * factor,
                       period: this.period * factor });
  }

  /** The same noise with some fields changed; a field given as undefined stays as it is. */
  with(changes: NoiseOptions & { amplitude?: number }): Noise {
    const given = Object.entries(changes).filter(([, value]) => value !== undefined);
    const { amplitude, size, speed, direction, seed, period } =
      { ...this, ...Object.fromEntries(given) } as Noise;
    return new Noise(amplitude, { size, speed, direction, seed, period });
  }

  /**
   * The noise at the nodes for output frame f, from -1 to 1, and how to blend it back to
   * each column and row.
   */
  grid(width: number, height: number, f: number): NoiseGrid {
    const across = nodes(width, this.size), down = nodes(height, this.size);
    let z = f * this.speed / this.size, cells = 0;
    if (this.period !== 0 && this.speed !== 0) {
      // A whole number of lattice cells per period, for the noise to wrap round in. At least
      // 2: in a ring of 1 the planes either side of a cell are the same, so the bumps barely
      // change.
      cells = Math.max(2, Math.round(this.period * Math.abs(this.speed) / this.size));
      z = Math.sign(this.speed) * f * cells / this.period;
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
