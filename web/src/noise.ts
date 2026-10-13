// Perlin noise, for pushing the slicing surface off its plane. This is Ken Perlin's improved
// noise (2002): a smooth random function of three coordinates, zero at every point of a
// whole-number lattice and swelling up or down in between. The slicer works it out at every
// pixel (gpu/shaders.ts); this is where its settings and lattice hash come from.

import { floorMod } from "./numbers";
import type { Sweep, TX } from "./planner";

// No value of the noise is further from 0 than this. Each lattice corner adds g.d, the dot
// product of its gradient g (one of 12 vectors like (1, 1, 0)) with the offset d from that
// corner, which is at most the two largest components of |d| added together. Weighting those
// by how much each corner counts and maximising over the cell gives 1.03635.
export const NOISE_BOUND = 1.0364;

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
   * Where output frame f lies along the noise's third axis, z in lattice cells, and the ring of
   * cells the noise runs round on that axis: with a period, a whole number of cells for each,
   * and otherwise the 256 after which the lattice repeats. z is reduced to lie within the ring.
   */
  depth(f: number): { z: number; ring: number } {
    if (this.period === 0 || this.speed === 0)
      return { z: floorMod(f * this.speed / this.size, 256), ring: 256 };
    // At least 2: in a ring of 1 the planes either side of a cell are the same, so the bumps
    // barely change.
    const ring = Math.max(2, Math.round(this.period * Math.abs(this.speed) / this.size));
    return { z: floorMod(Math.sign(this.speed) * f * ring / this.period, ring), ring };
  }
}

/** The noise as the sweep uses it: a loop's comes back round with it. */
export function looping(noise: Noise, sweep: Sweep): Noise {
  return sweep.loop ? noise.with({ period: sweep.frames }) : noise;
}
