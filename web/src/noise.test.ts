import { describe, expect, test } from "vitest";
import {
  looping, MAX_SEED, Noise, NOISE_BOUND, noise3, permutation,
} from "./noise";
import { rotationSweep, shearSweep } from "./planner";

/** The largest difference relative to max(1, |expected|), or Infinity if the lengths differ. */
function worst(got: ArrayLike<number>, expected: ArrayLike<number>): number {
  if (got.length !== expected.length) return Infinity;
  let most = 0;
  for (let i = 0; i < got.length; i++) {
    most = Math.max(most, Math.abs(got[i] - expected[i]) / Math.max(1, Math.abs(expected[i])));
  }
  return most;
}

describe("permutation", () => {
  test("shuffles 0 to 255, written out twice", () => {
    for (const seed of [0, 1, 7, 12345, MAX_SEED]) {
      const table = Array.from(permutation(seed));
      expect(table).toHaveLength(512);
      expect([...table.slice(0, 256)].sort((a, b) => a - b))
        .toEqual(Array.from({ length: 256 }, (_, i) => i));
      expect(table.slice(256)).toEqual(table.slice(0, 256));
    }
  });

  test("is the same for a seed every time, and differs between seeds", () => {
    expect(permutation(3)).toEqual(permutation(3));
    const tables = new Set([0, 1, 2, 3, MAX_SEED].map((seed) => permutation(seed).join()));
    expect(tables.size).toBe(5);
  });
});


/** Row-major height × width noise as rows. */
function rows(values: Float64Array, width: number): number[][] {
  return Array.from({ length: values.length / width },
                    (_, i) => Array.from(values.subarray(i * width, (i + 1) * width)));
}

describe("surface noise", () => {
  test("stays within its amplitude, and comes close to it", () => {
    for (let seed = 0; seed < 4; seed++) {
      const noise = new Noise(1, { size: 4.3, speed: 0.37, seed });
      let most = 0;
      for (let f = 0; f < 10; f++) {
        for (const v of noise.field(200, 150, f)) most = Math.max(most, Math.abs(v));
      }
      expect(most).toBeLessThanOrEqual(1);
      expect(most).toBeGreaterThan(0.8);
    }
  });

  test("is smooth and set by its seed", () => {
    const bumps = new Noise(1, { size: 8 }).field(64, 48, 3);
    expect(bumps).toEqual(new Noise(1, { size: 8 }).field(64, 48, 3));
    const other = new Noise(1, { size: 8, seed: 1 }).field(64, 48, 3);
    expect(worst(bumps, other)).toBeGreaterThan(0.1);
    const grid = rows(bumps, 64);
    for (let i = 0; i < 48; i++) {
      for (let j = 0; j < 64; j++) {
        if (i) expect(Math.abs(grid[i][j] - grid[i - 1][j])).toBeLessThan(0.3);
        if (j) expect(Math.abs(grid[i][j] - grid[i][j - 1])).toBeLessThan(0.3);
      }
    }
  });

  test("changes over the sweep unless its speed is zero", () => {
    const still = new Noise(1, { size: 8, speed: 0 }), moving = new Noise(1, { size: 8 });
    expect(still.field(32, 24, 0)).toEqual(still.field(32, 24, 9));
    expect(worst(moving.field(32, 24, 0), moving.field(32, 24, 9))).toBeGreaterThan(0.1);
  });

  test("with a period repeats", () => {
    for (const speed of [0.37, 1.0, -0.6, 3.0]) {
      const noise = new Noise(1, { size: 8, speed, period: 40 });
      const first = noise.field(32, 24, 3);
      expect(worst(first, noise.field(32, 24, 43))).toBeLessThan(1e-9);
      expect(worst(first, noise.field(32, 24, 23))).toBeGreaterThan(0.1);
      // and runs smoothly from the end of one period into the next.
      const step = worst(noise.field(32, 24, 1), noise.field(32, 24, 0));
      expect(worst(noise.field(32, 24, 39), noise.field(32, 24, 0))).toBeLessThan(2 * step);
    }
    // The speed is rounded to make that work, but never down to standing still.
    const slow = new Noise(1, { size: 64, speed: 0.1, period: 40 });
    expect(worst(slow.field(32, 24, 0), slow.field(32, 24, 20))).toBeGreaterThan(0.01);
    expect(new Noise(1, { size: 8, speed: 0, period: 40 }).field(32, 24, 7))
      .toEqual(new Noise(1, { size: 8, speed: 0 }).field(32, 24, 7));
  });

  test("for a preview is the full noise at half size", () => {
    for (const size of [64, 3]) {
      for (const period of [0, 30]) {
        const full = new Noise(1, { size, speed: 0.75, seed: 2, period });
        const half = full.scaled(0.5);
        for (const f of [0, 3, 10]) {
          const big = rows(full.field(160, 120, 2 * f), 160);
          const everyOther = big.filter((_, i) => i % 2 === 0)
            .flatMap((row) => row.filter((_, j) => j % 2 === 0));
          expect(Array.from(half.field(80, 60, f))).toEqual(everyOther);
        }
      }
    }
  });

  test("under 16 pixels apart is worked out at every pixel", () => {
    for (const size of [3, 15.9]) {
      const noise = new Noise(1, { size, speed: 0.6, seed: 1 });
      const perm = permutation(1);
      const exact = Array.from({ length: 80 * 100 }, (_, k) => noise3(
        perm, (k % 100) / size, Math.floor(k / 100) / size, 4 * 0.6 / size, 0) / NOISE_BOUND);
      expect(Array.from(noise.field(100, 80, 4))).toEqual(exact);
    }
  });

  test("pushes through time, or perpendicular to the plane", () => {
    expect(new Noise(2).push([0.6, -0.8])).toEqual([2, 0]);
    expect(new Noise(2, { direction: "perpendicular" }).push([0.6, -0.8])).toEqual([1.2, -1.6]);
  });

  test("scales, and changes a field at a time", () => {
    const noise = new Noise(2, { size: 32, speed: 0.5, direction: "perpendicular", seed: 3,
                                 period: 40 });
    expect({ ...noise.scaled(0.5) }).toEqual({ ...noise, amplitude: 1, size: 16, period: 20 });
    expect({ ...noise.with({ seed: 9 }) }).toEqual({ ...noise, seed: 9 });
    expect(noise.with({ seed: 9 })).toBeInstanceOf(Noise);
    // A field given as undefined stays as it was.
    expect({ ...noise.with({ size: undefined, seed: 9, amplitude: undefined }) })
      .toEqual({ ...noise, seed: 9 });
  });

  test("comes back round with a loop", () => {
    const noise = new Noise(1.5, { size: 3, speed: 0.6, seed: 1 });
    const loop = rotationSweep(7, 11, 30, { noise, loop: true });
    expect(looping(noise, loop).period).toBe(7);
    expect(looping(noise, shearSweep(7, 11, 30, { noise }))).toBe(noise);
  });



  test("permutation and grid give arrays of their own, not the cache's", () => {
    const table = permutation(5);
    table[0] = -1;
    expect(permutation(5)[0]).not.toBe(-1);
    const noise = new Noise(1, { size: 8 });
    const grid = noise.grid(20, 10, 0);
    grid.colW[0] = grid.rowW[0] = 99;
    grid.cols[0] = grid.rows[0] = 99;
    const again = noise.grid(20, 10, 0);
    expect([again.colW[0], again.rowW[0], again.cols[0], again.rows[0]]).not.toContain(99);
  });
});
