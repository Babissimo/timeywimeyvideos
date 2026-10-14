import { describe, expect, test } from "vitest";
import { looping, MAX_SEED, Noise, permutation } from "./noise";
import { floorMod } from "./numbers";
import { rotationSweep, shearSweep } from "./planner";

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

describe("surface noise", () => {
  test("moves through its third axis with the sweep, unless its speed is zero", () => {
    expect(new Noise(1, { size: 8, speed: 0.5 }).depth(12)).toEqual({ z: 0.75, ring: 256 });
    expect(new Noise(1, { size: 8, speed: 0 }).depth(12)).toEqual({ z: 0, ring: 256 });
    // Taken round the 256 cells after which the lattice repeats.
    expect(new Noise(1, { size: 2, speed: -1 }).depth(3)).toEqual({ z: 254.5, ring: 256 });
  });

  test("with a period comes back round, through a whole number of cells", () => {
    for (const speed of [0.37, 1.0, -0.6, 3.0]) {
      const noise = new Noise(1, { size: 8, speed, period: 40 });
      const { z, ring } = noise.depth(3);
      expect(Number.isInteger(ring)).toBe(true);
      expect(noise.depth(43).z).toBeCloseTo(z, 9);
      expect(noise.depth(43).ring).toBe(ring);
      // and runs on from the end of one period into the next as from one frame to the next.
      const step = (from: number, to: number) => floorMod(noise.depth(to).z - noise.depth(from).z,
                                                          ring);
      expect(step(39, 40)).toBeCloseTo(step(0, 1), 9);
    }
    // The speed is rounded to make that work, but never down to standing still.
    const slow = new Noise(1, { size: 64, speed: 0.1, period: 40 });
    expect(slow.depth(20)).toEqual({ z: 1, ring: 2 });
    expect(new Noise(1, { size: 8, speed: 0, period: 40 }).depth(7))
      .toEqual(new Noise(1, { size: 8, speed: 0 }).depth(7));
  });

  test("for a preview is the full noise at half size", () => {
    for (const period of [0, 30]) {
      const full = new Noise(1, { size: 64, speed: 0.75, seed: 2, period });
      const half = full.scaled(0.5);
      expect(half.size).toBe(full.size / 2);
      for (const f of [0, 3, 10]) expect(half.depth(f)).toEqual(full.depth(2 * f));
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
});
