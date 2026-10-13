import { afterAll, describe, expect, test } from "vitest";
import { blockClip, type Clip, greyClip, rgba, upload } from "../../test/gpu";
import { Noise } from "../noise";
import { floorMod } from "../numbers";
import type { Line, TX } from "../planner";
import { Slicer, type SliceOptions } from "./slicer";
import type { Volume } from "./volume";

const canvas = document.createElement("canvas");
const gl = canvas.getContext("webgl2")!;
const slicer = new Slicer(gl);
afterAll(() => {
  slicer.dispose();
  gl.getExtension("WEBGL_lose_context")?.loseContext();
});

/** A frame along a line, read back. */
function slice(volume: Volume, width: number, line: Line, options?: SliceOptions): Uint8Array {
  slicer.slice(volume, width, line, options);
  return slicer.read();
}

/** Each pixel's level, of a slice of greys, which every colour channel holds. */
function greys(pixels: Uint8Array): number[] {
  const out: number[] = [];
  for (let p = 0; p < pixels.length; p += 4) {
    expect([pixels[p + 1], pixels[p + 2], pixels[p + 3]]).toEqual([pixels[p], pixels[p], 255]);
    out.push(pixels[p]);
  }
  return out;
}

/** Expect each level within `within` of what's expected of it. */
function near(got: ArrayLike<number>, expected: number[], within = 1): void {
  expect(got.length).toBe(expected.length);
  const off = expected.findIndex((v, i) => !(Math.abs(got[i] - v) <= within));
  expect(off, `level ${off}: ${got[off]}, expected ${expected[off]}`).toBe(-1);
}

const range = (n: number) => Array.from({ length: n }, (_, i) => i);
const BLACK = [0, 0, 0, 255];

// Clips of greys two rows high whose levels say where a pixel read: 20 a frame, or 15 a column.
const T = 10, W = 16;
const byTime = greyClip(T, 2, W, (t) => 20 * t);
const byColumn = greyClip(T, 2, W, (_, x) => 15 * x);

/** The levels of a slice two rows high whose columns' levels are `columns`. */
const twoRows = (columns: number[]) => [...columns, ...columns];

/** A test run on volumes holding these clips. */
function using(clips: Clip[], run: (...volumes: Volume[]) => void): () => void {
  return () => {
    const volumes = clips.map((clip) => upload(gl, clip));
    try {
      run(...volumes);
    } finally {
      for (const volume of volumes) volume.dispose();
    }
  };
}

describe("Slicer", () => {
  test("slices each frame of a clip as it was, at 0°", () => {
    const clip = blockClip(5, 6, 9);  // 2×2 blocks of one colour, which YUV 4:2:0 holds
    const volume = upload(gl, clip);
    for (const nearest of [false, true])
      for (let t = 0; t < clip.frames; t++)
        near(slice(volume, 9, { origin: [t, 0], step: [0, 1] }, { nearest }),
             Array.from(rgba(clip, t)), 2);
    volume.dispose();
  });

  test("reads each column along the line, blending the voxels around it",
       using([byTime, byColumn], (time, column) => {
    const t = (j: number) => 1.25 + 0.5 * j, x = (j: number) => 0.35 + 0.7 * j;
    const line: Line = { origin: [t(0), x(0)], step: [0.5, 0.7] };
    near(greys(slice(time, W, line)), twoRows(range(W).map((j) => 20 * t(j))));
    near(greys(slice(column, W, line)), twoRows(range(W).map((j) => 15 * x(j))));
    const closest = (v: number) => Math.floor(v + 0.5);
    near(greys(slice(time, W, line, { nearest: true })),
         twoRows(range(W).map((j) => 20 * closest(t(j)))));
    near(greys(slice(column, W, line, { nearest: true })),
         twoRows(range(W).map((j) => 15 * closest(x(j)))));
  }));

  test("shows black outside the clip, and holds its edges within half a voxel",
       using([byTime, byColumn], (time, column) => {
    // Column j at t = j - 2.5, from 2.5 frames before the clip to 2.5 after it.
    const pixels = slice(time, 15, { origin: [-2.5, 3], step: [1, 0] });
    for (const j of range(15)) {
      const t = j - 2.5, pixel = pixels.subarray(4 * j, 4 * j + 4);
      if (t < -0.5 || t >= T - 0.5) expect(Array.from(pixel), `t = ${t}`).toEqual(BLACK);
      else near(pixel, [...Array(3).fill(20 * Math.min(Math.max(t, 0), T - 1)), 255]);
    }
    const sideways = slice(column, 22, { origin: [4, -3], step: [0, 1] });
    for (const j of range(22)) {
      const x = j - 3, pixel = sideways.subarray(4 * j, 4 * j + 4);
      if (x < 0 || x >= W) expect(Array.from(pixel), `x = ${x}`).toEqual(BLACK);
      else near(pixel, [...Array(3).fill(15 * x), 255]);
    }
  }));

  test("wraps round in time, and in x", using([byTime, byColumn], (time, column) => {
    // A level `per` a voxel, along an axis of n that wraps round, blended at `at`.
    const wrapped = (at: number, n: number, per: number) => {
      const lo = Math.floor(at), f = at - lo;
      return per * ((1 - f) * floorMod(lo, n) + f * floorMod(lo + 1, n));
    };
    // Twice round, from half a voxel before the start, half a voxel a column.
    near(greys(slice(time, 4 * T, { origin: [-0.5, 2], step: [0.5, 0] }, { wrap: true })),
         twoRows(range(4 * T).map((j) => wrapped(j / 2 - 0.5, T, 20))));
    near(greys(slice(column, 4 * W, { origin: [3, -0.5], step: [0, 0.5] }, { wrapX: true })),
         twoRows(range(4 * W).map((j) => wrapped(j / 2 - 0.5, W, 15))));
  }));

  test("wraps round in x between columns that share no chroma", () => {
    for (const width of [11, 10]) {
      const clip = blockClip(3, 4, width);
      const volume = upload(gl, clip);
      // From the second-last column, past the last and round to the second, at t = 1.25.
      const x = (j: number) => width - 2 + j / 8;
      const pixels = slice(volume, 32, { origin: [1.25, x(0)], step: [0, 1 / 8] },
                           { wrapX: true });
      const voxel = (t: number, y: number, x: number, c: number) =>
        clip.rgb[((t * 4 + y) * width + floorMod(x, width)) * 3 + c];
      for (let y = 0; y < 4; y++)
        for (let j = 0; j < 32; j++) {
          const lo = Math.floor(x(j)), fx = x(j) - lo;
          for (let c = 0; c < 3; c++) {
            const at = (t: number) => voxel(t, y, lo, c) * (1 - fx) + voxel(t, y, lo + 1, c) * fx;
            expect(Math.abs(pixels[4 * (32 * y + j) + c] - (at(1) * 0.75 + at(2) * 0.25)),
                   `${width} wide, row ${y}, column ${j}, channel ${c}`).toBeLessThanOrEqual(3);
          }
        }
      volume.dispose();
    }
  });

  test("makes a frame as wide as asked and as high as the volume", using([byTime], (volume) => {
    expect(slice(volume, 3, { origin: [1, 4], step: [1, 1] }).length).toBe(3 * 2 * 4);
  }));

  test("refuses a frame larger than the GPU can draw", using([byTime], (volume) => {
    const size = Math.min(gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
                          ...(gl.getParameter(gl.MAX_VIEWPORT_DIMS) as Int32Array));
    const wide = () => slicer.slice(volume, size + 1, { origin: [0, 0], step: [0, 1] });
    expect(wide).toThrow(RangeError);
    expect(wide).toThrow(`a slice of ${size + 1} × 2 pixels is more than the GPU's limit of `
                         + `${size} each way`);
  }));

  test("has nothing to read or draw before it slices", () => {
    const fresh = new Slicer(gl);
    expect(() => fresh.read()).toThrow("nothing has been sliced yet");
    expect(() => fresh.draw()).toThrow("nothing has been sliced yet");
    fresh.dispose();
  });

  test("draws the last slice upright, leaving no GL error", () => {
    const volume = upload(gl, blockClip(4, 6, 9));
    const sliced = slice(volume, 9, { origin: [3, 0], step: [0, 1] });
    canvas.width = 9;
    canvas.height = 6;
    slicer.draw();
    expect(gl.getError()).toBe(gl.NO_ERROR);
    const drawn = new Uint8Array(9 * 6 * 4);
    gl.readPixels(0, 0, 9, 6, gl.RGBA, gl.UNSIGNED_BYTE, drawn);
    // The canvas's bottom row is its last.
    const row = 9 * 4;
    for (let y = 0; y < 6; y++)
      expect(drawn.subarray(y * row, (y + 1) * row))
        .toEqual(sliced.subarray((5 - y) * row, (6 - y) * row));
    volume.dispose();
  });

  test("slices again after drawing, as the live view does every frame",
       using([byTime], (volume) => {
    canvas.width = W;
    canvas.height = 2;
    for (let t = 0; t < T; t++) {
      near(greys(slice(volume, W, { origin: [t, 0], step: [0, 1] })), Array(2 * W).fill(20 * t));
      slicer.draw();
    }
    expect(gl.getError()).toBe(gl.NO_ERROR);
  }));
});

describe("fades", () => {
  // The first n of an axis `length` long once faded: voxel i is (i + 1) / (n + 1) of itself
  // and the rest of the voxel `length` on, which the fade drops.
  const faded = (level: (i: number) => number, n: number, length: number) => (i: number) =>
    i < n ? (level(i) * (i + 1) + level(i + length) * (n - i)) / (n + 1) : level(i);

  test("blend the clip's last frames into its first", using([byTime], (volume) => {
    const fade: TX = [3, 0], frames = T - 3;
    const level = faded((t) => 20 * t, 3, frames);
    for (let t = 0; t < frames; t++) {
      near(greys(slice(volume, W, { origin: [t, 0], step: [0, 1] }, { wrap: true, fade })),
           Array(2 * W).fill(level(t)));
    }
    expect(level(1)).toBe((20 * 1 + 20 * 8) / 2);  // the middle of the fade, half and half
    // The last frame kept runs on into the first, as faded.
    near(greys(slice(volume, 2, { origin: [frames - 0.5, 0], step: [0.5, 0] },
                     { wrap: true, fade })),
         twoRows([(level(frames - 1) + level(0)) / 2, level(0)]));
  }));

  test("blend the picture's last columns into its first", using([byColumn], (volume) => {
    const fade: TX = [0, 4], columns = W - 4;
    const level = faded((x) => 15 * x, 4, columns);
    near(greys(slice(volume, columns, { origin: [2, 0], step: [0, 1] }, { wrapX: true, fade })),
         twoRows(range(columns).map(level)));
    // The last column kept runs on into the first, as faded.
    near(greys(slice(volume, 1, { origin: [2, columns - 0.5], step: [0, 1] },
                     { wrapX: true, fade })),
         twoRows([(level(columns - 1) + level(0)) / 2]));
  }));

  test("fade the ends and the sides together", () => {
    const volume = upload(gl, greyClip(T, 2, W, (t, x) => 10 * t + 5 * x));
    const fade: TX = [3, 4], [frames, columns] = [T - 3, W - 4];
    // Fading is linear, so each fade fades its own part of the level.
    const inTime = faded((t) => 10 * t, 3, frames), inX = faded((x) => 5 * x, 4, columns);
    for (let t = 0; t < frames; t++) {
      near(greys(slice(volume, columns, { origin: [t, 0], step: [0, 1] },
                       { wrap: true, wrapX: true, fade })),
           twoRows(range(columns).map((x) => inTime(t) + inX(x))));
    }
    volume.dispose();
  });
});

describe("surface noise", () => {
  // A flat frame through the middle of a clip of 10 levels a frame, pushed through time by up
  // to 4 frames: where each pixel lands says how far the noise pushed it.
  const [height, width, middle, push] = [24, 40, 11.5, 4];
  const ramp = greyClip(24, height, width, (t) => 10 * t);

  /** How far the noise pushes each pixel, rows top-first, as a share of the most it can. */
  const bumps = (volume: Volume, surface: Noise, f: number): number[] =>
    greys(slice(volume, volume.width, { origin: [middle, 0], step: [0, 1] },
                { noise: { surface, push: [push, 0], f } }))
      .map((level) => (level / 10 - middle) / push);

  const furthest = (a: number[], b: number[]) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));

  test("bends the frame, the same way for the same seed", using([ramp], (volume) => {
    const surface = new Noise(1, { size: 8, seed: 4 });
    const bumpy = bumps(volume, surface, 3);
    expect(Math.max(...bumpy.map(Math.abs))).toBeGreaterThan(0.3);
    expect(bumps(volume, surface, 3)).toEqual(bumpy);
    expect(furthest(bumps(volume, surface.with({ seed: 5 }), 3), bumpy)).toBeGreaterThan(0.3);
  }));

  test("is Perlin's improved noise", using([ramp], (volume) => {
    // At depth 2.3 of the noise for seed 7, 6.5 pixels a cell, the noise at these pixels
    // over NOISE_BOUND, as a float64 implementation works it out.
    const bumpy = bumps(volume, new Noise(1, { size: 6.5, speed: 2.3 * 6.5, seed: 7 }), 1);
    for (const [j, y, expected] of [[3, 2, -0.258], [10, 5, -0.651], [17, 11, -0.038],
                                    [25, 20, 0.182], [31, 7, 0.281], [38, 23, -0.022],
                                    [12, 16, 0.141], [5, 19, 0.289]])
      expect(Math.abs(bumpy[y * width + j] - expected), `(${j}, ${y})`).toBeLessThan(0.02);
  }));

  test("for a preview is the full noise at half size",
       using([ramp, greyClip(24, height / 2, width / 2, (t) => 10 * t)], (full, half) => {
    const noise = new Noise(1, { size: 8, speed: 0.75, seed: 2 });
    const big = bumps(full, noise, 6), small = bumps(half, noise.scaled(0.5), 3);
    for (let y = 0; y < height / 2; y++)
      for (let j = 0; j < width / 2; j++)
        expect(Math.abs(small[y * width / 2 + j] - big[2 * y * width + 2 * j]), `(${j}, ${y})`)
          .toBeLessThan(0.03);
  }));

  test("never pushes further than it can, and comes close", using([ramp], (volume) => {
    let most = 0;
    for (let seed = 0; seed < 4; seed++)
      for (let f = 0; f < 10; f++)
        for (const v of bumps(volume, new Noise(1, { size: 4.3, speed: 0.37, seed }), f))
          most = Math.max(most, Math.abs(v));
    expect(most).toBeLessThanOrEqual(1 + 0.05 / push);  // half a level either way
    expect(most).toBeGreaterThan(0.8);
  }));

  test("is smooth, and nothing at the lattice's points", using([ramp], (volume) => {
    const size = 8;
    const bumpy = bumps(volume, new Noise(1, { size, speed: 0 }), 5);
    const at = (y: number, j: number) => bumpy[y * width + j];
    for (let y = 0; y < height; y++)
      for (let j = 0; j < width; j++) {
        if (y) expect(Math.abs(at(y, j) - at(y - 1, j))).toBeLessThan(0.3);
        if (j) expect(Math.abs(at(y, j) - at(y, j - 1))).toBeLessThan(0.3);
        if (y % size === 0 && j % size === 0) expect(Math.abs(at(y, j))).toBeLessThan(0.05);
      }
  }));

  test("changes over the sweep unless its speed is 0, and comes back round with a period",
       using([ramp], (volume) => {
    const moving = new Noise(1, { size: 8, speed: 0.6, seed: 2 });
    expect(furthest(bumps(volume, moving, 9), bumps(volume, moving, 0))).toBeGreaterThan(0.3);
    const still = moving.with({ speed: 0 });
    expect(bumps(volume, still, 9)).toEqual(bumps(volume, still, 0));
    const looping = moving.with({ period: 40 });
    expect(furthest(bumps(volume, looping, 43), bumps(volume, looping, 3))).toBeLessThan(0.05);
    expect(furthest(bumps(volume, looping, 23), bumps(volume, looping, 3))).toBeGreaterThan(0.3);
  }));

  test("pushes across the frame as it does through time",
       using([ramp, greyClip(4, height, 3 * width, (_, x) => 2 * x)], (time, columns) => {
    const surface = new Noise(1, { size: 8, seed: 3 });
    // Columns from x = width on, at 2 levels a column.
    const sideways = greys(slice(columns, width, { origin: [1, width], step: [0, 1] },
                                 { noise: { surface, push: [0, push], f: 2 } }))
      .map((level, p) => (level / 2 - width - p % width) / push);
    expect(furthest(sideways, bumps(time, surface, 2))).toBeLessThan(0.15);
  }));
});
