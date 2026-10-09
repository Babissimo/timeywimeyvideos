import { describe, expect, test } from "vitest";
import raw from "../test/fixtures/options.json?raw";
import { Noise } from "./noise";
import {
  endpoints, fullSize, liveNoise, loopFades, plan, Problem, readOptions, type FullSize,
  type OptionValues,
} from "./options";
import { planSweep, PREVIEW_SCALE } from "./planner";
import type { ClipInfo } from "./decode";

interface PyCase {
  values: OptionValues;
  options?: Record<string, unknown> & { fps: [number, number] | null };
  error?: string;
  kind?: string;
}

const cases = JSON.parse(raw) as PyCase[];

function thrown(fn: () => unknown): unknown {
  try { fn(); } catch (error) { return error; }
  return undefined;
}

/** What fn throws, as the page shows it. */
function problem(fn: () => unknown): { message: string; kind: string } {
  const error = thrown(fn);
  expect(error).toBeInstanceOf(Problem);
  const { message, kind } = error as Problem;
  return { message, kind };
}

describe("readOptions matches webapp.read_options", () => {
  for (const { values, options, error, kind } of cases) {
    test(JSON.stringify(values), () => {
      if (error !== undefined) {
        expect(problem(() => readOptions(values))).toEqual({ message: error, kind });
        return;
      }
      const got = readOptions(values);
      const { fps, ...rest } = options!;
      const noise = got.noise && { ...got.noise };
      expect({ ...got, noise, fps: got.fps && [got.fps.num, got.fps.den] })
        .toEqual({ ...rest, fps });
    });
  }
});

describe("readOptions reads frame rates", () => {
  const refused = { message: "Output fps must be a positive number, like 24 or 30000/1001.",
                    kind: "error" };
  const digits = (n: number) => `${"0".repeat(n - 1)}1`;

  test("with at most 4300 digits in each part, as Python's int() reads them", () => {
    for (const fps of [digits(4300), `1/${digits(4300)}`, [...digits(4300)].join("_")])
      expect(readOptions({ fps }).fps).toEqual({ num: 1, den: 1 });
    expect(readOptions({ fps: `1e-${digits(4300)}` }).fps).toEqual({ num: 1, den: 10 });
    for (const fps of [digits(4301), `1/${digits(4301)}`, `0.${digits(4301)}`,
                       `1e-${digits(4301)}`])
      expect(problem(() => readOptions({ fps }))).toEqual(refused);
  });

  test("refusing, without working them out, those a double can't hold", () => {
    // Fraction() takes these, but as doubles their numerators or denominators are infinite.
    for (const fps of ["1e400", "1e-400", `${"9".repeat(400)}/7`, "1e1000000", "1e-1000000",
                       "1e300000000", "0e999999999"])
      expect(problem(() => readOptions({ fps })), fps).toEqual(refused);
  });
});

describe("readOptions reads noise seeds", () => {
  const seed = (noise_seed: unknown) => () => readOptions({ noise: "2", noise_seed });

  test("exactly, up to the largest whole number a double holds", () => {
    for (const given of ["9007199254740991", "9_007_199_254_740_991", Number.MAX_SAFE_INTEGER])
      expect(seed(given)().noise!.seed, String(given)).toBe(Number.MAX_SAFE_INTEGER);
  });

  test("refusing larger ones, which Python's int() reads but a double can't hold", () => {
    const range = { message: "The noise seed must be a whole number from 0 to "
                      + "9007199254740991.", kind: "error" };
    for (const given of ["9007199254740992", "9007199254740993", 2 ** 53, "9".repeat(400)])
      expect(problem(seed(given)), String(given)).toEqual(range);
    // As Python refuses these, with its message.
    const whole = { message: "The noise seed must be a whole number, 0 or more.", kind: "error" };
    for (const given of [`-${"9".repeat(400)}`, "1".repeat(4301)])
      expect(problem(seed(given)), given.slice(0, 10)).toEqual(whole);
  });
});

// The clip test_webapp.py makes, 64×48 at 20 fps for 3 s: at half size, 30 frames 32 pixels
// wide.
const clip: ClipInfo = { width: 64, height: 48, fps: { num: 20, den: 1 }, duration: 3 };
const read = (values: Record<string, string | number>) =>
  readOptions(Object.fromEntries(Object.entries(values).map(([k, v]) => [k, String(v)])));

describe("plan", () => {
  test("too steep for inside says so", () => {
    const opts = read({ angle: 80, inside: 1 });
    expect(problem(() => plan(30, 32, opts, liveNoise(opts)))).toEqual({
      message: "At 80 degrees a 32-pixel-wide frame covers 32 frames of time, but the clip "
        + "has only 30. Angles up to 69.3 degrees fit this clip. Use a longer clip or a "
        + "smaller angle, or choose Let black in or Wrap round at the video's edges.",
      kind: "does_not_fit",
    });
    const noisy = read({ angle: 80, inside: 1, noise: 3 });
    expect(problem(() => plan(30, 32, noisy, liveNoise(noisy))).message).toBe(
      "At 80 degrees a 32-pixel-wide frame covers 32 frames of time, and the noise can push "
      + "it 1.5 frames either way, but the clip has only 30. Angles up to 57 degrees fit this "
      + "clip. Use a longer clip or a smaller angle or less noise, or choose Let black in or "
      + "Wrap round at the video's edges.");
  });

  test("a loop pushed sideways offers to wrap the sides", () => {
    const opts = read({ angle: 20, loop: 1, noise: 8, noise_direction: "perpendicular" });
    expect(problem(() => plan(30, 32, opts, liveNoise(opts)))).toEqual({
      message: "At 20 degrees the noise can push the ends of the frame 1.37 pixels sideways, "
        + "out of the video. Angles of 0 and 29 to 90 degrees fit this clip. Use less noise, "
        + "push it through time, or wrap round the sides too.",
      kind: "sideways",
    });
    const sides = read({ angle: 20, loop: 1, loop_sides: 1, noise: 8,
                         noise_direction: "perpendicular" });
    const sweep = plan(30, 32, sides, liveNoise(sides));
    expect([sweep.width, sweep.frames, sweep.loop, sweep.sides]).toEqual([32, 30, true, true]);
  });

  test("other refusals are plain problems", () => {
    expect(problem(() => plan(30, 32, read({ slice: "shear", angle: 95 })))).toEqual({
      message: "shear needs an angle between -90 and 90 degrees (at 90 the delay would be "
        + "infinite)",
      kind: "error",
    });
  });
});

describe("loopFades", () => {
  test("rounds the fades to frames and columns of the clip as loaded", () => {
    const opts = read({ loop: 1, loop_fade: 0.5, loop_sides: 1, loop_side_fade: 8 });
    expect(loopFades(30, 32, 10, opts, PREVIEW_SCALE)).toEqual([5, 4]);
    expect(loopFades(30, 32, 10, read({ loop: 1, loop_fade: 0.25 }), PREVIEW_SCALE))
      .toEqual([2, 0]);  // halves round to even
    expect(loopFades(30, 32, 10, read({ loop: 1, loop_fade: 0.35 }), PREVIEW_SCALE))
      .toEqual([4, 0]);
  });

  test("refuses fades over half the clip", () => {
    expect(problem(() => loopFades(30, 32, 10, read({ loop: 1, loop_fade: 2 }), PREVIEW_SCALE)))
      .toEqual({ message: "The crossfade at the ends can be at most half the clip, 1.5 s.",
                 kind: "error" });
    const sides = read({ loop: 1, loop_sides: 1, loop_side_fade: 40 });
    expect(problem(() => loopFades(30, 32, 10, sides, PREVIEW_SCALE)).message)
      .toBe("The crossfade at the sides can be at most half the width, 32 pixels.");
    expect(problem(() => loopFades(7, 32, 3, read({ loop: 1, loop_fade: 1.2 }), PREVIEW_SCALE))
      .message).toBe("The crossfade at the ends can be at most half the clip, 1.17 s.");
    const wide = read({ loop: 1, loop_sides: 1, loop_side_fade: 34 });
    expect(problem(() => loopFades(30, 33, 10, wide, PREVIEW_SCALE)).message)
      .toBe("The crossfade at the sides can be at most half the width, 33 pixels.");
  });
});

describe("fullSize", () => {
  test("reports the full-size render", () => {
    expect(fullSize(clip, read({ angle: 20, slice: "shear" })))
      .toEqual({ width: 64, height: 48, frames: 82, seconds: 4.1, memory: 276480 });
    expect(fullSize(clip, read({ angle: 30, fps: "30000/1001", scale: 0.5, start: 1,
                                 duration: 1.5 })))
      .toEqual({ width: 43, height: 24, frames: 42, seconds: 1.4014, memory: 34560 });
    expect(fullSize(clip, read({ angle: 30, start: 5 })))
      .toEqual({ width: 56, height: 48, frames: 33, seconds: 1.65, memory: 4608 });
  });

  test("gives the GPU memory of the clip in YUV 4:2:0, chroma rounded up at odd sizes", () => {
    // 60 frames of 58×43: luma 58 × 43, and Cb and Cr 29 × 22 each.
    expect((fullSize(clip, read({ angle: 0, scale: 0.9 })) as FullSize).memory)
      .toBe(60 * (58 * 43 + 2 * 29 * 22));
  });

  test("plans the full-size noise", () => {
    const noise = new Noise(3, { size: 10, speed: 0.5, direction: "perpendicular", seed: 4 });
    const full = fullSize(clip, read({ angle: 30, noise: 3, noise_size: 10, noise_speed: 0.5,
                                       noise_direction: "perpendicular", noise_seed: 4 }));
    expect(full).toEqual({ width: 85, height: 48, frames: 90, seconds: 4.5, memory: 276480 });
    expect((full as FullSize).frames).toBe(planSweep(60, 64, { angle: 30, noise }).frames);
  });

  test("allows for a loop's fades", () => {
    expect((fullSize(clip, read({ angle: 30, loop: 1, loop_fade: 0.5 })) as FullSize).frames)
      .toBe(50);  // 60 frames less a second's fade
    expect(fullSize(clip, read({ angle: 80, loop: 1, loop_sides: 1, motion: "perpendicular",
                                 loop_side_fade: 8, inside: 1 })))
      .toEqual({ width: 56, height: 48, frames: 341, seconds: 17.05, memory: 276480 });
  });

  test("says why it can't be made", () => {
    expect(fullSize(clip, read({ angle: 80, inside: 1 }))).toEqual({
      error: "At 80 degrees a 64-pixel-wide frame covers 64 frames of time, but the clip has "
        + "only 60. Angles up to 69.4 degrees fit this clip. Use a longer clip or a smaller "
        + "angle, or choose Let black in or Wrap round at the video's edges.",
    });
    expect(fullSize(clip, read({ angle: 30, loop: 1, loop_fade: 2 })))
      .toEqual({ error: "The crossfade at the ends can be at most half the clip, 1.5 s." });
  });

  test("needs the clip's length, or a duration", () => {
    const unknown = { ...clip, duration: null };
    expect(fullSize(unknown, read({ angle: 30 }))).toBeNull();
    expect(fullSize(unknown, read({ angle: 30, duration: 2 })))
      .toEqual({ width: 75, height: 48, frames: 67, seconds: 3.35, memory: 184320 });
  });
});

test("endpoints are where a frame's first and last columns come from", () => {
  const sweep = planSweep(30, 32, { angle: 30 });
  const close = (got: number[], expected: number[]) =>
    got.forEach((v, i) => expect(v).toBeCloseTo(expected[i], 9));
  close(endpoints(sweep, 0), [-13.753520777572, 7.563466520536, 7.246479222428, 43.936533479464]);
  close(endpoints(sweep, 7), [-7.691342951084, 4.063466520536, 13.308657048916, 40.436533479464]);
});

test("the live view's noise is the page's at half size", () => {
  expect(liveNoise(read({ angle: 30 }))).toBeNull();
  const opts = read({ noise: 3, noise_size: 10, noise_seed: 4 });
  expect({ ...liveNoise(opts) }).toEqual({ ...opts.noise!.scaled(PREVIEW_SCALE) });
  expect(liveNoise(opts)!.amplitude).toBe(1.5);
});
