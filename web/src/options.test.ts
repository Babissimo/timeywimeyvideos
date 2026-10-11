import { describe, expect, test } from "vitest";
import { MAX_SEED, Noise } from "./noise";
import {
  endpoints, fullSize, liveNoise, loopFades, plan, Problem, readOptions, type FullSize,
  type OptionValues,
} from "./options";
import { planSweep, PREVIEW_SCALE } from "./planner";
import type { ClipInfo } from "./decode";

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

const refused = (message: string) => ({ message, kind: "error" });

describe("readOptions", () => {
  test("gives the defaults for empty fields", () => {
    const defaults = { slice: "rotate", angle: 45, inside: false, motion: null, noise: null,
                       scale: 1, start: null, duration: null, fps: null, loop: false,
                       sides: false, loopFade: 0, sideFade: 0 };
    expect(readOptions({})).toEqual(defaults);
    expect(readOptions({ angle: "", scale: " ", start: null, fps: "" })).toEqual(defaults);
  });

  test("reads numbers, and refuses what isn't one or is out of range", () => {
    expect(readOptions({ angle: " 12.5 " }).angle).toBe(12.5);
    expect(readOptions({ angle: "-1e1" }).angle).toBe(-10);
    expect(readOptions({ angle: 30 }).angle).toBe(30);
    expect(problem(() => readOptions({ angle: "steep" })))
      .toEqual(refused("angle must be a number."));
    expect(problem(() => readOptions({ angle: "Infinity" })))
      .toEqual(refused("angle is out of range."));
    const ranges: [string, OptionValues][] = [
      ["scale", { scale: "0" }], ["start", { start: "-1" }], ["duration", { duration: "0" }],
      ["loop_fade", { loop: "1", loop_fade: "-1" }], ["noise", { noise: "-2" }],
    ];
    for (const [name, values] of ranges)
      expect(problem(() => readOptions(values)), name).toEqual(refused(`${name} is out of range.`));
  });

  test("reads flags, a loop overriding inside and the sides wrapping only on a loop", () => {
    for (const on of ["1", "true", "True", true])
      expect(readOptions({ inside: on }).inside, String(on)).toBe(true);
    expect(readOptions({ inside: "" }).inside).toBe(false);
    expect(readOptions({ inside: "1", loop: "1" })).toMatchObject({ inside: false, loop: true });
    expect(readOptions({ loop_sides: "1" }).sides).toBe(false);
    expect(readOptions({ loop: "1", loop_sides: "1" }).sides).toBe(true);
  });

  test("keeps motion only where a frame can move different ways", () => {
    expect(readOptions({ motion: "time" }).motion).toBe(null);
    expect(readOptions({ motion: "time", inside: "1", slice: "shear" }).motion).toBe(null);
    expect(readOptions({ motion: "time", inside: "1" }).motion).toBe("time");
    expect(readOptions({ motion: "perpendicular", loop: "1", loop_sides: "1" }).motion)
      .toBe("perpendicular");
    expect(problem(() => readOptions({ motion: "longest", loop: "1", loop_sides: "1" })))
      .toEqual(refused("A loop round the sides moves through time or perpendicular to the "
                       + "frame: no line on it is the longest."));
    expect(problem(() => readOptions({ motion: "up" })))
      .toEqual(refused('Unknown motion "up".'));
    expect(problem(() => readOptions({ slice: "twist" })))
      .toEqual(refused('Unknown slice "twist".'));
  });

  test("reads the loop's crossfades only where they apply", () => {
    expect(readOptions({ loop_fade: "2", loop_side_fade: "8" }))
      .toMatchObject({ loopFade: 0, sideFade: 0 });
    expect(readOptions({ loop: "1", loop_sides: "1", loop_fade: "2", loop_side_fade: "8" }))
      .toMatchObject({ loopFade: 2, sideFade: 8 });
  });

  test("reads frame rates as whole numbers, decimals or fractions", () => {
    const rates: [string, number, number][] = [
      ["24", 24, 1], ["30000/1001", 30000, 1001], ["29.97", 2997, 100], [" 25 / 2 ", 25, 2],
      ["50/2", 25, 1],
    ];
    for (const [fps, num, den] of rates) expect(readOptions({ fps }).fps, fps).toEqual({ num, den });
    const message = "Output fps must be a positive number, like 24 or 30000/1001.";
    for (const fps of ["0", "-24", "1/0", "24fps", "1e3", "abc", "9".repeat(20)])
      expect(problem(() => readOptions({ fps })), fps).toEqual(refused(message));
  });

  test("reads noise, its seed a whole number up to MAX_SEED", () => {
    expect(readOptions({ noise: "0" }).noise).toBe(null);
    expect({ ...readOptions({ noise: "2" }).noise }).toEqual({ ...new Noise(2) });
    const noise = readOptions({ noise: "2", noise_size: "16", noise_speed: "-0.5",
                                noise_direction: "perpendicular", noise_seed: "7" }).noise;
    expect({ ...noise }).toEqual({ ...new Noise(2, { size: 16, speed: -0.5,
                                                    direction: "perpendicular", seed: 7 }) });
    expect(readOptions({ noise: "2", noise_seed: String(MAX_SEED) }).noise!.seed).toBe(MAX_SEED);
    const seed = refused(`The noise seed must be a whole number from 0 to ${MAX_SEED}.`);
    for (const noise_seed of [String(MAX_SEED + 1), "-1", "1.5", "seven"])
      expect(problem(() => readOptions({ noise: "2", noise_seed })), noise_seed).toEqual(seed);
    expect(problem(() => readOptions({ noise: "2", noise_size: "0" })))
      .toEqual(refused("noise_size is out of range."));
    expect(problem(() => readOptions({ noise: "2", noise_direction: "up" })))
      .toEqual(refused('Unknown noise direction "up".'));
  });
});


// A clip 64×48 at 20 fps for 3 s: 30 frames 32 pixels wide at half size.
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
      .toEqual([3, 0]);
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
