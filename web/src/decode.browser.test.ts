import { describe, expect, test, vi } from "vitest";
import {
  clipSeconds, load, planLoad, probe, VideoError, type LoadOptions, type LoadPlan,
} from "./decode";

const urls = import.meta.glob<string>("../test/fixtures/decode/*.{mp4,webm,mkv,wav,avi}",
                                     { query: "?url", import: "default", eager: true });

async function file(name: string): Promise<File> {
  const blob = await (await fetch(urls[`../test/fixtures/decode/${name}`]!)).blob();
  return new File([blob], name, { type: blob.type });
}

// The clips' source frames are 64 × 48. Frame i is PALETTE[i] throughout, made lighter or darker
// in each quadrant (top left, top right, bottom left, bottom right) by SHADES, so it shows which
// way up it is. The longer clips loop through the 40 frames.
const PALETTE = [
  [104, 44, 44], [164, 104, 44], [104, 164, 224], [44, 164, 44], [164, 224, 44], [104, 104, 224],
  [224, 104, 104], [44, 104, 44], [224, 164, 164], [224, 44, 164], [44, 164, 164], [44, 44, 164],
  [164, 164, 164], [164, 44, 164], [104, 44, 224], [164, 224, 224], [44, 164, 224], [224, 164, 104],
  [164, 104, 104], [104, 104, 44], [104, 44, 164], [224, 224, 104], [44, 44, 224], [44, 44, 104],
  [104, 224, 164], [104, 164, 44], [104, 44, 104], [164, 224, 164], [104, 104, 104], [164, 44, 224],
  [104, 224, 44], [164, 164, 224], [44, 44, 44], [44, 104, 164], [104, 104, 164], [104, 164, 164],
  [224, 44, 224], [224, 44, 44], [224, 224, 164], [164, 44, 44],
];
const SHADES = [-30, -10, 10, 30];

// Where each of the source frame's quadrants shows, as indices into SHADES: as it was, and
// turned a quarter anticlockwise, as the turned clips say to show them.
const UPRIGHT = [0, 1, 2, 3];
const TURNED = [1, 3, 0, 2];

interface Seen { index: number; quadrants: number[] }

const mean = (values: number[]) => values.reduce((a, b) => a + b, 0) / values.length;

/** The index of the value at the least distance. */
function nearest<T>(values: T[], distance: (value: T) => number): number {
  return values.reduce((best, value, i) => distance(value) < distance(values[best]!) ? i : best, 0);
}

/** Which source frame a bitmap shows, and which way up. */
function identify(bitmap: ImageBitmap): Seen {
  const { width, height } = bitmap;
  const context = new OffscreenCanvas(width, height).getContext("2d")!;
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, width, height);
  // Each quadrant's mean colour, away from its edges, where coding blurs the shades together.
  const [w, h] = [width / 2, height / 2];
  const quadrants = [0, 1, 2, 3].map((q) => {
    const sums = [0, 0, 0];
    let n = 0;
    for (let y = Math.ceil((q >> 1) * h + h / 4); y < (q >> 1) * h + 3 * h / 4; y++) {
      for (let x = Math.ceil((q & 1) * w + w / 4); x < (q & 1) * w + 3 * w / 4; x++) {
        for (let c = 0; c < 3; c++) sums[c] += data[(y * width + x) * 4 + c]!;
        n++;
      }
    }
    return sums.map((s) => s / n);
  });
  // The shades cancel out over the four quadrants.
  const colour = [0, 1, 2].map((c) => mean(quadrants.map((q) => q[c]!)));
  const away = (p: number[]) => Math.max(...p.map((v, c) => Math.abs(v - colour[c]!)));
  const index = nearest(PALETTE, away);
  // Firefox's decoders, and Chrome's on Linux, use libyuv's BT.709 table, which puts colours
  // some 10 levels out in blue; the palette's are 60 apart.
  expect(away(PALETTE[index]!), "distance from the nearest palette colour").toBeLessThan(20);
  const base = mean(PALETTE[index]!);
  const shade = (q: number[]) => nearest(SHADES, (s) => Math.abs(s - (mean(q) - base)));
  return { index, quadrants: quadrants.map(shade) };
}

/** What `load` delivers with these options, each frame identified, having checked that it
 * delivers what `planLoad` said it would. */
async function loaded(name: string, options: LoadOptions = {}):
    Promise<{ plan: LoadPlan; seen: Seen[] }> {
  const source = await file(name);
  const plan = await planLoad(source, options);
  const seen: Seen[] = [];
  const progress: [number, number][] = [];
  const done = await load(source, options, (index, bitmap) => {
    expect(index).toBe(seen.length);
    expect([bitmap.width, bitmap.height]).toEqual([plan.width, plan.height]);
    seen.push(identify(bitmap));
    bitmap.close();
  }, (count, total) => progress.push([count, total]));
  expect(done).toEqual(plan);
  expect(seen).toHaveLength(plan.frames);
  expect(progress).toEqual(seen.map((_, i) => [i + 1, plan.frames]));
  return { plan, seen };
}

/** The source frames each loaded frame shows. */
async function indices(name: string, options: LoadOptions = {}): Promise<number[]> {
  return (await loaded(name, options)).seen.map((frame) => frame.index);
}

/** from, from + step, … up to but not including `to`. */
const range = (from: number, to: number, step = 1) =>
  Array.from({ length: Math.ceil((to - from) / step) }, (_, i) => from + i * step);

/** Expect each frame to show the same source frame as the one before, or one of the next two,
 * counting on through each loop of the palette. */
function expectSteady(indices: number[]): void {
  let loops = 0;
  const shown = indices.map((index, i) => {
    if (i > 0 && index < indices[i - 1]! - 20) loops++;
    return index + 40 * loops;
  });
  for (let i = 1; i < shown.length; i++) expect(shown[i]! - shown[i - 1]!).toBeOneOf([0, 1, 2]);
}

const rate = (num: number, den = 1) => ({ num, den });

// Whether the browser decodes the HEVC clip, as neither Playwright's Firefox nor Chrome on Linux
// does.
const hevc = (await VideoDecoder.isConfigSupported({ codec: "hev1.1.6.L30.90" })).supported;

describe("probe", () => {
  test.each([
    ["h264-30.mp4", 64, 48, rate(30), 40 / 30],
    ["h264-ntsc.mp4", 64, 48, rate(30000, 1001), 40 * 1001 / 30000],
    ["vp9-25.webm", 64, 48, rate(25), 40 / 25],
    // Timed to the millisecond, in Matroska.
    ["h264-30.mkv", 64, 48, rate(30), 40 / 30],
    // Turned upright.
    ["h264-rot90.mp4", 48, 64, rate(30), 40 / 30],
    ["vp9-alpha-rot90.webm", 48, 64, rate(25), 40 / 25],
    // Pixels half as wide again as they are high, made square.
    ["h264-anamorphic.mp4", 96, 48, rate(30), 40 / 30],
    // The video's length, not the 1.7 s of sound.
    ["h264-30-aac.mp4", 64, 48, rate(30), 40 / 30],
    // From the video's first frame, half a second into the file.
    ["h264-30-late.mp4", 64, 48, rate(30), 40 / 30],
    // Frames 10 to 19 last two frames each.
    ["h264-vfr.mp4", 64, 48, rate(30), 50 / 30],
    // 120 frames at 240 fps, the track's header saying each lasts a quarter second, as the
    // last does.
    ["vp9-240-at-4.webm", 64, 48, rate(240), 0.496 + 0.25],
    // A single frame, held as one frame of its own duration, or with none, a second.
    ["h264-single.mp4", 64, 48, rate(30), 1 / 30],
    ["vp9-single.webm", 64, 48, rate(1), 1],
  ])("%s", async (name, width, height, fps, duration) => {
    const info = await probe(await file(name));
    expect({ width: info.width, height: info.height, fps: info.fps })
      .toEqual({ width, height, fps });
    expect(info.duration).toBeCloseTo(duration, 3);  // to the millisecond some clips count in
  });

  test("measures a clip without WebCodecs, which the page reports itself", async () => {
    vi.stubGlobal("VideoDecoder", undefined);
    try {
      expect((await probe(await file("h264-30.mp4"))).fps).toEqual(rate(30));
    } finally {
      vi.unstubAllGlobals();
    }
  });
});

describe("load", () => {
  // The last three beside a timecode track, an empty sound track, and sound sharing the video's
  // track ID.
  test.each([
    "h264-30.mp4", "h264-ntsc.mp4", "vp9-25.webm",
    "h264-tmcd.mp4", "h264-empty-audio.mp4", "h264-aac-same-id.mp4",
  ])("%s: every frame", async (name) => {
    const { plan, seen } = await loaded(name);
    expect([plan.width, plan.height, plan.frames]).toEqual([64, 48, 40]);
    expect(seen.map((frame) => frame.index)).toEqual(range(0, 40));
    for (const frame of seen) expect(frame.quadrants).toEqual(UPRIGHT);
  });

  test.each([
    [{ scale: 0.5 }, 32, 24], [{ scale: 0.5, fast: true }, 32, 24], [{ scale: 1.5 }, 96, 72],
  ])("resizes with %j", async (options, width, height) => {
    const { plan, seen } = await loaded("h264-30.mp4", options);
    expect([plan.width, plan.height]).toEqual([width, height]);
    expect(seen.map((frame) => frame.index)).toEqual(range(0, 40));
    for (const frame of seen) expect(frame.quadrants).toEqual(UPRIGHT);
  });

  test.each(["h264-rot90.mp4", "vp9-alpha-rot90.webm"])("%s: turned upright", async (name) => {
    for (const options of [{}, { scale: 0.5, fast: true }]) {
      const { plan, seen } = await loaded(name, options);
      expect([plan.width, plan.height]).toEqual(options.scale ? [24, 32] : [48, 64]);
      expect(seen.map((frame) => frame.index)).toEqual(range(0, 40));
      for (const frame of seen) expect(frame.quadrants).toEqual(TURNED);
    }
  });

  test("keeps the colour beneath a clear or half-clear frame", async () => {
    // Clear in the top left quadrant and half clear in the bottom right; `identify` checks the
    // colours there are the palette's.
    const { seen } = await loaded("vp9-alpha.webm");
    expect(seen.map((frame) => frame.index)).toEqual(range(0, 40));
    for (const frame of seen) expect(frame.quadrants).toEqual(UPRIGHT);
  });

  test("makes non-square pixels square", async () => {
    const { plan, seen } = await loaded("h264-anamorphic.mp4", { scale: 0.5 });
    expect([plan.width, plan.height, plan.frames]).toEqual([48, 24, 40]);
    for (const frame of seen) expect(frame.quadrants).toEqual(UPRIGHT);
  });

  // Each slot holds the source frame on screen at the middle of the frame nearest the slot's
  // own middle, the earlier of two as near; the same whether the times are exact or rounded,
  // and counting from the video's first frame wherever in the file that is.
  describe.each(["h264-30.mp4", "h264-30.mkv", "h264-30-late.mp4"])("%s", (name) => {
    test.each([
      [{ timeScale: 0.5 }, rate(15), range(0, 40, 2)],
      [{ timeScale: 1 / 3 }, rate(10), range(1, 40, 3)],
      [{ timeScale: 2, duration: 0.2 }, rate(60), range(0, 12).map((i) => i >> 1)],
      [{ start: 0.5, duration: 0.5 }, rate(30), range(15, 30)],
      [{ start: 1.2 }, rate(30), range(36, 40)],
      [{ duration: 0.7 }, rate(30), range(0, 21)],
      // The frame on screen at 0.51 s started at 0.5 s.
      [{ start: 0.51, duration: 0.51 }, rate(30), range(15, 30)],
      [{ start: 0.5, duration: 0.02 }, rate(30), [15]],
      [{ start: 0.25, timeScale: 0.5 }, rate(15), range(8, 40, 2)],
    ])("picks frames at a constant rate with %j", async (options, fps, expected) => {
      const { plan, seen } = await loaded(name, options);
      expect(plan.fps).toEqual(fps);
      expect(seen.map((frame) => frame.index)).toEqual(expected);
    });
  });

  test("holds a frame for as many slots as it lasts", async () => {
    // Frames 10 to 19 last two frames each.
    expect(await indices("h264-vfr.mp4"))
      .toEqual([...range(0, 10), ...range(10, 20).flatMap((i) => [i, i]), ...range(20, 40)]);
    // Frames 9 and 24 last 3 and 4 frames, with B-frames decoded out of order.
    expect(await indices("h264-bf2-vfr.mp4"))
      .toEqual([...range(0, 9), 9, 9, 9, ...range(10, 24), 24, 24, 24, 24, ...range(25, 40)]);
    // The last of 120 frames at 240 fps lasts a quarter second.
    const { plan, seen } = await loaded("vp9-240-at-4.webm", { scale: 0.5, fast: true });
    expect([plan.fps, plan.frames]).toEqual([rate(240), 179]);
    expect(seen.map((frame) => frame.index))
      .toEqual([...range(0, 40), ...range(0, 40), ...range(0, 40), ...Array(59).fill(39)]);
  });

  test("holds a clip at the rate most frames come, though a few come between", async () => {
    // At 30 fps, frames 5, 15, 25 and 35 half a frame early: on a 60 fps lattice. Each early
    // frame starts at the time its slot is taken at, so counts as later, keeping the one before.
    const { plan, seen } = await loaded("h264-30-early.mp4", { scale: 0.5, fast: true });
    expect([plan.fps, plan.frames]).toEqual([rate(30), 40]);
    expect(seen.map((frame) => frame.index)).toEqual(range(0, 40));
  });

  test("starts a clip an edit list cuts mid-frame with the frame on screen then", async () => {
    // Cut a quarter second in, halfway through frame 7, decoding from frame 5, and timed to the
    // millisecond, so slots taken where frames change would show some twice and skip others.
    const { plan, seen } = await loaded("h264-30-cut.mp4");
    expect([plan.fps, plan.frames]).toEqual([rate(30), 33]);
    expect(seen.map((frame) => frame.index)).toEqual(range(7, 40));
  });

  test("takes a rate its timestamps round unevenly for the common rate it is near", async () => {
    // 240 fps timed in 600ths of a second, 2 and 3 ticks apart by turns, as phones' slow motion
    // is; the last frame lasts a thirtieth of a second.
    const { plan, seen } = await loaded("h264-240-at-600.mp4", { scale: 0.5, fast: true });
    expect(plan.fps).toEqual(rate(240));
    expect(seen.map((frame) => frame.index))
      .toEqual([...range(0, 40), ...Array(7).fill(39)]);
  });

  test("takes a slow rate its timestamps wander about for the common rate it is near", async () => {
    // About 5 fps timed to the millisecond, each frame up to 70 ms early or late, and frame 22
    // held a second.
    const { plan, seen } = await loaded("vp9-slow-stall.mp4", { scale: 0.5, fast: true });
    expect(plan.fps).toEqual(rate(5));
    expect(seen.map((frame) => frame.index))
      .toEqual([...range(0, 22), 22, 22, 22, 22, 22, ...range(23, 40)]);
  });

  test("takes a rate between one tick a frame and two for the common rate it is near", async () => {
    // 60 fps timed in hundredths of a second: frames two ticks apart, but every third one.
    const { plan, seen } = await loaded("h264-60-at-100.mp4", { scale: 0.5, fast: true });
    expect([plan.fps, plan.frames]).toEqual([rate(60), 40]);
    expect(seen.map((frame) => frame.index)).toEqual(range(0, 40));
  });

  test("holds a clip whose rate its timestamps can't place at its mean rate", async () => {
    // 45 frames at 30 fps then 35 at 20 fps, timed in sixtieths of a second.
    const { plan, seen } = await loaded("h264-ts60.mp4", { scale: 0.5, fast: true });
    const fps = plan.fps.num / plan.fps.den;
    expect(fps).toBeGreaterThan(20);
    expect(fps).toBeLessThan(30);
    expect(Math.abs(plan.frames - 80)).toBeLessThanOrEqual(2);
    expect(seen[0]!.index).toBe(0);
    expect(seen.at(-1)!.index).toBe(39);
    expectSteady(seen.map((frame) => frame.index));
  });

  test("holds timestamps jittered about a common rate at that rate", async () => {
    // 120 fps timed to the millisecond, each frame up to 4 ms early or late, under half a frame.
    const { plan, seen } = await loaded("h264-uneven-120.mp4", { scale: 0.5, fast: true });
    expect([plan.fps, plan.frames]).toEqual([rate(120), 120]);
    expect(seen.map((frame) => frame.index)).toEqual([0, 1, 2].flatMap(() => range(0, 40)));
  });

  // Fragmented MP4, with B-frames, indexed by a sidx box, and with VP9.
  test.each([
    "h264-uneven-frag.mp4", "h264-uneven-frag-bf.mp4", "h264-uneven-sidx.mp4", "vp9-uneven-frag.mp4",
  ])("%s: keeps frames that wander about their rate in order", async (name) => {
    // Each frame up to 10 ms either side of a steady 30.14 fps, timed to the millisecond, so held
    // at 30. From the first frame, a few sit over half a frame early, so are skipped and others
    // shown twice.
    const { plan, seen } = await loaded(name, { scale: 0.5, fast: true });
    expect([plan.fps, plan.frames]).toEqual([rate(30), 40]);
    const shown = seen.map((frame) => frame.index);
    expect([shown[0], shown.at(-1)]).toEqual([0, 39]);
    expectSteady(shown);
  });

  test.each(["h264-single.mp4", "vp9-single.webm"])("%s: a single frame", async (name) => {
    expect(await indices(name)).toEqual([0]);
  });

  test("loads HEVC where the browser decodes it", async () => {
    const source = await file("hevc-uneven-30.mp4");
    if (!hevc) {
      const refused = new VideoError("The browser has no decoder for the hevc video in "
                                     + "hevc-uneven-30.mp4. Convert it to an H.264 MP4 first.");
      await expect(probe(source)).rejects.toThrow(refused);
      await expect(load(source, {}, (_, bitmap) => bitmap.close())).rejects.toThrow(refused);
      return;
    }
    const shown = await indices("hevc-uneven-30.mp4");
    expect(shown).toHaveLength(40);
    expectSteady(shown);
  });
});

describe("errors", () => {
  const junk = new File(["this is not a video\n".repeat(8)], "junk.mp4");
  test.each([
    [junk, {}, "can't read a video stream from junk.mp4"],
    ["audio.wav", {}, "can't read a video stream from audio.wav"],
    ["h264-30.mp4", { start: 5 }, "no frames decoded from h264-30.mp4"],
    ["clip.avi", {}, "The browser can't open AVI files such as clip.avi. Convert it to an H.264 "
                     + "MP4 first."],
  ] as const)("%s %j", async (clip, options, message) => {
    const source = typeof clip === "string" ? await file(clip) : clip;
    const attempts = await Promise.allSettled([
      planLoad(source, options), load(source, options, (_, bitmap) => bitmap.close()),
      ...Object.keys(options).length === 0 ? [probe(source)] : [],
    ]);
    for (const attempt of attempts) {
      expect(attempt.status).toBe("rejected");
      const error = (attempt as PromiseRejectedResult).reason as Error;
      expect(error).toBeInstanceOf(VideoError);
      expect(error.message).toBe(message);
    }
  });

  test("a codec the browser can't decode", async () => {
    // Mediabunny warns of the codec it doesn't know as it reads the file.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const source = await file("mpeg4.mp4");
      const attempts = await Promise.allSettled([
        probe(source), planLoad(source), load(source, {}, (_, bitmap) => bitmap.close()),
      ]);
      for (const attempt of attempts) {
        expect(attempt.status).toBe("rejected");
        const error = (attempt as PromiseRejectedResult).reason as Error;
        expect(error).toBeInstanceOf(VideoError);
        expect(error.message)
          .toMatch(/^The browser has no decoder for the \S+ video in mpeg4\.mp4\. /);
      }
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("mp4v"));
    } finally {
      warn.mockRestore();
    }
  });
});

describe("an AbortSignal", () => {
  test("stops load part way", async () => {
    const source = await file("h264-30.mp4");
    const controller = new AbortController();
    const indices: number[] = [];
    const loading = load(source, { signal: controller.signal }, (index, bitmap) => {
      bitmap.close();
      indices.push(index);
      if (indices.length === 3) controller.abort();
    });
    await expect(loading).rejects.toMatchObject({ name: "AbortError" });
    expect(indices).toEqual([0, 1, 2]);
  });

  test("already aborted loads nothing", async () => {
    const source = await file("h264-30.mp4");
    let puts = 0;
    const loading = load(source, { signal: AbortSignal.abort() }, () => { puts++; });
    await expect(loading).rejects.toMatchObject({ name: "AbortError" });
    await expect(planLoad(source, { signal: AbortSignal.abort() }))
      .rejects.toMatchObject({ name: "AbortError" });
    expect(puts).toBe(0);
  });

  test("aborted while opening the file to plan, rejects", async () => {
    const controller = new AbortController();
    const planning = planLoad(await file("h264-30.mp4"), { signal: controller.signal });
    controller.abort();
    await expect(planning).rejects.toMatchObject({ name: "AbortError" });
  });

  test("aborted while measuring the open file, rejects", async () => {
    const source = await file("h264-30.mp4");
    const controller = new AbortController();
    const ask = VideoDecoder.isConfigSupported.bind(VideoDecoder);
    // Whether the browser decodes the clip is asked once the file is open.
    const asked = vi.spyOn(VideoDecoder, "isConfigSupported").mockImplementation((config) => {
      controller.abort();
      return ask(config);
    });
    try {
      await expect(planLoad(source, { signal: controller.signal }))
        .rejects.toMatchObject({ name: "AbortError" });
      expect(asked).toHaveBeenCalled();
    } finally {
      asked.mockRestore();
    }
  });
});

test("clipSeconds", () => {
  expect(clipSeconds(10)).toBe(10);
  expect(clipSeconds(10, 0.25)).toBe(9.75);
  expect(clipSeconds(10, 0.25, 3)).toBe(3);
  expect(clipSeconds(10, 9, 3)).toBe(1);
  expect(clipSeconds(10, 12)).toBe(0);
  expect(clipSeconds(null)).toBeNull();
  expect(clipSeconds(null, 0.25)).toBeNull();
  expect(clipSeconds(null, 0.25, 3)).toBe(3);
});
