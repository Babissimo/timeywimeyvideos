import { describe, expect, test, vi } from "vitest";
import { clipSeconds, load, planLoad, probe, VideoError, type LoadOptions } from "./decode";
import fixture from "../test/fixtures/decode/decode.json?raw";
import framesUrl from "../test/fixtures/decode/frames.bin.gz?url";

interface PyOptions {
  scale?: number; timeScale?: number; start?: number | null; duration?: number | null; fast?: boolean;
}
interface PyCase {
  clip: string; options: PyOptions; width: number; height: number;
  fps: [number, number]; frames: number[]; offset: number;
}
interface PyClip { width: number; height: number; fps: [number, number]; duration: number | null }
interface PyError { clip: string; options: PyOptions; message: string }

const { junk, undecodable, unopenable, clips, errors, cases, clipSeconds: seconds } = JSON.parse(fixture) as {
  junk: string; undecodable: string; unopenable: string; clips: Record<string, PyClip>; errors: PyError[];
  cases: PyCase[]; clipSeconds: [number | null, number | null, number | null, number | null][];
};
const urls = import.meta.glob<string>("../test/fixtures/decode/*.{mp4,webm,wav,avi}",
                                     { query: "?url", import: "default", eager: true });

async function file(name: string): Promise<File> {
  if (name === "junk.mp4") return new File([junk], name);
  const blob = await (await fetch(urls[`../test/fixtures/decode/${name}`]!)).blob();
  return new File([blob], name, { type: blob.type });
}

function loadOptions(options: PyOptions): LoadOptions {
  return {
    scale: options.scale, timeScale: options.timeScale, start: options.start ?? undefined,
    duration: options.duration ?? undefined, fast: options.fast,
  };
}

function label(c: PyCase): string {
  const o = c.options;
  const parts = [c.clip];
  if (o.scale !== 1) parts.push(`scale=${o.scale}`);
  if (o.timeScale !== 1) parts.push(`timeScale=${o.timeScale}`);
  if (o.start != null) parts.push(`start=${o.start}`);
  if (o.duration != null) parts.push(`duration=${o.duration}`);
  if (o.fast) parts.push("fast");
  return parts.join(" ");
}

let pythonFrames: Promise<Uint8Array> | undefined;
function pythonPixels(): Promise<Uint8Array> {
  pythonFrames ??= fetch(framesUrl).then(async (response) => {
    const bytes = new Uint8Array(await response.arrayBuffer());
    // Vite sends a .gz file as Content-Encoding: gzip, so it may arrive inflated.
    if (bytes[0] !== 0x1f || bytes[1] !== 0x8b) return bytes;
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  });
  return pythonFrames;
}

/** A bitmap's pixels as rgb24, rows top first, like a frame of load_video's array. */
function rgb(bitmap: ImageBitmap): Uint8Array {
  const canvas = new OffscreenCanvas(bitmap.width, bitmap.height);
  const context = canvas.getContext("2d")!;
  context.drawImage(bitmap, 0, 0);
  const rgba = context.getImageData(0, 0, bitmap.width, bitmap.height).data;
  const out = new Uint8Array(bitmap.width * bitmap.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    out[j] = rgba[i]!;
    out[j + 1] = rgba[i + 1]!;
    out[j + 2] = rgba[i + 2]!;
  }
  return out;
}

function meanAbsDiff(a: Uint8Array, b: Uint8Array): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i]! - b[i]!);
  return sum / a.length;
}

describe("probe", () => {
  test.each(Object.entries(clips))("%s", async (name, py) => {
    const info = await probe(await file(name));
    expect({ width: info.width, height: info.height, fps: info.fps })
      .toEqual({ width: py.width, height: py.height, fps: { num: py.fps[0], den: py.fps[1] } });
    // ffprobe prints the duration to the microsecond.
    expect(Math.abs(info.duration! - py.duration!)).toBeLessThanOrEqual(1e-6);
  });
});

describe("planLoad and load match load_video", () => {
  test.each(cases.map((c) => [label(c), c] as const))("%s", async (_, c) => {
    const source = await file(c.clip);
    const options = loadOptions(c.options);
    const expected = {
      width: c.width, height: c.height, frames: c.frames.length,
      fps: { num: c.fps[0], den: c.fps[1] },
    };
    expect(await planLoad(source, options)).toEqual(expected);

    const got: Uint8Array[] = [];
    const progress: [number, number][] = [];
    const sizes = new Set<string>();
    const plan = await load(source, options, (index, bitmap) => {
      expect(index).toBe(got.length);
      sizes.add(`${bitmap.width}x${bitmap.height}`);
      got.push(rgb(bitmap));
      bitmap.close();
    }, (done, total) => progress.push([done, total]));
    expect(plan).toEqual(expected);
    expect([...sizes]).toEqual([`${c.width}x${c.height}`]);
    expect(progress).toEqual(c.frames.map((_, i) => [i + 1, c.frames.length]));

    // Name each frame by the nearest of Python's, then compare it with the one
    // Python returned at the same place.
    const pixels = await pythonPixels();
    const size = c.width * c.height * 3;
    const py = c.frames.map((_, i) => pixels.subarray(c.offset + i * size, c.offset + (i + 1) * size));
    const named = got.map((frame) => {
      const diffs = py.map((p) => meanAbsDiff(frame, p));
      return c.frames[diffs.indexOf(Math.min(...diffs))];
    });
    expect(named).toEqual(c.frames);
    const worst = Math.max(...got.map((frame, i) => meanAbsDiff(frame, py[i]!)));
    expect(worst, "largest mean absolute difference from Python's frame").toBeLessThanOrEqual(4);
  });
});

describe("errors", () => {
  test.each(errors.map((e) => [`${e.clip} ${JSON.stringify(e.options)}`, e] as const))(
    "%s",
    async (_, e) => {
      const source = await file(e.clip);
      const attempts = await Promise.allSettled([
        planLoad(source, loadOptions(e.options)),
        load(source, loadOptions(e.options), (_, bitmap) => bitmap.close()),
        ...Object.keys(e.options).length === 0 ? [probe(source)] : [],
      ]);
      for (const attempt of attempts) {
        expect(attempt.status).toBe("rejected");
        const error = (attempt as PromiseRejectedResult).reason as Error;
        expect(error).toBeInstanceOf(VideoError);
        expect(error.message).toBe(e.message);
      }
    },
  );

  test("a codec the browser can't decode", async () => {
    // Mediabunny warns of the codec it doesn't know as it reads the file.
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    try {
      const source = await file(undecodable);
      const attempts = await Promise.allSettled([
        planLoad(source), load(source, {}, (_, bitmap) => bitmap.close()),
      ]);
      for (const attempt of attempts) {
        expect(attempt.status).toBe("rejected");
        const error = (attempt as PromiseRejectedResult).reason as Error;
        expect(error).toBeInstanceOf(VideoError);
        expect(error.message).toMatch(/^can't decode mpeg4\.mp4: the browser has no decoder for/);
      }
      expect(warn).toHaveBeenCalledWith(expect.stringContaining("mp4v"));
    } finally {
      warn.mockRestore();
    }
  });

  test("a container the browser can't open", async () => {
    const source = await file(unopenable);
    const attempts = await Promise.allSettled([probe(source), planLoad(source), load(source, {}, () => {})]);
    for (const attempt of attempts) {
      expect(attempt.status).toBe("rejected");
      const error = (attempt as PromiseRejectedResult).reason as Error;
      expect(error).toBeInstanceOf(VideoError);
      expect(error.message).toBe("can't read clip.avi: the browser can't open AVI files");
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
});

test("clipSeconds matches clip_seconds", () => {
  for (const [length, start, duration, expected] of seconds) {
    expect(clipSeconds(length, start ?? undefined, duration ?? undefined)).toBe(expected);
  }
});
