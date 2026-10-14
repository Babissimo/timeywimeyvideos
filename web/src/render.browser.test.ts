import { describe, expect, test } from "vitest";
import { server } from "vitest/browser";
import raw from "../test/fixtures/render.json?raw";
import { Problem } from "./options";
import { framesOf, readBack, thumbnail } from "./readback";
import { render, type Progress, type Rendered } from "./render";

interface Recorded {
  clip: string; values: Record<string, string>; preview: boolean; output: string;
  codec: string; width: number; height: number; frames: number; fps: [number, number];
  cols: number; rows: number; chosen: number[]; thumbnails: number[][];
}

const { renders } = JSON.parse(raw) as { renders: Recorded[] };
const urls = import.meta.glob<string>("../test/fixtures/decode/*.{mp4,avi}",
                                     { query: "?url", import: "default", eager: true });

async function clip(name: string): Promise<File> {
  const blob = await (await fetch(urls[`../test/fixtures/decode/${name}`]!)).blob();
  return new File([blob], name, { type: blob.type });
}

/** A worker as render starts it, and whether it has been ended. */
function watched(): { start: () => Worker; ended: () => boolean } {
  let ended = false;
  return {
    start: () => {
      const worker = new Worker(new URL("./render.worker.ts", import.meta.url),
                                { type: "module" });
      const terminate = worker.terminate.bind(worker);
      worker.terminate = () => {
        ended = true;
        terminate();
      };
      return worker;
    },
    ended: () => ended,
  };
}

async function failure(promise: Promise<unknown>): Promise<unknown> {
  return promise.then(() => expect.fail("the render was made"), (error: unknown) => error);
}

// Firefox's decoders, and Chrome's on Linux, make RGB with libyuv's BT.709 table, whose 2 for
// Cb's part in blue, not 2.112, takes saturated colours' blue some 10 levels towards grey: once
// as the clip is loaded, and again as the render is read back.
const MOST = server.browser === "firefox" || server.platform === "linux" ? 24 : 16;

describe("renders as recorded", () => {
  for (const c of renders) {
    test(`${c.clip} ${JSON.stringify(c.values)}${c.preview ? " preview" : ""}`, async () => {
      const steps: Progress[] = [];
      const { start, ended } = watched();
      const made: Rendered = await render(await clip(c.clip), c.values,
                                          { preview: c.preview, progress: (p) => steps.push(p),
                                            worker: start });
      expect(ended()).toBe(true);
      expect(made.name).toBe(c.output);
      const fps = { num: c.fps[0], den: c.fps[1] };
      expect([made.width, made.height, made.frames, made.fps])
        .toEqual([c.width, c.height, c.frames, fps]);
      expect(made.seconds).toBeCloseTo(c.frames * c.fps[1] / c.fps[0], 12);
      expect(await readBack(made.video)).toEqual({
        codec: "avc", width: c.width, height: c.height, frames: c.frames, fps,
        colour: { primaries: "bt709", transfer: "bt709", matrix: "bt709", fullRange: false },
      });

      // Progress runs through decoding, then each frame, then finishing.
      const stages = steps.map((p) => p.stage);
      expect(stages.indexOf("slicing")).toBeGreaterThan(0);
      expect(stages.slice(0, stages.indexOf("slicing")).every((s) => s === "loading")).toBe(true);
      const sliced = steps.filter((p) => p.stage === "slicing");
      expect(sliced.map((p) => p.done)).toEqual(Array.from({ length: c.frames }, (_, i) => i + 1));
      expect(steps.at(-1)).toEqual({ stage: "finishing", done: c.frames, total: c.frames });

      // Both videos are lossy, and from different encoders, so their thumbnails match only
      // loosely: about as closely as neighbouring frames of the slower sweeps do. Each of
      // the clip's frames is a colour of its own, so frames or columns from the wrong part of
      // it would differ by far more. Firefox's encoder garbles the luma of some P-frames as
      // small as these previews, whatever it is asked, so there a preview is judged by its first
      // frame, a key frame, alone.
      const judged = c.preview && server.browser === "firefox" ? c.chosen.slice(0, 1) : c.chosen;
      const frames = await framesOf(made.video, judged);
      frames.forEach((frame, n) => {
        const got = thumbnail(frame, c.width, c.height, c.cols, c.rows);
        const expected = c.thumbnails[n];
        const diffs = got.map((v, k) => Math.abs(v - expected[k]));
        const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        const most = Math.max(...diffs);
        console.log(`frame ${judged[n]}: mean difference ${mean.toFixed(2)}, most ${most}`);
        expect(mean).toBeLessThanOrEqual(5);
        expect(most).toBeLessThanOrEqual(MOST);
      });
    });
  }
});

describe("render", () => {
  test("bends the slice with surface noise", async () => {
    const file = await clip("h264-30.mp4");
    const flat = await render(file, { angle: "45", loop: "1" }, { preview: true });
    const bumpy = await render(file, { angle: "45", loop: "1", noise: "3", noise_seed: "2" },
                               { preview: true });
    expect([bumpy.width, bumpy.height, bumpy.frames]).toEqual([flat.width, flat.height,
                                                                 flat.frames]);
    const [[a], [b]] = [await framesOf(flat.video, [3]), await framesOf(bumpy.video, [3])];
    let sum = 0;
    for (let i = 0; i < a.length; i++) if (i % 4 !== 3) sum += Math.abs(a[i] - b[i]);
    expect(sum / (a.length * 3 / 4)).toBeGreaterThan(2);
  });

  test("fades a loop's ends and sides", async () => {
    const file = await clip("h264-30.mp4");  // 40 frames 64 × 48, at 30 fps
    const made = await render(file, { angle: "0", loop: "1", loop_sides: "1", loop_fade: "0.4",
                                      loop_side_fade: "8" });
    expect([made.width, made.height, made.frames]).toEqual([56, 48, 28]);
    // Each of the clip's frames is a colour of its own, and at 0° each output frame is one of
    // them, so the faded loop's first is mostly the frame its fade begins with, 12/13 of it.
    const [start, end] = await framesOf(file, [0, 28]);
    const [first] = await framesOf(made.video, [0]);
    const off = (frame: Uint8ClampedArray) => {
      let sum = 0;  // over the columns the side fade leaves alone
      for (let y = 0; y < 48; y++)
        for (let x = 8; x < 56; x++)
          for (let k = 0; k < 3; k++)
            sum += Math.abs(first[4 * (y * 56 + x) + k] - frame[4 * (y * 64 + x) + k]);
      return sum / (48 * 48 * 3);
    };
    expect(off(end)).toBeLessThan(off(start) / 4);
  });

  test("gives an odd sweep's frames the next column along, not black", async () => {
    // A loop round the sides of the clip at 63 × 47, whose left and right halves differ, so
    // that the 64th column is its first come round again.
    const made = await render(await clip("h264-30.mp4"),
                              { angle: "0", loop: "1", loop_sides: "1", scale: "0.98" });
    expect([made.width, made.height]).toEqual([64, 48]);
    for (const frame of await framesOf(made.video, [0, 20])) {
      const means = thumbnail(frame, 64, 48, 64, 1);  // each column's mean colour
      const column = (x: number) => means.slice(3 * x, 3 * x + 3);
      const off = (a: number[], b: number[]) => Math.max(...a.map((v, k) => Math.abs(v - b[k])));
      expect(off(column(0), column(62))).toBeGreaterThan(12);
      expect(off(column(63), column(0))).toBeLessThanOrEqual(6);
    }
  });

  test("stops the worker when cancelled", async () => {
    const { start, ended } = watched();
    const controller = new AbortController();
    let after = 0;
    const making = render(await clip("h264-30.mp4"), { angle: "30" }, {
      signal: controller.signal, worker: start,
      progress: ({ stage }) => {
        if (controller.signal.aborted) after++;
        else if (stage === "slicing") controller.abort();
      },
    });
    expect(await failure(making)).toMatchObject({ name: "AbortError" });
    expect(ended()).toBe(true);
    await new Promise((resolve) => setTimeout(resolve, 200));
    expect(after).toBe(0);
  });

  test("refuses options before starting a worker", async () => {
    const { start, ended } = watched();
    const error = await failure(render(await clip("h264-30.mp4"), { angle: "steep" },
                                       { worker: start }));
    expect(error).toEqual(new Problem("angle must be a number."));
    expect(ended()).toBe(false);
  });

  test("says why a sweep doesn't fit", async () => {
    const error = await failure(render(await clip("h264-30.mp4"), { angle: "80", inside: "1" }));
    expect(error).toBeInstanceOf(Problem);
    expect((error as Problem).kind).toBe("does_not_fit");
    expect((error as Problem).message).toContain("Let black in or Wrap round");
  });

  test("says when it can't read the file", async () => {
    const error = await failure(render(new File(["not a video"], "notes.mp4"), {}));
    expect(error).toEqual(new Problem("can't read a video stream from notes.mp4"));
  });

  test("says what to do with a file the browser can't open or decode", async () => {
    const avi = await failure(render(await clip("clip.avi"), {}));
    expect(avi).toEqual(new Problem("The browser can't open AVI files such as clip.avi. Convert "
                                    + "it to an H.264 MP4 first."));
    const mpeg4 = await failure(render(await clip("mpeg4.mp4"), {}, { preview: true }));
    expect(mpeg4).toBeInstanceOf(Problem);
    expect((mpeg4 as Problem).message).toMatch(
      /^The browser has no decoder for the \S+ video in mpeg4\.mp4\. Convert it to an H\.264 MP4 first\.$/);
  });

  test("says how to make a clip the GPU can't hold fit", async () => {
    const gl = document.createElement("canvas").getContext("webgl2")!;
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    const file = await clip("h264-30.mp4");

    // At 90° a frame is as wide as the clip is long, so at a scale between size / 64 and
    // size / 48 only the clip, 64 × 48 before it, is too large.
    const scale = Math.round(size / 56);
    const wide = await failure(render(file, { angle: "90", scale: String(scale) }));
    expect(wide).toEqual(new Problem(`The GPU can hold frames at most ${size} pixels across, `
                                     + `and this clip's are ${64 * scale}×${48 * scale}. `
                                     + "Set a smaller scale."));

    const huge = await failure(render(file, { angle: "90", scale: String(size / 64) }));
    expect(huge).toBeInstanceOf(Problem);
    expect((huge as Problem).message).toMatch(
      /^This clip needs \d+\.\d GB of GPU memory, .*\. Set a shorter duration or a smaller scale\.$/);
  });
});
