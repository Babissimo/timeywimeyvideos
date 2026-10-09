import { describe, expect, test } from "vitest";
import raw from "../test/fixtures/render.json?raw";
import { Problem } from "./options";
import { framesOf, readBack, thumbnail } from "./readback";
import { handOver, render, type Progress, type Rendered } from "./render";
import { Unrenderable } from "./unrenderable";

interface PyRender {
  clip: string; values: Record<string, string>; preview: boolean; output: string;
  codec: string; width: number; height: number; frames: number; fps: [number, number];
  cols: number; rows: number; chosen: number[]; thumbnails: number[][];
}

const { renders } = JSON.parse(raw) as { renders: PyRender[] };
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

describe("renders what timeslice.py renders", () => {
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
      // it would differ by far more.
      const frames = await framesOf(made.video, c.chosen);
      frames.forEach((frame, n) => {
        const got = thumbnail(frame, c.width, c.height, c.cols, c.rows);
        const expected = c.thumbnails[n];
        const diffs = got.map((v, k) => Math.abs(v - expected[k]));
        const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
        const most = Math.max(...diffs);
        console.log(`frame ${c.chosen[n]}: mean difference ${mean.toFixed(2)}, most ${most}`);
        expect(mean).toBeLessThanOrEqual(5);
        expect(most).toBeLessThanOrEqual(16);
      });
    });
  }
});

describe("render", () => {
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
    expect(error).not.toBeInstanceOf(Unrenderable);
  });

  test("offers timeslice.py's command for a file the browser can't open or decode", async () => {
    const avi = await failure(render(await clip("clip.avi"), {}));
    expect(avi).toBeInstanceOf(Unrenderable);
    expect(handOver(avi as Unrenderable))
      .toBe("Can't read clip.avi: the browser can't open AVI files. timeslice.py can render it: "
            + "run this beside it, with the video's path in place of its name.");
    expect((avi as Unrenderable).command)
      .toBe("uv run timeslice.py clip.avi clip_rotate_45deg.mp4 --angle=45 --slice=rotate");
    const mpeg4 = await failure(render(await clip("mpeg4.mp4"), {}, { preview: true }));
    expect(mpeg4).toBeInstanceOf(Unrenderable);
    expect((mpeg4 as Unrenderable).message)
      .toMatch(/^Can't decode mpeg4\.mp4: the browser has no decoder for its \S+ video\.$/);
    expect((mpeg4 as Unrenderable).command)
      .toBe("uv run timeslice.py mpeg4.mp4 mpeg4_rotate_45deg_preview.mp4 --angle=45 "
            + "--slice=rotate --preview");
  });

  test("offers timeslice.py's command for a clip the GPU can't hold", async () => {
    const gl = document.createElement("canvas").getContext("webgl2")!;
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    const file = new File([await clip("h264-30.mp4")], "my clip.mp4");

    // At 90° a frame is as wide as the clip is long, so only the clip is too large.
    const wide = await failure(render(file, { angle: "90", scale: "300" }));
    expect(wide).toBeInstanceOf(Unrenderable);
    expect((wide as Unrenderable).message)
      .toBe(`The GPU can hold frames at most ${size} pixels across, and this clip's are `
            + "19200×14400.");
    expect((wide as Unrenderable).command)
      .toBe("uv run timeslice.py 'my clip.mp4' 'my clip_rotate_90deg_scale300.mp4' "
            + "--angle=90 --slice=rotate --scale=300");

    const huge = await failure(render(file, { angle: "90", scale: String(size / 64) }));
    expect(huge).toBeInstanceOf(Unrenderable);
    expect((huge as Unrenderable).message).toMatch(
      /^This clip needs \d+\.\d GB of GPU memory, more than the GPU can give it\.$/);
  });
});
