import { afterAll, afterEach, describe, expect, test, vi } from "vitest";
import raw from "../test/fixtures/live.json?raw";
import h264 from "../test/fixtures/decode/h264-30.mp4?url";
import turned from "../test/fixtures/decode/h264-rot90.mp4?url";
import { probe } from "./decode";
import { NOISY } from "./gpu/shaders";
import { Volume, VolumeTooLarge } from "./gpu/volume";
import { fetchSource, Live, type FrameInfo } from "./live";
import { Noise } from "./noise";
import { endpoints, fullSize, liveNoise, plan, Problem, readOptions } from "./options";

type PyFrame = Omit<FrameInfo, "memory" | "full"> & {
  values: Record<string, string>; pos: number; rgb: string;
};

const cases = JSON.parse(raw) as PyFrame[];
const SOURCE = "videos/h264-30.mp4";

let bytes: Promise<Blob> | undefined;
async function clip(): Promise<File> {
  bytes ??= fetch(h264).then((res) => res.blob());
  return new File([await bytes], "h264-30.mp4");
}

/** Let a canvas's GPU context go, as browsers allow only so many at once. */
const free = (canvas: HTMLCanvasElement) =>
  canvas.getContext("webgl2")?.getExtension("WEBGL_lose_context")?.loseContext();

const made: HTMLCanvasElement[] = [];
afterEach(() => {
  made.splice(0).forEach(free);
  vi.restoreAllMocks();
});

type Open = (source: string) => Promise<Blob>;

/** A live view on a canvas of its own, and how many times it has opened a file. */
function view(open?: Open): { live: Live; canvas: HTMLCanvasElement; opened: () => number } {
  const canvas = document.createElement("canvas");
  let count = 0;
  const live = new Live(canvas, async (source) => {
    count++;
    return open ? open(source) : clip();
  });
  return { live, canvas, opened: () => count };
}

/** A view for one test, freed after it. */
function make(open?: Open): ReturnType<typeof view> {
  const one = view(open);
  made.push(one.canvas);
  return one;
}

/** What the canvas shows, as RGBA rows top-first. */
function drawn(canvas: HTMLCanvasElement): Uint8Array {
  const gl = canvas.getContext("webgl2")!;  // the live view's own
  const { width, height } = canvas;
  const pixels = new Uint8Array(width * height * 4);
  gl.readPixels(0, 0, width, height, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  const out = new Uint8Array(pixels.length), row = width * 4;
  for (let y = 0; y < height; y++)  // the canvas's bottom row comes first
    out.set(pixels.subarray((height - 1 - y) * row, (height - y) * row), y * row);
  return out;
}

function base64(text: string): Uint8Array {
  return Uint8Array.from(atob(text), (c) => c.charCodeAt(0));
}

async function problem(promise: Promise<unknown>): Promise<{ message: string; kind: string }> {
  const error = await promise.then(() => undefined, (error: unknown) => error);
  expect(error).toBeInstanceOf(Problem);
  const { message, kind } = error as Problem;
  return { message, kind };
}

const close = (got: number[], expected: number[]) =>
  got.forEach((v, i) => expect(Math.abs(v - expected[i]))
    .toBeLessThanOrEqual(1e-9 * Math.max(1, Math.abs(expected[i]))));

describe("shows the frames timeslice --preview makes", () => {
  const shared = view();
  afterAll(() => {
    shared.live.dispose();
    free(shared.canvas);
  });
  for (const c of cases) {
    test(`${JSON.stringify(c.values)} at ${c.pos}`, async () => {
      const info = await shared.live.show(SOURCE, c.values, c.pos);
      expect([info.width, info.height, info.frames, info.frame, info.loop, info.sides])
        .toEqual([c.width, c.height, c.frames, c.frame, c.loop, c.sides]);
      expect(info.volume).toEqual(c.volume);
      close([info.fps, info.seconds], [c.fps, c.seconds]);
      close(info.line, c.line);
      close(info.first, c.first);
      close(info.last, c.last);
      expect(info.memory).toBeGreaterThanOrEqual(4 * c.volume[0] * c.volume[1] * c.volume[2]);
      expect(info.full).toEqual(fullSize(await probe(await clip()), readOptions(c.values)));

      // The browser decodes a little differently from ffmpeg, so frames match closely but
      // not exactly; the wrong frame or column of the clip would differ by far more.
      expect([shared.canvas.width, shared.canvas.height]).toEqual([c.width, c.height]);
      const got = drawn(shared.canvas), expected = base64(c.rgb);
      let sum = 0, near = 0;
      for (let p = 0; p < c.width * c.height; p++) {
        let most = 0;
        for (let k = 0; k < 3; k++) {
          const diff = Math.abs(got[4 * p + k] - expected[3 * p + k]);
          sum += diff;
          most = Math.max(most, diff);
        }
        if (most <= 16) near++;
        expect(got[4 * p + 3]).toBe(255);
      }
      const mean = sum / (3 * c.width * c.height), share = near / (c.width * c.height);
      console.log(`mean difference ${mean.toFixed(2)}, ${(100 * share).toFixed(1)}% of `
                  + "pixels within 16 levels");
      expect(mean).toBeLessThanOrEqual(4);
      expect(share).toBeGreaterThanOrEqual(0.97);
    });
  }
});

describe("Live", () => {
  test("loads a clip only when the options it is loaded with change", async () => {
    const { live, opened } = make();
    const progress = vi.fn();
    await live.show(SOURCE, { angle: "30" }, 0.5, progress);
    expect(progress).toHaveBeenCalledTimes(20);
    expect(progress).toHaveBeenLastCalledWith(20, 20);
    progress.mockClear();
    await live.show(SOURCE, { angle: "-20", inside: "1", noise: "2", fps: "12" }, 0.1, progress);
    await live.show(SOURCE, { slice: "shear", angle: "10" }, 0.9, progress);
    expect(progress).not.toHaveBeenCalled();
    await live.show(SOURCE, { angle: "30", scale: "0.5" }, 0.5, progress);
    expect(progress).toHaveBeenCalledTimes(20);
    expect(opened()).toBe(1);
    live.unload();
    await live.show(SOURCE, { angle: "30", scale: "0.5" }, 0.5);
    expect(opened()).toBe(2);
    live.dispose();
  });

  test("keeps the clip it has through options it refuses", async () => {
    const { live, canvas } = make();
    const gl = canvas.getContext("webgl2")!;
    const progress = vi.fn();
    const values = { angle: "30", loop: "1", loop_fade: "0.4" };
    await live.show(SOURCE, values, 0.5, progress);
    expect(progress).toHaveBeenCalledTimes(20);
    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    for (const [refused, message] of [
      [{ ...values, loop_fade: "2" },
       "The crossfade at the ends can be at most half the clip, 0.667 s."],
      [{ ...values, scale: String(size) },
       `The live view can hold frames at most ${size} pixels across on this GPU, and this `
       + `clip's are ${32 * size}×${24 * size} at half size. Set a smaller scale.`],
    ] as const) {
      progress.mockClear();
      expect(await problem(live.show(SOURCE, refused, 0.5, progress)))
        .toEqual({ message, kind: "error" });
      await live.show(SOURCE, values, 0.5, progress);
      expect(progress, JSON.stringify(refused)).not.toHaveBeenCalled();
    }
    live.dispose();
  });

  test("shows the clip's first frame as loaded for the cuboid", async () => {
    const { live, canvas } = make();
    expect(live.face()).toBeNull();
    const values = { angle: "0", loop: "1", loop_fade: "0.4", loop_sides: "1",
                     loop_side_fade: "8" };
    await live.show(SOURCE, values, 0);
    const first = drawn(canvas);
    await live.show(SOURCE, { ...values, angle: "70" }, 0.3);
    const face = live.face()!;
    expect([face.width, face.height]).toEqual([28, 24]);
    expect(new Uint8Array(face.data.buffer)).toEqual(first);

    // One face for each clip loaded, so the page can tell when it changes.
    await live.show(SOURCE, { ...values, angle: "20" }, 0.6);
    expect(live.face()).toBe(face);
    await live.show(SOURCE, { ...values, loop_side_fade: "4" }, 0.6);
    const next = live.face()!;
    expect(next).not.toBe(face);
    expect([next.width, next.height]).toEqual([30, 24]);
    expect(live.face()).toBe(next);
    live.unload();
    expect(live.face()).toBeNull();
    live.dispose();
  });

  test("plans a sweep once for all the frames of the same options", async () => {
    const { live } = make();
    const values = { angle: "30", loop: "1", noise: "2" };
    const first = await live.show(SOURCE, values, 0);
    const opts = readOptions(values);
    const sweep = plan(first.volume[0], first.volume[2], opts, liveNoise(opts));
    const scaled = vi.spyOn(Noise.prototype, "scaled");  // as each plan makes its noise
    for (const pos of [0.2, 0.5, 0.9]) {
      const info = await live.show(SOURCE, { ...values }, pos);
      expect({ ...info, frame: 0, line: [] }).toEqual({ ...first, frame: 0, line: [] });
      expect(info.line).toEqual(endpoints(sweep, info.frame));
    }
    expect(scaled).not.toHaveBeenCalled();
    const steeper = await live.show(SOURCE, { ...values, angle: "40" }, 0.5);
    expect(scaled).toHaveBeenCalledTimes(1);
    expect(steeper.first).not.toEqual(first.first);
    live.dispose();
  });

  test("plans afresh for another clip with the same options", async () => {
    const { live } = make(async (source) =>
      source === SOURCE ? clip() : fetch(turned).then((res) => res.blob()));
    const values = { angle: "30" };
    expect((await live.show(SOURCE, values, 0.5)).volume).toEqual([20, 24, 32]);
    expect((await live.show("videos/h264-rot90.mp4", values, 0.5)).volume)
      .toEqual([20, 32, 24]);
    live.dispose();
  });

  test("refuses fades too long for the clip before decoding it", async () => {
    const { live } = make();
    const progress = vi.fn();
    expect(await problem(live.show(SOURCE, { loop: "1", loop_fade: "2" }, 0, progress)))
      .toEqual({ message: "The crossfade at the ends can be at most half the clip, 0.667 s.",
                 kind: "error" });
    expect(await problem(live.show(SOURCE, { loop: "1", loop_sides: "1",
                                             loop_side_fade: "40" }, 0, progress)))
      .toEqual({ message: "The crossfade at the sides can be at most half the width, 32 "
                          + "pixels.", kind: "error" });
    expect(progress).not.toHaveBeenCalled();
    live.dispose();
  });

  test("says why a sweep doesn't fit", async () => {
    const { live } = make();
    const { message, kind } = await problem(live.show(SOURCE, { angle: "80", inside: "1" }, 0));
    expect(kind).toBe("does_not_fit");
    expect(message).toContain("Let black in or Wrap round");
    live.dispose();
  });

  test("says when the GPU can't hold the clip, before decoding it", async () => {
    const { live, canvas } = make();
    const gl = canvas.getContext("webgl2")!;
    const progress = vi.fn();
    vi.spyOn(Volume, "create").mockImplementation(() => { throw new VolumeTooLarge(2.46e9); });
    expect(await problem(live.show(SOURCE, {}, 0, progress))).toEqual({
      message: "The live view would need 2.5 GB of GPU memory for this clip, more than the GPU "
        + "can give it. Set a shorter duration or a smaller scale.",
      kind: "error",
    });
    vi.restoreAllMocks();

    const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
    const scale = String((size + 64) / 32);  // half size is 64 pixels too wide
    expect((await problem(live.show(SOURCE, { scale }, 0, progress))).message)
      .toBe(`The live view can hold frames at most ${size} pixels across on this GPU, and this `
            + `clip's are ${size + 64}×${(size + 64) * 3 / 4} at half size. Set a smaller scale.`);

    const real = gl.getParameter.bind(gl);
    vi.spyOn(gl, "getParameter").mockImplementation((name: GLenum) =>
      name === gl.MAX_ARRAY_TEXTURE_LAYERS ? 10 : real(name));
    expect((await problem(live.show(SOURCE, {}, 0, progress))).message)
      .toBe("The live view can hold at most 10 frames on this GPU, and this clip has 20 at "
            + "half its frame rate. Set a shorter duration.");
    expect(progress).not.toHaveBeenCalled();
    live.dispose();
  });

  test("says when a frame is larger than the GPU can draw", async () => {
    const canvas = document.createElement("canvas");
    made.push(canvas);
    const gl = canvas.getContext("webgl2")!;  // the one the view will take
    const real = gl.getParameter.bind(gl);
    vi.spyOn(gl, "getParameter").mockImplementation((name: GLenum) =>
      name === gl.MAX_VIEWPORT_DIMS ? Int32Array.of(30, 30) : real(name));
    const live = new Live(canvas, async () => clip());
    const advice = ". Set a smaller angle, a shorter duration or a smaller scale.";
    expect(await problem(live.show(SOURCE, { angle: "30" }, 0))).toEqual({
      message: "The live view can't draw frames this large on this GPU: a slice of 38 × 24 "
        + `pixels is more than the GPU's limit of 30 each way${advice}`,
      kind: "error",
    });
    const error = (() => { try { live.face(); } catch (error) { return error; } })();
    expect(error).toBeInstanceOf(Problem);
    expect((error as Problem).message).toBe("The live view can't draw frames this large on this "
      + `GPU: a slice of 32 × 24 pixels is more than the GPU's limit of 30 each way${advice}`);
    live.dispose();
  });

  test("turns a file it can't read into a problem", async () => {
    const { live } = make(async () => new File(["not a video"], "notes.mp4"));
    expect(await problem(live.show(SOURCE, {}, 0)))
      .toEqual({ message: "can't read a video stream from notes.mp4", kind: "error" });
    live.dispose();
  });

  test("refuses sources outside the server's folders", async () => {
    const { live, opened } = make();
    for (const source of ["videos/../webapp.py", "videos/.clip.mp4", "elsewhere/clip.mp4",
                          "videos/", "videos", ""]) {
      expect(await problem(live.show(source, {}, 0)), source)
        .toEqual({ message: "Pick a video first.", kind: "error" });
    }
    expect(opened()).toBe(0);
    live.dispose();
  });

  test("stops a load when unloaded", async () => {
    const { live } = make();
    const showing = live.show(SOURCE, {}, 0);
    live.unload();
    await expect(showing).rejects.toMatchObject({ name: "AbortError" });
    expect((await live.show(SOURCE, {}, 0)).volume).toEqual([20, 24, 32]);
    live.dispose();
  });

  test("says the GPU is lost when it goes before the clip has room", async () => {
    const canvas = document.createElement("canvas");
    made.push(canvas);
    // Keep the news from the live view, as a load runs on until the event comes.
    canvas.addEventListener("webglcontextlost", (event) => event.stopImmediatePropagation());
    const live = new Live(canvas, async () => {
      canvas.getContext("webgl2")!.getExtension("WEBGL_lose_context")!.loseContext();
      return clip();
    });
    expect((await problem(live.show(SOURCE, {}, 0))).message).toContain("lost the GPU");
    live.dispose();
  });

  test("gives up the GPU when it can't make its shaders", () => {
    const canvas = document.createElement("canvas");
    made.push(canvas);
    const gl = canvas.getContext("webgl2")!;
    const real = gl.getShaderParameter.bind(gl);
    vi.spyOn(gl, "getShaderParameter").mockImplementation((shader, name) =>
      gl.getShaderSource(shader) === NOISY.rgba ? false : real(shader, name));
    const error = (() => { try { new Live(canvas); } catch (error) { return error; } })();
    expect(error).toBeInstanceOf(Problem);
    expect((error as Problem).message)
      .toMatch(/^The live view couldn't start on this GPU: shader didn't compile/);
    expect(gl.isContextLost()).toBe(true);  // and with it whatever the shaders left
  });

  test("lets go of its canvas when disposed", async () => {
    const { live, canvas } = make();
    live.dispose();
    const lost = new Promise<Event>((resolve) =>
      canvas.addEventListener("webglcontextlost", resolve));
    canvas.getContext("webgl2")!.getExtension("WEBGL_lose_context")!.loseContext();
    expect((await lost).defaultPrevented).toBe(false);  // nothing waits for it to come back
  });

  test("comes back after losing the GPU", async () => {
    let opening: () => void = () => {};
    const opened = new Promise<void>((resolve) => { opening = resolve; });
    const { live, canvas } = make(async () => {
      await opened;  // held, so the GPU goes while the clip loads
      return clip();
    });
    const lose = canvas.getContext("webgl2")!.getExtension("WEBGL_lose_context")!;
    const loading = live.show(SOURCE, { angle: "30" }, 0.5);
    const lost = new Promise((resolve) => canvas.addEventListener("webglcontextlost", resolve));
    lose.loseContext();
    await lost;
    opening();
    expect((await problem(loading)).message).toContain("lost the GPU");
    expect((await problem(live.show(SOURCE, { angle: "30" }, 0.5))).message)
      .toContain("lost the GPU");
    expect(live.face()).toBeNull();
    const restored = new Promise((resolve) =>
      canvas.addEventListener("webglcontextrestored", resolve));
    await new Promise((resolve) => setTimeout(resolve));  // Chrome won't restore any sooner
    lose.restoreContext();
    await restored;
    const progress = vi.fn();
    await live.show(SOURCE, { angle: "30" }, 0.5, progress);
    expect(progress).toHaveBeenCalled();
    expect(live.face()).not.toBeNull();
    const after = drawn(canvas);
    const shown = make();
    await shown.live.show(SOURCE, { angle: "30" }, 0.5);
    expect(after).toEqual(drawn(shown.canvas));
    shown.live.dispose();
    live.dispose();
  });
});

describe("fetchSource", () => {
  const { signal } = new AbortController();

  test("fetches a source from the server's media", async () => {
    const fetch = vi.spyOn(globalThis, "fetch").mockImplementation(async () =>
      new Response("the bytes"));
    const file = await fetchSource("videos/my clip#1?.mp4", signal);
    expect(fetch).toHaveBeenCalledWith("/media/videos/my%20clip%231%3F.mp4", { signal });
    expect([file.name, await file.text()]).toEqual(["my clip#1?.mp4", "the bytes"]);
    await fetchSource("uploads/clip.mp4", signal);
    expect(fetch).toHaveBeenLastCalledWith("/media/uploads/clip.mp4", { signal });
  });

  test("says why it can't", async () => {
    const fetch = vi.spyOn(globalThis, "fetch")
      .mockResolvedValueOnce(new Response("", { status: 404 }))
      .mockResolvedValueOnce(new Response("", { status: 500 }));
    expect(await problem(fetchSource("uploads/gone.mp4", signal)))
      .toEqual({ message: "gone.mp4 isn't in uploads any more.", kind: "error" });
    expect(await problem(fetchSource("videos/clip.mp4", signal)))
      .toEqual({ message: "The server said 500.", kind: "error" });
    expect(await problem(fetchSource("videos/../webapp.py", signal)))
      .toEqual({ message: "Pick a video first.", kind: "error" });
    expect(fetch).toHaveBeenCalledTimes(2);
  });
});
