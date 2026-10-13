/**
 * Renders a clip in a worker of its own: decodes it onto the GPU, slices every frame of the
 * sweep there, fading a loop's ends as it reads them, and encodes it. A full render keeps the
 * clip at the page's scale and every frame, and blends neighbouring voxels; a preview halves
 * it in x, y and time and copies the nearest. The page ends the worker once it reports, or to
 * cancel, and its GPU memory and encoder go with it.
 */
import { load, planLoad, VideoError, type LoadOptions } from "./decode";
import { Encoder, settings, Unencodable } from "./encode";
import { Slicer } from "./gpu/slicer";
import { Volume, VolumeTooLarge } from "./gpu/volume";
import { looping } from "./noise";
import {
  planFaded, Problem, readOptions, type OptionValues, type ProblemKind,
} from "./options";
import { PREVIEW_SCALE } from "./planner";
import { gcd, limitDenominator } from "./numbers";
import type { Rate } from "./types";

/** What the page asks for: a render of a clip with the page's options. */
export interface RenderRequest { file: Blob; values: OptionValues; preview: boolean }

/** What the worker is doing: decoding the clip, slicing and encoding, or finishing the file. */
export type Stage = "loading" | "slicing" | "finishing";

/** A finished render. */
export interface Made {
  video: Blob; width: number; height: number; frames: number; fps: Rate; rateControl: string;
}

/** What the worker says: how far it has got, then the video, or why it failed. */
export type RenderMessage =
  | { type: "progress"; stage: Stage; done: number; total: number }
  | ({ type: "done" } & Made)
  | { type: "failed"; kind: ProblemKind; message: string };

const LOST = "The GPU gave out, perhaps for want of memory.";

function post(message: RenderMessage): void {
  self.postMessage(message);
}

function failure(error: unknown): RenderMessage {
  const { message } = error instanceof Error ? error : { message: String(error) };
  if (error instanceof Problem) return { type: "failed", kind: error.kind, message };
  if (error instanceof VideoError || error instanceof Unencodable)
    return { type: "failed", kind: "error", message };
  return { type: "failed", kind: "error", message: `The render failed: ${message}` };
}

function times(a: Rate, b: Rate): Rate {
  const [num, den] = [a.num * b.num, a.den * b.den];
  const divisor = gcd(num, den) || 1;
  return { num: num / divisor, den: den / divisor };
}

/** A volume for the clip, frames × height × width, or the Problem of a GPU that can't hold it. */
function allocate(gl: WebGL2RenderingContext, frames: number, height: number, width: number,
                  preview: boolean): Volume {
  if (gl.isContextLost()) throw new Problem(LOST);  // its limits would read as 0
  const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (frames > layers) {
    throw new Problem(`The GPU can hold at most ${layers} frames, and this clip has `
                      + `${frames}${preview ? " at half its frame rate" : ""}. Set a shorter `
                      + "duration.");
  }
  if (width > size || height > size) {
    throw new Problem(`The GPU can hold frames at most ${size} pixels across, and this clip's `
                      + `are ${width}×${height}${preview ? " at half size" : ""}. Set a smaller `
                      + "scale.");
  }
  try {
    return Volume.create(gl, frames, height, width);
  } catch (error) {
    if (!(error instanceof VolumeTooLarge)) throw error;
    throw new Problem(`This clip needs ${(error.bytes / 1e9).toFixed(1)} GB of GPU memory, more `
                      + "than the GPU can give it. Set a shorter duration or a smaller scale.");
  }
}

async function run({ file, values, preview }: RenderRequest,
                   progress: (stage: Stage, done: number, total: number) => void): Promise<Made> {
  const opts = readOptions(values);
  const shrink = preview ? PREVIEW_SCALE : 1;
  const options: LoadOptions = {
    scale: opts.scale * shrink, timeScale: shrink, start: opts.start ?? undefined,
    duration: opts.duration ?? undefined, fast: preview,
  };
  const planned = await planLoad(file, options);
  // Refuse what can't be done before decoding the clip.
  const noise = opts.noise && opts.noise.scaled(shrink);
  const { fade, sweep } = planFaded(planned.frames, planned.width,
                                    planned.fps.num / planned.fps.den, opts, noise, shrink);
  const { height } = planned;
  // A preview has fewer frames, so it plays slower to last as long as the full render.
  const fps = opts.fps ? times(opts.fps, limitDenominator(shrink)) : planned.fps;

  // Frames are drawn into textures, never onto the canvas.
  const canvas = new OffscreenCanvas(1, 1);
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false,
                                           stencil: false });
  // No context: the browser has no WebGL2, or won't give any more to this page, as after the
  // GPU has been reset.
  if (!gl) throw new Problem("The browser won't give this render a WebGL2 context.");
  try {
    // Everything the GPU and encoder need is made before the clip is decoded.
    const slicer = new Slicer(gl);
    if (sweep.width > slicer.maxSize || height > slicer.maxSize) {
      throw new Problem(`The GPU can't make frames ${sweep.width}×${height}, as this render's `
                        + "are. Set a smaller angle, a shorter duration or a smaller scale.");
    }
    const volume = allocate(gl, planned.frames, height, planned.width, preview);
    const encoder = await Encoder.open(gl, sweep.width, height, fps,
                                       settings(sweep.width, height, fps, preview));
    await load(file, options, (i, frame) => {
      volume.upload(i, frame);
      frame.close();
      if (gl.isContextLost()) throw new Problem(LOST);
    }, (done, total) => progress("loading", done, total));

    const surface = noise && looping(noise, sweep);
    const bend = surface && { surface, push: surface.push(sweep.normal) };
    const flat = { nearest: preview, wrap: sweep.loop, wrapX: sweep.sides, fade };
    for (let f = 0; f < sweep.frames; f++) {
      slicer.slice(volume, sweep.width, sweep.at(f),
                   { ...flat, noise: bend ? { ...bend, f } : undefined });
      await encoder.add(slicer.texture);
      if (gl.isContextLost()) throw new Problem(LOST);
      progress("slicing", f + 1, sweep.frames);
    }
    // The clip's GPU memory goes before the encoder flushes.
    slicer.dispose();
    volume.dispose();
    progress("finishing", sweep.frames, sweep.frames);
    const { video, width, height: codedHeight, rateControl } = await encoder.finish();
    return { video, width, height: codedHeight, frames: sweep.frames, fps, rateControl };
  } catch (error) {
    // A lost context shows first as whatever fails, such as a shader that won't compile.
    if (gl.isContextLost() && !(error instanceof Problem)) throw new Problem(LOST);
    throw error;
  }
}

self.addEventListener("message", (event: MessageEvent<RenderRequest>) => {
  run(event.data, (stage, done, total) => post({ type: "progress", stage, done, total }))
    .then((made) => post({ type: "done", ...made }), (error: unknown) => post(failure(error)));
});
