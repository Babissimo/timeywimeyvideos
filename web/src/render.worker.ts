/**
 * Renders a clip in a worker of its own: decodes it onto the GPU, slices every frame of the
 * sweep there, fading a loop's ends as it reads them, and encodes it. A full render keeps the
 * clip at the page's scale and every frame, and blends neighbouring voxels; a preview halves
 * it in x, y and time and copies the nearest. The page ends the worker once it reports, or to
 * cancel, and its GPU memory and encoder go with it.
 */
import { planLoad, VideoError, type LoadOptions } from "./decode";
import { Encoder, even, settings, Unencodable } from "./encode";
import { Slicer } from "./gpu/slicer";
import { LOST, Volume } from "./gpu/volume";
import { looping } from "./noise";
import {
  planFaded, Problem, readOptions, type OptionValues, type ProblemKind,
} from "./options";
import { PREVIEW_SCALE } from "./planner";
import { gcd, limitDenominator } from "./numbers";
import { destination } from "./store";
import type { Rate } from "./types";

/** What the page asks for: a render of a clip with the page's options, and its file name. */
export interface RenderRequest { file: Blob; values: OptionValues; preview: boolean; name: string }

/** What the worker is doing: decoding the clip, slicing and encoding, or finishing the file. */
export type Stage = "loading" | "slicing" | "finishing";

/** A finished render, and whether its video is kept in the origin private file system. */
export interface Made {
  video: Blob; kept: boolean; width: number; height: number; frames: number; fps: Rate;
  rateControl: string;
}

/** What the worker says: how far it has got, then the video, or why it failed. */
export type RenderMessage =
  | { type: "progress"; stage: Stage; done: number; total: number }
  | ({ type: "done" } & Made)
  | { type: "failed"; kind: ProblemKind; message: string };

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

async function run({ file, values, preview, name }: RenderRequest,
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
  // H.264 frames are an even number of pixels wide, so an odd sweep's frames take one more
  // column along their line, where the encoder would otherwise add a black one.
  const width = even(sweep.width);
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
    const out = await destination();
    const slicer = new Slicer(gl);
    if (width > slicer.maxSize || height > slicer.maxSize) {
      throw new Problem(`The GPU can't make frames ${width}×${height}, as this render's `
                        + "are. Set a smaller angle, a shorter duration or a smaller scale.");
    }
    const volume = Volume.create(gl, planned, { preview });
    const encoder = await Encoder.open(gl, width, height, fps,
                                       settings(width, height, fps, preview), out.target);
    await volume.decode(file, options, (done, total) => progress("loading", done, total));

    const surface = noise && looping(noise, sweep);
    const bend = surface && { surface, push: surface.push(sweep.normal) };
    const flat = { nearest: preview, wrap: sweep.loop, wrapX: sweep.sides, fade };
    for (let f = 0; f < sweep.frames; f++) {
      slicer.slice(volume, width, sweep.at(f),
                   { ...flat, noise: bend ? { ...bend, f } : undefined });
      await encoder.add(slicer.texture);
      if (gl.isContextLost()) throw new Problem(LOST);
      progress("slicing", f + 1, sweep.frames);
    }
    // The clip's GPU memory goes before the encoder flushes.
    slicer.dispose();
    volume.dispose();
    progress("finishing", sweep.frames, sweep.frames);
    await encoder.finish();
    const { video, kept } = await out.finish(name);
    return { video, kept, width: encoder.width, height: encoder.height, frames: sweep.frames,
             fps, rateControl: encoder.rateControl };
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
