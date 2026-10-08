/**
 * Renders a clip as timeslice.py does, in a worker of its own: decodes it onto the GPU, fades
 * a loop's ends, slices every frame of the sweep there and encodes it. A full render keeps the
 * clip at the page's scale and every frame, and blends neighbouring voxels; a preview, as
 * `--preview` does, halves it in x, y and time and copies the nearest. The page ends the worker
 * once it reports, or to cancel, and its GPU memory and encoder go with it.
 */
import { load, planLoad, VideoError, type LoadOptions } from "./decode";
import { Encoder, settings, Unencodable } from "./encode";
import { Slicer, type SliceOptions } from "./gpu/slicer";
import { Volume, VolumeTooLarge } from "./gpu/volume";
import { looping } from "./noise";
import {
  loopFades, plan, Problem, readOptions, type OptionValues, type ProblemKind,
} from "./options";
import { PREVIEW_SCALE } from "./planner";
import { gcd, limitDenominator } from "./pymath";
import type { Columns, Rate } from "./types";
import { Unrenderable } from "./unrenderable";

/** What the page asks for: a render of a clip with the page's options. */
export interface RenderRequest { file: Blob; values: OptionValues; preview: boolean }

/** What the worker is doing: decoding the clip, slicing and encoding, or finishing the file. */
export type Stage = "loading" | "slicing" | "finishing";

/** A finished render. */
export interface Made {
  video: Blob; width: number; height: number; frames: number; fps: Rate; rateControl: string;
}

/**
 * What the worker says: how far it has got, then the video, or why it failed. A failure of
 * kind "unrenderable" is a clip the browser can't render but timeslice.py can.
 */
export type RenderMessage =
  | { type: "progress"; stage: Stage; done: number; total: number }
  | ({ type: "done" } & Made)
  | { type: "failed"; kind: ProblemKind | "unrenderable"; message: string };

const LOST = "The GPU gave out, perhaps for want of memory.";

// How decode.ts says the browser can't open a container, or has no decoder for a codec, where
// ffmpeg may still read the file.
const UNREADABLE = /: the browser (can't open|has no decoder)/;

function post(message: RenderMessage): void {
  self.postMessage(message);
}

/** A message as a sentence, as the page's note on timeslice.py follows it: capital, full stop. */
function sentence(message: string): string {
  const stop = /[.!?]$/.test(message) ? "" : ".";
  return `${message.charAt(0).toUpperCase()}${message.slice(1)}${stop}`;
}

function failure(error: unknown): RenderMessage {
  const { message } = error instanceof Error ? error : { message: String(error) };
  if (error instanceof Unrenderable || error instanceof Unencodable
      || (error instanceof VideoError && UNREADABLE.test(message)))
    return { type: "failed", kind: "unrenderable", message: sentence(message) };
  if (error instanceof Problem) return { type: "failed", kind: error.kind, message };
  if (error instanceof VideoError) return { type: "failed", kind: "error", message };
  return { type: "failed", kind: "error", message: `The render failed: ${message}` };
}

function times(a: Rate, b: Rate): Rate {
  const [num, den] = [a.num * b.num, a.den * b.den];
  const divisor = gcd(num, den) || 1;
  return { num: num / divisor, den: den / divisor };
}

/**
 * A volume for the clip, frames × height × width, on the context of a canvas the size of an
 * output frame; or Unrenderable if the GPU can't hold either.
 */
function allocate(gl: WebGL2RenderingContext, frames: number, height: number, width: number,
                  preview: boolean): Volume {
  if (gl.isContextLost()) throw new Unrenderable(LOST);  // its limits would read as 0
  const layers = gl.getParameter(gl.MAX_ARRAY_TEXTURE_LAYERS) as number;
  const size = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  const { canvas } = gl;
  // A canvas larger than the GPU can draw gets a smaller drawing buffer instead.
  if (gl.drawingBufferWidth !== canvas.width || gl.drawingBufferHeight !== canvas.height
      || canvas.width > size) {
    throw new Unrenderable(`The GPU can't make frames ${canvas.width}×${canvas.height}, as `
                           + "this render's are.");
  }
  if (frames > layers) {
    throw new Unrenderable(`The GPU can hold at most ${layers} frames, and this clip has `
                           + `${frames}${preview ? " at half its frame rate" : ""}.`);
  }
  if (width > size || height > size) {
    throw new Unrenderable(`The GPU can hold frames at most ${size} pixels across, and this `
                           + `clip's are ${width}×${height}${preview ? " at half size" : ""}.`);
  }
  try {
    return Volume.create(gl, frames, height, width, { format: "yuv420" });
  } catch (error) {
    if (!(error instanceof VolumeTooLarge)) throw error;
    throw new Unrenderable(`This clip needs ${(error.bytes / 1e9).toFixed(1)} GB of GPU memory, `
                           + "more than the GPU can give it.");
  }
}

/** Slice a frame, or throw Unrenderable for one larger than the GPU can draw. */
function slice(slicer: Slicer, volume: Volume, columns: Columns, options: SliceOptions): void {
  try {
    slicer.slice(volume, columns, options);
  } catch (error) {
    if (!(error instanceof RangeError)) throw error;
    throw new Unrenderable(`The GPU can't draw this render's frames: ${error.message}.`);
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
  const [fadeFrames, fadeColumns] = loopFades(planned.frames, planned.width,
                                              planned.fps.num / planned.fps.den, opts, shrink);
  const noise = opts.noise && opts.noise.scaled(shrink);
  const sweep = plan(planned.frames - fadeFrames, planned.width - fadeColumns, opts, noise);
  const { height } = planned;
  // A preview has fewer frames, so it plays slower to last as long as the full render.
  const fps = opts.fps ? times(opts.fps, limitDenominator(shrink)) : planned.fps;

  const canvas = new OffscreenCanvas(sweep.width, height);
  const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false,
                                           stencil: false });
  // No context: the browser has no WebGL2, or won't give any more to this page, as after the
  // GPU has been reset.
  if (!gl) throw new Unrenderable("The browser won't give this render a WebGL2 context.");
  try {
    // Everything the GPU and encoder need is made before the clip is decoded.
    const volume = allocate(gl, planned.frames, height, planned.width, preview);
    const slicer = new Slicer(gl);
    const encoder = await Encoder.open(gl, fps, settings(sweep.width, height, fps, preview));
    await load(file, options, (i, frame) => {
      volume.upload(i, frame);
      frame.close();
      if (gl.isContextLost()) throw new Unrenderable(LOST);
    }, (done, total) => progress("loading", done, total));
    volume.crossfade(fadeFrames, 0);
    volume.crossfade(fadeColumns, 2);

    const surface = noise && looping(noise, sweep);
    const flat = { nearest: preview, wrap: sweep.loop, wrapX: sweep.sides };
    for (let f = 0; f < sweep.frames; f++) {
      const columns = sweep.at(f);
      slice(slicer, volume, columns, !surface ? flat : { ...flat, noise: {
        push: surface.push(sweep.normal), grid: surface.grid(sweep.width, height, f),
      } });
      slicer.draw();
      await encoder.add();
      if (gl.isContextLost()) throw new Unrenderable(LOST);
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
    if (gl.isContextLost() && !(error instanceof Unrenderable)) throw new Unrenderable(LOST);
    throw error;
  }
}

self.addEventListener("message", (event: MessageEvent<RenderRequest>) => {
  run(event.data, (stage, done, total) => post({ type: "progress", stage, done, total }))
    .then((made) => post({ type: "done", ...made }), (error: unknown) => post(failure(error)));
});
