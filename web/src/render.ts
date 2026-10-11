/**
 * Renders made in the browser: each runs in a worker of its own (render.worker.ts) and is named
 * from its source and the options that differ from the page's defaults.
 */
import { Noise } from "./noise";
import { Problem, readOptions, type OptionValues, type Options } from "./options";
import type { Made, RenderMessage, RenderRequest, Stage } from "./render.worker";

export type { Stage } from "./render.worker";

/** How far a render has got: `done` of `total` frames decoded, or sliced and encoded. */
export interface Progress { stage: Stage; done: number; total: number }

/** A finished render: the video, its file name, and how long it plays in seconds. */
export interface Rendered extends Made { name: string; seconds: number }

export interface RenderOptions {
  preview?: boolean;
  /** Aborting it ends the worker, and the render rejects with the signal's reason. */
  signal?: AbortSignal;
  progress?: (progress: Progress) => void;
  /** Starts the worker. */
  worker?: () => Worker;
}

const startWorker = () =>
  new Worker(new URL("./render.worker.ts", import.meta.url), { type: "module" });

/**
 * Render a video file with the page's options, as a preview or at full quality. Rejects with a
 * Problem for the page to show.
 */
export async function render(file: File, values: OptionValues,
                             { preview = false, signal, progress, worker = startWorker }:
                               RenderOptions = {}): Promise<Rendered> {
  const opts = readOptions(values);
  const name = outputName(file.name, opts, preview);
  signal?.throwIfAborted();
  const thread = worker();
  try {
    return await new Promise<Rendered>((resolve, reject) => {
      signal?.addEventListener("abort", () => reject(signal.reason), { once: true });
      thread.addEventListener("message", ({ data }: MessageEvent<RenderMessage>) => {
        if (signal?.aborted) return;  // a message sent before the worker was ended
        if (data.type === "progress") {
          const { stage, done, total } = data;
          progress?.({ stage, done, total });
        } else if (data.type === "done") {
          const { type: _, ...made } = data;
          resolve({ ...made, name, seconds: made.frames * made.fps.den / made.fps.num });
        } else {
          reject(new Problem(data.message, data.kind));
        }
      });
      thread.addEventListener("error", (event) => {
        reject(new Problem(`The render failed: ${event.message || "its worker stopped"}`));
      });
      thread.postMessage({ file, values, preview } satisfies RenderRequest);
    });
  } finally {
    thread.terminate();
  }
}


/** A file name less its extension. */
function stem(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 && dot < name.length - 1 ? name.slice(0, dot) : name;
}

/**
 * The file name of a render of the file `name`: its stem, the slice and angle, then each option
 * that differs from the page's defaults.
 */
export function outputName(name: string, opts: Options, preview: boolean): string {
  const parts = [stem(name), opts.slice, `${opts.angle}deg`];
  if (opts.inside) parts.push("inside");
  if (opts.motion) parts.push(opts.motion);
  if (opts.loop) {
    parts.push("loop");
    if (opts.loopFade) parts.push(`loopfade${opts.loopFade}s`);
  }
  if (opts.sides) {
    parts.push("sides");
    if (opts.sideFade) parts.push(`sidefade${opts.sideFade}px`);
  }
  const { noise } = opts, plain = new Noise(0);
  if (noise) {
    parts.push(`noise${noise.amplitude}`);
    if (noise.size !== plain.size) parts.push(`noisesize${noise.size}`);
    if (noise.speed !== plain.speed) parts.push(`noisespeed${noise.speed}`);
    if (noise.direction !== plain.direction) parts.push(`noise${noise.direction}`);
    if (noise.seed !== plain.seed) parts.push(`noiseseed${noise.seed}`);
  }
  if (opts.start !== null) parts.push(`from${opts.start}s`);
  if (opts.duration !== null) parts.push(`for${opts.duration}s`);
  if (opts.scale !== 1) parts.push(`scale${opts.scale}`);
  const { fps } = opts;
  if (fps) parts.push(fps.den === 1 ? `${fps.num}fps` : `${fps.num}over${fps.den}fps`);
  if (preview) parts.push("preview");
  return `${parts.join("_")}.mp4`;
}
