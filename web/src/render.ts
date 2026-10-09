/**
 * Renders made in the browser: each runs in a worker of its own (render.worker.ts) and is named
 * from its source and the options that differ from timeslice.py's defaults. A clip the browser
 * can't render is left to timeslice.py, whose command the page offers instead.
 */
import { Noise } from "./noise";
import { Problem, readOptions, type OptionValues, type Options } from "./options";
import { formatG } from "./pymath";
import type { Made, RenderMessage, RenderRequest, Stage } from "./render.worker";
import { Unrenderable } from "./unrenderable";

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
 * Problem for the page to show: an Unrenderable, with its command, for a clip the browser can't
 * render.
 */
export async function render(file: File, values: OptionValues,
                             { preview = false, signal, progress, worker = startWorker }:
                               RenderOptions = {}): Promise<Rendered> {
  const opts = readOptions(values);
  const fps = opts.fps && String(values.fps);
  const name = outputName(file.name, opts, fps, preview);
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
        } else if (data.kind === "unrenderable") {
          const command = renderCommand(file.name, name, opts, fps, preview);
          reject(new Unrenderable(data.message, shellJoin(command)));
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

/**
 * What the page says of a clip the browser can't render: why, and that timeslice.py can, with
 * the command shown after it.
 */
export function handOver(error: Unrenderable): string {
  return `${error.message} timeslice.py can render it: run this beside it, with the video's `
    + "path in place of its name.";
}

const g = (x: number) => formatG(x);

/** A file name's stem, as Python's Path.stem gives it. */
function stem(name: string): string {
  const dot = name.lastIndexOf(".");
  return dot > 0 && dot < name.length - 1 ? name.slice(0, dot) : name;
}

/**
 * The file name of a render of the file `name`: its stem, the slice and angle, then each option
 * that differs from timeslice.py's defaults. `fps` is the output frame rate as the page gave
 * it, or null.
 */
export function outputName(name: string, opts: Options, fps: string | null,
                           preview: boolean): string {
  const parts = [stem(name), opts.slice, `${g(opts.angle)}deg`];
  if (opts.inside) parts.push("inside");
  if (opts.motion) parts.push(opts.motion);
  if (opts.loop) {
    parts.push("loop");
    if (opts.loopFade) parts.push(`loopfade${g(opts.loopFade)}s`);
  }
  if (opts.sides) {
    parts.push("sides");
    if (opts.sideFade) parts.push(`sidefade${g(opts.sideFade)}px`);
  }
  const { noise } = opts, plain = new Noise(0);
  if (noise) {
    parts.push(`noise${g(noise.amplitude)}`);
    if (noise.size !== plain.size) parts.push(`noisesize${g(noise.size)}`);
    if (noise.speed !== plain.speed) parts.push(`noisespeed${g(noise.speed)}`);
    if (noise.direction !== plain.direction) parts.push(`noise${noise.direction}`);
    if (noise.seed !== plain.seed) parts.push(`noiseseed${noise.seed}`);
  }
  if (opts.start !== null) parts.push(`from${g(opts.start)}s`);
  if (opts.duration !== null) parts.push(`for${g(opts.duration)}s`);
  if (opts.scale !== 1) parts.push(`scale${g(opts.scale)}`);
  if (fps) parts.push(`${fps.replaceAll("/", "over")}fps`);
  if (preview) parts.push("preview");
  return `${parts.join("_")}.mp4`;
}

/**
 * The timeslice.py command that renders the file `name` to `output` as these options ask, run
 * with uv from beside timeslice.py.
 */
export function renderCommand(name: string, output: string, opts: Options, fps: string | null,
                              preview: boolean): string[] {
  const command = ["uv", "run", "timeslice.py", name, output, `--angle=${g(opts.angle)}`,
                   `--slice=${opts.slice}`];
  if (opts.inside) command.push("--inside");
  if (opts.motion) command.push(`--motion=${opts.motion}`);
  if (opts.loop) command.push("--loop", `--loop-fade=${g(opts.loopFade)}`);
  if (opts.sides) command.push("--loop-sides", `--loop-side-fade=${g(opts.sideFade)}`);
  const { noise } = opts;
  if (noise) {
    command.push(`--noise=${g(noise.amplitude)}`, `--noise-size=${g(noise.size)}`,
                 `--noise-speed=${g(noise.speed)}`, `--noise-direction=${noise.direction}`,
                 `--noise-seed=${noise.seed}`);
  }
  if (opts.scale !== 1) command.push(`--scale=${g(opts.scale)}`);
  if (opts.start !== null) command.push(`--start=${g(opts.start)}`);
  if (opts.duration !== null) command.push(`--duration=${g(opts.duration)}`);
  if (fps) command.push(`--fps=${fps}`);
  if (preview) command.push("--preview");
  return command;
}

/** An argument quoted for a POSIX shell where it needs it, as Python's shlex.quote does. */
export function shellQuote(arg: string): string {
  if (!arg) return "''";
  if (!/[^\w@%+=:,./-]/.test(arg)) return arg;
  return `'${arg.replaceAll("'", `'"'"'`)}'`;
}

/** A command as one line for a POSIX shell, as Python's shlex.join gives it. */
export function shellJoin(command: string[]): string {
  return command.map(shellQuote).join(" ");
}
