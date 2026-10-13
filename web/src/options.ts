/**
 * The page's options, and what the live view makes of them: how to plan its sweep, how much of
 * a loop to crossfade, and what a full-quality render would make.
 */
import { clipSeconds, type ClipInfo } from "./decode";
import { MAX_SEED, Noise } from "./noise";
import { decimal, gcd, roundHalfEven } from "./numbers";
import {
  DoesNotFit, OptionError, planSweep, PREVIEW_SCALE, type Motion, type SliceKind, type Sweep,
} from "./planner";
import type { Rate } from "./types";

/** What kind of problem it is, for the way out the page offers. */
export type ProblemKind = "error" | "does_not_fit" | "sideways";

/** Something the page should show the user. */
export class Problem extends Error {
  override name = "Problem";
  readonly kind: ProblemKind;

  constructor(message: string, kind: ProblemKind = "error") {
    super(message);
    this.kind = kind;
  }
}

/** The page's options by name: strings from its fields, or numbers and booleans. */
export type OptionValues = Readonly<Record<string, unknown>>;

export interface Options {
  slice: SliceKind;
  angle: number;
  inside: boolean;
  motion: Motion | null;
  noise: Noise | null;
  scale: number;
  start: number | null;
  duration: number | null;
  fps: Rate | null;
  loop: boolean;
  sides: boolean;
  /** Seconds of the loop's ends to crossfade; 0 without a loop. */
  loopFade: number;
  /** Pixels of the picture's sides to crossfade; 0 unless they wrap. */
  sideFade: number;
}

/**
 * What a full-quality render with some options would make, and the GPU memory its clip takes,
 * held in YUV 4:2:0.
 */
export interface FullSize { width: number; height: number; frames: number; seconds: number;
                            memory: number }

/** Whether a field was left empty. */
function empty(value: unknown): boolean {
  return value === undefined || value === null || (typeof value === "string" && !value.trim());
}

/** A field's value as a number: NaN where it isn't one. */
function toNumber(value: unknown): number {
  if (typeof value === "number") return value;
  return typeof value === "string" ? Number(value) : NaN;
}

/** A positive frame rate written as a number or a fraction, such as 24, 29.97 or 30000/1001. */
function readRate(text: string): Rate | undefined {
  const match = /^\s*(\d+)(?:\.(\d+))?\s*(?:\/\s*(\d+)\s*)?$/.exec(text);
  if (!match) return undefined;
  const [, whole, fraction = "", over = "1"] = match;
  const num = Number(whole + fraction), den = Number(over) * 10 ** fraction.length;
  if (!(num > 0 && den > 0 && Number.isSafeInteger(num) && Number.isSafeInteger(den)))
    return undefined;
  const divisor = gcd(num, den);
  return { num: num / divisor, den: den / divisor };
}

/** Read the page's options, or throw the Problem with the first value refused. */
export function readOptions(values: OptionValues): Options {
  function number<T extends number | null>(name: string, fallback: T, low = -Infinity,
                                           above?: number): number | T {
    const value = values[name];
    if (empty(value)) return fallback;
    const n = toNumber(value);
    if (Number.isNaN(n)) throw new Problem(`${name} must be a number.`);
    if (!Number.isFinite(n) || n < low || (above !== undefined && n <= above))
      throw new Problem(`${name} is out of range.`);
    return n;
  }

  const flag = (name: string) => ["1", "true"].includes(String(values[name]).toLowerCase());
  const quoted = (value: unknown) => JSON.stringify(value);

  const slice = values.slice || "rotate";
  if (slice !== "rotate" && slice !== "shear")
    throw new Problem(`Unknown slice ${quoted(slice)}.`);
  const loop = flag("loop");
  const sides = loop && flag("loop_sides");
  const inside = flag("inside") && !loop;  // a looping frame is always inside
  let motion = values.motion || null;
  if (motion !== null && motion !== "perpendicular" && motion !== "time" && motion !== "longest")
    throw new Problem(`Unknown motion ${quoted(motion)}.`);
  if (slice === "shear" || !(inside || sides)) {
    motion = null;  // only these rotate frames can move different ways
  } else if (sides && motion === "longest") {
    throw new Problem("A loop round the sides moves through time or perpendicular to the "
                      + "frame: no line on it is the longest.");
  }
  let fps: Rate | null = null;
  if (!empty(values.fps)) {
    fps = readRate(String(values.fps)) ?? null;
    if (!fps) throw new Problem("Output fps must be a positive number, like 24 or 30000/1001.");
  }

  let noise: Noise | null = null;
  const amplitude = number("noise", 0, 0);
  if (amplitude) {
    const plain = new Noise(amplitude);  // for its defaults
    const direction = values.noise_direction || plain.direction;
    if (direction !== "time" && direction !== "perpendicular")
      throw new Problem(`Unknown noise direction ${quoted(direction)}.`);
    const seed = empty(values.noise_seed) ? plain.seed : toNumber(values.noise_seed);
    if (!(Number.isInteger(seed) && seed >= 0 && seed <= MAX_SEED))
      throw new Problem(`The noise seed must be a whole number from 0 to ${MAX_SEED}.`);
    const size = number("noise_size", plain.size, -Infinity, 0);
    const speed = number("noise_speed", plain.speed);
    noise = new Noise(amplitude, { size, speed, direction, seed });
  }

  const angle = number("angle", 45);
  const scale = number("scale", 1, -Infinity, 0);
  const start = number("start", null, 0);
  const duration = number("duration", null, -Infinity, 0);
  const loopFade = loop ? number("loop_fade", 0, 0) : 0;
  const sideFade = sides ? number("loop_side_fade", 0, 0) : 0;
  return { slice, angle, inside, motion, noise, scale, start, duration, fps, loop, sides,
           loopFade, sideFade };
}

/** The noise for the live view's half-size clip. */
export function liveNoise(opts: Options): Noise | null {
  return opts.noise && opts.noise.scaled(PREVIEW_SCALE);
}

/** Plan the sweep, with `noise` scaled to match the clip. */
export function plan(nFrames: number, width: number, opts: Options,
                     noise: Noise | null = null): Sweep {
  const { slice, angle, inside, motion, loop, sides } = opts;
  try {
    return planSweep(nFrames, width, { slice, angle, inside, motion, noise, loop, sides });
  } catch (error) {
    if (error instanceof DoesNotFit) {
      if (loop) {  // only sideways noise can leave a loop's frame
        throw new Problem(`${error.message} Use less noise, push it through time, or wrap `
                          + "round the sides too.", "sideways");
      }
      const smaller = noise ? "a smaller angle or less noise" : "a smaller angle";
      throw new Problem(`${error.message} Use a longer clip or ${smaller}, or choose `
                        + "Let black in or Wrap round at the video's edges.", "does_not_fit");
    }
    if (error instanceof OptionError) throw new Problem(error.message);
    throw error;
  }
}

/**
 * How many frames and columns the loop's crossfades blend, for a clip of nFrames at `rate`
 * frames a second, `width` pixels wide once shrunk by `shrink`.
 */
export function loopFades(nFrames: number, width: number, rate: number, opts: Options,
                          shrink = 1): [number, number] {
  const frames = Math.round(opts.loopFade * rate);
  const columns = Math.round(opts.sideFade * shrink);
  if (2 * frames > nFrames) {
    throw new Problem("The crossfade at the ends can be at most half the clip, "
                      + `${decimal(nFrames / 2 / rate, 3)} s.`);
  }
  if (2 * columns > width) {
    throw new Problem("The crossfade at the sides can be at most half the width, "
                      + `${decimal(width / shrink / 2)} pixels.`);
  }
  return [frames, columns];
}

/**
 * The loop's crossfades for a clip of nFrames at `rate` frames a second, `width` pixels wide
 * once shrunk by `shrink`, and the sweep planned on what they leave of it, with `noise` scaled
 * to match the clip.
 */
export function planFaded(nFrames: number, width: number, rate: number, opts: Options,
                          noise: Noise | null, shrink = 1):
    { fade: [number, number]; sweep: Sweep } {
  const fade = loopFades(nFrames, width, rate, opts, shrink);
  return { fade, sweep: plan(nFrames - fade[0], width - fade[1], opts, noise) };
}

/** Where output frame f's first and last columns come from: [t0, x0, t1, x1]. */
export function endpoints(sweep: Sweep, f: number): [number, number, number, number] {
  const { origin: [t, x], step: [dt, dx] } = sweep.at(f), last = sweep.width - 1;
  return [t, x, t + last * dt, x + last * dx];
}

/**
 * Roughly what a full-quality render with these options would make of a clip, why it can't,
 * or null if the clip's length is unknown and no duration is set.
 */
export function fullSize(clip: ClipInfo, opts: Options): FullSize | { error: string } | null {
  const seconds = clipSeconds(clip.duration, opts.start ?? undefined, opts.duration ?? undefined);
  if (seconds === null) return null;
  // Scaled as decode.ts scales the clip.
  const width = Math.max(1, roundHalfEven(clip.width * opts.scale));
  const height = Math.max(1, roundHalfEven(clip.height * opts.scale));
  const rate = clip.fps.num / clip.fps.den;
  const nFrames = Math.max(1, Math.round(seconds * rate));
  let sweep: Sweep;
  try {
    ({ sweep } = planFaded(nFrames, width, rate, opts, opts.noise));
  } catch (error) {
    if (error instanceof Problem) return { error: error.message };
    throw error;
  }
  const out = opts.fps ?? clip.fps;
  // Luma a byte a pixel, and Cb and Cr a byte each for every 2×2 block.
  const memory = nFrames * (height * width + 2 * Math.ceil(height / 2) * Math.ceil(width / 2));
  return { width: sweep.width, height, frames: sweep.frames,
           seconds: sweep.frames * out.den / out.num, memory };
}
