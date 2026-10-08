/**
 * The page's options in timeslice.py's terms, and what the live view makes of them: how to
 * plan its sweep, how much of a loop to crossfade, and what a full-quality render would make.
 */
import { clipSeconds, type ClipInfo } from "./decode";
import { Noise } from "./noise";
import {
  DoesNotFit, OptionError, planSweep, PREVIEW_SCALE, type Motion, type SliceKind, type Sweep,
} from "./planner";
import { formatG, pyRepr, roundHalfEven } from "./pymath";
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

const DIGITS = String.raw`\d(?:_?\d)*`;  // underscores only between digits
const MAX_DIGITS = 4300;  // the most int() reads, as sys.get_int_max_str_digits() gives
const FLOAT = new RegExp(String.raw`^[+-]?(?:(?:${DIGITS}(?:\.(?:${DIGITS})?)?|\.${DIGITS})`
                         + String.raw`(?:e[+-]?${DIGITS})?|inf(?:inity)?|nan)$`, "i");
const INT = new RegExp(String.raw`^[+-]?${DIGITS}$`);
// Python's fractions._RATIONAL_FORMAT.
const FRACTION = new RegExp(String.raw`^\s*([-+]?)(?=\d|\.\d)(\d*|\d+(?:_\d+)*)`
                            + String.raw`(?:(?:\s*/\s*(\d+(?:_\d+)*))?`
                            + String.raw`|(?:\.(\d*|\d+(?:_\d+)*))?(?:E([-+]?\d+(?:_\d+)*))?)\s*$`,
                            "i");

/** A value as Python's float() reads it, or undefined where float() would raise. */
function pyFloat(value: unknown): number | undefined {
  if (typeof value === "number") return value;
  if (typeof value === "boolean") return Number(value);
  if (typeof value !== "string") return undefined;
  const text = value.trim();
  if (!FLOAT.test(text)) return undefined;
  const word = text.replace(/^[+-]/, "").toLowerCase();
  if (word === "nan") return NaN;
  if (word.startsWith("inf")) return text.startsWith("-") ? -Infinity : Infinity;
  return Number(text.replaceAll("_", ""));
}

/** A value as Python's int() reads it, or undefined where int() would raise. */
function pyInt(value: unknown): bigint | undefined {
  if (typeof value === "number")
    return Number.isFinite(value) ? BigInt(Math.trunc(value)) : undefined;
  if (typeof value === "boolean") return BigInt(value);
  if (typeof value !== "string" || !INT.test(value.trim())) return undefined;
  const text = value.trim().replaceAll("_", "");
  return text.replace(/^[+-]/, "").length <= MAX_DIGITS ? BigInt(text) : undefined;
}

/**
 * A string as Python's Fraction() reads it, or undefined where Fraction() would raise or a
 * double can't hold the numerator or denominator.
 */
function pyFraction(text: string): Rate | undefined {
  const match = FRACTION.exec(text);
  if (!match) return undefined;
  const [, sign, whole = "", over, decimal = "", exp = "0"] =
    match.map((group) => group?.replaceAll("_", ""));
  if ([whole, over ?? "", decimal, exp.replace(/^[-+]/, "")].some((d) => d.length > MAX_DIGITS))
    return undefined;
  let num: bigint, den: bigint;
  if (over) {
    num = BigInt(whole || "0");
    den = BigInt(over);
    if (den === 0n) return undefined;
  } else {
    // The digits times 10 to the power shift.
    const digits = whole + decimal, shift = Number(exp) - decimal.length;
    // Past these, a double can't hold the numerator or denominator.
    if (shift > 308 || -shift > 308 + digits.length) return undefined;
    num = BigInt(digits || "0") * 10n ** BigInt(Math.max(shift, 0));
    den = 10n ** BigInt(Math.max(-shift, 0));
  }
  if (sign === "-") num = -num;
  let [a, b] = [num < 0n ? -num : num, den];  // their greatest common divisor, once b is 0
  while (b) [a, b] = [b, a % b];
  const rate = { num: Number(num / a), den: Number(den / a) };
  return Number.isFinite(rate.num) && Number.isFinite(rate.den) ? rate : undefined;
}

/** Python's str() of a value, as far as the page's values go. */
function pyStr(value: unknown): string {
  if (value === undefined) return "";
  if (value === null) return "None";
  if (typeof value === "boolean") return value ? "True" : "False";
  return String(value);
}

const repr = (value: unknown) => typeof value === "string" ? pyRepr(value) : pyStr(value);

/**
 * Read the page's options, as webapp.read_options does: the same defaults, and the same
 * Problem for a value it refuses.
 */
export function readOptions(values: OptionValues): Options {
  function number<T extends number | null>(name: string, fallback: T, low = -Infinity,
                                           above?: number): number | T {
    const value = values[name];
    if (value === undefined || value === null || value === "") return fallback;
    const n = pyFloat(value);
    if (n === undefined) throw new Problem(`${name} must be a number.`);
    if (!Number.isFinite(n) || n < low || (above !== undefined && n <= above))
      throw new Problem(`${name} is out of range.`);
    return n;
  }

  const flag = (name: string) => ["1", "true"].includes(pyStr(values[name]).toLowerCase());

  const slice = values.slice || "rotate";
  if (slice !== "rotate" && slice !== "shear") throw new Problem(`Unknown slice ${repr(slice)}.`);
  const loop = flag("loop");
  const sides = loop && flag("loop_sides");
  const inside = flag("inside") && !loop;  // a looping frame is always inside
  let motion = values.motion || null;
  if (motion !== null && motion !== "perpendicular" && motion !== "time" && motion !== "longest")
    throw new Problem(`Unknown motion ${repr(motion)}.`);
  if (slice === "shear" || !(inside || sides)) {
    motion = null;  // only these rotate frames can move different ways
  } else if (sides && motion === "longest") {
    throw new Problem("A loop round the sides moves through time or perpendicular to the "
                      + "frame: no line on it is the longest.");
  }
  const fpsValue = values.fps || null;
  let fps: Rate | null = null;
  if (fpsValue !== null) {
    const rate = pyFraction(pyStr(fpsValue));
    if (rate === undefined || rate.num <= 0)
      throw new Problem("Output fps must be a positive number, like 24 or 30000/1001.");
    fps = rate;
  }

  let noise: Noise | null = null;
  const amplitude = number("noise", 0, 0);
  if (amplitude) {
    const plain = new Noise(amplitude);  // for its defaults
    const direction = values.noise_direction || plain.direction;
    if (direction !== "time" && direction !== "perpendicular")
      throw new Problem(`Unknown noise direction ${repr(direction)}.`);
    let seed = plain.seed;
    const given = values.noise_seed;
    if (given !== undefined && given !== null && given !== "") {
      const whole = pyInt(given);
      if (whole === undefined || whole < 0n)
        throw new Problem("The noise seed must be a whole number, 0 or more.");
      // Python reads larger seeds, but a double can't hold them exactly.
      if (whole > BigInt(Number.MAX_SAFE_INTEGER)) {
        throw new Problem("The noise seed must be a whole number from 0 to "
                          + `${Number.MAX_SAFE_INTEGER}.`);
      }
      seed = Number(whole);
    }
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
  const frames = roundHalfEven(opts.loopFade * rate);
  const columns = roundHalfEven(opts.sideFade * shrink);
  if (2 * frames > nFrames) {
    throw new Problem("The crossfade at the ends can be at most half the clip, "
                      + `${formatG(nFrames / 2 / rate, 3)} s.`);
  }
  if (2 * columns > width) {
    throw new Problem("The crossfade at the sides can be at most half the width, "
                      + `${formatG(width / shrink / 2)} pixels.`);
  }
  return [frames, columns];
}

/** Where output frame f's first and last columns come from: [t0, x0, t1, x1]. */
export function endpoints(sweep: Sweep, f: number): [number, number, number, number] {
  const { t, x } = sweep.at(f);
  return [t[0], x[0], t[t.length - 1], x[x.length - 1]];
}

/**
 * Roughly what a full-quality render with these options would make of a clip, why it can't,
 * or null if the clip's length is unknown and no duration is set.
 */
export function fullSize(clip: ClipInfo, opts: Options): FullSize | { error: string } | null {
  const seconds = clipSeconds(clip.duration, opts.start ?? undefined, opts.duration ?? undefined);
  if (seconds === null) return null;
  const width = Math.max(1, roundHalfEven(clip.width * opts.scale));
  const height = Math.max(1, roundHalfEven(clip.height * opts.scale));
  const rate = clip.fps.num / clip.fps.den;
  const nFrames = Math.max(1, roundHalfEven(seconds * rate));
  let sweep: Sweep;
  try {
    const [frames, columns] = loopFades(nFrames, width, rate, opts);
    sweep = plan(nFrames - frames, width - columns, opts, opts.noise);
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
