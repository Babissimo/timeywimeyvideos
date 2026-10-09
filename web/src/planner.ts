// Slice planning. Each function returns a Sweep: the output frame width, the number of output
// frames, at(f), which gives the (t, x) source position of every column of output frame f, the
// (t, x) normal of the frame's plane, which perpendicular noise pushes along, and whether it
// loops. A loop runs round time in a ring, so at(f) can lie past either end of the clip, and is
// read with wrap. With sides, x wraps round too (wrapX).
//
// Given noise, a plan allows for it pushing points as far as it can: the whole-plane sweep
// starts and ends early and late enough to catch the bumps, and inside and loop frames keep far
// enough from the edges that none leave the video.

import { LruCache } from "./lru";
import {
  degrees, formatG, gcd, hypot, ieeeRemainder, pyRepr, radians, roundHalfEven, TAU,
} from "./pymath";
import type { Columns } from "./types";

export const PREVIEW_SCALE = 0.5;  // a preview shrinks x, y and t by this factor

/** An option, or a combination of options, that can't be planned. */
export class OptionError extends Error {
  name = "OptionError";
}

/** The frame can't fit inside the video at the requested angle. */
export class DoesNotFit extends OptionError {
  name = "DoesNotFit";
}

/** x, unless it is NaN or infinite: Python's round() and int() raise on those, as a plan does. */
export function refuseNonFinite(x: number): number {
  if (Number.isNaN(x)) throw new OptionError("cannot convert float NaN to integer");
  if (!Number.isFinite(x)) throw new OptionError("cannot convert float infinity to integer");
  return x;
}

/** A direction or a move in the x-t plane, as (t, x). */
export type TX = [number, number];

export interface Sweep {
  width: number;
  frames: number;
  /** The source position of every column of output frame f, in new arrays. */
  at(f: number): Columns;
  normal: TX;
  loop: boolean;
  sides: boolean;
}

/**
 * What a plan needs of the noise (a Noise has it): the (t, x) move of a point where the noise
 * is strongest, on a frame whose plane has this (t, x) unit normal.
 */
export interface NoisePush {
  push(normal: Readonly<TX>): Readonly<TX>;
}

export type Motion = "perpendicular" | "time" | "longest";
export type SliceKind = "rotate" | "shear";

export interface ShearOptions {
  inside?: boolean;
  noise?: NoisePush | null;
  loop?: boolean;
  sides?: boolean;
}

export interface RotationOptions extends ShearOptions {
  motion?: Motion | null;
}

export interface PlanOptions extends RotationOptions {
  slice?: SliceKind;
  angle?: number;
}

function push(noise: NoisePush | null | undefined, normal: TX): Readonly<TX> {
  return noise == null ? [0, 0] : noise.push(normal);
}

/**
 * Say which angles from 0 to `limit` degrees `fits` accepts, to the nearest 0.1. (Inside
 * frames fit at -a and 180 - a just as at a.)
 */
function anglesThatFit(fits: (angle: number) => boolean, limit: number): string {
  const runs: [number, number][] = [];  // [first, last] in tenths of a degree
  const last = roundHalfEven(limit * 10);
  for (let tenth = 0; tenth <= last; tenth++) {
    if (!fits(tenth / 10)) continue;
    const run = runs[runs.length - 1];
    if (run && run[1] === tenth - 1) run[1] = tenth;
    else runs.push([tenth, tenth]);
  }
  if (!runs.length) return "No angle fits this clip.";
  if (runs.length === 1 && runs[0][0] === 0 && runs[0][1] === 0) {
    return "Only 0 degrees fits this clip.";
  }
  if (runs.length === 1 && runs[0][0] === 0) {
    return `Angles up to ${formatG(runs[0][1] / 10)} degrees fit this clip.`;
  }
  const spans = runs.map(([a, b]) =>
    a === b ? formatG(a / 10) : `${formatG(a / 10)} to ${formatG(b / 10)}`);
  return `Angles of ${spans.join(" and ")} degrees fit this clip.`;
}

/** Why a frame covering spanT frames of time doesn't fit in the clip. */
function tooLong(angle: number, width: number, spanT: number, pushT: number, nFrames: number,
                 kind = ""): string {
  const noise = pushT !== 0
    ? `, and the noise can push it ${formatG(Math.abs(pushT), 3)} frames either way` : "";
  return `At ${formatG(angle)} degrees a ${width}-pixel-wide ${kind}frame covers `
    + `${Math.ceil(spanT) + 1} frames of time${noise}, but the clip has only ${nFrames}.`;
}

export const LOOP_TOLERANCE = 2.0;  // degrees a loop round the sides may stray from its aim
export const LOOP_WINDINGS = 16;    // most times across the width, or round time, it tries

/** How many times a loop goes across the width, and round time. */
export type Windings = [across: number, roundT: number];

const windingsCache = new LruCache<string, Windings>(64);  // the live view plans every frame

/**
 * How many times (a, b) a loop through a clip whose sides and ends wrap round should go across
 * the width and round time, to close up while moving about along (dx, dt): the shortest within
 * LOOP_TOLERANCE degrees of it, or failing that the closest.
 */
export function windings(width: number, nFrames: number, dx: number, dt: number): Windings {
  const key = `${width},${nFrames},${dx},${dt}`;  // -0 and 0 share an entry
  const cached = windingsCache.get(key);
  if (cached) return [...cached];
  const aim = Math.atan2(dt, dx);
  let bestKey: [rank: number, cost: number] | null = null;
  let best: Windings = [0, 0];
  for (let a = -LOOP_WINDINGS; a <= LOOP_WINDINGS; a++) {
    for (let b = -LOOP_WINDINGS; b <= LOOP_WINDINGS; b++) {
      if (gcd(a, b) !== 1) continue;  // not a closed path, or one gone round twice
      const off = Math.abs(ieeeRemainder(Math.atan2(b * nFrames, a * width) - aim, TAU));
      const length = hypot(a * width, b * nFrames);
      const k: [number, number] = degrees(off) <= LOOP_TOLERANCE ? [0, length] : [1, off];
      if (bestKey === null || k[0] < bestKey[0] || (k[0] === bestKey[0] && k[1] < bestKey[1])) {
        bestKey = k;
        best = [a, b];
      }
    }
  }
  windingsCache.set(key, best);
  return [...best];
}

function pushedSideways(angle: number, pushX: number): string {
  return `At ${formatG(angle)} degrees the noise can push the ends of the frame `
    + `${formatG(Math.abs(pushX), 3)} pixels sideways, out of the video.`;
}

/** cos and sin of the angle, rounded so that 0, 90, 180... degrees land exactly on the grid. */
export function rotation(angle: number): TX {
  const theta = radians(angle);
  // Python has no cosine of infinity.
  if (Math.abs(theta) === Infinity) throw new OptionError("math domain error");
  return [roundHalfEven(Math.cos(theta), 12), roundHalfEven(Math.sin(theta), 12)];
}

/**
 * How far an inside frame's centre can move from the cuboid's centre, in t and in x, before
 * any point of it leaves the video. Negative if it can't fit at all.
 */
function rotationRoom(nFrames: number, width: number, angle: number,
                      noise?: NoisePush | null): TX {
  const [c, s] = rotation(angle);
  const [pushT, pushX] = push(noise, [c, -s]);
  const spanT = (width - 1) * Math.abs(s);  // frames of time one output frame covers
  return [(nFrames - 1 - spanT) / 2 - Math.abs(pushT),
          (width - 1) * (1 - Math.abs(c)) / 2 - Math.abs(pushX)];
}

/**
 * Plan a sweep with the slicing plane rotated `angle` degrees about y.
 *
 * Without `inside`, the frame is wide enough to hold the plane's whole cut through the cuboid,
 * and the plane moves perpendicular to itself from where it first touches the cuboid to where
 * it leaves.
 *
 * With `inside`, the frame is as wide as the input and stays entirely inside the cuboid,
 * moving along a straight line through the cuboid's centre: "perpendicular" to itself,
 * straight through "time", or along the "longest" line that fits (the default). Throws
 * DoesNotFit if the clip is too short for the angle.
 *
 * With `loop`, time runs round in a ring, the clip's first frame following its last. The frame
 * is as wide as the input and moves straight through time once round the ring, so the last
 * output frame leads back into the first. It stays inside the video, so `inside` makes no
 * difference, and no clip is too short.
 *
 * With `sides` as well, x wraps round too, the picture's left edge following its right, so the
 * frame can also move "perpendicular" to itself. It then goes across the width and round time
 * each a whole number of times, along the shortest such path within LOOP_TOLERANCE degrees of
 * perpendicular, and no noise can take it out of the video.
 *
 * With `noise`, the sweep allows for the noise pushing points off the plane.
 */
export function rotationSweep(nFrames: number, width: number, angle: number,
                              { inside = false, motion = null, noise = null, loop = false,
                                sides = false }: RotationOptions = {}): Sweep {
  if (sides && !loop) throw new OptionError("the sides only wrap round on a loop");
  const [c, s] = rotation(angle);
  const normal: TX = [c, -s];  // the way the whole plane sweeps
  const [pushT, pushX] = push(noise, normal);
  // Move forward in time unless the plane is turned past 90 degrees, where the sweep runs
  // backwards (180 degrees plays the clip in reverse).
  const signT = c >= 0 ? 1 : -1;
  let outWidth: number, outFrames: number, dx: number, dt: number;

  if (loop) {
    if (motion != null && motion !== "time" && !(sides && motion === "perpendicular")) {
      throw new OptionError("a loop moves straight through time, or with the sides wrapping, "
                            + "perpendicular to the frame");
    }
    if (!sides && rotationRoom(nFrames, width, angle, noise)[1] < 0) {
      throw new DoesNotFit(pushedSideways(angle, pushX) + " " + anglesThatFit(
        (a) => rotationRoom(nFrames, width, a, noise)[1] >= 0, 90));
    }
    outWidth = width;
    if (motion === "perpendicular") {
      const [across, roundT] = windings(width, nFrames, -s, c);
      const spanX = across * width, spanT = roundT * nFrames;
      // About a pixel a frame.
      outFrames = Math.max(1, roundHalfEven(refuseNonFinite(hypot(spanX, spanT))));
      dx = spanX / outFrames;
      dt = spanT / outFrames;
    } else {
      outFrames = nFrames;
      dx = 0;
      dt = signT;
    }
  } else if (!inside) {
    const ahead = Math.abs(pushT * c - pushX * s);  // how far bumps reach off the plane
    outWidth = Math.max(1, roundHalfEven(refuseNonFinite(
      width * Math.abs(c) + nFrames * Math.abs(s))));
    outFrames = Math.max(1, roundHalfEven(refuseNonFinite(
      width * Math.abs(s) + nFrames * Math.abs(c) + 2 * ahead)));
    dx = -s;
    dt = c;
  } else {
    outWidth = width;
    // How far the frame's centre can move from the cuboid's centre, in x and in t, before any
    // point of the frame leaves the video.
    const [roomT, roomX] = rotationRoom(nFrames, width, angle, noise);
    if (roomT < 0 || roomX < 0) {
      const why = roomT < 0
        ? tooLong(angle, width, (width - 1) * Math.abs(s), pushT, nFrames)
        : pushedSideways(angle, pushX);
      throw new DoesNotFit(why + " " + anglesThatFit(
        (a) => Math.min(...rotationRoom(nFrames, width, a, noise)) >= 0, 90));
    }
    motion = motion || "longest";
    if (motion === "perpendicular") {
      dx = -s;
      dt = c;
    } else if (motion === "time") {
      dx = 0;
      dt = signT;
    } else if (motion === "longest") {
      // The diagonal of the region the centre can move in. Of the two diagonals, take the one
      // closer to perpendicular, so the frame sweeps through the video rather than sliding
      // along itself.
      dx = s >= 0 ? -roomX : roomX;
      dt = signT * roomT;
      const length = hypot(dx, dt);
      [dx, dt] = length !== 0 ? [dx / length, dt / length] : [0, signT];  // NaN is a length
    } else {
      throw new OptionError(`unknown motion ${pyRepr(motion)}`);
    }
    // As in Python, a NaN is a direction, and a NaN second loses a min to the first.
    const alongX = dx !== 0 ? roomX / Math.abs(dx) : Infinity;
    const alongT = dt !== 0 ? roomT / Math.abs(dt) : Infinity;
    const reach = alongT < alongX ? alongT : alongX;
    outFrames = Math.trunc(refuseNonFinite(2 * reach + 1e-9)) + 1;
  }

  const centreX = (width - 1) / 2, centreT = (nFrames - 1) / 2;
  const middle = (outWidth - 1) / 2;
  return {
    width: outWidth, frames: outFrames, normal, loop, sides,
    at(f) {
      const step = f - (outFrames - 1) / 2;  // distance moved from the centre
      const t0 = centreT + step * dt, x0 = centreX + step * dx;
      const t = new Float64Array(outWidth), x = new Float64Array(outWidth);
      for (let j = 0; j < outWidth; j++) {
        const across = j - middle;  // position along the frame
        t[j] = t0 + across * s;
        x[j] = x0 + across * c;
      }
      return { t, x };
    },
  };
}

/** The delay per pixel across a sheared frame, and its plane's (t, x) unit normal. */
function shear(angle: number): [number, TX] {
  // Math.tan can differ from libm's by an ulp, which the rounding keeps at steep angles: k, and
  // so each position per pixel across the frame, can be 1e-12 from Python's.
  const k = roundHalfEven(Math.tan(radians(angle)), 12);
  const h = hypot(1, k);
  return [k, [1 / h, -k / h]];
}

/**
 * How many frames of time a sheared inside frame can move through, and how far its ends can
 * move sideways. Negative if it can't fit at all.
 */
function shearRoom(nFrames: number, width: number, angle: number,
                   noise?: NoisePush | null): TX {
  const [k, normal] = shear(angle);
  const [pushT, pushX] = push(noise, normal);
  // A sheared frame always spans the video's full width, so any sideways push takes its ends
  // out of the video.
  return [nFrames - 1 - Math.abs(k) * (width - 1) - 2 * Math.abs(pushT), -Math.abs(pushX)];
}

/**
 * Plan a sweep where output column x is input column x, delayed in time by tan(angle) frames
 * per pixel across the frame.
 *
 * The frame is always as wide as the input and moves forward one frame of time per output
 * frame. Without `inside`, it runs from where the slice first touches the video to where it
 * leaves, black where it's outside. With `inside`, only positions entirely inside the video
 * are kept; throws DoesNotFit if the clip is too short for the angle. With `loop`, time runs
 * round in a ring, the clip's first frame following its last, and the frame goes once round
 * it, so the last output frame leads back into the first. With `sides` as well, x wraps round
 * too, the picture's left edge following its right.
 *
 * With `noise`, the sweep allows for the noise pushing points off the plane. Perpendicular
 * noise pushes the frame's ends sideways, so with `inside` or `loop` it only fits at 0
 * degrees, unless the sides wrap.
 */
export function shearSweep(nFrames: number, width: number, angle: number,
                           { inside = false, noise = null, loop = false,
                             sides = false }: ShearOptions = {}): Sweep {
  if (sides && !loop) throw new OptionError("the sides only wrap round on a loop");
  if (!(-90 < angle && angle < 90)) {
    throw new OptionError("shear needs an angle between -90 and 90 degrees "
                          + "(at 90 the delay would be infinite)");
  }
  const [k, normal] = shear(angle);
  const [pushT, pushX] = push(noise, normal);
  // Python lays the columns out with arange, which refuses these.
  if (Number.isNaN(width)) throw new OptionError("arange: cannot compute length");
  if (!Number.isFinite(width)) throw new OptionError("Maximum allowed size exceeded");
  const middle = (width - 1) / 2;
  const spanT = Math.abs(k) * (width - 1);  // frames of time one output frame covers
  let outFrames: number;

  if (inside || loop) {
    const [roomT, roomX] = shearRoom(nFrames, width, angle, noise);
    if (roomX < 0 && !sides) {
      throw new DoesNotFit(
        `At ${formatG(angle)} degrees perpendicular noise can push the ends of a sheared frame `
        + `${formatG(Math.abs(pushX), 3)} pixels sideways, out of the video, and a sheared `
        + "frame always spans the video's whole width. Push the noise through time instead.");
    }
    if (roomT < 0 && !loop) {
      throw new DoesNotFit(
        tooLong(angle, width, spanT, pushT, nFrames, "sheared ") + " " + anglesThatFit(
          (a) => Math.min(...shearRoom(nFrames, width, a, noise)) >= 0, 89.9));
    }
    outFrames = loop ? nFrames : Math.trunc(refuseNonFinite(roomT + 1e-9)) + 1;
  } else {
    outFrames = Math.trunc(refuseNonFinite(
      nFrames - 1 + spanT + 2 * Math.abs(pushT) + 1e-9)) + 1;
  }

  const centreT = (nFrames - 1) / 2;
  return {
    width, frames: outFrames, normal, loop, sides,
    at(f) {
      const t0 = centreT + (f - (outFrames - 1) / 2);
      const t = new Float64Array(width), x = new Float64Array(width);
      for (let j = 0; j < width; j++) {
        t[j] = t0 + (j - middle) * k;  // delayed k frames per pixel from the middle
        x[j] = j;
      }
      return { t, x };
    },
  };
}

/** Plan a sweep of either kind. motion only applies to rotate, with inside or loop. */
export function planSweep(nFrames: number, width: number,
                          { slice = "rotate", angle = 45.0, inside = false, motion = null,
                            noise = null, loop = false, sides = false }: PlanOptions = {}): Sweep {
  if (slice === "shear") return shearSweep(nFrames, width, angle, { inside, noise, loop, sides });
  if (slice === "rotate") {
    return rotationSweep(nFrames, width, angle, { inside, motion, noise, loop, sides });
  }
  throw new OptionError(`unknown slice ${pyRepr(slice)}`);
}
