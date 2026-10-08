import { describe, expect, test } from "vitest";
import raw from "../test/fixtures/planner.json?raw";
import {
  DoesNotFit, OptionError, planSweep, rotation, rotationSweep, shearSweep, windings,
  type Motion, type PlanOptions, type Sweep,
} from "./planner";
import { Noise, type NoiseDirection } from "./noise";
import { floorMod } from "./pymath";

interface Case {
  nFrames: number;
  width: number;
  options: Omit<PlanOptions, "noise">
    & { noise?: { amplitude: number; direction?: NoiseDirection } };
  error?: { type: string; message: string };
  sweep?: {
    width: number; frames: number; normal: [number, number]; loop: boolean; sides: boolean;
    cols: number[]; at: [number, number[], number[]][];
  };
}

const py = JSON.parse(raw) as { sweeps: Case[] };

function close(got: number, expected: number): boolean {
  return Math.abs(got - expected) <= 1e-9 * Math.max(1, Math.abs(expected));
}

/** How a case differs from what Python gave, or null if it doesn't. */
function mismatch({ nFrames, width, options, error, sweep }: Case): string | null {
  const noise = options.noise && new Noise(options.noise.amplitude, options.noise);
  let got: Sweep;
  try {
    got = planSweep(nFrames, width, { ...options, noise });
  } catch (err) {
    if (!(err instanceof OptionError)) throw err;
    if (!error) return `raised ${err.name}: ${err.message}`;
    const kind = error.type === "DoesNotFit" ? DoesNotFit : OptionError;
    if (!(err instanceof kind) || (kind === OptionError && err instanceof DoesNotFit)) {
      return `raised ${err.name}, not ${error.type}`;
    }
    return err.message === error.message ? null : `said ${JSON.stringify(err.message)}`;
  }
  if (!sweep) return `planned a sweep, but Python raised ${error!.type}: ${error!.message}`;
  for (const key of ["width", "frames", "loop", "sides"] as const) {
    if (got[key] !== sweep[key]) return `${key} ${got[key]}, not ${sweep[key]}`;
  }
  if (!got.normal.every((v, i) => close(v, sweep.normal[i]))) return `normal ${got.normal}`;
  for (const [f, t, x] of sweep.at) {
    const at = got.at(f);
    for (const [i, j] of sweep.cols.entries()) {
      if (!close(at.t[j], t[i]) || !close(at.x[j], x[i])) {
        return `at(${f}) column ${j} is (${at.t[j]}, ${at.x[j]}), not (${t[i]}, ${x[i]})`;
      }
    }
  }
  return null;
}

test("plans every sweep in the fixture as Python does", () => {
  const failures = py.sweeps.flatMap((c) => {
    const why = mismatch(c);
    return why ? [`${c.nFrames} frames × ${c.width} ${JSON.stringify(c.options)}: ${why}`] : [];
  });
  expect(failures).toEqual([]);
  expect(py.sweeps.filter((c) => c.sweep).length).toBeGreaterThan(400);
  expect(py.sweeps.filter((c) => c.error?.type === "DoesNotFit").length).toBeGreaterThan(100);
  expect(py.sweeps.filter((c) => c.sweep && c.options.noise).length).toBeGreaterThan(200);
});

// Geometry, after test_timeslice.py. The cuboid is T frames of W columns.
const T = 7, W = 11;
const MOTIONS: Motion[] = ["perpendicular", "time", "longest"];

/** Every column of every output frame, as [t, x] lists. */
function positions(sweep: Sweep): [number[], number[]][] {
  return Array.from({ length: sweep.frames }, (_, f) => {
    const { t, x } = sweep.at(f);
    return [Array.from(t), Array.from(x)];
  });
}

function expectPositions(sweep: Sweep, frames: number, width: number,
                         where: (f: number, j: number) => [number, number]) {
  expect([sweep.frames, sweep.width]).toEqual([frames, width]);
  for (const [f, [t, x]] of positions(sweep).entries()) {
    for (let j = 0; j < width; j++) {
      const [tt, xx] = where(f, j);
      expect(t[j]).toBeCloseTo(tt, 9);
      expect(x[j]).toBeCloseTo(xx, 9);
    }
  }
}

describe("geometry", () => {
  test("0 degrees reads every voxel in place, in every mode", () => {
    const identity = (f: number, j: number): [number, number] => [f, j];
    expectPositions(rotationSweep(T, W, 0), T, W, identity);
    for (const motion of MOTIONS) {
      expectPositions(rotationSweep(T, W, 0, { inside: true, motion }), T, W, identity);
    }
    expectPositions(shearSweep(T, W, 0), T, W, identity);
    expectPositions(shearSweep(T, W, 0, { inside: true }), T, W, identity);
  });

  test("90 degrees gives y-t slices swept right to left", () => {
    expectPositions(rotationSweep(T, W, 90), W, T, (f, j) => [j, W - 1 - f]);
  });

  test("-90 degrees sweeps left to right, with time reversed", () => {
    expectPositions(rotationSweep(T, W, -90), W, T, (f, j) => [T - 1 - j, f]);
  });

  test("180 degrees reverses time and mirrors", () => {
    const mirrored = (f: number, j: number): [number, number] => [T - 1 - f, W - 1 - j];
    expectPositions(rotationSweep(T, W, 180), T, W, mirrored);
    expectPositions(rotationSweep(T, W, 180, { inside: true }), T, W, mirrored);
  });

  test("an upright inside frame moves across the columns", () => {
    // 5 wide, standing in time centred on frame 6 of 13, moving right to left.
    expectPositions(rotationSweep(13, 5, 90, { inside: true, motion: "perpendicular" }), 5, 5,
                    (f, j) => [4 + j, 4 - f]);
  });

  test("inside frames never leave the video", () => {
    for (const angle of [10, 30, 60, 90, 135, -45]) {
      for (const motion of MOTIONS) {
        const sweep = rotationSweep(13, 5, angle, { inside: true, motion });
        expect(sweep.width).toBe(5);
        for (const [t, x] of positions(sweep)) {
          expect(Math.min(...t)).toBeGreaterThan(-1e-9);
          expect(Math.max(...t)).toBeLessThan(12 + 1e-9);
          expect(Math.min(...x)).toBeGreaterThan(-1e-9);
          expect(Math.max(...x)).toBeLessThan(4 + 1e-9);
        }
      }
    }
  });

  test("the longest motion is at least as long as the others", () => {
    for (const angle of [5, 30, 60, 90]) {
      const frames = MOTIONS.map((motion) =>
        rotationSweep(13, 5, angle, { inside: true, motion }).frames);
      expect(frames[2]).toBe(Math.max(...frames));
    }
  });

  test("an inside frame that doesn't fit says which angles do", () => {
    // sin(45 degrees) × 10 pixels is about 7 frames of time; the clip spans 6.
    expect(() => rotationSweep(T, W, 45, { inside: true }))
      .toThrow(/Angles up to 36\.8 degrees fit this clip\.$/);
    expect(() => rotationSweep(T, W, 45, { inside: true })).toThrow(DoesNotFit);
    expect(() => shearSweep(T, W, 45, { inside: true })).toThrow(DoesNotFit);
  });

  test("shear delays each column", () => {
    // At 45 degrees column x is one frame later than column x - 1.
    expectPositions(shearSweep(13, 5, 45, { inside: true }), 9, 5, (f, j) => [f + j, j]);
  });

  test("a whole-plane shear starts where only its last column is in the video", () => {
    const sweep = shearSweep(13, 5, 45);
    expect(sweep.frames).toBe(13 + 4);
    const { t } = sweep.at(0);
    expect(t[4]).toBeCloseTo(0, 9);
    expect(Math.max(...t.subarray(0, 4))).toBeLessThan(-0.5);
  });

  test("shear refuses 90 degrees as a bad option, not as a misfit", () => {
    expect(() => shearSweep(T, W, 90)).toThrow(OptionError);
    expect(() => shearSweep(T, W, 90)).not.toThrow(DoesNotFit);
    expect(() => shearSweep(T, W, NaN)).toThrow("shear needs an angle between -90 and 90");
  });

  test("a NaN or infinite angle or size fails as it does in Python", () => {
    const nan = "cannot convert float NaN to integer";
    const inf = "cannot convert float infinity to integer";
    const domain = "math domain error";
    const shearAngle = "shear needs an angle between -90 and 90 degrees "
      + "(at 90 the delay would be infinite)";
    const cases: [() => unknown, string][] = [
      [() => rotationSweep(T, W, NaN), nan],
      [() => rotationSweep(T, W, NaN, { inside: true }), nan],
      [() => rotationSweep(T, W, NaN, { inside: true, motion: "perpendicular" }), nan],
      // Python's min drops the NaN, leaving a reach of infinity.
      [() => rotationSweep(T, W, NaN, { inside: true, motion: "time" }), inf],
      [() => rotationSweep(T, W, Infinity), domain],
      [() => rotationSweep(T, W, -Infinity, { inside: true }), domain],
      [() => rotationSweep(T, W, Infinity, { loop: true }), domain],
      [() => rotation(-Infinity), domain],
      [() => shearSweep(T, W, NaN), shearAngle],
      [() => shearSweep(T, W, Infinity, { inside: true }), shearAngle],
      [() => shearSweep(T, W, -Infinity, { loop: true }), shearAngle],
      [() => shearSweep(NaN, W, 30), nan],
      [() => shearSweep(NaN, W, 30, { inside: true }), nan],
      [() => shearSweep(Infinity, W, 30), inf],
      [() => shearSweep(T, NaN, 30), "arange: cannot compute length"],
      [() => shearSweep(T, Infinity, 30, { loop: true }), "Maximum allowed size exceeded"],
    ];
    for (const [plan, message] of cases) {
      expect(plan).toThrow(OptionError);
      expect(plan).toThrow(new OptionError(message));
    }
  });

  test("planSweep picks the kind of slice", () => {
    expect(positions(planSweep(T, W, { slice: "shear", angle: 20 })))
      .toEqual(positions(shearSweep(T, W, 20)));
    expect(positions(planSweep(13, 5, { angle: 30, inside: true, motion: "time" })))
      .toEqual(positions(rotationSweep(13, 5, 30, { inside: true, motion: "time" })));
    expect(() => planSweep(T, W, { slice: "twist" as never })).toThrow("unknown slice 'twist'");
  });

  test("at(f) gives fresh arrays each time", () => {
    const sweep = shearSweep(T, W, 20);
    expect(sweep.at(1).x).not.toBe(sweep.at(1).x);
  });
});

describe("loops", () => {
  test("a loop at 0 degrees is the clip", () => {
    for (const slice of ["rotate", "shear"] as const) {
      const sweep = planSweep(T, W, { slice, angle: 0, loop: true });
      expect(sweep.loop).toBe(true);
      expectPositions(sweep, T, W, (f, j) => [f, j]);
    }
  });

  test("a loop comes back round to its first frame", () => {
    for (const [slice, angle] of [["rotate", 30], ["rotate", 90], ["rotate", 135], ["rotate", -60],
                                  ["shear", 45], ["shear", -70]] as const) {
      const sweep = planSweep(T, W, { slice, angle, loop: true });
      expect([sweep.frames, sweep.width]).toEqual([T, W]);
      const first = sweep.at(0), after = sweep.at(sweep.frames);
      for (let j = 0; j < W; j++) {
        expect(Math.abs(after.t[j] - first.t[j])).toBeCloseTo(T, 9);
        expect(after.x[j]).toBeCloseTo(first.x[j], 9);
      }
    }
  });

  test("a loop goes straight through time, however short the clip", () => {
    expect(planSweep(2, 5, { angle: 80, loop: true }).frames).toBe(2);
    expect(planSweep(T, W, { angle: 30, loop: true, motion: "time" }).frames).toBe(T);
    expect(() => rotationSweep(T, W, 30, { motion: "longest", loop: true })).toThrow(OptionError);
    expect(() => rotationSweep(T, W, 30, { motion: "perpendicular", loop: true }))
      .toThrow("a loop moves straight through time");
  });

  test("a loop round the sides at 90 degrees crosses every column", () => {
    // Column j of frame f is column 10 - f at time j - 2, both wrapped.
    const sweep = rotationSweep(T, W, 90, { loop: true, sides: true, motion: "perpendicular" });
    expect([sweep.frames, sweep.width, sweep.sides]).toEqual([W, W, true]);
    for (const [f, [t, x]] of positions(sweep).entries()) {
      for (let j = 0; j < W; j++) {
        expect(floorMod(t[j], T)).toBeCloseTo(floorMod(j - 2, T), 9);
        expect(floorMod(x[j], W)).toBeCloseTo(W - 1 - f, 9);
      }
    }
  });

  test("a loop round the sides takes the closest path when none is close enough", () => {
    // A wide, short clip at 3 degrees would need over a thousand turns round time per turn
    // across, so it goes straight through time instead, 3 degrees off.
    const [c, s] = rotation(3);
    expect(windings(1920, 30, -s, c)).toEqual([0, 1]);
  });

  test("sides only wrap on a loop, and never take the longest line", () => {
    expect(() => rotationSweep(T, W, 30, { sides: true }))
      .toThrow("the sides only wrap round on a loop");
    expect(() => shearSweep(T, W, 30, { inside: true, sides: true })).toThrow(OptionError);
    expect(() => rotationSweep(T, W, 30, { motion: "longest", loop: true, sides: true }))
      .toThrow(OptionError);
  });
});

describe("noise", () => {
  test("makes room inside", () => {
    const noise = new Noise(2);
    expect(rotationSweep(13, 5, 30, { inside: true, motion: "time", noise }).frames)
      .toBe(rotationSweep(13, 5, 30, { inside: true, motion: "time" }).frames - 4);
    expect(shearSweep(13, 5, 20, { inside: true, noise }).frames)
      .toBe(shearSweep(13, 5, 20, { inside: true }).frames - 4);
    expect(rotationSweep(13, 5, 0, { inside: true, noise }).frames).toBe(13 - 4);
  });

  test("starts a whole-plane sweep early enough for the bumps", () => {
    expect(rotationSweep(T, W, 30, { noise: new Noise(2, { direction: "perpendicular" }) }).frames)
      .toBe(rotationSweep(T, W, 30).frames + 4);
    expect(shearSweep(T, W, 30, { noise: new Noise(2) }).frames)
      .toBe(shearSweep(T, W, 30).frames + 4);
  });

  test("perpendicular noise pushes straight off the plane", () => {
    const noise = new Noise(1.5, { direction: "perpendicular" });
    const sweeps = [0, 30, 90, 135].map((angle) => rotationSweep(T, W, angle, { noise }));
    sweeps.push(shearSweep(T, W, 30, { noise }));
    for (const sweep of sweeps) {
      const { t, x } = sweep.at(2);
      const [pushT, pushX] = noise.push(sweep.normal);
      expect(Math.abs(pushT * (t[1] - t[0]) + pushX * (x[1] - x[0]))).toBeLessThan(1e-9);
      expect(Math.hypot(pushT, pushX)).toBeCloseTo(1.5, 9);
    }
  });

  test("an inside frame pushed out of the video says which angles fit", () => {
    // Perpendicular noise pushes the ends of a nearly flat frame sideways, out of the video,
    // until the frame turns far enough to leave room.
    const sideways = new Noise(1, { direction: "perpendicular" });
    expect(() => rotationSweep(13, 5, 20, { inside: true, noise: sideways }))
      .toThrow(/sideways.*Angles of 0 and 53\.2 to 90 degrees fit this clip\.$/);
    expect(() => rotationSweep(13, 5, 0, { inside: true, noise: new Noise(7) }))
      .toThrow(/No angle fits this clip\.$/);
  });

  test("a sheared inside frame only takes perpendicular noise at 0 degrees", () => {
    const sideways = new Noise(1, { direction: "perpendicular" });
    expect(() => shearSweep(13, 5, 20, { inside: true, noise: sideways }))
      .toThrow("Push the noise through time");
    expect(shearSweep(13, 5, 0, { inside: true, noise: sideways }).frames).toBe(11);
  });

  test("a loop only takes perpendicular noise that keeps inside, unless the sides wrap", () => {
    const sideways = new Noise(1, { direction: "perpendicular" });
    expect(() => rotationSweep(13, 5, 20, { noise: sideways, loop: true }))
      .toThrow(/Angles of 0 and 53\.2 to 90 degrees fit this clip\.$/);
    expect(rotationSweep(13, 5, 60, { noise: sideways, loop: true }).frames).toBe(13);
    expect(() => shearSweep(13, 5, 20, { noise: sideways, loop: true })).toThrow(DoesNotFit);
    expect(rotationSweep(13, 5, 20, { noise: sideways, loop: true, sides: true }).frames).toBe(13);
    expect(shearSweep(13, 5, 20, { noise: sideways, loop: true, sides: true }).frames).toBe(13);
    // Noise through time never leaves a ring.
    expect(shearSweep(3, 5, 80, { noise: new Noise(9), loop: true }).frames).toBe(3);
  });
});
