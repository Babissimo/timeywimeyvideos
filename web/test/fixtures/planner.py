"""Reference values for web/src/pymath.ts, planner.ts and noise.ts, from Python.

Run from the repository root:

    uv run python web/test/fixtures/planner.py

It writes pymath.json, planner.json and noise.json beside this script.
"""

import json
import math
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import timeslice  # noqa: E402


def write(name, data):
    text = json.dumps(data, allow_nan=False, separators=(",", ":"))
    (HERE / name).write_text(text + "\n")


def pymath():
    halves = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 0.49999999999999994, 4503599627370497.0,
              -0.4, 0.0, 1e300, 123.456, -7.5]
    with_digits = [(2.675, 2), (0.125, 2), (0.375, 2), (-0.125, 2), (1.005, 2), (0.5, 1),
                   (0.25, 1), (0.35, 1), (1e-13, 12), (-1e-13, 12), (5e-13, 12), (1.5e-12, 12),
                   (6.123233995736766e-17, 12), (-1.8369701987210297e-16, 12),
                   (0.8660254037844387, 12), (0.49999999999999994, 12), (1.0000000000004998, 12),
                   (572.9572133542884, 12), (1234.5, -1), (1250.0, -2), (-0.0, 3),
                   (123456.789, 400), (123456.789, -400)]
    for angle in [30, 45, 60, 89.9, 37.3, 135]:
        theta = math.radians(angle)
        with_digits += [(math.cos(theta), 12), (math.sin(theta), 12), (math.tan(theta), 12)]
    # Decimal halves, just off half way in binary, and arbitrary values.
    with_digits += [(k / 20, 1) for k in range(-39, 40, 2)]
    with_digits += [(k / 2000, 3) for k in range(1, 40, 2)]
    rng = np.random.default_rng(5)
    with_digits += [(x, n) for x, n in zip(rng.uniform(-1000, 1000, 60).tolist(),
                                           [1, 2, 3, 6, 9, 12] * 10)]
    with_digits += [(x, 12) for x in rng.uniform(-1, 1, 40).tolist()]
    with_digits += [(5729.577893130245, 12), (1e-300, 12), (4503.599627370497, 12)]
    rounds = [[x, None, round(x)] for x in halves]
    rounds += [[x, n, round(x, n)] for x, n in with_digits]

    pairs = [(7, 3), (-7, 3), (7, -3), (-7, -3), (6, 3), (-6, 3), (6, -3), (0, 5), (-0.0, 5),
             (5.5, 2), (-5.5, 2), (-1e-17, 1), (-1, 256), (-257, 256), (-3, 7), (13.25, -4)]
    mods = [[a, n, a % n] for a, n in pairs]

    remainder_pairs = [(5, 2), (7, 2), (-5, 2), (-7, 2), (1, 3), (2, 3), (10, 3), (3, -2),
                       (0.0, 1), (-0.0, 1), (1e300, 3), (1, 1e300), (-4.71238898038469, math.tau),
                       (9.42477796076938, math.tau), (math.pi, math.tau), (-math.pi, math.tau),
                       (3 * math.pi, math.tau)]
    remainders = [[x, y, math.remainder(x, y)] for x, y in remainder_pairs]

    gcds = [[a, b, math.gcd(a, b)]
            for a, b in [(0, 0), (0, 5), (5, 0), (12, 18), (-12, 18), (12, -18), (-16, -4),
                         (1, 16), (7, 13), (15, -10)]]

    limits = [[x, m, list(Fraction(x).limit_denominator(m).as_integer_ratio())]
              for x, m in [(0.5, 1000), (1 / 3, 1000), (0.1, 1000), (2 / 3, 1000), (0.25, 1000),
                           (1.0, 1000), (0.75, 1000), (1 / 7, 1000), (0.999, 1000),
                           (30000 / 1001, 1000), (30000 / 1001, 10000), (math.pi, 10),
                           (math.pi, 100), (-math.pi, 100), (-0.1, 1000), (0.0, 1000),
                           (1e-7, 1000), (123.456, 1)]]

    hypots = [[x, y, math.hypot(x, y)]
              for x, y in [(3, 4), (1, 1), (0, 0), (-5, 12), (1e-200, 1e-200), (1e300, 1e300),
                           (0.1, 0.2), (8.660254037844387, 5.000000000005), (1920, 30),
                           (11, 7), (-7, 0), (2.5, -1e-20), (16 * 1920, 15 * 30),
                           (4 * 11, 7 * 7), (8 * 11, 1 * 7)]]

    angles = [0, 1, 30, -45, 89.9, 90, 135, 180, -180, 37.3, 1e-7]
    radians = [[a, math.radians(a)] for a in angles]
    degrees = [[r, math.degrees(r)] for r in [0, 1, -0.5, math.pi, math.tau, 0.034906585,
                                              1e-12]]

    gs = [(x, p) for x in [0.0, -0.0, 1.0, 30.0, -30.0, 37.3, 89.9, -89.9, 1e-7, 0.0001,
                           0.00001234, 123456.0, 1234567.0, 999999.5, 9999995.0, 1e16,
                           2.5, 0.125, 36.8, 53.2, 0.939692620786, 1.5, 2.675, 0.30000000000000004,
                           12.345, 100.0, 0.9995, 1e-300, 5e-324, 1.7976931348623157e308]
          for p in [6, 3, 2, 1]]
    formats = [[x, p, format(x, f".{p}g")] for x, p in gs]
    formats += [[x, None, format(x, "g")] for x in [37.3, 1e-7, 1234567.0, 0.5]]

    reprs = [[s, repr(s)] for s in ["sideways", "", "it's", 'say "hi"', "both ' and \"",
                                     "back\\slash", "tab\there", "line\nbreak", "\x01",
                                     "café"]]

    copysigns = [[x, y, math.copysign(x, y)]
                 for x, y in [(1.5, -0.0), (1.5, 0.0), (-2, 3), (2, -3), (0.0, -1), (-0.0, 1)]]

    write("pymath.json", {
        "round": rounds, "mod": mods, "remainder": remainders, "gcd": gcds,
        "limitDenominator": limits, "hypot": hypots, "radians": radians,
        "degrees": degrees, "formatG": formats, "repr": reprs, "copysign": copysigns,
    })


ANGLES = [0, 30, -30, 45, 89.9, -89.9, 90, -90, 120, 135, 180, -180, 37.3, -61.7, 3]
ROTATE_MODES = [{}] + [
    {"inside": True, "motion": m} for m in [None, "perpendicular", "time", "longest", "sideways"]
] + [{"loop": True}] + [{"loop": True, "sides": True, "motion": m} for m in [None, "perpendicular"]]
SHEAR_MODES = [{}, {"inside": True}, {"loop": True}, {"loop": True, "sides": True}]
# Noise only changes how far a plan keeps from the edges, which a loop round the sides
# ignores.
NOISY_ROTATE_MODES = [{}, {"loop": True}] + [
    {"inside": True, "motion": m} for m in ["perpendicular", "time", "longest"]]
NOISY_SHEAR_MODES = [{}, {"inside": True}, {"loop": True}]
NOISES = [{"amplitude": a, "direction": d}
          for d, amplitudes in [("time", [0.5, 9.0]), ("perpendicular", [0.5, 2.0])]
          for a in amplitudes]
# (sizes as (frames, width), angles, noises, rotate modes, shear modes)
GRID = [
    ([(7, 11), (13, 5)], ANGLES, [None], ROTATE_MODES, SHEAR_MODES),
    ([(2, 5)], [0, 30, 80, 90, 180], [None], ROTATE_MODES, SHEAR_MODES),
    ([(300, 640), (151, 1920), (30, 1920)], [0, 30, 89.9, 90, 135, 180, 37.3, -61.7, 3], [None],
     ROTATE_MODES, SHEAR_MODES),
    ([(7, 11), (13, 5)], [0, 30, 89.9, 90, 135, 37.3, -61.7], NOISES, NOISY_ROTATE_MODES,
     NOISY_SHEAR_MODES),
    ([(300, 640)], [0, 30, 37.3], NOISES, NOISY_ROTATE_MODES, NOISY_SHEAR_MODES),
]
ONE_OFFS = [
    (7, 11, {"slice": "twist", "angle": 30}),
    (7, 11, {}),  # plan_sweep's defaults
    (7, 11, {"slice": "rotate", "angle": 30, "motion": "sideways"}),  # not inside: ignored
    (7, 11, {"slice": "rotate", "angle": 30, "sides": True}),
    (7, 11, {"slice": "rotate", "angle": 30, "inside": True, "sides": True}),
    (7, 11, {"slice": "rotate", "angle": 30, "loop": True, "motion": "time"}),
    (7, 11, {"slice": "rotate", "angle": 30, "loop": True, "motion": "perpendicular"}),
    (7, 11, {"slice": "rotate", "angle": 30, "loop": True, "motion": "longest"}),
    (7, 11, {"slice": "rotate", "angle": 30, "loop": True, "motion": ""}),
    (7, 11, {"slice": "rotate", "angle": 30, "loop": True, "sides": True, "motion": "time"}),
    (7, 11, {"slice": "rotate", "angle": 30, "loop": True, "sides": True, "motion": "longest"}),
    (13, 5, {"slice": "rotate", "angle": 30, "inside": True, "motion": ""}),
    (13, 5, {"slice": "shear", "angle": 20, "motion": "longest"}),  # ignored
    (13, 5, {"slice": "shear", "angle": 20, "sides": True}),
    (13, 5, {"slice": "shear", "angle": 20, "inside": True, "loop": True}),
    (13, 5, {"slice": "rotate", "angle": 20, "noise": {"amplitude": 1.0, "direction": "up"}}),
    (13, 5, {"slice": "rotate", "angle": 20, "inside": True,
             "noise": {"amplitude": 1.0, "direction": "perpendicular"}}),
    (13, 5, {"slice": "rotate", "angle": 0, "inside": True, "noise": {"amplitude": 7.0}}),
    (13, 5, {"slice": "rotate", "angle": 20, "loop": True, "sides": True,
             "noise": {"amplitude": 1.0, "direction": "perpendicular"}}),
    (13, 5, {"slice": "shear", "angle": 20, "loop": True, "sides": True,
             "noise": {"amplitude": 1.0, "direction": "perpendicular"}}),
    (3, 5, {"slice": "shear", "angle": 80, "loop": True, "noise": {"amplitude": 9.0}}),
]


def sampled(n):
    """The columns recorded of a frame n wide: the first, the last and a few between, a
    multiple of 37 apart on a wide frame."""
    step = 37 * math.ceil(n / 185) if n > 64 else math.ceil(n / 4)
    return sorted(set(range(0, n, step)) | {n - 1})


def plan(n_frames, width, options):
    """A plan_sweep call and what it gives: its error, or the sweep with at(f) at a few f."""
    case = {"nFrames": n_frames, "width": width, "options": options}
    noise = options.get("noise")
    kwargs = {**options, "noise": noise and timeslice.Noise(**noise)}
    try:
        sweep = timeslice.plan_sweep(n_frames, width, **kwargs)
    except ValueError as err:
        return {**case, "error": {"type": type(err).__name__, "message": str(err)}}
    cols = sampled(sweep.width)
    at = []
    for f in sorted({0, 1, sweep.frames // 2, sweep.frames - 1} & set(range(sweep.frames))):
        t, x = sweep.at(f)
        at.append([f, [float(t[j]) for j in cols], [float(x[j]) for j in cols]])
    return {**case, "sweep": {"width": sweep.width, "frames": sweep.frames,
                              "normal": [float(v) for v in sweep.normal], "loop": sweep.loop,
                              "sides": sweep.sides, "cols": cols, "at": at}}


def planner():
    cases = []
    for sizes, angles, noises, rotate_modes, shear_modes in GRID:
        for n_frames, width in sizes:
            for angle in angles:
                for kind, modes in [("rotate", rotate_modes), ("shear", shear_modes)]:
                    for mode in modes:
                        for noise in noises:
                            options = {"slice": kind, "angle": angle, **mode}
                            if noise:
                                options["noise"] = noise
                            cases.append(plan(n_frames, width, options))
    cases += [plan(n_frames, width, options) for n_frames, width, options in ONE_OFFS]
    write("planner.json", {"sweeps": cases})


# SeedSequence takes the seed in 32-bit words: two for 2**32 + 5, then 3, 3, 4, 5, 7 and 32.
# The test reads each as a JSON number, so every one is a whole double.
SEEDS = [0, 1, 2, 7, 42, 12345, 2**31 - 1, 2**32 + 5,
         2**64, 2**64 + 2**12, 3 * 2**96 + 5 * 2**64, 2**128, 2**200 + 2**150, int(1.7e308)]
assert all(int(float(seed)) == seed for seed in SEEDS)
# (width, height, f, Noise options): nodes 8 to the bump and nodes at every pixel (size under
# 16), negative and zero speeds, and periods, one so short for its size the noise rounds up to
# 2 cells.
FIELDS = [
    (24, 18, 0, {}),
    (24, 18, 5, {"size": 24.0, "speed": 0.6, "seed": 1}),
    (19, 11, 4, {"size": 3.0, "speed": 0.6, "seed": 1}),
    (19, 11, 7, {"size": 15.9, "speed": 0.37, "seed": 2}),
    (20, 14, 3, {"size": 8.0, "speed": -0.6, "period": 40.0}),
    (20, 14, 43, {"size": 8.0, "speed": 0.37, "period": 40.0}),
    (21, 13, 9, {"size": 64.0, "speed": 0.1, "period": 40.0, "seed": 3}),
    (17, 13, 2, {"size": 10.0, "speed": 0.0, "period": 40.0, "seed": 5}),
    (18, 12, 6, {"size": 5.0, "speed": 0.75, "period": 15.0, "seed": 2**32 + 5}),
    (16, 12, 1, {"size": 32.0, "speed": 1.5, "seed": 42}),
]


def noise():
    permutations = [[seed, timeslice._permutation(seed).tolist()] for seed in SEEDS]

    rng = np.random.default_rng(1)
    points = rng.uniform(-300, 300, (60, 3)).tolist() + rng.uniform(-2, 2, (60, 3)).tolist()
    points += [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [0.5, 0.5, 0.5], [-0.5, -1.25, -2.75],
               [255.5, 256.5, 257.5], [1e6 + 0.3, -1e6 - 0.7, 12345.678], [0.999999, 1e-9, -1e-9]]
    samples = []
    for seed in [0, 42]:
        perm = timeslice._permutation(seed)
        for i, (x, y, z) in enumerate(points):
            period = [0, 2, 3, 7, 300][i % 5]
            samples.append([seed, x, y, z, period, timeslice._noise3(perm, x, y, z, period)])

    fields = []
    for width, height, f, options in FIELDS:
        bumps = timeslice.Noise(1.0, **options)
        nodes, cols, col_w, rows, row_w = bumps._grid(width, height, f)
        fields.append({
            "width": width, "height": height, "f": f, "options": options,
            "grid": {"nodes": nodes.ravel().tolist(), "nodeRows": nodes.shape[0],
                     "nodeCols": nodes.shape[1], "cols": cols.tolist(),
                     "colW": col_w.ravel().tolist(), "rows": rows.tolist(),
                     "rowW": row_w.ravel().tolist()},
            "field": bumps.field(width, height, f).ravel().tolist(),
        })

    write("noise.json", {"permutations": permutations, "noise3": samples, "fields": fields})


if __name__ == "__main__":
    pymath()
    planner()
    noise()
