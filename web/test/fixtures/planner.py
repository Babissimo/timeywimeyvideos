"""Reference values for web/src/pymath.ts, from Python.

Run from the repository root:

    uv run python web/test/fixtures/planner.py

It writes pymath.json beside this script.
"""

import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


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


if __name__ == "__main__":
    pymath()
