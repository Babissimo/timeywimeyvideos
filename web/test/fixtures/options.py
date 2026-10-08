"""What webapp.read_options makes of the page's options, for web/src/options.test.ts.

Run from the repository root:

    uv run python web/test/fixtures/options.py

It writes options.json beside this script: for each set of values, the options
read from them, or the message and kind of the problem they raise.
"""

import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import webapp  # noqa: E402

CASES = [
    {},
    # Numbers, as Python's float() reads them.
    {"angle": "30"}, {"angle": "-45"}, {"angle": " 12.5 "}, {"angle": "1_000"},
    {"angle": "1e2"}, {"angle": ".5"}, {"angle": "5."}, {"angle": "+7"},
    {"angle": "steep"}, {"angle": "inf"}, {"angle": "-Infinity"}, {"angle": "nan"},
    {"angle": "1__0"}, {"angle": "_1"}, {"angle": "1_"}, {"angle": "0x10"}, {"angle": "."},
    {"angle": "e5"}, {"angle": " "},
    {"slice": "shear", "angle": "20"}, {"slice": "twist"}, {"slice": ""},
    {"scale": "0"}, {"scale": "-1"}, {"scale": "0.5"}, {"scale": ""},
    {"start": "-1"}, {"start": "0"}, {"start": "2.5"},
    {"duration": "0"}, {"duration": "1.5"}, {"duration": "-2"},
    # Frame rates, as Fraction() reads them.
    {"fps": "24"}, {"fps": "30000/1001"}, {"fps": "29.97"}, {"fps": " 30000 / 1001 "},
    {"fps": "2.5e1"}, {"fps": "1_000/1_001"}, {"fps": "1E-1"}, {"fps": "+12"}, {"fps": "6/4"},
    {"fps": "-3"}, {"fps": "0"}, {"fps": "0/5"}, {"fps": "1/0"}, {"fps": "abc"},
    {"fps": "inf"}, {"fps": "1.5/2"}, {"fps": "1/-2"}, {"fps": ""}, {"fps": "1e300"},
    {"fps": "1e-300"},
    # Flags, and the edges.
    {"inside": "1"}, {"inside": "true"}, {"inside": "TRUE"}, {"inside": "yes"},
    {"inside": "0"}, {"inside": "1", "loop": "1"},
    {"inside": "1", "motion": "perpendicular"}, {"inside": "1", "motion": "time"},
    {"inside": "1", "motion": "longest"}, {"inside": "1", "motion": "sideways"},
    {"inside": "1", "motion": ""}, {"motion": "time"}, {"motion": "sideways"},
    {"slice": "shear", "inside": "1", "motion": "time"},
    {"loop": "1", "motion": "time"},
    {"loop": "1", "loop_sides": "1"},
    {"loop": "1", "loop_sides": "1", "motion": "perpendicular"},
    {"loop": "1", "loop_sides": "1", "motion": "longest"},
    {"loop_sides": "1"}, {"loop_sides": "1", "motion": "longest"},
    {"loop": "1", "loop_fade": "0.5"}, {"loop": "1", "loop_fade": "-1"},
    {"loop": "1", "loop_fade": ""}, {"loop_fade": "0.5"}, {"loop_fade": "-1"},
    {"loop": "1", "loop_sides": "1", "loop_side_fade": "8"},
    {"loop": "1", "loop_sides": "1", "loop_side_fade": "x"},
    {"loop": "1", "loop_side_fade": "8"},
    # Noise.
    {"noise": "0"}, {"noise": "-1"}, {"noise": ""}, {"noise": "2"},
    {"noise": "3", "noise_size": "10", "noise_speed": "0.5",
     "noise_direction": "perpendicular", "noise_seed": "4"},
    {"noise": "2", "noise_size": "", "noise_speed": "", "noise_direction": "",
     "noise_seed": ""},
    {"noise": "2", "noise_size": "0"}, {"noise": "2", "noise_size": "-3"},
    {"noise": "2", "noise_speed": "-0.6"}, {"noise": "2", "noise_speed": "fast"},
    {"noise": "2", "noise_direction": "sideways"},
    {"noise": "2", "noise_seed": "1.5"}, {"noise": "2", "noise_seed": "-1"},
    {"noise": "2", "noise_seed": " 7 "}, {"noise": "2", "noise_seed": "1_0"},
    {"noise": "2", "noise_seed": "+3"}, {"noise": "2", "noise_seed": "-0"},
    {"noise": "2", "noise_seed": "seven"},
    {"noise": "", "noise_seed": "-1", "noise_direction": "sideways"},
    # Numbers and booleans, as a JSON body gives them.
    {"angle": 30, "inside": True, "noise": 2.5, "noise_seed": 3, "noise_size": 12},
    {"loop": 1, "loop_fade": 0.4, "scale": 0.5, "start": 0, "duration": 2},
    {"noise": 1, "noise_seed": 2.0}, {"inside": False, "loop": 0},
    # Several problems: the first is reported.
    {"slice": "twist", "angle": "steep"}, {"angle": "steep", "scale": "0"},
    {"fps": "0", "noise": "-1"}, {"motion": "up", "inside": "1", "fps": "0"},
    {"noise": "2", "noise_seed": "-1", "noise_direction": "up"},
    {"noise": "2", "noise_size": "0", "noise_speed": "x"},
    {"scale": "0", "start": "-1", "duration": "0", "loop": "1", "loop_fade": "-1"},
]


def described(opts):
    noise = opts["noise"]
    fps = opts["fps"]
    if fps is not None:
        fps = Fraction(fps)
    return dict(
        slice=opts["slice"], angle=opts["angle"], inside=opts["inside"],
        motion=opts["motion"],
        noise=noise and dict(amplitude=noise.amplitude, size=noise.size, speed=noise.speed,
                             direction=noise.direction, seed=noise.seed,
                             period=noise.period),
        scale=opts["scale"], start=opts["start"], duration=opts["duration"],
        fps=fps and [fps.numerator, fps.denominator], loop=opts["loop"],
        sides=opts["sides"], loopFade=opts["loop_fade"], sideFade=opts["side_fade"])


def main():
    cases = []
    for values in CASES:
        try:
            cases.append(dict(values=values, options=described(webapp.read_options(values))))
        except webapp.Problem as err:
            cases.append(dict(values=values, error=str(err), kind=err.kind))
    text = json.dumps(cases, allow_nan=False, separators=(",", ":"))
    (HERE / "options.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
