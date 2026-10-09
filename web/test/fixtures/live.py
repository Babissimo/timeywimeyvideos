"""What the live view shows of a clip, for web/src/live.browser.test.ts.

The live view slices a copy of the clip at half size in x, y and time, nearest
voxel, as `timeslice.py --preview` renders it. For each set of the page's
options and position this records the frame timeslice makes, and the numbers
that describe it.

Run from the repository root:

    uv run python web/test/fixtures/live.py

It writes live.json beside this script. The clip is decode/h264-30.mp4.
"""

import base64
import json
import sys
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2]))

import timeslice  # noqa: E402
import page_options  # noqa: E402

CLIP = HERE / "decode" / "h264-30.mp4"
SHRINK = timeslice.PREVIEW_SCALE

CASES = [
    ({"angle": "0"}, 0),
    ({"angle": "30"}, 0.5),
    ({"slice": "shear", "angle": "20"}, 0.3),
    ({"angle": "30", "noise": "3", "noise_size": "10", "noise_speed": "0.5",
      "noise_direction": "perpendicular", "noise_seed": "4"}, 0.25),
    ({"angle": "30", "loop": "1", "loop_fade": "0.4"}, 0.25),
    ({"angle": "80", "loop": "1", "loop_sides": "1", "motion": "perpendicular",
      "loop_side_fade": "8", "inside": "1"}, 0.36),
    ({"angle": "45", "loop": "1", "noise": "4", "noise_seed": "1"}, 0.6),
    ({"angle": "20", "inside": "1", "scale": "0.5", "start": "0.2", "duration": "0.8"}, 0.7),
    ({"angle": "-60", "fps": "24"}, 1),
]


def endpoints(sweep, f):
    t, x = sweep.at(f)
    return [float(t[0]), float(x[0]), float(t[-1]), float(x[-1])]


def frame(values, pos):
    opts = page_options.read_options(values)
    volume, fps = timeslice.load_video(str(CLIP), opts["scale"] * SHRINK, SHRINK,
                                       opts["start"], opts["duration"], fast=True)
    frames = round(opts["loop_fade"] * float(fps) * SHRINK)
    columns = round(opts["side_fade"] * SHRINK)
    volume = timeslice.crossfade(timeslice.crossfade(volume, frames), columns, axis=2)
    n_frames, height, width, _ = volume.shape
    noise = opts["noise"] and opts["noise"].scaled(SHRINK)
    sweep = timeslice.plan_sweep(n_frames, width, opts["slice"], opts["angle"], opts["inside"],
                                 opts["motion"], noise, opts["loop"], opts["sides"])
    f = round(pos * (sweep.frames - 1))
    image = timeslice.slice_frame(volume, sweep, f, noise, nearest=True)
    out_fps = (Fraction(opts["fps"]) if opts["fps"] else fps) * Fraction(SHRINK)
    return dict(
        values=values, pos=pos,
        width=sweep.width, height=height, frames=sweep.frames, frame=f,
        fps=float(out_fps), seconds=float(sweep.frames / out_fps),
        volume=[n_frames, height, width], loop=sweep.loop, sides=sweep.sides,
        line=endpoints(sweep, f), first=endpoints(sweep, 0),
        last=endpoints(sweep, sweep.frames - 1),
        rgb=base64.b64encode(image.tobytes()).decode())


def main():
    cases = [frame(values, pos) for values, pos in CASES]
    text = json.dumps(cases, allow_nan=False, separators=(",", ":"))
    (HERE / "live.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
