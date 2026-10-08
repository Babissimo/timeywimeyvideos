"""What timeslice.py renders, for web/src/render.browser.test.ts and render.test.ts.

Renders clips from decode/ with the real command line, at full quality and with
--preview, for a few sets of the page's options, and records what ffprobe and
ffmpeg find in each video: its frame count, coded size (after padding to even
sizes), frame rate, and thumbnails of a few frames, each shrunk to COLS x ROWS
blocks of mean red, green and blue (block i covers columns floor(i W / COLS) to
floor((i + 1) W / COLS) of the decoded frame, and the same down).

It also records, for a range of options, the file name a render is given and
the timeslice.py command that makes it, as output_name and command below make
them: the reference for web/src/render.ts.

Run from the repository root:

    uv run python web/test/fixtures/render.py

It writes render.json beside this script.
"""

import json
import shlex
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import timeslice  # noqa: E402
import webapp  # noqa: E402

COLS, ROWS = 8, 6


def output_name(name, opts, preview):
    """The file name of a render of the file `name`: its stem, the slice and angle,
    then each option that differs from timeslice.py's defaults."""
    parts = [Path(name).stem, opts["slice"], f"{opts['angle']:g}deg"]
    if opts["inside"]:
        parts.append("inside")
    if opts["motion"]:
        parts.append(opts["motion"])
    if opts["loop"]:
        parts.append("loop")
        if opts["loop_fade"]:
            parts.append(f"loopfade{opts['loop_fade']:g}s")
    if opts["sides"]:
        parts.append("sides")
        if opts["side_fade"]:
            parts.append(f"sidefade{opts['side_fade']:g}px")
    noise, plain = opts["noise"], timeslice.Noise(0)
    if noise:
        parts.append(f"noise{noise.amplitude:g}")
        if noise.size != plain.size:
            parts.append(f"noisesize{noise.size:g}")
        if noise.speed != plain.speed:
            parts.append(f"noisespeed{noise.speed:g}")
        if noise.direction != plain.direction:
            parts.append(f"noise{noise.direction}")
        if noise.seed != plain.seed:
            parts.append(f"noiseseed{noise.seed}")
    if opts["start"] is not None:
        parts.append(f"from{opts['start']:g}s")
    if opts["duration"] is not None:
        parts.append(f"for{opts['duration']:g}s")
    if opts["scale"] != 1:
        parts.append(f"scale{opts['scale']:g}")
    if opts["fps"]:
        parts.append(f"{opts['fps'].replace('/', 'over')}fps")
    if preview:
        parts.append("preview")
    return "_".join(parts) + ".mp4"


def command(name, output, opts, preview):
    """The timeslice.py command that renders the file `name` to `output`, run with
    uv from beside timeslice.py."""
    cmd = ["uv", "run", "timeslice.py", name, output, f"--angle={opts['angle']:g}",
           f"--slice={opts['slice']}"]
    if opts["inside"]:
        cmd.append("--inside")
    if opts["motion"]:
        cmd.append(f"--motion={opts['motion']}")
    if opts["loop"]:
        cmd += ["--loop", f"--loop-fade={opts['loop_fade']:g}"]
    if opts["sides"]:
        cmd += ["--loop-sides", f"--loop-side-fade={opts['side_fade']:g}"]
    if opts["noise"]:
        noise = opts["noise"]
        cmd += [f"--noise={noise.amplitude:g}", f"--noise-size={noise.size:g}",
                f"--noise-speed={noise.speed:g}",
                f"--noise-direction={noise.direction}", f"--noise-seed={noise.seed}"]
    if opts["scale"] != 1:
        cmd.append(f"--scale={opts['scale']:g}")
    if opts["start"] is not None:
        cmd.append(f"--start={opts['start']:g}")
    if opts["duration"] is not None:
        cmd.append(f"--duration={opts['duration']:g}")
    if opts["fps"]:
        cmd.append(f"--fps={opts['fps']}")
    if preview:
        cmd.append("--preview")
    return cmd


NAMES = [
    ("clip.mp4", {}),
    ("clip.mp4", {"angle": "-30.5", "slice": "shear", "inside": "1"}),
    ("my clip's.mov", {"angle": "80", "inside": "1", "motion": "perpendicular"}),
    ("a.b.mp4", {"angle": "1e-05", "loop": "1", "loop_fade": "0.25"}),
    ("clip", {"angle": "60", "loop": "1", "loop_sides": "1", "motion": "time",
              "loop_side_fade": "12"}),
    (".hidden.mp4", {"loop": "1", "loop_sides": "1", "loop_fade": "0", "loop_side_fade": "0"}),
    ("clip.mp4", {"noise": "4"}),
    ("clip.mp4", {"noise": "2.5", "noise_size": "32", "noise_speed": "0.5",
                  "noise_direction": "perpendicular", "noise_seed": "7"}),
    ("clip.mp4", {"start": "1.5", "duration": "2", "scale": "0.5", "fps": "30000/1001"}),
    ("clip.mp4", {"start": "0", "scale": "1234567", "fps": "24"}),
]

RENDERS = [
    ("h264-30.mp4", {"angle": "30"}),
    ("h264-30.mp4", {"slice": "shear", "angle": "20", "scale": "0.9"}),
    ("h264-ntsc.mp4", {"angle": "60", "loop": "1", "loop_fade": "0.2", "loop_sides": "1",
                       "motion": "perpendicular", "loop_side_fade": "6", "noise": "3",
                       "noise_seed": "2"}),
    ("h264-30-aac.mp4", {"angle": "-20", "inside": "1", "fps": "24", "start": "0.2",
                         "duration": "1"}),
]


def names():
    cases = []
    for name, values in NAMES:
        opts = webapp.read_options(values)
        for preview in (False, True):
            output = output_name(name, opts, preview)
            cmd = command(name, output, opts, preview)
            cases.append(dict(name=name, values=values, preview=preview, output=output,
                              command=cmd, line=shlex.join(cmd)))
    return cases


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True)
    return json.loads(out.stdout)["streams"][0]


def thumbnails(path, width, height, frames):
    out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    video = np.frombuffer(out, np.uint8).reshape(-1, height, width, 3)
    shrunk = []
    for f in frames:
        blocks = [video[f, j * height // ROWS:(j + 1) * height // ROWS,
                        i * width // COLS:(i + 1) * width // COLS].reshape(-1, 3).mean(axis=0)
                  for j in range(ROWS) for i in range(COLS)]
        shrunk.append(np.round(blocks).astype(int).ravel().tolist())
    return shrunk


def render(clip, values, preview, folder):
    opts = webapp.read_options(values)
    output = output_name(clip, opts, preview)
    flags = command(clip, output, opts, preview)[5:]
    path = folder / output
    subprocess.run(["uv", "run", "timeslice.py", str(HERE / "decode" / clip), str(path), *flags],
                   cwd=ROOT, check=True, capture_output=True)
    stream = probe(path)
    width, height, frames = stream["width"], stream["height"], int(stream["nb_read_frames"])
    rate = Fraction(stream["r_frame_rate"])
    chosen = sorted({0, frames // 2, frames - 1})
    return dict(clip=clip, values=values, preview=preview, output=output,
                codec=stream["codec_name"], width=width, height=height, frames=frames,
                fps=[rate.numerator, rate.denominator], cols=COLS, rows=ROWS, chosen=chosen,
                thumbnails=thumbnails(path, width, height, chosen))


def main():
    with tempfile.TemporaryDirectory() as folder:
        renders = [render(clip, values, preview, Path(folder))
                   for clip, values in RENDERS for preview in (False, True)]
    text = json.dumps(dict(names=names(), renders=renders), separators=(",", ":"))
    (HERE / "render.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
