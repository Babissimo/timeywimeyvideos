# timeywimeyvideos

Treat a video as a cuboid of voxels with axes x, y and time, then play it back
by sweeping a slicing plane through it at an angle, instead of the usual
"one plane per moment" playback.

For now the plane can be rotated about the **y axis** (the vertical axis of
the picture).

## Setup

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (which
includes `ffprobe`) on your `PATH`.

```sh
pip install -r requirements.txt
```

The sampling loops are compiled by [numba](https://numba.pydata.org/) the first
time they run, which takes several seconds. The compiled code is cached in
`__pycache__/`, so later runs start straight away.

## Usage

```sh
python timeslice.py input.mp4 output.mp4 --angle 30
```

| option | meaning |
| --- | --- |
| `--angle DEG` | rotation of the slicing plane about the y axis (default 45) |
| `--scale F` | shrink the input first, e.g. `0.5`, which is much faster and uses less memory |
| `--start S`, `--duration S` | use only part of the input (in seconds) |
| `--fps R` | output frame rate (defaults to the input's) |

The whole clip is held in memory at once (width × height × frames × 3 bytes),
so start with short clips or `--scale 0.5`. The script prints how much memory
it's using.

## How the angle works

Picture the cuboid from above: x runs left to right and time runs up the
page, like a space-time diagram. Ordinary playback slides a horizontal line
(one moment) up the page. `--angle` rotates that line anticlockwise:

```
 t ↑     0°              30°              90°
   │  ─────────         ╱ (tilted)        │ (vertical)
   │  sweeps up ↑       sweeps up-left    sweeps right → left
   └──────────→ x
```

- **0°** reproduces the original video.
- **Positive angles**: the right edge of each output frame is later in time
  than the left edge.
- **90°**: each output frame is a y-time slice through one column of the
  original (time runs left to right), and the sweep moves from right to left.
  Use **-90°** to sweep left to right instead (time then runs right to left
  in each frame).
- **180°**: the video plays backwards and mirrored.

One frame of time is treated as one pixel of distance. Parts of the tilted
plane that fall outside the cuboid come out black, so tilted outputs are a bit
wider than the input and start and end with a mostly black frame as the plane
enters and leaves a corner.

## Code layout

`timeslice.py` is split so that new kinds of slices only need new coordinates:

- `sample(volume, t, y, x)` reads the video at any (t, y, x) positions, so it
  works for any surface. Each point blends the 8 voxels around it, weighted
  by closeness (linear interpolation); points outside the cuboid are black.
- `sample_columns(volume, t, x)` is a faster version for slices made of whole
  source columns, which is what a rotation about the y axis gives. It only
  blends across x and t (4 voxels).
- `y_rotation_slices(...)` generates the (t, x) position of every output
  column, one output frame at a time, for the rotated plane.
- `main()` handles decoding and encoding (via ffmpeg).

Run the geometry checks with `python test_timeslice.py` (or `pytest`).
