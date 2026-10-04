# timeywimeyvideos

Treat a video as a cuboid of voxels with axes x, y and time, then play it back
by sweeping a slicing plane through it at an angle, instead of the usual
"one plane per moment" playback.

For now the plane can be tilted about the **y axis** (the vertical axis of
the picture).

## Setup

You need Python 3.10+ and [ffmpeg](https://ffmpeg.org/download.html) (which
includes `ffprobe`) on your `PATH`.

```sh
pip install -r requirements.txt
```

(Flask is only needed for the web front end.)

The sampling loops are compiled by [numba](https://numba.pydata.org/) the first
time they run, which takes several seconds. The compiled code is cached in
`__pycache__/`, so later runs start straight away.

## Usage

```sh
python timeslice.py input.mp4 output.mp4 --angle 30 --preview   # quick look
python timeslice.py input.mp4 output.mp4 --angle 30             # final cut
```

| option | meaning |
| --- | --- |
| `--angle DEG` | tilt of the slicing plane about the y axis (default 45) |
| `--slice rotate\|shear` | how to read along the tilted plane (default `rotate`; see below) |
| `--inside` | keep each frame the input's width and entirely inside the video: no black edges |
| `--motion perpendicular\|time\|longest` | with `--slice rotate --inside`: which way the frame moves (default `longest`) |
| `--preview` | quick rough render (see below) |
| `--scale F` | resize each input frame first, e.g. `0.5`; keeps every frame |
| `--start S`, `--duration S` | use only part of the input (in seconds) |
| `--fps R` | output frame rate (defaults to the input's) |

The whole clip is held in memory at once (width × height × frames × 3 bytes),
so start with short clips. The script prints how much memory it's using, and
at the end how long the video it made is against how long it took.

### Preview

`--preview` halves the clip in x, y **and** time (every other pixel, every
other frame) before slicing. Because one frame counts as one pixel, shrinking
all three keeps the cuboid's shape, so the preview shows the same slices as
the full render at half size. It also copies the nearest voxel instead of
blending, and encodes with x264's `ultrafast` preset. It plays at half the
frame rate, so it runs as long as the full render would.

Note that `--scale` only resizes the frames and keeps every frame, so it
changes the cuboid's shape: the same angle gives a different slice at a
different `--scale`.

## Web front end

```sh
python webapp.py --videos ~/Movies
```

then open <http://127.0.0.1:8000>. The page offers the videos in the folder
you name (the top level only; `videos/` here by default) and any you upload,
which are saved in `uploads/`.

- **Live preview** plays the slice while you change the angle and other
  options. It shows the same frames as `--preview` would render: the server
  holds a half-size copy of the clip in memory and slices one frame per
  request. Space plays and pauses; the arrow keys step a frame.
- **Cuboid** draws the clip as a box (one unit per pixel or frame), with the
  first frame at the front and time running back, the current output frame
  drawn where it cuts through, and the region the sweep covers shaded on top.
  Drag to turn it, scroll to zoom, or pick a view; **Top** is the x-t diagram
  below.
- **Render preview** and **Render full quality** run `timeslice.py` in a
  separate process with the options on the page. Renders are saved in
  `renders/` and can be watched and downloaded from the page.

The live view refuses clips that would need more than 2 GB of memory at half
size; set a duration or scale to use part of a long video. The server only
listens on this machine unless you pass `--host`.

## How the angle works

Picture the cuboid from above: x runs left to right and time runs up the
page, like a space-time diagram. Ordinary playback slides a horizontal line
(one moment) up the page. `--angle` tilts that line anticlockwise:

```
 t ↑     0°              30°              90°
   │  ─────────         ╱ (tilted)        │ (vertical)
   │  sweeps up ↑       sweeps up-left    sweeps right → left
   └──────────→ x
```

- **0°** reproduces the original video, in every mode.
- **Positive angles**: the right edge of each output frame is later in time
  than the left edge.
- **90°** (rotate only): each output frame is a y-time slice through one
  column of the original (time runs left to right), and the sweep moves from
  right to left. Use **-90°** to sweep left to right instead (time then runs
  right to left in each frame).
- **180°** (rotate only): the video plays backwards and mirrored.

One frame of time is treated as one pixel of distance.

### Rotate or shear

Both read along the same tilted line; they space the samples differently.

- **rotate** spaces samples one pixel apart along the line itself, as if the
  cuboid were rotated. The steeper the angle, the less of the scene's width a
  frame covers, and the more of it is made of time.
- **shear** takes one sample per input column: output column x is input
  column x, delayed by tan(angle) frames per pixel across. The whole scene
  stays in view at every angle, but the delay becomes infinite at 90°, so
  shear needs an angle between -90° and 90°. Past 45° it skips frames between
  neighbouring columns, so fast motion can break up.

### Whole plane or `--inside`

By default the whole plane is swept through the cuboid, from where it first
touches a corner to where it leaves. Parts of the plane outside the video
come out black, and rotated frames are wider than the input.

With `--inside`, each frame is exactly the input's width and stays entirely
inside the video. A frame W pixels wide tilted at an angle covers about
W × sin(angle) frames of time (W × tan(angle) for shear), so steep angles
need long clips. If the clip is too short, the script stops, says which
angles would fit, and suggests dropping `--inside`.

### Motion (`--inside` with rotate)

An inside frame can move through the cuboid along different straight lines:

- **perpendicular** to itself, as the whole-plane sweep does. Every frame is a
  fresh slice, but a full-width frame has little room to move sideways, so
  this gives short videos at small angles.
- **time**: straight forward through time, like ordinary playback but tilted.
- **longest** (default): the longest straight line that fits. It usually
  runs mostly through time, drifting sideways as far as there is room.

Shear frames always use every input column, so they can only move through
time. The whole-plane sweep always moves perpendicular, because there the
frame already holds the plane's whole cut through the video and another
direction would only change the speed.

## Code layout

`timeslice.py` is split so that new kinds of slices only need new coordinates:

- `sample(volume, t, y, x)` reads the video at any (t, y, x) positions, so it
  works for any surface. Each point blends the 8 voxels around it, weighted
  by closeness (linear interpolation); points outside the cuboid are black.
- `sample_columns(volume, t, x)` is a faster version for slices made of whole
  source columns, which is what any tilt about the y axis gives. It only
  blends across x and t (4 voxels), or copies the nearest one for previews.
- `rotation_sweep(...)` and `shear_sweep(...)` plan a sweep: the output size,
  and `at(f)`, the (t, x) position of every column of output frame f.
- `plan_sweep(...)` picks between them.
- `main()` handles decoding and encoding (via ffmpeg).

`webapp.py` is the web server (Flask), and `web/` holds the page it serves.

Run the geometry checks with `python test_timeslice.py`, and the server's with
`python test_webapp.py` (or run both with `pytest`).
