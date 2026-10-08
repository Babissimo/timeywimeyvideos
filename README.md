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
| `--noise A` | push each point of the surface up to A frames off the plane with Perlin noise (default 0: flat; see below) |
| `--noise-size PX` | roughly how far apart the bumps are, in pixels (default 64) |
| `--noise-speed PX` | how fast the bumps change, in pixels per output frame (default 1; 0 keeps them still) |
| `--noise-direction time\|perpendicular` | which way the noise pushes points (default `time`) |
| `--noise-seed N` | which noise pattern to use (default 0) |
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
  below. With surface noise on it still draws the flat plane the points are
  pushed off, not the bumps.
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

### Surface noise (`--noise`)

`--noise A` makes the slicing surface bumpy instead of flat: Perlin noise
(Ken Perlin's "Improving Noise", SIGGRAPH 2002) pushes each pixel of the
frame off the plane by up to A frames (one frame = one pixel). The noise is a
smooth random function of the pixel's position in the frame and of how far
the sweep has got, so the bumps change as the frame moves. It works with
every slice, angle and motion.

- `--noise-direction time` (default) reads each pixel up to A frames earlier
  or later than the flat frame would. It only shows where something moves:
  still parts of the scene look the same at every moment. On a rotated frame
  near 90° the frame itself runs through time, so this slides points along
  the frame rather than off it.
- `--noise-direction perpendicular` moves each point straight off the plane
  in x–t, which is a true bump at every angle. At 0° it's the same as `time`;
  otherwise it moves points sideways in x too, so a sheared frame's column x
  is no longer exactly input column x.
- `--noise-size` is roughly how far apart the bumps are, and `--noise-speed`
  how fast they change: the pattern moves that many pixels through a third
  noise dimension per output frame. With 0 the surface keeps one shape as it
  sweeps. `--noise-seed` picks a different pattern.

The noise never moves a point further than A, and the sweep allows for that.
The whole-plane sweep starts earlier and ends later, by as far as the bumps
can reach ahead of the plane, so their first and last contact with the video
is included. With `--inside`, the frame keeps that far from the video's
edges, so no bump comes out black. That makes inside videos shorter,
and some combinations don't fit:

- Perpendicular noise pushes the ends of a full-width rotated frame sideways.
  At small angles (but not exactly 0°) there's no room for that, so these
  fit only once the frame has turned far enough. The error message says
  which angles fit.
- A sheared frame always spans the video's whole width, so with
  perpendicular noise and `--inside` it only fits at 0°.

To save time, the noise is worked out at points `size`/8 pixels apart and
blended smoothly between them (Catmull-Rom). That comes within half a
percent of the amplitude of the exact Perlin noise, and still never moves a
point further than A. For bumps under 16 pixels apart, where those points
would be under 2 pixels apart, it's worked out at every pixel instead.

Previews and the live view shrink A and the size with the clip, so they show
the same bumps as the full render. (For bumps 16 to 32 pixels apart the full
render uses those points but the half-size preview works the noise out at
every pixel, so the two can differ by up to that half percent.)

Noise costs little time: on a 4-core machine, slicing a 1080p frame took
about 22 ms against 17 flat, and full renders, which spend most of their
time encoding, took no measurably longer. Preview frames took about 5 ms
against 4.4.

## Code layout

`timeslice.py` is split so that new kinds of slices only need new coordinates:

- `sample(volume, t, y, x)` reads the video at any (t, y, x) positions, so it
  works for any surface. Each point blends the 8 voxels around it, weighted
  by closeness (linear interpolation); points outside the cuboid are black.
- `sample_columns(volume, t, x)` is a faster version for slices made of whole
  source columns, which is what any tilt about the y axis gives. It only
  blends across x and t (4 voxels), or copies the nearest one for previews.
- `rotation_sweep(...)` and `shear_sweep(...)` plan a sweep: the output size,
  `at(f)`, the (t, x) position of every column of output frame f, and the
  normal of the frame's plane. Given noise, they make room for it.
- `plan_sweep(...)` picks between them.
- `Noise` is the surface noise: `field(...)` gives its values over a frame,
  blended from points `NOISE_STEPS` to every `size` pixels, and
  `push(normal)` which way and how far they move a point.
- `surface(...)` gives the (t, y, x) position of every pixel of a noisy
  frame, which `sample` can read.
- `sample_noisy_columns(...)` is a faster way to read a noisy frame: it
  works out the noise as it goes, and since each pixel stays in its row it
  only blends across x and t, like `sample_columns`. It gives exactly what
  `sample` gives at `surface`'s points.
- `slice_frame(...)` makes an output frame, with `sample_columns` when the
  surface is flat and `sample_noisy_columns` when it's bumpy.
- `main()` handles decoding and encoding (via ffmpeg).

`webapp.py` is the web server (Flask), and `web/` holds the page it serves.

Run the geometry checks with `python test_timeslice.py`, and the server's with
`python test_webapp.py` (or run both with `pytest`).
