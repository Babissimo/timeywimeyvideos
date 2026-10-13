# timeywimeyvideos

Treat a video as a cuboid of voxels with axes x, y and time, then play it back
by sweeping a slicing plane through it at an angle, instead of the usual
"one plane per moment" playback.

For now the plane can be tilted about the **y axis** (the vertical axis of
the picture).

It runs in the browser, at <https://timeywimey.babissimo.net/>: the page
decodes a clip, slices it on the GPU and encodes the result there.

## Using the page

Open videos or drop them onto the page, or open a folder to list the videos
at its top level (`.mp4`, `.mov`, `.m4v`, `.mkv` and `.webm`). The page reads
a video where it is, a piece at a time as it needs it: nothing is uploaded or
copied.

- **Live preview** plays the slice while you change the angle and other
  options. The browser decodes a half-size copy of the clip, holds it on the
  GPU and slices each frame there. Space plays and pauses; the arrow keys
  step a frame.
- **Cuboid** draws the clip as a box (one unit per pixel or frame), with the
  first frame at the front and time running back, the current output frame
  drawn where it cuts through, and the region the sweep covers shaded on top.
  Drag to turn it, scroll to zoom, or pick a view; **Top** is the x-t diagram
  below. With surface noise on it still draws the flat plane the points are
  pushed off, not the bumps. On a loop, each part of the frame past an end
  (or a side) of the box is drawn where it wraps round to.
- **Render preview** and **Render full quality** render the slice with the
  options on the page and encode it as H.264 in an MP4, named after the clip
  and the options that differ from the defaults. A render runs on its own
  while you carry on with the page, and can be cancelled. Renders are listed
  below the player, to watch, download or remove, until you leave or reload
  the page, which asks first while a render runs or one hasn't been
  downloaded.

| option | meaning |
| --- | --- |
| Rotate, Shear | how to read along the tilted plane (see [Rotate or shear](#rotate-or-shear)) |
| Angle about y | tilt of the slicing plane, in degrees (default 45) |
| At the video's edges | **Let black in** (default) sweeps the whole plane; **Stay inside** keeps each frame the clip's width and entirely inside the video; **Wrap round** makes a loop (see [Loops](#loops)) |
| Crossfade the ends | with Wrap round: blend the clip's last so many seconds into its first to hide the join |
| Wrap round the sides too | with Wrap round: the picture's left edge follows its right, so the frame can move sideways (see [Loops round the sides](#loops-round-the-sides)) |
| Crossfade the sides | blend the picture's last so many columns into its first to hide the seam |
| Motion | which way an inside or side-wrapping frame moves (see [Motion](#motion)) |
| Surface noise | push each point of the surface off the plane with Perlin noise (see [Surface noise](#surface-noise)) |
| Start, Duration | use only part of the clip, in seconds |
| Scale | resize each frame first, e.g. `0.5`; keeps every frame |
| Output fps | output frame rate, such as `24` or `30000/1001` (defaults to the clip's) |

A preview, and the live view, halve the clip in x, y **and** time (every
other pixel, every other frame) before slicing. Because one frame counts as
one pixel, shrinking all three keeps the cuboid's shape, so a preview shows
the same slices as the full render at half size. It also copies the nearest
voxel instead of blending. It plays at half the frame rate, so it runs as
long as the full render would.

Scale only resizes the frames and keeps every frame, so it changes the
cuboid's shape: the same angle gives a different slice at a different scale.

The page needs a browser with WebGL2 and WebCodecs, such as a recent Chrome
or Edge, that can decode the clip's video and encode H.264. Opening a folder
needs Chrome or Edge too; in other browsers, open or drop the videos
themselves.

The GPU holds the clip as luma (brightness) at full size and colour at half
size each way, 1.5 bytes for every pixel of every frame, in two pieces: the
luma's 1 byte a pixel in one, the colour in the other. A GPU may give no more
than 1 or 2 GB to one piece, and holds at most so many frames in one (2048 on
many). The live view and previews hold the clip at half size and half the
frame rate, a full-quality render at full size and every frame, so a full
render of 30 fps 1080p video runs out of room at some 20 to 30 seconds of it.
The page says when a clip needs more than the GPU can give: set a duration or
scale to use part of a long video.

## How the angle works

Picture the cuboid from above: x runs left to right and time runs up the
page, like a space-time diagram. Ordinary playback slides a horizontal line
(one moment) up the page. The angle tilts that line anticlockwise:

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

- **Rotate** spaces samples one pixel apart along the line itself, as if the
  cuboid were rotated. The steeper the angle, the less of the scene's width a
  frame covers, and the more of it is made of time.
- **Shear** takes one sample per input column: output column x is input
  column x, delayed by tan(angle) frames per pixel across. The whole scene
  stays in view at every angle, but the delay becomes infinite at 90°, so
  shear needs an angle between -90° and 90°. Past 45° it skips frames between
  neighbouring columns, so fast motion can break up.

### Let black in, or stay inside

With **Let black in**, the whole plane is swept through the cuboid, from
where it first touches a corner to where it leaves. Parts of the plane
outside the video come out black, and rotated frames are wider than the
input.

With **Stay inside**, each frame is exactly the input's width and stays
entirely inside the video. A frame W pixels wide tilted at an angle covers
about W × sin(angle) frames of time (W × tan(angle) for shear), so steep
angles need long clips. If the clip is too short, the page says which angles
would fit.

### Motion

An inside rotated frame can move through the cuboid along different
straight lines:

- **Perpendicular to the frame**, as the whole-plane sweep does. Every frame
  is a fresh slice, but a full-width frame has little room to move sideways,
  so this gives short videos at small angles.
- **Straight through time**, like ordinary playback but tilted.
- **Longest line that fits** (default): it usually runs mostly through time,
  drifting sideways as far as there is room.

Shear frames always use every input column, so they can only move through
time. The whole-plane sweep always moves perpendicular, because there the
frame already holds the plane's whole cut through the video and another
direction would only change the speed. A loop round the sides moves straight
through time (the default there) or perpendicular to the frame; no line on
it is the longest (see [Loops round the sides](#loops-round-the-sides)).

### Surface noise

Surface noise makes the slicing surface bumpy instead of flat: Perlin noise
(Ken Perlin's "Improving Noise", SIGGRAPH 2002) pushes each pixel of the
frame off the plane by up to the amplitude, in frames (one frame = one
pixel). The noise is a smooth random function of the pixel's position in the
frame and of how far the sweep has got, so the bumps change as the frame
moves. It works with every slice, angle and motion.

- **Push through time** (default) reads each pixel up to the amplitude
  earlier or later than the flat frame would. It only shows where something
  moves: still parts of the scene look the same at every moment. On a rotated
  frame near 90° the frame itself runs through time, so this slides points
  along the frame rather than off it.
- **Push perpendicular** moves each point straight off the plane in x–t,
  which is a true bump at every angle. At 0° it's the same as pushing through
  time; otherwise it moves points sideways in x too, so a sheared frame's
  column x is no longer exactly input column x.
- **Size** is roughly how far apart the bumps are, and **Speed** how fast
  they change: the pattern moves that many pixels through a third noise
  dimension per output frame. With 0 the surface keeps one shape as it
  sweeps. **Seed** picks a different pattern.

The noise never moves a point further than the amplitude, and the sweep
allows for that. The whole-plane sweep starts earlier and ends later, by as
far as the bumps can reach ahead of the plane, so their first and last
contact with the video is included. Inside, the frame keeps that far from
the video's edges, so no bump comes out black. That makes inside videos
shorter, and some combinations don't fit:

- Perpendicular noise pushes the ends of a full-width rotated frame sideways.
  At small angles (but not exactly 0°) there's no room for that, so these
  fit only once the frame has turned far enough. The page says which angles
  fit.
- A sheared frame always spans the video's whole width, so with
  perpendicular noise it stays inside only at 0°.

To save time, the noise is worked out at points `size`/8 pixels apart and
blended smoothly between them (Catmull-Rom). That comes within half a
percent of the amplitude of the exact Perlin noise, and still never moves a
point further than the amplitude. For bumps under 16 pixels apart, where
those points would be under 2 pixels apart, it's worked out at every pixel
instead.

Previews and the live view shrink the amplitude and the size with the clip,
so they show the same bumps as the full render. (For bumps 16 to 32 pixels
apart the full render uses those points but the half-size preview works the
noise out at every pixel, so the two can differ by up to that half percent.)

### Loops

**Wrap round** makes a video that plays on repeat without a jump. Time runs
round in a ring, the clip's first frame following its last, and the frame
moves straight through time once round it. So the video is as long as the
clip, and its last frame leads back into its first. At 0° it is the clip
itself.

Each frame is the input's width and stays inside the video, as with Stay
inside, but since time wraps round, no clip is too short for an angle. Only
perpendicular noise can take a frame out of the video, by pushing its ends
sideways. As with Stay inside, a rotated frame has no room for that at small
angles other than 0° (the page says which angles fit), and a sheared one at
any angle but 0°.

The join from the clip's last frame back to its first becomes part of the
video. Unless the clip already loops, it shows as a tear in time that
crosses each frame once per loop. **Crossfade the ends** hides it by blending
the clip's last so many seconds into its first, which makes the loop that
much shorter. Frames in the fade mix two moments, so anything moving shows
twice as the fade passes, in a band that crosses the frame (at 0°, a plain
crossfade). The fade can be at most half the clip.

Surface noise loops too: after one loop the bumps are back in the shape they
started in. For that, the noise has to move through a whole number of its
cells (each Size across) per loop, so the speed is rounded to the nearest
speed that does, and to at least two cells, since over one the bumps barely
change. A preview's loop has half as many frames, so where the rounding is
close it can land on one cell more or fewer than the full render's.

### Loops round the sides

A loop that only wraps time has to move straight through time, since a
frame drifting sideways would leave the video and never come back. At steep
angles that is the dull way to move: the frame lies nearly along time, so
moving through time mostly slides it along itself.

**Wrap round the sides too** wraps x as well, the picture's left edge
following its right, which makes the clip a torus. Motion perpendicular to
the frame can then move it straight off itself, as the whole-plane sweep
does, and keep going. The loop closes once the frame has gone across the
width a whole number of times and round time a whole number of times, so its
direction is nudged to the shortest such path within 2° of perpendicular (or
the closest, if none within 16 times round either way is that close). Loops
can be much longer than the clip: at 80°, a 6-second clip 320 pixels wide
gives a 32-second loop, three times across the width and once round time.

With the sides wrapping round, noise can't push the frame out of the video
either, so perpendicular noise fits at every angle.

The picture's right edge now meets its left, so unless the footage wraps
round (a 360° panorama, say), a seam shows wherever a frame crosses it.
**Crossfade the sides** hides it the way Crossfade the ends hides the join in
time, blending the picture's last so many columns into its first, which
makes it that much narrower (after scaling). Like the ends' fade, it can be
at most half.

## Working on it

The page is built with npm, which comes with [Node.js](https://nodejs.org/).
Run

```sh
npm install
npm run dev
```

and open the address it prints. It serves the page straight from `web/` and
reloads it as you edit.

To host it, run `npm run build` and serve `web/dist/` as static files from
any web server, such as `npx vite preview`. WebCodecs runs only in a secure
context, so serve the page over HTTPS or open it at `localhost` or
`127.0.0.1`; at any other plain `http://` address the page says it can't
decode video. Browsers won't run its scripts from a file opened straight from
disk. On every push to `main`, `.github/workflows/deploy-pages.yml` runs the
checks, builds the page and publishes it on GitHub Pages.

`web/app.js` runs the page. The TypeScript in `web/src/` does the work:

- `decode.ts` reads a clip's frames, `encode.ts` writes a render's.
- `planner.ts` plans a sweep: the output size and where each column of each
  frame reads the cuboid. `noise.ts` is the surface noise, and `options.ts`
  reads the page's options into a plan.
- `gpu/` holds the clip on the GPU (`volume.ts`) and slices it there
  (`slicer.ts`, `shaders.ts`).
- `live.ts` runs the live preview; `render.ts` runs a render in a worker of
  its own (`render.worker.ts`).
- `sources.ts` lists the videos the page offers.

`npm test` runs the checks, some of them in both Google Chrome, which has
to be installed, and Playwright's Firefox, which `npx playwright install firefox`
installs; `npm run test:unit` runs only those that need no browser. On
Linux, Firefox needs more to decode H.264 and draw with WebGL: run it as
`.github/workflows/deploy-pages.yml` does. `npm run typecheck` checks the
TypeScript. The expected values in
`web/test/fixtures/` (the JSON files and `decode/frames.bin.gz`) were
recorded from an earlier reference implementation, and nothing can make them
afresh: when a change breaks a test that reads one, replace it with a test of
the behaviour itself.
