#!/usr/bin/env python3
"""Slice through a video's space-time cuboid at an angle.

A video is a block of voxels indexed (t, y, x). Playing it normally means
sweeping a plane of constant t through that block, front to back. This script
tilts the plane about the y axis, so each output frame shows different moments
in time across its width, and moves it through the block.

Picture the x-t plane as a space-time diagram, x to the right and t up. A
normal frame is a horizontal line; a positive angle tilts that line
anticlockwise, so the right edge of each output frame is later in time than the
left edge. There are two ways to read along the tilted line:

  rotate  Samples are evenly spaced along the line itself, as if the cuboid
          were rotated. At 90 degrees each output frame is a y-t slice.
  shear   Output column x is input column x, delayed by tan(angle) frames per
          pixel across. Every column of the original stays in view.

By default the whole plane is swept through the cuboid and the parts of it
outside the video are black. With --inside, each frame is the input's width
and stays entirely inside the video.

With --noise, Perlin noise pushes each point of the frame off the plane, so
the surface is bumpy instead of flat, and the bumps change as it sweeps.

Units: one frame of time is treated as one pixel of distance, so the cuboid is
width x height x frame-count voxels.

    python timeslice.py input.mp4 output.mp4 --angle 30
"""

import argparse
import functools
import json
import math
import subprocess
import sys
import time
from fractions import Fraction
from typing import Callable, NamedTuple

import numpy as np
from numba import njit, prange

PREVIEW_SCALE = 0.5  # --preview shrinks x, y and t by this factor


class DoesNotFit(ValueError):
    """The frame can't fit inside the video at the requested angle."""


class VideoError(RuntimeError):
    """ffmpeg couldn't read or write a video."""


def probe(path):
    """Return (width, height, frame rate, duration in seconds or None)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:stream_tags=rotate"
                          ":stream_side_data=rotation:format=duration",
         "-of", "json", path],
        capture_output=True, text=True)
    info = json.loads(out.stdout or "{}")
    if out.returncode != 0 or not info.get("streams"):
        raise VideoError(f"can't read a video stream from {path}")
    stream = info["streams"][0]
    width, height = stream["width"], stream["height"]

    # Phone videos are often stored sideways with a rotation flag; ffmpeg
    # applies it when decoding, so the decoded frames have swapped dimensions.
    rotation = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []):
        rotation = side_data.get("rotation", rotation)
    if abs(int(float(rotation))) % 180 == 90:
        width, height = height, width

    duration = info.get("format", {}).get("duration")
    return (width, height, Fraction(stream["r_frame_rate"]),
            float(duration) if duration else None)


def clip_seconds(length, start=None, duration=None):
    """How many seconds of a video `length` seconds long (None if unknown)
    are used when starting at `start` and keeping at most `duration`."""
    seconds = None if length is None else max(length - (start or 0), 0)
    if duration is not None:
        seconds = duration if seconds is None else min(seconds, duration)
    return seconds


def _read_exactly(pipe, buffer):
    """Fill a numpy array from a pipe. Returns False if the pipe ran out first."""
    view = memoryview(buffer).cast("B")
    got = 0
    while got < len(view):
        n = pipe.readinto(view[got:])
        if not n:
            return False
        got += n
    return True


def load_video(path, scale=1.0, time_scale=1.0, start=None, duration=None,
               fast=False):
    """Decode a video into a uint8 array of shape (frames, height, width, 3).

    scale resizes each frame; time_scale keeps that fraction of the frames
    (0.5 = every other frame). Returns the array and the input's frame rate.
    fast trades decoding and resizing quality for speed.
    """
    width, height, fps, length = probe(path)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))
    rate = fps * Fraction(time_scale).limit_denominator(1000)

    filters = [f"scale={width}:{height}" + (":flags=neighbor" if fast else "")]
    if rate != fps:
        filters.insert(0, f"fps={rate.numerator}/{rate.denominator}")
    cmd = ["ffmpeg", "-v", "error"]
    if fast:
        cmd += ["-skip_loop_filter", "all"]  # skip H.264's deblocking pass
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vf", ",".join(filters),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    # Read frames straight into one array sized from the expected frame
    # count, growing it if the estimate was short; this avoids holding two
    # copies of the video in memory.
    seconds = clip_seconds(length, start, duration)
    capacity = 256 if seconds is None else int(seconds * rate * 1.05) + 2
    volume = np.empty((capacity, height, width, 3), np.uint8)
    n = 0
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0) as proc:
        while True:
            if n == len(volume):
                volume = np.concatenate([volume, np.empty_like(volume[:n // 2 + 1])])
            if not _read_exactly(proc.stdout, volume[n]):
                break
            n += 1
    if proc.returncode != 0:
        raise VideoError(f"ffmpeg failed to decode {path}")
    if n == 0:
        raise VideoError(f"no frames decoded from {path}")
    return volume[:n], fps


def crossfade(volume, frames):
    """Blend the last `frames` frames of a (T, H, W, C) volume into its first,
    in place, and return the first T - frames: a clip that runs on smoothly
    from its end into its start when time runs round in a ring.

    Frame i of the fade mixes the clip's frame i with frame T - frames + i,
    as far past the loop's end as frame i is past its start, taking more of
    frame i the further in it is. The fade can be at most half the clip.
    """
    length = len(volume) - frames
    if not 0 <= frames <= length:
        raise ValueError(f"a {frames}-frame fade doesn't fit in a "
                         f"{len(volume)}-frame clip: it can be at most half")
    for i in range(frames):
        w = np.float32((i + 1) / (frames + 1))  # float32 halves the frame-sized temporaries
        mixed = volume[length + i] * (1 - w)
        mixed += volume[i] * w
        volume[i] = (mixed + 0.5).astype(np.uint8)  # rounded, as the samplers do
    return volume[:length]


def open_writer(path, width, height, fps, preset="medium", crf=18):
    """Start an ffmpeg process that encodes raw RGB frames written to its stdin."""
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
           # H.264 needs even dimensions; pad with a black row/column if not.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:v", "libx264", "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", path]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


# Sampling. Voxel centres sit at whole-number coordinates. A point between
# voxels is a blend of its neighbours, each weighted by how close the point is
# to it (linear interpolation). Points outside the cuboid come out black,
# unless time wraps: then it runs round in a ring, the first frame following
# the last, and no point is outside it in time. x can wrap the same way, the
# picture's left edge following its right. The loops are compiled by numba and
# spread across all CPU cores.

@njit(cache=True)
def _neighbours(coord, n, wrap=False):
    """The voxel indices either side of coord along an axis of length n, and
    how far coord is past the first (0 = on it, 0.999 = nearly on the second).
    With wrap the axis is a ring, so past the last voxel comes the first."""
    lo = int(np.floor(coord))
    if wrap:
        return lo % n, (lo + 1) % n, coord - lo
    return max(lo, 0), min(lo + 1, n - 1), coord - lo


@njit(parallel=True, cache=True)
def _sample_points(volume, t, y, x, nearest, wrap, wrap_x, out):
    n_t, n_y, n_x, n_c = volume.shape
    for i in prange(out.shape[0]):
        for j in range(out.shape[1]):
            tt, yy, xx = t[i, j], y[i, j], x[i, j]
            if not ((wrap or -0.5 <= tt < n_t - 0.5) and -0.5 <= yy < n_y - 0.5
                    and (wrap_x or -0.5 <= xx < n_x - 0.5)):
                out[i, j, :] = 0
                continue
            t0, t1, ft = _neighbours(tt, n_t, wrap)
            y0, y1, fy = _neighbours(yy, n_y)
            x0, x1, fx = _neighbours(xx, n_x, wrap_x)
            if nearest:
                ti = t1 if ft >= 0.5 else t0
                yi = y1 if fy >= 0.5 else y0
                xi = x1 if fx >= 0.5 else x0
                out[i, j, :] = volume[ti, yi, xi, :]
                continue
            for c in range(n_c):
                # Blend along x, then y, then t: 8 voxels in all.
                v = (((volume[t0, y0, x0, c] * (1 - fx) + volume[t0, y0, x1, c] * fx) * (1 - fy)
                      + (volume[t0, y1, x0, c] * (1 - fx) + volume[t0, y1, x1, c] * fx) * fy) * (1 - ft)
                     + ((volume[t1, y0, x0, c] * (1 - fx) + volume[t1, y0, x1, c] * fx) * (1 - fy)
                        + (volume[t1, y1, x0, c] * (1 - fx) + volume[t1, y1, x1, c] * fx) * fy) * ft)
                out[i, j, c] = min(255, int(v + 0.5))


@njit(cache=True, inline="always")  # a call per pixel would cost more than the read
def _read_in_row(volume, tt, row, xx, nearest, wrap, wrap_x, out, j):
    """Read the volume at (tt, row, xx) into out[row, j], blending across x
    and t only, since row is a whole number."""
    n_t, _, n_x, n_c = volume.shape
    if not ((wrap or -0.5 <= tt < n_t - 0.5) and (wrap_x or -0.5 <= xx < n_x - 0.5)):
        out[row, j, :] = 0
        return
    t0, t1, ft = _neighbours(tt, n_t, wrap)
    x0, x1, fx = _neighbours(xx, n_x, wrap_x)
    if nearest:
        ti = t1 if ft >= 0.5 else t0
        xi = x1 if fx >= 0.5 else x0
        out[row, j, :] = volume[ti, row, xi, :]
        return
    for c in range(n_c):
        # Blend along x, then t: 4 voxels in all.
        v = ((volume[t0, row, x0, c] * (1 - fx) + volume[t0, row, x1, c] * fx) * (1 - ft)
             + (volume[t1, row, x0, c] * (1 - fx) + volume[t1, row, x1, c] * fx) * ft)
        out[row, j, c] = min(255, int(v + 0.5))


@njit(parallel=True, cache=True)
def _sample_columns(volume, t, x, nearest, wrap, wrap_x, out):
    n_t, n_y, n_x, n_c = volume.shape
    n = t.shape[0]
    # Every row of output column j reads the same source columns with the same
    # weights, so work those out once per column rather than once per pixel.
    inside = np.empty(n, np.bool_)
    t0s, t1s = np.empty(n, np.int64), np.empty(n, np.int64)
    x0s, x1s = np.empty(n, np.int64), np.empty(n, np.int64)
    fts, fxs = np.empty(n), np.empty(n)
    for j in range(n):
        inside[j] = ((wrap or -0.5 <= t[j] < n_t - 0.5)
                     and (wrap_x or -0.5 <= x[j] < n_x - 0.5))
        if inside[j]:
            t0s[j], t1s[j], fts[j] = _neighbours(t[j], n_t, wrap)
            x0s[j], x1s[j], fxs[j] = _neighbours(x[j], n_x, wrap_x)
            if nearest:  # keep only the closest voxel, in t0s and x0s
                if fts[j] >= 0.5:
                    t0s[j] = t1s[j]
                if fxs[j] >= 0.5:
                    x0s[j] = x1s[j]

    for row in prange(n_y):
        for j in range(n):
            if not inside[j]:
                for c in range(n_c):
                    out[row, j, c] = 0
                continue
            t0, t1, ft = t0s[j], t1s[j], fts[j]
            x0, x1, fx = x0s[j], x1s[j], fxs[j]
            if nearest:
                for c in range(n_c):
                    out[row, j, c] = volume[t0, row, x0, c]
                continue
            for c in range(n_c):
                # Blend along x, then t: 4 voxels in all.
                v = ((volume[t0, row, x0, c] * (1 - fx) + volume[t0, row, x1, c] * fx) * (1 - ft)
                     + (volume[t1, row, x0, c] * (1 - fx) + volume[t1, row, x1, c] * fx) * ft)
                out[row, j, c] = min(255, int(v + 0.5))


def sample(volume, t, y, x, nearest=False, wrap=False, wrap_x=False):
    """Sample a (T, H, W, C) volume at any continuous (t, y, x) points.

    t, y and x broadcast together to the (height, width) of the output frame,
    so any surface through the cuboid can be read this way. With nearest,
    each pixel is copied from the closest voxel instead of blended. With
    wrap, time runs round in a ring: time t reads frame t mod T. With wrap_x,
    so does x: position x reads column x mod W.
    """
    shape = np.broadcast_shapes(np.shape(t), np.shape(y), np.shape(x))
    t, y, x = (np.broadcast_to(np.asarray(a, np.float64), shape) for a in (t, y, x))
    out = np.empty(shape + volume.shape[3:], np.uint8)
    _sample_points(volume, t, y, x, nearest, wrap, wrap_x, out)
    return out


def sample_columns(volume, t, x, nearest=False, wrap=False, wrap_x=False):
    """A faster sample() for slices built from whole source columns.

    Output column j is the full-height column of the input at time t[j] and
    position x[j], so only x and t need blending (4 voxels, not 8). Slices that
    only tilt about the y axis have this shape. nearest, wrap and wrap_x are
    as for sample().
    """
    out = np.empty((volume.shape[1], len(t)) + volume.shape[3:], np.uint8)
    _sample_columns(volume, np.asarray(t, np.float64), np.asarray(x, np.float64),
                    nearest, wrap, wrap_x, out)
    return out


# Perlin noise, for pushing the slicing surface off its plane. This is Ken
# Perlin's improved noise (2002): a smooth random function of three
# coordinates, zero at every point of a whole-number lattice and swelling up or
# down in between.

# No value of _noise3 is further from 0 than this. Each lattice corner adds
# g.d, the dot product of its gradient g (one of 12 vectors like (1, 1, 0))
# with the offset d from that corner, which is at most the two largest
# components of |d| added together. Weighting those by how much each corner
# counts and maximising over the cell gives 1.03635.
NOISE_BOUND = 1.0364


@njit(cache=True)
def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


@njit(cache=True)
def _lerp(t, a, b):
    return a + t * (b - a)


@njit(cache=True)
def _grad(h, x, y, z):
    """The dot product of (x, y, z) with one of the 12 vectors from a cube's
    centre to the middles of its edges, picked by the hash h."""
    h &= 15
    u = x if h < 8 else y
    v = y if h < 4 else (x if h == 12 or h == 14 else z)
    return (u if h & 1 == 0 else -u) + (v if h & 2 == 0 else -v)


@njit(cache=True)
def _noise3(perm, x, y, z, period):
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    X, Y = int(fx) & 255, int(fy) & 255
    # The lattice planes either side in z. With a period they wrap round
    # every `period` cells, so the noise repeats along z.
    Z0, Z1 = int(fz), int(fz) + 1
    if period:
        Z0, Z1 = Z0 % period, Z1 % period
    Z0, Z1 = Z0 & 255, Z1 & 255
    x, y, z = x - fx, y - fy, z - fz
    u, v, w = _fade(x), _fade(y), _fade(z)
    # Hash the 8 corners of the lattice cell around the point.
    a, b = perm[X] + Y, perm[X + 1] + Y
    aa, ab, ba, bb = perm[a], perm[a + 1], perm[b], perm[b + 1]
    return _lerp(w, _lerp(v, _lerp(u, _grad(perm[aa + Z0], x, y, z),
                                      _grad(perm[ba + Z0], x - 1, y, z)),
                             _lerp(u, _grad(perm[ab + Z0], x, y - 1, z),
                                      _grad(perm[bb + Z0], x - 1, y - 1, z))),
                    _lerp(v, _lerp(u, _grad(perm[aa + Z1], x, y, z - 1),
                                      _grad(perm[ba + Z1], x - 1, y, z - 1)),
                             _lerp(u, _grad(perm[ab + Z1], x, y - 1, z - 1),
                                      _grad(perm[bb + Z1], x - 1, y - 1, z - 1))))


@njit(parallel=True, cache=True)
def _noise_grid(perm, xs, ys, z, period, out):
    for i in prange(len(ys)):
        for j in range(len(xs)):
            out[i, j] = _noise3(perm, xs[j], ys[i], z, period)


@functools.lru_cache(maxsize=16)
def _permutation(seed):
    """The shuffled lattice hash for a seed, written out twice so that
    indexing past 255 wraps around."""
    p = np.random.default_rng(seed).permutation(256)
    return np.concatenate([p, p]).astype(np.int64)


# Working the noise out at every pixel is slow, and it changes little from
# one pixel to the next, so it's worked out at nodes NOISE_STEPS to every
# `size` pixels and blended smoothly in between (Catmull-Rom). That comes
# within about half a percent of the amplitude of the noise itself. Nodes
# under 2 pixels apart would save nothing, so then there's one at every
# pixel, which gives the noise exactly.
NOISE_STEPS = 8


def _catmull_rom(t):
    """For points t of the way (0 to 1) between the middle two of 4 evenly
    spaced nodes, how much each node counts, shape (len(t), 4)."""
    t2, t3 = t * t, t * t * t
    return np.stack([-t3 + 2 * t2 - t, 3 * t3 - 5 * t2 + 2,
                     -3 * t3 + 4 * t2 + t, t3 - t2], -1) / 2


def _nodes(n, size):
    """Where to work out the noise along an axis of n pixels, and how to
    blend it back: the nodes' noise coordinates, and for each pixel the
    first of the 4 nodes around it and their weights."""
    step = size / NOISE_STEPS
    if step < 2:
        step = 1.0
    pos = np.arange(n) / step
    first = np.floor(pos).astype(np.int64)
    # One node before the first pixel and two past the last, for the blend.
    return (np.arange(first[-1] + 4) - 1) * step / size, first, \
        _catmull_rom(pos - first)


@njit(cache=True, inline="always")
def _blend_row(nodes, cols, col_w, first_row, row_w, across, out):
    """Blend the noise at the nodes back to one row of pixels, into out:
    first the 4 rows of nodes around it into one (across), then each pixel
    from the 4 nodes around it. Clamped to -1 to 1, which a blend could
    otherwise overshoot by a hair."""
    r = first_row
    for m in range(nodes.shape[1]):
        across[m] = (row_w[0] * nodes[r, m] + row_w[1] * nodes[r + 1, m]
                     + row_w[2] * nodes[r + 2, m] + row_w[3] * nodes[r + 3, m])
    for j in range(cols.shape[0]):
        c = cols[j]
        v = (col_w[j, 0] * across[c] + col_w[j, 1] * across[c + 1]
             + col_w[j, 2] * across[c + 2] + col_w[j, 3] * across[c + 3])
        out[j] = min(1.0, max(-1.0, v))


@njit(parallel=True, cache=True)
def _fill_noise(nodes, cols, col_w, rows, row_w, out):
    for row in prange(out.shape[0]):
        _blend_row(nodes, cols, col_w, rows[row], row_w[row],
                   np.empty(nodes.shape[1]), out[row])


class Noise(NamedTuple):
    """Perlin noise that pushes each point of a sweep's frame off its plane.

    Points move up to `amplitude` frames (one frame = one pixel), either
    through "time" or "perpendicular" to the plane, in the x-t plane. `size` is
    roughly how far apart the bumps are, in pixels. The bumps change as the
    sweep goes on, `speed` pixels' worth per output frame; 0 keeps one fixed
    bumpy surface. `seed` picks the pattern. With a `period`, the bumps come
    back to the same shape every `period` output frames, the speed rounded
    to the nearest that does so.
    """
    amplitude: float
    size: float = 64.0
    speed: float = 1.0
    direction: str = "time"
    seed: int = 0
    period: float = 0.0

    def push(self, normal):
        """The (t, x) move of a point where the noise is strongest, on a frame
        whose plane has this (t, x) unit normal."""
        if self.direction == "time":
            return self.amplitude, 0.0
        if self.direction == "perpendicular":
            return self.amplitude * normal[0], self.amplitude * normal[1]
        raise ValueError(f"unknown noise direction {self.direction!r}")

    def scaled(self, factor):
        """The same noise on a video shrunk by factor in x, y and t. speed is
        pixels per output frame, and both shrink together, so it stays."""
        return self._replace(amplitude=self.amplitude * factor,
                             size=self.size * factor, period=self.period * factor)

    def _grid(self, width, height, f):
        """The noise at the nodes for output frame f, from -1 to 1, and how
        to blend it back to each column and row."""
        us, cols, col_w = _nodes(width, self.size)
        vs, rows, row_w = _nodes(height, self.size)
        z, cells = f * self.speed / self.size, 0
        if self.period and self.speed:
            # A whole number of lattice cells per period, for the noise to
            # wrap round in. At least 2: in a ring of 1 the planes either
            # side of a cell are the same, so the bumps barely change.
            cells = max(2, round(self.period * abs(self.speed) / self.size))
            z = math.copysign(f * cells / self.period, self.speed)
        nodes = np.empty((len(vs), len(us)))
        _noise_grid(_permutation(self.seed), us, vs, z, cells, nodes)
        return nodes / NOISE_BOUND, cols, col_w, rows, row_w

    def field(self, width, height, f):
        """The noise over output frame f, from -1 to 1, shape (height, width)."""
        out = np.empty((height, width))
        _fill_noise(*self._grid(width, height, f), out)
        return out


@njit(parallel=True, cache=True)
def _sample_noisy_columns(volume, t, x, push_t, push_x, nodes, cols, col_w,
                          rows, row_w, nearest, wrap, wrap_x, out):
    for row in prange(volume.shape[1]):
        # Blend the whole row's bumps from the nodes, then read the row.
        bumps = np.empty(t.shape[0])
        _blend_row(nodes, cols, col_w, rows[row], row_w[row],
                   np.empty(nodes.shape[1]), bumps)
        for j in range(t.shape[0]):
            _read_in_row(volume, t[j] + push_t * bumps[j], row,
                         x[j] + push_x * bumps[j], nearest, wrap, wrap_x, out, j)


def sample_noisy_columns(volume, t, x, push, noise, f, nearest=False, wrap=False,
                         wrap_x=False):
    """A faster sample() for column slices pushed off their plane by noise.

    Output pixel (row, j) is read at time t[j] and position x[j], moved by
    push (a (t, x) pair) times the noise at that pixel of output frame f.
    It works out the noise as it reads, and since every pixel stays in its
    own row, only blends across x and t (4 voxels, not 8). It gives exactly
    what sample() would at surface()'s points. nearest, wrap and wrap_x are
    as for sample().
    """
    height = volume.shape[1]
    out = np.empty((height, len(t)) + volume.shape[3:], np.uint8)
    _sample_noisy_columns(volume, np.asarray(t, np.float64), np.asarray(x, np.float64),
                          float(push[0]), float(push[1]),
                          *noise._grid(len(t), height, f), nearest, wrap, wrap_x,
                          out)
    return out


# Slice planning. Each function returns a Sweep: the output frame width, the
# number of output frames, at(f), which gives the (t, x) source position of
# every column of output frame f, for use with sample_columns, the (t, x)
# normal of the frame's plane, which perpendicular noise pushes along, and
# whether it loops. A loop runs round time in a ring, so at(f) can lie past
# either end of the clip, and is read with wrap. With sides, x wraps round too
# (wrap_x).
#
# Given noise, a plan allows for it pushing points as far as it can: the
# whole-plane sweep starts and ends early and late enough to catch the bumps,
# and --inside and --loop frames keep far enough from the edges that none leave
# the video.

class Sweep(NamedTuple):
    width: int
    frames: int
    at: Callable[[int], tuple]
    normal: tuple
    loop: bool = False
    sides: bool = False


def _push(noise, normal):
    return (0.0, 0.0) if noise is None else noise.push(normal)


def _angles_that_fit(fits, limit):
    """Say which angles from 0 to `limit` degrees `fits` accepts, to the
    nearest 0.1. (Inside frames fit at -a and 180 - a just as at a.)"""
    runs = []  # [first, last] in tenths of a degree
    for tenth in range(round(limit * 10) + 1):
        if fits(tenth / 10):
            if runs and runs[-1][1] == tenth - 1:
                runs[-1][1] = tenth
            else:
                runs.append([tenth, tenth])
    if not runs:
        return "No angle fits this clip."
    if runs == [[0, 0]]:
        return "Only 0 degrees fits this clip."
    if len(runs) == 1 and runs[0][0] == 0:
        return f"Angles up to {runs[0][1] / 10:g} degrees fit this clip."
    spans = [f"{a / 10:g}" if a == b else f"{a / 10:g} to {b / 10:g}"
             for a, b in runs]
    return f"Angles of {' and '.join(spans)} degrees fit this clip."


def _too_long(angle, width, span_t, push_t, n_frames, kind=""):
    """Why a frame covering span_t frames of time doesn't fit in the clip."""
    noise = f", and the noise can push it {abs(push_t):.3g} frames either way" \
        if push_t else ""
    return (f"At {angle:g} degrees a {width}-pixel-wide {kind}frame covers "
            f"{math.ceil(span_t) + 1} frames of time{noise}, but the clip has "
            f"only {n_frames}.")


LOOP_TOLERANCE = 2.0  # degrees a loop round the sides may stray from its aim
LOOP_WINDINGS = 16    # most times across the width, or round time, it tries


@functools.lru_cache(maxsize=64)  # the live view plans a sweep for every frame
def _windings(width, n_frames, dx, dt):
    """How many times (a, b) a loop through a clip whose sides and ends wrap
    round should go across the width and round time, to close up while moving
    about along (dx, dt): the shortest within LOOP_TOLERANCE degrees of it, or
    failing that the closest."""
    aim = math.atan2(dt, dx)
    best = None
    for a in range(-LOOP_WINDINGS, LOOP_WINDINGS + 1):
        for b in range(-LOOP_WINDINGS, LOOP_WINDINGS + 1):
            if math.gcd(a, b) != 1:  # not a closed path, or one gone round twice
                continue
            off = abs(math.remainder(math.atan2(b * n_frames, a * width) - aim, math.tau))
            length = math.hypot(a * width, b * n_frames)
            key = (0, length) if math.degrees(off) <= LOOP_TOLERANCE else (1, off)
            if best is None or key < best[0]:
                best = key, (a, b)
    return best[1]


def _pushed_sideways(angle, push_x):
    return (f"At {angle:g} degrees the noise can push the ends of the frame "
            f"{abs(push_x):.3g} pixels sideways, out of the video.")


def _rotation(angle):
    theta = math.radians(angle)
    # Round so that 0, 90, 180... degrees land exactly on the voxel grid.
    return round(math.cos(theta), 12), round(math.sin(theta), 12)


def _rotation_room(n_frames, width, angle, noise=None):
    """How far an inside frame's centre can move from the cuboid's centre, in
    t and in x, before any point of it leaves the video. Negative if it can't
    fit at all."""
    c, s = _rotation(angle)
    push_t, push_x = _push(noise, (c, -s))
    span_t = (width - 1) * abs(s)  # frames of time one output frame covers
    return ((n_frames - 1 - span_t) / 2 - abs(push_t),
            (width - 1) * (1 - abs(c)) / 2 - abs(push_x))


def rotation_sweep(n_frames, width, angle, inside=False, motion=None,
                   noise=None, loop=False, sides=False):
    """Plan a sweep with the slicing plane rotated `angle` degrees about y.

    Without `inside`, the frame is wide enough to hold the plane's whole cut
    through the cuboid, and the plane moves perpendicular to itself from where
    it first touches the cuboid to where it leaves.

    With `inside`, the frame is as wide as the input and stays entirely inside
    the cuboid, moving along a straight line through the cuboid's centre:
    "perpendicular" to itself, straight through "time", or along the "longest"
    line that fits (the default). Raises DoesNotFit if the clip is too short
    for the angle.

    With `loop`, time runs round in a ring, the clip's first frame following
    its last. The frame is as wide as the input and moves straight through
    time once round the ring, so the last output frame leads back into the
    first. It stays inside the video, so `inside` makes no difference, and no
    clip is too short.

    With `sides` as well, x wraps round too, the picture's left edge following
    its right, so the frame can also move "perpendicular" to itself. It then
    goes across the width and round time each a whole number of times, along
    the shortest such path within LOOP_TOLERANCE degrees of perpendicular, and
    no noise can take it out of the video.

    With `noise`, the sweep allows for the noise pushing points off the plane.
    """
    if sides and not loop:
        raise ValueError("the sides only wrap round on a loop")
    c, s = _rotation(angle)
    normal = (c, -s)  # (t, x): the way the whole plane sweeps
    push_t, push_x = _push(noise, normal)
    # Move forward in time unless the plane is turned past 90 degrees,
    # where the sweep runs backwards (180 degrees plays the clip in reverse).
    sign_t = 1 if c >= 0 else -1

    if loop:
        if motion not in (None, "time") and not (sides and motion == "perpendicular"):
            raise ValueError("a loop moves straight through time, or with the "
                             "sides wrapping, perpendicular to the frame")
        if not sides and _rotation_room(n_frames, width, angle, noise)[1] < 0:
            raise DoesNotFit(_pushed_sideways(angle, push_x) + " " + _angles_that_fit(
                lambda a: _rotation_room(n_frames, width, a, noise)[1] >= 0, 90))
        out_width = width
        if motion == "perpendicular":
            across, round_t = _windings(width, n_frames, -s, c)
            span_x, span_t = across * width, round_t * n_frames
            out_frames = max(1, round(math.hypot(span_x, span_t)))  # about a pixel a frame
            dx, dt = span_x / out_frames, span_t / out_frames
        else:
            out_frames = n_frames
            dx, dt = 0.0, sign_t
    elif not inside:
        ahead = abs(push_t * c - push_x * s)  # how far bumps reach off the plane
        out_width = max(1, round(width * abs(c) + n_frames * abs(s)))
        out_frames = max(1, round(width * abs(s) + n_frames * abs(c) + 2 * ahead))
        dx, dt = -s, c
    else:
        out_width = width
        # How far the frame's centre can move from the cuboid's centre, in x
        # and in t, before any point of the frame leaves the video.
        room_t, room_x = _rotation_room(n_frames, width, angle, noise)
        if room_t < 0 or room_x < 0:
            if room_t < 0:
                why = _too_long(angle, width, (width - 1) * abs(s), push_t, n_frames)
            else:
                why = _pushed_sideways(angle, push_x)
            raise DoesNotFit(why + " " + _angles_that_fit(
                lambda a: min(_rotation_room(n_frames, width, a, noise)) >= 0, 90))

        motion = motion or "longest"
        if motion == "perpendicular":
            dx, dt = -s, c
        elif motion == "time":
            dx, dt = 0.0, sign_t
        elif motion == "longest":
            # The diagonal of the region the centre can move in. Of the two
            # diagonals, take the one closer to perpendicular, so the frame
            # sweeps through the video rather than sliding along itself.
            dx, dt = (-room_x if s >= 0 else room_x), sign_t * room_t
            length = math.hypot(dx, dt)
            dx, dt = (dx / length, dt / length) if length else (0.0, sign_t)
        else:
            raise ValueError(f"unknown motion {motion!r}")
        reach = min(room_x / abs(dx) if dx else math.inf,
                    room_t / abs(dt) if dt else math.inf)
        out_frames = int(2 * reach + 1e-9) + 1

    centre_x, centre_t = (width - 1) / 2, (n_frames - 1) / 2
    across = np.arange(out_width) - (out_width - 1) / 2  # position along the frame

    def at(f):
        step = f - (out_frames - 1) / 2  # distance moved from the centre
        return centre_t + step * dt + across * s, centre_x + step * dx + across * c

    return Sweep(out_width, out_frames, at, normal, loop, sides)


def _shear(angle):
    """The delay per pixel across a sheared frame, and its plane's (t, x)
    unit normal."""
    k = round(math.tan(math.radians(angle)), 12)
    return k, (1 / math.hypot(1, k), -k / math.hypot(1, k))


def _shear_room(n_frames, width, angle, noise=None):
    """How many frames of time a sheared inside frame can move through, and
    how far its ends can move sideways. Negative if it can't fit at all."""
    k, normal = _shear(angle)
    push_t, push_x = _push(noise, normal)
    # A sheared frame always spans the video's full width, so any sideways
    # push takes its ends out of the video.
    return n_frames - 1 - abs(k) * (width - 1) - 2 * abs(push_t), -abs(push_x)


def shear_sweep(n_frames, width, angle, inside=False, noise=None, loop=False,
                sides=False):
    """Plan a sweep where output column x is input column x, delayed in time
    by tan(angle) frames per pixel across the frame.

    The frame is always as wide as the input and moves forward one frame of
    time per output frame. Without `inside`, it runs from where the slice
    first touches the video to where it leaves, black where it's outside.
    With `inside`, only positions entirely inside the video are kept; raises
    DoesNotFit if the clip is too short for the angle. With `loop`, time runs
    round in a ring, the clip's first frame following its last, and the frame
    goes once round it, so the last output frame leads back into the first.
    With `sides` as well, x wraps round too, the picture's left edge following
    its right.

    With `noise`, the sweep allows for the noise pushing points off the plane.
    Perpendicular noise pushes the frame's ends sideways, so with `inside` or
    `loop` it only fits at 0 degrees, unless the sides wrap.
    """
    if sides and not loop:
        raise ValueError("the sides only wrap round on a loop")
    if not -90 < angle < 90:
        raise ValueError("shear needs an angle between -90 and 90 degrees "
                         "(at 90 the delay would be infinite)")
    k, normal = _shear(angle)
    push_t, push_x = _push(noise, normal)
    x = np.arange(width, dtype=np.float64)
    delay = (x - (width - 1) / 2) * k
    span_t = abs(k) * (width - 1)  # frames of time one output frame covers

    if inside or loop:
        room_t, room_x = _shear_room(n_frames, width, angle, noise)
        if room_x < 0 and not sides:
            raise DoesNotFit(
                f"At {angle:g} degrees perpendicular noise can push the ends of "
                f"a sheared frame {abs(push_x):.3g} pixels sideways, out of the "
                "video, and a sheared frame always spans the video's whole "
                "width. Push the noise through time instead.")
        if room_t < 0 and not loop:
            raise DoesNotFit(
                _too_long(angle, width, span_t, push_t, n_frames, "sheared ")
                + " " + _angles_that_fit(
                    lambda a: min(_shear_room(n_frames, width, a, noise)) >= 0, 89.9))
        out_frames = n_frames if loop else int(room_t + 1e-9) + 1
    else:
        out_frames = int(n_frames - 1 + span_t + 2 * abs(push_t) + 1e-9) + 1

    centre_t = (n_frames - 1) / 2

    def at(f):
        return centre_t + (f - (out_frames - 1) / 2) + delay, x

    return Sweep(width, out_frames, at, normal, loop, sides)


def plan_sweep(n_frames, width, slice="rotate", angle=45.0, inside=False,
               motion=None, noise=None, loop=False, sides=False):
    """Plan a sweep of either kind. motion only applies to rotate, with
    inside or loop."""
    if slice == "shear":
        return shear_sweep(n_frames, width, angle, inside, noise, loop, sides)
    if slice == "rotate":
        return rotation_sweep(n_frames, width, angle, inside, motion, noise, loop,
                              sides)
    raise ValueError(f"unknown slice {slice!r}")


def _looping(noise, sweep):
    """The noise as the sweep uses it: a loop's comes back round with it."""
    return noise._replace(period=sweep.frames) if sweep.loop else noise


def surface(sweep, f, height, noise):
    """The (t, y, x) source position of every pixel of output frame f, with
    the frame pushed off its plane by the noise, as arrays that broadcast to
    (height, sweep.width) for sample(), with wrap if the sweep loops and
    wrap_x if its sides do too. Use the noise the sweep was planned with."""
    noise = _looping(noise, sweep)
    t, x = sweep.at(f)
    bump = noise.field(sweep.width, height, f)
    push_t, push_x = noise.push(sweep.normal)
    return t + push_t * bump, np.arange(height)[:, None], x + push_x * bump


def slice_frame(volume, sweep, f, noise=None, nearest=False):
    """Output frame f of a sweep through a (T, H, W, C) volume, pushed off its
    plane by the noise the sweep was planned with, if any."""
    t, x = sweep.at(f)
    if noise is None:
        return sample_columns(volume, t, x, nearest, sweep.loop, sweep.sides)
    noise = _looping(noise, sweep)
    return sample_noisy_columns(volume, t, x, noise.push(sweep.normal), noise, f,
                                nearest, sweep.loop, sweep.sides)


def main():
    parser = argparse.ArgumentParser(
        description="Re-slice a video's x-y-t cuboid with a plane tilted "
                    "about the y axis.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--angle", type=float, default=45.0,
                        help="tilt of the slicing plane about the y axis, in "
                             "degrees (default 45; 0 reproduces the input)")
    parser.add_argument("--slice", choices=["rotate", "shear"], default="rotate",
                        help="rotate: sample evenly along the tilted plane "
                             "(default). shear: keep every input column, "
                             "delayed by tan(angle) frames per pixel across")
    parser.add_argument("--inside", action="store_true",
                        help="keep each frame the input's width and entirely "
                             "inside the video, so there are no black edges")
    parser.add_argument("--motion", choices=["perpendicular", "time", "longest"],
                        help="with --slice rotate --inside: the straight line "
                             "the frame moves along (default longest). With "
                             "--loop-sides: time (default) or perpendicular")
    parser.add_argument("--loop", action="store_true",
                        help="make a seamless loop: time runs round in a ring, "
                             "the first frame following the last, and a frame "
                             "the input's width goes once round it")
    parser.add_argument("--loop-fade", type=float, metavar="S",
                        help="with --loop: blend the clip's last S seconds into "
                             "its first, so the join from its end back to its "
                             "start doesn't show (default 0)")
    parser.add_argument("--loop-sides", action="store_true",
                        help="with --loop: let the picture's sides wrap round "
                             "too, its left edge following its right, so the "
                             "frame can move sideways (--motion perpendicular)")
    parser.add_argument("--noise", type=float, default=0.0, metavar="A",
                        help="push each point of the slicing surface up to A "
                             "frames off the plane with Perlin noise (default "
                             "0: a flat plane)")
    parser.add_argument("--noise-size", type=float, metavar="PX",
                        help="roughly how far apart the bumps are, in pixels "
                             "(default 64)")
    parser.add_argument("--noise-speed", type=float, metavar="PX",
                        help="how fast the bumps change, in pixels per output "
                             "frame (default 1; 0 keeps one fixed bumpy surface)")
    parser.add_argument("--noise-direction", choices=["time", "perpendicular"],
                        help="push points through time (default), or "
                             "perpendicular to the plane")
    parser.add_argument("--noise-seed", type=int, metavar="N",
                        help="which noise pattern to use (default 0)")
    parser.add_argument("--preview", action="store_true",
                        help="quick rough render: half size in x, y and time, "
                             "no blending, fast low-quality encoding")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="resize each input frame by this factor before "
                             "slicing (e.g. 0.5); keeps every frame")
    parser.add_argument("--start", type=float,
                        help="start this many seconds into the input")
    parser.add_argument("--duration", type=float,
                        help="use only this many seconds of the input")
    parser.add_argument("--fps",
                        help="output frame rate (default: same as input)")
    args = parser.parse_args()

    if args.motion and args.slice == "shear":
        parser.error("--motion doesn't apply to --slice shear: a sheared frame "
                     "always uses every input column, so it can only move "
                     "through time")
    if args.loop and args.inside:
        parser.error("--inside doesn't apply to --loop: a looping frame always "
                     "stays inside the video")
    if args.loop_sides and not args.loop:
        parser.error("--loop-sides needs --loop")
    if args.loop and args.motion and not args.loop_sides:
        parser.error("--motion needs --loop-sides on a loop: unless the sides "
                     "wrap round, a looping frame can only move straight "
                     "through time")
    if args.loop_sides and args.motion == "longest":
        parser.error("--motion longest doesn't apply to --loop-sides: with the "
                     "sides wrapping round, no straight line is the longest")
    if args.loop_fade is not None and not args.loop:
        parser.error("--loop-fade needs --loop")
    if (args.loop_fade or 0) < 0:
        parser.error("--loop-fade can't be negative")
    if args.motion and not (args.inside or args.loop):
        parser.error("--motion needs --inside: without it the frame already "
                     "holds the plane's whole cut through the video, so moving "
                     "it another way would only change the speed")
    noise_options = {"size": args.noise_size, "speed": args.noise_speed,
                     "direction": args.noise_direction, "seed": args.noise_seed}
    noise_options = {k: v for k, v in noise_options.items() if v is not None}
    if args.noise < 0:
        parser.error("--noise can't be negative")
    if noise_options and not args.noise:
        parser.error("--noise-size, --noise-speed, --noise-direction and "
                     "--noise-seed need --noise")
    if noise_options.get("size", 1) <= 0:
        parser.error("--noise-size must be more than 0")
    if noise_options.get("seed", 0) < 0:
        parser.error("--noise-seed can't be negative")

    began = time.monotonic()
    shrink = PREVIEW_SCALE if args.preview else 1.0
    noise = Noise(args.noise, **noise_options).scaled(shrink) if args.noise else None
    try:
        volume, fps = load_video(args.input, args.scale * shrink, shrink,
                                 args.start, args.duration, fast=args.preview)
    except VideoError as err:
        sys.exit(str(err))
    n_frames, height, width, _ = volume.shape
    print(f"Loaded {n_frames} frames of {width}x{height} "
          f"({volume.nbytes / 1e9:.2f} GB in memory)")
    if args.loop_fade:
        rate = float(fps) * shrink  # of the frames loaded
        fade = round(args.loop_fade * rate)
        if 2 * fade > n_frames:
            sys.exit(f"--loop-fade can be at most half the clip, "
                     f"{n_frames / 2 / rate:.3g} s.")
        volume = crossfade(volume, fade)
        n_frames = len(volume)

    try:
        sweep = plan_sweep(n_frames, width, args.slice, args.angle, args.inside,
                           args.motion, noise, args.loop, args.loop_sides)
    except DoesNotFit as err:
        if args.loop:  # only sideways noise can leave a loop's frame
            sys.exit(f"{err}\nUse less --noise, --noise-direction time, or "
                     f"--loop-sides.")
        smaller = "a smaller angle or less --noise" if noise else "a smaller angle"
        sys.exit(f"{err}\nUse a longer clip or {smaller}, or drop --inside "
                 f"to sweep the whole plane with black edges.")
    except ValueError as err:
        sys.exit(str(err))
    out_width, out_frames = sweep.width, sweep.frames

    # A preview has fewer frames, so play it slower to keep the running time
    # of the full render.
    out_fps = (Fraction(args.fps) if args.fps else fps) * Fraction(shrink)
    print(f"Writing {out_frames} frames of {out_width}x{height} to {args.output}")

    if args.preview:
        writer = open_writer(args.output, out_width, height, out_fps,
                             preset="ultrafast")
    else:
        writer = open_writer(args.output, out_width, height, out_fps)
    for i in range(out_frames):
        frame = slice_frame(volume, sweep, i, noise, nearest=args.preview)
        writer.stdin.write(frame.tobytes())
        if i % 10 == 0 or i == out_frames - 1:
            print(f"\r  frame {i + 1}/{out_frames}", end="", flush=True)
    print()
    writer.stdin.close()
    if writer.wait() != 0:
        sys.exit("ffmpeg failed to encode the output")
    print(f"Made {float(out_frames / out_fps):.1f} s of video in "
          f"{time.monotonic() - began:.1f} s")


if __name__ == "__main__":
    main()
