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
# to it (linear interpolation). Points outside the cuboid come out black. The
# loops are compiled by numba and spread across all CPU cores.

@njit(cache=True)
def _neighbours(coord, n):
    """The voxel indices either side of coord along an axis of length n, and
    how far coord is past the first (0 = on it, 0.999 = nearly on the second)."""
    lo = int(np.floor(coord))
    return max(lo, 0), min(lo + 1, n - 1), coord - lo


@njit(parallel=True, cache=True)
def _sample_points(volume, t, y, x, nearest, out):
    n_t, n_y, n_x, n_c = volume.shape
    for i in prange(out.shape[0]):
        for j in range(out.shape[1]):
            tt, yy, xx = t[i, j], y[i, j], x[i, j]
            if not (-0.5 <= tt < n_t - 0.5 and -0.5 <= yy < n_y - 0.5
                    and -0.5 <= xx < n_x - 0.5):
                out[i, j, :] = 0
                continue
            t0, t1, ft = _neighbours(tt, n_t)
            y0, y1, fy = _neighbours(yy, n_y)
            x0, x1, fx = _neighbours(xx, n_x)
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


@njit(parallel=True, cache=True)
def _sample_columns(volume, t, x, nearest, out):
    n_t, n_y, n_x, n_c = volume.shape
    for row in prange(n_y):
        for j in range(t.shape[0]):
            tt, xx = t[j], x[j]
            if not (-0.5 <= tt < n_t - 0.5 and -0.5 <= xx < n_x - 0.5):
                out[row, j, :] = 0
                continue
            t0, t1, ft = _neighbours(tt, n_t)
            x0, x1, fx = _neighbours(xx, n_x)
            if nearest:
                ti = t1 if ft >= 0.5 else t0
                xi = x1 if fx >= 0.5 else x0
                out[row, j, :] = volume[ti, row, xi, :]
                continue
            for c in range(n_c):
                # Blend along x, then t: 4 voxels in all.
                v = ((volume[t0, row, x0, c] * (1 - fx) + volume[t0, row, x1, c] * fx) * (1 - ft)
                     + (volume[t1, row, x0, c] * (1 - fx) + volume[t1, row, x1, c] * fx) * ft)
                out[row, j, c] = min(255, int(v + 0.5))


def sample(volume, t, y, x, nearest=False):
    """Sample a (T, H, W, C) volume at any continuous (t, y, x) points.

    t, y and x broadcast together to the (height, width) of the output frame,
    so any surface through the cuboid can be read this way. With nearest,
    each pixel is copied from the closest voxel instead of blended.
    """
    shape = np.broadcast_shapes(np.shape(t), np.shape(y), np.shape(x))
    t, y, x = (np.broadcast_to(np.asarray(a, np.float64), shape) for a in (t, y, x))
    out = np.empty(shape + volume.shape[3:], np.uint8)
    _sample_points(volume, t, y, x, nearest, out)
    return out


def sample_columns(volume, t, x, nearest=False):
    """A faster sample() for slices built from whole source columns.

    Output column j is the full-height column of the input at time t[j] and
    position x[j], so only x and t need blending (4 voxels, not 8). Slices that
    only tilt about the y axis have this shape. With nearest, each pixel is
    copied from the closest voxel instead of blended.
    """
    out = np.empty((volume.shape[1], len(t)) + volume.shape[3:], np.uint8)
    _sample_columns(volume, np.asarray(t, np.float64), np.asarray(x, np.float64),
                    nearest, out)
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
def _noise3(perm, x, y, z):
    fx, fy, fz = np.floor(x), np.floor(y), np.floor(z)
    X, Y, Z = int(fx) & 255, int(fy) & 255, int(fz) & 255
    x, y, z = x - fx, y - fy, z - fz
    u, v, w = _fade(x), _fade(y), _fade(z)
    # Hash the 8 corners of the lattice cell around the point.
    a, b = perm[X] + Y, perm[X + 1] + Y
    aa, ab, ba, bb = perm[a] + Z, perm[a + 1] + Z, perm[b] + Z, perm[b + 1] + Z
    return _lerp(w, _lerp(v, _lerp(u, _grad(perm[aa], x, y, z),
                                      _grad(perm[ba], x - 1, y, z)),
                             _lerp(u, _grad(perm[ab], x, y - 1, z),
                                      _grad(perm[bb], x - 1, y - 1, z))),
                    _lerp(v, _lerp(u, _grad(perm[aa + 1], x, y, z - 1),
                                      _grad(perm[ba + 1], x - 1, y, z - 1)),
                             _lerp(u, _grad(perm[ab + 1], x, y - 1, z - 1),
                                      _grad(perm[bb + 1], x - 1, y - 1, z - 1))))


@njit(parallel=True, cache=True)
def _noise_grid(perm, xs, ys, z, out):
    for i in prange(len(ys)):
        for j in range(len(xs)):
            out[i, j] = _noise3(perm, xs[j], ys[i], z)


@functools.lru_cache(maxsize=16)
def _permutation(seed):
    """The shuffled lattice hash for a seed, written out twice so that
    indexing past 255 wraps around."""
    p = np.random.default_rng(seed).permutation(256)
    return np.concatenate([p, p]).astype(np.int64)


class Noise(NamedTuple):
    """Perlin noise that pushes each point of a sweep's frame off its plane.

    Points move up to `amplitude` frames (one frame = one pixel), either
    through "time" or "perpendicular" to the plane, in the x-t plane. `size` is
    roughly how far apart the bumps are, in pixels. The bumps change as the
    sweep goes on, `speed` pixels' worth per output frame; 0 keeps one fixed
    bumpy surface. `seed` picks the pattern.
    """
    amplitude: float
    size: float = 64.0
    speed: float = 1.0
    direction: str = "time"
    seed: int = 0

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
                             size=self.size * factor)

    def field(self, width, height, f):
        """The noise over output frame f, from -1 to 1, shape (height, width)."""
        out = np.empty((height, width))
        _noise_grid(_permutation(self.seed), np.arange(width) / self.size,
                    np.arange(height) / self.size, f * self.speed / self.size, out)
        return out / NOISE_BOUND


# Slice planning. Each function returns a Sweep: the output frame width, the
# number of output frames, at(f), which gives the (t, x) source position of
# every column of output frame f, for use with sample_columns, and the (t, x)
# normal of the frame's plane, which perpendicular noise pushes along.
#
# Given noise, a plan allows for it pushing points as far as it can: the
# whole-plane sweep starts and ends early and late enough to catch the bumps,
# and --inside frames keep far enough from the edges that none leave the video.

class Sweep(NamedTuple):
    width: int
    frames: int
    at: Callable[[int], tuple]
    normal: tuple


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


def rotation_sweep(n_frames, width, angle, inside=False, motion="longest",
                   noise=None):
    """Plan a sweep with the slicing plane rotated `angle` degrees about y.

    Without `inside`, the frame is wide enough to hold the plane's whole cut
    through the cuboid, and the plane moves perpendicular to itself from where
    it first touches the cuboid to where it leaves.

    With `inside`, the frame is as wide as the input and stays entirely inside
    the cuboid, moving along a straight line through the cuboid's centre:
    "perpendicular" to itself, straight through "time", or along the "longest"
    line that fits. Raises DoesNotFit if the clip is too short for the angle.

    With `noise`, the sweep allows for the noise pushing points off the plane.
    """
    c, s = _rotation(angle)
    normal = (c, -s)  # (t, x): the way the whole plane sweeps
    push_t, push_x = _push(noise, normal)

    if not inside:
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
                why = (f"At {angle:g} degrees the noise can push the ends of the "
                       f"frame {abs(push_x):.3g} pixels sideways, out of the video.")
            raise DoesNotFit(why + " " + _angles_that_fit(
                lambda a: min(_rotation_room(n_frames, width, a, noise)) >= 0, 90))

        # Move forward in time unless the plane is turned past 90 degrees,
        # where the sweep runs backwards (180 degrees plays the clip in reverse).
        sign_t = 1 if c >= 0 else -1
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

    return Sweep(out_width, out_frames, at, normal)


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


def shear_sweep(n_frames, width, angle, inside=False, noise=None):
    """Plan a sweep where output column x is input column x, delayed in time
    by tan(angle) frames per pixel across the frame.

    The frame is always as wide as the input and moves forward one frame of
    time per output frame. Without `inside`, it runs from where the slice
    first touches the video to where it leaves, black where it's outside.
    With `inside`, only positions entirely inside the video are kept; raises
    DoesNotFit if the clip is too short for the angle.

    With `noise`, the sweep allows for the noise pushing points off the plane.
    Perpendicular noise pushes the frame's ends sideways, so with `inside` it
    only fits at 0 degrees.
    """
    if not -90 < angle < 90:
        raise ValueError("shear needs an angle between -90 and 90 degrees "
                         "(at 90 the delay would be infinite)")
    k, normal = _shear(angle)
    push_t, push_x = _push(noise, normal)
    x = np.arange(width, dtype=np.float64)
    delay = (x - (width - 1) / 2) * k
    span_t = abs(k) * (width - 1)  # frames of time one output frame covers

    if inside:
        room_t, room_x = _shear_room(n_frames, width, angle, noise)
        if room_x < 0:
            raise DoesNotFit(
                f"At {angle:g} degrees perpendicular noise can push the ends of "
                f"a sheared frame {abs(push_x):.3g} pixels sideways, out of the "
                "video, and a sheared frame always spans the video's whole "
                "width. Push the noise through time instead.")
        if room_t < 0:
            raise DoesNotFit(
                _too_long(angle, width, span_t, push_t, n_frames, "sheared ")
                + " " + _angles_that_fit(
                    lambda a: min(_shear_room(n_frames, width, a, noise)) >= 0, 89.9))
        out_frames = int(room_t + 1e-9) + 1
    else:
        out_frames = int(n_frames - 1 + span_t + 2 * abs(push_t) + 1e-9) + 1

    centre_t = (n_frames - 1) / 2

    def at(f):
        return centre_t + (f - (out_frames - 1) / 2) + delay, x

    return Sweep(width, out_frames, at, normal)


def plan_sweep(n_frames, width, slice="rotate", angle=45.0, inside=False,
               motion=None, noise=None):
    """Plan a sweep of either kind. motion only applies to rotate with
    inside, and defaults to "longest" there."""
    if slice == "shear":
        return shear_sweep(n_frames, width, angle, inside, noise)
    if slice == "rotate":
        return rotation_sweep(n_frames, width, angle, inside, motion or "longest",
                              noise)
    raise ValueError(f"unknown slice {slice!r}")


def surface(sweep, f, height, noise):
    """The (t, y, x) source position of every pixel of output frame f, with
    the frame pushed off its plane by the noise, as arrays that broadcast to
    (height, sweep.width) for sample(). Use the noise the sweep was planned
    with."""
    t, x = sweep.at(f)
    bump = noise.field(sweep.width, height, f)
    push_t, push_x = noise.push(sweep.normal)
    return t + push_t * bump, np.arange(height)[:, None], x + push_x * bump


def slice_frame(volume, sweep, f, noise=None, nearest=False):
    """Output frame f of a sweep through a (T, H, W, C) volume, pushed off its
    plane by the noise the sweep was planned with, if any."""
    if noise is None:
        return sample_columns(volume, *sweep.at(f), nearest)
    return sample(volume, *surface(sweep, f, volume.shape[1], noise), nearest)


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
                             "the frame moves along (default longest)")
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
    if args.motion and not args.inside:
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

    try:
        sweep = plan_sweep(n_frames, width, args.slice, args.angle, args.inside,
                           args.motion, noise)
    except DoesNotFit as err:
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
