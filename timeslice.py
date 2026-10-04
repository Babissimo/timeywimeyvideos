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

Units: one frame of time is treated as one pixel of distance, so the cuboid is
width x height x frame-count voxels.

    python timeslice.py input.mp4 output.mp4 --angle 30
"""

import argparse
import json
import math
import subprocess
import sys
import time
from fractions import Fraction

import numpy as np
from numba import njit, prange

PREVIEW_SCALE = 0.5  # --preview shrinks x, y and t by this factor


class DoesNotFit(ValueError):
    """The frame can't fit inside the video at the requested angle."""


def probe(path):
    """Return (width, height, frame rate, duration in seconds or None)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:stream_tags=rotate"
                          ":stream_side_data=rotation:format=duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    info = json.loads(out)
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
    seconds = None if length is None else length - (start or 0)
    if duration is not None:
        seconds = duration if seconds is None else min(seconds, duration)
    capacity = 256 if seconds is None else int(max(seconds, 0) * rate * 1.05) + 2
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
        sys.exit(f"ffmpeg failed to decode {path}")
    if n == 0:
        sys.exit(f"no frames decoded from {path}")
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
def _sample_points(volume, t, y, x, out):
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


def sample(volume, t, y, x):
    """Sample a (T, H, W, C) volume at any continuous (t, y, x) points.

    t, y and x broadcast together to the (height, width) of the output frame,
    so any surface through the cuboid can be read this way.
    """
    shape = np.broadcast_shapes(np.shape(t), np.shape(y), np.shape(x))
    t, y, x = (np.broadcast_to(np.asarray(a, np.float64), shape) for a in (t, y, x))
    out = np.empty(shape + volume.shape[3:], np.uint8)
    _sample_points(volume, t, y, x, out)
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


# Slice planning. Each function returns (out_width, out_frames, slices), where
# slices yields, for each output frame, the (t, x) source position of every
# output column, for use with sample_columns.

def rotation_sweep(n_frames, width, angle, inside=False, motion="longest"):
    """Plan a sweep with the slicing plane rotated `angle` degrees about y.

    Without `inside`, the frame is wide enough to hold the plane's whole cut
    through the cuboid, and the plane moves perpendicular to itself from where
    it first touches the cuboid to where it leaves.

    With `inside`, the frame is as wide as the input and stays entirely inside
    the cuboid, moving along a straight line through the cuboid's centre:
    "perpendicular" to itself, straight through "time", or along the "longest"
    line that fits. Raises DoesNotFit if the clip is too short for the angle.
    """
    theta = math.radians(angle)
    # Round so that 0, 90, 180... degrees land exactly on the voxel grid.
    c, s = round(math.cos(theta), 12), round(math.sin(theta), 12)

    if not inside:
        out_width = max(1, round(width * abs(c) + n_frames * abs(s)))
        out_frames = max(1, round(width * abs(s) + n_frames * abs(c)))
        dx, dt = -s, c
    else:
        out_width = width
        # How far the frame's centre can move from the cuboid's centre, in x
        # and in t, before the frame's ends leave the video.
        span_t = (width - 1) * abs(s)  # frames of time one output frame covers
        room_x = (width - 1) * (1 - abs(c)) / 2
        room_t = (n_frames - 1 - span_t) / 2
        if room_t < 0:
            steepest = math.degrees(math.asin((n_frames - 1) / (width - 1)))
            raise DoesNotFit(
                f"At {angle:g} degrees a {width}-pixel-wide frame covers "
                f"{math.ceil(span_t) + 1} frames of time, but the clip has only "
                f"{n_frames}. Angles up to {steepest:.1f} degrees fit this clip.")

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

    def frames():
        for f in range(out_frames):
            step = f - (out_frames - 1) / 2  # distance moved from the centre
            x = centre_x + step * dx + across * c
            t = centre_t + step * dt + across * s
            yield t, x

    return out_width, out_frames, frames()


def shear_sweep(n_frames, width, angle, inside=False):
    """Plan a sweep where output column x is input column x, delayed in time
    by tan(angle) frames per pixel across the frame.

    The frame is always as wide as the input and moves forward one frame of
    time per output frame. Without `inside`, it runs from where the slice
    first touches the video to where it leaves, black where it's outside.
    With `inside`, only positions entirely inside the video are kept; raises
    DoesNotFit if the clip is too short for the angle.
    """
    if not -90 < angle < 90:
        raise ValueError("shear needs an angle between -90 and 90 degrees "
                         "(at 90 the delay would be infinite)")
    k = round(math.tan(math.radians(angle)), 12)
    x = np.arange(width, dtype=np.float64)
    delay = (x - (width - 1) / 2) * k
    span_t = abs(k) * (width - 1)  # frames of time one output frame covers

    if inside:
        if span_t > n_frames - 1:
            steepest = math.degrees(math.atan((n_frames - 1) / (width - 1)))
            raise DoesNotFit(
                f"At {angle:g} degrees a {width}-pixel-wide sheared frame covers "
                f"{math.ceil(span_t) + 1} frames of time, but the clip has only "
                f"{n_frames}. Angles up to {steepest:.1f} degrees fit this clip.")
        out_frames = int(n_frames - 1 - span_t + 1e-9) + 1
    else:
        out_frames = int(n_frames - 1 + span_t + 1e-9) + 1

    centre_t = (n_frames - 1) / 2

    def frames():
        for f in range(out_frames):
            yield centre_t + (f - (out_frames - 1) / 2) + delay, x

    return width, out_frames, frames()


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

    began = time.monotonic()
    shrink = PREVIEW_SCALE if args.preview else 1.0
    volume, fps = load_video(args.input, args.scale * shrink, shrink,
                             args.start, args.duration, fast=args.preview)
    n_frames, height, width, _ = volume.shape
    print(f"Loaded {n_frames} frames of {width}x{height} "
          f"({volume.nbytes / 1e9:.2f} GB in memory)")

    try:
        if args.slice == "shear":
            plan = shear_sweep(n_frames, width, args.angle, args.inside)
        else:
            plan = rotation_sweep(n_frames, width, args.angle, args.inside,
                                  args.motion or "longest")
    except DoesNotFit as err:
        sys.exit(f"{err}\nUse a longer clip or a smaller angle, or drop --inside "
                 f"to sweep the whole plane with black edges.")
    except ValueError as err:
        sys.exit(str(err))
    out_width, out_frames, slices = plan

    # A preview has fewer frames, so play it slower to keep the running time
    # of the full render.
    out_fps = (Fraction(args.fps) if args.fps else fps) * Fraction(shrink)
    print(f"Writing {out_frames} frames of {out_width}x{height} to {args.output}")

    if args.preview:
        writer = open_writer(args.output, out_width, height, out_fps,
                             preset="ultrafast")
    else:
        writer = open_writer(args.output, out_width, height, out_fps)
    for i, (t, x) in enumerate(slices):
        frame = sample_columns(volume, t, x, nearest=args.preview)
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
