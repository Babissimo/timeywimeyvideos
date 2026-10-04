#!/usr/bin/env python3
"""Slice through a video's space-time cuboid at an angle.

A video is a block of voxels indexed (t, y, x). Playing it normally means
sweeping a plane of constant t through that block, front to back. This script
tilts the plane by an angle about the y axis, so each output frame shows
different moments in time across its width, and sweeps it through the block
along its normal.

Picture the x-t plane as a space-time diagram, x to the right and t up. A
normal frame is a horizontal line; a positive angle rotates that line
anticlockwise, so the right edge of each output frame is later in time than the
left edge. At 90 degrees each output frame is a y-t slice (time runs left to
right) and the sweep moves from right to left across the original picture.

Units: one frame of time is treated as one pixel of distance, so the cuboid is
width x height x frame-count voxels. Parts of the tilted plane that fall
outside the cuboid come out black.

    python timeslice.py input.mp4 output.mp4 --angle 30
"""

import argparse
import json
import subprocess
import sys

import numpy as np
from numba import njit, prange


def probe(path):
    """Return (width, height, frame rate string) of the first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate:stream_tags=rotate"
                          ":stream_side_data=rotation",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    width, height = stream["width"], stream["height"]

    # Phone videos are often stored sideways with a rotation flag; ffmpeg
    # applies it when decoding, so the decoded frames have swapped dimensions.
    rotation = stream.get("tags", {}).get("rotate", 0)
    for side_data in stream.get("side_data_list", []):
        rotation = side_data.get("rotation", rotation)
    if abs(int(float(rotation))) % 180 == 90:
        width, height = height, width

    return width, height, stream["r_frame_rate"]


def load_video(path, scale=1.0, start=None, duration=None):
    """Decode a video into a uint8 array of shape (frames, height, width, 3)."""
    width, height, fps = probe(path)
    width, height = max(1, round(width * scale)), max(1, round(height * scale))

    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-i", path, "-vf", f"scale={width}:{height}",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    frame_bytes = width * height * 3
    frames = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE) as proc:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frames.append(np.frombuffer(buf, np.uint8).reshape(height, width, 3))
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed to decode {path}")
    if not frames:
        sys.exit(f"no frames decoded from {path}")
    return np.stack(frames), fps


def open_writer(path, width, height, fps):
    """Start an ffmpeg process that encodes raw RGB frames written to its stdin."""
    cmd = ["ffmpeg", "-v", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", fps, "-i", "-",
           # H.264 needs even dimensions; pad with a black row/column if not.
           "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", path]
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
def _sample_columns(volume, t, x, out):
    n_t, n_y, n_x, n_c = volume.shape
    for row in prange(n_y):
        for j in range(t.shape[0]):
            tt, xx = t[j], x[j]
            if not (-0.5 <= tt < n_t - 0.5 and -0.5 <= xx < n_x - 0.5):
                out[row, j, :] = 0
                continue
            t0, t1, ft = _neighbours(tt, n_t)
            x0, x1, fx = _neighbours(xx, n_x)
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


def sample_columns(volume, t, x):
    """A faster sample() for slices built from whole source columns.

    Output column j is the full-height column of the input at time t[j] and
    position x[j], so only x and t need blending (4 voxels, not 8). Slices that
    only rotate about the y axis have this shape.
    """
    out = np.empty((volume.shape[1], len(t)) + volume.shape[3:], np.uint8)
    _sample_columns(volume, np.asarray(t, np.float64), np.asarray(x, np.float64), out)
    return out


def y_rotation_slices(n_frames, width, angle):
    """Plan a sweep with the slicing plane rotated `angle` degrees about y.

    Returns (out_width, out_frames, slices), where slices yields, for each
    output frame, the (t, x) source position of every output column (for use
    with sample_columns). The output frame is wide enough to hold the plane's
    whole intersection with the cuboid, and the sweep runs from where the plane
    first touches the cuboid to where it leaves.
    """
    theta = np.radians(angle)
    # Round so that 0, 90, 180... degrees land exactly on the voxel grid.
    c, s = round(np.cos(theta), 12), round(np.sin(theta), 12)

    out_width = max(1, round(width * abs(c) + n_frames * abs(s)))
    out_frames = max(1, round(width * abs(s) + n_frames * abs(c)))

    centre_x, centre_t = (width - 1) / 2, (n_frames - 1) / 2
    across = np.arange(out_width) - (out_width - 1) / 2  # position along the frame

    def frames():
        for f in range(out_frames):
            depth = f - (out_frames - 1) / 2  # plane's distance from the centre
            x = centre_x + across * c - depth * s
            t = centre_t + across * s + depth * c
            yield t, x

    return out_width, out_frames, frames()


def main():
    parser = argparse.ArgumentParser(
        description="Re-slice a video's x-y-t cuboid with a plane rotated "
                    "about the y axis.")
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--angle", type=float, default=45.0,
                        help="rotation of the slicing plane about the y axis, "
                             "in degrees (default 45; 0 reproduces the input)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="resize the input by this factor before slicing, "
                             "to save memory and time (e.g. 0.5)")
    parser.add_argument("--start", type=float,
                        help="start this many seconds into the input")
    parser.add_argument("--duration", type=float,
                        help="use only this many seconds of the input")
    parser.add_argument("--fps",
                        help="output frame rate (default: same as input)")
    args = parser.parse_args()

    volume, fps = load_video(args.input, args.scale, args.start, args.duration)
    n_frames, height, width, _ = volume.shape
    print(f"Loaded {n_frames} frames of {width}x{height} "
          f"({volume.nbytes / 1e9:.2f} GB in memory)")

    out_width, out_frames, slices = y_rotation_slices(
        n_frames, width, args.angle)
    print(f"Writing {out_frames} frames of {out_width}x{height} to {args.output}")

    writer = open_writer(args.output, out_width, height, args.fps or fps)
    for i, (t, x) in enumerate(slices):
        writer.stdin.write(sample_columns(volume, t, x).tobytes())
        if i % 10 == 0 or i == out_frames - 1:
            print(f"\r  frame {i + 1}/{out_frames}", end="", flush=True)
    print()
    writer.stdin.close()
    if writer.wait() != 0:
        sys.exit("ffmpeg failed to encode the output")


if __name__ == "__main__":
    main()
