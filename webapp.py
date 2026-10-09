#!/usr/bin/env python3
"""A local web front end for timeslice.py.

    npm install && npm run build    # once, and after changing the page
    python webapp.py --videos ~/Movies

then open http://127.0.0.1:8000. Pick a video from that folder or upload one,
drag the angle and watch the slice play live, then render it.

The page (web/, built into web/dist/) plays the live view itself: it decodes a
half-size copy of the clip in the browser and slices it on the GPU, the same
way `timeslice.py --preview` does, so what it shows is what a preview render
makes. This server offers the videos and takes uploads, saved in uploads/.
Renders run timeslice.py itself in a separate process, and land in renders/.
"""

import argparse
import functools
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path
from urllib.parse import unquote

from flask import Flask, Response, abort, jsonify, request, send_from_directory
from werkzeug.utils import secure_filename

import timeslice

HERE = Path(__file__).resolve().parent
VIDEO_TYPES = {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".gif"}
PAGE = HERE / "web" / "dist"  # the page, as `npm run build` makes it
NOT_BUILT = ("The page isn't built yet. Run `npm install` and then `npm run build` beside "
             "webapp.py, and reload. While working on the page, `npm run dev` serves it "
             "instead.\n")

app = Flask(__name__, static_folder=None)
folders = {  # where sources come from and renders go; main() can change these
    "videos": HERE / "videos",
    "uploads": HERE / "uploads",
    "renders": HERE / "renders",
}


class Problem(Exception):
    """Something the page should show the user, sent as a JSON error."""

    def __init__(self, message, status=400, kind="error"):
        super().__init__(message)
        self.status, self.kind = status, kind


@app.errorhandler(Problem)
def problem(err):
    return jsonify(error=str(err), kind=err.kind), err.status


# Sources: videos in the folder named at startup, and uploaded ones. The page
# names them "videos/<file>" or "uploads/<file>".

def list_videos(folder):
    if not folder.is_dir():
        return []
    return sorted(p.name for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in VIDEO_TYPES)


def source_path(source):
    """The file a source name refers to; refuses anything outside the folders."""
    kind, _, name = (source or "").partition("/")
    if kind not in ("videos", "uploads") or not name or "/" in name \
            or name.startswith("."):
        raise Problem("Pick a video first.", 404)
    path = folders[kind] / name
    if not path.is_file():
        raise Problem(f"{name} isn't in {kind} any more.", 404)
    return path


@functools.lru_cache(maxsize=64)
def _probe(path, mtime):
    return timeslice.probe(path)


def probe(path):
    try:
        return _probe(str(path), path.stat().st_mtime)
    except timeslice.VideoError as err:
        raise Problem(str(err), 422)


@app.get("/api/sources")
def sources():
    return jsonify(videos=list_videos(folders["videos"]),
                   uploads=list_videos(folders["uploads"]),
                   folder=str(folders["videos"]))


@app.post("/api/upload")
def upload():
    name = secure_filename(unquote(request.headers.get("X-Filename", "")))
    if Path(name).suffix.lower() not in VIDEO_TYPES:
        raise Problem("That doesn't look like a video file "
                      f"(expected one of {', '.join(sorted(VIDEO_TYPES))}).")
    folders["uploads"].mkdir(parents=True, exist_ok=True)
    path = folders["uploads"] / name
    partial = path.with_name(f".{name}.part")  # hidden until complete
    with open(partial, "wb") as out:
        while chunk := request.stream.read(1 << 20):
            out.write(chunk)
    partial.replace(path)
    return jsonify(source=f"uploads/{name}")


@app.get("/api/info")
def info():
    path = source_path(request.args.get("source"))
    width, height, fps, length = probe(path)
    return jsonify(width=width, height=height, fps=float(fps), duration=length,
                   size=path.stat().st_size)


@app.get("/media/<kind>/<name>")
def media(kind, name):
    """Serve a source or render to the page's video players (with seeking)."""
    if kind not in folders:
        abort(404)
    return send_from_directory(folders[kind], name)


@app.get("/")
def index():
    if not (PAGE / "index.html").is_file():
        return Response(NOT_BUILT, 503, mimetype="text/plain")
    # Never cached: it names the current build's files.
    return send_from_directory(PAGE, "index.html", max_age=0)


@app.get("/<path:name>")
def page(name):
    """The page's scripts and styles, at the paths the build gave them."""
    return send_from_directory(PAGE, name)


# Options, from a render request's JSON body, in timeslice.py's terms.

def read_options(values):
    def number(name, default=None, low=-math.inf, above=None):
        value = values.get(name)
        if value in (None, ""):
            return default
        try:
            value = float(value)
        except (TypeError, ValueError):
            raise Problem(f"{name} must be a number.")
        if not math.isfinite(value) or value < low or \
                (above is not None and value <= above):
            raise Problem(f"{name} is out of range.")
        return value

    def flag(name):
        return str(values.get(name, "")).lower() in ("1", "true")

    slice_ = values.get("slice") or "rotate"
    if slice_ not in ("rotate", "shear"):
        raise Problem(f"Unknown slice {slice_!r}.")
    loop = flag("loop")
    sides = loop and flag("loop_sides")
    inside = flag("inside") and not loop  # a looping frame is always inside
    motion = values.get("motion") or None
    if motion not in (None, "perpendicular", "time", "longest"):
        raise Problem(f"Unknown motion {motion!r}.")
    if slice_ == "shear" or not (inside or sides):
        motion = None  # only these rotate frames can move different ways
    elif sides and motion == "longest":
        raise Problem("A loop round the sides moves through time or "
                      "perpendicular to the frame: no line on it is the longest.")
    fps = values.get("fps") or None
    if fps is not None:
        try:
            if Fraction(fps) <= 0:
                raise ValueError
        except (ValueError, ZeroDivisionError):
            raise Problem("Output fps must be a positive number, like 24 or 30000/1001.")

    noise = None
    amplitude = number("noise", 0.0, low=0)
    if amplitude:
        noise = timeslice.Noise(amplitude)  # for its defaults
        direction = values.get("noise_direction") or noise.direction
        if direction not in ("time", "perpendicular"):
            raise Problem(f"Unknown noise direction {direction!r}.")
        seed = values.get("noise_seed")
        try:
            seed = noise.seed if seed in (None, "") else int(seed)
            if seed < 0:
                raise ValueError
        except (TypeError, ValueError):
            raise Problem("The noise seed must be a whole number, 0 or more.")
        noise = noise._replace(size=number("noise_size", noise.size, above=0),
                               speed=number("noise_speed", noise.speed),
                               direction=direction, seed=seed)

    return dict(slice=slice_, angle=number("angle", 45.0), inside=inside,
                motion=motion, noise=noise, scale=number("scale", 1.0, above=0),
                start=number("start", low=0), duration=number("duration", above=0),
                fps=fps, loop=loop, sides=sides,
                loop_fade=number("loop_fade", 0.0, low=0) if loop else 0.0,
                side_fade=number("loop_side_fade", 0.0, low=0) if sides else 0.0)


# Renders: one at a time, each running timeslice.py in its own process so it
# frees its memory when done and can be cancelled cleanly.

job_lock = threading.Lock()
job = {"state": "idle"}


def output_path(source, opts, preview):
    parts = [source.stem, opts["slice"], f"{opts['angle']:g}deg"]
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
    return folders["renders"] / ("_".join(parts) + ".mp4")


def render_command(source, output, opts, preview):
    cmd = [sys.executable, "-u", str(HERE / "timeslice.py"), str(source),
           str(output), f"--angle={opts['angle']:g}", f"--slice={opts['slice']}"]
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


def follow(proc, output):
    """Track a render's progress from timeslice.py's output."""
    log = []
    buffer = b""
    while chunk := proc.stdout.read1(4096):
        buffer += chunk
        *lines, buffer = re.split(rb"[\r\n]", buffer)
        for line in lines:
            text = line.decode(errors="replace").strip()
            if not text:
                continue
            progress = re.fullmatch(r"frame (\d+)/(\d+)", text)
            with job_lock:
                if progress:
                    job.update(stage="Slicing", frame=int(progress[1]),
                               frames=int(progress[2]))
                else:
                    log.append(text)
                    job["log"] = log[-8:]
                    if text.startswith("Loaded "):
                        job["stage"] = "Slicing"
                    elif text.startswith("Made "):
                        job["message"] = text
    code = proc.wait()
    with job_lock:
        if job.get("cancelled"):
            job["state"] = "cancelled"
            output.unlink(missing_ok=True)
        elif code == 0:
            job["state"] = "done"
        else:
            job["state"] = "failed"
            job["message"] = "\n".join(log[-3:]) or f"timeslice.py exited with {code}"
            output.unlink(missing_ok=True)
        job["elapsed"] = time.monotonic() - job["began"]


def job_status():
    status = {k: v for k, v in job.items() if k not in ("proc", "began")}
    if job.get("state") == "running":
        status["elapsed"] = time.monotonic() - job["began"]
    return status


@app.post("/api/render")
def start_render():
    body = request.get_json(force=True, silent=True) or {}
    source = source_path(body.get("source"))
    opts = read_options(body)
    preview = bool(body.get("preview"))
    with job_lock:
        if job["state"] == "running":
            raise Problem("A render is already running.", 409)
        folders["renders"].mkdir(parents=True, exist_ok=True)
        output = output_path(source, opts, preview)
        proc = subprocess.Popen(
            render_command(source, output, opts, preview), cwd=HERE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix")
        job.clear()
        job.update(state="running", stage="Loading the clip", output=output.name,
                   preview=preview, frame=0, frames=None, log=[], message="",
                   began=time.monotonic(), proc=proc)
        threading.Thread(target=follow, args=(proc, output), daemon=True).start()
        return jsonify(job_status())


@app.get("/api/render")
def render_status():
    with job_lock:
        return jsonify(job_status())


@app.post("/api/render/cancel")
def cancel_render():
    with job_lock:
        if job["state"] == "running":
            job["cancelled"] = True
            proc = job["proc"]
            try:
                if os.name == "posix":  # stop its ffmpeg processes too
                    os.killpg(proc.pid, signal.SIGTERM)
                else:
                    proc.terminate()
            except ProcessLookupError:
                pass  # it had just finished
        return jsonify(job_status())


@app.get("/api/renders")
def renders():
    folder = folders["renders"]
    with job_lock:
        busy = job.get("output") if job["state"] == "running" else None
    files = [p for p in folder.glob("*.mp4") if p.name != busy] \
        if folder.is_dir() else []
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return jsonify([dict(name=p.name, size=p.stat().st_size,
                         modified=p.stat().st_mtime) for p in files])


def main():
    parser = argparse.ArgumentParser(description="Web front end for timeslice.py")
    parser.add_argument("--videos", default=str(folders["videos"]),
                        help="folder of videos to offer (default: videos/ here)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="address to listen on (default: this machine only)")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    folders["videos"] = Path(args.videos).expanduser().resolve()
    if args.videos == parser.get_default("videos"):
        folders["videos"].mkdir(exist_ok=True)
    if not folders["videos"].is_dir():
        sys.exit(f"{folders['videos']} isn't a folder")
    if not (PAGE / "index.html").is_file():
        print(NOT_BUILT, end="")
    print(f"Offering videos from {folders['videos']}")
    print(f"Open http://{args.host}:{args.port} in a browser")
    app.run(args.host, args.port, threaded=True)


if __name__ == "__main__":
    main()
