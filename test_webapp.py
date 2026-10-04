"""Checks for the web front end's server. Run with `python test_webapp.py` (or
pytest). They make a tiny clip with ffmpeg in a temporary folder."""

import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

import timeslice
import webapp

tmp = Path(tempfile.mkdtemp(prefix="timeslice-test-"))
for name in ["videos", "uploads", "renders"]:
    webapp.folders[name] = tmp / name
    webapp.folders[name].mkdir()
clip = webapp.folders["videos"] / "clip.mp4"
subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                "testsrc2=size=64x48:rate=20:duration=3", "-pix_fmt", "yuv420p",
                str(clip)], check=True)
client = webapp.app.test_client()


def frame(**params):
    res = client.get("/api/frame", query_string={"source": "videos/clip.mp4", **params})
    if res.status_code != 200:
        return res, None, None
    info = json.loads(res.headers["X-Info"])
    image = np.frombuffer(res.data, np.uint8).reshape(info["height"], info["width"], 3)
    return res, info, image


def test_lists_sources():
    assert client.get("/api/sources").get_json()["videos"] == ["clip.mp4"]


def test_live_frames_match_a_preview_render():
    volume, _ = timeslice.load_video(str(clip), 0.5, 0.5, fast=True)
    _, info, image = frame(angle=0, pos=0)
    assert info["volume"] == list(volume.shape[:3])
    assert np.array_equal(image, volume[0])

    _, info, image = frame(angle=30, slice="rotate", pos=0.5)
    sweep = timeslice.plan_sweep(len(volume), volume.shape[2], "rotate", 30)
    f = round(0.5 * (sweep.frames - 1))
    assert info["frame"] == f and info["frames"] == sweep.frames
    assert np.array_equal(image, timeslice.sample_columns(volume, *sweep.at(f), nearest=True))


def test_frame_reports_full_size_render():
    _, info, _ = frame(angle=20, slice="shear")
    assert info["full"]["width"] == 64 and info["full"]["height"] == 48
    assert info["fps"] == 10  # half the input's 20 fps, like a preview render


def test_too_steep_for_inside_says_so():
    res, _, _ = frame(angle=80, inside=1)  # up to about 69 degrees fits
    assert res.status_code == 400
    assert res.get_json()["kind"] == "does_not_fit"
    assert "untick Inside" in res.get_json()["error"]


def test_bad_options_are_refused():
    assert frame(angle="steep")[0].status_code == 400
    assert frame(slice="twist")[0].status_code == 400
    assert frame(scale=0)[0].status_code == 400
    assert frame(fps="-3")[0].status_code == 400


def test_only_serves_files_in_its_folders():
    for source in ["videos/../webapp.py", "videos/.clip.mp4", "elsewhere/clip.mp4",
                   "videos/missing.mp4", ""]:
        res = client.get("/api/frame", query_string={"source": source})
        assert res.status_code == 404, source
    assert client.get("/media/videos/..%2Fwebapp.py").status_code == 404


def test_upload_adds_a_source():
    res = client.post("/api/upload", data=clip.read_bytes(),
                      headers={"X-Filename": "my%20clip.mp4"})
    assert res.get_json()["source"] == "uploads/my_clip.mp4"
    assert client.get("/api/sources").get_json()["uploads"] == ["my_clip.mp4"]
    res = client.post("/api/upload", data=b"hello", headers={"X-Filename": "notes.txt"})
    assert res.status_code == 400


def test_render_runs_timeslice():
    res = client.post("/api/render", json={"source": "videos/clip.mp4", "angle": 30,
                                           "preview": True})
    assert res.get_json()["state"] == "running"
    assert client.post("/api/render", json={"source": "videos/clip.mp4"}).status_code == 409
    for _ in range(600):
        job = client.get("/api/render").get_json()
        if job["state"] != "running":
            break
        time.sleep(0.1)
    assert job["state"] == "done", job
    assert job["output"] == "clip_rotate_30deg_preview.mp4"
    names = [r["name"] for r in client.get("/api/renders").get_json()]
    assert names == ["clip_rotate_30deg_preview.mp4"]
    assert client.get(f"/media/renders/{names[0]}").status_code == 200


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
