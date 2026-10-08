"""Checks for the web front end's server. Run with `python test_webapp.py` (or
pytest). They make a tiny clip with ffmpeg in a temporary folder."""

import subprocess
import tempfile
import time
from pathlib import Path

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


def test_serves_the_built_page():
    page = tmp / "dist"
    (page / "assets").mkdir(parents=True)
    (page / "index.html").write_text("<script src=/assets/app.js></script>")
    (page / "assets" / "app.js").write_text("// the page")
    built = webapp.PAGE
    try:
        webapp.PAGE = page
        res = client.get("/")
        assert res.status_code == 200 and b"/assets/app.js" in res.data
        assert res.cache_control.max_age == 0
        assert client.get("/assets/app.js").data == b"// the page"
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/assets/..%2F..%2Fwebapp.py").status_code == 404
        webapp.PAGE = tmp / "unbuilt"
        res = client.get("/")
        assert res.status_code == 503 and b"npm run build" in res.data
    finally:
        webapp.PAGE = built


def test_lists_sources():
    assert client.get("/api/sources").get_json()["videos"] == ["clip.mp4"]


def test_bad_options_are_refused():
    for options in [dict(angle="steep"), dict(slice="twist"), dict(scale=0), dict(fps="-3"),
                    dict(noise=-1), dict(noise=2, noise_size=0),
                    dict(noise=2, noise_direction="sideways"), dict(noise=2, noise_seed="1.5"),
                    dict(noise=2, noise_seed=-1), dict(loop=1, loop_fade=-1),
                    dict(loop=1, loop_sides=1, motion="longest")]:
        res = client.post("/api/render", json={"source": "videos/clip.mp4", **options})
        assert res.status_code == 400, options  # before any render starts


def test_only_serves_files_in_its_folders():
    for source in ["videos/../webapp.py", "videos/.clip.mp4", "elsewhere/clip.mp4",
                   "videos/missing.mp4", ""]:
        res = client.get("/api/info", query_string={"source": source})
        assert res.status_code == 404, source
    assert client.get("/media/videos/..%2Fwebapp.py").status_code == 404


def test_upload_adds_a_source():
    res = client.post("/api/upload", data=clip.read_bytes(),
                      headers={"X-Filename": "my%20clip.mp4"})
    assert res.get_json()["source"] == "uploads/my_clip.mp4"
    assert client.get("/api/sources").get_json()["uploads"] == ["my_clip.mp4"]
    res = client.post("/api/upload", data=b"hello", headers={"X-Filename": "notes.txt"})
    assert res.status_code == 400


def finish_render():
    for _ in range(600):
        job = client.get("/api/render").get_json()
        if job["state"] != "running":
            return job
        time.sleep(0.1)
    raise AssertionError("the render took more than a minute")


def test_render_runs_timeslice():
    res = client.post("/api/render", json={"source": "videos/clip.mp4", "angle": 30,
                                           "preview": True})
    assert res.get_json()["state"] == "running"
    assert client.post("/api/render", json={"source": "videos/clip.mp4"}).status_code == 409
    job = finish_render()
    assert job["state"] == "done", job
    assert job["output"] == "clip_rotate_30deg_preview.mp4"
    names = [r["name"] for r in client.get("/api/renders").get_json()]
    assert names == ["clip_rotate_30deg_preview.mp4"]
    assert client.get(f"/media/renders/{names[0]}").status_code == 200


# At half size, as the live view and a preview render hold it, the clip is 30
# frames at 10 fps, 32 pixels wide.

def test_render_with_noise_makes_what_the_live_view_shows():
    options = dict(angle=30, inside=1, noise=4, noise_direction="perpendicular",
                   noise_seed=2)
    noise = timeslice.Noise(4, direction="perpendicular", seed=2).scaled(0.5)
    sweep = timeslice.plan_sweep(30, 32, "rotate", 30, inside=True, noise=noise)
    res = client.post("/api/render", json={"source": "videos/clip.mp4", "preview": True,
                                           **options})
    assert res.get_json()["state"] == "running"
    job = finish_render()
    assert job["state"] == "done", job
    assert job["output"] == \
        "clip_rotate_30deg_inside_noise4_noiseperpendicular_noiseseed2_preview.mp4"
    frames = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0",
         str(webapp.folders["renders"] / job["output"])],
        capture_output=True, text=True, check=True).stdout
    assert int(frames) == sweep.frames


def test_render_of_a_loop_makes_what_the_live_view_shows():
    options = dict(angle=60, loop=1, loop_fade=0.4, loop_sides=1, motion="perpendicular",
                   loop_side_fade=6)
    # The fades take 4 frames and 3 columns.
    sweep = timeslice.plan_sweep(26, 29, "rotate", 60, motion="perpendicular", loop=True,
                                 sides=True)
    res = client.post("/api/render", json={"source": "videos/clip.mp4", "preview": True,
                                           **options})
    assert res.get_json()["state"] == "running"
    job = finish_render()
    assert job["state"] == "done", job
    assert job["output"] == ("clip_rotate_60deg_perpendicular_loop_loopfade0.4s_sides_"
                             "sidefade6px_preview.mp4")
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries", "stream=nb_read_frames,width", "-of", "csv=p=0",
         str(webapp.folders["renders"] / job["output"])],
        capture_output=True, text=True, check=True).stdout.strip()
    width, frames = probe.split(",")
    assert int(frames) == sweep.frames and int(width) == sweep.width + sweep.width % 2


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
