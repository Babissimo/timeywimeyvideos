"""Checks for the web front end's server. Run with `python test_webapp.py` (or
pytest). They make a tiny clip with ffmpeg in a temporary folder."""

import subprocess
import tempfile
from pathlib import Path

import webapp

tmp = Path(tempfile.mkdtemp(prefix="timeslice-test-"))
for name in ["videos", "uploads"]:
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
        try:
            webapp.read_options(options)
        except webapp.Problem as err:
            assert err.status == 400, options
        else:
            raise AssertionError(f"accepted {options}")


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


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
