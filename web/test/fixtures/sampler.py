"""Write sampler.json: what timeslice.py's samplers give, for the GPU slicer's tests.

Three volumes: the two random ones from test_timeslice.py, stored, and a larger one
whose voxels both sides work out from FORMULA, so it needn't be. For a spread of
sweeps, each case holds the columns (t, x) the planner gives for some of its frames,
the noise grid and push where the sweep is noisy, and what slice_frame makes of them
with nearest off and on.

Arrays are base64: uint8 as is, float64 and int32 little-endian.

    uv run python web/test/fixtures/sampler.py
"""

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import timeslice  # noqa: E402
from test_timeslice import long_volume, volume  # noqa: E402

FORMULA = "(t*131 + y*71 + x*37 + c*17 + (t*x) % 23) % 256"
BIG_SHAPE = (200, 4, 300)


def formula_volume(frames, height, width):
    t, y, x, c = np.ogrid[:frames, :height, :width, :3]
    return ((t * 131 + y * 71 + x * 37 + c * 17 + (t * x) % 23) % 256).astype(np.uint8)


VOLUMES = {"small": volume, "long": long_volume, "big": formula_volume(*BIG_SHAPE)}
Noise = timeslice.Noise
SMALL_NOISE = Noise(1.5, size=3.0, speed=0.6, seed=1)  # a node at every pixel
BIG_NOISE = Noise(4.0, size=40.0, speed=2.0, seed=3)
SIDEWAYS = Noise(6.0, size=64.0, direction="perpendicular", seed=2)

# name, volume, slice, angle, plan_sweep options
SWEEPS = [
    ("rotate 30", "small", "rotate", 30, {}),
    ("rotate 90", "small", "rotate", 90, {}),
    ("rotate 135", "small", "rotate", 135, {}),
    ("rotate -60", "small", "rotate", -60, {}),
    ("rotate 45 inside", "long", "rotate", 45, {"inside": True}),
    ("rotate 60 inside perpendicular", "long", "rotate", 60,
     {"inside": True, "motion": "perpendicular"}),
    ("shear 20", "small", "shear", 20, {}),
    ("shear 30 inside", "long", "shear", 30, {"inside": True}),
    ("loop rotate 30", "small", "rotate", 30, {"loop": True}),
    ("loop shear 45", "small", "shear", 45, {"loop": True}),
    ("loop sides rotate 70", "small", "rotate", 70,
     {"loop": True, "sides": True, "motion": "perpendicular"}),
    ("noise time rotate 30", "small", "rotate", 30, {"noise": SMALL_NOISE}),
    ("noise perpendicular rotate 70", "small", "rotate", 70,
     {"noise": SMALL_NOISE._replace(direction="perpendicular")}),
    ("noise time shear 20", "small", "shear", 20, {"noise": SMALL_NOISE}),
    ("noise loop rotate 30", "small", "rotate", 30, {"loop": True, "noise": SMALL_NOISE}),
    ("noise loop sides rotate 70", "small", "rotate", 70,
     {"loop": True, "sides": True, "motion": "perpendicular",
      "noise": SMALL_NOISE._replace(direction="perpendicular")}),
    ("big rotate 30", "big", "rotate", 30, {}),
    ("big rotate 35 inside", "big", "rotate", 35, {"inside": True}),
    ("big shear 25", "big", "shear", 25, {}),
    ("big loop rotate 60", "big", "rotate", 60, {"loop": True}),
    ("big loop sides rotate 50", "big", "rotate", 50,
     {"loop": True, "sides": True, "motion": "perpendicular"}),
    ("big noise time rotate 30", "big", "rotate", 30, {"noise": BIG_NOISE}),
    ("big noise perpendicular rotate 45", "big", "rotate", 45, {"noise": SIDEWAYS}),
    ("big noise time shear 15 inside", "big", "shear", 15,
     {"inside": True, "noise": BIG_NOISE}),
    ("big noise loop rotate 20", "big", "rotate", 20, {"loop": True, "noise": BIG_NOISE}),
    ("big noise loop sides rotate 60", "big", "rotate", 60,
     {"loop": True, "sides": True, "motion": "perpendicular", "noise": SIDEWAYS}),
]


def b64(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype).tobytes()).decode()


def frames_of(sweep, name):
    """Every frame of a small sweep (at most 12, evenly spread), two of a big one."""
    if name.startswith("big"):
        return [sweep.frames // 3, 2 * sweep.frames // 3]
    if sweep.frames <= 12:
        return list(range(sweep.frames))
    return sorted({round(i * (sweep.frames - 1) / 11) for i in range(12)})


def slice_case(name, vol_name, kind, angle, options):
    vol = VOLUMES[vol_name]
    sweep = timeslice.plan_sweep(len(vol), vol.shape[2], kind, angle, **options)
    noise = options.get("noise")
    case = {"name": name, "volume": vol_name, "wrap": sweep.loop, "wrapX": sweep.sides,
            "width": sweep.width, "frames": []}
    if noise is not None:
        noise = timeslice._looping(noise, sweep)
        # Only the nodes change from frame to frame.
        _, cols, col_w, rows, row_w = noise._grid(sweep.width, vol.shape[1], 0)
        case["noise"] = {"push": list(noise.push(sweep.normal)),
                         "cols": b64(cols, "<i4"), "colW": b64(col_w, "<f8"),
                         "rows": b64(rows, "<i4"), "rowW": b64(row_w, "<f8")}
    for f in frames_of(sweep, name):
        t, x = sweep.at(f)
        frame = {"f": f, "t": b64(t, "<f8"), "x": b64(x, "<f8")}
        if noise is not None:
            nodes = noise._grid(sweep.width, vol.shape[1], f)[0]
            frame["nodes"] = b64(nodes, "<f8")
            frame["nodeRows"], frame["nodeCols"] = nodes.shape
        for nearest in (False, True):
            out = timeslice.slice_frame(vol, sweep, f, options.get("noise"), nearest)
            frame["nearest" if nearest else "bilinear"] = b64(out, "u1")
        case["frames"].append(frame)
    return case


def main():
    fixture = {
        "formula": FORMULA,
        "volumes": {name: {"shape": list(vol.shape[:3])} for name, vol in VOLUMES.items()},
        "slices": [slice_case(*sweep) for sweep in SWEEPS],
    }
    for name in ("small", "long"):  # the big volume comes from FORMULA
        fixture["volumes"][name]["rgb"] = b64(VOLUMES[name], "u1")
    path = Path(__file__).with_name("sampler.json")
    path.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}, {path.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
