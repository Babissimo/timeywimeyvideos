"""The page's options in timeslice.py's terms: the reference for web/src/options.ts.

The generators beside it import this once the repository root is on the path.
"""

import math
from fractions import Fraction

import timeslice


class Problem(Exception):
    """Something the page should show the user, and what kind of problem it is."""

    def __init__(self, message, kind="error"):
        super().__init__(message)
        self.kind = kind


def read_options(values):
    """The options the page's values ask for, or the Problem with the first one refused."""
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
