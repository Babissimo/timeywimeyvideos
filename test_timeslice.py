"""Geometry checks for the slicer. Run with `python test_timeslice.py` (or
pytest). They use small random volumes, except the decoding checks, which make
tiny clips with ffmpeg in a temporary folder."""

import math
import subprocess
import tempfile
from fractions import Fraction

import numpy as np

import timeslice

rng = np.random.default_rng(0)
# A short wide clip, and a long narrow one that steep --inside frames fit in.
T, H, W = 7, 5, 11
volume = rng.integers(0, 256, (T, H, W, 3), np.uint8)
long_volume = rng.integers(0, 256, (13, 3, 5, 3), np.uint8)


def positions(sweep):
    """The (t, x) columns of every output frame."""
    return [sweep.at(f) for f in range(sweep.frames)]


def render(vol, sweep, nearest=False):
    frames = np.stack([timeslice.sample_columns(vol, t, x, nearest)
                       for t, x in positions(sweep)])
    assert frames.shape == (sweep.frames, vol.shape[1], sweep.width, 3)
    return frames


def rotate(angle, vol=volume, **kwargs):
    return render(vol, timeslice.rotation_sweep(len(vol), vol.shape[2], angle, **kwargs))


def shear(angle, vol=volume, **kwargs):
    return render(vol, timeslice.shear_sweep(len(vol), vol.shape[2], angle, **kwargs))


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return
    raise AssertionError(f"{error.__name__} not raised")


def test_zero_degrees_reproduces_input_in_every_mode():
    assert np.array_equal(rotate(0), volume)
    for motion in ["perpendicular", "time", "longest"]:
        assert np.array_equal(rotate(0, inside=True, motion=motion), volume)
    assert np.array_equal(shear(0), volume)
    assert np.array_equal(shear(0, inside=True), volume)


def test_ninety_degrees_gives_y_t_slices_swept_right_to_left():
    # Output frame f is the column x = W-1-f, with time running left to right.
    expected = volume.transpose(2, 1, 0, 3)[::-1]
    assert np.array_equal(rotate(90), expected)


def test_minus_ninety_degrees_sweeps_left_to_right():
    expected = volume.transpose(2, 1, 0, 3)[:, :, ::-1]
    assert np.array_equal(rotate(-90), expected)


def test_one_eighty_degrees_reverses_time_and_mirrors():
    assert np.array_equal(rotate(180), volume[::-1, :, ::-1])
    assert np.array_equal(rotate(180, inside=True), volume[::-1, :, ::-1])


def test_tilted_plane_is_black_outside_the_cuboid():
    frames = rotate(30)
    # The plane first touches the cuboid at one corner, so most of the first
    # output frame lies outside it.
    assert (frames[0] == 0).mean() > 0.5


def test_inside_ninety_degrees_perpendicular():
    # A 5-wide frame standing upright in time, centred on frame 6, moving
    # right to left across the 5 columns.
    frames = rotate(90, long_volume, inside=True, motion="perpendicular")
    expected = long_volume[4:9].transpose(2, 1, 0, 3)[::-1]
    assert np.array_equal(frames, expected)


def test_inside_frames_never_leave_the_video():
    n_frames, width = len(long_volume), long_volume.shape[2]
    for angle in [10, 30, 60, 90, 135, -45]:
        for motion in ["perpendicular", "time", "longest"]:
            sweep = timeslice.rotation_sweep(
                n_frames, width, angle, inside=True, motion=motion)
            assert sweep.width == width
            for t, x in positions(sweep):
                assert t.min() > -1e-9 and t.max() < n_frames - 1 + 1e-9
                assert x.min() > -1e-9 and x.max() < width - 1 + 1e-9


def test_longest_motion_is_at_least_as_long_as_the_others():
    for angle in [5, 30, 60, 90]:
        lengths = {motion: timeslice.rotation_sweep(
                       len(long_volume), long_volume.shape[2], angle,
                       inside=True, motion=motion).frames
                   for motion in ["perpendicular", "time", "longest"]}
        assert lengths["longest"] >= max(lengths.values()), (angle, lengths)


def test_inside_raises_when_the_clip_is_too_short():
    # sin(45 degrees) * 10 pixels is about 7 frames of time; the clip has 7,
    # spanning only 6 between its first and last.
    raises(timeslice.DoesNotFit, timeslice.rotation_sweep, T, W, 45, inside=True)
    raises(timeslice.DoesNotFit, timeslice.shear_sweep, T, W, 45, inside=True)


def test_shear_delays_each_column():
    # At 45 degrees column x is one frame later than column x-1.
    frames = shear(45, long_volume, inside=True)
    n_frames, width = len(long_volume), long_volume.shape[2]
    assert len(frames) == n_frames - (width - 1)
    for f in range(len(frames)):
        for x in range(width):
            assert np.array_equal(frames[f, :, x], long_volume[f + x, :, x])


def test_shear_without_inside_is_black_outside_the_video():
    frames = shear(45, long_volume)
    n_frames, width = len(long_volume), long_volume.shape[2]
    assert len(frames) == n_frames + (width - 1)
    # The first frame only reaches the video in its rightmost column.
    assert (frames[0, :, :-1] == 0).all()
    assert np.array_equal(frames[0, :, -1], long_volume[0, :, -1])


def test_shear_rejects_ninety_degrees():
    raises(ValueError, timeslice.shear_sweep, T, W, 90)


def test_plan_sweep_picks_the_kind_of_slice():
    assert np.array_equal(render(volume, timeslice.plan_sweep(T, W, "shear", 20)),
                          shear(20))
    assert np.array_equal(
        render(long_volume, timeslice.plan_sweep(13, 5, "rotate", 30, True, "time")),
        rotate(30, long_volume, inside=True, motion="time"))


def test_nearest_copies_the_closest_voxel():
    frame = timeslice.sample_columns(volume, [2.4, 2.6], [3.6, 3.4], nearest=True)
    assert np.array_equal(frame[:, 0], volume[2, :, 4])
    assert np.array_equal(frame[:, 1], volume[3, :, 3])


def test_general_sampler_matches_column_sampler():
    rows = np.arange(H)[:, None]
    for t, x in positions(timeslice.rotation_sweep(T, W, 30)):
        assert np.array_equal(timeslice.sample(volume, t, rows, x),
                              timeslice.sample_columns(volume, t, x))


def test_general_sampler_blends_between_voxels():
    # Halfway between two voxels along each axis is the mean of all eight.
    point = timeslice.sample(volume, [[2.5]], [[1.5]], [[3.5]])
    expected = volume[2:4, 1:3, 3:5].reshape(-1, 3).mean(axis=0)
    assert np.allclose(point[0, 0], expected, atol=0.5)


def test_general_sampler_can_copy_the_nearest_voxel():
    points = timeslice.sample(volume, [[2.4, 2.6]], [[1.6, 0.4]], [[3.6, 3.4]],
                              nearest=True)
    assert np.array_equal(points[0, 0], volume[2, 2, 4])
    assert np.array_equal(points[0, 1], volume[3, 0, 3])


def test_samplers_can_run_round_time_in_a_ring():
    # Time t reads frame t mod T, so the first frame follows the last and
    # halfway between them is a blend of the two. x still has edges.
    t = np.array([-1.0, T, T + 2.4, 2 * T - 0.5, 1.0])
    x = np.array([3.0, 4.0, 5.0, 6.0, -1.0])
    rows = np.arange(H)[:, None]
    for nearest in [False, True]:
        frame = timeslice.sample_columns(volume, t, x, nearest, wrap=True)
        assert np.array_equal(frame, timeslice.sample(volume, t, rows, x, nearest, wrap=True))
        assert np.array_equal(frame, timeslice.sample_noisy_columns(
            volume, t, x, (0.0, 0.0), Noise(1), 0, nearest, wrap=True))
        assert np.array_equal(frame[:, 0], volume[T - 1, :, 3])
        assert np.array_equal(frame[:, 1], volume[0, :, 4])
        assert (frame[:, 4] == 0).all()
    assert np.array_equal(frame[:, 2], volume[2, :, 5])
    assert np.array_equal(frame[:, 3], volume[0, :, 6])
    blend = timeslice.sample_columns(volume, t, x, wrap=True)[:, 3]
    assert np.allclose(blend, (volume[T - 1, :, 6] / 2 + volume[0, :, 6] / 2), atol=0.5)
    # Without wrap, the same points past either end are black.
    assert (timeslice.sample_columns(volume, t, x)[:, :4] == 0).all()


def test_samplers_can_run_round_the_sides_in_a_ring():
    # With wrap_x, x reads column x mod W too, the left edge following the
    # right. A strided view of the clip reads the same as a copy.
    t = np.array([2.0, 3.0, 4.0, 5.0, 3.0])
    x = np.array([-1.0, W, W + 2.4, 2 * W - 0.5, -W - 7.0])
    rows = np.arange(H)[:, None]
    strided = np.concatenate([volume, volume[:, :, :3]], axis=2)[:, :, :W]
    assert not strided.flags.c_contiguous
    for vol in [volume, strided]:
        for nearest in [False, True]:
            frame = timeslice.sample_columns(vol, t, x, nearest, wrap_x=True)
            assert np.array_equal(frame, timeslice.sample(vol, t, rows, x, nearest,
                                                          wrap_x=True))
            assert np.array_equal(frame, timeslice.sample_noisy_columns(
                vol, t, x, (0.0, 0.0), Noise(1), 0, nearest, wrap_x=True))
            assert np.array_equal(frame[:, 0], volume[2, :, W - 1])
            assert np.array_equal(frame[:, 1], volume[3, :, 0])
            assert np.array_equal(frame[:, 4], volume[3, :, (-7) % W])
        assert np.array_equal(frame[:, 2], volume[4, :, 2])
        assert np.array_equal(frame[:, 3], volume[5, :, 0])
    blend = timeslice.sample_columns(volume, t, x, wrap_x=True)[:, 3]
    assert np.allclose(blend, (volume[5, :, W - 1] / 2 + volume[5, :, 0] / 2), atol=0.5)
    # Both at once: a point past the end of the clip and past its right edge.
    corner = timeslice.sample_columns(volume, [T + 1.0], [W + 1.0], wrap=True, wrap_x=True)
    assert np.array_equal(corner[:, 0], volume[1, :, 1])
    assert (timeslice.sample_columns(volume, t, x)[:, :4] == 0).all()


# Surface noise.

Noise = timeslice.Noise


def test_noise_stays_within_its_amplitude():
    for seed in range(4):
        noise = Noise(1, size=4.3, speed=0.37, seed=seed)
        values = np.stack([noise.field(200, 150, f) for f in range(10)])
        assert np.abs(values).max() <= 1
        assert np.abs(values).max() > 0.8  # and comes close to it


def test_noise_is_smooth_and_set_by_its_seed():
    bumps = Noise(1, size=8).field(64, 48, 3)
    assert np.array_equal(bumps, Noise(1, size=8).field(64, 48, 3))
    assert not np.allclose(bumps, Noise(1, size=8, seed=1).field(64, 48, 3))
    assert np.abs(np.diff(bumps, axis=0)).max() < 0.3
    assert np.abs(np.diff(bumps, axis=1)).max() < 0.3


def test_noise_changes_over_the_sweep_unless_its_speed_is_zero():
    still, moving = Noise(1, size=8, speed=0), Noise(1, size=8, speed=1)
    assert np.array_equal(still.field(32, 24, 0), still.field(32, 24, 9))
    assert not np.allclose(moving.field(32, 24, 0), moving.field(32, 24, 9))


def test_noise_with_a_period_repeats():
    for speed in [0.37, 1.0, -0.6, 3.0]:
        noise = Noise(1, size=8, speed=speed, period=40)
        first = noise.field(32, 24, 3)
        assert np.allclose(first, noise.field(32, 24, 43), atol=1e-9), speed
        assert not np.allclose(first, noise.field(32, 24, 23)), speed
        # and runs smoothly from the end of one period into the next.
        step = np.abs(noise.field(32, 24, 1) - noise.field(32, 24, 0)).max()
        assert np.abs(noise.field(32, 24, 39) - noise.field(32, 24, 0)).max() < 2 * step
    # The speed is rounded to make that work, but never down to standing still.
    assert not np.allclose(Noise(1, size=64, speed=0.1, period=40).field(32, 24, 0),
                           Noise(1, size=64, speed=0.1, period=40).field(32, 24, 20))
    assert np.array_equal(Noise(1, size=8, speed=0, period=40).field(32, 24, 7),
                          Noise(1, size=8, speed=0).field(32, 24, 7))


def test_preview_noise_is_the_full_noise_at_half_size():
    # Sizes where both work the noise out at nodes, and where both work it
    # out at every pixel, with and without a period.
    for size in [64, 3]:
        for period in [0, 30]:
            full = Noise(1, size=size, speed=0.75, seed=2, period=period)
            half = full.scaled(0.5)
            for f in [0, 3, 10]:
                assert np.array_equal(half.field(80, 60, f),
                                      full.field(160, 120, 2 * f)[::2, ::2]), size


def exact_noise(noise, width, height, f):
    """Perlin noise worked out at every pixel, from -1 to 1."""
    out = np.empty((height, width))
    timeslice._noise_grid(timeslice._permutation(noise.seed), np.arange(width) / noise.size,
                          np.arange(height) / noise.size, f * noise.speed / noise.size, 0,
                          out)
    return out / timeslice.NOISE_BOUND


def test_noise_from_nodes_is_close_to_perlin_noise():
    for seed in range(4):
        noise = Noise(1, size=64, speed=0.37, seed=seed)
        for f in [0, 5]:
            error = np.abs(noise.field(640, 360, f) - exact_noise(noise, 640, 360, f))
            assert error.max() < 0.01 and error.mean() < 0.001, (seed, f, error.max())


def test_small_noise_is_worked_out_at_every_pixel():
    # Bumps under 16 pixels apart would put nodes under 2 pixels apart.
    for size in [3.0, 15.9]:
        noise = Noise(1, size=size, speed=0.6, seed=1)
        assert np.array_equal(noise.field(100, 80, 4), exact_noise(noise, 100, 80, 4))


def test_zero_noise_gives_the_flat_slice():
    flat = Noise(0)
    for kind, angle in [("rotate", 30), ("shear", 20)]:
        sweep = timeslice.plan_sweep(T, W, kind, angle, noise=flat)
        for nearest in [False, True]:
            bumpy = np.stack([timeslice.slice_frame(volume, sweep, f, flat, nearest)
                              for f in range(sweep.frames)])
            assert np.array_equal(bumpy, render(volume, sweep, nearest))


def test_noisy_column_sampler_matches_general_sampler():
    # Whole-plane sweeps, so some points fall outside the video too.
    cases = [(kind, angle, direction, size)
             for kind, angle in [("rotate", 30), ("rotate", 120), ("shear", 20)]
             for direction in ["time", "perpendicular"]
             for size in [3.0, 24.0]]  # noise at every pixel, and from nodes
    for kind, angle, direction, size in cases:
        noise = Noise(1.5, size=size, speed=0.6, direction=direction, seed=1)
        sweep = timeslice.plan_sweep(T, W, kind, angle, noise=noise)
        for f in range(sweep.frames):
            t, x = sweep.at(f)
            for nearest in [False, True]:
                fast = timeslice.sample_noisy_columns(
                    volume, t, x, noise.push(sweep.normal), noise, f, nearest)
                slow = timeslice.sample(
                    volume, *timeslice.surface(sweep, f, H, noise), nearest)
                assert np.array_equal(fast, slow), (kind, angle, direction, f)


def test_time_noise_reads_each_pixel_earlier_or_later():
    # At 0 degrees pixel (y, x) of output frame f comes from frame f + 2 (the
    # room the noise needs), plus the noise.
    noise = Noise(2, size=3.0, speed=0.5)
    n_frames, height, width = long_volume.shape[:3]
    sweep = timeslice.rotation_sweep(n_frames, width, 0, inside=True, noise=noise)
    assert sweep.frames == n_frames - 4
    y, x = np.indices((height, width))
    for f in range(sweep.frames):
        t = f + 2 + 2 * noise.field(width, height, f)
        expected = long_volume[np.floor(t + 0.5).astype(int), y, x]
        assert np.array_equal(
            timeslice.slice_frame(long_volume, sweep, f, noise, nearest=True), expected)


def test_perpendicular_noise_pushes_straight_off_the_plane():
    noise = Noise(1.5, direction="perpendicular")
    sweeps = [timeslice.rotation_sweep(T, W, angle, noise=noise)
              for angle in [0, 30, 90, 135]]
    sweeps.append(timeslice.shear_sweep(T, W, 30, noise=noise))
    for sweep in sweeps:
        t, x = sweep.at(2)
        push_t, push_x = noise.push(sweep.normal)
        assert abs(push_t * (t[1] - t[0]) + push_x * (x[1] - x[0])) < 1e-9
        assert math.isclose(math.hypot(push_t, push_x), 1.5)


def test_noisy_inside_frames_never_leave_the_video():
    n_frames, height, width = long_volume.shape[:3]
    fitted = 0
    for direction in ["time", "perpendicular"]:
        noise = Noise(1.0, size=2.0, speed=0.7, direction=direction)
        sweeps = [timeslice.shear_sweep(n_frames, width, 20, True, noise)] \
            if direction == "time" else []
        for angle in [0, 20, 60, 90, 135]:
            for motion in ["perpendicular", "time", "longest"]:
                try:
                    sweeps.append(timeslice.rotation_sweep(
                        n_frames, width, angle, True, motion, noise))
                except timeslice.DoesNotFit:
                    pass
        for sweep in sweeps:
            fitted += 1
            # Even pushed as far as the noise could ever push them...
            push_t, push_x = (abs(p) for p in noise.push(sweep.normal))
            for f in range(sweep.frames):
                t, x = sweep.at(f)
                assert t.min() - push_t > -1e-9 and t.max() + push_t < n_frames - 1 + 1e-9
                assert x.min() - push_x > -1e-9 and x.max() + push_x < width - 1 + 1e-9
                t, _, x = timeslice.surface(sweep, f, height, noise)
                assert t.min() > -1e-9 and t.max() < n_frames - 1 + 1e-9
                assert x.min() > -1e-9 and x.max() < width - 1 + 1e-9
    assert fitted >= 20


def test_noise_makes_room_inside():
    noise = Noise(2.0)
    plain = timeslice.rotation_sweep(13, 5, 30, inside=True, motion="time")
    bumpy = timeslice.rotation_sweep(13, 5, 30, inside=True, motion="time", noise=noise)
    assert bumpy.frames == plain.frames - 4
    plain = timeslice.shear_sweep(13, 5, 20, inside=True)
    bumpy = timeslice.shear_sweep(13, 5, 20, inside=True, noise=noise)
    assert bumpy.frames == plain.frames - 4


def test_whole_plane_sweep_starts_early_enough_for_the_bumps():
    noise = Noise(2.0, direction="perpendicular")
    assert timeslice.rotation_sweep(T, W, 30, noise=noise).frames == \
        timeslice.rotation_sweep(T, W, 30).frames + 4
    assert timeslice.shear_sweep(T, W, 30, noise=Noise(2.0)).frames == \
        timeslice.shear_sweep(T, W, 30).frames + 4


def message(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except timeslice.DoesNotFit as err:
        return str(err)
    raise AssertionError("DoesNotFit not raised")


def test_does_not_fit_says_which_angles_do():
    # sin(36.8 degrees) * 10 pixels is just under the clip's 6 frames.
    assert message(timeslice.rotation_sweep, T, W, 45, inside=True) \
        .endswith("Angles up to 36.8 degrees fit this clip.")
    # Perpendicular noise pushes the ends of a nearly flat frame sideways,
    # out of the video, until the frame turns far enough to leave room.
    sideways = Noise(1.0, direction="perpendicular")
    text = message(timeslice.rotation_sweep, 13, 5, 20, inside=True, noise=sideways)
    assert "sideways" in text and text.endswith("Angles of 0 and 53.2 to 90 degrees fit this clip.")
    assert message(timeslice.rotation_sweep, 13, 5, 0, inside=True, noise=Noise(7)) \
        .endswith("No angle fits this clip.")


def test_sheared_inside_frames_only_take_perpendicular_noise_at_zero_degrees():
    noise = Noise(1.0, direction="perpendicular")
    assert "Push the noise through time" in message(
        timeslice.shear_sweep, 13, 5, 20, inside=True, noise=noise)
    assert timeslice.shear_sweep(13, 5, 0, inside=True, noise=noise).frames == 11



# Loops.

def loop(kind, angle, vol=volume, **kwargs):
    return timeslice.plan_sweep(len(vol), vol.shape[2], kind, angle, loop=True, **kwargs)


def frames_of(vol, sweep, noise=None, nearest=False, frames=None):
    return np.stack([timeslice.slice_frame(vol, sweep, f, noise, nearest)
                     for f in range(sweep.frames if frames is None else frames)])


def test_a_loop_at_zero_degrees_is_the_clip():
    for kind in ["rotate", "shear"]:
        sweep = loop(kind, 0)
        assert sweep.loop and sweep.frames == T and sweep.width == W
        assert np.array_equal(frames_of(volume, sweep), volume)


def test_a_loop_comes_back_round_to_its_first_frame():
    # The frame after the last is the first again, so played on repeat the
    # video never jumps, at any angle and with noise too.
    noise = Noise(1.5, size=3.0, speed=0.6, seed=1)
    for kind, angle in [("rotate", 30), ("rotate", 90), ("rotate", 135),
                        ("rotate", -60), ("shear", 45), ("shear", -70)]:
        sweep = loop(kind, angle, noise=noise)
        assert sweep.frames == T and sweep.width == W
        t0, x0 = sweep.at(0)
        t1, x1 = sweep.at(sweep.frames)
        assert np.allclose(np.abs(t1 - t0), T) and np.allclose(x1, x0)
        for bumps in [None, noise]:
            first = timeslice.slice_frame(volume, sweep, 0, bumps).astype(int)
            after = timeslice.slice_frame(volume, sweep, sweep.frames, bumps).astype(int)
            assert np.abs(after - first).max() <= 1, (kind, angle, bumps)


def test_a_loop_moves_straight_through_time_inside_the_video():
    for angle in [10, 45, 80, 90, 135, -30]:
        sweep = loop("rotate", angle)
        sign = 1 if math.cos(math.radians(angle)) >= 0 else -1
        for f in range(sweep.frames):
            t, x = sweep.at(f)
            assert np.allclose(t - sweep.at(0)[0], sign * f)
            assert x.min() > -1e-9 and x.max() < W - 1 + 1e-9
    # However short the clip, since time wraps round.
    assert loop("rotate", 80, long_volume[:2]).frames == 2


def test_a_loop_reads_round_the_end_of_the_clip():
    # At 45 degrees column x of shear frame f is frame f + x - 5, wrapped.
    frames = frames_of(volume, loop("shear", 45))
    for f in range(T):
        for x in range(W):
            assert np.array_equal(frames[f, :, x], volume[(f + x - 5) % T, :, x])


def test_noisy_loops_match_the_general_sampler():
    for kind, angle, direction in [("rotate", 30, "time"), ("rotate", 70, "perpendicular"),
                                   ("shear", 20, "time")]:
        noise = Noise(1.5, size=3.0, speed=0.6, direction=direction, seed=1)
        sweep = loop(kind, angle, noise=noise)
        for f in range(sweep.frames):
            for nearest in [False, True]:
                slow = timeslice.sample(volume, *timeslice.surface(sweep, f, H, noise),
                                        nearest, wrap=True)
                assert np.array_equal(
                    timeslice.slice_frame(volume, sweep, f, noise, nearest), slow)


def test_a_loop_only_takes_perpendicular_noise_that_keeps_inside():
    sideways = Noise(1.0, direction="perpendicular")
    text = message(timeslice.rotation_sweep, 13, 5, 20, noise=sideways, loop=True)
    assert "sideways" in text and text.endswith("Angles of 0 and 53.2 to 90 degrees fit this clip.")
    assert timeslice.rotation_sweep(13, 5, 60, noise=sideways, loop=True).frames == 13
    assert "Push the noise through time" in message(
        timeslice.shear_sweep, 13, 5, 20, noise=sideways, loop=True)
    # Noise through time never leaves a ring.
    assert timeslice.shear_sweep(3, 5, 80, noise=Noise(9.0), loop=True).frames == 3


def test_a_loop_only_moves_through_time():
    raises(ValueError, timeslice.rotation_sweep, T, W, 30, motion="longest", loop=True)
    raises(ValueError, timeslice.rotation_sweep, T, W, 30, motion="perpendicular",
           loop=True)
    assert loop("rotate", 30, motion="time").frames == T


# Loops round the sides too.

def test_a_loop_round_the_sides_at_ninety_degrees_crosses_every_column():
    # Upright frames, each the y-t slice of one column, moving right to left
    # across all of them: column j of frame f is column 10 - f at time j - 2.
    frames = frames_of(volume, loop("rotate", 90, motion="perpendicular", sides=True))
    assert len(frames) == W
    for f in range(W):
        for j in range(W):
            assert np.array_equal(frames[f, :, j], volume[(j - 2) % T, :, W - 1 - f])


def test_a_loop_round_the_sides_moves_about_perpendicular_and_closes():
    noise = Noise(2.5, size=3.0, speed=0.6, direction="perpendicular", seed=1)
    for angle in [0, 20, 45, 70, 80, 90, 135, -60]:
        sweep = loop("rotate", angle, motion="perpendicular", sides=True, noise=noise)
        assert sweep.sides and sweep.width == W
        (t0, x0), (t1, x1) = sweep.at(0), sweep.at(1)
        step_t, step_x = t1[0] - t0[0], x1[0] - x0[0]
        assert 0.9 < math.hypot(step_t, step_x) < 1.1  # about a pixel a frame
        c, s = timeslice._rotation(angle)
        along = (step_t * c - step_x * s) / math.hypot(step_t, step_x)
        off = math.degrees(math.acos(min(1.0, along)))
        assert off <= timeslice.LOOP_TOLERANCE, (angle, off)
        # After the last frame it has gone a whole number of times across the
        # width and round time, so the next frame is the first again.
        t_end, x_end = sweep.at(sweep.frames)
        assert np.allclose((t_end - t0) / T, round((t_end - t0)[0] / T))
        assert np.allclose((x_end - x0) / W, round((x_end - x0)[0] / W))
        for bumps in [None, noise]:
            first = timeslice.slice_frame(volume, sweep, 0, bumps).astype(int)
            after = timeslice.slice_frame(volume, sweep, sweep.frames, bumps).astype(int)
            assert np.abs(after - first).max() <= 1, (angle, bumps)


def test_a_loop_round_the_sides_takes_the_closest_path_when_none_is_close_enough():
    # A wide, short clip at 3 degrees would need over a thousand turns round
    # time per turn across to stay within the tolerance, so it goes straight
    # through time instead, 3 degrees off.
    aim = timeslice._rotation(3)
    assert timeslice._windings(1920, 30, -aim[1], aim[0]) == (0, 1)


def test_noise_cannot_push_a_loop_round_the_sides_out_of_the_video():
    sideways = Noise(1.0, size=2.0, direction="perpendicular")
    for sweep in [timeslice.rotation_sweep(13, 5, 20, noise=sideways, loop=True, sides=True),
                  timeslice.shear_sweep(13, 5, 20, noise=sideways, loop=True, sides=True)]:
        assert sweep.frames == 13
        for f in range(sweep.frames):
            t, y, x = timeslice.surface(sweep, f, 3, sideways)
            assert np.array_equal(
                timeslice.slice_frame(long_volume, sweep, f, sideways),
                timeslice.sample(long_volume, t, y, x, wrap=True, wrap_x=True))


def test_sides_only_wrap_on_a_loop_and_never_take_the_longest_line():
    raises(ValueError, timeslice.rotation_sweep, T, W, 30, sides=True)
    raises(ValueError, timeslice.shear_sweep, T, W, 30, inside=True, sides=True)
    raises(ValueError, timeslice.rotation_sweep, T, W, 30, motion="longest", loop=True,
           sides=True)


def test_crossfade_spreads_the_join_round_the_loop():
    # A clip that brightens steadily jumps back from 120 to 0 when played on
    # repeat. Faded over 4 frames, the jump is shared out in small steps.
    ramp = np.broadcast_to(np.arange(13, dtype=np.uint8)[:, None, None, None] * 10,
                           (13, 2, 2, 3)).copy()
    looped = timeslice.crossfade(ramp, 4)
    assert len(looped) == 9
    levels = looped[:, 0, 0, 0].astype(int)
    assert list(levels) == [72, 64, 56, 48, 40, 50, 60, 70, 80]
    steps = np.abs(np.diff(np.append(levels, levels[0])))
    assert steps.max() <= 10


def test_crossfade_only_changes_the_frames_it_blends():
    looped = timeslice.crossfade(long_volume.copy(), 3)
    assert np.array_equal(looped[3:], long_volume[3:10])
    for i in range(3):
        w = (i + 1) / 4
        mixed = long_volume[10 + i] * (1 - w) + long_volume[i] * w
        assert np.abs(looped[i] - mixed).max() <= 0.5
    assert np.array_equal(timeslice.crossfade(long_volume.copy(), 0), long_volume)
    raises(ValueError, timeslice.crossfade, long_volume.copy(), 7)  # over half of 13


def test_crossfade_can_blend_across_the_sides():
    # The same blend, of the last columns into the first rather than frames.
    sideways = np.ascontiguousarray(long_volume.transpose(2, 1, 0, 3))
    expected = timeslice.crossfade(sideways, 2).transpose(2, 1, 0, 3)
    looped = timeslice.crossfade(long_volume.copy(), 2, axis=2)
    assert looped.shape == (13, 3, 3, 3)
    assert np.array_equal(looped, expected)
    raises(ValueError, timeslice.crossfade, long_volume.copy(), 3, axis=2)  # over half of 5


def test_variable_rate_clips_decode_at_the_rate_ffmpeg_paces_them_to():
    # Both clips are 240 fps to ffprobe. At a time base of 1/600, as phones and
    # screen recorders write, two uneven steps bring the average to about 30,
    # and ffmpeg paces to that. At 1/2400 one frame lasts twice as long, the
    # average stays over 70, and ffmpeg keeps to 240.
    with tempfile.TemporaryDirectory() as folder:
        decoded = {}
        for base, frames, pts in [(600, 120, "N*20+5*gte(N,30)+7*gte(N,40)"),
                                  (2400, 24, "N*10+10*gte(N,12)")]:
            clip = f"{folder}/{base}.mp4"
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x24",
                            "-frames:v", str(frames), "-vf", f"settb=1/{base},setpts='{pts}'",
                            "-fps_mode", "passthrough", "-enc_time_base", f"1/{base}",
                            "-video_track_timescale", str(base), "-pix_fmt", "yuv420p", clip],
                           check=True)
            judged = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                                     "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0",
                                     clip], capture_output=True, text=True, check=True).stdout
            assert judged.strip() == "240/1"
            _, _, fps, length = timeslice.probe(clip)
            volume, loaded_fps = timeslice.load_video(clip)
            assert fps == loaded_fps
            assert abs(len(volume) / fps - length) < 3 / fps
            decoded[base] = len(volume)
        # time_scale keeps a share of the frames ffmpeg paces to.
        halved, _ = timeslice.load_video(f"{folder}/600.mp4", time_scale=0.5)
        assert abs(len(halved) - decoded[600] / 2) <= 1


# 40 frames about 30 fps apart at a time base of 1/1000, each up to 8 ms early
# or late: too uneven for ffmpeg to find a frame rate in.
UNEVEN_MS = "settb=1/1000,setpts='N*33+trunc(8*sin(N*2.3))'"
# The same about 120 fps apart, each up to 3 ms early or late.
UNEVEN_120 = "settb=1/1000,setpts='trunc(N*25/3+3*sin(N*2.3))'"


def timed_clip(path, *options, timestamps=UNEVEN_MS, time_base="1/1000"):
    """Encode 40 frames to `path` at these timestamps, in ticks of `time_base`,
    and return the clip's r_frame_rate."""
    subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=32x24",
                    "-frames:v", "40", "-vf", timestamps, "-fps_mode", "passthrough",
                    "-enc_time_base", time_base, *options, "-pix_fmt", "yuv420p", path],
                   check=True)
    judged = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", path],
                            capture_output=True, text=True, check=True).stdout
    return Fraction(judged.strip())


def test_h264_decodes_at_the_rate_it_declares_where_ffmpeg_switches_to_that():
    # ffprobe takes the 30 fps x264 declares as 60, since H.264 can count
    # fields, but ffmpeg paces to the 30, the average being so far from 60.
    with tempfile.TemporaryDirectory() as folder:
        clip = f"{folder}/declared.mp4"
        assert timed_clip(clip, "-c:v", "libx264", "-x264-params", "force-cfr=1:fps=30",
                          "-video_track_timescale", "1000") == 60
        _, _, fps, length = timeslice.probe(clip)
        volume, loaded_fps = timeslice.load_video(clip)
        assert fps == loaded_fps == 30
        assert abs(len(volume) / fps - length) < 3 / fps


def test_a_rate_over_210_fps_twice_the_frames_gives_way_to_their_mean():
    # An IVF file, like a browser's WebM recording, gives no rate in its
    # header, so ffmpeg would pace to the 1000 Hz time base, 33 copies of every
    # frame. The mean rate of the frames stands in, 39 steps in 1294 ms.
    with tempfile.TemporaryDirectory() as folder:
        clip = f"{folder}/recording.ivf"
        assert timed_clip(clip, "-c:v", "libvpx-vp9") == 1000
        _, _, fps, _ = timeslice.probe(clip)
        volume, loaded_fps = timeslice.load_video(clip)
        assert fps == loaded_fps == Fraction(39_000, 1294)
        assert abs(len(volume) - 40) <= 1
        # time_scale keeps a share of them.
        assert abs(len(timeslice.load_video(clip, time_scale=0.5)[0]) - 20) <= 1
        # Likewise where ffmpeg paces to the 1000 fps x264 declares for its time
        # base, here in an MP4 counting 1/16000 s: 39 steps in 327 ms.
        declared = f"{folder}/declared.mp4"
        assert timed_clip(declared, "-c:v", "libx264", timestamps=UNEVEN_120) == 2000
        assert timeslice.probe(declared)[2] == Fraction(39_000, 327)
        # A single frame spans no time, so ffmpeg's rate stands.
        single = f"{folder}/single.ivf"
        timed_clip(single, "-c:v", "libvpx-vp9", "-frames:v", "1")
        assert timeslice.probe(single)[2] == 1000
        assert len(timeslice.load_video(single)[0]) == 1
        # So does a rate the frames come at, here 240 fps at a time base of 1/240
        # with frame 5 a tick late.
        steady = f"{folder}/steady.mp4"
        assert timed_clip(steady, "-video_track_timescale", "240", time_base="1/240",
                          timestamps="settb=1/240,setpts='N+gte(N,5)'") == 240
        assert timeslice.probe(steady)[2] == 240

if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
