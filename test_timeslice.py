"""Geometry checks for the slicer. Run with `python test_timeslice.py` (or
pytest). They use small random volumes, so no video files are needed."""

import math

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


def test_preview_noise_is_the_full_noise_at_half_size():
    full = Noise(1, size=16, speed=0.75, seed=2)
    half = full.scaled(0.5)
    for f in [0, 3, 10]:
        assert np.array_equal(half.field(32, 24, f), full.field(64, 48, 2 * f)[::2, ::2])


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
    for kind, angle in [("rotate", 30), ("rotate", 120), ("shear", 20)]:
        for direction in ["time", "perpendicular"]:
            noise = Noise(1.5, size=3.0, speed=0.6, direction=direction, seed=1)
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


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
