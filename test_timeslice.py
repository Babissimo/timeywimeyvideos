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


def render(vol, plan, nearest=False):
    out_width, out_frames, slices = plan
    frames = np.stack([timeslice.sample_columns(vol, t, x, nearest) for t, x in slices])
    assert frames.shape == (out_frames, vol.shape[1], out_width, 3)
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
            out_width, _, slices = timeslice.rotation_sweep(
                n_frames, width, angle, inside=True, motion=motion)
            assert out_width == width
            for t, x in slices:
                assert t.min() > -1e-9 and t.max() < n_frames - 1 + 1e-9
                assert x.min() > -1e-9 and x.max() < width - 1 + 1e-9


def test_longest_motion_is_at_least_as_long_as_the_others():
    for angle in [5, 30, 60, 90]:
        lengths = {motion: timeslice.rotation_sweep(
                       len(long_volume), long_volume.shape[2], angle,
                       inside=True, motion=motion)[1]
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


def test_nearest_copies_the_closest_voxel():
    frame = timeslice.sample_columns(volume, [2.4, 2.6], [3.6, 3.4], nearest=True)
    assert np.array_equal(frame[:, 0], volume[2, :, 4])
    assert np.array_equal(frame[:, 1], volume[3, :, 3])


def test_general_sampler_matches_column_sampler():
    _, _, slices = timeslice.rotation_sweep(T, W, 30)
    rows = np.arange(H)[:, None]
    for t, x in slices:
        assert np.array_equal(timeslice.sample(volume, t, rows, x),
                              timeslice.sample_columns(volume, t, x))


def test_general_sampler_blends_between_voxels():
    # Halfway between two voxels along each axis is the mean of all eight.
    point = timeslice.sample(volume, [[2.5]], [[1.5]], [[3.5]])
    expected = volume[2:4, 1:3, 3:5].reshape(-1, 3).mean(axis=0)
    assert np.allclose(point[0, 0], expected, atol=0.5)


if __name__ == "__main__":
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("ok", name)
