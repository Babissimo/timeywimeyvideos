"""Geometry checks for the y-axis slicer. Run with `python test_timeslice.py`
(or pytest). They use a small random volume, so no video files are needed."""

import numpy as np

import timeslice

T, H, W = 7, 5, 11
volume = np.random.default_rng(0).integers(0, 256, (T, H, W, 3), np.uint8)


def sweep(angle):
    out_width, out_frames, slices = timeslice.y_rotation_slices(T, W, angle)
    frames = np.stack([timeslice.sample_columns(volume, t, x) for t, x in slices])
    assert frames.shape == (out_frames, H, out_width, 3)
    return frames


def test_zero_degrees_reproduces_input():
    assert np.array_equal(sweep(0), volume)


def test_ninety_degrees_gives_y_t_slices_swept_right_to_left():
    # Output frame f is the column x = W-1-f, with time running left to right.
    expected = volume.transpose(2, 1, 0, 3)[::-1]
    assert np.array_equal(sweep(90), expected)


def test_minus_ninety_degrees_sweeps_left_to_right():
    expected = volume.transpose(2, 1, 0, 3)[:, :, ::-1]
    assert np.array_equal(sweep(-90), expected)


def test_one_eighty_degrees_reverses_time_and_mirrors():
    assert np.array_equal(sweep(180), volume[::-1, :, ::-1])


def test_tilted_plane_is_black_outside_the_cuboid():
    frames = sweep(30)
    # The plane first touches the cuboid at one corner, so most of the first
    # output frame lies outside it.
    assert (frames[0] == 0).mean() > 0.5


def test_general_sampler_matches_column_sampler():
    _, _, slices = timeslice.y_rotation_slices(T, W, 30)
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
