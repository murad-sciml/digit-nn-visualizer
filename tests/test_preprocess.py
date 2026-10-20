import numpy as np
from PIL import Image

from nn_viz.preprocess import to_8x8
from synthetic import draw_digit


def test_blank_canvas_returns_none():
    assert to_8x8(Image.new("L", (320, 320), 0)) is None


def test_output_shape_and_range():
    x = to_8x8(draw_digit(3))
    assert x.shape == (64,)
    assert x.min() >= 0 and x.max() <= 1


def test_digit_fills_full_height_like_training_data():
    # the training digits always span all 8 rows
    x = to_8x8(draw_digit(1, scale=0.4)).reshape(8, 8)
    assert (x.sum(axis=1) > 0).all()


def test_position_and_size_do_not_matter():
    a = to_8x8(draw_digit(4, scale=0.8))
    b = to_8x8(draw_digit(4, scale=0.45, dx=0.2, dy=-0.15))
    assert np.abs(a - b).mean() < 0.08


def test_thin_strokes_are_thickened():
    thin = to_8x8(draw_digit(7, width=6))
    assert thin.mean() > 0.2
