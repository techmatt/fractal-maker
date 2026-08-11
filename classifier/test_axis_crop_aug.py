"""`Transform(axis_crop=...)` — the (28) render-mode aug dial.

Two properties matter and both are silent when broken: the dial must be OFF by default (it
was added under eleven live recipes, none of which may move), and the DEPLOY path must be
untouched, because a train-only aug that leaked into deploy would shift every stored score.

The third set pins the shape difference from `border_crop`, which is the whole reason a
second dial exists rather than a different number in the old one.
"""
from __future__ import annotations

import random

import numpy as np
import pytest
from PIL import Image

from classifier.data import TARGET_H, TARGET_W, Transform

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def _tf(**kw):
    base = dict(geometry="stretch", interp="bicubic", mean=MEAN, std=STD, train=True,
                border_crop=0.0, jpeg_q=None, brightness=0.0, contrast=0.0,
                hflip=0.0, vflip=0.0)
    base.update(kw)
    return Transform(**base)


@pytest.fixture
def img():
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (720, 1280, 3), dtype=np.uint8))


def test_axis_crop_is_off_by_default(img):
    """Eleven trainers construct a Transform without naming this dial."""
    assert Transform(geometry="stretch", interp="bicubic", mean=MEAN, std=STD,
                     train=True).axis_crop == 0.0
    a = _tf()(img, random.Random(1))
    b = _tf()(img, random.Random(2))
    assert np.allclose(a.numpy(), b.numpy()), "with every dial off, train must be deterministic"


def test_deploy_path_ignores_axis_crop(img):
    """`train=False` is the deploy mirror and must not see any aug at all."""
    d0 = _tf(train=False, axis_crop=0.0)(img)
    d1 = _tf(train=False, axis_crop=0.03)(img)
    assert np.allclose(d0.numpy(), d1.numpy())


def test_output_geometry_is_unchanged(img):
    t = _tf(axis_crop=0.03)(img, random.Random(3))
    assert tuple(t.shape) == (3, TARGET_H, TARGET_W)


def test_axis_crop_actually_varies_the_frame(img):
    outs = [_tf(axis_crop=0.03)(img, random.Random(s)).numpy() for s in range(6)]
    assert any(not np.allclose(outs[0], o) for o in outs[1:]), "the dial must do something"


def test_axis_crop_is_bounded_by_a_not_2a():
    """THE shape difference. `border_crop=b` takes U(0,b) off EACH edge, so an axis can lose
    up to 2b and the frame translates; `axis_crop=a` takes ONE U(0,a) per axis. Measured on
    the crop box itself by intercepting PIL, because the resize hides it afterwards."""
    seen = []
    src = Image.fromarray(np.zeros((720, 1280, 3), dtype=np.uint8))
    real_crop = Image.Image.crop

    def spy(self, box):
        seen.append((self.size, box))
        return real_crop(self, box)

    Image.Image.crop = spy
    try:
        for s in range(200):
            _tf(axis_crop=0.03)(src, random.Random(s))
    finally:
        Image.Image.crop = real_crop

    assert seen, "axis_crop must reach PIL.crop"
    lost_x = [(w - (r - l)) / w for (w, h), (l, t, r, b) in seen]
    lost_y = [(h - (b - t)) / h for (w, h), (l, t, r, b) in seen]
    # The cap is 3% plus at most half a pixel of rounding: `int(round(0.03*720)) == 22`
    # is 3.06% of the axis. Stated rather than absorbed into a fudge factor — the bound
    # this test defends is "a, not 2a", and one pixel is not the difference.
    assert max(lost_x) <= 0.03 + 0.5 / 1280 and max(lost_y) <= 0.03 + 0.5 / 720
    assert max(lost_x) > 0.02 and max(lost_y) > 0.02, "the draw should reach near its cap"


def test_border_and_axis_crop_compose(img):
    """The strong arm sets both; neither may silently disable the other."""
    tf = _tf(border_crop=0.10, axis_crop=0.03)
    outs = [tf(img, random.Random(s)).numpy() for s in range(4)]
    assert any(not np.allclose(outs[0], o) for o in outs[1:])
    assert tf.border_crop == 0.10 and tf.axis_crop == 0.03


def test_a_degenerate_cap_cannot_crop_the_image_away():
    tiny = Image.fromarray(np.zeros((12, 12, 3), dtype=np.uint8))
    t = _tf(axis_crop=0.99)(tiny, random.Random(0))
    assert tuple(t.shape) == (3, TARGET_H, TARGET_W)
