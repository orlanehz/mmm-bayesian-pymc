import numpy as np
import pytest

from mmm.transforms import geometric_adstock, log_saturation


def test_adstock_decay_zero_identity():
    x = np.array([0, 1, 2, 0], dtype=float)
    y = geometric_adstock(x, decay=0.0)
    assert np.allclose(y, x)


def test_adstock_impulse_response():
    x = np.array([0, 10, 0, 0], dtype=float)
    y = geometric_adstock(x, decay=0.5)
    assert y[1] == 10
    assert y[2] == 5
    assert y[3] == 2.5


def test_adstock_rejects_negative_x():
    with pytest.raises(ValueError):
        geometric_adstock(np.array([1, -1], dtype=float), decay=0.5)


def test_adstock_rejects_bad_decay():
    with pytest.raises(ValueError):
        geometric_adstock(np.array([1, 2], dtype=float), decay=1.0)


def test_log_saturation_monotonic_increasing():
    x = np.array([0, 1, 10, 100], dtype=float)
    y = log_saturation(x)
    assert np.all(np.diff(y) > 0)