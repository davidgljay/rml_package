"""Utility functions tests."""

import numpy as np # type: ignore

from relationality.util import dims, shape_lift


def test_dims():
    assert dims(()) == 0
    assert dims(4) == 1
    assert dims((4,)) == 1
    assert dims((3, 5)) == 2
    assert dims((1, 2, 3)) == 3
    assert dims(np.array([1, 2, 3])) == 1
    assert dims(np.array([[1, 2, 3]])) == 2


def test_shape_lift():
    assert shape_lift(()) == ()
    assert shape_lift(5) == (5,)
    assert shape_lift((5,)) == (5,)
    assert shape_lift((3, 2)) == (3, 2)
