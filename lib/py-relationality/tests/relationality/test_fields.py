"""Fibers tests."""

import numpy as np # type: ignore
from numpy.testing import assert_equal, assert_almost_equal # type: ignore

from relationality.fields import (
    Histo, Distro, normalize, information,
    entropy, relative_entropy, accumulate, draw
)


def test_histo_construction():
    assert_equal(Histo.uniform(3), Histo.from_iterable([1., 1., 1.]))
    assert_equal(Histo.uniform(3, 0.), Histo.from_iterable([0., 0., 0.]))
    assert_equal(Histo.dirac(3), Histo.from_iterable([1., 0., 0.]))
    assert_equal(Histo.dirac(3, 1), Histo.from_iterable([0., 1., 0.]))


def test_distro_construction():
    assert_equal(Distro.uniform(4), Distro.from_iterable([1., 1., 1., 1.]))
    assert_equal(Distro.dirac(3), Distro.from_iterable([1., 0., 0.]))
    assert_equal(Distro.dirac(3, 1), Distro.from_iterable([0., 1., 0.]))
    assert_equal(Distro.uniform(2), np.array([0.5, 0.5]))


def test_normalize():
    assert_equal(normalize(Histo.uniform(2)), Distro.uniform(2))
    assert_equal(normalize(Histo.uniform(1)), np.array([1.]))
    assert_equal(normalize(Histo.uniform(2)), np.array([0.5, 0.5]))


def test_information():
    assert_equal(information(Distro.uniform(2)), np.array([1., 1.]))
    assert_equal(information(Distro.uniform(4)), np.array([2., 2., 2., 2.]))
    assert_equal(information(Distro.dirac(3)), np.array([0., 0., 0.]))


def test_entropy():
    for k in range(5):
        assert entropy(Distro.uniform(2**k)) == k

    for k in range(1, 6):
        assert entropy(Distro.dirac(k)) == 0

    for dist, ent_val in (
        ([2., 1., 1.], 1.5),
        ([1., 2., 1.], 1.5),
        ([4., 2., 1., 1.], 1.75),
        ([1., 2., 1., 4.], 1.75),
        ([4., 1., 1., 1., 1.], 2.),
        ([1., 2., 2., 2., 1.], 2.25),
        ([2., 2., 1., 2., 1.], 2.25)
    ):
        assert entropy(Distro.from_iterable(dist)) == ent_val


def test_entropy_density():
    assert relative_entropy(Distro.uniform(1)) == 0
    for k in range(1, 5):
        assert relative_entropy(Distro.uniform(2**k)) == 1

    for k in range(1, 6):
        assert relative_entropy(Distro.dirac(k)) == 0

    for dist, ent_val in (
        ([2., 1., 1.], 0.94639),
        ([1., 2., 1.], 0.94639),
        ([4., 2., 1., 1.], 0.875),
        ([1., 2., 1., 4.], 0.875),
        ([4., 1., 1., 1., 1.], 0.86135),
        ([1., 2., 2., 2., 1.], 0.96902),
        ([2., 2., 1., 2., 1.], 0.96902)
    ):
        rent = relative_entropy(Distro.from_iterable(dist))
        assert_almost_equal(rent, ent_val, 4)


def test_cumulate_field():
    for dist, cumu in (
        ([2., 1., 1.], [0.5, 0.75, 1.]),
        ([1., 2., 1.], [0.25, 0.75, 1.]),
        ([4., 2., 1., 1.], [0.5, 0.75, 0.875, 1.]),
    ):
        cumulate = accumulate(Distro.from_iterable(dist))
        assert_equal(cumulate, np.array(cumu))


def test_entropy_field():
    test_field = Distro.from_iterable([
        [[4., 2.], [1., 1.]],
        [[1., 1.], [1., 1.]],
        [[1., 2.], [1., 4.]],
    ], base_rank=1)
    ents = entropy(test_field)
    assert_equal(ents, np.array([1.75, 2., 1.75]))


def test_relative_entropy_field():
    test_field = Distro.from_iterable([
        [[4., 2.], [1., 1.]],
        [[1., 1.], [1., 1.]],
        [[1., 2.], [1., 4.]],
    ], base_rank=1)
    ents = relative_entropy(test_field)
    assert_almost_equal(ents, np.array([1.104127, 1.26186 , 1.104127]), 4)


def test_draw():
    for shape, index in (
        ((5,), (3,)),
        ((5, 4), (3, 2)),
        ((5, 4, 2), (3, 2, 0)),
    ):
        distro = Distro.dirac(shape, index)
        assert draw(distro) == index
