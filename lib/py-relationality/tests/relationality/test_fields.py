"""Fibers tests."""

import numpy as np # type: ignore
from numpy.testing import assert_equal, assert_allclose # type: ignore

from relationality.fields import (
    Histo, Distro, norm, normalize, information,
    entropy, relative_entropy, accumulate, draw
)

# Histo 3
_histo_l_1_a = [1, 1, 1]
_histo_l_2_a = [1, 2, 1]
_histo_l_3_a = [3, 2, 4]


# Histo 2x3
_histo_m_1_a = [_histo_l_1_a, _histo_l_1_a]
_histo_m_2_a = [_histo_l_2_a, _histo_l_3_a]


# Histo field 2x3 over 4x2
_histo_field_1_a = [
    [_histo_m_1_a, _histo_m_1_a],
    [_histo_m_1_a, _histo_m_2_a],
    [_histo_m_2_a, _histo_m_1_a],
    [_histo_m_2_a, _histo_m_2_a]]


_histo_field_1_norms = [
    [6, 6],
    [6, 13],
    [13, 6],
    [13, 13]]


def _H(m, base_rank: int = 0) -> Histo:
    return Histo.from_iterable(m, base_rank)


def _D(m, base_rank: int = 0) -> Distro:
    return Distro.from_iterable(m, base_rank)


def test_histo_construction():
    assert_equal(Histo.uniform(3), _H([1., 1., 1.]))
    assert_equal(Histo.uniform(3, 0.), _H([0., 0., 0.]))
    assert_equal(Histo.dirac(3), _H([1., 0., 0.]))
    assert_equal(Histo.dirac(3, 1), _H([0., 1., 0.]))


def test_fiber_size():
    assert _H(_histo_m_1_a).fiber_size() == 6
    assert _H(_histo_field_1_a, 2).fiber_size() == 6
    assert _H(_histo_field_1_a).fiber_size() == 48


def test_distro_construction():
    assert_equal(Distro.uniform(4), Distro.from_iterable([1., 1., 1., 1.]))
    assert_equal(Distro.dirac(3), Distro.from_iterable([1., 0., 0.]))
    assert_equal(Distro.dirac(3, 1), Distro.from_iterable([0., 1., 0.]))
    assert_equal(Distro.uniform(2), np.array([0.5, 0.5]))


def test_norm():
    assert_equal(norm(_H(_histo_l_1_a)), np.array([3]))
    assert_equal(norm(_H(_histo_l_2_a)), np.array([4]))
    assert_equal(norm(_H(_histo_l_3_a)), np.array([9]))
    assert_equal(norm(_H(_histo_m_1_a)), np.array([6]))
    assert_equal(norm(_H(_histo_m_2_a)), np.array([13]))
    assert_equal(norm(_H(_histo_field_1_a, 2)), np.array(_histo_field_1_norms))


def test_normalize():
    assert_equal(normalize(Histo.uniform(1)), np.array([1.]))
    assert_equal(normalize(Histo.uniform(2)), np.array([0.5, 0.5]))

    normed_1 = [0.333, 0.333, 0.333]
    normed_2 = [0.25, 0.5, 0.25]
    normed_3 = [0.333, 0.222, 0.444]
    normed_m_1 = [[0.167, 0.167, 0.167], [0.167, 0.167, 0.167]]
    normed_m_2 = [[0.077, 0.154, 0.077], [0.231, 0.154, 0.308]]

    assert_allclose(normalize(_H(_histo_l_1_a)), np.array(normed_1), atol=0.001)
    assert_allclose(normalize(_H(_histo_l_2_a)), np.array(normed_2), atol=0.001)
    assert_allclose(normalize(_H(_histo_l_3_a)), np.array(normed_3), atol=0.001)
    assert_allclose(normalize(_H(_histo_m_1_a)), np.array(normed_m_1), atol=0.001)
    assert_allclose(normalize(_H(_histo_m_2_a)), np.array(normed_m_2), atol=0.001)
    assert_allclose(normalize(_H(_histo_field_1_a, 2)), np.array([
        [normed_m_1, normed_m_1],
        [normed_m_1, normed_m_2],
        [normed_m_2, normed_m_1],
        [normed_m_2, normed_m_2]]),
        atol=0.001)


def test_information():
    assert_equal(information(Distro.uniform(2)), np.array([1., 1.]))
    assert_equal(information(Distro.uniform(4)), np.array([2., 2., 2., 2.]))
    assert_equal(information(Distro.dirac(3)), np.array([0., 0., 0.]))


def test_entropy_special_cases():
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
        assert entropy(_D(dist)) == ent_val


def test_histro_entropy():
    assert_allclose(entropy(_H(_histo_l_1_a)), np.array([1.585]), atol=0.001)
    assert_allclose(entropy(_H(_histo_l_2_a)), np.array([1.5]), atol=0.001)
    assert_allclose(entropy(_H(_histo_l_3_a)), np.array([1.530]), atol=0.001)
    assert_allclose(entropy(_H(_histo_m_1_a)), np.array([2.585]), atol=0.001)
    assert_allclose(entropy(_H(_histo_m_2_a)), np.array([2.412]), atol=0.001)
    assert_allclose(entropy(_H(_histo_field_1_a, 2)),
            np.array([[2.585, 2.585], [2.585, 2.412], [2.412, 2.585], [2.412, 2.412]]),
            atol=0.001)


def test_distro_entropy():
    assert_allclose(entropy(_D(_histo_l_1_a)), np.array([1.585]), atol=0.001)
    assert_allclose(entropy(_D(_histo_l_2_a)), np.array([1.5]), atol=0.001)
    assert_allclose(entropy(_D(_histo_l_3_a)), np.array([1.530]), atol=0.001)
    assert_allclose(entropy(_D(_histo_m_1_a)), np.array([2.585]), atol=0.001)
    assert_allclose(entropy(_D(_histo_m_2_a)), np.array([2.412]), atol=0.001)
    assert_allclose(entropy(_D(_histo_field_1_a, 2)),
            np.array([[2.585, 2.585], [2.585, 2.412], [2.412, 2.585], [2.412, 2.412]]),
            atol=0.001)


def test_entropy_density_special_cases():
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
        assert_allclose(relative_entropy(_D(dist)), ent_val, 0.00001)


def test_histro_entropy_density():
    assert_allclose(relative_entropy(_H(_histo_l_1_a)), np.array([1]), atol=0.001)
    assert_allclose(relative_entropy(_H(_histo_l_2_a)), np.array([0.946]), atol=0.001)
    assert_allclose(relative_entropy(_H(_histo_l_3_a)), np.array([0.966]), atol=0.001)
    assert_allclose(relative_entropy(_H(_histo_m_1_a)), np.array([1]), atol=0.001)
    assert_allclose(relative_entropy(_H(_histo_m_2_a)), np.array([0.933]), atol=0.001)
    assert_allclose(relative_entropy(_H(_histo_field_1_a, 2)),
            np.array([[1, 1], [1, 0.933], [0.933, 1], [0.933, 0.933]]),
            atol=0.001)


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
    test_field = _D([
        [[4., 2.], [1., 1.]],
        [[1., 1.], [1., 1.]],
        [[1., 2.], [1., 4.]],
    ], base_rank=1)
    assert_allclose(relative_entropy(test_field),
            np.array([0.875, 1, 0.875]), atol=0.00001)


def test_draw():
    for shape, index in (
        ((5,), (3,)),
        ((5, 4), (3, 2)),
        ((5, 4, 2), (3, 2, 0)),
    ):
        distro = Distro.dirac(shape, index)
        assert draw(distro) == index
