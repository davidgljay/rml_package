"""Histogram and distribution fields.
    The Histo and Distro classes are respectively for histogram and
    distribution fields. A simple histogram or distribution is just an
    array of values, the latter being normalized to sum to 1. A field is
    an array where the first "base_rank" dimensions index a base space.
    Attached to each point of this space is a histogram or distribution.
    In the case of the distribution, at each point in the base space the
    values add up to 1.
"""
from typing import Iterable
import numpy as np  # type: ignore
import numpy.random as nprand  # type: ignore

from relationality.util import Shape, dims, shape_lift, Indexer


class Histo(np.ndarray):
    """Histogram of arbitrary shape.
        If the base_rank is set > 0, the first base_rank dimensions of the
        array form a base space, and the rest the dimensions of the histo
        itself as a fiber.
    """
    base_rank: int = 0

    @classmethod
    def from_array(cls, a: np.ndarray, base_rank: int = 0):
        """Downcast from ndarray."""
        if dims(a) < base_rank:
            raise ValueError("base_rank larger than dimensions of array")
        histo = a.view(cls)
        histo.base_rank = base_rank
        return histo

    @classmethod
    def from_iterable(cls, a: Iterable, base_rank: int = 0):
        """Downcast from ndarray."""
        return cls.from_array(np.array(a), base_rank)

    @classmethod
    def uniform(
        cls, shape: Shape, value: float = 1, base_rank: int = 0
    ) -> 'Histo':
        """Uniform histogram of constant value."""
        if dims(shape) < base_rank:
            raise ValueError("base_rank larger than given shape")
        return cls.from_array(np.full(shape, value), base_rank)

    @classmethod
    def dirac(
        cls, shape: Shape, index: Shape = 0, value: float = 1
    ) -> 'Histo':
        """Histo with only one non-zero value."""
        rank, fiber_rank = dims(shape), dims(index)
        if rank < fiber_rank:
            raise ValueError("index has more dims than given shape")
        base_rank = rank - fiber_rank
        under = np.zeros(shape)
        indexer = (Ellipsis,) + shape_lift(index)
        under[indexer] = value
        return cls.from_array(under, base_rank)

    def fiber_axis(self):
        """Get the axes for the fiber."""
        return tuple(range(self.base_rank, dims(self)))

    def point(self, index: Indexer):
        """Index point from the base space."""
        rank = dims(index)
        if rank > self.base_rank:
            raise ValueError("rank of index is bigger than base")
        seg = self[index]
        seg.base_rank = self.base_rank - rank
        return seg

    def update(self, updater: 'Histo'):
        """Multiplicative update for field."""
        if self.shape != updater.shape:
            raise ValueError("updater shape does not match the histo")
        if self.base_rank != updater.base_rank:
            raise ValueError("updater base_rank does not match the histo")
        return self.__class__.from_array(self*updater, base_rank=self.base_rank)

    def update_at(self, updater: 'Histo', index: Indexer):
        """Multiplicative in-place update for field at a point."""
        subfield = self.point(index)
        self[index] = subfield.update(updater)


class Distro(Histo):
    """Distribution of arbitrary shape."""
    @classmethod
    def from_array(Self, a: np.ndarray, base_rank: int = 0) -> 'Distro':
        """Downcast from ndarray."""
        rank = dims(a)
        if rank < base_rank:
            raise ValueError("base_rank larger than dimensions of array")
        axis = tuple(range(base_rank, rank))
        norms = np.sum(a, axis=axis, keepdims=True)
        distro = (a / norms).view(Self)
        distro.base_rank = base_rank
        return distro


class Cumulate(np.ndarray):
    """Cumulative distribution field derived from distribution."""
    base_rank: int = 0

    @classmethod
    def from_distro(cls, distro: Distro):
        """Derive from distro."""
        unwound = distro.reshape(distro.shape[:distro.base_rank] + (-1,))
        cumu = np.cumsum(unwound, axis=distro.base_rank).view(cls)
        cumu.base_rank = distro.base_rank
        return cumu


def normalize(histo: Histo) -> Distro:
    """Normalizes histogram into distribution."""
    return Distro.from_array(histo)


def information(distro: Distro) -> Histo:
    """Shannon information of distro in nats."""
    under = -np.log2(distro, where=(distro > 0.))
    return Histo.from_array(under, base_rank=distro.base_rank)


def entropy(distro: Distro) -> np.ndarray:
    """Shannon entropy of distro in average nats."""
    if len(distro) == 1:
        return 0.0
    neg_info = np.log2(distro, where=(distro > 0.))
    return -np.sum(neg_info*distro, axis=distro.fiber_axis())


def relative_entropy(distro: Distro) -> np.ndarray:
    """Entropy as a ratio with maximum entropy for size.
        Will be between 0 and 1.
        0 indicates certitude
        1 indicates total equivocality
    """
    size = len(distro)
    if size <= 1:
        return 0.
    max_entropy = np.log2(float(size))
    return entropy(distro)/max_entropy


def accumulate(distro: Distro) -> Cumulate:
    """Derive cumulative field from distro field."""
    return Cumulate.from_distro(distro)


def draw(distro: Distro) -> Shape:
    """Draw randomly from a distro."""
    if distro.base_rank > 0:
        raise NotImplementedError(
            "Only single distros are currently supported. "
            "No field support yet."
        )
    flat_index = nprand.choice(a=distro.size, p=distro.flat)
    return np.unravel_index(flat_index, distro.shape)
