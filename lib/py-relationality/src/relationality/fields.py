"""Histogram and distribution fields.
    The Histo and Distro classes are respectively for histogram and
    distribution fields. A simple histogram or distribution is just an
    array of values, the latter being normalized to sum to 1. A field is
    an array where the first "base_rank" dimensions index a base space.
    Attached to each point of this space is a histogram or distribution.
    In the case of the distribution, at each point in the base space the
    values add up to 1.
"""
from typing import TypeVar, Type, Iterable, Tuple, Any
import numpy as np
import numpy.random as nprand
from functools import reduce
from operator import mul

from relationality.util import Shape, dims, shape_lift, Indexer

T = TypeVar('T', bound='Histo')


class Histo(np.ndarray):
    """Histogram of arbitrary shape.
        If the base_rank is set > 0, the first base_rank dimensions of the
        array form a base space, and the rest the dimensions of the histo
        itself as a fiber.
    """
    base_rank: int = 0

    @classmethod
    def from_array(cls: Type[T], a: np.ndarray, base_rank: int = 0) -> T:
        """Downcast from ndarray."""
        if dims(a) < base_rank:
            raise ValueError("base_rank larger than dimensions of array")
        histo = a.view(cls)
        histo.base_rank = base_rank
        return histo

    @classmethod
    def from_iterable(cls: Type[T], a: Iterable, base_rank: int = 0) -> T:
        """Downcast from ndarray."""
        return cls.from_array(np.array(a), base_rank)

    @classmethod
    def uniform(cls: Type[T], shape: Shape, value: float = 1, base_rank: int = 0) -> T:
        """Uniform histogram of constant value."""
        if dims(shape) < base_rank:
            raise ValueError("base_rank larger than given shape")
        return cls.from_array(np.full(shape, value), base_rank)

    @classmethod
    def dirac(cls: Type[T], shape: Shape, index: Shape = 0, value: float = 1) -> T:
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
    
    def base_axis(self):
        """Get the axes for the base."""
        return tuple(range(0, self.base_rank))

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

    def fiber_size(self):
        return reduce(mul, self.shape[self.base_rank:], 1)

def norm(histo: Histo) -> np.ndarray:
    """Norm of histogram."""
    return np.sum(histo, axis=histo.fiber_axis())


class Distro(Histo):
    """Distribution of arbitrary shape."""
    @classmethod
    def from_array(cls: Type[T], a: np.ndarray, base_rank: int = 0) -> T:
        """Downcast from ndarray."""
        histo = Histo.from_array(a, base_rank)
        normer = np.reshape(np.repeat(norm(histo), histo.fiber_size()), histo.shape)
        distro = (histo / normer).view(cls)
        distro.base_rank = histo.base_rank
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
    return Distro.from_array(histo, histo.base_rank)


def magnitude(histo: Histo) -> Histo:
    """Shannon magnitude of histo in nats."""
    under = np.log2(histo, where=(histo > 0.))
    return Histo.from_array(under, base_rank=histo.base_rank)


def information(distro: Distro) -> Histo:
    """Shannon information of distro in nats."""
    return Histo.from_array(-magnitude(distro), base_rank=distro.base_rank)


def entropy(histo: Histo) -> np.ndarray:
    """Shannon entropy of histo in average nats."""
    if len(histo) == 1:
        return np.array(0.)
    
    direct = -np.sum(magnitude(histo)*histo, axis=histo.fiber_axis())

    if isinstance(histo, Distro):
        return direct

    normed = norm(histo)
    return direct/normed + np.log2(normed)


def relative_entropy(histo: Histo) -> np.ndarray:
    """Entropy as a ratio with maximum entropy for size.
        Will be between 0 and 1.
        0 indicates certitude
        1 indicates total equivocality
    """
    size = histo.fiber_size()
    if size <= 1:
        return np.array(0.)

    max_entropy = np.log2(float(size))
    return entropy(histo)/max_entropy


def accumulate(distro: Distro) -> Cumulate:
    """Derive cumulative field from distro field."""
    return Cumulate.from_distro(distro)


def draw(distro: Distro) -> Tuple[Any, ...]:
    """Draw randomly from a distro.
        The typing on this is messy.
    """
    if distro.base_rank > 0:
        raise NotImplementedError(
            "Only single distros are currently supported. "
            "No field support yet."
        )
    flat_index = nprand.choice(a=distro.size, p=distro.flat)
    return np.unravel_index(flat_index, distro.shape)
