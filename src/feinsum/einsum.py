from __future__ import annotations

import abc
import numpy as np

from typing import Union, Tuple, Any, FrozenSet
from dataclasses import dataclass


IntegralT = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8,
                  np.uint16, np.uint32, np.uint64]
INT_CLASSES = (int, np.int8, np.int16, np.int32, np.int64, np.uint8,
               np.uint16, np.uint32, np.uint64)


ShapeComponentT = Union[IntegralT, "VeryLongAxis"]
ShapeT = Tuple[ShapeComponentT, ...]


class VeryLongAxis:
    """
    Describes a shape which can be assumed to be very large.
    """
    # TODO: Record the threshold over which an axis could be considered as
    # "VeryLong."


@dataclass(frozen=True, repr=True, eq=True)
class EinsumAxisAccess(abc.ABC):
    """
    Base class for axis access types in an einsum expression.
    """


@dataclass(frozen=True, repr=True, eq=True)
class FreeAxis(EinsumAxisAccess):
    """
    Records the axis of an einsum argument over which contraction is not performed.

    .. attribute:: output_index

        Position of the corresponding index in the einsum's output.
    """
    output_index: int


@dataclass(frozen=True, repr=True, eq=True)
class SummationAxis(EinsumAxisAccess):
    """
    Records an index in an einsum expression over which reduction is performed.
    Sometimes also referred to as an axis with a corresponding "dummy index" in
    Ricci Calculus.

    .. attribute:: index

        An integer which is unique to a reduction index of an einsum.
    """
    idx: int


@dataclass(frozen=True, eq=True, repr=True)
class Einsum:
    """
    An einsum expression.
    """
    arg_shapes: Tuple[ShapeT, ...]
    arg_dtypes: Tuple[np.dtype[Any], ...]
    access_descriptors: Tuple[Tuple[EinsumAxisAccess, ...], ...]
    use_matrix: Tuple[Tuple[FrozenSet[str], ...]]
