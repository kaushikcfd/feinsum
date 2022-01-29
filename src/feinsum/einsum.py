"""
.. currentmodule:: feinsum.einsum

.. autoclass:: FusedEinsum
.. autoclass:: VeryLongAxis
.. autoclass:: EinsumAxisAccess
.. autoclass:: FreeAxis
.. autoclass:: SummationAxis
"""


from __future__ import annotations

import abc
import numpy as np

from pyrsistent.typing import PMap as PMapT
from pyrsistent import pmap
from typing import Union, Tuple, Any, FrozenSet
from dataclasses import dataclass, field
from functools import cached_property, cache


IntegralT = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8,
                  np.uint16, np.uint32, np.uint64]
INT_CLASSES = (int, np.int8, np.int16, np.int32, np.int64, np.uint8,
               np.uint16, np.uint32, np.uint64)


ShapeComponentT = Union[IntegralT, "VeryLongAxis"]
ShapeT = Tuple[ShapeComponentT, ...]


@dataclass(frozen=True, eq=True, repr=True)
class VeryLongAxis:
    """
    Describes a dimension length which is to be assumed to be very large.
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
    index: int


@dataclass(frozen=True, eq=True, repr=True)
class FusedEinsum:
    """
    A fused einsum expression.

    .. attribute:: shape
    .. attribute:: ndim
    .. automethod:: index_to_dim_length
    """
    arg_shapes: Tuple[ShapeT, ...]
    value_to_dtype: PMapT[str, np.dtype[Any]]
    access_descriptors: Tuple[Tuple[EinsumAxisAccess, ...], ...]
    use_matrix: Tuple[Tuple[FrozenSet[str], ...]]
    index_names: PMapT[EinsumAxisAccess, str]

    @property
    def noutputs(self) -> int:
        return len(self.use_matrix)

    @cache
    def index_to_dim_length(self) -> PMapT[EinsumAxisAccess, ShapeComponentT]:
        index_to_dim = {}
        for arg_shape, arg_axes in zip(self.arg_shapes,
                                       self.access_descriptors,
                                       strict=True):
            for dim, index in zip(arg_shape, arg_axes,
                                  strict=True):
                if dim not in index_to_dim:
                    index_to_dim[index] = dim
                else:
                    assert dim == index_to_dim[index]

        return pmap(index_to_dim)

    @cached_property
    def shape(self) -> ShapeT:
        free_index_to_dim = {idx: dim
                             for idx, dim in self.index_to_dim_length().items()
                             if isinstance(idx, FreeAxis)}
        assert all(FreeAxis(idim) in free_index_to_dim
                   for idim in range(len(free_index_to_dim)))

        return tuple(dim
                     for _, dim in sorted(free_index_to_dim.items(),
                                          key=lambda x: x[0].output_index))

    @property
    def ndim(self) -> int:
        return len(self.shape)
