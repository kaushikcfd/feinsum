"""
.. currentmodule:: feinsum.einsum

.. autoclass:: FusedEinsum
.. autoclass:: VeryLongAxis
.. autoclass:: EinsumAxisAccess
.. autoclass:: FreeAxis
.. autoclass:: SummationAxis
.. autoclass:: Argument
.. autoclass:: EinsumOperand
.. autoclass:: IntermediateResult
.. autoclass:: ContractionSchedule
.. autoclass:: SizeParam


Helper routines
^^^^^^^^^^^^^^^

.. autofunction:: construct_subscripts_from_einsum
.. autofunction:: get_trivial_contract_schedule
.. autofunction:: contraction_schedule_from_opt_einsum
"""


from __future__ import annotations

import abc
import numpy as np

from pyrsistent.typing import PMap as PMapT
from pyrsistent import pmap
from typing import Union, Tuple, Any, FrozenSet, TYPE_CHECKING, List
from dataclasses import dataclass
from functools import cached_property, cache
from more_itertools import zip_equal as zip
from pytools import UniqueNameGenerator


IntegralT = Union[int, np.int8, np.int16, np.int32, np.int64, np.uint8,
                  np.uint16, np.uint32, np.uint64]
INT_CLASSES = (int, np.int8, np.int16, np.int32, np.int64, np.uint8,
               np.uint16, np.uint32, np.uint64)


ShapeComponentT = Union[IntegralT, "SizeParam"]
ShapeT = Tuple[ShapeComponentT, ...]


if TYPE_CHECKING:
    from opt_einsum.contract import PathInfo


@dataclass(frozen=True, eq=True, repr=True)
class VeryLongAxis:
    """
    Describes a dimension length which is to be assumed to be very large.
    """
    # TODO: Record the threshold over which an axis could be considered as
    # "VeryLong."


@dataclass(frozen=True, eq=True, repr=True)
class SizeParam:
    name: str


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
    .. automethod:: get_subscripts
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
                                       self.access_descriptors):
            for dim, index in zip(arg_shape, arg_axes):
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

    @cache
    def get_subscripts(self) -> str:
        return (",".join("".join(self.index_names[axis]
                                 for axis in axes)
                        for axes in self.access_descriptors)
                + "->"
                + "".join(self.index_names[FreeAxis(i)]
                          for i in range(self.ndim))
                )

    def copy(self, **kwargs: Any) -> FusedEinsum:
        from dataclasses import replace
        return replace(self, **kwargs)


class Argument(abc.ABC):
    """
    An abstract class denoting an argument to an einsum in
    :class:`ContractionSchedule`. See :attr:`ContractionSchedule.arguments`.
    """


@dataclass(frozen=True, eq=True, repr=True)
class IntermediateResult(Argument):
    """
    An :class:`Argument` representing an intermediate result available during
    the current contraction.
    """
    name: str


@dataclass(frozen=True, eq=True, repr=True)
class EinsumOperand(Argument):
    """
    An :class:`Argument` representing the *ioperand*-th argument that was
    passed to the parent einsum whose :class:`ContractionSchedule` is being
    specified.
    """
    ioperand: int


@dataclass(frozen=True, eq=True, repr=True)
class ContractionSchedule:
    """
    Records the schedule in which contractions are to be performed in an einsum
    as a series of einsums with the i-th einsum having subscript
    ``subscript[i]`` operating on ``arguments[i]`` and writing its result to
    ``result_names[i]``.

    .. attribute:: result_names

        Names of the result generated by each

    .. attribute:: arguments

       A :class:`tuple` containing :class:`tuple` of :class:`` for each
       contraction in the schedule.

    .. attribute:: nsteps
    """
    subscripts: Tuple[str, ...]
    result_names: Tuple[str, ...]
    arguments: Tuple[Tuple[Argument, ...], ...]

    def __post_init__(self) -> None:
        assert len(self.subscripts) == len(self.result_names) == len(self.arguments)

    @property
    def nsteps(self) -> int:
        """
        Returns the number of steps involved in scheduling the einsum.
        """
        return len(self.subscripts)

    def copy(self, **kwargs: Any) -> ContractionSchedule:
        from dataclasses import replace
        return replace(self, **kwargs)


def contraction_schedule_from_opt_einsum(path: "PathInfo") -> ContractionSchedule:
    current_args: List[Argument] = [
        EinsumOperand(i)
        for i in range(path.input_subscripts.count(",") + 1)]
    vng = UniqueNameGenerator()

    subscripts: List[str] = []
    result_names: List[str] = []
    arguments: List[Tuple[Argument, ...]] = []
    for contraction in path.contraction_list:
        arg_indices, _, subscript, _, _ = contraction
        arguments.append(tuple(current_args[idx]
                               for idx in arg_indices))
        subscripts.append(subscript)
        result_names.append(vng("_fe_tmp"))
        current_args = ([arg
                         for idx, arg in enumerate(current_args)
                         if idx not in arg_indices]
                        + [IntermediateResult(result_names[-1])])

    assert len(current_args) == 1
    result_names[-1] = vng("_fe_out")

    return ContractionSchedule(tuple(subscripts),
                               tuple(result_names),
                               tuple(arguments))


def construct_subscripts_from_einsum(einsum: FusedEinsum) -> str:
    """
    Reconstruct the subscripts used in the building the *einsum* from it.
    """
    input_subscripts = ",".join("".join(einsum.index_names[axis]
                                        for axis in axes)
                                for axes in einsum.access_descriptors)
    output_subscripts = "".join(einsum.index_names[FreeAxis(i)]
                                for i in range(einsum.ndim))
    return f"{input_subscripts}->{output_subscripts}"


def get_trivial_contract_schedule(einsum: FusedEinsum) -> ContractionSchedule:
    """
    Returns the :class:`ContractionSchedule` for *einsum* scheduled as a single
    contraction.
    """
    return ContractionSchedule((construct_subscripts_from_einsum(einsum),),
                               ("_fe_out",),
                               (tuple(EinsumOperand(i)
                                      for i, _ in enumerate(einsum.arg_shapes)),)
                               )
