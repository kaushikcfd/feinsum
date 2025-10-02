"""
.. currentmodule:: feinsum.einsum

.. autoclass:: BatchedEinsum
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

.. autofunction:: get_trivial_contraction_schedule
.. autofunction:: get_opt_einsum_contraction_schedule
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from immutables import Map
from more_itertools import zip_equal as zip
from pytools import UniqueNameGenerator, memoize_method

IntegralT = int | np.integer
ScalarT = np.number | int, np.bool_ | bool | float | complex
INT_CLASSES = (int, np.integer)
SCALAR_CLASSES = (np.number, int, np.bool_, bool, float, complex)


@dataclass(frozen=True, eq=True, repr=True)
class VeryLongAxis:
    """
    Describes a dimension length which is to be assumed to be very large.
    """

    # TODO: Record the threshold over which an axis could be considered as
    # "VeryLong."


@dataclass(frozen=True)
class SizeParam:
    name: str

    def __truediv__(self, other: IntegralT | SizeParam) -> IntegralT:
        # only present to help writing easier param-getters in the tuner
        # implementations.
        return NotImplemented

    __rtruediv__ = __truediv__


ShapeComponentT = IntegralT | SizeParam
ShapeT = tuple[ShapeComponentT, ...]


@dataclass(frozen=True)
class EinsumAxisAccess:
    """
    Base class for axis access types in an einsum expression.
    """

    def __new__(cls, *args, **kwargs):
        if cls is EinsumAxisAccess:
            raise TypeError(
                "EinsumAxisAccess is abstract and cannot be instantiated directly"
            )
        return super().__new__(cls)


@dataclass(frozen=True)
class FreeAxis(EinsumAxisAccess):
    """
    Records the axis of an einsum argument over which contraction is not performed.

    .. attribute:: output_index

        Position of the corresponding index in the einsum's output.
    """

    output_index: int


@dataclass(frozen=True)
class SummationAxis(EinsumAxisAccess):
    """
    Records an index in an einsum expression over which reduction is performed.
    Sometimes also referred to as an axis with a corresponding "dummy index" in
    Ricci Calculus.

    .. attribute:: index

        An integer which is unique to a reduction index of an einsum.
    """

    index: int


@dataclass(frozen=True)
class BatchedEinsum:
    """
    A batched einsum expression.

    .. attribute:: shape
    .. attribute:: ndim
    .. automethod:: index_to_dim_length
    .. automethod:: get_subscripts
    .. automethod:: get_arg_shape
    """

    arg_shapes: tuple[ShapeT, ...]
    value_to_dtype: Map[str, np.dtype[Any]]
    access_descriptors: tuple[tuple[EinsumAxisAccess, ...], ...]
    use_matrix: tuple[tuple[frozenset[str], ...], ...]
    index_names: Map[EinsumAxisAccess, str]

    @property
    def noutputs(self) -> int:
        return len(self.use_matrix)

    @memoize_method
    def index_to_dim_length(self) -> Map[EinsumAxisAccess, ShapeComponentT]:
        index_to_dim = {}
        for arg_shape, arg_axes in zip(self.arg_shapes, self.access_descriptors):
            for dim, index in zip(arg_shape, arg_axes):
                if dim not in index_to_dim:
                    index_to_dim[index] = dim
                else:
                    assert dim == index_to_dim[index]

        return Map(index_to_dim)

    @cached_property
    def shape(self) -> ShapeT:
        free_index_to_dim = {
            idx: dim
            for idx, dim in self.index_to_dim_length().items()
            if isinstance(idx, FreeAxis)
        }
        assert all(
            FreeAxis(idim) in free_index_to_dim
            for idim in range(len(free_index_to_dim))
        )

        return tuple(
            dim
            for _, dim in sorted(
                free_index_to_dim.items(), key=lambda x: x[0].output_index
            )
        )

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @memoize_method
    def get_subscripts(self) -> str:
        """
        Returns the subscripts used in the building the *einsum* from it.
        """
        return (
            ",".join(
                "".join(self.index_names[axis] for axis in axes)
                for axes in self.access_descriptors
            )
            + "->"
            + "".join(self.index_names[FreeAxis(i)] for i in range(self.ndim))
        )

    @memoize_method
    def get_arg_shape(self, name: str) -> ShapeT:
        """
        Returns the shape for argument named *name*.
        """
        for argument_row in self.use_matrix:
            for arguments, access_descrs in zip(
                argument_row, self.access_descriptors
            ):
                if name in arguments:
                    return tuple(
                        self.index_to_dim_length()[acc_descr]
                        for acc_descr in access_descrs
                    )

        raise ValueError(f"'{name}' is not one of the arguments.")

    def copy(self, **kwargs: Any) -> BatchedEinsum:
        from dataclasses import replace

        return replace(self, **kwargs)


class Argument:
    """
    An abstract class denoting an argument to an einsum in
    :class:`ContractionSchedule`. See :attr:`ContractionSchedule.arguments`.
    """

    def __new__(cls, *args, **kwargs):
        if cls is Argument:
            raise TypeError(
                "EinsumAxisAccess is abstract and cannot be instantiated directly"
            )
        return super().__new__(cls)


@dataclass(frozen=True)
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

        Names of the result generated by each step.

    .. attribute:: arguments

       A :class:`tuple` containing :class:`tuple` of :class:`str` for each
       contraction in the schedule.

    .. attribute:: nsteps
    """

    subscripts: tuple[str, ...]
    result_names: tuple[str, ...]
    arguments: tuple[tuple[Argument, ...], ...]

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


def get_trivial_contraction_schedule(einsum: BatchedEinsum) -> ContractionSchedule:
    """
    Returns the :class:`ContractionSchedule` for *einsum* scheduled as a single
    contraction.
    """
    return ContractionSchedule(
        (einsum.get_subscripts(),),
        ("_fe_out",),
        (tuple(EinsumOperand(i) for i, _ in enumerate(einsum.arg_shapes)),),
    )


def get_opt_einsum_contraction_schedule(
    expr: BatchedEinsum,
    **opt_einsum_kwargs: Any,
) -> ContractionSchedule:
    """
    Returns a :class:`ContractionSchedule` as computed by
    :func:`opt_einsum.contract_path`.

    :param opt_einsum_kwargs: kwargs to be passed to
        :func:`opt_einsum.contract_path`.

    .. note::

        The following defaults are populated in *opt_einsum_kwargs*, if left
        unspecified:

        - ``optimize="optimal"``
        - ``use_blas=False``
    """
    import opt_einsum

    from feinsum.make_einsum import array

    long_dim_length = opt_einsum_kwargs.pop("long_dim_length", 1_000_000)

    if "optimize" not in opt_einsum_kwargs:
        opt_einsum_kwargs["optimize"] = "optimal"

    if "use_blas" not in opt_einsum_kwargs:
        opt_einsum_kwargs["use_blas"] = False

    _, path = opt_einsum.contract_path(
        expr.get_subscripts(),
        *[
            array(
                [
                    d if isinstance(op_shape, INT_CLASSES) else long_dim_length
                    for d in op_shape
                ],
                "float64",
            )
            for op_shape in expr.arg_shapes
        ],
        **opt_einsum_kwargs,
    )

    current_args: list[Argument] = [
        EinsumOperand(i) for i in range(path.input_subscripts.count(",") + 1)
    ]
    vng = UniqueNameGenerator()

    subscripts: list[str] = []
    result_names: list[str] = []
    arguments: list[tuple[Argument, ...]] = []
    for contraction in path.contraction_list:
        arg_indices, _, subscript, _, _ = contraction
        arguments.append(tuple(current_args[idx] for idx in arg_indices))
        subscripts.append(subscript)
        result_names.append(vng("_fe_tmp"))
        current_args = [
            arg for idx, arg in enumerate(current_args) if idx not in arg_indices
        ] + [IntermediateResult(result_names[-1])]

    assert len(current_args) == 1
    result_names[-1] = vng("_fe_out")

    return ContractionSchedule(
        tuple(subscripts), tuple(result_names), tuple(arguments)
    )
