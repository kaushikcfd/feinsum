"""
.. currentmodule:: feinsum.einsum

.. autoclass:: BatchedEinsum
.. autoclass:: EinsumAxisAccess
.. autoclass:: FreeAxis
.. autoclass:: SummationAxis
.. autoclass:: Argument
.. autoclass:: Array
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
from typing import Any, Self, cast

import numpy as np
from immutables import Map
from pytools import UniqueNameGenerator, memoize_method

IntegralT = int | np.integer
INT_CLASSES = (int, np.integer)


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


@dataclass(frozen=True, eq=True, repr=True)
class Array:
    name: str
    shape: ShapeT
    dtype: np.dtype[Any]

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def copy(
        self,
        *,
        name: str | None = None,
        shape: ShapeT | None = None,
        dtype: np.dtype[Any] | None = None,
    ) -> Self:
        from dataclasses import replace

        return replace(
            self,
            name=self.name if name is None else name,
            shape=self.shape if shape is None else shape,
            dtype=self.dtype if dtype is None else dtype,
        )


@dataclass(frozen=True)
class EinsumAxisAccess:
    """
    Base class for axis access types in an einsum expression.
    """

    def __init__(self) -> None:
        if type(self) is EinsumAxisAccess:
            raise TypeError(
                "EinsumAxisAccess is abstract and cannot be instantiated directly"
            )


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

    .. attribute:: output_indices
    .. attribute:: input_indices
    .. attribute:: args
    """

    out_idx_set: tuple[str, ...]
    in_idx_sets: tuple[tuple[str, ...], ...]
    args: tuple[tuple[Array, ...], ...]

    def __post_init__(self) -> None:
        from functools import reduce

        assert all(
            len(idx) == 1 and idx.islower() for idx in self.out_idx_set
        ), "Obtained invalid output index (RHS of ->)."
        assert all(
            all(len(idx) == 1 and idx.islower() for idx in in_idx_set)
            for in_idx_set in self.in_idx_sets
        ), "Obtained invalid input index (LHS of ->)."
        all_in_indices = reduce(
            frozenset.union,
            (frozenset(idx_set) for idx_set in self.in_idx_sets),
            cast("frozenset[str]", frozenset()),
        )
        assert (
            frozenset(self.out_idx_set) <= all_in_indices
        ), "Obtained an out index which is not present in the input indices."

        assert all(
            len(arg_row) == len(self.in_idx_sets) for arg_row in self.args
        ), "Mismatch in #operands between subscript expression and input arrays."
        assert all(
            all(
                arg.ndim == len(idx_set)
                for arg, idx_set in zip(arg_row, self.in_idx_sets, strict=True)
            )
            for arg_row in self.args
        ), "Dimensionality of input operands do no match the provided subscripts."

        _ = self.arg_to_dtype
        _ = self.arg_to_shape
        _ = self.index_to_dim_length
        assert (
            len(self.all_args) + len(self.all_indices) + len(self.all_size_params)
        ) == len(
            self.all_args | self.all_indices | {p.name for p in self.all_size_params}
        ), "Must use different names for arguments, indices, and size params."

    @cached_property
    def b(self) -> int:
        """
        Returns the number of batches in the batched einsum.
        """
        return len(self.args)

    @cached_property
    def n(self) -> int:
        """
        Return the number of operands of each einsum in the batched einsum.
        """
        return len(self.in_idx_sets)

    @cached_property
    def index_to_dim_length(self) -> Map[str, ShapeComponentT]:
        index_to_dim: dict[str, ShapeComponentT] = {}
        for arg_row in self.args:
            for arg, idx_set in zip(arg_row, self.in_idx_sets, strict=True):
                for axis_len, idx in zip(arg.shape, idx_set, strict=True):
                    assert (
                        index_to_dim.setdefault(idx, axis_len) == axis_len
                    ), "Shape mismatch for indices across the arguments."

        return Map(index_to_dim)

    @cached_property
    def shape(self) -> ShapeT:
        """
        Returns the shape of an output of the batched einsum.
        """
        return tuple(self.index_to_dim_length[idx] for idx in self.out_idx_set)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @memoize_method
    def get_subscripts(self) -> str:
        """
        Returns the subscripts used in the building the *einsum* from it.
        """
        joined_input_induces = ["".join(idx_set) for idx_set in self.in_idx_sets]
        return f"{','.join(joined_input_induces)} -> {''.join(self.out_idx_set)}"

    @cached_property
    def arg_to_shape(self) -> Map[str, ShapeT]:
        result: dict[str, ShapeT] = {}
        for arg_row in self.args:
            for arg in arg_row:
                assert (
                    result.setdefault(arg.name, arg.shape) == arg.shape
                ), f"Inconsistent shapes for arg {arg.name}."
        return Map(result)

    @cached_property
    def arg_to_dtype(self) -> Map[str, np.dtype[Any]]:
        result: dict[str, np.dtype[Any]] = {}
        for arg_row in self.args:
            for arg in arg_row:
                assert (
                    result.setdefault(arg.name, arg.dtype) == arg.dtype
                ), f"Inconsistent dtypes for arg {arg.name}."
        return Map(result)

    @cached_property
    def index_to_access_descr(self) -> Map[str, EinsumAxisAccess]:
        result: dict[str, EinsumAxisAccess] = {}
        for i, idx in enumerate(self.out_idx_set):
            result[idx] = FreeAxis(i)

        i_redn = 0
        for idx_set in self.in_idx_sets:
            for idx in idx_set:
                if idx not in result:
                    result[idx] = SummationAxis(i_redn)
                    i_redn += 1
        return Map(result)

    @cached_property
    def sum_indices(self) -> tuple[str, ...]:
        all_sum_indices = {
            idx: access.index
            for idx, access in self.index_to_access_descr.items()
            if isinstance(access, SummationAxis)
        }
        return tuple(sorted(all_sum_indices, key=lambda x: all_sum_indices[x]))

    @cached_property
    def all_args(self) -> frozenset[str]:
        return frozenset(self.arg_to_shape)

    @cached_property
    def all_indices(self) -> frozenset[str]:
        return frozenset(self.index_to_dim_length)

    @cached_property
    def all_size_params(self) -> frozenset[SizeParam]:
        return frozenset(
            v for v in self.index_to_dim_length.values() if isinstance(v, SizeParam)
        )

    def copy(self, **kwargs: Any) -> BatchedEinsum:
        from dataclasses import replace

        return replace(self, **kwargs)


class Argument:
    """
    An abstract class denoting an argument to an einsum in
    :class:`ContractionSchedule`. See :attr:`ContractionSchedule.arguments`.
    """

    def __init__(self) -> None:
        if type(self) is Argument:
            raise TypeError(
                "Argument is abstract and cannot be instantiated directly."
            )


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
        (tuple(EinsumOperand(i) for i in range(einsum.n)),),
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

    long_dim_length = opt_einsum_kwargs.pop("long_dim_length", 1_000_000)

    if "optimize" not in opt_einsum_kwargs:
        opt_einsum_kwargs["optimize"] = "optimal"

    if "use_blas" not in opt_einsum_kwargs:
        opt_einsum_kwargs["use_blas"] = False

    _, path = opt_einsum.contract_path(
        expr.get_subscripts(),
        *[
            arg.copy(
                shape=tuple(
                    long_dim_length if isinstance(dim, SizeParam) else dim
                    for dim in arg.shape
                )
            )
            for arg in expr.args[0]
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
