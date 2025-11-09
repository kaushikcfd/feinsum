"""
.. currentmodule:: feinsum.einsum

.. autoclass:: BatchedEinsum
.. autoclass:: EinsumAxisAccess
.. autoclass:: FreeAxis
.. autoclass:: SummationAxis
.. autoclass:: Array
.. autoclass:: SizeParam
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any, Self, cast

import numpy as np
from immutables import Map
from pytools import memoize_method

IntegralT = int | np.integer
INT_CLASSES = (int, np.integer)


@dataclass(frozen=True)
class SizeParam:
    """
    Parametric axis length.

    :attr str: Name of the parameter.
    """

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
    """
    Represents a multidimensional array operand.

    :attr name: Name of the operand.
    :attr shape: Shape of the n-dimension array.
    :attr dtype: Numeric data-type of an element of the array.
    """

    name: str
    shape: ShapeT
    dtype: np.dtype[Any]

    @property
    def ndim(self) -> int:
        """
        Returns the rank (i.e. dimensionality) of the multidimensional array.
        """
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

        A 2-d matrix of :class:`Array` corresponding
        to the array operands.

    .. autoattribute:: b
    .. autoattribute:: n
    .. autoattribute:: shape
    .. autoattribute:: ndim
    .. autoattribute:: arg_to_shape
    .. autoattribute:: arg_to_dtype
    .. autoattribute:: sum_indices
    .. autoattribute:: all_args
    .. autoattribute:: all_size_params

    .. automethod:: get_subscripts
    .. automethod:: copy
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
        """
        Mapping for from the name of the index to its corresponding axis-length.
        """
        index_to_dim: dict[str, ShapeComponentT] = {}
        for arg_row in self.args:
            for arg, idx_set in zip(arg_row, self.in_idx_sets, strict=True):
                for axis_len, idx in zip(arg.shape, idx_set, strict=True):
                    if index_to_dim.setdefault(idx, axis_len) != axis_len:
                        raise AssertionError(
                            "Shape mismatch for indices across the arguments."
                        )

        return Map(index_to_dim)

    @cached_property
    def shape(self) -> ShapeT:
        """
        Returns the shape of an output of the batched einsum.
        """
        return tuple(self.index_to_dim_length[idx] for idx in self.out_idx_set)

    @property
    def ndim(self) -> int:
        """
        Returns the rank (dimensionality) of each output of the batched einsum.
        """
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
        """
        Mapping from an array operand's name to its shape.
        """
        result: dict[str, ShapeT] = {}
        for arg_row in self.args:
            for arg in arg_row:
                if result.setdefault(arg.name, arg.shape) != arg.shape:
                    raise AssertionError(f"Inconsistent shapes for arg {arg.name}.")
        return Map(result)

    @cached_property
    def arg_to_dtype(self) -> Map[str, np.dtype[Any]]:
        """
        Mapping from an array operand's name to its data-type.
        """
        result: dict[str, np.dtype[Any]] = {}
        for arg_row in self.args:
            for arg in arg_row:
                if result.setdefault(arg.name, arg.dtype) != arg.dtype:
                    raise AssertionError(f"Inconsistent dtypes for arg {arg.name}.")
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
        """
        A :class:`tuple` of index names that denote the contraction indices.
        """
        all_sum_indices = {
            idx: access.index
            for idx, access in self.index_to_access_descr.items()
            if isinstance(access, SummationAxis)
        }
        return tuple(sorted(all_sum_indices, key=lambda x: all_sum_indices[x]))

    @cached_property
    def all_args(self) -> frozenset[str]:
        """
        Names of all array operands in the batched einsum.
        """
        return frozenset(self.arg_to_shape)

    @cached_property
    def all_indices(self) -> frozenset[str]:
        """
        Names of indices of the batched einsum.
        """
        return frozenset(self.index_to_dim_length)

    @cached_property
    def all_size_params(self) -> frozenset[SizeParam]:
        """
        All instances of :class:`SizeParam` involved in
        this einsum.
        """
        return frozenset(
            v for v in self.index_to_dim_length.values() if isinstance(v, SizeParam)
        )

    def copy(
        self,
        *,
        out_idx_set: tuple[str, ...] | None = None,
        in_idx_sets: tuple[tuple[str, ...], ...] | None = None,
        args: tuple[tuple[Array, ...], ...] | None = None,
    ) -> BatchedEinsum:
        """
        Returns a copy of *self*.
        """
        from dataclasses import replace

        if out_idx_set is None:
            out_idx_set = self.out_idx_set
        if in_idx_sets is None:
            in_idx_sets = self.in_idx_sets
        if args is None:
            args = self.args

        return replace(
            self, out_idx_set=out_idx_set, in_idx_sets=in_idx_sets, args=args
        )

    @memoize_method
    def __str__(self) -> str:
        from tabulate import tabulate

        from feinsum.codegen.loopy import _get_isl_basic_set

        dtypes = "\n".join(
            f"{arg_name}: {dtype}"
            for arg_name, dtype in sorted(
                self.arg_to_dtype.items(), key=lambda x: x[0]
            )
        )
        output_names = ["_fe_out", *[f"_fe_out_{i}" for i in range(self.b - 1)]]
        joined_sum_idxs = "{" + ", ".join(self.sum_indices) + "}"
        joined_out_idxs = ", ".join(self.out_idx_set)
        statement_lines = [
            [
                " ",
                f"{out_name}[{joined_out_idxs}]",
                "<-",
                f"Σ_{joined_sum_idxs} {'×'.join(arg.name + '[' + ', '.join(in_idx_set) + ']' for in_idx_set, arg in zip(self.in_idx_sets, arg_row, strict=True))}",  # noqa: E501, RUF001
            ]
            for out_name, arg_row in zip(output_names, self.args, strict=True)
        ]
        statements = tabulate(
            statement_lines,
            tablefmt="plain",
            colalign=("left", "right", "left", "left"),
        )

        return f"""---------------------------------------------------------------------------
DOMAINS:
{_get_isl_basic_set(self.index_to_dim_length)}
---------------------------------------------------------------------------
Data-types:
{dtypes}
---------------------------------------------------------------------------
for {','.join(self.out_idx_set)}
{statements}
end
---------------------------------------------------------------------------"""  # noqa: E501
