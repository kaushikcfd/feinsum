"""
.. currentmodule:: feinsum.make_einsum

.. autofunction:: einsum
.. autofunction:: fused_einsum
"""

__copyright__ = """
Copyright (C) 2020 Andreas Kloeckner
Copyright (C) 2020 Matt Wala
Copyright (C) 2020 Xiaoyu Wei
Copyright (C) 2022 Kaushik Kulkarni
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import re
import numpy as np
import numpy.typing as npt

from typing import (Tuple, Protocol, Any, List, Sequence, Optional, Mapping,
                    FrozenSet, Union, Dict)
from dataclasses import dataclass
from pyrsistent import pmap
from pyrsistent.typing import PMap as PMapT
from feinsum.einsum import (FusedEinsum, FreeAxis,
                            SummationAxis, EinsumAxisAccess, VeryLongAxis,
                            INT_CLASSES, IntegralT, SizeParam)
from more_itertools import zip_equal as szip  # strict zip
from pytools import UniqueNameGenerator


ShapeComponentT = Union[VeryLongAxis, IntegralT, SizeParam]
ShapeT = Tuple[ShapeComponentT, ...]


class ArrayT(Protocol):
    """
    A protocol specifying the minimal interface requirements for concrete
    array data.

    .. attribute:: shape
    .. attribute:: dtype
    """

    @property
    def shape(self) -> ShapeT:
        pass

    @property
    def dtype(self) -> np.dtype[Any]:
        pass


@dataclass(frozen=True, eq=True, repr=True)
class Array:
    shape: ShapeT
    dtype: np.dtype[Any]


def _preprocess_component(s: Any) -> ShapeComponentT:
    if isinstance(s, SizeParam):
        return s
    elif (isinstance(s, VeryLongAxis) or np.isposinf(s)):
        return VeryLongAxis()
    elif (isinstance(s, INT_CLASSES) and (s >= 0)):
        return s
    else:
        raise ValueError(f"Cannot infer shape component '{s}'.")


def _preprocess_shape(shape: Any) -> ShapeT:
    from collections.abc import Sequence
    if not isinstance(shape, Sequence):
        shape = shape,

    return tuple(_preprocess_component(d) for d in shape)


def array(shape: Any, dtype: npt.DTypeLike) -> Array:
    return Array(shape=_preprocess_shape(shape), dtype=np.dtype(dtype))


EINSUM_FIRST_INDEX = re.compile(r"^\s*((?P<alpha>[a-zA-Z])|(?P<ellipsis>\.\.\.))\s*")


def _normalize_einsum_out_subscript(subscript: str) -> PMapT[str,
                                                             EinsumAxisAccess]:
    """
    Normalizes the output subscript of an einsum (provided in the explicit
    mode). Returns a mapping from index name to an instance of
    :class:`FreeAxis`.
    .. testsetup::
        >>> from feinsum.make_einsum import _normalize_einsum_out_subscript
    .. doctest::
        >>> result = _normalize_einsum_out_subscript("kij")
        >>> sorted(result.keys())  # sorting to have a deterministic print order
        ['i', 'j', 'k']
        >>> result["i"], result["j"], result["k"]
        (FreeAxis(dim=1), FreeAxis(dim=2), FreeAxis(dim=0))
    """

    normalized_indices: List[str] = []
    acc = subscript.strip()
    while acc:
        match = EINSUM_FIRST_INDEX.match(acc)
        if match:
            if "alpha" in match.groupdict():
                normalized_indices.append(match.groupdict()["alpha"])
            else:
                assert "ellipsis" in match.groupdict()
                raise NotImplementedError("Broadcasting in einsums not supported")
            assert match.span()[0] == 0
            acc = acc[match.span()[-1]:]
        else:
            raise ValueError(f"Cannot parse '{acc}' in provided einsum"
                             f" '{subscript}'.")

    if len(set(normalized_indices)) != len(normalized_indices):
        raise ValueError("Used an input more than once to refer to the"
                         f" output axis in '{subscript}")

    return pmap({idx: FreeAxis(i)
                 for i, idx in enumerate(normalized_indices)})


def _normalize_einsum_in_subscript(subscript: str,
                                   in_operand_shape: ShapeT,
                                   index_to_descr: PMapT[str,
                                                        EinsumAxisAccess],
                                   index_to_axis_length: PMapT[str,
                                                               ShapeComponentT],
                                   ) -> Tuple[Tuple[EinsumAxisAccess, ...],
                                              PMapT[str, EinsumAxisAccess],
                                              PMapT[str, ShapeComponentT]]:
    """
    Normalizes the subscript for an input operand in an einsum. Returns
    ``(access_descrs, updated_index_to_descr, updated_to_index_to_axis_length)``,
    where, *access_descrs* is a :class:`tuple` of
    :class`EinsumAxisAccess` corresponding to *subscript*,
    *updated_index_to_descr* is the updated version of *index_to_descr* while
    inferring *subscript*. Similarly, *updated_index_to_axis_length* is the updated
    version of *index_to_axis_length*.
    :arg index_to_descr: A mapping from index names to instance of
        :class:`EinsumAxisAccess`. These constraints would most likely
        recorded during normalizing other parts of an einsum's subscripts.
    :arg index_to_axis_length: A mapping from index names to instance of
        :class:`ShapeComponentT` denoting the iteration extent of the index.
        These constraints would most likely recorded during normalizing other
        parts of an einsum's subscripts.
    """
    normalized_indices: List[str] = []
    acc = subscript.strip()
    while acc:
        match = EINSUM_FIRST_INDEX.match(acc)
        if match:
            if "alpha" in match.groupdict():
                normalized_indices.append(match.groupdict()["alpha"])
            else:
                assert "ellipsis" in match.groupdict()
                raise NotImplementedError("Broadcasting in einsums not supported")
            assert match.span()[0] == 0
            acc = acc[match.span()[-1]:]
        else:
            raise ValueError(f"Cannot parse '{acc}' in provided einsum"
                             f" '{subscript}'.")

    if len(normalized_indices) != len(in_operand_shape):
        raise ValueError(f"Subscript '{subscript}' doesn't match the dimensionality "
                         f"of corresponding operand ({len(in_operand_shape)}).")

    in_operand_axis_descrs = []

    for iaxis, index_char in enumerate(normalized_indices):
        in_axis_len = in_operand_shape[iaxis]
        if index_char in index_to_descr:
            if index_char in index_to_axis_length:
                seen_axis_len = index_to_axis_length[index_char]
                if in_axis_len != seen_axis_len:
                    if in_axis_len == 1:
                        # Broadcast the current axis
                        pass
                    elif seen_axis_len == 1:
                        # Broadcast to the length of the current axis
                        index_to_axis_length = (index_to_axis_length
                                                .set(index_char, in_axis_len))
                    else:
                        raise ValueError("Got conflicting lengths for"
                                         f" '{index_char}' -- {in_axis_len},"
                                         f" {seen_axis_len}.")
            else:
                index_to_axis_length = index_to_axis_length.set(index_char,
                                                                in_axis_len)
        else:
            redn_sr_no = len([descr for descr in index_to_descr.values()
                              if isinstance(descr, SummationAxis)])
            redn_axis_descr = SummationAxis(redn_sr_no)
            index_to_descr = index_to_descr.set(index_char, redn_axis_descr)
            index_to_axis_length = index_to_axis_length.set(index_char,
                                                             in_axis_len)

        in_operand_axis_descrs.append(index_to_descr[index_char])

    return (tuple(in_operand_axis_descrs),
            index_to_descr, index_to_axis_length)


def _parse_subscripts(subscripts: str,
                      operand_shapes: Tuple[ShapeT, ...]
                      ) -> Tuple[Tuple[Tuple[EinsumAxisAccess, ...], ...],
                                 PMapT[str, EinsumAxisAccess]]:
    if len(operand_shapes) == 0:
        raise ValueError("must specify at least one operand")

    if "->" not in subscripts:
        # implicit-mode: output spec matched by alphabetical ordering of
        # indices (ewwwww)
        raise NotImplementedError("Implicit mode not supported. 'subscripts'"
                                  " must contain '->', followed by the output's"
                                  " indices.")
    in_spec, out_spec = subscripts.split("->")

    in_specs = in_spec.split(",")

    if len(operand_shapes) != len(in_specs):
        raise ValueError(
            f"Number of operands should match the number"
            f" of arg specs: '{in_specs}'. Length of operands is"
            f" {len(operand_shapes)}; expecting"
            f" {len(in_specs)} operands."
        )

    index_to_descr = _normalize_einsum_out_subscript(out_spec)
    index_to_axis_length: PMapT[str, ShapeComponentT] = pmap()
    access_descriptors = []

    for in_spec, in_operand_shape in szip(in_specs, operand_shapes):
        access_descriptor, index_to_descr, index_to_axis_length = (
            _normalize_einsum_in_subscript(in_spec,
                                           in_operand_shape,
                                           index_to_descr,
                                           index_to_axis_length))
        access_descriptors.append(access_descriptor)

    return tuple(access_descriptors), index_to_descr


def fused_einsum(subscripts: str,
                 operand_shapes: Sequence[Any],
                 use_matrix: npt.ArrayLike,
                 dtypes: Optional[npt.DTypeLike] = None,
                 value_to_dtype: Optional[Mapping[str, npt.DTypeLike]] = None,
                 ) -> FusedEinsum:
    """
    Returns a :class:`~feinsum.einsum.FusedEinsum` with an interface similar to
    :func:`numpy.einsum`.

    :param subscripts: A :class:`str` describing the Einstein summation as
        accepted by :func:`numpy.einsum`.
    :param dtypes: The dtype of all the value the operands use. Cannot be
        provide both *value_to_dtype* and *dtypes*.
    :param value_to_dtype: A mapping from the values the operands of the einsum
        depend on to their dtypes. Cannot be provide both *value_to_dtype* and
        *dtypes*.
    :param operand_shapes: A sequence of shapes of the operands of the Einstein
        Summation.
    :param use_matrix: A 2D :mod:`numpy` array-like object, where ``i,j``-th
        entry corresponds to the :class:`frozenset` of :class:`str` of value
        names that the ``j``-th operand of the ``i``-th einsum accesses.
    """

    from functools import reduce

    proc_op_shapes = tuple(_preprocess_shape(shape) for shape in operand_shapes)
    access_descriptors, index_to_descr = _parse_subscripts(subscripts,
                                                           proc_op_shapes)

    use_matrix = np.array(use_matrix)

    # {{{ sanity checks

    if use_matrix.ndim != 2:
        raise ValueError("``use_matrix`` is not a matrix.")

    if not np.all(
            np.vectorize(lambda x: (isinstance(x, (frozenset, set))
                                    and all(isinstance(k, str) for k in x))
                         )(use_matrix)):
        raise ValueError("Each element of the array-like ``use_matrix`` must be"
                         " an instance of FrozenSet[str].")

    use_matrix = np.vectorize(lambda x: frozenset(x))(use_matrix)
    assert isinstance(use_matrix, np.ndarray)
    all_values_from_use_matrix: FrozenSet[str] = reduce(frozenset.union,
                                                        use_matrix.ravel(),
                                                        frozenset())
    if use_matrix.shape[1] != len(proc_op_shapes):
        raise ValueError("use_matrix.shape[1] != len(proc_op_shapes)")

    if dtypes is not None:
        if value_to_dtype is not None:
            raise ValueError("cannot pass both ``dtypes`` and ``value_to_dtype``")

        value_to_proc_dtype = {value: np.dtype(dtypes)
                               for value in all_values_from_use_matrix}
    else:
        if value_to_dtype is None:
            raise ValueError("must pass either ``value_to_dtype`` or ``dtypes``")
        value_to_proc_dtype = {value: np.dtype(dtype)
                               for value, dtype in value_to_dtype.items()}

    if all_values_from_use_matrix != frozenset(value_to_proc_dtype.keys()):
        raise ValueError("The values inferred via ``value_to_dtype`` do"
                         " not match the values inferred via ``use_matrix``")

    # }}}

    axis_to_name = pmap({v: k for k, v in index_to_descr.items()})
    vng = UniqueNameGenerator()
    vng.add_names(value_to_proc_dtype.keys())

    # {{{ process operand shapes to

    from feinsum.einsum import ShapeComponentT as ProcessedShapeComponentT

    size_param_op_shapes = []
    axis_to_dim: Dict[EinsumAxisAccess, ProcessedShapeComponentT] = {}

    for axes, op_shape in szip(access_descriptors,
                               proc_op_shapes):
        size_param_op_shape = []
        for axis, dim in szip(axes, op_shape):
            if axis in axis_to_dim:
                if isinstance(dim, INT_CLASSES + (SizeParam,)):
                    assert axis_to_dim[axis] == dim
                else:
                    assert isinstance(dim, VeryLongAxis)
            else:
                if isinstance(dim, INT_CLASSES + (SizeParam,)):
                    axis_to_dim[axis] = dim
                else:
                    assert isinstance(dim, VeryLongAxis)
                    axis_to_dim[axis] = SizeParam(vng(f"N_{axis_to_name[axis]}"))

            size_param_op_shape.append(axis_to_dim[axis])

        size_param_op_shapes.append(tuple(size_param_op_shape))

    # }}}

    return FusedEinsum(tuple(size_param_op_shapes),
                       pmap(value_to_proc_dtype),
                       access_descriptors,
                       tuple(tuple(use_row)
                             for use_row in use_matrix),
                       index_names=axis_to_name,
                       )


def einsum(subscripts: str,
           *operands: ArrayT,
           arg_names: Optional[Sequence[str]] = None) -> FusedEinsum:
    """
    Returns a :class:`~feinsum.einsum.FusedEinsum` with an interface similar to
    :func:`numpy.einsum`.

    :param arg_names: An optional sequence of :class:`str`. If not provided,
        defaults to the sequence: ``"arg_0", "arg_1", "arg_2", ...``.
    """

    if arg_names is None:
        arg_names = [f"arg_{i}" for i in range(len(operands))]

    if len(arg_names) != len(operands):
        raise ValueError(f"Number of argument names ({len(arg_names)}) "
                         " does not match the number of operands"
                         f" ({len(operands)}).")

    use_matrix = [[{arg_name} for arg_name in arg_names]]
    value_to_dtype = {arg_name: operand.dtype
                      for arg_name, operand in szip(arg_names, operands)}

    return fused_einsum(subscripts,
                        tuple(op.shape for op in operands),
                        use_matrix=use_matrix,  # type: ignore[arg-type]
                        value_to_dtype=value_to_dtype)

# vim: foldmethod=marker
