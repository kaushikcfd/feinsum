"""
.. currentmodule:: feinsum
.. autofunction:: einsum
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

from typing import Tuple, Protocol, Any, List, Optional, FrozenSet
from dataclasses import dataclass
from pyrsistent import pmap
from pyrsistent.typing import PMap as PMapT
from feinsum.einsum import (Einsum, ShapeComponentT, ShapeT, FreeAxis, SummationAxis,
                            EinsumAxisAccess, VeryLongAxis, INT_CLASSES)


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
    if (isinstance(s, VeryLongAxis)
            or np.isposinf(s)):
        return VeryLongAxis()
    elif (isinstance(s, INT_CLASSES) and (s >= 0)):
        return s
    else:
        raise ValueError(f"Cannot infer shape component '{s}'.")


def _preprocess_shape(shape: Any) -> ShapeT:
    if not isinstance(shape, tuple):
        shape = shape,

    return tuple(_preprocess_component(d) for d in shape)


def array(shape: Any, dtype: Any) -> Array:
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
                                   in_operand: ArrayT,
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

    if len(normalized_indices) != len(in_operand.shape):
        raise ValueError(f"Subscript '{subscript}' doesn't match the dimensionality "
                         f"of corresponding operand ({len(in_operand.shape)}).")

    in_operand_axis_descrs = []

    for iaxis, index_char in enumerate(normalized_indices):
        in_axis_len = in_operand.shape[iaxis]
        if index_char in index_to_descr:
            if index_char in index_to_axis_length:
                seen_axis_len = index_to_axis_length[index_char]
                if in_axis_len != seen_axis_len:
                    if in_axis_len != 1:
                        # Broadcast the current axis
                        pass
                    elif seen_axis_len != 1:
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


def einsum(subscripts: str, *operands: ArrayT,
           use_matrix: Optional[Tuple[Tuple[FrozenSet[str], ...]]] = None) -> Einsum:
    """
    Returns a :class:`Einsum` with an interface similar to
    :func:`numpy.einsum`, but corresponds to multiple einsums fused in a single
    loop. The number of einsum expressions in the computation is inferred from
    the 0-th dimension length of *use_matrix*.


    :param use_matrix: Denotes a matrix of shape ``(noutputs, noperands)`` with
        the ``(i, j)``-th entry denoting the names of arrays making up the
        ``j``-th operand in the ``i``-th output einsum.
    """
    if len(operands) == 0:
        raise ValueError("must specify at least one operand")

    if "->" not in subscripts:
        # implicit-mode: output spec matched by alphabetical ordering of
        # indices (ewwwww)
        raise NotImplementedError("Implicit mode not supported. 'subscripts'"
                                  " must contain '->', followed by the output's"
                                  " indices.")
    in_spec, out_spec = subscripts.split("->")

    in_specs = in_spec.split(",")

    if len(operands) != len(in_specs):
        raise ValueError(
            f"Number of operands should match the number "
            f"of arg specs: '{in_specs}'. Length of operands is {len(operands)}; "
            f"expecting {len(in_specs)} operands."
        )

    index_to_descr = _normalize_einsum_out_subscript(out_spec)
    index_to_axis_length: PMapT[str, ShapeComponentT] = pmap()
    access_descriptors = []

    for in_spec, in_operand in zip(in_specs, operands):
        access_descriptor, index_to_descr, index_to_axis_length = (
            _normalize_einsum_in_subscript(in_spec,
                                           in_operand,
                                           index_to_descr,
                                           index_to_axis_length))
        access_descriptors.append(access_descriptor)

    if use_matrix is None:
        use_matrix = tuple(frozenset([f"arg_{i}"])
                           for i in range(len(operands))),

    assert isinstance(use_matrix, tuple)
    assert all(isinstance(k, tuple) for k in use_matrix)
    assert all(all((isinstance(use, frozenset)
                    and all(isinstance(k, str) for k in use))
                   for use in use_row)
               for use_row in use_matrix)
    assert all(len(use_row) == len(operands)
               for use_row in use_matrix)

    return Einsum(tuple(operand.shape
                        for operand in operands),
                  tuple(operand.dtype
                        for operand in operands),
                  tuple(access_descriptors),
                  use_matrix)
