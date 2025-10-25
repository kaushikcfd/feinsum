"""
.. currentmodule:: feinsum.make_einsum

.. autofunction:: einsum
.. autofunction:: batched_einsum
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
from collections.abc import Sequence
from typing import (
    Any,
)

import numpy as np
import numpy.typing as npt

from feinsum.einsum import (
    INT_CLASSES,
    Array,
    BatchedEinsum,
    ShapeComponentT,
    ShapeT,
    SizeParam,
)


def _preprocess_component(s: Any) -> ShapeComponentT:
    if isinstance(s, str):
        return SizeParam(s)
    elif isinstance(s, INT_CLASSES) and (s >= 0):
        return s
    elif isinstance(s, SizeParam):
        return s
    else:
        raise ValueError(f"Cannot infer shape component '{s}'.")


def _preprocess_shape(shape: Any) -> ShapeT:
    from collections.abc import Iterable

    if not isinstance(shape, Iterable):
        shape = (shape,)

    return tuple(_preprocess_component(d) for d in shape)


def array(name: str, shape: Any, dtype: npt.DTypeLike = "float64") -> Array:
    """
    Return an
    """
    return Array(name=name, shape=_preprocess_shape(shape), dtype=np.dtype(dtype))


EINSUM_FIRST_INDEX = re.compile(r"^\s*((?P<alpha>[a-zA-Z])|(?P<ellipsis>\.\.\.))\s*")


def _normalize_einsum_subscript(subscript: str, is_output: bool) -> tuple[str, ...]:
    """
    Normalizes the subscript of an einsum (provided in the explicit
    mode).
    """

    normalized_indices: list[str] = []
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
            acc = acc[match.span()[-1] :]
        else:
            raise ValueError(
                f"Cannot parse '{acc}' in provided einsum" f" '{subscript}'."
            )

    if is_output:
        if len(set(normalized_indices)) != len(normalized_indices):
            raise ValueError(
                "Used an input more than once to refer to the"
                f" output axis in '{subscript}"
            )
    return tuple(normalized_indices)


def batched_einsum(
    subscripts: str,
    args: Sequence[Sequence[Array]],
) -> BatchedEinsum:
    """
    Returns a :class:`~feinsum.einsum.BatchedEinsum` with an interface similar to
    :func:`numpy.einsum`.

    :param subscripts: A :class:`str` describing the Einstein summation as
        accepted by :func:`numpy.einsum`.
    :param args: A b-long sequence each each comprising of n-long
        :class:`~feinsum.einsum.Array` instances, where, "b" is the number of
        einsums in the batched einsum, and, "n" is the number of array operands
        accepted by each of those b-einsums.
    """

    if "->" not in subscripts:
        # implicit-mode: output spec matched by alphabetical ordering of
        # indices (ewwwww)
        raise ValueError(
            "Missing -> in 'subscripts'. If the expected behavior"
            " is implicit mode, feinsum does not support it."
        )
    in_specs, out_spec = subscripts.split("->")
    out_idx_set = _normalize_einsum_subscript(out_spec, is_output=True)
    in_idx_sets = tuple(
        _normalize_einsum_subscript(in_spec, is_output=False)
        for in_spec in in_specs.split(",")
    )
    try:
        return BatchedEinsum(
            out_idx_set, in_idx_sets, tuple(tuple(arg_row) for arg_row in args)
        )
    except AssertionError as exc:
        raise TypeError(f"{exc}") from exc


def einsum(subscripts: str, *operands: Array) -> BatchedEinsum:
    """
    Returns a :class:`~feinsum.einsum.BatchedEinsum` with an interface similar to
    :func:`numpy.einsum`.
    """
    return batched_einsum(subscripts, [operands])


# vim: foldmethod=marker
