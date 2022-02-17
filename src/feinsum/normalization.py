"""
.. autofunction:: normalize_einsum
"""

from typing import List, Dict
from pyrsistent import pmap
from feinsum.einsum import (FusedEinsum, SizeParam, FreeAxis, SummationAxis,
                            EinsumAxisAccess)


def normalize_einsum(einsum: FusedEinsum) -> FusedEinsum:
    """
    Returns a normalized form of *einsum*.
    """
    nfree_indices = einsum.ndim
    nredn_indices = len([idx
                         for idx in einsum.index_to_dim_length()
                         if isinstance(idx, SummationAxis)])

    # there are only 26 letters :)
    assert nfree_indices + nredn_indices <= 26

    index_to_new_name = {}
    # type-ignore reason: List is invariant
    sorted_axes: List[EinsumAxisAccess] = ([FreeAxis(i)  # type: ignore[assignment]
                                            for i in range(nfree_indices)]
                                           + [SummationAxis(i)  # type: ignore[misc]
                                              for i in range(nredn_indices)])
    for idx, ichr in zip(sorted_axes, range(97, 123)):
        index_to_new_name[idx] = chr(ichr)

    old_value_to_new_value: Dict[str, str] = {}
    new_use_matrix = []
    for use_row in einsum.use_matrix:
        new_use_row = []
        for values in use_row:
            if len(values) > 1:
                raise NotImplementedError("Multi-values per use not yet supported.")
            old_value, = values
            new_value = old_value_to_new_value.setdefault(
                old_value,
                f"arg_{len(old_value_to_new_value)}")
            new_use_row.append(frozenset([new_value]))

        new_use_matrix.append(tuple(new_use_row))

    new_value_to_dtypes = {old_value_to_new_value[old_val]: dtype
                           for old_val, dtype in einsum.value_to_dtype.items()}

    old_size_param_to_new_size_param: Dict[SizeParam, SizeParam] = {
        old_sz_par: SizeParam(f"N_{index_to_new_name[old_idx]}")
        for old_idx, old_sz_par in einsum.index_to_dim_length().items()
        if isinstance(old_sz_par, SizeParam)
    }

    # type-ignore reason: mypy isn't smart to see that only SizeParams of the
    # dict are queried
    new_arg_shapes = tuple(
        tuple(old_size_param_to_new_size_param[dim]
              if isinstance(dim, SizeParam)
              else dim
              for dim in old_arg_shape)
        for old_arg_shape in einsum.arg_shapes
    )

    return FusedEinsum(new_arg_shapes,
                       pmap(new_value_to_dtypes),
                       einsum.access_descriptors,
                       tuple(new_use_matrix),
                       pmap(index_to_new_name))
