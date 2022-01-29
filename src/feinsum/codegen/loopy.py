"""
.. currentmodule:: feinsum.codegen.loopy

.. autofunction:: generate_loopy
"""

import loopy as lp
import islpy as isl

from typing import Union, Tuple
from pytools import UniqueNameGenerator
from more_itertools import zip_equal as zip
from feinsum.einsum import (FusedEinsum, FreeAxis, SummationAxis,
                            EinsumAxisAccess, VeryLongAxis, IntegralT,
                            INT_CLASSES)
import pymbolic.primitives as p

LOOPY_LANG_VERSION = (2018, 2)


def get_isl_basic_set(einsum: FusedEinsum) -> isl.BasicSet:
    dim_name_to_ubound = {}
    vng = UniqueNameGenerator()

    for idx, dim in einsum.index_to_dim_length().items():
        if isinstance(dim, VeryLongAxis):
            proc_dim: Union[str, IntegralT] = vng(f"N_{einsum.index_names[idx]}")
        else:
            proc_dim = dim

        dim_name_to_ubound[einsum.index_names[idx]] = proc_dim

    space = isl.Space.create_from_names(isl.DEFAULT_CONTEXT,
                                        set=sorted(dim_name_to_ubound),
                                        params=sorted(
                                            bound
                                            for bound in dim_name_to_ubound.values()
                                            if isinstance(bound, str)))
    bset = isl.BasicSet.universe(space)

    for dim_name, ubound in sorted(dim_name_to_ubound.items()):
        if isinstance(ubound, str):
            bset = bset.add_constraint(
                isl.Constraint.ineq_from_names(space, {1: -1,
                                                       ubound: 1,
                                                       dim_name: -1}))
        else:
            assert isinstance(ubound, INT_CLASSES)
            bset = bset.add_constraint(
                isl.Constraint.ineq_from_names(space, {1: ubound-1, dim_name: -1}))

        bset = bset.add_constraint(
            isl.Constraint.ineq_from_names(space, {1: 0, dim_name: 1}))

    return bset


def make_subscript(name: str,
                   axes: Tuple[EinsumAxisAccess, ...],
                   einsum: FusedEinsum,
                   ) -> p.Subscript:
    return p.Variable(name)[tuple(p.Variable(einsum.index_names[axis])
                                  for axis in axes)]


def generate_loopy(einsum: FusedEinsum) -> "lp.TranslationUnit":
    domain = get_isl_basic_set(einsum)
    statements = []
    dummy_indices = tuple(sorted(einsum.index_names[axis]
                                 for axis in einsum.index_to_dim_length()
                                 if isinstance(axis, SummationAxis)))

    for i_out in range(einsum.noutputs):
        lhs = make_subscript(f"out_{i_out}",
                             tuple(FreeAxis(idim)
                                                   for idim in
                                   range(einsum.ndim)),
                             einsum)
        rhs = p.Product(tuple(p.Sum(tuple(make_subscript(dep, axes, einsum)
                                          for dep in deps)
                                    )
                              for deps, axes in zip(einsum.use_matrix[i_out],
                                                    einsum.access_descriptors))
                        )
        if dummy_indices:
            rhs = lp.Reduction("sum", tuple(p.Variable(idx)
                                            for idx in dummy_indices), rhs)

        statements.append(lp.Assignment(lhs, rhs))

    return lp.make_kernel([domain],
                          statements,
                          kernel_data=([lp.GlobalArg(value, dtype=dtype,
                                                     shape=lp.auto)
                                        for value, dtype in sorted(einsum.
                                                                   value_to_dtype
                                                                   .items())]
                                       + [...]
                                       ),
                          lang_version=LOOPY_LANG_VERSION)
