"""
.. currentmodule:: feinsum.codegen.loopy

.. autofunction:: generate_loopy
"""

import loopy as lp
import islpy as isl
import pymbolic.primitives as p
import numpy as np

from typing import Union, Tuple, Optional, Any, Dict
from pytools import UniqueNameGenerator
from feinsum.einsum import (FusedEinsum, FreeAxis, SummationAxis,
                            EinsumAxisAccess, IntegralT, INT_CLASSES,
                            ContractionSchedule, EinsumOperand,
                            IntermediateResult, SizeParam, Argument,
                            ShapeT)
from feinsum.make_einsum import fused_einsum
from more_itertools import zip_equal as szip
from pyrsistent import pmap


LOOPY_LANG_VERSION = (2018, 2)


def get_isl_basic_set(einsum: FusedEinsum) -> isl.BasicSet:
    dim_name_to_ubound = {}

    for idx, dim in einsum.index_to_dim_length().items():
        if isinstance(dim, SizeParam):
            proc_dim: Union[str, IntegralT] = dim.name
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


def _generate_trivial_einsum(einsum: FusedEinsum,
                             output_names: Tuple[str, ...],
                             ) -> Tuple[isl.BasicSet, "lp.TranslationUnit"]:
    assert len(output_names) == einsum.noutputs

    domain = get_isl_basic_set(einsum)
    statements = []
    dummy_indices = tuple(sorted(einsum.index_names[axis]
                                 for axis in einsum.index_to_dim_length()
                                 if isinstance(axis, SummationAxis)))

    for i_out, output_name in enumerate(output_names):
        lhs = make_subscript(output_name,
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

    return domain, statements


def generate_loopy(einsum: FusedEinsum,
                   schedule: Optional[ContractionSchedule] = None
                   ) -> "lp.TranslationUnit":
    """
    Returns a :class:`loopy.TranslationUnit` with the reductions scheduled by
    *contract_path*.

    :param schedule: An optional instance of
        :class:`~feinsum.einsum.ContractionSchedule`. Defaults to the trivial
        contraction schedule if not provided.
    """

    if schedule is None:
        from feinsum.einsum import get_trivial_contract_schedule
        schedule = get_trivial_contract_schedule(einsum)

    assert isinstance(schedule, ContractionSchedule)

    # {{{ prepare unique name generator

    vng = UniqueNameGenerator()
    vng.add_names(einsum.value_to_dtype.keys())
    for dim in einsum.index_to_dim_length().values():
        if isinstance(dim, SizeParam):
            vng.add_name(dim.name)

    # }}}

    # {{{ start holding a mapping from argument to shapes

    arg_to_shape: Dict[Argument, ShapeT] = {}

    for ioperand, arg_shape in enumerate(einsum.arg_shapes):
        arg_to_shape[EinsumOperand(ioperand)] = arg_shape

    # }}}

    # {{{

    result_name_in_lpy_knl = tuple(tuple(vng(result_name)
                                         for result_name in schedule.result_names)
                                   for _ in range(einsum.noutputs))

    name_in_feinsum_to_lpy = tuple(pmap({feinsum_name: lpy_name
                                         for feinsum_name, lpy_name in szip(
                                                 schedule.result_names,
                                                 result_name_in_lpy_knl[i_output])})
                                   for i_output in range(einsum.noutputs))

    # }}}

    # {{{ update value_to_dtype

    value_to_dtype = einsum.value_to_dtype

    for i_output in range(einsum.noutputs):
        arg_to_dtype: Dict[Argument, np.dtype[Any]] = {
            EinsumOperand(ioperand): np.find_common_type({value_to_dtype[use]
                                                          for use in uses},
                                                         [])
            for ioperand, uses in enumerate(einsum
                                            .use_matrix[i_output])}

        for name_in_lpy_knl, name_in_feinsum, args in (
                zip(result_name_in_lpy_knl[i_output],
                    schedule.result_names,
                    schedule.arguments)):
            dtype = np.find_common_type({arg_to_dtype[arg] for arg in args}, [])
            value_to_dtype = value_to_dtype.set(name_in_lpy_knl, dtype)
            arg_to_dtype[IntermediateResult(name_in_feinsum)] = dtype

    # }}}

    statements = []
    domains = []
    kernel_data = []

    for istep, (name_in_feinsum, subscripts, args) in enumerate(
            zip(schedule.result_names,
                schedule.subscripts,
                schedule.arguments)):

        subeinsum_value_to_dtype = {}
        subeinsum_use_matrix = []
        for i_output in range(einsum.noutputs):
            subeinsum_use_row = []
            for arg in args:
                if isinstance(arg, EinsumOperand):
                    subeinsum_use_row.append(einsum
                                              .use_matrix[i_output][arg.ioperand])
                    for value in einsum.use_matrix[i_output][arg.ioperand]:
                        subeinsum_value_to_dtype[value] = value_to_dtype[value]
                elif isinstance(arg, IntermediateResult):
                    lpy_name = name_in_feinsum_to_lpy[i_output][arg.name]
                    subeinsum_use_row.append(frozenset({lpy_name}))
                    subeinsum_value_to_dtype[lpy_name] = value_to_dtype[lpy_name]
                else:
                    raise NotImplementedError(type(arg))

            subeinsum_use_matrix.append(subeinsum_use_row)

        subeinsum = fused_einsum(subscripts,
                                 [arg_to_shape[arg] for arg in args],
                                 subeinsum_use_matrix,  # type: ignore[arg-type]
                                 value_to_dtype=pmap(subeinsum_value_to_dtype))
        subeinsum = subeinsum.copy(
            index_names=pmap({idx: name if istep == 0 else f"{name}_{istep-1}"
                              for idx, name in subeinsum.index_names.items()}))
        arg_to_shape[IntermediateResult(name_in_feinsum)] = subeinsum.shape

        subeinsum_domain, subeinsum_statements = _generate_trivial_einsum(
            subeinsum,
            tuple([result_name_in_lpy_knl[i_output][istep]
                   for i_output in range(einsum.noutputs)]))

        domains.append(subeinsum_domain)
        statements.extend(subeinsum_statements)

    # {{{ Populate kernel_data

    # Inputs:
    for value, dtype in einsum.value_to_dtype.items():
        kernel_data.append(lp.GlobalArg(value, shape=lp.auto, dtype=dtype))

    # Outputs
    for i_output in range(einsum.noutputs):
        kernel_data.append(lp.GlobalArg(result_name_in_lpy_knl[i_output][-1],
                                        dtype=lp.auto, shape=lp.auto))

    # Temporary Variables
    for i_output in range(einsum.noutputs):
        for istep in range(schedule.nsteps-1):
            kernel_data.append(
                lp.TemporaryVariable(result_name_in_lpy_knl[i_output][istep],
                                     dtype=lp.auto, shape=lp.auto,
                                     address_space=lp.AddressSpace.GLOBAL))

    # }}}

    return lp.make_kernel(
        domains, statements,
        kernel_data=kernel_data+[...],
        lang_version=LOOPY_LANG_VERSION)
