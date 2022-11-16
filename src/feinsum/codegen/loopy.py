"""
.. currentmodule:: feinsum.codegen.loopy

.. autofunction:: generate_loopy
.. autofunction:: generate_loopy_with_opt_einsum_schedule
"""

import loopy as lp
import islpy as isl
import pymbolic.primitives as p
import numpy as np

from typing import Union, Tuple, Optional, Any, Dict
from pytools import UniqueNameGenerator, memoize_on_first_arg
from feinsum.einsum import (FusedEinsum, FreeAxis, SummationAxis,
                            EinsumAxisAccess, IntegralT, INT_CLASSES,
                            ContractionSchedule, EinsumOperand,
                            IntermediateResult, SizeParam, Argument,
                            ShapeT,
                            get_opt_einsum_contraction_schedule,
                            get_trivial_contraction_schedule)
from feinsum.make_einsum import fused_einsum
from more_itertools import zip_equal as szip
from immutables import Map


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


def make_access_expr(name: str,
                     axes: Tuple[EinsumAxisAccess, ...],
                     einsum: FusedEinsum,
                     ) -> p.Call:
    return p.Variable(_get_input_subst_name(name))(
        *tuple(p.Variable(einsum.index_names[axis])
               for axis in axes))


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
        rhs = p.Product(tuple((p.Sum(tuple(make_access_expr(dep, axes, einsum)
                                           for dep in deps)
                                     )
                               if len(deps) > 1
                               else make_access_expr(list(deps)[0], axes, einsum))
                              for deps, axes in zip(einsum.use_matrix[i_out],
                                                    einsum.access_descriptors))
                        )
        if dummy_indices:
            rhs = lp.Reduction("sum", tuple(p.Variable(idx)
                                            for idx in dummy_indices), rhs)

        statements.append(lp.Assignment(lhs, rhs))

    return domain, statements


def _get_input_subst_name(x: str) -> str:
    return f"_fe_subst_{x}"


@memoize_on_first_arg
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
        schedule = get_trivial_contraction_schedule(einsum)

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

    name_in_feinsum_to_lpy = tuple(Map({feinsum_name: lpy_name
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
                                 subeinsum_use_matrix,
                                 value_to_dtype=Map(subeinsum_value_to_dtype))
        subeinsum = subeinsum.copy(
            index_names=Map({idx: name if istep == 0 else f"{name}_{istep-1}"
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
        kernel_data.append(lp.GlobalArg(value,
                                        shape=lp.auto,
                                        dtype=dtype))

    # Outputs
    for i_output in range(einsum.noutputs):
        kernel_data.append(lp.GlobalArg(result_name_in_lpy_knl[i_output][-1],
                                        shape=lp.auto))

    # Temporary Variables
    for i_output in range(einsum.noutputs):
        for istep in range(schedule.nsteps-1):
            kernel_data.append(
                lp.TemporaryVariable(result_name_in_lpy_knl[i_output][istep],
                                     shape=lp.auto,
                                     address_space=lp.AddressSpace.GLOBAL))

    # }}}

    # Substitutions
    substitutions: Dict[str, lp.SubstitutionRule] = {}
    for val in einsum.value_to_dtype:
        val_ndim = len(einsum.get_arg_shape(val))
        subst_name = _get_input_subst_name(val)
        substitutions[subst_name] = lp.SubstitutionRule(
            subst_name,
            tuple(f"_{idim}"
                  for idim in range(val_ndim)),
            p.Variable(val)[
                tuple(p.Variable(f"_{idim}")
                      for idim in range(val_ndim))]
        )

    t_unit = lp.make_kernel(
        domains, statements,
        kernel_data=kernel_data+[...],
        lang_version=LOOPY_LANG_VERSION)

    # TODO: Once https://github.com/inducer/loopy/issues/705
    # is fixed avoid this copy
    t_unit = t_unit.with_kernel(
        t_unit.default_entrypoint
        .copy(substitutions=substitutions))

    return t_unit


def generate_loopy_with_opt_einsum_schedule(expr: FusedEinsum,
                                            **opt_einsum_kwargs: Any
                                            ) -> "lp.TranslationUnit":
    """
    Returns a :class:`loopy.TranslationUnit` with the
    :class:`~feinsum.einsum.ContractionSchedule` specified via
    :func:`opt_einsum.contract_path`.

    :param opt_einsum_kwargs: kwargs to be passed to
        :func:`~feinsum.einsum.get_opt_einsum_contraction_schedule`.
    """
    return generate_loopy(expr,
                          get_opt_einsum_contraction_schedule(expr))
