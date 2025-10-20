"""
.. currentmodule:: feinsum.codegen.loopy

.. autofunction:: generate_loopy
.. autofunction:: generate_loopy_with_opt_einsum_schedule
"""

from collections.abc import Mapping
from typing import Any, cast

import islpy as isl
import loopy as lp
import numpy as np
import pymbolic.primitives as p
from pytools import UniqueNameGenerator, memoize_on_first_arg

from feinsum.einsum import (
    INT_CLASSES,
    Argument,
    BatchedEinsum,
    ContractionSchedule,
    EinsumOperand,
    IntegralT,
    IntermediateResult,
    ShapeComponentT,
    ShapeT,
    SizeParam,
    get_opt_einsum_contraction_schedule,
    get_trivial_contraction_schedule,
)

LOOPY_LANG_VERSION = (2018, 2)


def _get_isl_basic_set(
    index_to_dim_length: Mapping[str, ShapeComponentT],
) -> isl.BasicSet:
    dim_name_to_ubound: dict[str, str | IntegralT] = {
        idx: dim.name if isinstance(dim, SizeParam) else dim
        for idx, dim in index_to_dim_length.items()
    }

    space = isl.Space.create_from_names(
        isl.DEFAULT_CONTEXT,
        set=sorted(dim_name_to_ubound),
        params=sorted(
            bound for bound in dim_name_to_ubound.values() if isinstance(bound, str)
        ),
    )
    bset = isl.BasicSet.universe(space)

    for dim_name, ubound in sorted(dim_name_to_ubound.items()):
        if isinstance(ubound, str):
            bset = bset.add_constraint(
                isl.Constraint.ineq_from_names(
                    space, {1: -1, ubound: 1, dim_name: -1}
                )
            )
        else:
            assert isinstance(ubound, INT_CLASSES)
            bset = bset.add_constraint(
                isl.Constraint.ineq_from_names(space, {1: ubound - 1, dim_name: -1})
            )

        bset = bset.add_constraint(
            isl.Constraint.ineq_from_names(space, {1: 0, dim_name: 1})
        )

    return bset


def make_subscript(
    name: str,
    idx_set: tuple[str, ...],
) -> p.Subscript:
    return p.Subscript(
        p.Variable(name),
        tuple(p.Variable(idx) for idx in idx_set),
    )


def make_access_expr(
    name: str,
    idx_set: tuple[str, ...],
) -> p.Call:
    return p.Call(
        p.Variable(_get_input_subst_name(name)),
        tuple(p.Variable(idx) for idx in idx_set),
    )


def _get_input_subst_name(x: str) -> str:
    return f"_fe_subst_{x}"


def _get_out_in_idx_sets(
    subscripts: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, ...], ...]]:
    from feinsum.make_einsum import _normalize_einsum_subscript

    in_specs, out_spec = subscripts.split("->")
    out_idx_set = _normalize_einsum_subscript(out_spec, is_output=True)
    in_idx_sets = tuple(
        _normalize_einsum_subscript(in_spec, is_output=False)
        for in_spec in in_specs.split(",")
    )
    return out_idx_set, in_idx_sets


@memoize_on_first_arg
def generate_loopy(
    einsum: BatchedEinsum, schedule: ContractionSchedule | None = None
) -> "lp.TranslationUnit":
    """
    Returns a :class:`loopy.TranslationUnit` with the reductions scheduled by
    *contract_path*.

    :param schedule: An optional instance of
        :class:`~feinsum.einsum.ContractionSchedule`. Defaults to the trivial
        contraction schedule if not provided.
    """
    from functools import reduce

    if schedule is None:
        schedule = get_trivial_contraction_schedule(einsum)

    assert isinstance(schedule, ContractionSchedule)

    # {{{ prepare unique name generator

    vng = UniqueNameGenerator()
    vng.add_names(einsum.all_args)
    vng.add_names(einsum.all_indices)
    vng.add_names(p.name for p in einsum.all_size_params)

    # }}}

    statements: list[lp.Assignment] = []
    knl_args: list[lp.ArrayArg | lp.TemporaryVariable] = []
    substitutions: dict[str, lp.SubstitutionRule] = {}

    # {{{ populate operands in knl_args

    for arg_name in sorted(einsum.all_args):
        dtype = einsum.arg_to_dtype[arg_name]
        shape = tuple(
            dim.name if isinstance(dim, SizeParam) else dim
            for dim in einsum.arg_to_shape[arg_name]
        )
        knl_args.append(lp.GlobalArg(arg_name, dtype=dtype, shape=shape))

    # }}}

    # {{{ populate substitutions with arguments

    for arg in sorted(einsum.all_args):
        subst_name = _get_input_subst_name(arg)
        ndim = len(einsum.arg_to_shape[arg])
        subst = lp.SubstitutionRule(
            subst_name,
            tuple(f"d_{i}" for i in range(ndim)),
            make_subscript(arg, tuple(f"d_{i}" for i in range(ndim))),
        )
        substitutions[subst_name] = subst

    # }}}

    # {{{ start maintaining the inames involved.

    istepxindex_to_iname: dict[tuple[int, str], str] = {}
    iname_to_ubound: dict[str, ShapeComponentT] = {}

    for i_step, subscripts in enumerate(schedule.subscripts):
        _, in_idx_sets = _get_out_in_idx_sets(subscripts)
        all_indices = reduce(
            frozenset.union,
            (frozenset(idx_set) for idx_set in in_idx_sets),
            cast("frozenset[str]", frozenset()),
        )
        for idx in sorted(all_indices):
            istepxindex_to_iname[i_step, idx] = vng("i")

    sched_arg_to_shape: dict[Argument, ShapeT] = {}
    for ioperand, einsum_arg in enumerate(einsum.args[0]):
        sched_arg_to_shape[EinsumOperand(ioperand)] = einsum_arg.shape

    for i_step, (result_name, subscripts, operands) in enumerate(
        zip(
            schedule.result_names,
            schedule.subscripts,
            schedule.arguments,
            strict=True,
        )
    ):
        idx_to_length: dict[str, ShapeComponentT] = {}
        out_idx_set, in_idx_sets = _get_out_in_idx_sets(subscripts)

        for idx_set, operand in zip(in_idx_sets, operands, strict=True):
            for idx, axis_len in zip(
                idx_set, sched_arg_to_shape[operand], strict=True
            ):
                assert idx_to_length.setdefault(idx, axis_len) == axis_len

        iname_to_ubound.update(
            {
                istepxindex_to_iname[i_step, idx]: axis_len
                for idx, axis_len in idx_to_length.items()
            }
        )
        sched_arg_to_shape[IntermediateResult(result_name)] = tuple(
            idx_to_length[idx] for idx in out_idx_set
        )
    del sched_arg_to_shape

    # }}}

    # {{{ generate domains.

    domains: list[isl.BasicSet] = []

    for i_step in range(schedule.nsteps):
        inames = tuple(
            sorted(
                {
                    iname
                    for (istepxindex, iname) in istepxindex_to_iname.items()
                    if i_step == istepxindex[0]
                }
            )
        )
        domains.append(
            _get_isl_basic_set({iname: iname_to_ubound[iname] for iname in inames})
        )

    # }}}

    # {{{ generate statements and substitutions

    for args in einsum.args:
        dtypes = {arg.name: arg.dtype for arg in args}
        sched_arg_to_lp_name: dict[Argument, str] = {}

        for j, einsum_arg in enumerate(args):
            sched_arg_to_lp_name[EinsumOperand(j)] = einsum_arg.name

        for i_step, (result_name, operands, subscripts) in enumerate(
            zip(
                schedule.result_names,
                schedule.arguments,
                schedule.subscripts,
                strict=True,
            )
        ):
            lp_name = vng(result_name)
            lp_dtype = np.result_type(
                *[dtypes[sched_arg_to_lp_name[operand]] for operand in operands]
            )
            sched_arg_to_lp_name[IntermediateResult(result_name)] = lp_name
            dtypes[lp_name] = lp_dtype
            if result_name != schedule.result_names[-1]:
                knl_args.append(
                    lp.TemporaryVariable(
                        lp_name,
                        dtype=lp_dtype,
                        shape=lp.auto,
                        address_space=lp.AddressSpace.GLOBAL,
                    )
                )
            else:
                knl_args.append(lp.GlobalArg(lp_name, dtype=lp_dtype, shape=lp.auto))

            out_idx_set, in_idx_sets = _get_out_in_idx_sets(subscripts)
            out_idx_set = tuple(
                istepxindex_to_iname[i_step, idx] for idx in out_idx_set
            )
            in_idx_sets = tuple(
                tuple(istepxindex_to_iname[i_step, idx] for idx in idx_set)
                for idx_set in in_idx_sets
            )
            inames_to_sum_over = reduce(
                frozenset.union,
                (frozenset(idx_set) for idx_set in in_idx_sets),
                cast("frozenset[str]", frozenset()),
            ) - frozenset(out_idx_set)

            lhs = make_subscript(lp_name, out_idx_set)
            rhs: p.ExpressionNode = p.Product(
                tuple(
                    make_access_expr(sched_arg_to_lp_name[operand], in_idx_set)
                    for operand, in_idx_set in zip(
                        operands, in_idx_sets, strict=True
                    )
                )
            )
            if inames_to_sum_over:
                rhs = lp.Reduction(
                    "sum",
                    tuple(p.Variable(iname) for iname in inames_to_sum_over),
                    rhs,
                )

            statements.append(lp.Assignment(lhs, rhs))

            subst_name = _get_input_subst_name(lp_name)
            subst = lp.SubstitutionRule(
                subst_name,
                tuple(f"d_{i}" for i in range(len(out_idx_set))),
                make_subscript(
                    lp_name, tuple(f"d_{i}" for i in range(len(out_idx_set)))
                ),
            )
            substitutions[subst_name] = subst

    # }}}

    t_unit = lp.make_kernel(
        domains,
        statements,
        kernel_data=[*knl_args, ...],
        substitutions=substitutions,
        lang_version=LOOPY_LANG_VERSION,
    )
    return t_unit


def generate_loopy_with_opt_einsum_schedule(
    expr: BatchedEinsum, **opt_einsum_kwargs: Any
) -> "lp.TranslationUnit":
    """
    Returns a :class:`loopy.TranslationUnit` with the
    :class:`~feinsum.einsum.ContractionSchedule` specified via
    :func:`opt_einsum.contract_path`.

    :param opt_einsum_kwargs: kwargs to be passed to
        :func:`~feinsum.einsum.get_opt_einsum_contraction_schedule`.
    """
    return generate_loopy(expr, get_opt_einsum_contraction_schedule(expr))
