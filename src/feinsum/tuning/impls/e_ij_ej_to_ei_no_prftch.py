from typing import Any

import loopy as lp
import loopy.match as lp_match

import feinsum as fnsm
import feinsum.loopy_utils as lp_utils
from feinsum.tuning import IntParameter


@fnsm.tuning.einsum_arg("noutputs", lambda e: e.b)
@fnsm.tuning.einsum_arg("ndofs", lambda e: e.shape[1])
@fnsm.tuning.transform_param("nworkitems_per_e", lambda e: IntParameter(8, 8))
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(4, 4))
def transform(
    t_unit: lp.TranslationUnit,
    ndofs: int,
    noutputs: int,
    n_e_per_wg: int,
    nworkitems_per_e: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    if n_e_per_wg * nworkitems_per_e > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.batched_einsum(
        "e,ij,ej->ei",
        [
            [
                fnsm.array("J", "Nel"),
                fnsm.array("D", (ndofs, ndofs)),
                fnsm.array(f"u{i}", ("Nel", ndofs)),
            ]
            for i in range(noutputs)
        ],
    )
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, kernel_name=kernel_name, insn_match=within
    )
    vng = t_unit[kernel_name].get_var_name_generator()

    e = subst_map["e"]
    i = subst_map["i"]
    j = subst_map["j"]
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
    i_inner, i_outer = vng(f"{i}_inner"), vng(f"{i}_outer")
    J = subst_map["J"]
    ref_outs = ["_fe_out"] + [f"_fe_out_{iout}" for iout in range(noutputs - 1)]
    outs = tuple(subst_map[ref_out] for ref_out in ref_outs)
    del ref_outs

    t_unit = t_unit.with_kernel(
        lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(
            t_unit[kernel_name], j
        )
    )

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg, inner_tag="l.1", outer_tag="g.0")
    t_unit = lp.split_iname(
        t_unit, i, nworkitems_per_e, inner_tag="l.0", outer_tag="unr"
    )

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        J,
        sweep_inames=[],
        precompute_outer_inames=frozenset([e_outer, e_inner, i_inner]),
        default_tag="unr",
        within=within,
        temporary_address_space=lp.AddressSpace.PRIVATE,
    )

    for out in outs[1:]:
        t_unit = lp.duplicate_inames(
            t_unit,
            (i_outer, j),
            within=lp_match.And((within, lp_match.Writes(out))),
            tags={
                i_outer: t_unit[kernel_name].inames[i_outer].tags,
                j: t_unit[kernel_name].inames[j].tags,
            },
        )
    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    Ndof = 4
    Nfields = 16

    cl_ctx = cl.create_some_context()

    expr = fnsm.batched_einsum(
        "e,ij,ej->ei",
        [
            [
                fnsm.array("J", "Nel"),
                fnsm.array("D", (Ndof, Ndof)),
                fnsm.array(f"u{i}", ("Nel", Ndof)),
            ]
            for i in range(Nfields)
        ],
    )

    fnsm.autotune(expr, os.path.abspath(__file__), cl_ctx)

# vim: fdm=marker
