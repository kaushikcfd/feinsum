from typing import Any

import loopy as lp
import loopy.match as lp_match

import feinsum as fnsm
from feinsum.tuning import IntParameter


@fnsm.tuning.einsum_arg("ndofs", lambda e: e.shape[1])
@fnsm.tuning.transform_param(
    "nworkitems_per_e", lambda e: IntParameter(1, e.shape[1])
)
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(1, 32))
def transform(
    t_unit: lp.TranslationUnit,
    ndofs: int,
    n_e_per_wg: int,
    nworkitems_per_e: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    within = lp_match.parse_match(insn_match)
    (insn_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if within(t_unit[kernel_name], insn)
    ]

    ref_einsum = fnsm.einsum(
        "ij,ej->ei", fnsm.array("D", (ndofs, ndofs)), fnsm.array("u", ("Nel", ndofs))
    )
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, kernel_name=kernel_name, insn_match=insn_match
    )
    vng = t_unit[kernel_name].get_var_name_generator()
    ing = t_unit[kernel_name].get_instruction_id_generator()

    e = subst_map["e"]
    i = subst_map["i"]
    j = subst_map["j"]
    i_inner, i_outer = vng(f"{i}_inner"), vng(f"{i}_outer")
    e_outer = vng(f"{e}_outer")
    # out = subst_map["_fe_out"]
    u = subst_map["u"]
    D = subst_map["D"]
    uprftch_insn_id = ing("u_prftch_id")
    Dprftch_insn_id = ing("D_prftch_id")
    jprftch_u = vng("jprftch_u")
    iprftch_D, jprftch_D = vng("iprftch_D"), vng("jprftch_D")

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        u,
        sweep_inames=[j],
        precompute_outer_inames=frozenset([e]),
        compute_insn_id=uprftch_insn_id,
        precompute_inames=(
            None,
            jprftch_u,
        ),
        default_tag="unr",
        temporary_address_space=lp.AddressSpace.PRIVATE,
        within=within,
    )

    t_unit = lp.split_iname(
        t_unit, e, n_e_per_wg, outer_iname=e_outer, inner_tag="l.1", outer_tag="g.0"
    )

    # {{{ prefetch "D"

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        sweep_inames=[i, j],
        precompute_outer_inames=frozenset([e_outer]),
        compute_insn_id=Dprftch_insn_id,
        precompute_inames=(iprftch_D, jprftch_D),
        default_tag=None,
        temporary_address_space=lp.AddressSpace.LOCAL,
        within=within,
    )
    t_unit = lp.split_iname(t_unit, iprftch_D, nworkitems_per_e, inner_tag="l.0")
    t_unit = lp.split_iname(t_unit, jprftch_D, n_e_per_wg, inner_tag="l.1")

    # }}}

    t_unit = lp.split_iname(
        t_unit,
        i,
        nworkitems_per_e,
        outer_iname=i_outer,
        inner_iname=i_inner,
        inner_tag="l.0",
        outer_tag="unr",
    )

    t_unit = lp.add_inames_to_insn(
        t_unit, frozenset([i_inner]), insn_match=lp_match.Id(uprftch_insn_id)
    )
    t_unit = lp.realize_reduction(t_unit, insn_id_filter=frozenset([insn_id]))
    (acc_name,) = t_unit[kernel_name].id_to_insn[
        insn_id
    ].read_dependency_names() & set(t_unit[kernel_name].temporary_variables)
    inames_to_duplicate = frozenset([i_outer])
    assert acc_name.startswith("acc")
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, inames_to_duplicate, only_var_names={acc_name}
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"writes:{acc_name} and not reads:{acc_name}",
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"reads:{acc_name} and not writes:{acc_name}",
    )
    t_unit = lp.tag_inames(t_unit, {j: "unr"})

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    Ndof = 15
    Nfields = 1

    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum(
        "ij,ej->ei", fnsm.array("D", (Ndof, Ndof)), fnsm.array("u", ("Nel", Ndof))
    )

    fnsm.autotune(expr, os.path.abspath(__file__), cl_ctx)

# vim: fdm=marker
