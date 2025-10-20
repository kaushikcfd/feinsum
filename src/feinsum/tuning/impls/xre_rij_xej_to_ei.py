import logging
import math
from typing import Any

import loopy as lp

import feinsum as fnsm
import feinsum.loopy_utils as lp_utils
from feinsum.tuning import IntParameter

logger = logging.getLogger(__name__)


@fnsm.tuning.einsum_arg("ndim", lambda e: e.args[0][0].shape)
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(2, 32))
@fnsm.tuning.transform_param(
    "nwork_items_per_e", lambda e: IntParameter(1, e.shape[1])
)
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    ndim: int,
    ndof: int,
    n_e_per_wg: int,
    nwork_items_per_e: int,
    i_tiles: int,
    j_tiles: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    from loopy.match import parse_match

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = parse_match(insn_match)
    knl = t_unit[kernel_name]
    (insn_id,) = [insn.id for insn in knl.instructions if within(knl, insn)]
    del knl

    ref_einsum = fnsm.einsum(
        "xre,rij,xej->ei",
        fnsm.array("J", (ndim, ndim, "Nel"), "float64"),
        fnsm.array("D", (ndim, ndof, ndof), "float64"),
        fnsm.array("u", (ndim, "Nel", ndof), "float64"),
    )

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match, kernel_name=kernel_name
    )
    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    D = subst_map["D"]
    u = subst_map["u"]
    x = subst_map["x"]
    r = subst_map["r"]

    j_inner, j_tile = vng(f"{j}_inner"), vng(f"{j}_tile")
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
    i_inner, i_tile = vng(f"{i}_inner"), vng(f"{i}_tile")
    i_inner_inner, i_inner_outer = (vng(f"{i_inner}_inner"), vng(f"{i_inner}_outer"))
    rprftchD, iprftchD, jprftchD = (
        vng(f"{r}prftchD"),
        vng(f"{i}prftchD"),
        vng(f"{j}prftchD"),
    )
    D_fetch = vng(f"{D}_fetch")
    prcmpt_x_redn = ing(f"prcmpt_{x}_redn")
    e_prcmpt_subst, r_prcmpt_subst, j_prcmpt_subst = (
        vng(f"{e}prcmpt_subst"),
        vng(f"{r}prcmpt_subst"),
        vng(f"{j}prcmpt_subst"),
    )

    j_prcmpt_subst_inner, j_prcmpt_subst_outer = (
        vng(f"{j_prcmpt_subst}_inner"),
        vng(f"{j_prcmpt_subst}_outer"),
    )

    # }}}

    # {{{ term hoisting to match the flop count of opt_einsum

    from loopy.symbolic import get_dependencies
    from pymbolic import variables

    knl = t_unit[kernel_name]

    knl = lp.split_reduction_inward(knl, x)
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(knl, x)
    knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
        knl,
        subst_name="subst",
        arguments=variables(f"{e} {j} {r}"),
        within=insn_match,
        terms_filter=lambda x: (
            (get_dependencies(x) & knl.all_inames()) <= {r, e, j}
        ),
    )

    knl = lp.split_iname(
        knl, j, math.ceil(ndof / j_tiles), outer_iname=j_tile, inner_iname=j_inner
    )

    t_unit = t_unit.with_kernel(knl)

    # }}}

    t_unit = lp.split_iname(
        t_unit,
        e,
        n_e_per_wg,
        inner_iname=e_inner,
        outer_iname=e_outer,
        inner_tag="l.1",
        outer_tag="g.0",
    )

    if 0:
        # TODO: This is valid only if len(i_inner_inner) is greater than
        # len(iprftch_D_inner). Enable this after fixing that.
        J = subst_map["J"]
        t_unit = lp.precompute(
            t_unit,
            J,
            sweep_inames=[x, r],
            precompute_outer_inames=frozenset({e_outer, e_inner, i_inner_inner}),
            temporary_address_space=lp.AddressSpace.PRIVATE,
            default_tag="unr",
            within=within,
        )

    # {{{ tile and prefetch D

    t_unit = lp.split_iname(
        t_unit,
        i,
        math.ceil(ndof / i_tiles),
        inner_iname=i_inner,
        outer_iname=i_tile,
        outer_tag="unr",
    )
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        [i_inner, r, j_inner],
        precompute_outer_inames=frozenset([e_outer, i_tile, j_tile]),
        precompute_inames=[rprftchD, iprftchD, jprftchD],
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=D_fetch,
        within=within,
        default_tag=None,
    )
    t_unit = lp.split_iname(t_unit, jprftchD, n_e_per_wg, inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, iprftchD, nwork_items_per_e, inner_tag="l.0")

    # }}}

    # {{{ precompute 'subst'

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        "subst",
        sweep_inames=[r, j_inner, e_inner],
        precompute_inames=[e_prcmpt_subst, j_prcmpt_subst, r_prcmpt_subst],
        precompute_outer_inames=frozenset({e_outer, j_tile}),
        default_tag=None,
        compute_insn_id=prcmpt_x_redn,
        temporary_address_space=lp.AddressSpace.LOCAL,
    )
    t_unit = lp.tag_inames(t_unit, {e_prcmpt_subst: "l.1"})

    # TODO: It might be worth exploring joining 'r_prcmpt_subst',
    # 'j_prcmpt_subst'.

    t_unit = lp.split_iname(
        t_unit,
        j_prcmpt_subst,
        nwork_items_per_e,
        inner_iname=j_prcmpt_subst_inner,
        outer_iname=j_prcmpt_subst_outer,
        inner_tag="l.0",
        outer_tag="unr",
    )

    # }}}

    t_unit = lp.split_iname(
        t_unit,
        i_inner,
        nwork_items_per_e,
        inner_iname=i_inner_inner,
        outer_iname=i_inner_outer,
        inner_tag="l.0",
        outer_tag="unr",
    )

    u_fetch = vng(f"{u}_fetch")
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        u,
        sweep_inames=[x, j_prcmpt_subst_outer],
        precompute_outer_inames=frozenset(
            [j_prcmpt_subst_inner, e_prcmpt_subst, e_outer, j_tile]
        ),
        precompute_inames=[vng("prftch_u_x"), None, vng("prftch_u_j")],
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_name=u_fetch,
        default_tag="unr",
    )

    # {{{ TODO: remove once github.com/inducer/loopy/issues/666 is resolved.

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_id)
    inames_to_duplicate = (
        frozenset({i_tile, i_inner_outer}) & t_unit[kernel_name].all_inames()
    )
    acc_name = f"acc_{r}_{j_tile}_{j_inner}"
    assert acc_name in t_unit[kernel_name].temporary_variables
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

    # }}}

    t_unit = lp.tag_inames(t_unit, {r: "unr", x: "unr"})

    return t_unit


if __name__ == "__main__":
    import os
    from functools import partial

    import pyopencl as cl

    Ndim = 3
    Ndof = 35

    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum(
        "xre,rij,xej->ei",
        fnsm.array("J", (Ndim, Ndim, "Nel"), "float64"),
        fnsm.array("D", (Ndim, Ndof, Ndof), "float64"),
        fnsm.array("u", (Ndim, "Nel", Ndof), "float64"),
    )

    if 1:
        fnsm.autotune(expr, os.path.abspath(__file__), cl_ctx)
    else:
        # Enable while debugging ->
        # evaluate a point in the parameter space.
        bound_transform = partial(
            transform,
            ndim=Ndim,
            ndof=Ndof,
            n_e_per_wg=27,
            nwork_items_per_e=9,
            i_tiles=4,
            j_tiles=4,
        )

        print(
            fnsm.stringify_comparison_vs_roofline(
                expr, transform=bound_transform, cl_ctx=cl_ctx
            )
        )

# vim: fdm=marker
