import feinsum as fnsm
from feinsum.tuning import IntParameter, BoolParameter
import numpy as np
import loopy as lp
import math
import logging

logger = logging.getLogger(__name__)


@fnsm.tuning.einsum_arg(
    "ndim", lambda e: e.shape[0])
@fnsm.tuning.einsum_arg(
    "ndof", lambda e: e.shape[2])
@fnsm.tuning.transform_param(
    "n_e_per_wg", lambda e: IntParameter(2, 32))
@fnsm.tuning.transform_param(
    "nwork_items_per_e", lambda e: IntParameter(1, e.shape[2]+1))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[2] / 2)))
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[2] / 2)))
@fnsm.tuning.transform_param(
    "prftch_u_to_local", lambda e: BoolParameter())
def transform(t_unit,
              ndim, ndof,
              n_e_per_wg, nwork_items_per_e,
              prftch_u_to_local, i_tiles, j_tiles,
              insn_match=None, kernel_name=None):

    if n_e_per_wg * nwork_items_per_e > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    if ((math.ceil((ndof*ndim)/i_tiles) * math.ceil(ndof/j_tiles))
            + int(prftch_u_to_local) * ndof * n_e_per_wg
            + ndim * ndof * n_e_per_wg)*8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

    from loopy.match import parse_match
    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = parse_match(insn_match)
    knl = t_unit[kernel_name]
    insn_id, = [insn.id
                for insn in knl.instructions
                if within(knl, insn)]
    del knl

    ref_einsum = fnsm.einsum("xre,rij,ej->xei",
                             fnsm.array((ndim, ndim, np.inf), "float64"),
                             fnsm.array((ndim, ndof, ndof), "float64"),
                             fnsm.array((np.inf, ndof), "float64"),
                             arg_names=["J", "D", "u"])

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_var_name_generator()
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match)
    i = subst_map["i"]
    j = subst_map["j"]
    J = subst_map["J"]
    e = subst_map["e"]
    D = subst_map["D"]
    u = subst_map["u"]
    x = subst_map["x"]
    r = subst_map["r"]
    e_inner, e_outer = f"{e}_inner", f"{e}_outer"
    u_fetch = vng(u+"_prftch")
    j_inner = vng(f"{j}_inner")
    j_tile = f"{j}_tile"
    i_tile, i_inner = f"{i}_tile", f"{i}_inner"
    e_prcmpt_subst = vng(f"{e}_prcmpt")
    r_prcmpt_subst = vng(f"{r}_prcmpt")
    i_prcmpt_subst = vng(f"{i}_prcmpt")
    rprftch_D, iprftch_D, jprftch_D = (vng("rprftchD"),
                                       vng("iprftchD"),
                                       vng("jprftchD"))

    prcmpt_j_redn = ing(f"prcmpt_{j}_redn")
    D_fetch = vng(f"{D}_fetch")
    J_fetch = vng(f"{J}_fetch")
    i_inner_inner, i_inner_outer = vng(f"{i_inner}_inner"), vng(f"{i_inner}_outer")
    # prftch_J = ing(f"prftch_{J}")

    # }}}

    # {{{ term hoisting to match the flop count of opt_einsum

    t_unit = lp.split_reduction_inward(t_unit, j)
    t_unit = fnsm.hoist_reduction_invariant_terms(t_unit, j)
    t_unit = fnsm.extract_einsum_terms_as_subst(
        t_unit,
        f"subst({e}, {i}, {r})",
        f"sum({j}, {D}[{r}, {i}, {j}]*{u}[{e}, {j}])",
        insn_match=insn_match
    )

    t_unit = lp.split_iname(t_unit, i, math.ceil(ndof/i_tiles),
                            outer_iname=i_tile, inner_iname=i_inner)

    # }}}

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg,
                            inner_iname=e_inner, outer_iname=e_outer,
                            inner_tag="l.1", outer_tag="g.0")

    # {{{ prefetch 'u'

    if prftch_u_to_local:
        eprftch_u, jprftch_u = vng("eprftch_u"), vng("jprftch_u")
        t_unit = lp.add_prefetch(t_unit, u,
                                 sweep_inames=[e_inner, j],
                                 fetch_outer_inames=frozenset([e_outer]),
                                 temporary_address_space=lp.AddressSpace.LOCAL,
                                 temporary_name=u_fetch,
                                 dim_arg_names=[eprftch_u, jprftch_u],
                                 default_tag=None,
                                 within=within
                                 )
        t_unit = lp.tag_inames(t_unit, {eprftch_u: "l.1"})
        t_unit = lp.split_iname(t_unit, jprftch_u, nwork_items_per_e,
                                inner_tag="l.0")
    else:
        t_unit = lp.add_prefetch(t_unit, u,
                                 sweep_inames=[j],
                                 fetch_outer_inames=frozenset([e_inner,
                                                               e_outer]),
                                 temporary_address_space=lp.AddressSpace.PRIVATE,
                                 temporary_name=u_fetch,
                                 default_tag="unr",
                                 within=within
                                 )

    # }}}

    t_unit = lp.add_prefetch(t_unit, J,
                             sweep_inames=[x, r],
                             fetch_outer_inames=frozenset({e_outer, e_inner,
                                                           i_inner_inner}),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name=J_fetch,
                             default_tag="unr",
                             within=within)

    # {{{ tile and prefetch D

    t_unit = lp.split_iname(t_unit, j, math.ceil(ndof/j_tiles),
                            inner_iname=j_inner, outer_iname=j_tile,
                            inner_tag="unr", outer_tag="unr")
    t_unit = lp.add_prefetch(t_unit, D, [i_inner, r, j_inner],
                             fetch_outer_inames=frozenset([e_outer,
                                                           i_tile,
                                                           j_tile]),
                             dim_arg_names=[rprftch_D, iprftch_D, jprftch_D],
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             temporary_name=D_fetch,
                             default_tag=None)
    t_unit = lp.split_iname(t_unit, iprftch_D, n_e_per_wg, inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, jprftch_D, nwork_items_per_e, inner_tag="l.0")

    # }}}

    # {{{ precompute 'subst'

    t_unit = lp.split_iname(t_unit, i_inner, nwork_items_per_e,
                            inner_iname=i_inner_inner,
                            outer_iname=i_inner_outer,
                            inner_tag="l.0", outer_tag="unr",
                            )

    t_unit = lp.precompute(t_unit, "subst",
                           sweep_inames=[r, i_inner_outer],
                           precompute_inames=[e_prcmpt_subst,
                                              i_prcmpt_subst,
                                              r_prcmpt_subst],
                           # storage_axes=[0, 1],
                           precompute_outer_inames=frozenset({e_inner,
                                                              e_outer,
                                                              i_tile,
                                                              i_inner_inner}),
                           default_tag="unr",
                           compute_insn_id=prcmpt_j_redn,
                           temporary_address_space=lp.AddressSpace.PRIVATE)

    # }}}

    # {{{ TODO: remove once github.com/inducer/loopy/issues/666 is resolved.

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_j_redn)
    inames_to_duplicate = (frozenset({i_prcmpt_subst,
                                      r_prcmpt_subst})
                           & t_unit[kernel_name].all_inames())

    acc_name = f"acc_{j_tile}_{j_inner}"
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  inames_to_duplicate,
                                                  only_var_names={acc_name})

    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"writes:{acc_name} and not reads:{acc_name}")
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"reads:{acc_name} and not writes:{acc_name}")

    # }}}

    t_unit = lp.tag_inames(t_unit, {r: "unr"})

    if not prftch_u_to_local:
        # TODO: Yet another headache to ensure that the fetch instruction uses all
        # the hw axes.
        t_unit = lp.add_inames_to_insn(t_unit,
                                       i_inner_inner,
                                       f"writes:{u_fetch}")

    return t_unit


if __name__ == "__main__":
    import pyopencl as cl
    import os
    from functools import partial

    Ndim = 3
    Ndof = 35

    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum("xre,rij,ej->xei",
                       fnsm.array((Ndim, Ndim, np.inf), "float64"),
                       fnsm.array((Ndim, Ndof, Ndof), "float64"),
                       fnsm.array((np.inf, Ndof), "float64"),
                       arg_names=["J", "D", "u"])

    if 1:
        fnsm. autotune(expr, os.path.abspath(__file__), cl_ctx)
    else:
        # Enable while debugging ->
        # evaluate a point in the parameter space.
        bound_transform = partial(transform,
                                  ndim=Ndim, ndof=Ndof,
                                  n_e_per_wg=27,
                                  nwork_items_per_e=9,
                                  i_tiles=4, j_tiles=4,
                                  prftch_u_to_local=True)

        print(fnsm.stringify_comparison_with_roofline(expr,
                                                      bound_transform,
                                                      cl_ctx))

# vim: fdm=marker
