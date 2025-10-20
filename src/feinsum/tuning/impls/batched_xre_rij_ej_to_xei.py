import logging
import math
from typing import TYPE_CHECKING, Any, cast

import islpy as isl
import loopy as lp
import loopy.match as lp_match
from more_itertools import chunked
from more_itertools import zip_equal as szip

import feinsum as fnsm
import feinsum.loopy_utils as lp_utils
from feinsum.tuning import IntParameter

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


@fnsm.tuning.einsum_arg("noutputs", lambda e: e.b)
@fnsm.tuning.einsum_arg("ndim", lambda e: e.shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[2])
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(2, 32))
@fnsm.tuning.transform_param(
    "nwork_items_per_e", lambda e: IntParameter(1, e.shape[2])
)
@fnsm.tuning.transform_param(
    "n_stmt_tile", lambda e: IntParameter(1, math.ceil(e.b))
)
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[2] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[2] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    noutputs: int,
    ndim: int,
    ndof: int,
    n_e_per_wg: int,
    nwork_items_per_e: int,
    n_stmt_tile: int,
    i_tiles: int,
    j_tiles: int,
    # FIXME: Making this is BoolParameters leads to an error in validation.
    prftch_u_to_local: bool = False,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:

    if n_e_per_wg * nwork_items_per_e > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    if (
        (ndim * math.ceil((ndof) / i_tiles) * math.ceil(ndof / j_tiles))
        + int(prftch_u_to_local)
        * ndof
        * n_e_per_wg
        * math.ceil(noutputs / n_stmt_tile)
    ) * 8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

    from loopy.match import parse_match
    from pymbolic import variables

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = parse_match(insn_match)

    ref_einsum = fnsm.batched_einsum(
        "xre,rij,ej->xei",
        [
            [
                fnsm.array("J", (ndim, ndim, "Nel")),
                fnsm.array("D", (ndim, ndof, ndof)),
                fnsm.array(f"u{i}", ("Nel", ndof)),
            ]
            for i in range(noutputs)
        ],
    )

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match, kernel_name=kernel_name
    )
    i = subst_map["i"]
    j = subst_map["j"]
    J = subst_map["J"]
    e = subst_map["e"]
    D = subst_map["D"]
    x = subst_map["x"]
    r = subst_map["r"]
    e_inner, e_outer = f"{e}_inner", f"{e}_outer"
    us = tuple(subst_map[f"u{i}"] for i in range(noutputs))
    ref_outputs = ["_fe_out"] + [
        f"_fe_out_{ioutput}" for ioutput in range(noutputs - 1)
    ]
    outputs = tuple(subst_map[ref_output] for ref_output in ref_outputs)
    del ref_outputs
    # prftch_J = ing(f"prftch_{J}")

    # }}}

    # {{{ get new names

    n_stmt_tile = math.ceil(noutputs / math.ceil(noutputs / n_stmt_tile))

    new_is = [vng("i") for _ in range(n_stmt_tile)]
    new_js = [vng("j") for _ in range(n_stmt_tile)]
    new_xs = [vng("x") for _ in range(n_stmt_tile)]
    new_rs = [vng("r") for _ in range(n_stmt_tile)]
    new_es = [vng("e") for _ in range(n_stmt_tile)]
    i_stmt_tile_to_fields = tuple(
        list(el) for el in chunked(us, math.ceil(noutputs / n_stmt_tile))
    )
    i_stmt_tile_to_outputs = tuple(
        list(el) for el in chunked(outputs, math.ceil(noutputs / n_stmt_tile))
    )

    ufetch_ids = tuple(
        tuple(ing("u_fetch_insn") for _ in fields)
        for fields in i_stmt_tile_to_fields
    )

    # }}}

    for new_i, new_j, new_x, new_r, new_e, outputs_in_tile in szip(
        new_is, new_js, new_xs, new_rs, new_es, i_stmt_tile_to_outputs
    ):
        t_unit = lp.duplicate_inames(
            t_unit,
            [i, j, x, r, e],
            within=lp_match.And(
                (
                    within,
                    lp_match.Or(
                        tuple(lp_match.Writes(output) for output in outputs_in_tile)
                    ),
                )
            ),
            new_inames=[new_i, new_j, new_x, new_r, new_e],
        )
        t_unit = t_unit.with_kernel(
            lp_utils.decouple_domain(
                t_unit[kernel_name],
                [new_i, new_j, new_x, new_r, new_e],
                parent_inames=cast(
                    "Collection[str]",
                    t_unit[kernel_name]
                    .get_inames_domain(e)
                    .get_var_names(isl.dim_type.param),
                ),
            )
        )

    t_unit = lp.remove_unused_inames(t_unit)
    D_fetch = vng(f"{D}_fetch")
    D_fetch_ids = tuple(ing("D_fetch_id") for _ in range(n_stmt_tile))

    if 0:
        # TODO: Assumes that len(i_inner_inner) > len(jprftch_D_inner).
        # Re-enable this after that ambiguity is fixed.
        J_fetch = vng(f"{J}_fetch")
        J_prftch_0, J_prftch_1 = vng(f"{J}_prftch_x"), vng(f"{J}_prftch_r")
        t_unit = lp.precompute(
            t_unit,
            J,
            sweep_inames=[x, r],
            precompute_outer_inames=frozenset({}),
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_name=J_fetch,
            precompute_inames=(J_prftch_0, J_prftch_1),
            default_tag=None,
            within=within,
        )

    for istmt_tile in range(n_stmt_tile):

        i = new_is[istmt_tile]
        j = new_js[istmt_tile]
        x = new_xs[istmt_tile]
        r = new_rs[istmt_tile]
        e = new_es[istmt_tile]

        # {{{ set var names

        j_inner = vng(f"{j}_inner")
        j_tile = f"{j}_tile"
        i_tile, i_inner = f"{i}_tile", f"{i}_inner"
        e_prcmpt_subst = vng(f"{e}_prcmpt")
        r_prcmpt_subst = vng(f"{r}_prcmpt")
        i_prcmpt_subst = vng(f"{i}_prcmpt")
        rprftch_D, iprftch_D, jprftch_D = (
            vng("rprftchD"),
            vng("iprftchD"),
            vng("jprftchD"),
        )

        prcmpt_j_redns = tuple(
            ing(f"prcmpt_{j}_redn") for _ in i_stmt_tile_to_fields[istmt_tile]
        )
        i_inner_inner, i_inner_outer = (
            vng(f"{i_inner}_inner"),
            vng(f"{i_inner}_outer"),
        )

        subst_names = tuple(vng("subst") for _ in i_stmt_tile_to_fields[istmt_tile])

        # }}}

        # {{{ term hoisting to match the flop count of opt_einsum

        knl = t_unit[kernel_name]
        knl = lp.split_reduction_inward(knl, j)
        knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(knl, j)

        for subst_name, u, output in szip(
            subst_names,
            i_stmt_tile_to_fields[istmt_tile],
            i_stmt_tile_to_outputs[istmt_tile],
        ):
            knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
                knl,
                subst_name=subst_name,
                arguments=variables(f"{e} {i} {r}"),
                within=lp_match.And((within, lp_match.Writes(output))),
                terms_filter=lambda x: fnsm.get_call_ids(x) <= {D, u},
            )

        t_unit = t_unit.with_kernel(knl)

        t_unit = lp.split_iname(
            t_unit,
            i,
            math.ceil(ndof / i_tiles),
            outer_iname=i_tile,
            inner_iname=i_inner,
        )

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

        # {{{ prefetch 'u'

        if prftch_u_to_local:
            raise NotImplementedError
        else:
            jprftch_u = vng("j_prftch_u")
            for u, ufetch_id in zip(
                i_stmt_tile_to_fields[istmt_tile],
                ufetch_ids[istmt_tile],
                strict=False,
            ):
                t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                    t_unit,
                    u,
                    sweep_inames=[j],
                    precompute_outer_inames=frozenset([e_inner, e_outer]),
                    temporary_address_space=lp.AddressSpace.PRIVATE,
                    temporary_name=vng("u_prftch_var"),
                    default_tag="unr",
                    precompute_inames=(
                        None,
                        jprftch_u,
                    ),
                    compute_insn_id=ufetch_id,
                    within=within,
                )

        # }}}

        # {{{ tile and prefetch D

        t_unit = lp.split_iname(
            t_unit,
            j,
            math.ceil(ndof / j_tiles),
            inner_iname=j_inner,
            outer_iname=j_tile,
            inner_tag="unr",
            outer_tag="unr",
        )
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            D,
            [i_inner, r, j_inner],
            precompute_outer_inames=frozenset([e_outer, i_tile, j_tile]),
            precompute_inames=[rprftch_D, iprftch_D, jprftch_D],
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_name=D_fetch,
            compute_insn_id=D_fetch_ids[istmt_tile],
            within=lp_match.And(
                (
                    within,
                    lp_match.Or(
                        tuple(
                            lp_match.Writes(output)
                            for output in i_stmt_tile_to_outputs[istmt_tile]
                        )
                    ),
                )
            ),
            default_tag=None,
        )
        t_unit = lp.split_iname(t_unit, iprftch_D, n_e_per_wg, inner_tag="l.1")
        t_unit = lp.split_iname(
            t_unit, jprftch_D, nwork_items_per_e, inner_tag="l.0"
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

        # {{{ precompute 'subst'

        for subst_name, prcmpt_j_redn in szip(subst_names, prcmpt_j_redns):
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                subst_name,
                sweep_inames=[r, i_inner_outer],
                precompute_inames=[e_prcmpt_subst, i_prcmpt_subst, r_prcmpt_subst],
                # storage_axes=[0, 1],
                precompute_outer_inames=frozenset(
                    {e_inner, e_outer, i_tile, i_inner_inner}
                ),
                default_tag="unr",
                compute_insn_id=prcmpt_j_redn,
                temporary_name=vng("tmp_hoist_j_redn"),
                temporary_address_space=lp.AddressSpace.PRIVATE,
            )

        # }}}

        # {{{ TODO: remove once github.com/inducer/loopy/issues/666 is resolved.

        t_unit = lp.realize_reduction(
            t_unit, insn_id_filter=frozenset(prcmpt_j_redns)
        )
        inames_to_duplicate = (
            frozenset({i_prcmpt_subst, r_prcmpt_subst})
            & t_unit[kernel_name].all_inames()
        )
        acc_names = set()
        for prcmpt_j_redn in prcmpt_j_redns:
            (acc_name,) = t_unit[kernel_name].id_to_insn[
                prcmpt_j_redn
            ].read_dependency_names() & set(t_unit[kernel_name].temporary_variables)
            assert acc_name.startswith("acc")
            acc_names.add(acc_name)

        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, inames_to_duplicate, only_var_names=acc_names
        )

        t_unit = lp.duplicate_inames(
            t_unit,
            inames_to_duplicate,
            within=lp_match.Or(
                tuple(
                    lp_match.And(
                        (
                            lp_match.Writes(acc_name),
                            lp_match.Not(lp_match.Reads(acc_name)),
                        )
                    )
                    for acc_name in acc_names
                )
            ),
        )
        t_unit = lp.duplicate_inames(
            t_unit,
            inames_to_duplicate,
            within=lp_match.Or(
                tuple(
                    lp_match.And(
                        (
                            lp_match.Reads(acc_name),
                            lp_match.Not(lp_match.Writes(acc_name)),
                        )
                    )
                    for acc_name in acc_names
                )
            ),
        )

        # }}}

        if not prftch_u_to_local:
            # TODO: Yet another headache to ensure that the fetch instruction uses
            # all the hw axes.
            for ufetch_id in ufetch_ids[istmt_tile]:
                t_unit = lp.add_inames_to_insn(
                    t_unit, i_inner_inner, lp_match.Id(ufetch_id)
                )

    if prftch_u_to_local:
        # add dependencies correctly.
        raise NotImplementedError

    for istmt_tile in range(1, n_stmt_tile):
        preceding_outputs = i_stmt_tile_to_outputs[istmt_tile - 1]
        t_unit = lp.add_dependency(
            t_unit,
            insn_match=lp_match.Or(
                (
                    lp_match.Id(D_fetch_ids[istmt_tile]),
                    lp_match.Or(
                        tuple(
                            lp_match.Id(ufetch_id)
                            for ufetch_id in ufetch_ids[istmt_tile]
                        )
                    ),
                )
            ),
            depends_on=lp_match.And(
                (
                    within,
                    lp_match.Or(
                        tuple(
                            lp_match.Writes(output) for output in preceding_outputs
                        )
                    ),
                )
            ),
        )

    return t_unit


if __name__ == "__main__":
    import os
    from functools import partial

    import pyopencl as cl

    Ndim = 3
    Ndof = 4
    Nfields = 5

    cl_ctx = cl.create_some_context()

    expr = fnsm.batched_einsum(
        "xre,rij,ej->xei",
        [
            [
                fnsm.array("J", (Ndim, Ndim, "Nel")),
                fnsm.array("D", (Ndim, Ndof, Ndof)),
                fnsm.array(f"u{i}", ("Nel", Ndof)),
            ]
            for i in range(Nfields)
        ],
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
            noutputs=Nfields,
            n_e_per_wg=21,
            nwork_items_per_e=12,
            n_stmt_tile=Nfields,
            i_tiles=3,
            j_tiles=1,
            prftch_u_to_local=False,
        )

        print(
            fnsm.stringify_comparison_vs_roofline(
                expr, transform=bound_transform, cl_ctx=cl_ctx
            )
        )

# vim: fdm=marker
