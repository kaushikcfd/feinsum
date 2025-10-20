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
@fnsm.tuning.einsum_arg("ndim", lambda e: e.args[0][0].shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(2, 32))
@fnsm.tuning.transform_param(
    "nwork_items_per_e", lambda e: IntParameter(1, e.shape[1])
)
@fnsm.tuning.transform_param("n_stmt_tile", lambda e: IntParameter(1, e.b))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 3))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 3))
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
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:

    if n_e_per_wg * nwork_items_per_e > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    if (
        math.ceil(noutputs / n_stmt_tile)
        * ndim
        * n_e_per_wg
        * math.ceil(ndof / j_tiles)
        + ndim * math.ceil(ndof / i_tiles) * math.ceil(ndof / j_tiles)
    ) * 8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

    from loopy.match import parse_match

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = parse_match(insn_match)
    ref_outputs = ["_fe_out"] + [
        f"_fe_out_{ioutput}" for ioutput in range(noutputs - 1)
    ]

    ref_einsum = fnsm.batched_einsum(
        "xre,rij,xej->ei",
        [
            [
                fnsm.array("J", (ndim, ndim, "Nel")),
                fnsm.array("D", (ndim, ndof, ndof)),
                fnsm.array(f"u{i}", (ndim, "Nel", ndof)),
            ]
            for i in range(noutputs)
        ],
    )

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=within, kernel_name=kernel_name
    )
    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    D = subst_map["D"]
    us = tuple(subst_map[f"u{ioutput}"] for ioutput in range(noutputs))
    outputs = tuple(subst_map[ref_output] for ref_output in ref_outputs)
    x = subst_map["x"]
    r = subst_map["r"]

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
    hoisted_tmp_names = tuple(
        vng("J_times_u_fetch") for _ in i_stmt_tile_to_fields[0]
    )
    D_prftch_id = tuple(ing("D_fetch_insn") for _ in range(n_stmt_tile))

    prcmpt_x_redn_ids = tuple(
        tuple(ing(f"prcmpt_{x}_redn") for _ in fields)
        for fields in i_stmt_tile_to_fields
    )

    for istmt_tile in range(n_stmt_tile):
        i = new_is[istmt_tile]
        j = new_js[istmt_tile]
        x = new_xs[istmt_tile]
        r = new_rs[istmt_tile]
        e = new_es[istmt_tile]
        outputs_in_tile = i_stmt_tile_to_outputs[istmt_tile]
        fields_in_tile = i_stmt_tile_to_fields[istmt_tile]
        subst_names = [vng("hoist_subst") for _ in fields_in_tile]

        knl = t_unit[kernel_name]
        insn_ids = [
            insn.id
            for insn in knl.instructions
            if within(knl, insn)
            and (frozenset(outputs_in_tile) & insn.write_dependency_names())
        ]
        assert len(insn_ids) == len(outputs_in_tile)
        del knl

        # {{{ define variables

        j_inner, j_tile = vng(f"{j}_inner"), vng(f"{j}_tile")
        e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
        i_inner, i_tile = vng(f"{i}_inner"), vng(f"{i}_tile")
        i_inner_inner, i_inner_outer = (
            vng(f"{i_inner}_inner"),
            vng(f"{i_inner}_outer"),
        )
        rprftchD, iprftchD, jprftchD = (
            vng(f"{r}prftchD"),
            vng(f"{i}prftchD"),
            vng(f"{j}prftchD"),
        )
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
        for subst_name, output in szip(subst_names, outputs_in_tile):
            knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
                knl,
                subst_name=subst_name,
                arguments=variables(f"{e} {j} {r}"),
                within=lp_match.And((within, lp_match.Writes(output))),
                terms_filter=lambda x: (
                    (get_dependencies(x) & knl.all_inames()) <= {r, e, j}
                ),
            )

        knl = lp.split_iname(
            knl,
            j,
            math.ceil(ndof / j_tiles),
            outer_iname=j_tile,
            inner_iname=j_inner,
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
            compute_insn_id=D_prftch_id[istmt_tile],
            within=lp_match.And(
                (
                    within,
                    lp_match.Or(
                        tuple(lp_match.Writes(out) for out in outputs_in_tile)
                    ),
                )
            ),
            default_tag=None,
        )
        t_unit = lp.split_iname(t_unit, jprftchD, n_e_per_wg, inner_tag="l.1")
        t_unit = lp.split_iname(t_unit, iprftchD, nwork_items_per_e, inner_tag="l.0")

        # }}}

        # {{{ precompute hoisted substs

        for isubst, (subst_name, tmp_name) in enumerate(
            zip(subst_names, hoisted_tmp_names, strict=False)
        ):
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                subst_name,
                sweep_inames=[r, j_inner, e_inner],
                precompute_inames=[e_prcmpt_subst, j_prcmpt_subst, r_prcmpt_subst],
                precompute_outer_inames=frozenset({e_outer, j_tile}),
                default_tag=None,
                temporary_name=tmp_name,
                compute_insn_id=prcmpt_x_redn_ids[istmt_tile][isubst],
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

        for field in fields_in_tile:
            u_fetch = vng("u_fetch")
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                field,
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

        t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_ids)
        inames_to_duplicate = (
            frozenset({i_tile, i_inner_outer}) & t_unit[kernel_name].all_inames()
        )
        acc_names = set()

        for insn_id in insn_ids:
            (acc_name,) = t_unit[kernel_name].id_to_insn[
                insn_id
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

        t_unit = lp.tag_inames(t_unit, {r: "unr", x: "unr"})

    for istmt_tile in range(1, n_stmt_tile):
        predecessor_match = lp_match.Or(
            tuple(
                lp_match.Writes(output)
                for output in i_stmt_tile_to_outputs[istmt_tile - 1]
            )
        )
        for ifield, _ in enumerate(i_stmt_tile_to_fields[istmt_tile]):
            successor_match = lp_match.Or(
                (
                    lp_match.Id(prcmpt_x_redn_ids[istmt_tile][ifield]),
                    lp_match.Id(D_prftch_id[istmt_tile]),
                )
            )

            t_unit = lp.add_dependency(
                t_unit,
                insn_match=successor_match,
                depends_on=lp_match.And((within, predecessor_match)),
            )

    return t_unit


if __name__ == "__main__":
    import os
    from functools import partial

    import pyopencl as cl

    Ndim = 3
    Ndof = 4
    Nfield = 5

    cl_ctx = cl.create_some_context()

    expr = fnsm.batched_einsum(
        "xre,rij,xej->ei",
        [
            [
                fnsm.array("J", (Ndim, Ndim, "Nel")),
                fnsm.array("D", (Ndim, Ndof, Ndof)),
                fnsm.array(f"u{i}", (Ndim, "Nel", Ndof)),
            ]
            for i in range(Nfield)
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
            noutputs=Nfield,
            n_e_per_wg=10,
            nwork_items_per_e=5,
            n_stmt_tile=4,
            i_tiles=15,
            j_tiles=2,
        )

        print(
            fnsm.stringify_comparison_vs_roofline(
                expr, transform=bound_transform, cl_ctx=cl_ctx
            )
        )

# vim: fdm=marker
