import logging
import math
from typing import TYPE_CHECKING, Any, cast

import islpy as isl
import loopy as lp

import feinsum as fnsm
import feinsum.loopy_utils as lp_utils
from feinsum.loopy_utils.cse import hoist_cses
from feinsum.tuning import IntParameter

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


def transform_with_single_j_tile_i_tile(
    t_unit: lp.TranslationUnit,
    nface: int,
    nvoldof: int,
    nfacedof: int,
    nfields: int,
    n_e_per_wg_log2: int,
    nwork_items_per_e: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import loopy.match as lp_match
    from loopy.symbolic import get_dependencies
    from pymbolic import variables
    n_e_per_wg = 2 ** n_e_per_wg_log2

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = lp_match.parse_match(insn_match)
    within = lp_match.Or(
        tuple(
            lp_match.Id(insn.id)
            for insn in t_unit[kernel_name].instructions
            if within(t_unit[kernel_name], insn)
        )
    )

    ref_einsum = fnsm.batched_einsum(
        "jfi,fe,fej->ei",
        [
            [
                fnsm.array("L", (nfacedof, nface, nvoldof)),
                fnsm.array("J", (nface, "Nel")),
                fnsm.array(f"v{i}", (nface, "Nel", nfacedof)),
            ]
            for i in range(nfields)
        ],
    )

    # {{{ get corresponding variables in t_unit

    vng = t_unit[kernel_name].get_var_name_generator()
    ing = t_unit[kernel_name].get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit,
        ref_einsum,
        insn_match=insn_match,
        kernel_name=kernel_name,
        long_dim_length=36,
    )
    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    f = subst_map["f"]
    L = subst_map["L"]
    e_outer, e_inner = vng(f"{e}_outer"), vng(f"{e}_inner")
    fields = [subst_map[f"v{i}"] for i in range(nfields)]
    outputs = [subst_map["_fe_out"]] + [
        subst_map[f"_fe_out_{i}"] for i in range(nfields - 1)
    ]
    subst_names = {field: vng("subst_hoist") for field in fields}
    i_outer_name = vng(f"{i}_outer")
    i_inner_name = vng(f"{i}_inner")
    L_fetch = vng(f"{L}_fetch")
    # L_insns_ids = vng(f"prftch_{L}")
    i_stmt_to_subst_prcmpt_tmp = [vng("prcmpt_stage1") for _ in range(nfields)]
    e_prcmpt_stage1 = vng(f"{e}_prcmpt_stage1")
    f_prcmpt_stage1 = vng(f"{f}_prcmpt_stage1")
    j_prcmpt_stage1 = vng(f"{j}_prcmpt_stage1")
    compute_fxj_id = {field: ing(f"compute_fxj_{field}") for field in fields}

    # }}}

    knl = t_unit[kernel_name]

    for field, output in zip(fields, outputs, strict=True):
        subst_name = subst_names[field]
        insn_match = lp_match.And((within, lp_match.Writes(output)))
        knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
            knl,
            within=insn_match,
            subst_name=subst_name,
            arguments=variables(f"{f} {e} {j}"),
            terms_filter=lambda x: (get_dependencies(x) & knl.all_inames())
            <= {f, e, j},
        )
    t_unit = t_unit.with_kernel(knl)

    f_prftchL, i_prftchL, j_prftchL = (
        vng(f"{f}prftch{L}"),
        vng(f"{i}prftch{L}"),
        vng(f"{j}prftch{L}"),
    )

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        L,
        sweep_inames=[f, i, j],
        precompute_outer_inames=frozenset(),
        precompute_inames=[i_prftchL, f_prftchL, j_prftchL],
        default_tag=None,
        within=within,
        temporary_name=L_fetch,
    )

    t_unit = lp.split_iname(
        t_unit, i_prftchL, n_e_per_wg, inner_tag="l.1", outer_tag="unr"
    )
    t_unit = lp.split_iname(
        t_unit, j_prftchL, nwork_items_per_e, inner_tag="l.0", outer_tag="unr"
    )

    parent_inames = cast(
        "Collection[str]",
        t_unit[kernel_name].get_inames_domain(e).get_var_names(isl.dim_type.param),
    )
    assert all(isinstance(parent_iname, str) for parent_iname in parent_inames)

    t_unit = lp.split_iname(
        t_unit,
        e,
        n_e_per_wg,
        inner_iname=e_inner,
        outer_iname=e_outer,
        inner_tag="l.1",
        outer_tag="g.0",
    )

    for istmt, field in enumerate(fields):
        subst_name = subst_names[field]
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            subst_name,
            sweep_inames=[e_inner, j, f],
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_name=i_stmt_to_subst_prcmpt_tmp[istmt],
            precompute_outer_inames=frozenset({e_outer}),
            precompute_inames=[
                f_prcmpt_stage1,
                e_prcmpt_stage1,
                j_prcmpt_stage1,
            ],
            compute_insn_id=compute_fxj_id[field],
            default_tag=None,
        )

    # {{{ CSE

    t_unit = lp.expand_subst(
        t_unit,
        within=lp_match.And(
            (
                lp_match.Iname(e_prcmpt_stage1),
                lp_match.Iname(f_prcmpt_stage1),
                lp_match.Iname(j_prcmpt_stage1),
            )
        ),
    )
    t_unit = hoist_cses(
        t_unit,
        within=lp_match.And(
            (
                lp_match.Iname(e_prcmpt_stage1),
                lp_match.Iname(f_prcmpt_stage1),
                lp_match.Iname(j_prcmpt_stage1),
            )
        ),
    )

    # }}}

    t_unit = lp.tag_inames(t_unit, {e_prcmpt_stage1: "l.1"})
    t_unit = lp.split_iname(
        t_unit,
        j_prcmpt_stage1,
        nwork_items_per_e,
        inner_tag="l.0",
        outer_tag="unr",
    )

    t_unit = lp.split_iname(
        t_unit,
        i,
        nwork_items_per_e,
        inner_iname=i_inner_name,
        outer_iname=i_outer_name,
        inner_tag="l.0",
        outer_tag="unr",
    )
    t_unit = lp.prioritize_loops(t_unit, [j, f])
    t_unit = lp.tag_inames(t_unit, {f: "unr"})

    t_unit = lp.add_inames_to_insn(
        t_unit, inames=e_outer, insn_match=lp_match.Writes(L_fetch)
    )
    # lp.generate_code_v2(t_unit)
    return t_unit


@fnsm.tuning.einsum_arg("nface", lambda e: e.args[0][1].shape[0])
@fnsm.tuning.einsum_arg("nvoldof", lambda e: e.shape[1])
@fnsm.tuning.einsum_arg("nfacedof", lambda e: e.args[0][2].shape[0])
@fnsm.tuning.einsum_arg("nfields", lambda e: e.b)
@fnsm.tuning.transform_param("n_e_per_wg_log2", lambda e: IntParameter(1, 5))
@fnsm.tuning.transform_param(
    "nwork_items_per_e", lambda e: IntParameter(1, e.args[0][2].shape[0])
)
@fnsm.tuning.transform_param(
    "n_i_tile", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "n_j_tile", lambda e: IntParameter(1, math.ceil(e.args[0][2].shape[0] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    nface: int,
    nvoldof: int,
    nfacedof: int,
    nfields: int,
    n_e_per_wg_log2: int,
    nwork_items_per_e: int,
    n_i_tile: int,
    n_j_tile: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import loopy.match as lp_match
    from loopy.symbolic import get_dependencies
    from pymbolic import variables

    if n_j_tile == 1 and n_i_tile == 1:
        return transform_with_single_j_tile_i_tile(
            t_unit,
            nface,
            nvoldof,
            nfacedof,
            nfields,
            n_e_per_wg_log2,
            nwork_items_per_e,
            insn_match=insn_match,
            kernel_name=kernel_name,
        )

    n_e_per_wg = 2 ** n_e_per_wg_log2
    kernel_name = kernel_name or t_unit.default_entrypoint.name

    if n_e_per_wg * nwork_items_per_e > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    within = lp_match.parse_match(insn_match)
    within = lp_match.Or(
        tuple(
            lp_match.Id(insn.id)
            for insn in t_unit[kernel_name].instructions
            if within(t_unit[kernel_name], insn)
        )
    )

    ref_einsum = fnsm.batched_einsum(
        "jfi,fe,fej->ei",
        [
            [
                fnsm.array("L", (nfacedof, nface, nvoldof)),
                fnsm.array("J", (nface, "Nel")),
                fnsm.array(f"v{i}", (nface, "Nel", nfacedof)),
            ]
            for i in range(nfields)
        ],
    )
    len_j_tile = math.ceil(nfacedof / n_j_tile)
    len_i_tile = math.ceil(nvoldof / n_i_tile)

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit,
        ref_einsum,
        insn_match=insn_match,
        kernel_name=kernel_name,
        long_dim_length=36,
    )
    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    f = subst_map["f"]
    L = subst_map["L"]
    fields = [subst_map[f"v{i}"] for i in range(nfields)]
    outputs = [subst_map["_fe_out"]] + [
        subst_map[f"_fe_out_{i}"] for i in range(nfields - 1)
    ]
    subst_names = {field: vng("subst_hoist") for field in fields}
    j_tile_name = vng(f"{j}_tile")
    j_inner_name = vng(f"{j}_inner")
    i_tile_name = vng(f"{i}_tile")
    i_inner_name = vng(f"{i}_inner")
    i_inner_inner_name = vng(f"{i}_inner_inner")
    i_inner_outer_name = vng(f"{i}_inner_outer")
    L_fetch = vng(f"{L}_fetch")
    prefetch_L_insns_id = ing(f"prftch_{L}")
    e_outer_name = vng(f"{e}_outer")
    e_inner_name = vng(f"{e}_inner")
    i_stmt_to_subst_prcmp_tmp = [vng("prcmpt_stage1") for _ in range(nfields)]
    itile_init = vng(f"{i}_tile_init")
    i_inner_outer_assign = vng(f"{i}_inner_outer_assign")
    itile_assign = vng(f"{i}_tile_assign")
    i_inner_outer_init = vng(f"{i}_inner_outer_init")
    e_prcmpt_stage1 = vng(f"{e}_prcmpt_stage1")
    f_prcmpt_stage1 = vng(f"{f}_prcmpt_stage1")
    j_prcmpt_stage1 = vng(f"{j}_prcmpt_stage1")

    # }}}

    f_prftchL, i_prftchL, j_prftchL = (
        vng(f"{f}prftch{L}"),
        vng(f"{i}prftch{L}"),
        vng(f"{j}prftch{L}"),
    )

    # There's a problem here. The accumulator names are sort of random
    # here which is obnoxious. We probably need to use some metadata here.

    knl = t_unit[kernel_name]

    for field, output in zip(fields, outputs, strict=True):
        subst_name = subst_names[field]
        # FIXME: use precompute inames based on which inner statement tile
        # does the field belong to.

        insn_match = lp_match.And((within, lp_match.Writes(output)))

        knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
            knl,
            within=insn_match,
            subst_name=subst_name,
            arguments=variables(f"{e} {j} {f}"),
            terms_filter=lambda x: (get_dependencies(x) & knl.all_inames())
            <= {e, f, j},
        )

    t_unit = t_unit.with_kernel(knl)

    t_unit = lp.split_iname(
        t_unit,
        j,
        len_j_tile,
        outer_iname=j_tile_name,
        inner_iname=j_inner_name,
    )
    t_unit = lp.split_iname(
        t_unit,
        i,
        len_i_tile,
        outer_iname=i_tile_name,
        inner_iname=i_inner_name,
    )
    t_unit = lp.split_iname(
        t_unit,
        e,
        n_e_per_wg,
        inner_iname=e_inner_name,
        outer_iname=e_outer_name,
        outer_tag="g.0",
        inner_tag="l.1",
    )
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        L,
        sweep_inames=[j_inner_name, f, i_inner_name],
        precompute_outer_inames=frozenset({j_tile_name, i_tile_name, e_outer_name}),
        precompute_inames=[j_prftchL, f_prftchL, i_prftchL],
        temporary_name=L_fetch,
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag=None,
        compute_insn_id=prefetch_L_insns_id,
        within=lp_match.Iname(i_inner_name),
    )
    if len_i_tile > nwork_items_per_e:
        t_unit = lp.split_iname(
            t_unit, i_prftchL, nwork_items_per_e, inner_tag="l.0", outer_tag="unr"
        )
    else:
        t_unit = lp.tag_inames(t_unit, {i_prftchL: "l.0"}, ignore_nonexistent=True)

    if len_j_tile > n_e_per_wg:
        t_unit = lp.split_iname(
            t_unit,
            j_prftchL,
            nwork_items_per_e,
            inner_tag="l.1",
            outer_iname="unr",
        )
    else:
        t_unit = lp.tag_inames(t_unit, {j_prftchL: "l.1"}, ignore_nonexistent=True)

    for istmt, field in enumerate(fields):
        subst_name = subst_names[field]
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            subst_name,
            sweep_inames=[
                e_inner_name,
                j_inner_name,
                f,
            ],
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_name=i_stmt_to_subst_prcmp_tmp[istmt],
            precompute_outer_inames=frozenset(
                {
                    e_outer_name,
                    j_tile_name,
                }
            ),
            precompute_inames=[
                e_prcmpt_stage1,
                j_prcmpt_stage1,
                f_prcmpt_stage1,
            ],
            default_tag=None,
        )

    # {{{ CSE

    t_unit = lp.expand_subst(
        t_unit,
        within=lp_match.Iname(e_prcmpt_stage1),
    )
    t_unit = hoist_cses(
        t_unit,
        within=lp_match.Iname(e_prcmpt_stage1),
    )

    # }}}

    t_unit = lp.tag_inames(t_unit, {e_prcmpt_stage1: "l.1"})
    if len_j_tile > nwork_items_per_e:
        t_unit = lp.split_iname(
            t_unit,
            j_prcmpt_stage1,
            nwork_items_per_e,
            inner_tag="l.0",
        )
    else:
        t_unit = lp.tag_inames(
            t_unit, {j_prcmpt_stage1: "l.0"}, ignore_nonexistent=True
        )

    if len_i_tile > nwork_items_per_e:
        t_unit = lp.split_iname(
            t_unit,
            i_inner_name,
            nwork_items_per_e,
            inner_iname=i_inner_inner_name,
            outer_iname=i_inner_outer_name,
            inner_tag="l.0",
            outer_tag="unr",
        )
    else:
        t_unit = lp.tag_inames(t_unit, {i_inner_name: "l.0"})
        i_inner_inner_name = i_inner_name

    outputs_insn_match = lp_match.And(
        (
            within,
            lp_match.Or(tuple(lp_match.Writes(output) for output in outputs)),
        )
    )
    insn_ids = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if outputs_insn_match(t_unit[kernel_name], insn)
    ]

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_ids)
    inames_to_duplicate = sorted(
        frozenset({i_tile_name, i_inner_outer_name})
        & t_unit[kernel_name].all_inames()
    )
    acc_names = {vng(f"acc_{f}_{j_tile_name}_{j_inner_name}") for _ in fields}
    assert (set(t_unit[kernel_name].temporary_variables) & acc_names) == acc_names
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, set(inames_to_duplicate), only_var_names=acc_names
    )
    t_unit = lp.tag_inames(t_unit, {f: "unr"})

    new_iname_names_map = {
        i_tile_name: itile_init,
        i_inner_outer_name: i_inner_outer_init,
    }
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
        new_inames=[new_iname_names_map[iname] for iname in inames_to_duplicate],
        tags=dict.fromkeys(inames_to_duplicate, "unr"),
    )

    new_iname_names_map = {
        i_tile_name: itile_assign,
        i_inner_outer_name: i_inner_outer_assign,
    }
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
        new_inames=[new_iname_names_map[iname] for iname in inames_to_duplicate],
        tags=dict.fromkeys(inames_to_duplicate, "unr"),
    )
    t_unit = lp.prioritize_loops(t_unit, (j_inner_name, f))

    # lp.generate_code_v2(t_unit)

    return t_unit


if __name__ == "__main__":
    # t_unit = lp.make_kernel(
    #     "{[i,f,j,e]: 0<=f<4 and 0<=i<4 and 0<=j<3 and 0<=e<10000}",
    #     """
    #     _LIFT(_0, _1, _2) := M[_0, _1, _2]
    #     _sgeo(_0, _1) := J[_0, _1]
    #     _flux_0(_0, _1, _2) := u_0[_0, _1, _2]
    #     _flux_1(_0, _1, _2) := u_1[_0, _1, _2]
    #     _flux_2(_0, _1, _2) := u_2[_0, _1, _2]
    #     _flux_3(_0, _1, _2) := u_3[_0, _1, _2]
    #     out_0[e, i] = tmp_0[e, i] + sum([f, j], _LIFT(i, f, j)
    #                                            * _sgeo(f, e)
    #                                            * _flux_0(f, e, j))
    #     out_1[e, i] = tmp_1[e, i] + sum([f, j], _LIFT(i, f, j)
    #                                             * _sgeo(f, e)
    #                                             * _flux_1(f, e, j))
    #     out_2[e, i] = tmp_2[e, i] + sum([f, j], _LIFT(i, f, j)
    #                                             * _sgeo(f, e)
    #                                             * _flux_2(f, e, j))
    #     out_3[e, i] = tmp_3[e, i] + sum([f, j], _LIFT(i, f, j)
    #                                             * _sgeo(f, e)
    #                                             * _flux_3(f, e, j))
    #     """,
    #     [
    #         lp.GlobalArg(
    #             "M,J,tmp_0,tmp_1,tmp_2,tmp_3,out_0,out_1,out_2,out_3,"
    #             "u_0,u_1,u_2,u_3",
    #             dtype="float64",
    #             shape=lp.auto,
    #         ),
    #         ...,
    #     ],
    #     lang_version=(2018, 2),
    # )
    # t_unit = transform(
    #     t_unit,
    #     nface=4,
    #     nvoldof=4,
    #     nfacedof=3,
    #     nfields=4,
    #     n_e_per_wg_log2=1,
    #     n_i_tile=1,
    #     n_j_tile=1,
    #     nwork_items_per_e=1,
    # )
    # print(lp.generate_code_v2(t_unit).device_code())
    # 1/0

    import os

    import pyopencl as cl

    Nface = 4
    Nfields = 4

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    for Nfacedof, Nvoldof in [(10, 20)]:
        expr = fnsm.batched_einsum(
            "jfi,fe,fej->ei",
            [
                [
                    fnsm.array("L", (Nfacedof, Nface, Nvoldof)),
                    fnsm.array("J", (Nface, "Nel")),
                    fnsm.array(f"v{i}", (Nface, "Nel", Nfacedof)),
                ]
                for i in range(Nfields)
            ],
        )

        fnsm.autotune(
            expr,
            os.path.abspath(__file__),
            cq,
            test_limit=150,
        )

        best_config = min(
            fnsm.query(expr, cq.device, err_if_no_results=True),
            key=lambda query_info: query_info.runtime_in_sec,
        )
        print(75 * "-")
        print(best_config.transform_id)
        print(75 * "-")

        from feinsum.measure import _stringify_runtime_comparison_vs_roofline

        with open("log.txt", "a") as fp:
            fp.write(f"{Nfacedof = }, {Nvoldof = }.\n")
            fp.write(
                _stringify_runtime_comparison_vs_roofline(
                    expr, best_config.runtime_in_sec, cq.device.name
                )
            )
            fp.write("\n")

    # Enable while debugging ->
    # evaluate a point in the parameter space.
    # from functools import partial

    # bound_transform = partial(
    #     transform,
    #     n_e_per_wg=16,
    #     nwork_items_per_e=12,
    #     n_stmt_tile=2,
    #     n_i_tile=1,
    #     n_j_tile=1,
    # )

    # print(
    #     fnsm.stringify_comparison_vs_roofline(
    #         expr, transform=bound_transform, cq=cq
    #     )
    # )

# vim: fdm=marker
