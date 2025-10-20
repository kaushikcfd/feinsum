import logging
import math
from typing import TYPE_CHECKING, Any, cast

import islpy as isl
import loopy as lp
from more_itertools import zip_equal as szip

import feinsum as fnsm
import feinsum.loopy_utils as lp_utils
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
    n_e_per_wg: int,
    nwork_items_per_e: int,
    n_stmt_tile: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import loopy.match as lp_match
    from loopy.symbolic import get_dependencies
    from more_itertools import chunked
    from pymbolic import variables

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.batched_einsum(
        "ifj,fe,fej->ei",
        [
            [
                fnsm.array("L", (nvoldof, nface, nfacedof)),
                fnsm.array("J", (nface, "Nel")),
                fnsm.array(f"v{i}", (nface, "Nel", nfacedof)),
            ]
            for i in range(nfields)
        ],
    )
    len_stmt_tile = math.ceil(nfields / n_stmt_tile)

    # {{{ get corresponding variables in t_unit

    vng = t_unit[kernel_name].get_var_name_generator()
    ing = t_unit[kernel_name].get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match, kernel_name=kernel_name
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
    i_s = [vng(i) for _ in range(n_stmt_tile)]
    f_s = [vng(f) for _ in range(n_stmt_tile)]
    j_s = [vng(j) for _ in range(n_stmt_tile)]
    e_s = [vng(e) for _ in range(n_stmt_tile)]
    e_outer_names = [vng("e_outer") for _ in range(n_stmt_tile)]
    e_inner_names = [vng("e_inner") for _ in range(n_stmt_tile)]
    i_outer_names = [vng(f"{i}_outer") for _ in range(n_stmt_tile)]
    i_inner_names = [vng(f"{i}_inner") for _ in range(n_stmt_tile)]
    L_fetch = vng(f"{L}_fetch")
    # prefetch_L_insns_ids = [vng(f"prftch_{L}") for _ in range(n_stmt_tile)]
    i_stmt_to_subst_prcmpt_tmp = [vng("prcmpt_stage1") for _ in range(len_stmt_tile)]
    i_stmt_tile_to_e_prcmpt_stage1 = [
        vng(f"{e}_prcmpt_stage1") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_f_prcmpt_stage1 = [
        vng(f"{f}_prcmpt_stage1") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_j_prcmpt_stage1 = [
        vng(f"{j}_prcmpt_stage1") for _ in range(n_stmt_tile)
    ]
    compute_fxj_id = {field: ing(f"compute_fxj_{field}") for field in fields}

    # }}}

    # {{{ splitting fields across outer_statement_tiles

    i_stmt_tile_to_fields = [
        list(el) for el in chunked(fields, math.ceil(nfields / n_stmt_tile))
    ]
    i_stmt_tile_to_outputs = [
        list(el) for el in chunked(outputs, math.ceil(nfields / n_stmt_tile))
    ]
    assert all(len(el) <= len_stmt_tile for el in i_stmt_tile_to_fields)
    assert all(
        len(el1) == len(el2)
        for el1, el2 in szip(i_stmt_tile_to_fields, i_stmt_tile_to_outputs)
    )

    # }}}

    knl = t_unit[kernel_name]

    for fields_in_tile, outputs_in_tile in szip(
        i_stmt_tile_to_fields, i_stmt_tile_to_outputs
    ):
        for field, output in szip(fields_in_tile, outputs_in_tile):
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

    for i_stmt_tile, fields_in_tile in enumerate(i_stmt_tile_to_fields):
        new_j = j_s[i_stmt_tile]
        new_f = f_s[i_stmt_tile]
        new_i = i_s[i_stmt_tile]
        new_e = e_s[i_stmt_tile]
        e_inner = e_inner_names[i_stmt_tile]
        e_outer = e_outer_names[i_stmt_tile]
        i_inner_name = i_inner_names[i_stmt_tile]
        i_outer_name = i_outer_names[i_stmt_tile]

        outputs_insn_match = lp_match.And(
            (
                within,
                lp_match.Or(
                    tuple(
                        lp_match.Writes(output)
                        for output in i_stmt_tile_to_outputs[i_stmt_tile]
                    )
                ),
            )
        )
        t_unit = lp.duplicate_inames(
            t_unit,
            [e, f, i, j],
            within=outputs_insn_match,
            new_inames=[new_e, new_f, new_i, new_j],
        )
        parent_inames = cast(
            "Collection[str]",
            t_unit[kernel_name]
            .get_inames_domain(e)
            .get_var_names(isl.dim_type.param),
        )
        assert all(isinstance(parent_iname, str) for parent_iname in parent_inames)
        t_unit = t_unit.with_kernel(
            lp_utils.decouple_domain(
                t_unit[kernel_name],
                [new_e, new_f, new_i, new_j],
                parent_inames=parent_inames,
            ),
        )

        t_unit = lp.split_iname(
            t_unit,
            new_e,
            n_e_per_wg,
            inner_iname=e_inner,
            outer_iname=e_outer,
            inner_tag="l.1",
            outer_tag="g.0",
        )

        for istmt, field in enumerate(fields_in_tile):
            subst_name = subst_names[field]
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                subst_name,
                sweep_inames=[e_inner, new_j, new_f],
                temporary_address_space=lp.AddressSpace.LOCAL,
                temporary_name=i_stmt_to_subst_prcmpt_tmp[istmt],
                precompute_outer_inames=frozenset({e_outer}),
                precompute_inames=[
                    i_stmt_tile_to_f_prcmpt_stage1[i_stmt_tile],
                    i_stmt_tile_to_e_prcmpt_stage1[i_stmt_tile],
                    i_stmt_tile_to_j_prcmpt_stage1[i_stmt_tile],
                ],
                compute_insn_id=compute_fxj_id[field],
                default_tag=None,
            )

        t_unit = lp.tag_inames(
            t_unit, {i_stmt_tile_to_e_prcmpt_stage1[i_stmt_tile]: "l.1"}
        )
        t_unit = lp.split_iname(
            t_unit,
            i_stmt_tile_to_j_prcmpt_stage1[i_stmt_tile],
            nwork_items_per_e,
            inner_tag="l.0",
            # TODO: uncommenting this leads to 20% slow down in POCL & 10%
            # speedup in Nv-CL.
            # outer_tag="unr"
        )

        t_unit = lp.split_iname(
            t_unit,
            new_i,
            nwork_items_per_e,
            inner_iname=i_inner_name,
            outer_iname=i_outer_name,
            inner_tag="l.0",
            outer_tag="unr",
        )
        t_unit = lp.prioritize_loops(t_unit, [new_j, new_f])
        # t_unit = lp.tag_inames(t_unit, {new_f: "unr"})

    for i_stmt_tile in range(1, n_stmt_tile):
        predecessors = lp_match.And(
            (
                within,
                lp_match.Or(
                    tuple(
                        lp_match.Writes(output)
                        for output in i_stmt_tile_to_outputs[i_stmt_tile - 1]
                    )
                ),
            )
        )
        successors = lp_match.Or(
            tuple(
                lp_match.Id(compute_fxj_id[field])
                for field in i_stmt_tile_to_fields[i_stmt_tile]
            )
        )
        t_unit = lp.add_dependency(t_unit, successors, predecessors)

    t_unit = lp.add_inames_to_insn(
        t_unit, inames=e_outer_names[0], insn_match=lp_match.Writes(L_fetch)
    )

    return t_unit


@fnsm.tuning.einsum_arg("nface", lambda e: e.args[0][2].shape[0])
@fnsm.tuning.einsum_arg("nvoldof", lambda e: e.shape[1])
@fnsm.tuning.einsum_arg("nfacedof", lambda e: e.args[0][2].shape[2])
@fnsm.tuning.einsum_arg("nfields", lambda e: e.b)
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(2, 32))
@fnsm.tuning.transform_param(
    "nwork_items_per_e", lambda e: IntParameter(1, e.args[0][2].shape[2])
)
@fnsm.tuning.transform_param(
    "n_stmt_tile", lambda e: IntParameter(1, math.ceil(e.b))
)
@fnsm.tuning.transform_param(
    "n_i_tile", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "n_j_tile", lambda e: IntParameter(1, math.ceil(e.args[0][2].shape[2] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    nface: int,
    nvoldof: int,
    nfacedof: int,
    nfields: int,
    n_e_per_wg: int,
    nwork_items_per_e: int,
    n_stmt_tile: int,
    n_i_tile: int,
    n_j_tile: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import loopy.match as lp_match
    from loopy.symbolic import get_dependencies
    from more_itertools import chunked
    from pymbolic import variables

    n_stmt_tile = math.ceil(nfields / math.ceil(nfields / n_stmt_tile))

    if n_j_tile == 1 and n_i_tile == 1:
        return transform_with_single_j_tile_i_tile(
            t_unit,
            nface,
            nvoldof,
            nfacedof,
            nfields,
            n_e_per_wg,
            nwork_items_per_e,
            n_stmt_tile,
            insn_match=insn_match,
            kernel_name=kernel_name,
        )
    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = lp_match.parse_match(insn_match)
    ref_einsum = fnsm.batched_einsum(
        "ifj,fe,fej->ei",
        [
            [
                fnsm.array("L", (nvoldof, nface, nfacedof)),
                fnsm.array("J", (nface, "Nel")),
                fnsm.array(f"v{i}", (nface, "Nel", nfacedof)),
            ]
            for i in range(nfields)
        ],
    )
    len_stmt_tile = math.ceil(nfields / n_stmt_tile)
    len_j_tile = math.ceil(nfacedof / n_j_tile)
    len_i_tile = math.ceil(nvoldof / n_i_tile)

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match, kernel_name=kernel_name
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
    e_s = [vng(e) for _ in range(n_stmt_tile)]
    i_s = [vng(i) for _ in range(n_stmt_tile)]
    j_s = [vng(j) for _ in range(n_stmt_tile)]
    f_s = [vng(f) for _ in range(n_stmt_tile)]
    j_tile_names = [vng(f"{j}_tile") for _ in range(n_stmt_tile)]
    j_inner_names = [vng(f"{j}_inner") for _ in range(n_stmt_tile)]
    i_tile_names = [vng(f"{i}_tile") for _ in range(n_stmt_tile)]
    i_inner_names = [vng(f"{i}_inner") for _ in range(n_stmt_tile)]
    i_inner_inner_names = [vng(f"{i}_inner_inner") for _ in range(n_stmt_tile)]
    i_inner_outer_names = [vng(f"{i}_inner_outer") for _ in range(n_stmt_tile)]
    L_fetch = vng(f"{L}_fetch")
    prefetch_L_insns_ids = [ing(f"prftch_{L}") for _ in range(n_stmt_tile)]
    e_outer_names = [vng(f"{e}_outer") for _ in range(n_stmt_tile)]
    e_inner_names = [vng(f"{e}_inner") for _ in range(n_stmt_tile)]
    i_stmt_to_subst_prcmp_tmp = [vng("prcmpt_stage1") for _ in range(len_stmt_tile)]
    i_stmt_tile_to_itile_init = [vng(f"{i}_tile_init") for _ in range(n_stmt_tile)]
    i_stmt_tile_to_i_inner_outer_assign = [
        vng(f"{i}_inner_outer_assign") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_itile_assign = [
        vng(f"{i}_tile_assign") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_i_inner_outer_init = [
        vng(f"{i}_inner_outer_init") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_e_prcmpt_stage1 = [
        vng(f"{e}_prcmpt_stage1") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_f_prcmpt_stage1 = [
        vng(f"{f}_prcmpt_stage1") for _ in range(n_stmt_tile)
    ]
    i_stmt_tile_to_j_prcmpt_stage1 = [
        vng(f"{j}_prcmpt_stage1") for _ in range(n_stmt_tile)
    ]

    # }}}

    # {{{ splitting fields across outer_statement_tiles

    i_stmt_tile_to_fields = [
        list(el) for el in chunked(fields, math.ceil(nfields / n_stmt_tile))
    ]
    i_stmt_tile_to_outputs = [
        list(el) for el in chunked(outputs, math.ceil(nfields / n_stmt_tile))
    ]
    assert all(len(el) <= len_stmt_tile for el in i_stmt_tile_to_fields)
    assert all(
        len(el1) == len(el2)
        for el1, el2 in szip(i_stmt_tile_to_fields, i_stmt_tile_to_outputs)
    )

    # }}}

    # {{{ split the kernel into disparate chunks

    for outputs_in_tile, new_i, new_f, new_e, new_j in szip(
        i_stmt_tile_to_outputs, i_s, f_s, e_s, j_s
    ):
        insn_match = lp_match.And(
            (
                within,
                lp_match.Or(
                    tuple(
                        lp_match.Writes(output_name)
                        for output_name in outputs_in_tile
                    )
                ),
            )
        )
        # FIXME: These should pull the duplicated inames into basic sets of their
        # own -> brings down the computational time.
        t_unit = lp.duplicate_inames(
            t_unit,
            (i, f, e, j),
            within=insn_match,
            new_inames=[new_i, new_f, new_e, new_j],
        )
        t_unit = t_unit.with_kernel(
            lp_utils.decouple_domain(
                t_unit[kernel_name],
                [new_e, new_i, new_j, new_f],
                parent_inames=cast(
                    "Collection[str]",
                    t_unit[kernel_name]
                    .get_inames_domain(e)
                    .get_var_names(isl.dim_type.param),
                ),
            )
        )

    t_unit = lp.remove_unused_inames(t_unit)

    # }}}

    for i_stmt_tile, (fields_in_tile, outputs_in_tile) in enumerate(
        szip(i_stmt_tile_to_fields, i_stmt_tile_to_outputs)
    ):
        new_i = i_s[i_stmt_tile]
        new_j = j_s[i_stmt_tile]
        new_f = f_s[i_stmt_tile]
        new_e = e_s[i_stmt_tile]
        j_tile_name = j_tile_names[i_stmt_tile]
        j_inner_name = j_inner_names[i_stmt_tile]
        i_tile_name = i_tile_names[i_stmt_tile]
        i_inner_name = i_inner_names[i_stmt_tile]
        i_inner_inner_name = i_inner_inner_names[i_stmt_tile]
        i_inner_outer_name = i_inner_outer_names[i_stmt_tile]
        f_prftchL, i_prftchL, j_prftchL = (
            vng(f"{f}prftch{L}"),
            vng(f"{i}prftch{L}"),
            vng(f"{j}prftch{L}"),
        )

        # There's a problem here. The accumulator names are sort of random
        # here which is obnoxious. We probably need to use some metadata here.

        knl = t_unit[kernel_name]

        for field, output in szip(fields_in_tile, outputs_in_tile):
            subst_name = subst_names[field]
            # FIXME: use precompute inames based on which inner statement tile
            # does the field belong to.

            insn_match = lp_match.And((within, lp_match.Writes(output)))

            knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
                knl,
                within=insn_match,
                subst_name=subst_name,
                arguments=variables(f"{new_e} {new_j} {new_f}"),
                terms_filter=lambda x: (get_dependencies(x) & knl.all_inames())
                <= {new_e, new_f, new_j},
            )

        t_unit = t_unit.with_kernel(knl)

        t_unit = lp.split_iname(
            t_unit,
            new_j,
            len_j_tile,
            outer_iname=j_tile_name,
            inner_iname=j_inner_name,
        )
        t_unit = lp.split_iname(
            t_unit,
            new_i,
            len_i_tile,
            outer_iname=i_tile_name,
            inner_iname=i_inner_name,
        )
        t_unit = lp.split_iname(
            t_unit,
            new_e,
            n_e_per_wg,
            inner_iname=e_inner_names[i_stmt_tile],
            outer_iname=e_outer_names[i_stmt_tile],
            outer_tag="g.0",
            inner_tag="l.1",
        )
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            L,
            sweep_inames=[j_inner_name, i_inner_name, new_f],
            precompute_outer_inames=frozenset(
                {j_tile_name, i_tile_name, e_outer_names[i_stmt_tile]}
            ),
            precompute_inames=[i_prftchL, f_prftchL, j_prftchL],
            temporary_name=L_fetch,
            temporary_address_space=lp.AddressSpace.LOCAL,
            default_tag=None,
            compute_insn_id=prefetch_L_insns_ids[i_stmt_tile],
            within=lp_match.Iname(i_inner_name),
        )
        t_unit = lp.split_iname(
            t_unit, i_prftchL, n_e_per_wg, inner_tag="l.1", outer_tag="unr"
        )
        t_unit = lp.split_iname(
            t_unit, j_prftchL, nwork_items_per_e, inner_tag="l.0", outer_tag="unr"
        )

        for istmt, field in enumerate(fields_in_tile):
            subst_name = subst_names[field]
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                subst_name,
                sweep_inames=[
                    e_inner_names[i_stmt_tile],
                    j_inner_names[i_stmt_tile],
                    new_f,
                ],
                temporary_address_space=lp.AddressSpace.LOCAL,
                temporary_name=i_stmt_to_subst_prcmp_tmp[istmt],
                precompute_outer_inames=frozenset(
                    {
                        e_outer_names[i_stmt_tile],
                        j_tile_names[i_stmt_tile],
                    }
                ),
                precompute_inames=[
                    i_stmt_tile_to_e_prcmpt_stage1[i_stmt_tile],
                    i_stmt_tile_to_j_prcmpt_stage1[i_stmt_tile],
                    i_stmt_tile_to_f_prcmpt_stage1[i_stmt_tile],
                ],
                default_tag=None,
            )
        t_unit = lp.tag_inames(
            t_unit, {i_stmt_tile_to_e_prcmpt_stage1[i_stmt_tile]: "l.1"}
        )
        t_unit = lp.split_iname(
            t_unit,
            i_stmt_tile_to_j_prcmpt_stage1[i_stmt_tile],
            nwork_items_per_e,
            inner_tag="l.0",
        )

        t_unit = lp.split_iname(
            t_unit,
            i_inner_name,
            nwork_items_per_e,
            inner_iname=i_inner_inner_name,
            outer_iname=i_inner_outer_name,
            inner_tag="l.0",
            outer_tag="unr",
        )

        outputs_insn_match = lp_match.And(
            (
                within,
                lp_match.Or(
                    tuple(
                        lp_match.Writes(output)
                        for output in i_stmt_tile_to_outputs[i_stmt_tile]
                    )
                ),
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
        acc_names = {
            vng(f"acc_{new_f}_{j_tile_name}_{j_inner_name}") for _ in fields_in_tile
        }
        assert (
            set(t_unit[kernel_name].temporary_variables) & acc_names
        ) == acc_names
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, set(inames_to_duplicate), only_var_names=acc_names
        )
        t_unit = lp.tag_inames(t_unit, {new_f: "unr"})

        new_iname_names_map = {
            i_tile_name: i_stmt_tile_to_itile_init[i_stmt_tile],
            i_inner_outer_name: i_stmt_tile_to_i_inner_outer_init[i_stmt_tile],
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
            i_tile_name: i_stmt_tile_to_itile_assign[i_stmt_tile],
            i_inner_outer_name: i_stmt_tile_to_i_inner_outer_assign[i_stmt_tile],
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

    for i_stmt_tile in range(1, n_stmt_tile):
        predecessor = lp_match.Iname(j_tile_names[i_stmt_tile - 1])
        successor = lp_match.Iname(j_tile_names[i_stmt_tile])
        t_unit = lp.add_dependency(t_unit, successor, predecessor)

    return t_unit


if __name__ == "__main__":
    import os
    from functools import partial

    import pyopencl as cl

    Nfields = 4
    Nface = 4
    Nfacedof = 15
    Nvoldof = 35

    cl_ctx = cl.create_some_context()
    expr = fnsm.batched_einsum(
        "ifj,fe,fej->ei",
        [
            [
                fnsm.array("L", (Nvoldof, Nface, Nfacedof)),
                fnsm.array("J", (Nface, "Nel")),
                fnsm.array(f"v{i}", (Nface, "Nel", Nfacedof)),
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
            n_e_per_wg=16,
            nwork_items_per_e=12,
            n_stmt_tile=2,
            n_i_tile=1,
            n_j_tile=1,
        )

        print(
            fnsm.stringify_comparison_vs_roofline(
                expr, transform=bound_transform, cl_ctx=cl_ctx
            )
        )

# vim: fdm=marker
