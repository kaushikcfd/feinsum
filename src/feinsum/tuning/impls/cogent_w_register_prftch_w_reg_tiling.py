import itertools
import logging
import math
from typing import TYPE_CHECKING, Any, cast

import loopy as lp
import numpy as np

if TYPE_CHECKING:
    import pymbolic.primitives as prim
from pytools import memoize_on_first_arg

import feinsum as fnsm
from feinsum.einsum import INT_CLASSES, SizeParam
from feinsum.tuning import IntParameter, einsum_arg, transform_param
from feinsum.utils import get_n_redn_dim

logger = logging.getLogger(__name__)

# Values for Volta micro-arch. Generalize it via a class
MAX_SHARED_MEM_PER_WG = 48e3  # in bytes
REG_FILE_SPACE_PER_WI = 256 * 4  # in bytes


def _is_ensm_tensor_contraction(ensm: fnsm.BatchedEinsum) -> bool:
    # TC requires noperands == 2
    if ensm.n != 2 or ensm.b != 1:
        return False

    in_idx_set1, in_idx_set2 = ensm.in_idx_sets

    # TC requires each out index be indexed in exactly one operand
    for out_idx in ensm.out_idx_set:
        if not ((out_idx in in_idx_set1) ^ (out_idx in in_idx_set2)):
            return False
    if (len(set(in_idx_set1)) != len(in_idx_set1)) or (
        len(set(in_idx_set2)) != len(in_idx_set2)
    ):
        return False

    # TC requires all reduction indices be present in both the operands
    for redn_idx in ensm.sum_indices:
        if redn_idx not in in_idx_set1 or redn_idx not in in_idx_set2:
            return False

    return True


def _get_operand_names(ensm: fnsm.BatchedEinsum) -> tuple[str, str]:
    assert ensm.b == 1 and ensm.n == 2
    ((arg1, arg2),) = ensm.args
    return (arg1.name, arg2.name)


@einsum_arg("ensm", lambda ensm: ensm)
@transform_param(
    "i_thread_axis_mapping_perm",
    lambda ensm: IntParameter(0, math.perm(ensm.ndim, 2) - 1),
)
@transform_param(
    "i_reg_axis_mapping_perm",
    lambda ensm: IntParameter(0, math.perm(ensm.ndim, 2) - 1),
)
@transform_param(
    "log2_tx",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_ty",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_rx",
    lambda ensm: IntParameter(0, 5),
)
@transform_param(
    "log2_ry",
    lambda ensm: IntParameter(0, 5),
)
@transform_param(
    "log2_t_redns",
    lambda ensm: tuple(IntParameter(0, 5) for i in range(get_n_redn_dim(ensm))),
)
@transform_param("unroll_rx_ry", lambda ensm: IntParameter(0, 1))
@transform_param(
    "iredn_idx_to_prftch", lambda ensm: IntParameter(0, len(ensm.sum_indices) - 1)
)
@memoize_on_first_arg
def transform(
    t_unit: lp.TranslationUnit,
    ensm: fnsm.BatchedEinsum,
    i_thread_axis_mapping_perm: int,
    i_reg_axis_mapping_perm: int,
    log2_tx: int,
    log2_ty: int,
    log2_rx: int,
    log2_ry: int,
    log2_t_redns: tuple[int, ...],
    unroll_rx_ry: int,
    iredn_idx_to_prftch: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    if ensm.ndim < 2:
        raise ValueError(
            "This algorithm needs al least two dimensions" " in the output array."
        )
    if len(ensm.all_size_params) != 0:
        raise NotImplementedError("Parametric lengths are not allowed.")

    tx = 2**log2_tx
    ty = 2**log2_ty
    rx = 2**log2_rx
    ry = 2**log2_ry
    t_redns = tuple(2**log2_t_redn for log2_t_redn in log2_t_redns)

    i_tx: int
    i_ty: int
    i_rx: int
    i_ry: int
    for i, index_mapping in enumerate(
        itertools.permutations(list(range(ensm.ndim)), 2)
    ):
        if i == i_thread_axis_mapping_perm:
            i_tx, i_ty = index_mapping
        if i == i_reg_axis_mapping_perm:
            i_rx, i_ry = index_mapping
        if i >= i_thread_axis_mapping_perm and i >= i_reg_axis_mapping_perm:
            break
    else:
        raise AssertionError(
            f"{i_thread_axis_mapping_perm} or {i_reg_axis_mapping_perm} is an "
            " invalid permutation index."
        )

    import loopy.match as lp_match

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    knl = t_unit[kernel_name]
    within = lp_match.parse_match(insn_match)
    (insn_id,) = [insn.id for insn in knl.instructions if within(knl, insn)]
    del knl

    if not _is_ensm_tensor_contraction(ensm):
        raise ValueError(f"{ensm} is not a tensor contraction.")

    ensm_free_indices = ensm.out_idx_set
    ensm_redn_indices = ensm.sum_indices
    ensm_A, ensm_B = _get_operand_names(ensm)

    # {{{ sanity checks on param space

    # Verify legality of tile lengths
    # -------------------------------
    i_tx_dim_length = ensm.index_to_dim_length[ensm.out_idx_set[i_tx]]
    if isinstance(i_tx_dim_length, INT_CLASSES) and tx > i_tx_dim_length:
        raise fnsm.InvalidParameterError("Tx > Nx")

    i_ty_dim_length = ensm.index_to_dim_length[ensm.out_idx_set[i_ty]]
    if isinstance(i_ty_dim_length, INT_CLASSES) and ty > i_ty_dim_length:
        raise fnsm.InvalidParameterError("Ty > Ny")

    i_rx_dim_length = ensm.index_to_dim_length[ensm.out_idx_set[i_rx]]
    assert rx is not None
    if isinstance(i_rx_dim_length, INT_CLASSES) and rx > i_rx_dim_length:
        raise fnsm.InvalidParameterError("Rx > Nx")

    i_ry_dim_length = ensm.index_to_dim_length[ensm.out_idx_set[i_ry]]
    assert ry is not None
    if isinstance(i_ry_dim_length, INT_CLASSES) and ry > i_ry_dim_length:
        raise fnsm.InvalidParameterError("Ry > Ny")

    for redn_idx, redn_tile in zip(ensm_redn_indices, t_redns, strict=True):
        redn_idx_dim_length = ensm.index_to_dim_length[redn_idx]
        if (
            isinstance(redn_idx_dim_length, INT_CLASSES)
            and redn_tile > redn_idx_dim_length
        ):
            raise fnsm.InvalidParameterError(
                f"redn_dim {redn_idx}'s tile is greater the axis length."
            )

    # Verify Tx, Ty are distinct dimensions
    # ---------------------------------------------
    if i_tx == i_ty:
        raise AssertionError("i_tx, i_ty are not distinct.")

    # Verify Rx, Ry are distinct dimensions
    # ---------------------------------------------
    if i_rx == i_ry:
        raise AssertionError("i_rx, i_ry are not distinct.")

    # Verify shared memory
    # --------------------
    from pytools import product

    l_sm_a, l_sm_b = [
        product(
            cast("int", tile_length)
            for tile_length, i_tile in [
                (tx, i_tx),
                (ty, i_ty),
                (rx, i_rx),
                (ry, i_ry),
            ]
            if ensm.out_idx_set[i_tile] in idx_set
        )
        * product(t_redns)
        for idx_set in ensm.in_idx_sets
    ]
    if (
        l_sm_a * ensm.arg_to_dtype[ensm_A].itemsize
        + l_sm_b * ensm.arg_to_dtype[ensm_B].itemsize
    ) > MAX_SHARED_MEM_PER_WG:
        raise fnsm.InvalidParameterError("Exceeds local memory limits")

    # Verify register file usage
    # --------------------------
    if (
        rx
        * ry
        * np.result_type(
            ensm.arg_to_dtype[ensm_A], ensm.arg_to_dtype[ensm_B]
        ).itemsize
        > REG_FILE_SPACE_PER_WI
    ):
        raise fnsm.InvalidParameterError("Exceeds register file limits")

    assert tx <= 32 and ty <= 32

    new_t_redns = []
    for redn_idx, t_redn in zip(ensm_redn_indices, t_redns, strict=True):
        axis_len = ensm.index_to_dim_length[redn_idx]
        if isinstance(axis_len, SizeParam):
            new_t_redns.append(t_redn)
        else:
            assert isinstance(axis_len, INT_CLASSES)
            n_tiles = math.ceil(axis_len / t_redn)
            new_t_redns.append(math.ceil(axis_len / n_tiles))

    assert len(new_t_redns) == len(t_redns)
    assert all(
        new_t_redn <= old_t_redn
        for new_t_redn, old_t_redn in zip(new_t_redns, t_redns, strict=False)
    )
    t_redns = tuple(new_t_redns)
    del new_t_redns

    # }}}

    # type-ignore-reason: mypy is correct, but we have asserted earlier that we
    # won't accept parametric dim lengths
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit,
        ensm,
        insn_match=insn_match,
        kernel_name=kernel_name,
        long_dim_length=max(  # type: ignore[type-var,operator]
            ensm.index_to_dim_length.values(), default=1
        )
        + 1,
    )
    vng = t_unit[kernel_name].get_var_name_generator()
    ing = t_unit[kernel_name].get_instruction_id_generator()

    A = subst_map[ensm_A]
    B = subst_map[ensm_B]
    # C = subst_map["_fe_out"]
    free_indices = tuple(subst_map[idx] for idx in ensm_free_indices)
    redn_indices = tuple(subst_map[idx] for idx in ensm_redn_indices)
    tx_inner, tx_outer = vng("tx_inner"), vng("tx_outer")
    ty_inner, ty_outer = vng("ty_inner"), vng("ty_outer")
    rx_inner, rx_outer = vng("rx_inner"), vng("rx_outer")
    ry_inner, ry_outer = vng("ry_inner"), vng("ry_outer")
    inner_redn_inames = tuple(
        vng(f"{redn_iname}_inner") for redn_iname in redn_indices
    )
    outer_redn_inames = tuple(
        vng(f"{redn_iname}_outer") for redn_iname in redn_indices
    )

    izblock = vng("izblock")
    a_prftch_insn_id = ing("a_prftch_insn")
    b_prftch_insn_id = ing("b_prftch_insn")
    a_reg_prftch_insn_id = ing("a_reg_prftch_insn")
    b_reg_prftch_insn_id = ing("b_reg_prftch_insn")
    a_prftch_tmp = vng("a_prftch_tmp")
    b_prftch_tmp = vng("b_prftch_tmp")

    # {{{ split free indices

    if i_tx == i_rx or i_tx == i_ry:
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_tx],
            tx * (rx if i_tx == i_rx else ry),
            inner_iname=rx_outer,
            outer_iname=tx_outer,
            outer_tag="g.0",
            slabs=(0, 1),
        )
        t_unit = lp.split_iname(
            t_unit,
            rx_outer,
            tx,
            inner_iname=tx_inner,
            outer_iname=rx_inner if i_tx == i_rx else ry_inner,
            inner_tag="l.0",
            outer_tag="unr" if unroll_rx_ry else None,
        )
    else:
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_tx],
            tx,
            inner_iname=tx_inner,
            outer_iname=tx_outer,
            inner_tag="l.0",
            outer_tag="g.0",
            slabs=(0, 1),
        )

    if i_ty == i_rx or i_ty == i_ry:
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_ty],
            ty * (rx if i_ty == i_rx else ry),
            inner_iname=rx_outer,
            outer_iname=ty_outer,
            outer_tag="g.1",
            slabs=(0, 1),
        )
        t_unit = lp.split_iname(
            t_unit,
            rx_outer,
            ty,
            inner_iname=ty_inner,
            outer_iname=rx_inner if i_ty == i_rx else ry_inner,
            inner_tag="l.1",
            outer_tag="unr" if unroll_rx_ry else None,
        )
    else:
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_ty],
            ty,
            inner_iname=ty_inner,
            outer_iname=ty_outer,
            inner_tag="l.1",
            outer_tag="g.1",
            slabs=(0, 1),
        )
    if i_rx != i_tx and i_rx != i_ty:
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_rx],
            rx,
            inner_iname=rx_inner,
            outer_iname=rx_outer,
            inner_tag="unr" if unroll_rx_ry else None,
        )
    if i_ry != i_tx and i_ry != i_ty:
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_ry],
            ry,
            inner_iname=ry_inner,
            outer_iname=ry_outer,
            inner_tag="unr" if unroll_rx_ry else None,
        )

    # }}}

    # {{{ split redn indices

    for redn_iname, t_redn, inner_iname, outer_iname in zip(
        redn_indices, t_redns, inner_redn_inames, outer_redn_inames, strict=True
    ):
        if t_redn != 1:
            t_unit = lp.split_iname(
                t_unit,
                redn_iname,
                t_redn,
                inner_iname=inner_iname,
                outer_iname=outer_iname,
            )

    # }}}

    precompute_outer_inames = frozenset(
        {
            redn_outer_iname if t_redn != 1 else redn_iname
            for redn_outer_iname, redn_iname, t_redn in zip(
                outer_redn_inames, redn_indices, t_redns, strict=True
            )
        }
    )

    A_sweep_inames: list[str] = []
    B_sweep_inames: list[str] = []

    for sweep_inames, in_idx_set in zip(
        (A_sweep_inames, B_sweep_inames),
        ensm.in_idx_sets,
        strict=True,
    ):
        for in_idx in in_idx_set:
            if in_idx in ensm.out_idx_set:
                if ensm.out_idx_set.index(in_idx) == i_tx:
                    sweep_inames.append(tx_inner)
                if ensm.out_idx_set.index(in_idx) == i_ty:
                    sweep_inames.append(ty_inner)
                if ensm.out_idx_set.index(in_idx) == i_rx:
                    sweep_inames.append(rx_inner)
                if ensm.out_idx_set.index(in_idx) == i_ry:
                    sweep_inames.append(ry_inner)
            else:
                assert in_idx in ensm_redn_indices
                if t_redns[ensm_redn_indices.index(in_idx)] == 1:
                    # Not sweep this iname as it is purely a precompute outer iname.
                    pass
                else:
                    sweep_inames.append(
                        inner_redn_inames[ensm_redn_indices.index(in_idx)]
                    )

    # precompute A
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        A,
        sweep_inames=A_sweep_inames,
        precompute_outer_inames=precompute_outer_inames,
        temporary_address_space=lp.AddressSpace.LOCAL,
        within=within,
        compute_insn_id=a_prftch_insn_id,
        default_tag=None,
        temporary_name=a_prftch_tmp,
    )
    A_prftch_inames = tuple(
        cast("prim.Variable", idx).name
        for idx in cast(
            "prim.Subscript",
            cast(
                "lp.Assignment", t_unit[kernel_name].id_to_insn[a_prftch_insn_id]
            ).assignee,
        ).index_tuple
    )

    logger.info("Precomputed A")

    # prefetch B
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        B,
        sweep_inames=B_sweep_inames,
        precompute_outer_inames=precompute_outer_inames,
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id=b_prftch_insn_id,
        within=within,
        default_tag=None,
        temporary_name=b_prftch_tmp,
    )
    B_prftch_inames = tuple(
        cast("prim.Variable", idx).name
        for idx in cast(
            "prim.Subscript",
            cast(
                "lp.Assignment", t_unit[kernel_name].id_to_insn[b_prftch_insn_id]
            ).assignee,
        ).index_tuple
    )
    logger.info("Precomputed B")

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_id)
    logger.info("Realized reduction")

    (acc_name,) = (
        frozenset(t_unit[kernel_name].temporary_variables)
        & t_unit[kernel_name].id_to_insn[insn_id].read_dependency_names()
    )
    inames_to_duplicate = sorted({rx_inner, ry_inner})

    for iname_to_duplicate in inames_to_duplicate:
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, iname_to_duplicate, only_var_names={acc_name}
        )

    # {{{ prefetch outer products to registers

    if t_redns[iredn_idx_to_prftch] != 1:
        (sum_redn_update_insn,) = [
            insn
            for insn in t_unit[kernel_name].instructions
            if acc_name in insn.read_dependency_names()
            and acc_name in insn.write_dependency_names()
        ]

        t_unit = lp.add_prefetch(
            t_unit,
            a_prftch_tmp,
            sweep_inames=inner_redn_inames[iredn_idx_to_prftch],
            default_tag="unr",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            prefetch_insn_id=a_reg_prftch_insn_id,
            within=lp_match.Id(sum_redn_update_insn.id),
            fetch_outer_inames=sum_redn_update_insn.within_inames
            - {inner_redn_inames[iredn_idx_to_prftch]},
        )
        t_unit = lp.add_prefetch(
            t_unit,
            b_prftch_tmp,
            sweep_inames=inner_redn_inames[iredn_idx_to_prftch],
            default_tag="unr",
            temporary_address_space=lp.AddressSpace.PRIVATE,
            prefetch_insn_id=b_reg_prftch_insn_id,
            within=lp_match.Id(sum_redn_update_insn.id),
            fetch_outer_inames=sum_redn_update_insn.within_inames
            - {inner_redn_inames[iredn_idx_to_prftch]},
        )

    # }}}

    tags_to_inames_duplicate = (
        dict.fromkeys(inames_to_duplicate, "unr") if unroll_rx_ry else {}
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"writes:{acc_name} and not reads:{acc_name}",
        tags=tags_to_inames_duplicate,
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"reads:{acc_name} and not writes:{acc_name}",
        tags=tags_to_inames_duplicate,
    )

    logger.info("Done with iname duplication.")

    z_block_inames = tuple(
        (
            rx_outer
            if ifree_idx == i_rx
            else (ry_outer if ifree_idx == i_ry else free_index)
        )
        for ifree_idx, free_index in enumerate(free_indices)
        if ifree_idx not in [i_tx, i_ty]
    )

    if len(A_prftch_inames) > 0:
        t_unit = lp.split_iname(t_unit, A_prftch_inames[-1], tx, inner_tag="l.0")
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([tx_inner]), lp_match.Id(a_prftch_insn_id)
        )

    if len(A_prftch_inames) > 1:
        t_unit = lp.split_iname(t_unit, A_prftch_inames[-2], ty, inner_tag="l.1")
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([ty_inner]), lp_match.Id(a_prftch_insn_id)
        )
    logger.info("Done with parallelizing prftch(A).")

    if len(B_prftch_inames) > 0:
        t_unit = lp.split_iname(t_unit, B_prftch_inames[-1], tx, inner_tag="l.0")
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([tx_inner]), lp_match.Id(b_prftch_insn_id)
        )

    if len(B_prftch_inames) > 1:
        t_unit = lp.split_iname(t_unit, B_prftch_inames[-2], ty, inner_tag="l.1")
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([ty_inner]), lp_match.Id(b_prftch_insn_id)
        )
    logger.info("Done with parallelizing prftch(B).")

    t_unit = lp.add_inames_to_insn(
        t_unit,
        frozenset([tx_outer, ty_outer]),
        lp_match.Or(
            (
                lp_match.Id(a_prftch_insn_id),
                lp_match.Id(b_prftch_insn_id),
                lp_match.Id(a_reg_prftch_insn_id),
                lp_match.Id(b_reg_prftch_insn_id),
            )
        ),
    )

    if z_block_inames:
        t_unit = lp.join_inames(t_unit, z_block_inames, izblock)
        t_unit = lp.tag_inames(t_unit, {izblock: "g.2"})
        t_unit = lp.add_inames_to_insn(
            t_unit,
            frozenset([izblock]),
            lp_match.Or(
                (
                    lp_match.Id(a_prftch_insn_id),
                    lp_match.Id(b_prftch_insn_id),
                    lp_match.Id(a_reg_prftch_insn_id),
                    lp_match.Id(b_reg_prftch_insn_id),
                )
            ),
        )
    logger.info("Done with join inames.")

    t_unit = lp.prioritize_loops(
        t_unit,
        [
            redn_iname if t_redn == 1 else outer_iname
            for redn_iname, t_redn, outer_iname in zip(
                redn_indices, t_redns, outer_redn_inames, strict=True
            )
        ],
    )

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    from feinsum.utils import get_tccg_benchmark

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    for ibenchmark in range(12, 13):
        expr = get_tccg_benchmark(ibenchmark)
        print(f"{ibenchmark = }.")
        print(f"Autotuning for {expr} on device {cq.device}")

        fnsm.autotune(
            expr,
            os.path.abspath(__file__),
            cq,
            skip_value_mismatch=True,
            test_limit=3,
        )

        best_config = min(
            fnsm.query(expr, cq.device, err_if_no_results=True),
            key=lambda query_info: query_info.runtime_in_sec,
        )
        from feinsum.measure import _stringify_runtime_comparison_vs_roofline

        with open("log.txt", "a") as fp:
            fp.write(f"{ibenchmark = }.\n")
            fp.write(
                _stringify_runtime_comparison_vs_roofline(
                    expr, best_config.runtime_in_sec, cq.device.name
                )
            )
            fp.write("\n")

    # # Enable while debugging ->
    # # evaluate a point in the parameter space.
    # ibenchmark = 12
    # expr = get_tccg_benchmark(ibenchmark)
    # from functools import partial

    # bound_transform = partial(
    #     transform,
    #     ensm=expr,
    #     i_thread_axis_mapping_perm=1,
    #     i_reg_axis_mapping_perm=1,
    #     log2_tx=4,
    #     log2_ty=3,
    #     log2_rx=2,
    #     log2_ry=3,
    #     log2_t_redns=(4,),
    #     unroll_rx_ry=True,
    #     iredn_idx_to_prftch=0,
    # )
    # print(
    #     fnsm.stringify_comparison_vs_roofline(
    #         expr,
    #         transform=bound_transform,
    #         cq=cq,
    #     )
    # )
