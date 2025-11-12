import logging
import math
from typing import Any, cast

import loopy as lp
import numpy as np
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
    "i_axis_mapping_perm",
    lambda ensm: IntParameter(0, math.perm(ensm.ndim, min(ensm.ndim, 4)) - 1),
)
@transform_param(
    "log2_output_tile_lengths",
    lambda ensm: tuple(IntParameter(1, 5) for _ in range(min(ensm.ndim, 4))),
)
@transform_param(
    "log2_t_redns",
    lambda ensm: tuple(IntParameter(0, 5) for i in range(get_n_redn_dim(ensm))),
)
@memoize_on_first_arg
def transform(
    t_unit: lp.TranslationUnit,
    ensm: fnsm.BatchedEinsum,
    i_axis_mapping_perm: int,
    log2_output_tile_lengths: tuple[int, ...],
    log2_t_redns: tuple[int, ...],
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    if ensm.ndim < 2:
        raise ValueError(
            "This algorithm needs al least two dimensions" " in the output array."
        )
    if len(ensm.all_size_params) != 0:
        raise NotImplementedError("Parametric lengths are not allowed.")

    output_tile_lengths = tuple(
        2**log2_output_tile_length
        for log2_output_tile_length in log2_output_tile_lengths
    )
    t_redns = tuple(2**log2_t_redn for log2_t_redn in log2_t_redns)

    import itertools

    i_tx: int
    i_ty: int
    i_rx: int | None
    i_ry: int | None

    for i, index_mapping in enumerate(
        itertools.permutations(list(range(ensm.ndim)), min(4, ensm.ndim))
    ):
        if i == i_axis_mapping_perm:
            if len(index_mapping) == 2:
                i_tx, i_ty = index_mapping
                i_rx = None
                i_ry = None
            elif len(index_mapping) == 3:
                i_tx, i_ty, i_rx = index_mapping
                i_ry = None
            elif len(index_mapping) == 4:
                i_tx, i_ty, i_rx, i_ry = index_mapping
            else:
                raise AssertionError()
            break
    else:
        raise AssertionError(
            f"{i_axis_mapping_perm} is an invalid" " permutation index."
        )
    tx: int
    ty: int
    rx: int | None
    ry: int | None
    if len(output_tile_lengths) == 4:
        tx, ty, rx, ry = output_tile_lengths
    elif len(output_tile_lengths) == 3:
        ry = None
        tx, ty, rx = output_tile_lengths
    elif len(output_tile_lengths) == 2:
        rx, ry = None, None
        tx, ty = output_tile_lengths
    else:
        raise ValueError(
            "len(output_tile_lengths) can be either 2, 3, or 4;"
            f" got {len(output_tile_lengths)}."
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

    if i_rx is not None:
        i_rx_dim_length = ensm.index_to_dim_length[ensm.out_idx_set[i_rx]]
        assert rx is not None
        if isinstance(i_rx_dim_length, INT_CLASSES) and rx > i_rx_dim_length:
            raise fnsm.InvalidParameterError("Rx > Nx")

    if i_ry is not None:
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

    # Verify Tx, Ty, Rx, Ry are distinct dimensions
    # ---------------------------------------------
    if len({i_rx, i_tx, i_ry, i_ty, None}) != min(4, ensm.ndim) + 1:
        raise AssertionError("i_rx, i_tx, i_ty, i_ry are not distinct.")

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
            if i_tile is not None and ensm.out_idx_set[i_tile] in idx_set
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
    if (rx or 1) * (ry or 1) * np.result_type(
        ensm.arg_to_dtype[ensm_A], ensm.arg_to_dtype[ensm_B]
    ).itemsize > REG_FILE_SPACE_PER_WI:
        raise fnsm.InvalidParameterError("Exceeds register file limits")

    assert tx <= 32 and ty <= 32

    # Note: we do not restrict the tile-lengths(tx, ty) as in this case the
    # tile-lengths also correspond to the execution grid size.
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

    # {{{ split free indices

    t_unit = lp.split_iname(
        t_unit,
        free_indices[i_tx],
        tx,
        inner_iname=tx_inner,
        outer_iname=tx_outer,
        inner_tag="l.0",
        outer_tag="g.0",
    )
    t_unit = lp.split_iname(
        t_unit,
        free_indices[i_ty],
        ty,
        inner_iname=ty_inner,
        outer_iname=ty_outer,
        inner_tag="l.1",
        outer_tag="g.1",
    )
    if i_rx is not None:
        assert rx is not None
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_rx],
            rx,
            inner_iname=rx_inner,
            outer_iname=rx_outer,
        )
    if i_ry is not None:
        assert ry is not None
        t_unit = lp.split_iname(
            t_unit,
            free_indices[i_ry],
            ry,
            inner_iname=ry_inner,
            outer_iname=ry_outer,
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
    A_prftch_inames: list[str | None] = []
    B_sweep_inames: list[str] = []
    B_prftch_inames: list[str | None] = []

    for (sweep_inames, prftch_inames), in_idx_set in zip(
        ((A_sweep_inames, A_prftch_inames), (B_sweep_inames, B_prftch_inames)),
        ensm.in_idx_sets,
        strict=True,
    ):
        for in_idx in in_idx_set:
            if in_idx in ensm.out_idx_set:
                if ensm.out_idx_set.index(in_idx) == i_tx:
                    sweep_inames.append(tx_inner)
                    if tx != 1:
                        prftch_inames.append(vng("iprftch_tx"))
                    else:
                        prftch_inames.append(None)
                elif ensm.out_idx_set.index(in_idx) == i_ty:
                    sweep_inames.append(ty_inner)
                    if ty != 1:
                        prftch_inames.append(vng("iprftch_ty"))
                    else:
                        prftch_inames.append(None)
                elif ensm.out_idx_set.index(in_idx) == i_rx:
                    sweep_inames.append(rx_inner)
                    if rx != 1:
                        prftch_inames.append(vng("iprftch_rx"))
                    else:
                        prftch_inames.append(None)
                elif ensm.out_idx_set.index(in_idx) == i_ry:
                    sweep_inames.append(ry_inner)
                    if ry != 1:
                        prftch_inames.append(vng("iprftch_ry"))
                    else:
                        prftch_inames.append(None)
                else:
                    # Not sweep this iname as it is purely a precompute outer iname.
                    prftch_inames.append(None)
            else:
                assert in_idx in ensm_redn_indices
                if t_redns[ensm_redn_indices.index(in_idx)] == 1:
                    # Not sweep this iname as it is purely a precompute outer iname.
                    prftch_inames.append(None)
                else:
                    sweep_inames.append(
                        inner_redn_inames[ensm_redn_indices.index(in_idx)]
                    )
                    prftch_inames.append(vng(f"iprftch_{in_idx}"))

    # precompute A
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        A,
        sweep_inames=A_sweep_inames,
        precompute_inames=A_prftch_inames,
        precompute_outer_inames=precompute_outer_inames,
        temporary_address_space=lp.AddressSpace.LOCAL,
        within=within,
        compute_insn_id=a_prftch_insn_id,
        default_tag=None,
    )

    assert (frozenset(A_prftch_inames) - {None} | precompute_outer_inames) == (
        t_unit[kernel_name].id_to_insn[a_prftch_insn_id].within_inames
    )
    logger.info("Precomputed A")

    # prefetch B
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        B,
        sweep_inames=B_sweep_inames,
        precompute_inames=B_prftch_inames,
        precompute_outer_inames=precompute_outer_inames,
        temporary_address_space=lp.AddressSpace.LOCAL,
        compute_insn_id=b_prftch_insn_id,
        within=within,
        default_tag=None,
    )
    assert (frozenset(B_prftch_inames) - {None} | precompute_outer_inames) == (
        t_unit[kernel_name].id_to_insn[b_prftch_insn_id].within_inames
    )
    logger.info("Precomputed B")

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_id)
    logger.info("Realized reduction")

    (acc_name,) = (
        frozenset(t_unit[kernel_name].temporary_variables)
        & t_unit[kernel_name].id_to_insn[insn_id].read_dependency_names()
    )
    if i_rx is not None and i_ry is not None:
        inames_to_duplicate = sorted({rx_inner, ry_inner})
    elif i_rx is not None:
        inames_to_duplicate = sorted({rx_inner})
    elif i_ry is not None:
        inames_to_duplicate = sorted({ry_inner})
    else:
        inames_to_duplicate = []

    for iname_to_duplicate in inames_to_duplicate:
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, iname_to_duplicate, only_var_names={acc_name}
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

    A_prftch_inames_not_none = [iname for iname in A_prftch_inames[::-1] if iname]
    B_prftch_inames_not_none = [iname for iname in B_prftch_inames[::-1] if iname]
    if len(A_prftch_inames_not_none) > 0:
        t_unit = lp.split_iname(
            t_unit, A_prftch_inames_not_none[0], tx, inner_tag="l.0"
        )
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([tx_inner]), lp_match.Id(a_prftch_insn_id)
        )

    if len(A_prftch_inames_not_none) > 1:
        t_unit = lp.split_iname(
            t_unit, A_prftch_inames_not_none[1], ty, inner_tag="l.1"
        )
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([ty_inner]), lp_match.Id(a_prftch_insn_id)
        )
    logger.info("Done with parallelizing prftch(A).")

    if len(B_prftch_inames_not_none) > 0:
        t_unit = lp.split_iname(
            t_unit, B_prftch_inames_not_none[0], tx, inner_tag="l.0"
        )
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([tx_inner]), lp_match.Id(b_prftch_insn_id)
        )

    if len(B_prftch_inames_not_none) > 1:
        t_unit = lp.split_iname(
            t_unit, B_prftch_inames_not_none[1], ty, inner_tag="l.1"
        )
    else:
        t_unit = lp.add_inames_to_insn(
            t_unit, frozenset([ty_inner]), lp_match.Id(b_prftch_insn_id)
        )
    logger.info("Done with parallelizing prftch(B).")

    t_unit = lp.join_inames(t_unit, z_block_inames, izblock)
    logger.info("Done with join inames.")

    t_unit = lp.tag_inames(t_unit, {izblock: "g.2"})
    t_unit = lp.add_inames_to_insn(
        t_unit,
        frozenset([tx_outer, ty_outer, izblock]),
        lp_match.Or((lp_match.Id(a_prftch_insn_id), lp_match.Id(b_prftch_insn_id))),
    )

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    from feinsum.utils import get_tccg_benchmark

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    ibenchmark = 2
    expr = get_tccg_benchmark(ibenchmark)
    print(f"Autotuning for {expr} on device {cq.device}")

    if 1:
        fnsm.autotune(expr, os.path.abspath(__file__), cq)
    else:
        from functools import partial

        # Enable while debugging ->
        # evaluate a point in the parameter space.
        bound_transform = partial(
            transform,
            ensm=expr,
            i_axis_mapping_perm=5,
            log2_output_tile_lengths=(1, 5, 5),
            log2_t_redns=(5,),
        )
        print(
            fnsm.stringify_comparison_vs_roofline(
                expr, transform=bound_transform, cq=cq
            )
        )
