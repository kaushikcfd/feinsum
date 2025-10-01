import feinsum as fnsm
import numpy as np
import loopy as lp
import logging
import math
from more_itertools import zip_equal as szip

from typing import Optional, Any, Tuple, List
from feinsum.einsum import SizeParam, INT_CLASSES
from feinsum.tuning import einsum_arg, transform_param, IntParameter
from feinsum.utils import get_n_redn_dim

logger = logging.getLogger(__name__)

# Values for Volta micro-arch. Generalize it via a class
MAX_SHARED_MEM_PER_WG = 48e3  # in bytes
REG_FILE_SPACE_PER_WI = 256*4  # in bytes


def _is_ensm_tensor_contraction(ensm: fnsm.BatchedEinsum) -> bool:
    # TC requires noperands == 2
    noperands = len(ensm.access_descriptors)
    if noperands != 2:
        return False

    # TC requires each free index be indexed in exactly one operand
    for idim in range(ensm.ndim):
        free_axis = fnsm.FreeAxis(idim)
        if not ((free_axis in ensm.access_descriptors[0])
                ^ (free_axis in ensm.access_descriptors[1])):
            return False

    # TC requires all reduction indices be present in both the operands
    if ({redn_idx
         for redn_idx in ensm.access_descriptors[0]
         if isinstance(redn_idx, fnsm.SummationAxis)}
         != {redn_idx
             for redn_idx in ensm.access_descriptors[1]
             if isinstance(redn_idx, fnsm.SummationAxis)}):
        return False

    # For now let's only allow a single einsum with single arguments in each of
    # them.
    if ensm.noutputs != 1:
        return False

    if len(ensm.use_matrix[0][0]) != 1:
        return False

    if len(ensm.use_matrix[0][1]) != 1:
        return False

    return True


def _get_indices(ensm: fnsm.BatchedEinsum) -> Tuple[Tuple[str, ...],
                                                  Tuple[str, ...]]:
    from feinsum.einsum import FreeAxis, SummationAxis
    nredn_dim = len({idx
                     for idx in ensm.index_names
                     if isinstance(idx, SummationAxis)})
    return (tuple(ensm.index_names[FreeAxis(idim)]
                  for idim in range(ensm.ndim)),
            tuple(ensm.index_names[SummationAxis(iredn_dim)]
                  for iredn_dim in range(nredn_dim)))


def _get_operand_names(ensm: fnsm.BatchedEinsum) -> Tuple[str, str]:
    assert ensm.noutputs == 1
    assert len(ensm.access_descriptors) == 2
    operand1, = ensm.use_matrix[0][0]
    operand2, = ensm.use_matrix[0][1]
    return operand1, operand2


@einsum_arg("ensm", lambda ensm: ensm)
@transform_param(
    "i_axis_mapping_perm",
    lambda ensm: IntParameter(0, math.perm(ensm.ndim, min(ensm.ndim, 4))-1))
@transform_param(
    "output_tile_lengths",
    lambda ensm: tuple(IntParameter(1, 32)
                       for _ in range(min(ensm.ndim, 4))))
@transform_param("t_redns", lambda ensm: tuple(IntParameter(1, 16)
                                               for i in range(get_n_redn_dim(ensm))))
def transform(t_unit: lp.TranslationUnit,
              ensm: fnsm.BatchedEinsum,
              i_axis_mapping_perm: int,
              output_tile_lengths: Tuple[int, ...],
              t_redns: Tuple[int, ...],
              insn_match: Optional[Any] = None,
              kernel_name: Optional[str] = None) -> lp.TranslationUnit:
    if ensm.ndim < 2:
        raise ValueError("This algorithm needs al least two dimensions"
                         " in the output array.")
    if not all(isinstance(dim_length, INT_CLASSES)
               for dim_length in ensm.index_to_dim_length().values()):
        raise NotImplementedError("Parametric lengths are not allowed.")

    import itertools
    i_tx: int
    i_ty: int
    i_rx: Optional[int]
    i_ry: Optional[int]

    for i, index_mapping in enumerate(itertools.permutations(list(range(ensm.ndim)),
                                                             min(4, ensm.ndim))):
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
        raise AssertionError(f"{i_axis_mapping_perm} is an invalid"
                             " permutation index.")
    tx: int
    ty: int
    rx: Optional[int]
    ry: Optional[int]
    if len(output_tile_lengths) == 4:
        tx, ty, rx, ry = output_tile_lengths
    elif len(output_tile_lengths) == 3:
        ry = None
        tx, ty, rx = output_tile_lengths
    elif len(output_tile_lengths) == 2:
        rx, ry = None, None
        tx, ty = output_tile_lengths
    else:
        raise ValueError("len(output_tile_lengths) can be either 2, 3, or 4;"
                         f" got {len(output_tile_lengths)}.")

    import loopy.match as lp_match
    kernel_name = kernel_name or t_unit.default_entrypoint.name
    knl = t_unit[kernel_name]
    within = lp_match.parse_match(insn_match)
    insn_id, = [insn.id for insn in knl.instructions if within(knl, insn)]
    del knl

    if not _is_ensm_tensor_contraction(ensm):
        raise ValueError(f"{ensm} is not a tensor contraction.")

    ensm_free_indices, ensm_redn_indices = _get_indices(ensm)
    ensm_A, ensm_B = _get_operand_names(ensm)

    # {{{ sanity checks on param space

    # Verify legality of tile lengths
    # -------------------------------
    i_tx_dim_length = ensm.index_to_dim_length()[fnsm.FreeAxis(i_tx)]
    if isinstance(i_tx_dim_length, INT_CLASSES) and tx > i_tx_dim_length:
        raise fnsm.InvalidParameterError("Tx > Nx")

    i_ty_dim_length = ensm.index_to_dim_length()[fnsm.FreeAxis(i_ty)]
    if isinstance(i_ty_dim_length, INT_CLASSES) and ty > i_ty_dim_length:
        raise fnsm.InvalidParameterError("Ty > Ny")

    if i_rx is not None:
        i_rx_dim_length = ensm.index_to_dim_length()[fnsm.FreeAxis(i_rx)]
        assert rx is not None
        if isinstance(i_rx_dim_length, INT_CLASSES) and rx > i_rx_dim_length:
            raise fnsm.InvalidParameterError("Rx > Nx")

    if i_ry is not None:
        i_ry_dim_length = ensm.index_to_dim_length()[fnsm.FreeAxis(i_ry)]
        assert ry is not None
        if isinstance(i_ry_dim_length, INT_CLASSES) and ry > i_ry_dim_length:
            raise fnsm.InvalidParameterError("Ry > Ny")

    for iredn_dim, redn_tile in enumerate(t_redns):
        i_redn_dim_length = ensm.index_to_dim_length()[fnsm.SummationAxis(iredn_dim)]
        if (isinstance(i_redn_dim_length, INT_CLASSES)
                and redn_tile > i_redn_dim_length):
            raise fnsm.InvalidParameterError(f"redn_dim.{iredn_dim}'s tile is "
                                             " greater the axis length.")

    # Verify Tx, Ty, Rx, Ry are distinct dimensions
    # ---------------------------------------------
    if len({i_rx, i_tx, i_ry, i_ty, None}) != min(4, ensm.ndim) + 1:
        raise AssertionError("i_rx, i_tx, i_ty, i_ry are not distinct.")

    # Verify shared memory
    # --------------------
    from pytools import product
    l_sm_a, l_sm_b = [product(tile_length
                              for tile_length, i_tile in [(tx, i_tx),
                                                          (ty, i_ty)]
                              if fnsm.FreeAxis(i_tile) in acc_descrs
                              ) * product(t_redns)
                      for acc_descrs in ensm.access_descriptors]
    if (l_sm_a*ensm.value_to_dtype[ensm_A].itemsize
            + l_sm_b*ensm.value_to_dtype[ensm_B].itemsize) > MAX_SHARED_MEM_PER_WG:
        raise fnsm.InvalidParameterError("Exceeds local memory limits")

    # Verify register file usage
    # --------------------------
    if (rx or 1) * (ry or 1) * np.result_type(
            ensm.value_to_dtype[ensm_A],
            ensm.value_to_dtype[ensm_B]).itemsize > REG_FILE_SPACE_PER_WI:
        raise fnsm.InvalidParameterError("Exceeds register file limits")

    assert tx <= 32 and ty <= 32

    # Note: we do not restrict the tile-lengths(tx, ty) as in this case the
    # tile-lengths also correspond to the execution grid size.
    new_t_redns = []
    for iredn_dim, t_redn in enumerate(t_redns):
        axis_len = ensm.index_to_dim_length()[fnsm.SummationAxis(iredn_dim)]
        if isinstance(axis_len, SizeParam):
            new_t_redns.append(t_redn)
        else:
            assert isinstance(axis_len, INT_CLASSES)
            n_tiles = math.ceil(axis_len/t_redn)
            new_t_redns.append(math.ceil(axis_len/n_tiles))

    assert len(new_t_redns) == len(t_redns)
    assert all(new_t_redn <= old_t_redn
               for new_t_redn, old_t_redn in zip(new_t_redns, t_redns))
    t_redns = tuple(new_t_redns)
    del new_t_redns

    # }}}

    # type-ignore-reason: mypy is correct, but we have asserted earlier that we
    # won't accept parametric dim lengths
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ensm,
        insn_match=insn_match,
        kernel_name=kernel_name,
        long_dim_length=max(  # type: ignore[type-var,operator]
            ensm.index_to_dim_length().values(),
            default=1) + 1,
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
    inner_redn_inames = tuple(vng(f"{redn_iname}_inner")
                              for redn_iname in redn_indices)
    outer_redn_inames = tuple(vng(f"{redn_iname}_outer")
                              for redn_iname in redn_indices)

    iouter = vng("iouter")
    a_prftch_insn_id = ing("a_prftch_insn")
    b_prftch_insn_id = ing("b_prftch_insn")

    # {{{ split free indices

    t_unit = lp.split_iname(t_unit, free_indices[i_tx], tx,
                            inner_iname=tx_inner, outer_iname=tx_outer,
                            inner_tag="l.0")
    t_unit = lp.split_iname(t_unit, free_indices[i_ty], ty,
                            inner_iname=ty_inner, outer_iname=ty_outer,
                            inner_tag="l.1")
    if i_rx is not None:
        t_unit = lp.split_iname(t_unit, free_indices[i_rx], rx,
                                inner_iname=rx_inner, outer_iname=rx_outer)
    if i_ry is not None:
        t_unit = lp.split_iname(t_unit, free_indices[i_ry], ry,
                                inner_iname=ry_inner, outer_iname=ry_outer)

    # }}}

    # {{{ split redn indices

    for redn_iname, t_redn, inner_iname, outer_iname in szip(redn_indices,
                                                             t_redns,
                                                             inner_redn_inames,
                                                             outer_redn_inames):
        if t_redn != 1:
            t_unit = lp.split_iname(t_unit, redn_iname, t_redn,
                                    inner_iname=inner_iname,
                                    outer_iname=outer_iname)

    # }}}

    precompute_outer_inames = frozenset({
        redn_outer_iname if t_redn != 1 else redn_iname
        for redn_outer_iname, redn_iname, t_redn in szip(outer_redn_inames,
                                                         redn_indices,
                                                         t_redns)
    })

    A_sweep_inames: List[str] = []
    A_prftch_inames: List[Optional[str]] = []
    B_sweep_inames: List[str] = []
    B_prftch_inames: List[Optional[str]] = []

    for (sweep_inames, prftch_inames), acc_descrs in szip(
        ((A_sweep_inames, A_prftch_inames),
         (B_sweep_inames, B_prftch_inames)),
        ensm.access_descriptors
    ):
        for acc_descr in acc_descrs:
            if isinstance(acc_descr, fnsm.FreeAxis):
                if acc_descr.output_index == i_tx:
                    sweep_inames.append(tx_inner)
                    if tx != 1:
                        prftch_inames.append(vng("iprftch_tx"))
                    else:
                        prftch_inames.append(None)
                elif acc_descr.output_index == i_ty:
                    sweep_inames.append(ty_inner)
                    if ty != 1:
                        prftch_inames.append(vng("iprftch_ty"))
                    else:
                        prftch_inames.append(None)
                elif acc_descr.output_index == i_rx:
                    sweep_inames.append(rx_inner)
                    if rx != 1:
                        prftch_inames.append(vng("iprftch_rx"))
                    else:
                        prftch_inames.append(None)
                elif acc_descr.output_index == i_ry:
                    sweep_inames.append(ry_inner)
                    if ry != 1:
                        prftch_inames.append(vng("iprftch_ry"))
                    else:
                        prftch_inames.append(None)
                else:
                    # Not sweep this iname as it is purely a precompute outer iname.
                    prftch_inames.append(None)
            elif isinstance(acc_descr, fnsm.SummationAxis):
                if t_redns[acc_descr.index] == 1:
                    # Not sweep this iname as it is purely a precompute outer iname.
                    prftch_inames.append(None)
                else:
                    sweep_inames.append(inner_redn_inames[acc_descr.index])
                    prftch_inames.append(
                        vng(f"iprftch_{ensm.index_names[acc_descr]}"))
            else:
                raise AssertionError(type(acc_descr))

    # precompute A
    t_unit = lp.precompute(t_unit, A,
                           sweep_inames=A_sweep_inames,
                           precompute_inames=A_prftch_inames,
                           precompute_outer_inames=precompute_outer_inames,
                           temporary_address_space=lp.AddressSpace.LOCAL,
                           within=within,
                           compute_insn_id=a_prftch_insn_id,
                           default_tag=None)

    assert (frozenset(A_prftch_inames) - {None} | precompute_outer_inames) == (
        t_unit[kernel_name].id_to_insn[a_prftch_insn_id].within_inames)
    logger.info("Precomputed A")

    # prefetch B
    t_unit = lp.precompute(t_unit, B,
                           sweep_inames=B_sweep_inames,
                           precompute_inames=B_prftch_inames,
                           precompute_outer_inames=precompute_outer_inames,
                           temporary_address_space=lp.AddressSpace.LOCAL,
                           compute_insn_id=b_prftch_insn_id,
                           within=within,
                           default_tag=None)
    assert (frozenset(B_prftch_inames) - {None} | precompute_outer_inames) == (
        t_unit[kernel_name].id_to_insn[b_prftch_insn_id].within_inames)
    logger.info("Precomputed B")

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_id)
    logger.info("Realized reduction")

    acc_name, = (frozenset(t_unit[kernel_name].temporary_variables)
                 & t_unit[kernel_name].id_to_insn[insn_id].read_dependency_names())
    if i_rx is not None and i_ry is not None:
        inames_to_duplicate = sorted({rx_inner, ry_inner})
    elif i_rx is not None:
        inames_to_duplicate = sorted({rx_inner})
    elif i_ry is not None:
        inames_to_duplicate = sorted({ry_inner})
    else:
        inames_to_duplicate = []

    for iname_to_duplicate in inames_to_duplicate:
        t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                      iname_to_duplicate,
                                                      only_var_names={acc_name})

    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"writes:{acc_name} and not reads:{acc_name}")
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"reads:{acc_name} and not writes:{acc_name}")

    logger.info("Done with iname duplication")

    block_inames = [
        tx_outer if ifree_idx == i_tx
        else ty_outer if ifree_idx == i_ty
        else rx_outer if ifree_idx == i_rx
        else ry_outer if ifree_idx == i_ry
        else free_index
        for ifree_idx, free_index in enumerate(free_indices)]

    A_prftch_iname_x, A_prftch_iname_y = [iname
                                          for iname in A_prftch_inames[::-1]
                                          if iname][:2]
    B_prftch_iname_x, B_prftch_iname_y = [iname
                                          for iname in B_prftch_inames[::-1]
                                          if iname][:2]
    t_unit = lp.split_iname(t_unit, A_prftch_iname_x, tx, inner_tag="l.0")
    t_unit = lp.split_iname(t_unit, A_prftch_iname_y, ty, inner_tag="l.1")
    logger.info("Done with parallelizing prftch(A)")

    t_unit = lp.split_iname(t_unit, B_prftch_iname_x, tx, inner_tag="l.0")
    t_unit = lp.split_iname(t_unit, B_prftch_iname_y, ty, inner_tag="l.1")
    logger.info("Done with parallelizing prftch(B)")

    t_unit = lp.join_inames(t_unit, block_inames, iouter)
    logger.info("Done with join inames")

    t_unit = lp.tag_inames(t_unit, {iouter: "g.0"})
    t_unit = lp.add_inames_to_insn(t_unit, frozenset([iouter]),
                                   lp_match.Or((lp_match.Id(a_prftch_insn_id),
                                                lp_match.Id(b_prftch_insn_id))))

    return t_unit


if __name__ == "__main__":
    import pyopencl as cl
    import os

    cl_ctx = cl.create_some_context()

    # expr = fnsm.einsum("aebf,fdec->abcd",
    #                    fnsm.array((72, 72, 72, 72), np.float64),
    #                    fnsm.array((72, 72, 72, 72), np.float64),
    #                    arg_names=["A", "B"])

    expr = fnsm.einsum("il,ljk->ijk",
                       fnsm.array((72, 72), np.float64),
                       fnsm.array((72, 72, 72), np.float64),
                       arg_names=["A", "B"])

    fnsm.autotune(expr, os.path.abspath(__file__), cl_ctx)
