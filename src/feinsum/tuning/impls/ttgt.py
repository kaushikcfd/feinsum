import logging
from typing import Any, cast

import islpy as isl
import loopy as lp
import loopy.match as lp_match
import numpy as np
import pymbolic.primitives as prim
from constantdict import constantdict
from pytools import memoize_on_first_arg

import feinsum as fnsm
from feinsum.tuning import IntParameter, einsum_arg, transform_param

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
    "log2_gemm_tx",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_gemm_ty",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_gemm_rx",
    lambda ensm: IntParameter(0, 5),
)
@transform_param(
    "log2_gemm_ry",
    lambda ensm: IntParameter(0, 5),
)
@transform_param(
    "lgo2_gemm_t_redn",
    lambda ensm: IntParameter(0, 5),
)
@transform_param(
    "log2_a_transpose_tx",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_a_transpose_ty",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_b_transpose_tx",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_b_transpose_ty",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_c_transpose_tx",
    lambda ensm: IntParameter(1, 5),
)
@transform_param(
    "log2_c_transpose_ty",
    lambda ensm: IntParameter(1, 5),
)
@memoize_on_first_arg
def transform(
    t_unit: lp.TranslationUnit,
    ensm: fnsm.BatchedEinsum,
    log2_gemm_tx: int,
    log2_gemm_ty: int,
    log2_gemm_rx: int,
    log2_gemm_ry: int,
    log2_gemm_t_redn: int,
    log2_a_transpose_tx: int,
    log2_a_transpose_ty: int,
    log2_b_transpose_tx: int,
    log2_b_transpose_ty: int,
    log2_c_transpose_tx: int,
    log2_c_transpose_ty: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:

    if ensm.ndim < 2:
        raise ValueError(
            "This algorithm needs al least two dimensions" " in the output array."
        )
    if len(ensm.all_size_params) != 0:
        raise NotImplementedError("Parametric lengths are not allowed.")

    if not _is_ensm_tensor_contraction(ensm):
        raise ValueError(f"{ensm} is not a tensor contraction.")

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
    A_idx_set, B_idx_set = [
        tuple(subst_map[idx] for idx in in_idx_set)
        for in_idx_set in ensm.in_idx_sets
    ]
    free_indices = tuple(subst_map[idx] for idx in ensm_free_indices)
    redn_indices = tuple(subst_map[idx] for idx in ensm_redn_indices)
    a_transpose_tmp, b_transpose_tmp, c_transpose_tmp = (
        vng("a_transpose_tmp"),
        vng("b_transpose_tmp"),
        vng("c_transpose_tmp"),
    )
    a_transpose_insn, b_transpose_insn = vng("a_transpose_insn"), vng(
        "b_tranpose_insn"
    )
    a_tmp_base, b_tmp_base, c_tmp_base = (
        vng("a_tmp_base"),
        vng("b_tmp_base"),
        vng("c_tmp_base"),
    )
    a_gemm_tmp, b_gemm_tmp, c_gemm_tmp = vng("a_gemm"), vng("b_gemm"), vng("c_gemm")
    a_gemm_subst, b_gemm_subst = vng("a_gemm_subst"), vng("b_gemm_subst")
    i_gemm, j_gemm, k_gemm = vng("i_gemm"), vng("j_gemm"), vng("k_gemm")
    gemm_insn_id = ing("gemm")

    # Step 1. Precompute A with tranpose.
    a_prcmpt_inames = tuple(vng(f"iprftch_a_{i}") for i in range(len(A_idx_set)))
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        A,
        sweep_inames=frozenset(A_idx_set),
        precompute_inames=a_prcmpt_inames,
        precompute_outer_inames=frozenset(),
        temporary_address_space=lp.AddressSpace.GLOBAL,
        storage_axes=(
            [
                subst_arg
                for idx, subst_arg in zip(
                    A_idx_set,
                    t_unit[kernel_name].substitutions[A].arguments,
                    strict=True,
                )
                if idx in free_indices
            ]
            + [
                t_unit[kernel_name]
                .substitutions[A]
                .arguments[A_idx_set.index(redn_idx)]
                for redn_idx in redn_indices
            ]
        ),
        temporary_name=a_transpose_tmp,
        compute_insn_id=a_transpose_insn,
        default_tag=None,
        within=within,
    )

    # {{{ Step 2. Precompute A with tranpose.

    b_prcmpt_inames = tuple(vng(f"iprftch_b_{i}") for i in range(len(B_idx_set)))
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        B,
        sweep_inames=frozenset(B_idx_set),
        precompute_inames=b_prcmpt_inames,
        precompute_outer_inames=frozenset(),
        temporary_address_space=lp.AddressSpace.GLOBAL,
        temporary_name=b_transpose_tmp,
        compute_insn_id=b_transpose_insn,
        storage_axes=(
            [
                t_unit[kernel_name]
                .substitutions[B]
                .arguments[B_idx_set.index(redn_idx)]
                for redn_idx in redn_indices
            ]
            + [
                subst_arg
                for idx, subst_arg in zip(
                    B_idx_set,
                    t_unit[kernel_name].substitutions[B].arguments,
                    strict=True,
                )
                if idx in free_indices
            ]
        ),
        default_tag=None,
        within=lp_match.Id(insn_id),
    )

    # }}}

    # {{{ Step 3. Create GEMM of the form "ik,kj->ij" from the transposed operands.

    knl = t_unit[kernel_name]
    from pytools import product

    length_i, length_j = [
        product(
            cast("int", ensm.index_to_dim_length[idx])
            for idx in in_idx_set
            if idx in ensm_free_indices
        )
        for in_idx_set in ensm.in_idx_sets
    ]
    length_k = product(
        cast("int", ensm.index_to_dim_length[idx]) for idx in ensm_redn_indices
    )

    gemm_isl_space = isl.Space.create_from_names(
        isl.DEFAULT_CONTEXT, set=[i_gemm, j_gemm, k_gemm]
    )
    gemm_domain = (
        isl.BasicSet.universe(gemm_isl_space)
        .add_constraint(
            isl.Constraint.ineq_from_names(gemm_isl_space, {1: 0, i_gemm: 1})
        )
        .add_constraint(
            isl.Constraint.ineq_from_names(
                gemm_isl_space, {1: length_i - 1, i_gemm: -1}
            )
        )
        .add_constraint(
            isl.Constraint.ineq_from_names(gemm_isl_space, {1: 0, j_gemm: 1})
        )
        .add_constraint(
            isl.Constraint.ineq_from_names(
                gemm_isl_space, {1: length_j - 1, j_gemm: -1}
            )
        )
        .add_constraint(
            isl.Constraint.ineq_from_names(gemm_isl_space, {1: 0, k_gemm: 1})
        )
        .add_constraint(
            isl.Constraint.ineq_from_names(
                gemm_isl_space, {1: length_k - 1, k_gemm: -1}
            )
        )
    )
    gemm_insn = lp.Assignment(
        f"{c_gemm_tmp}[{i_gemm}, {j_gemm}]",
        (
            f"sum({k_gemm},"
            f"{a_gemm_subst}({i_gemm}, {k_gemm}) *"
            f"{b_gemm_subst}({k_gemm}, {j_gemm}))"
        ),
        id=gemm_insn_id,
        within_inames=frozenset({i_gemm, j_gemm}),
        depends_on=t_unit[kernel_name].id_to_insn[insn_id].depends_on,
    )
    gbarrier_insn = lp.BarrierInstruction(  # type: ignore[no-untyped-call]
        id=ing("gbarrier_gemm_transpose"),
        synchronization_kind="global",
        mem_kind="global",
        depends_on=frozenset([gemm_insn_id]),
    )
    c_transpose_indices = [
        *[idx for idx in A_idx_set if idx in free_indices],
        *[idx for idx in B_idx_set if idx in free_indices],
    ]
    c_transpose_insn = (
        t_unit[kernel_name]
        .id_to_insn[insn_id]
        .copy(
            expression=f"{c_transpose_tmp}[{','.join(c_transpose_indices)}]",
            depends_on=frozenset({gbarrier_insn.id}),
        )
    )
    new_temps = dict(knl.temporary_variables)
    new_temps[a_transpose_tmp] = new_temps[a_transpose_tmp].copy(
        dtype=ensm.arg_to_dtype[ensm_A],
        base_storage=a_tmp_base,
        offset=0,
    )
    new_temps[b_transpose_tmp] = new_temps[b_transpose_tmp].copy(
        dtype=ensm.arg_to_dtype[ensm_B],
        base_storage=b_tmp_base,
        offset=0,
    )
    new_temps[a_gemm_tmp] = lp.TemporaryVariable(
        name=a_gemm_tmp,
        dtype=ensm.arg_to_dtype[ensm_A],
        shape=(length_i, length_k),
        address_space=lp.AddressSpace.GLOBAL,
        base_storage=a_tmp_base,
        offset=0,
    )
    new_temps[b_gemm_tmp] = lp.TemporaryVariable(
        name=b_gemm_tmp,
        dtype=ensm.arg_to_dtype[ensm_B],
        shape=(length_k, length_j),
        address_space=lp.AddressSpace.GLOBAL,
        base_storage=b_tmp_base,
        offset=0,
    )
    new_temps[c_gemm_tmp] = lp.TemporaryVariable(
        name=c_gemm_tmp,
        dtype=np.result_type(ensm.arg_to_dtype[ensm_A], ensm.arg_to_dtype[ensm_B]),
        shape=(length_k, length_j),
        address_space=lp.AddressSpace.GLOBAL,
        base_storage=c_tmp_base,
        offset=0,
    )
    new_temps[c_transpose_tmp] = lp.TemporaryVariable(
        name=c_transpose_tmp,
        dtype=np.result_type(ensm.arg_to_dtype[ensm_A], ensm.arg_to_dtype[ensm_B]),
        shape=tuple(
            cast("int", ensm.index_to_dim_length[idx])
            for idx in ensm.in_idx_sets[0] + ensm.in_idx_sets[1]
            if idx in ensm_free_indices
        ),
        address_space=lp.AddressSpace.GLOBAL,
        base_storage=c_tmp_base,
        offset=0,
    )
    new_temps[a_tmp_base] = lp.TemporaryVariable(
        a_tmp_base,
        dtype=ensm.arg_to_dtype[ensm_A],
        shape=(length_i * length_k,),
        address_space=lp.AddressSpace.GLOBAL,
    )
    new_temps[b_tmp_base] = lp.TemporaryVariable(
        b_tmp_base,
        dtype=ensm.arg_to_dtype[ensm_B],
        shape=(length_k * length_j,),
        address_space=lp.AddressSpace.GLOBAL,
    )
    new_temps[c_tmp_base] = lp.TemporaryVariable(
        c_tmp_base,
        dtype=np.result_type(ensm.arg_to_dtype[ensm_A], ensm.arg_to_dtype[ensm_B]),
        shape=(length_i * length_j,),
        address_space=lp.AddressSpace.GLOBAL,
    )
    new_substs = {
        **knl.substitutions,
        **{
            a_gemm_subst: lp.SubstitutionRule(
                a_gemm_subst,
                ("d_0", "d_1"),
                prim.Subscript(
                    prim.Variable(a_gemm_tmp),
                    (prim.Variable("d_0"), prim.Variable("d_1")),
                ),
            ),
            b_gemm_subst: lp.SubstitutionRule(
                b_gemm_subst,
                ("d_0", "d_1"),
                prim.Subscript(
                    prim.Variable(b_gemm_tmp),
                    (prim.Variable("d_0"), prim.Variable("d_1")),
                ),
            ),
        },
    }

    knl = knl.copy(
        domains=[*knl.domains, gemm_domain],
        instructions=[
            *[
                c_transpose_insn if insn.id == insn_id else insn
                for insn in knl.instructions
            ],
            gemm_insn,
            gbarrier_insn,
        ],
        temporary_variables=constantdict(new_temps),
        substitutions=constantdict(new_substs),
    )
    knl = lp.remove_unused_inames(knl, inames=redn_indices)
    knl = knl.copy(
        silenced_warnings=knl.silenced_warnings
        | frozenset(
            [
                f"read_no_write({c_transpose_tmp})",
                f"read_no_write({a_gemm_tmp})",
                f"read_no_write({b_gemm_tmp})",
            ]
        )
    )
    t_unit = t_unit.with_kernel(knl)
    del knl

    # }}}

    # {{{ Parallelize A transpose

    t_unit = lp.split_iname(
        t_unit,
        a_prcmpt_inames[-1],
        2**log2_a_transpose_tx,
        inner_tag="l.0",
        outer_tag="g.0",
    )
    t_unit = lp.split_iname(
        t_unit,
        a_prcmpt_inames[-2],
        2**log2_a_transpose_ty,
        inner_tag="l.1",
        outer_tag="g.1",
    )

    # }}}

    # {{{ Parallelize B transpose

    t_unit = lp.split_iname(
        t_unit,
        b_prcmpt_inames[-1],
        2**log2_b_transpose_tx,
        inner_tag="l.0",
        outer_tag="g.0",
    )
    t_unit = lp.split_iname(
        t_unit,
        b_prcmpt_inames[-2],
        2**log2_b_transpose_ty,
        inner_tag="l.1",
        outer_tag="g.1",
    )

    # }}}

    # {{{ Transform GEMM.

    from feinsum.tuning.impls.cogent_w_register_prftch_w_reg_tiling import (
        transform as gemm_transform,
    )

    t_unit = gemm_transform(
        t_unit,
        fnsm.einsum(
            "ik,kj->ij",
            fnsm.array("A", (length_i, length_k), ensm.arg_to_dtype[ensm_A]),
            fnsm.array("B", (length_k, length_j), ensm.arg_to_dtype[ensm_B]),
        ),
        i_thread_axis_mapping_perm=1,
        i_reg_axis_mapping_perm=1,
        log2_tx=log2_gemm_tx,
        log2_ty=log2_gemm_ty,
        log2_rx=log2_gemm_rx,
        log2_ry=log2_gemm_ry,
        log2_t_redns=(log2_gemm_t_redn,),
        unroll_rx_ry=True,
        iredn_idx_to_prftch=0,
        insn_match=lp_match.Id(gemm_insn_id),
        kernel_name=kernel_name,
    )

    # }}}

    # {{{ Parallelize C transpose

    t_unit = lp.split_iname(
        t_unit,
        free_indices[-1],
        2**log2_c_transpose_tx,
        inner_tag="l.0",
        outer_tag="g.0",
    )
    t_unit = lp.split_iname(
        t_unit,
        free_indices[-2],
        2**log2_c_transpose_ty,
        inner_tag="l.1",
        outer_tag="g.1",
    )

    # }}}

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    from feinsum.utils import get_tccg_benchmark

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    for ibenchmark in range(22, 23):
        expr = get_tccg_benchmark(ibenchmark)
        from functools import partial

        bound_transform = partial(
            transform,
            ensm=expr,
            log2_gemm_tx=4,
            log2_gemm_ty=3,
            log2_gemm_rx=2,
            log2_gemm_ry=4,
            log2_gemm_t_redn=4,
            log2_a_transpose_tx=5,
            log2_a_transpose_ty=3,
            log2_b_transpose_tx=5,
            log2_b_transpose_ty=3,
            log2_c_transpose_tx=5,
            log2_c_transpose_ty=3,
        )
        print(
            fnsm.stringify_comparison_vs_roofline(
                expr,
                transform=bound_transform,
                cq=cq,
            )
        )
    _ = 1 / 0

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
