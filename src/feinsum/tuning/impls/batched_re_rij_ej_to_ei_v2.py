from typing import Any, cast

import loopy as lp
import loopy.match as lp_match
import numpy as np
from pymbolic.typing import Expression, Integer

import feinsum as fnsm
from feinsum.tuning import BoolParameter, IntParameter


def _get_derivative_op_params(
    t_unit: lp.TranslationUnit,
    kernel_name: str | None = None,
    insn_match: Any | None = None,
) -> tuple[int, int, int, int, np.dtype[Any]]:
    from feinsum.einsum import INT_CLASSES, SizeParam

    batched_einsum, _ = fnsm.get_a_matched_einsum(
        t_unit,
        kernel_name=kernel_name,
        insn_match=insn_match,
        long_dim_length=36,
    )

    einsum = fnsm.einsum(batched_einsum.get_subscripts(), *batched_einsum.args[0])
    assert einsum.ndim == 2
    assert isinstance(einsum.shape[0], SizeParam)
    ni = einsum.shape[1]
    assert isinstance(ni, INT_CLASSES)
    assert len(einsum.args[0]) == 3

    assert (
        len([arg for arg in einsum.args[0] if arg.ndim == 2]) == 2
        and len([arg for arg in einsum.args[0] if arg.ndim == 3]) == 1
    )
    (mat,) = [arg for arg in einsum.args[0] if arg.ndim == 3]
    nr = mat.shape[0]
    nj = mat.shape[2]
    ref_einsum = fnsm.einsum(
        "re,rij,ej->ei",
        fnsm.array("J", (nr, "Ne"), mat.dtype),
        fnsm.array("M", (nr, ni, nj), mat.dtype),
        fnsm.array("u", ("Ne", nj), mat.dtype),
    )

    assert fnsm.canonicalize_einsum(einsum) == fnsm.canonicalize_einsum(ref_einsum)
    assert isinstance(nr, Integer) and isinstance(nj, Integer)
    return (batched_einsum.b, int(nr), int(ni), int(nj), mat.dtype)


def fe_out(i: int) -> str:
    if i == 0:
        return "_fe_out"
    return f"_fe_out_{i - 1}"


def _get_reduction_expression_with_inames(
    expr: Expression, inames: frozenset[str]
) -> lp.Reduction:
    """
    Returns the first immeditate
    """
    from feinsum.loopy_utils import ReductionCollector

    get_redns = ReductionCollector()
    redn_with_inames = [
        redn for redn in get_redns(expr) if frozenset(redn.inames_set) == inames
    ]
    if len(redn_with_inames) != 1:
        raise RuntimeError(
            "_get_reduction_expression_with_inames expected exactly one reduction"
            f", got {len(redn_with_inames)}."
        )

    return redn_with_inames[0]


@fnsm.tuning.einsum_arg("b", lambda e: e.b)
@fnsm.tuning.einsum_arg("dim", lambda e: e.args[0][1].shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("unroll_j", lambda e: BoolParameter())
@fnsm.tuning.transform_param("n_e_per_wg_log2", lambda e: IntParameter(1, 4))
def transform(
    t_unit: lp.TranslationUnit,
    b: int,
    dim: int,
    ndof: int,
    n_e_per_wg_log2: int,
    unroll_j: bool,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    from feinsum.loopy_utils import (
        hoist_invariant_multiplicative_terms_in_sum_reduction,
    )

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
        "re,rij,ej->ei",
        [
            [
                fnsm.array(f"J_{i}", (dim, "Ne")),
                fnsm.array("D", (dim, ndof, ndof)),
                fnsm.array("u", ("Ne", ndof)),
            ]
            for i in range(b)
        ],
    )

    sigma = fnsm.identify_as_einsum(
        t_unit,
        ref_einsum,
        kernel_name=kernel_name,
        insn_match=insn_match,
        long_dim_length=36,
    )

    i_iname = sigma["i"]
    e_iname = sigma["e"]
    j_iname = sigma["j"]
    r_iname = sigma["r"]
    u_var = sigma["u"]

    knl = t_unit[kernel_name]
    vng = knl.get_var_name_generator()
    ing = knl.get_instruction_id_generator()
    e_inner_iname = vng(e_iname + "_inner")
    e_outer_iname = vng(e_iname + "_outer")

    knl = lp.split_iname(
        knl,
        e_iname,
        2**n_e_per_wg_log2,
        inner_iname=e_inner_iname,
        outer_iname=e_outer_iname,
        outer_tag="g.0",
        inner_tag="l.1",
        within=within,
        slabs=(0, 1),
    )

    knl = lp.tag_inames(knl, {i_iname: "l.0", r_iname: "unr"})

    t_unit = t_unit.with_kernel(knl)
    iprcmpt_e, iprcmpt_j = vng("iprcmpt_e"), vng("iprcmpt_j")

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        u_var,
        sweep_inames=[e_inner_iname, j_iname],
        precompute_inames=[iprcmpt_e, iprcmpt_j],
        temporary_address_space=lp.AddressSpace.LOCAL,
        within=within,
    )
    t_unit = lp.tag_inames(t_unit, {iprcmpt_e: "l.1", iprcmpt_j: "l.0"})

    knl = t_unit[kernel_name]
    knl = lp.split_reduction_outward(knl, r_iname, within=within)
    knl = hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, j_iname, within=within
    )
    t_unit = t_unit.with_kernel(knl)
    du_subst_name = vng("_tmp_Du")
    t_unit = cast(
        "lp.TranslationUnit",
        lp.extract_subst(  # pyright: ignore[reportUnknownMemberType]
            t_unit,
            template=next(
                iter(
                    _get_reduction_expression_with_inames(
                        insn.expression, frozenset({j_iname})
                    )
                    for insn in t_unit[kernel_name].instructions
                    if lp_match.Or(
                        tuple(lp_match.Writes(sigma[fe_out(i)]) for i in range(b))
                    )(t_unit[kernel_name], insn)
                )
            ),
            subst_name=du_subst_name,
            parameters=(r_iname,),
            within=within,
        ),
    )
    prcmpt_r = vng("_prcmpt_r_Du")
    prcmpt_Du_id = ing("_compute_Du")
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        du_subst_name,
        sweep_inames=[r_iname],
        precompute_outer_inames=frozenset({e_outer_iname, e_inner_iname, i_iname}),
        precompute_inames=prcmpt_r,
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id=prcmpt_Du_id,
        default_tag="unr",
        within=within,
    )
    t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_Du_id)
    (acc_name,) = (
        t_unit[kernel_name].id_to_insn[prcmpt_Du_id].read_dependency_names()
        - t_unit[kernel_name].all_inames()
    )
    (acc_j_init_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_name in insn.write_dependency_names()
            and acc_name not in insn.read_dependency_names()
        )
    ]
    (acc_j_assign_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_name in insn.read_dependency_names()
            and acc_name not in insn.write_dependency_names()
        )
    ]
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, frozenset([prcmpt_r]), acc_name
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        [prcmpt_r],
        within=lp_match.Id(acc_j_init_id),
        tags={prcmpt_r: "unr"},  # pyright: ignore[reportArgumentType]
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        [prcmpt_r],
        within=lp_match.Id(acc_j_assign_id),
        tags={prcmpt_r: "unr"},  # pyright: ignore[reportArgumentType]
    )
    t_unit = lp.prioritize_loops(t_unit, (j_iname, prcmpt_r))
    if unroll_j:
        t_unit = lp.tag_inames(t_unit, {j_iname: "unr"})

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    Ndim = 3
    Nfield = 3

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    for Ndof in [4, 10, 20, 35]:
        expr = fnsm.batched_einsum(
            "re,rij,ej->ei",
            [
                [
                    fnsm.array(f"J{i}", (Ndim, "Nel")),
                    fnsm.array("D", (Ndim, Ndof, Ndof)),
                    fnsm.array("u", ("Nel", Ndof)),
                ]
                for i in range(Nfield)
            ],
        )

        fnsm.autotune(expr, os.path.abspath(__file__), cq, test_limit=20)

    if 0:
        for Ndof in [35]:
            expr = fnsm.batched_einsum(
                "re,rij,ej->ei",
                [
                    [
                        fnsm.array(f"J{i}", (Ndim, Ndim, "Nel")),
                        fnsm.array("D", (Ndim, Ndof, Ndof)),
                        fnsm.array("u", (Ndim, "Nel", Ndof)),
                    ]
                    for i in range(Nfield)
                ],
            )

            best_config = min(
                fnsm.query(expr, cq.device, err_if_no_results=True),
                key=lambda query_info: query_info.runtime_in_sec,
            )

            from feinsum.measure import _stringify_runtime_comparison_vs_roofline

            with open("log.txt", "a") as fp:
                fp.write(f"{Ndof = }.\n")
                fp.write("Expected perf:\n")
                fp.write(
                    _stringify_runtime_comparison_vs_roofline(
                        expr, best_config.runtime_in_sec, cq.device.name
                    )
                )
                fp.write("Actual perf:\n")
                fp.write(
                    fnsm.stringify_comparison_vs_roofline(
                        expr, transform=best_config.transform, cq=cq
                    )
                )
                fp.write("\n")
                fp.write(f"Compiler version: {best_config.compiler_version}")
                fp.write("\n")

# vim: fdm=marker
