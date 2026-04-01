import logging
from typing import Any

import loopy as lp
import loopy.match as lp_match

import feinsum as fnsm
from feinsum.tuning import BoolParameter, IntParameter

logger = logging.getLogger(__name__)


def fe_out(i: int) -> str:
    if i == 0:
        return "_fe_out"

    return f"_fe_out_{i - 1}"


@fnsm.tuning.einsum_arg("b", lambda e: e.b)
@fnsm.tuning.einsum_arg("nf", lambda e: e.args[0][2].shape[0])
@fnsm.tuning.einsum_arg("ni", lambda e: e.shape[1])
@fnsm.tuning.einsum_arg("nj", lambda e: e.args[0][2].shape[2])
@fnsm.tuning.transform_param("n_e_per_wg_log2", lambda e: IntParameter(1, 4))
@fnsm.tuning.transform_param("unroll_j", lambda e: BoolParameter())
def transform(
    t_unit: lp.TranslationUnit,
    b: int,
    nf: int,
    ni: int,
    nj: int,
    n_e_per_wg_log2: int,
    unroll_j: bool,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    import pymbolic.primitives as prim

    from feinsum.loopy_utils import (
        extract_multiplicative_terms_in_sum_reduction_as_subst,
    )

    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.batched_einsum(
        "ifj,fe,fej->ei",
        [
            [
                fnsm.array("M", (ni, nf, nj)),
                fnsm.array("J", (nf, "Ne")),
                fnsm.array(f"u_{i}", (nf, "Ne", nj)),
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
    f_iname = sigma["f"]
    M_subst = sigma["M"]
    outs = tuple(sigma[fe_out(i)] for i in range(b))

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    vng = t_unit[kernel_name].get_var_name_generator()

    # Step 1: Split e -> (e_outer/g.0, e_inner/l.1); tag i -> l.0
    e_inner_iname = vng(e_iname + "_inner")
    e_outer_iname = vng(e_iname + "_outer")
    t_unit = lp.split_iname(
        t_unit,
        e_iname,
        2**n_e_per_wg_log2,
        outer_tag="g.0",
        inner_tag="l.1",
        inner_iname=e_inner_iname,
        outer_iname=e_outer_iname,
        within=within,
        slabs=(0, 1),
    )
    t_unit = lp.tag_inames(t_unit, {i_iname: "l.0"})

    # Step 2: For each batch k, extract J(f,e)*u_k(f,e,j) as subst _tmp_Mu_k.
    # Terms that don't depend on i_iname are J and u_k; M/LIFT depends on i.
    # All b instructions keep the SAME f,j inames (no duplication) so that
    # Steps 3 and 4 can exploit loop fusion across the b accumulations.
    from loopy.symbolic import get_dependencies

    knl = t_unit[kernel_name]
    mu_subst_names = tuple(vng(f"_tmp_Mu_{ib}") for ib in range(b))

    for ib in range(b):
        knl = extract_multiplicative_terms_in_sum_reduction_as_subst(
            knl,
            within=lp_match.Writes(outs[ib]),
            subst_name=mu_subst_names[ib],
            arguments=[
                prim.Variable(e_inner_iname),
                prim.Variable(f_iname),
                prim.Variable(j_iname),
            ],
            terms_filter=lambda t: i_iname not in get_dependencies(t),
        )

    t_unit = t_unit.with_kernel(knl)

    # Step 3: Precompute each _tmp_Mu_k into LOCAL memory [e_inner, f, j].
    # Shape: (p_NblockS, nf, nj)
    # Shared precompute_outer_inames across all b precomputes enables loop fusion
    iprcmpt_e, iprcmpt_f, iprcmpt_j = (
        vng("iprcmpt_e"),
        vng("iprcmpt_f"),
        vng("iprcmpt_j"),
    )
    for ib in range(b):
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            mu_subst_names[ib],
            sweep_inames=[e_inner_iname, f_iname, j_iname],
            precompute_inames=[iprcmpt_e, iprcmpt_f, iprcmpt_j],
            precompute_outer_inames=frozenset({e_outer_iname}),
            temporary_address_space=lp.AddressSpace.LOCAL,
        )
    t_unit = lp.tag_inames(
        t_unit, {iprcmpt_e: "l.1", iprcmpt_f: "unr", iprcmpt_j: "l.0"}
    )

    # Step 4: Precompute M[i,f,j] as a scalar temp per (f,j) step.
    # With f,j in precompute_outer_inames (empty sweep), loopy creates one
    # scalar _tmp_M computed once per (f,j) iteration, reused across all b
    # accumulations in the fused loop
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        M_subst,
        sweep_inames=[],
        precompute_outer_inames=frozenset({
            e_outer_iname,
            e_inner_iname,
            i_iname,
            f_iname,
            j_iname,
        }),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        within=within,
    )

    if unroll_j:
        t_unit = lp.tag_inames(t_unit, {j_iname: "unr"})

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    Nface = 4
    Nfacedof = 15
    Nvoldof = 35

    cl_ctx = cl.create_some_context()
    cq = cl.CommandQueue(cl_ctx)

    for Nfields in [4]:
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

        fnsm.autotune(expr, os.path.abspath(__file__), cq, test_limit=10)

        best_config = min(
            fnsm.query(expr, cq.device, err_if_no_results=True),
            key=lambda query_info: query_info.runtime_in_sec,
        )

        from feinsum.measure import _stringify_runtime_comparison_vs_roofline

        with open("log.txt", "a") as fp:
            fp.write(f"{Nfields = }.\n")
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
