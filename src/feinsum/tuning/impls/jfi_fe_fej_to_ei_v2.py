import logging
from typing import Any

import loopy as lp
import loopy.match as lp_match

import feinsum as fnsm
from feinsum.loopy_utils.cse import hoist_cses
from feinsum.tuning import BoolParameter, IntParameter

logger = logging.getLogger(__name__)


def fe_out(i: int) -> str:
    if i == 0:
        return "_fe_out"

    return f"_fe_out_{i - 1}"


@fnsm.tuning.einsum_arg("nf", lambda e: e.args[0][1].shape[0])
@fnsm.tuning.einsum_arg("ni", lambda e: e.shape[1])
@fnsm.tuning.einsum_arg("nj", lambda e: e.args[0][2].shape[0])
@fnsm.tuning.einsum_arg("b", lambda e: e.b)
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
                fnsm.array("M", (nj, nf, ni)),
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
    # M_subst = sigma["M"]
    outs = tuple(sigma[fe_out(i)] for i in range(b))

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    vng = t_unit[kernel_name].get_var_name_generator()

    # Step 1: Split e -> (e_outer/g.0, e_inner/l.1); tag i -> l.0.
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
        # slabs=(0, 1),
    )
    t_unit = lp.tag_inames(t_unit, {i_iname: "l.0"})

    # Step 2: For each batch k, extract J(f,e)*u_k(f,e,j) as subst _tmp_Mu_k.
    # These terms don't depend on i so they can be precomputed into shared mem.
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

    # Step 3: Precompute each _tmp_Mu_k into LOCAL (shared) memory with shape
    # [n_e_per_wg, nf, nj].  Shared precompute_inames across all b batches
    # enables loop fusion so all b arrays are filled by one precompute loop.
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

    # {{{ CSE

    t_unit = lp.expand_subst(
        t_unit,
        within=lp_match.And(
            (
                lp_match.Iname(iprcmpt_e),
                lp_match.Iname(iprcmpt_f),
                lp_match.Iname(iprcmpt_j),
            )
        ),
    )
    t_unit = hoist_cses(
        t_unit,
        within=lp_match.And(
            (
                lp_match.Iname(iprcmpt_e),
                lp_match.Iname(iprcmpt_f),
                lp_match.Iname(iprcmpt_j),
            )
        ),
    )

    # }}}

    # Step 4: Join f and j precompute inames into a single flat fj iname.
    # This gives shared memory shape [n_e_per_wg][nf*nj] and a single thread
    # dimension (l.0) covering all nf*nj face-dof pairs during the load phase,
    # matching the expected kernel's s_flux[es][n] with n = face*nj + jdof.
    iprcmpt_fj = vng("iprcmpt_fj")
    t_unit = lp.join_inames(t_unit, [iprcmpt_f, iprcmpt_j], iprcmpt_fj)
    t_unit = lp.tag_inames(t_unit, {iprcmpt_e: "l.1", iprcmpt_fj: "l.0"})

    # Step 5: Join f and j in the main reduction loop as well so that the
    # accumulation loop iterates over the same flat fj index and the shared
    # memory is accessed as s_flux[e_inner][fj] (flat).
    fj_iname = vng("fj")
    t_unit = lp.join_inames(t_unit, [f_iname, j_iname], fj_iname, within=within)

    if unroll_j:
        t_unit = lp.tag_inames(t_unit, {fj_iname: "unr"})

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

    for Nfacedof, Nvoldof in [(3, 4), (6, 10), (10, 20), (15, 35)]:
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
            test_limit=30,
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
