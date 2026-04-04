import math
from collections.abc import Iterable
from typing import Any, cast

import loopy as lp
import loopy.match as lp_match
from pymbolic.typing import Expression

import feinsum as fnsm
from feinsum import loopy_utils as lp_utils
from feinsum.tuning import IntParameter


def _fset_union(fsets: Iterable[frozenset[str]]) -> frozenset[str]:
    from functools import reduce

    return reduce(lambda x, y: x | y, fsets, cast("frozenset[str]", frozenset()))


def fe_out(i: int) -> str:
    if i == 0:
        return "_fe_out"
    return f"_fe_out_{i - 1}"


def _get_reduction_expression_with_inames(
    expr: Expression, inames: frozenset[str]
) -> lp.Reduction:
    """
    Returns all subexpressions of *expr* that are of type
    :class:`loopy.Reduction` and perform reductions across *inames*.
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


# Transformation to the following kernel:
# void grad_components(double* Jx, double* Jy, double* Jz, double* u, double* D,
#                      double* du_dx, double* du_dy, double* du_dz) {
#   int e = n_e_per_wg * gid(0) + lid(1);
#   double du_acc[i_tiles][3];
#
#   for (int i_tile = 0; i_tile < i_tiles; i_tile++)
#     for (int r =0; r < 3; r++)
#       du_acc[i_tile][r] = 0;
#
#   for (int j_tile = 0; j_tile < j_tiles; j_tile++) {
#     // Fetch uprcmpt[:,:] <- u[n_e_per_wg*gid(0):n_e_per_wg * gid(0)+n_e_per_wg,
#                                j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#     // uprcmpt is a local variable
#     for (int i_tile = 0; i_tile < i_tiles; i_tile++) {
#       // Fetch Dprcmpt[:,:] <- D[0:3,
#                                  i_tile*i_tile_len:(i_tile+1)*i_tile_len,
#                                  j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#       // Dprcmpt is a local variable.
#
#       for (int j = 0; j < j_tile_len; j++) {
#         for (int r = 0; r < 3; r++) {
#           du_acc[i_tile][r] += Dprcmpt[r,
#                                        i_tile*i_tilelen+lid(0),
#                                        j_tile*j_tilelen+j]
#                                * uprcmpt[lid(1), j];
#         }
#       }
#     }
#   }
#
#   double du_dx_acc[i_tiles], du_dy_acc[i_tiles], du_dz_acc[i_tiles];
#
#   for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#     du_dx_acc[i_tile]=0; du_dy_acc[i_tile]=0; du_dz_acc[i_tile]=0;
#   }
#
#   for (int r = 0; r < 3; r++) {
#     double Jxr = Jx[r][e], Jyr = Jy[r][e], Jzr = Jz[r][e];
#     for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#       du_dx_acc[i_tile] += Jxr * du_acc[i_tile][r];
#       du_dy_acc[i_tile] += Jyr * du_acc[i_tile][r];
#       du_dz_acc[i_tile] += Jzr * du_acc[i_tile][r];
#     }
#   }
#
#   for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#     du_dx[e][i_tile*i_tile_len + lid(0)] = du_dx_acc[i_tile];
#     du_dy[e][i_tile*i_tile_len + lid(0)] = du_dy_acc[i_tile];
#     du_dz[e][i_tile*i_tile_len + lid(0)] = du_dz_acc[i_tile];
#   }
# }
@fnsm.tuning.einsum_arg("noutputs", lambda e: e.b)
@fnsm.tuning.einsum_arg("ndim", lambda e: e.shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[2])
@fnsm.tuning.transform_param("n_e_per_wg", lambda e: IntParameter(2, 32))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[2] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[2] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    noutputs: int,
    ndim: int,
    ndof: int,
    n_e_per_wg: int,
    i_tiles: int,
    j_tiles: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:

    if n_e_per_wg * math.ceil((ndof) / i_tiles) > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    if (
        (ndim * math.ceil((ndof) / i_tiles) * math.ceil(ndof / j_tiles))
        + ndof * n_e_per_wg
    ) * 8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.batched_einsum(
        "re,rij,ej->ei",
        [
            [
                fnsm.array(f"J{i}", (ndim, "Nel")),
                fnsm.array("D", (ndim, ndof, ndof)),
                fnsm.array("u", ("Nel", ndof)),
            ]
            for i in range(noutputs)
        ],
    )

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    sigma = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=within, kernel_name=kernel_name
    )
    i = sigma["i"]
    j = sigma["j"]
    u = sigma["u"]
    e = sigma["e"]
    D = sigma["D"]
    r = sigma["r"]
    Js = tuple(sigma[f"J{i}"] for i in range(noutputs))
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")

    i_tile_iname = vng(f"{i}_tile")
    i_inner_iname = vng(f"{i}_inner")
    i_tile_len = math.ceil(ndof / i_tiles)

    j_tile_iname = vng(f"{j}_tile")
    j_inner_iname = vng(f"{j}_inner")
    j_tile_len = math.ceil(ndof / j_tiles)

    # names for u-prefetch inames
    uprcmpt_e = vng("uprcmpt_e")
    uprcmpt_j = vng("uprcmpt_j")

    # names for D-prefetch inames
    rprftch_D = vng("rprftchD")
    iprftch_D = vng("iprftchD")
    jprftch_D = vng("jprftchD")
    D_fetch = vng(f"{D}_fetch")
    D_fetch_id = ing("D_fetch_id")

    # names for Du substitution and its private precompute
    du_subst_name = vng("_subst_Du")
    du_tmp_name = vng("_tmp_Du")
    prcmpt_r = vng("_prcmpt_r_Du")
    prcmpt_itile = vng("_prcmpt_itile_Du")
    prcmpt_Du_id = ing("_compute_Du")

    # Instructions matching `within`
    matched_insn_ids = tuple(
        insn.id
        for insn in t_unit[kernel_name].instructions
        if within(t_unit[kernel_name], insn)
    )

    # }}}

    # {{{ Step 1: Split r outward and hoist J[r,e] out of the j-reduction
    #
    # Transforms: sum_{r,j} J[r,e]*D[r,i,j]*u[e,j]
    #          -> sum_r J[r,e] * (sum_j D[r,i,j]*u[e,j])
    #
    # This must happen before splitting j so that the hoisted form feeds into
    # extract_subst cleanly.

    t_unit = lp.split_reduction_outward(t_unit, r, within=within)
    knl = t_unit[kernel_name]
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, j, within=within
    )
    t_unit = t_unit.with_kernel(knl)
    del knl

    # }}}

    # {{{ Step 2: Extract Du(r) = sum_j D[r,i,j]*u[e,j] as a substitution rule
    #
    # The instruction now looks like: output[e,i] = sum_r J[r,e] * Du(r)
    # Du(r) will later be precomputed privately per work item and per i_tile,
    # reproducing du_acc[i_tile][r] from kernel.c.

    template = next(
        iter(
            _get_reduction_expression_with_inames(insn.expression, frozenset({j}))
            for insn in t_unit[kernel_name].instructions
            if lp_match.Or(
                tuple(lp_match.Writes(sigma[fe_out(k)]) for k in range(noutputs))
            )(t_unit[kernel_name], insn)
        )
    )
    t_unit = cast(
        "lp.TranslationUnit",
        lp.extract_subst(  # pyright: ignore[reportUnknownMemberType]
            t_unit,
            template=template,
            subst_name=du_subst_name,
            parameters=(r, e, i),
            within=within,
        ),
    )

    # }}}

    # {{{ Step 3: Split e, i, j inames

    t_unit = lp.split_iname(
        t_unit,
        e,
        n_e_per_wg,
        outer_tag="g.0",
        inner_tag="l.1",
        inner_iname=e_inner,
        outer_iname=e_outer,
        within=within,
        # slabs=(0, 1),
    )

    t_unit = lp.split_iname(
        t_unit,
        i,
        i_tile_len,
        outer_iname=i_tile_iname,
        inner_iname=i_inner_iname,
        within=within,
        inner_tag="l.0",
        outer_tag="unr",
    )

    t_unit = lp.split_iname(
        t_unit,
        j,
        j_tile_len,
        outer_iname=j_tile_iname,
        inner_iname=j_inner_iname,
        within=within,
    )

    # }}}

    # {{{ Step 4: Precompute u in LOCAL memory
    #
    # u_local[e_inner, j_inner] is loaded collaboratively per (e_outer, j_tile).

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        u,
        sweep_inames=[e_inner, j_inner_iname],
        precompute_inames=[uprcmpt_e, uprcmpt_j],
        precompute_outer_inames=frozenset({e_outer, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        within=within,
        default_tag=None,
    )
    t_unit = lp.tag_inames(t_unit, {uprcmpt_e: "l.1"})
    t_unit = lp.split_iname(t_unit, uprcmpt_j, n_e_per_wg, inner_tag="l.0")

    # }}}

    # {{{ Step 5: Precompute D in LOCAL memory
    #
    # D_local[r, i_inner, j_inner] is loaded collaboratively per
    # (e_outer, i_tile, j_tile), reproducing Dprcmpt from kernel.c.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        sweep_inames=[r, i_inner_iname, j_inner_iname],
        precompute_inames=[rprftch_D, iprftch_D, jprftch_D],
        precompute_outer_inames=frozenset({e_outer, i_tile_iname, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=D_fetch,
        compute_insn_id=D_fetch_id,
        default_tag=None,
        within=within,
    )
    t_unit = lp.tag_inames(t_unit, {iprftch_D: "l.0"})
    t_unit = lp.split_iname(t_unit, jprftch_D, n_e_per_wg, inner_tag="l.1")

    # }}}

    # {{{ Step 6: Precompute Du = sum_{j_tile,j_inner} D*u in PRIVATE memory
    #
    # Du_tmp[r] is computed per (e_outer, e_inner, i_tile, i_inner), accumulating
    # over (j_tile, j_inner) inside.  After privatize this becomes Du[i_tile, r],
    # reproducing du_acc[i_tile][r] from kernel.c.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        du_subst_name,
        sweep_inames=[r],
        precompute_inames=[prcmpt_r],
        precompute_outer_inames=frozenset(
            {
                e_outer,
                e_inner,
                i_tile_iname,
                i_inner_iname,
            }
        ),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        compute_insn_id=prcmpt_Du_id,
        default_tag="unr",
        temporary_name=du_tmp_name,
        within=within,
    )
    t_unit = lp.privatize_temporaries_with_inames(t_unit, i_tile_iname, du_tmp_name)
    t_unit = lp.duplicate_inames(
        t_unit,
        i_tile_iname,
        lp_match.Or((lp_match.Id(prcmpt_Du_id), lp_match.Id(D_fetch_id))),
        prcmpt_itile,
    )
    t_unit = lp.tag_inames(t_unit, {prcmpt_itile: "unr"})

    # }}}

    # {{{ Step 7: Realize reduction + privatize Du accumulator over (r, i_tile)

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_Du_id)

    (acc_name,) = (
        t_unit[kernel_name].id_to_insn[prcmpt_Du_id].read_dependency_names()
        - t_unit[kernel_name].all_inames()
    )

    inames_to_dup = (
        frozenset({prcmpt_r, prcmpt_itile}) & t_unit[kernel_name].all_inames()
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, inames_to_dup, only_var_names=frozenset({acc_name})
    )

    (acc_init_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_name in insn.write_dependency_names()
            and acc_name not in insn.read_dependency_names()
        )
    ]
    (acc_assign_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_name in insn.read_dependency_names()
            and acc_name not in insn.write_dependency_names()
        )
    ]

    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_dup,
        within=lp_match.Id(acc_init_id),
        tags={prcmpt_itile: "unr", prcmpt_r: "unr"},
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_dup,
        within=lp_match.Id(acc_assign_id),
        tags={prcmpt_itile: "unr", prcmpt_r: "unr"},
    )

    # }}}

    # {{{ Step 8: Realize outer reduction over r + privatize accumulators over i_tile

    assert len(matched_insn_ids) == noutputs, len(matched_insn_ids)
    t_unit = lp.realize_reduction(
        t_unit,
        insn_id_filter=matched_insn_ids,
    )
    acc_names = tuple(
        sorted(
            _fset_union(
                (
                    t_unit[kernel_name].id_to_insn[insn_id].read_dependency_names()
                    - t_unit[kernel_name].all_inames()
                )
                for insn_id in matched_insn_ids
            )
        )
    )
    assert len(acc_names) == noutputs
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, i_tile_iname, only_var_names=acc_names
    )

    acc_init_ids_tmp = []
    acc_assign_ids_tmp = []

    for acc_name in acc_names:
        (acc_init_id,) = [
            insn.id
            for insn in t_unit[kernel_name].instructions
            if (
                acc_name in insn.write_dependency_names()
                and acc_name not in insn.read_dependency_names()
            )
        ]
        (acc_assign_id,) = [
            insn.id
            for insn in t_unit[kernel_name].instructions
            if (
                acc_name in insn.read_dependency_names()
                and acc_name not in insn.write_dependency_names()
            )
        ]
        acc_init_ids_tmp.append(acc_init_id)
        acc_assign_ids_tmp.append(acc_assign_id)

    acc_init_ids = tuple(acc_init_ids_tmp)
    acc_assign_ids = tuple(acc_assign_ids_tmp)
    del acc_init_ids_tmp
    del acc_assign_ids_tmp

    t_unit = lp.duplicate_inames(
        t_unit,
        i_tile_iname,
        within=lp_match.Or(
            tuple(lp_match.Id(acc_init_id) for acc_init_id in acc_init_ids)
        ),
        tags={i_tile_iname: "unr"},
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        i_tile_iname,
        within=lp_match.Or(
            tuple(lp_match.Id(acc_assign_id) for acc_assign_id in acc_assign_ids)
        ),
        tags={i_tile_iname: "unr"},
    )
    t_unit = lp.tag_inames(t_unit, {r: "unr"})

    # }}}

    # {{{ Step 9. precompute Js

    for J in Js:
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            J,
            sweep_inames=[],
            precompute_outer_inames=frozenset((r, i_inner_iname, e_inner, e_outer)),
            temporary_address_space=lp.AddressSpace.PRIVATE,
            default_tag=None,
            within=within,
        )

    # }}}

    t_unit = lp.prioritize_loops(t_unit, (r, i_tile_iname))
    t_unit = lp.prioritize_loops(t_unit, (j_inner_iname, prcmpt_r))

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
            fp.write("\n")

# vim: fdm=marker
