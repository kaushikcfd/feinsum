import math
from typing import Any, cast

import loopy as lp
import loopy.match as lp_match
from pymbolic.typing import Expression

import feinsum as fnsm
from feinsum import loopy_utils as lp_utils
from feinsum.tuning import IntParameter


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


# Essentially needs multiple 'e' to be handled per WI.
# Transformation to the following kernel:
# extern int gid(int); // opencl workgroup id
# extern int lid(int); // opencl workitem id
# # define T_j ceil(20 / j_tiles)
# # define S_i ceil(20 / i_tiles)
#
# // block size => (S_i, S_e, 1)
# void divergence(int N,
#                 const double J[3][3][N],
#                 const double u[3][N][20],
#                 const double D[3][20][20],
#                 double div_u[N][20]) {
#
#   int e = S_e * R_e * gid(0);
#   double div_u_acc[R_e][i_tiles];
#   double Jprcmpt[R_e][3][3];
#   double tmp_Ju[3][S_e * R_e][T_j]; // local var.
#   double Dprcmpt[3][S_i][T_j]; // local var.
#   double tmp_Ju_priv[R_e][T_j]
#
#   for (int e_inner = 0; e_inner < R_e; e_inner++)
#     for (int x=0; x < 3; x++)
#       for (int r=0; r < 3; r++)
#         Jprcmpt[e_inner][x][r] =  J[x][r][e + S_e * e_inner + lid(1)];
#
#   for (int e_inner = 0; e_inner < R_e; e_inner++)
#     for (int i_tile=0; i_tile < i_tiles; i_tile++)
#       div_u_acc[e_inner][i_tile] = 0;
#
#   for (int j_tile = 0; j_tile < j_tiles; j_tile++) {
#     for (int e_inner = 0; e_inner < R_e; e_inner++) {
#       for (int j_inner = 0;
#            j_inner < ceil(T_j / S_i);
#             j_inner++) {
#         int j = T_j * j_tile + j_inner * S_i + lid(0):
#         if (j < 20) {
#           double acc_r[3] = 0, u_prcmpt[3];
#           // Precompute u_prcmpt[0:3] <- u[:][e + S_e * e_inner + lid(1)][j];
#           // u_prcmpt is a private var.
#           for (int x = 0; x < 3; x++)
#             for (int r = 0; r < 3; r++)
#               acc_r[r] += u_prcmpt[x] * J_prcmpt[x][r][e_inner];
#           for (int r = 0; r < 3; r++)
#             tmp_Ju[r][R_e * lid(1) + e_inner][j] = acc_r[r];
#         }
#       }
#     }
#     for (int r = 0; r < 3; r++) {
#       for (int e_inner = 0; e_inner < R_e; e_inner++)
#         for (int j = 0; j < T_j; j++)
#           tmp_Ju_priv[e_inner][j] = tmp_Ju[r][R_e * lid(1) + e_inner][j];
#       for (int i_tile = 0; i_tile < i_tiles; i_tile++) {
#         for (int j = 0; j < T_j; j++) {
#           double Dprcmpt_prftch = D[r][T_j*j_tile + j][T_i*i_tile + lid(0)];
#           for (int e_inner = 0; e_inner < R_e; e_inner++)
#             div_u_acc[e_inner][i_tile] += (
#                   Dprcmpt_prftch
#                   * tmp_Ju_priv[e_inner][j]
#             );
#         }
#       }
#     }
#   }
#
#   for (int e_inner=0; e_inner < R_e; e_inner++)
#     for (int i_tile=0; i_tile < i_tiles; i_tile++)
#       div_u[e + R_e * lid(1) + e_inner][i_tile*i_tile_len + lid(0)]
#            = div_u_acc[e_inner][i_tile];
# }
@fnsm.tuning.einsum_arg("b", lambda e: e.b)
@fnsm.tuning.einsum_arg("ndim", lambda e: e.args[0][0].shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("s_e_log2", lambda e: IntParameter(1, 4))
@fnsm.tuning.transform_param("r_e", lambda e: IntParameter(1, 8))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    b: int,
    ndim: int,
    ndof: int,
    s_e_log2: int,
    r_e: int,
    i_tiles: int,
    j_tiles: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    S_e = 2**s_e_log2

    if S_e * math.ceil(ndof / i_tiles) > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    i_tile_len = math.ceil(ndof / i_tiles)
    j_tile_len = math.ceil(ndof / j_tiles)

    # D shape in local memory: [ndim][i_tile_len][j_tile_len]
    # Ju shape: [ndim][n_e_per_wg][j_tile_len]
    if (b * ndim * S_e * r_e * j_tile_len) * 8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.batched_einsum(
        "xre,rji,xej->ei",
        [
            [
                fnsm.array("J", (ndim, ndim, "Nel")),
                fnsm.array("D", (ndim, ndof, ndof)),
                fnsm.array(f"u_{i}", (ndim, "Nel", ndof)),
            ]
            for i in range(b)
        ],
    )

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_instruction_id_generator()
    sigma = fnsm.match_t_unit_to_einsum(
        t_unit,
        ref_einsum,
        insn_match=within,
        kernel_name=kernel_name,
        long_dim_length=36,
    )
    i = sigma["i"]
    j = sigma["j"]
    e = sigma["e"]
    D = sigma["D"]
    r = sigma["r"]
    J = sigma["J"]
    x = sigma["x"]
    # e_inner: intermediate iname after first e-split (size S_e * r_e)
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
    # e_r_e: R_e elements per WI (unrolled); e_s_e: S_e WIs in e direction (l.1)
    e_r_e = vng(f"{e}_re")
    e_s_e = vng(f"{e}_se")

    i_tile_iname = vng(f"{i}_tile")
    i_inner_iname = vng(f"{i}_inner")

    j_tile_iname = vng(f"{j}_tile")
    j_inner_iname = vng(f"{j}_inner")

    # Instructions matching `within` — one per batched output
    matched_insn_ids = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if within(t_unit[kernel_name], insn)
    ]
    within = lp_match.Or(tuple(lp_match.Id(insn_id) for insn_id in matched_insn_ids))

    # }}}

    # {{{ Step 0: Split x inward and hoist D[r,i,j] (invariant in x) out
    #
    # Transforms: sum_{x,r,j} Jprcmpt[x][r]*D[r,i,j]*u[x,e,j]
    #          -> sum_{r,j} D[r,i,j] * (sum_x Jprcmpt[x][r]*u[x,e,j])

    t_unit = lp.split_reduction_inward(t_unit, x, within=within)
    knl = t_unit[kernel_name]
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, x, within=within
    )
    t_unit = t_unit.with_kernel(knl)
    del knl

    # }}}

    # {{{ Step 1: Extract Ju(r, e, j) = sum_x Jprcmpt[x][r]*u[x,e,j] as a subst rule
    #            (per output, since each uses a different u_i)

    ju_subst_names = []
    for iout in range(b):
        fe_out_name = sigma[fe_out(iout)]
        template = _get_reduction_expression_with_inames(
            next(
                insn.expression
                for insn in t_unit[kernel_name].instructions
                if lp_match.Writes(fe_out_name)(t_unit[kernel_name], insn)
            ),
            frozenset({x}),
        )
        ju_subst_name = vng("_subst_Ju")
        t_unit = cast(
            "lp.TranslationUnit",
            lp.extract_subst(  # pyright: ignore[reportUnknownMemberType]
                t_unit,
                template=template,
                subst_name=ju_subst_name,
                parameters=(r, e, j),
                within=lp_match.Writes(fe_out_name),
            ),
        )
        ju_subst_names.append(ju_subst_name)

    # }}}

    # {{{ Step 2: Split e (first stage only), i, j inames
    #
    # e -> e_outer (g.0) + e_inner (size S_e * r_e)
    # The second e split (e_inner -> e_r_e/e_s_e) happens AFTER Ju precompute so
    # the subst rule's single `e` parameter maps cleanly to e_inner.

    t_unit = lp.split_iname(
        t_unit,
        e,
        S_e * r_e,
        outer_tag="g.0",
        inner_iname=e_inner,
        outer_iname=e_outer,
        # slabs=(0, 1),
    )

    t_unit = lp.split_iname(
        t_unit,
        i,
        i_tile_len,
        outer_iname=i_tile_iname,
        inner_iname=i_inner_iname,
        inner_tag="l.0",
        outer_tag="unr",
    )

    t_unit = lp.split_iname(
        t_unit,
        j,
        j_tile_len,
        outer_iname=j_tile_iname,
        inner_iname=j_inner_iname,
        inner_tag="unr",
    )

    # }}}

    # {{{ Steps 3-5: Per-output Ju LOCAL precompute, J PRIVATE precompute, realize x

    # Split e_inner -> e_r_e + e_s_e (l.1); shared across all outputs.
    t_unit = lp.split_iname(
        t_unit,
        e_inner,
        r_e,
        outer_tag="l.1",
        inner_tag="unr",
        outer_iname=e_s_e,
        inner_iname=e_r_e,
    )

    import pymbolic.primitives as prim

    # Shared Ju precompute inames — all b outputs share the same loop.
    juprcmpt_r = vng("juprcmpt_r")
    juprcmpt_e = vng("juprcmpt_e")
    juprcmpt_e_re = vng("juprcmpt_e_re")
    juprcmpt_e_se = vng("juprcmpt_e_se")
    juprcmpt_j = vng("juprcmpt_j")
    J_fetch_id = ing("J_fetch_id")
    J_prcmpt_tmp = vng("Jprcmpt_tmp")
    jprcmpt_x = vng("Jprcmpt_x")
    jprcmpt_r = vng("Jprcmpt_r")

    # Step 3a: Precompute each Ju to LOCAL, all inside the same shared loop.
    # (Multiple precompute calls with the same precompute_inames is valid in
    # loopy — each creates its own temporary but shares the sweep loop.)
    ju_tmp_names: list[str] = []
    ju_tmp_insn_ids: list[str] = []
    for iout in range(b):
        ju_tmp_name = vng("_tmp_Ju")
        ju_tmp_insn_id = vng("_tmp_Ju_id")
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            ju_subst_names[iout],
            sweep_inames=[r, e_r_e, e_s_e, j_inner_iname],
            precompute_inames=[juprcmpt_r, juprcmpt_e, juprcmpt_j],
            precompute_outer_inames=frozenset({e_outer, j_tile_iname}),
            temporary_address_space=lp.AddressSpace.LOCAL,
            temporary_name=ju_tmp_name,
            within=within,
            compute_insn_id=ju_tmp_insn_id,
            default_tag=None,
        )
        ju_tmp_names.append(ju_tmp_name)
        ju_tmp_insn_ids.append(ju_tmp_insn_id)

    # Step 3b: Split juprcmpt_e ONCE (shared by all outputs).
    t_unit = lp.split_iname(
        t_unit,
        juprcmpt_e,
        r_e,
        outer_tag="l.1",
        inner_tag="unr",
        inner_iname=juprcmpt_e_re,
        outer_iname=juprcmpt_e_se,
    )

    t_unit = lp.tag_inames(t_unit, {juprcmpt_e_re: "unr", juprcmpt_r: "unr"})
    if j_tile_len == i_tile_len:
        t_unit = lp.tag_inames(t_unit, {juprcmpt_j: "l.0"})
    else:
        t_unit = lp.split_iname(
            t_unit, juprcmpt_j, i_tile_len, inner_tag="l.0", outer_tag="unr"
        )

    # Step 4: Precompute J[x,r,e] in PRIVATE memory (shared across all outputs).
    within_all_ju = lp_match.Or(tuple(lp_match.Id(id_) for id_ in ju_tmp_insn_ids))
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        J,
        sweep_inames=[x, juprcmpt_r],
        precompute_inames=[jprcmpt_x, jprcmpt_r],
        precompute_outer_inames=frozenset({e_outer, juprcmpt_e_se, juprcmpt_e_re}),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        default_tag=None,
        within=within_all_ju,
        compute_insn_id=J_fetch_id,
        temporary_name=J_prcmpt_tmp,
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit,
        juprcmpt_e_re,
        J_prcmpt_tmp,
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        juprcmpt_e_re,
        within=lp_match.Id(J_fetch_id),
        tags={juprcmpt_e_re: t_unit[kernel_name].iname_tags(juprcmpt_e_re)},
    )
    t_unit = lp.tag_inames(t_unit, {jprcmpt_x: "unr", jprcmpt_r: "unr"})

    # Step 5: Realize x-reduction per output, then share init/assign loops across
    # all outputs (one duplicate_inames call per role instead of one per output).
    acc_x_names: list[str] = []
    for iout in range(b):
        ju_tmp_insn_id = ju_tmp_insn_ids[iout]
        t_unit = lp.realize_reduction(t_unit, insn_id_filter=ju_tmp_insn_id)
        (acc_x,) = (
            t_unit[kernel_name].id_to_insn[ju_tmp_insn_id].read_dependency_names()
            - t_unit[kernel_name].all_inames()
        )
        acc_x_names.append(acc_x)
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, juprcmpt_r, only_var_names=frozenset({acc_x})
        )

    acc_x_init_ids = [
        next(
            insn.id
            for insn in t_unit[kernel_name].instructions
            if (
                acc_x in insn.write_dependency_names()
                and acc_x not in insn.read_dependency_names()
            )
        )
        for acc_x in acc_x_names
    ]
    acc_x_assign_ids = [
        next(
            insn.id
            for insn in t_unit[kernel_name].instructions
            if (
                acc_x in insn.read_dependency_names()
                and acc_x not in insn.write_dependency_names()
            )
        )
        for acc_x in acc_x_names
    ]
    juprcmpt_r_tags = {juprcmpt_r: t_unit[kernel_name].iname_tags(juprcmpt_r)}
    t_unit = lp.duplicate_inames(
        t_unit,
        juprcmpt_r,
        within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in acc_x_init_ids)),
        tags=juprcmpt_r_tags,
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        juprcmpt_r,
        within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in acc_x_assign_ids)),
        tags=juprcmpt_r_tags,
    )

    # }}}

    # {{{ Step 6: Per-output: realize redn over {r, j_inner} + privatize acc

    out_acc_names: list[str] = []
    main_update_insn_ids: list[str] = []

    for iout in range(b):
        matched_insn_id = matched_insn_ids[iout]
        t_unit = lp.realize_reduction(
            t_unit,
            insn_id_filter=matched_insn_id,
        )
        (out_acc_name,) = (
            t_unit[kernel_name].id_to_insn[matched_insn_id].read_dependency_names()
            - t_unit[kernel_name].all_inames()
        )
        out_acc_names.append(out_acc_name)
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, i_tile_iname, only_var_names=frozenset({out_acc_name})
        )
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, e_r_e, only_var_names=frozenset({out_acc_name})
        )
        (main_update_insn_id,) = [
            insn.id
            for insn in t_unit[kernel_name].instructions
            if ju_tmp_names[iout] in insn.read_dependency_names()
        ]
        main_update_insn_ids.append(main_update_insn_id)

    # Collect all init/assign IDs then duplicate their loops once, shared across
    # outputs.
    out_acc_init_ids = [
        next(
            insn.id
            for insn in t_unit[kernel_name].instructions
            if (
                out_acc_name in insn.write_dependency_names()
                and out_acc_name not in insn.read_dependency_names()
            )
        )
        for out_acc_name in out_acc_names
    ]
    out_acc_assign_ids = [
        next(
            insn.id
            for insn in t_unit[kernel_name].instructions
            if (
                out_acc_name in insn.read_dependency_names()
                and out_acc_name not in insn.write_dependency_names()
            )
        )
        for out_acc_name in out_acc_names
    ]
    out_acc_tags = {
        i_tile_iname: t_unit[kernel_name].iname_tags(i_tile_iname),
        e_r_e: t_unit[kernel_name].iname_tags(e_r_e),
    }
    t_unit = lp.duplicate_inames(
        t_unit,
        (e_r_e, i_tile_iname),
        within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in out_acc_init_ids)),
        tags=out_acc_tags,
    )
    e_r_e_assign = vng("e_r_e")
    i_tile_iname_assign = vng("itile")
    t_unit = lp.duplicate_inames(
        t_unit,
        (e_r_e, i_tile_iname),
        within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in out_acc_assign_ids)),
        new_inames=(e_r_e_assign, i_tile_iname_assign),
        tags=out_acc_tags,
    )

    # }}}

    # {{{ Step 7: Per-output: prefetch _tmp_Ju from LOCAL to PRIVATE
    #
    # _tmp_Ju is indexed as _tmp_Ju[r, e_re*S_e + e_se, j].  The non-trivial
    # e-subscript prevents add_prefetch from matching it, so we extract a
    # substitution rule first and then precompute it to PRIVATE memory.
    # Result: _tmp_Ju_priv[r][e_re][j] per work-item, filled once per j_tile.

    # Shared precompute inames — all outputs' Ju→private fetches share the same loop.
    ju_priv_e_r_e = vng("ju_priv_e_r_e")
    ju_priv_j = vng("ju_priv_j")

    for iout in range(b):
        main_update_insn_id = main_update_insn_ids[iout]
        ju_tmp_name = ju_tmp_names[iout]
        ju_tmp_insn_id = ju_tmp_insn_ids[iout]

        ju_priv_subst = vng("_subst_Ju_priv")
        t_unit = cast(
            "lp.TranslationUnit",
            lp.extract_subst(  # pyright: ignore[reportUnknownMemberType]
                t_unit,
                subst_name=ju_priv_subst,
                template=prim.Variable(ju_tmp_name)[
                    prim.Variable(r),
                    (
                        prim.Variable(e_s_e) * r_e + prim.Variable(e_r_e)
                        if r_e != 1
                        else prim.Variable(e_s_e) + prim.Variable(e_r_e)
                    ),
                    prim.Variable(j_inner_iname),
                ],
                parameters=(e_r_e, j_inner_iname),
                within=lp_match.Id(main_update_insn_id),
            ),
        )
        ju_priv_name = vng("_tmp_Ju_priv")
        ju_priv_fetch_id = ing("ju_prftch_id")
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            ju_priv_subst,
            sweep_inames=[e_r_e, j_inner_iname],
            precompute_inames=[ju_priv_e_r_e, ju_priv_j],
            precompute_outer_inames=frozenset(
                {e_outer, j_tile_iname, e_s_e, i_inner_iname, r}
            ),
            temporary_address_space=lp.AddressSpace.PRIVATE,
            temporary_name=ju_priv_name,
            default_tag=None,
            compute_insn_id=ju_priv_fetch_id,
            within=lp_match.Id(main_update_insn_id),
        )

        t_unit = t_unit.with_kernel(
            lp.map_instructions(  # type: ignore[no-untyped-call]
                t_unit[kernel_name],
                lp_match.Id(ju_priv_fetch_id),
                lambda insn, fid=ju_priv_fetch_id, tid=ju_tmp_insn_id: (
                    insn
                    if insn.id != fid
                    else insn.copy(depends_on=frozenset({tid}))
                ),
            )
        )

    t_unit = lp.tag_inames(
        t_unit, {ju_priv_e_r_e: "unr", ju_priv_j: "unr"}, ignore_nonexistent=True
    )

    # }}}

    # {{{ Step 9: Prefetch D from global to PRIVATE (shared across all outputs)
    #
    # D[r, j_inner, i_inner] -> scalar _D_priv per work-item,
    # computed once per (i_tile, j_tile), hoisting it out of the e_re loop.
    # D is the same array for all batched outputs, so one prefetch suffices.

    d_priv_name = vng("_D_priv")
    within_all_updates = lp_match.Or(
        tuple(lp_match.Id(mid) for mid in main_update_insn_ids)
    )
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        sweep_inames=[],
        precompute_outer_inames=frozenset(
            {
                e_outer,
                j_tile_iname,
                i_tile_iname,
                i_inner_iname,
                e_s_e,
                r,
                j_inner_iname,
            }
        ),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_name=d_priv_name,
        default_tag=None,
        within=within_all_updates,
    )

    # }}}

    t_unit = lp.add_inames_to_insn(t_unit, i_inner_iname, lp_match.Id(J_fetch_id))

    # x outer, r inner for both the J prefetch and the Ju update.
    t_unit = lp.prioritize_loops(t_unit, (jprcmpt_x, jprcmpt_r))
    t_unit = lp.prioritize_loops(t_unit, (x, juprcmpt_r))
    t_unit = lp.prioritize_loops(t_unit, (e_r_e_assign, i_tile_iname_assign))
    # In the main accumulation loop: r outer, j_inner middle, e_r_e inner.
    t_unit = lp.prioritize_loops(t_unit, (r, j_inner_iname))
    t_unit = lp.prioritize_loops(t_unit, (j_inner_iname, e_r_e))
    t_unit = lp.tag_inames(t_unit, {x: "unr", r: "unr"})

    if 0:
        # enable for debugging.
        from loopy.kernel.data import UnrollTag

        inames_to_unr = frozenset(
            iname
            for iname in t_unit[kernel_name].all_inames()
            if t_unit[kernel_name].iname_tags_of_type(iname, UnrollTag)
        )
        for iname in sorted(inames_to_unr):
            t_unit = lp.untag_inames(t_unit, iname, UnrollTag)

    return t_unit


if __name__ == "__main__":
    t_unit = lp.make_kernel(
        "{[x,r,e,i,j]: 0<=x,r<3 and 0<=i,j<20 and 0<=e<1024}",
        """
        _J(_0, _1, _2) := J[_0, _1, _2]
        _D(_0, _1, _2) := D[_0, _1, _2]
        _u(_0, _1, _2) := u[_0, _1, _2]
        _v(_0, _1, _2) := v[_0, _1, _2]
        div_u[e, i] = sum([x,r,j], _J(x,r,e)*_D(r,j,i)*_u(x,e,j))
        div_v[e, i] = sum([x,r,j], _J(x,r,e)*_D(r,j,i)*_v(x,e,j))
        """,
        [lp.GlobalArg("J,D,u,v", dtype="float64", shape=lp.auto), ...],
        lang_version=(2018, 2),
    )
    t_unit = transform(
        t_unit,
        b=2,
        ndim=3,
        ndof=20,
        s_e_log2=2,
        r_e=2,
        i_tiles=2,
        j_tiles=2,
    )
    print(lp.generate_code_v2(t_unit).device_code())  # type: ignore[no-untyped-call]

# vim: fdm=marker
