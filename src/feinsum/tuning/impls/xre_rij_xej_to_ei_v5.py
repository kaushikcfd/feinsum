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
#         // Precompute Dprcmpt[:,:] <- D[r,
#         //                              i_tile*S_i:(i_tile+1)*S_i,
#         //                              j_tile*T_j:(j_tile+1)*T_j]
#         // Dprcmpt is a local variable.
#         for (int j = 0; j < T_j; j++) {
#           double Dprcmpt_prftch = Dprcmpt[r][lid(0)][j];
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
@fnsm.tuning.einsum_arg("ndim", lambda e: e.args[0][0].shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("s_e_log2", lambda e: IntParameter(1, 4))
@fnsm.tuning.transform_param("r_e", lambda e: IntParameter(1, 5))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
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
    d_local_elems = i_tile_len * j_tile_len
    ju_local_elems = ndim * S_e * r_e * j_tile_len
    if (d_local_elems + ju_local_elems) * 8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

    kernel_name = kernel_name or t_unit.default_entrypoint.name
    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.einsum(
        "xre,rij,xej->ei",
        fnsm.array("J", (ndim, ndim, "Nel")),
        fnsm.array("D", (ndim, ndof, ndof)),
        fnsm.array("u", (ndim, "Nel", ndof)),
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

    # names for Ju-prefetch inames (r, e, j); e is split further after precompute
    juprcmpt_r = vng("juprcmpt_r")
    juprcmpt_e = vng("juprcmpt_e")
    juprcmpt_e_re = vng("juprcmpt_e_re")
    juprcmpt_e_se = vng("juprcmpt_e_se")
    juprcmpt_j = vng("juprcmpt_j")
    ju_tmp_name = vng("_tmp_Ju")

    # names for J private precompute inames
    J_fetch_id = ing("J_fetch_id")
    J_prcmpt_tmp = vng("Jprcmpt_tmp")
    jprcmpt_x = vng("Jprcmpt_x")
    jprcmpt_r = vng("Jprcmpt_r")

    # names for D-prefetch inames
    iprftch_D = vng("iprftchD")
    jprftch_D = vng("jprftchD")
    D_fetch = vng(f"{D}_fetch")
    D_fetch_id = ing("D_fetch_id")

    # name for Ju substitution rule
    ju_subst_name = vng("_subst_Ju")
    ju_tmp_insn_id = vng("_tmp_Ju_id")

    # Instructions matching `within`
    (matched_insn_id,) = tuple(
        insn.id
        for insn in t_unit[kernel_name].instructions
        if within(t_unit[kernel_name], insn)
    )

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

    template = _get_reduction_expression_with_inames(
        next(
            insn.expression
            for insn in t_unit[kernel_name].instructions
            if lp_match.Writes(sigma["_fe_out"])(t_unit[kernel_name], insn)
        ),
        frozenset({x}),
    )
    t_unit = cast(
        "lp.TranslationUnit",
        lp.extract_subst(  # pyright: ignore[reportUnknownMemberType]
            t_unit,
            template=template,
            subst_name=ju_subst_name,
            parameters=(r, e, j),
            within=within,
        ),
    )

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
        inner_tag="unr",
    )

    # }}}

    # {{{ Step 3: Precompute Ju = sum_x J[x,r,e]*u[x,e,j] in LOCAL memory
    #
    # Sweep over e_inner (whole S_e*r_e range) so the single `e` parameter in
    # the subst rule maps cleanly.  juprcmpt_e is then split into
    # juprcmpt_e_re (size r_e) + juprcmpt_e_se (l.1, size S_e) below.

    # Split e_inner -> e_r_e + e_s_e (l.1) now that Ju precompute is done.
    # This also splits any access to ju_tmp_name indexed by e_inner.
    # The J-fetch instruction also picks up e_r_e at the outer level (outside
    # j_tile); it gets its own duplicate later to avoid the scheduling conflict.
    t_unit = lp.split_iname(
        t_unit,
        e_inner,
        r_e,
        outer_tag="l.1",
        inner_tag="unr",
        outer_iname=e_s_e,
        inner_iname=e_r_e,
    )

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        ju_subst_name,
        sweep_inames=[r, e_r_e, e_s_e, j_inner_iname],
        precompute_inames=[juprcmpt_r, juprcmpt_e, juprcmpt_j],
        precompute_outer_inames=frozenset({e_outer, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=ju_tmp_name,
        within=within,
        compute_insn_id=ju_tmp_insn_id,
        default_tag=None,
    )

    # Split juprcmpt_e (covers the same S_e*r_e range) to match.
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

    # }}}

    # {{{ Step 4: Precompute J[x,r,e] in PRIVATE memory (Jprcmpt[x][r])
    #
    # Must happen before extracting Ju_subst (while J is still directly in the
    # main instruction and `r` is still the original iname).  Using e (unsplit)
    # as the sole outer iname places the load outside both j_tile and all
    # reduction loops — i.e. once per element, as in the pseudocode comment.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        J,
        sweep_inames=[x, juprcmpt_r],
        precompute_inames=[jprcmpt_x, jprcmpt_r],
        precompute_outer_inames=frozenset({e_outer, juprcmpt_e_se, juprcmpt_e_re}),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        default_tag=None,
        within=lp_match.Id(ju_tmp_insn_id),
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

    # }}}

    # {{{ Step 5: Realize reduction on ju_tmp_id

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=ju_tmp_insn_id)

    (acc_x,) = (
        t_unit[kernel_name].id_to_insn[ju_tmp_insn_id].read_dependency_names()
        - t_unit[kernel_name].all_inames()
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, juprcmpt_r, only_var_names=frozenset({acc_x})
    )

    (acc_x_init_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_x in insn.write_dependency_names()
            and acc_x not in insn.read_dependency_names()
        )
    ]
    (acc_x_assign_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            acc_x in insn.read_dependency_names()
            and acc_x not in insn.write_dependency_names()
        )
    ]

    t_unit = lp.duplicate_inames(
        t_unit,
        juprcmpt_r,
        within=lp_match.Id(acc_x_init_id),
        tags={juprcmpt_r: t_unit[kernel_name].iname_tags(juprcmpt_r)},
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        juprcmpt_r,
        within=lp_match.Id(acc_x_assign_id),
        tags={juprcmpt_r: t_unit[kernel_name].iname_tags(juprcmpt_r)},
    )

    # }}}

    # {{{ Step 6: Precompute D in LOCAL memory
    #
    # Dprcmpt[r, i_inner, j_inner] per (r, e_outer, i_tile, j_tile).
    # Shape: [i_tile_len][j_tile_len]

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        sweep_inames=[i_inner_iname, j_inner_iname],
        precompute_inames=[None, iprftch_D, jprftch_D],
        precompute_outer_inames=frozenset({r, e_outer, i_tile_iname, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=D_fetch,
        compute_insn_id=D_fetch_id,
        default_tag=None,
        within=within,
    )
    t_unit = lp.tag_inames(t_unit, {iprftch_D: "l.0"})
    t_unit = lp.split_iname(t_unit, jprftch_D, S_e, inner_tag="l.1", outer_tag="unr")

    # }}}

    # {{{ Step 7: Realize redn over {r, j_inner} + privatize acc over i_tile,e_r_e

    t_unit = lp.realize_reduction(
        t_unit,
        insn_id_filter=matched_insn_id,
    )
    (out_acc_name,) = (
        t_unit[kernel_name].id_to_insn[matched_insn_id].read_dependency_names()
        - t_unit[kernel_name].all_inames()
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, i_tile_iname, only_var_names=frozenset({out_acc_name})
    )
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, e_r_e, only_var_names=frozenset({out_acc_name})
    )

    (out_acc_init_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            out_acc_name in insn.write_dependency_names()
            and out_acc_name not in insn.read_dependency_names()
        )
    ]
    (out_acc_assign_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if (
            out_acc_name in insn.read_dependency_names()
            and out_acc_name not in insn.write_dependency_names()
        )
    ]

    # e_r_e,i_tile appear in the main computation (inside j_tile).  Give the
    # J-fetch its own copy first so the original e_r_e ends up only inside
    # j_tile after the init/assign duplications below.
    t_unit = lp.duplicate_inames(
        t_unit,
        (e_r_e, i_tile_iname),
        within=lp_match.Id(out_acc_init_id),
        tags={
            i_tile_iname: t_unit[kernel_name].iname_tags(i_tile_iname),
            e_r_e: t_unit[kernel_name].iname_tags(e_r_e),
        },
    )
    e_r_e_assign, i_tile_iname_assign = vng("e_r_e"), vng("itile")
    t_unit = lp.duplicate_inames(
        t_unit,
        (e_r_e, i_tile_iname),
        within=lp_match.Id(out_acc_assign_id),
        new_inames=(e_r_e_assign, i_tile_iname_assign),
        tags={
            i_tile_iname: t_unit[kernel_name].iname_tags(i_tile_iname),
            e_r_e: t_unit[kernel_name].iname_tags(e_r_e),
        },
    )

    # Find the update instruction (reads both _D_fetch and _tmp_Ju) so that
    # the private-prefetch steps below target the right instruction.
    (main_update_insn_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if D_fetch in insn.read_dependency_names()
    ]

    # }}}

    # {{{ Step 8: Prefetch _tmp_Ju from LOCAL to PRIVATE
    #
    # _tmp_Ju is indexed as _tmp_Ju[r, e_re*S_e + e_se, j].  The non-trivial
    # e-subscript prevents add_prefetch from matching it, so we extract a
    # substitution rule first and then precompute it to PRIVATE memory.
    # Result: _tmp_Ju_priv[r][e_re][j] per work-item, filled once per j_tile.

    import pymbolic.primitives as prim

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
    ju_priv_e_r_e = vng("ju_priv_e_r_e")
    ju_priv_j = vng("ju_priv_j")
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
    t_unit = lp.tag_inames(
        t_unit, {ju_priv_e_r_e: "unr", ju_priv_j: "unr"}, ignore_nonexistent=True
    )

    t_unit = t_unit.with_kernel(
        lp.map_instructions(  # type: ignore[no-untyped-call]
            t_unit[kernel_name],
            lp_match.Id(ju_priv_fetch_id),
            lambda insn: (
                insn
                if insn.id != ju_priv_fetch_id
                else insn.copy(depends_on=frozenset({ju_tmp_insn_id}))
            ),
        )
    )

    # }}}

    # {{{ Step 9: Prefetch _D_fetch from LOCAL to PRIVATE
    #
    # _D_fetch[r, i_inner, j] -> _D_priv[r][j] per work-item (i_inner=lid(0)),
    # computed once per (i_tile, j_tile), hoisting it out of the e_re loop.

    d_priv_name = vng("_D_priv")
    t_unit = lp.add_prefetch(
        t_unit,
        D_fetch,
        sweep_inames=[],
        fetch_outer_inames=frozenset(
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
        within=lp_match.Id(main_update_insn_id),
    )

    # }}}

    t_unit = lp.add_inames_to_insn(t_unit, i_inner_iname, lp_match.Id(J_fetch_id))

    # x outer, r inner for both the J prefetch and the Ju update.
    t_unit = lp.prioritize_loops(t_unit, (jprcmpt_x, jprcmpt_r))
    t_unit = lp.prioritize_loops(t_unit, (x, juprcmpt_r))
    # In the main accumulation loop: r outer, j_inner middle, e_r_e inner.
    t_unit = lp.prioritize_loops(t_unit, (r, j_inner_iname))
    t_unit = lp.prioritize_loops(t_unit, (j_inner_iname, e_r_e))
    t_unit = lp.prioritize_loops(t_unit, (e_r_e_assign, i_tile_iname_assign))
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
        "{[x,r,e,i,j]: 0<=x,r<3 and 0<=i,j<20 and 0<=e<64}",
        """
        _J(_0, _1, _2) := J[_0, _1, _2]
        _D(_0, _1, _2) := D[_0, _1, _2]
        _u(_0, _1, _2) := u[_0, _1, _2]
        div_u[e, i] = sum([x,r,j], _J(x,r,e)*_D(r,i,j)*_u(x,e,j))
        """,
        [lp.GlobalArg("J,D,u", dtype="float64", shape=lp.auto), ...],
        lang_version=(2018, 2),
    )
    t_unit = transform(
        t_unit,
        ndim=3,
        ndof=20,
        s_e_log2=2,
        r_e=2,
        i_tiles=2,
        j_tiles=2,
    )
    print(lp.generate_code_v2(t_unit).device_code())  # type: ignore[no-untyped-call]

# vim: fdm=marker
