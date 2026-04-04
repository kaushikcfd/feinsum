import math
from collections.abc import Iterable
from typing import Any, cast

import loopy as lp
import loopy.match as lp_match
from pymbolic.typing import Expression

import feinsum as fnsm
from feinsum import loopy_utils as lp_utils
from feinsum.tuning import BoolParameter, IntParameter


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
# extern int gid(int); // opencl workgroup id
# extern int lid(int); // opencl workitem id
#
# void divergence(int N,
#                 const double J[3][3][N],
#                 const double u[3][N][20],
#                 const double D[3][20][20],
#                 double div_u[N][20]) {
#
#   int e = ne_per_wg * gid(0) + lid(1);
#   double div_u_acc[i_tiles];
#   double Jprcmpt[3][3];
#   double tmp_Ju[3][ne_per_wg][j_tile_len]; // local var.
#   double Dprcmpt[i_tile_len][j_tile_len]; // local var.
#
#   for  (int x=0; x < 3; x++)
#     for  (int r=0; r < 3; r++)
#       Jprcmpt[x][r] =  J[x][r][e];
#
#   for (int i_tile=0; i_tile < i_tiles; i_tile++)
#     div_u_acc[i_tile] = 0;
#
#   for (int j_tile = 0; j_tile < j_tiles; j_tile++) {
#     double acc_r[3] = 0;
#     for (int x = 0; x < 3; x++)
#       for (int r = 0; r < 3; r++)
#         acc_r[r] += u[x][e][j_tile*j_tile_len+lid(0)];
#     for (int r = 0; r < 3; r++)
#       tmp_Ju[r][lid(1)][lid(0)] = acc_r[r];
#     for (int i_tile = 0; i_tile < i_tiles; i_tile++) {
#       for (int r = 0; r < 3; r++) {
#         // Precompute Dprcmpt[:,:] <- D[3,
#         //                              i_tile*i_tile_len:(i_tile+1)*i_tile_len,
#         //                              j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#         // Dprcmpt is a local variable.
#         // If `prcmpt_slices_of_D` is False, move this out "r" loop ->
#         // increase size of Dprcmpt.
#
#         for (int j = 0; j < j_tile_len; j++)
#           div_u_acc[i_tile] += (
#                 Dprcmpt[i_tile*i_tile_len + lid(0)][j_tile*j_tile_len+j]
#                 * uprcmpt[r][lid(1)][j]
#           );
#       }
#     }
#   }
#
#   for (int i_tile=0; i_tile < i_tiles; i_tile++)
#     div_u[e][i_tile*i_tile_len + lid(0)] = div_u_acc[i_tile];
# }


@fnsm.tuning.einsum_arg("ndim", lambda e: e.args[0][0].shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("n_e_per_wg_log2", lambda e: IntParameter(1, 5))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "precompute_slices_of_D", lambda e: BoolParameter()
)
def transform(
    t_unit: lp.TranslationUnit,
    ndim: int,
    ndof: int,
    n_e_per_wg_log2: int,
    i_tiles: int,
    j_tiles: int,
    precompute_slices_of_D: bool,  # noqa: N803
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    n_e_per_wg = 2**n_e_per_wg_log2

    if n_e_per_wg * math.ceil(ndof / i_tiles) > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    i_tile_len = math.ceil(ndof / i_tiles)
    j_tile_len = math.ceil(ndof / j_tiles)

    # D shape in local memory: [i_tile_len][j_tile_len] per r (if slices) else
    # [ndim][i_tile_len][j_tile_len]; Ju shape: [ndim][n_e_per_wg][j_tile_len]
    d_local_elems = (
        i_tile_len * j_tile_len
        if precompute_slices_of_D
        else ndim * i_tile_len * j_tile_len
    )
    ju_local_elems = ndim * n_e_per_wg * j_tile_len
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
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")

    i_tile_iname = vng(f"{i}_tile")
    i_inner_iname = vng(f"{i}_inner")

    j_tile_iname = vng(f"{j}_tile")
    j_inner_iname = vng(f"{j}_inner")

    # names for Ju-prefetch inames (r, e, j)
    juprcmpt_r = vng("juprcmpt_r")
    juprcmpt_e = vng("juprcmpt_e")
    juprcmpt_j = vng("juprcmpt_j")
    ju_tmp_name = vng("_tmp_Ju")

    # names for J private precompute inames
    J_fetch_id = ing("J_fetch_id")
    jprcmpt_x = vng("Jprcmpt_x")
    jprcmpt_r = vng("Jprcmpt_r")

    # names for D-prefetch inames
    rprftch_D = vng("rprftchD")
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

    # {{{ Step 0: Precompute J[x,r,e] in PRIVATE memory (Jprcmpt[x][r])
    #
    # Must happen before extracting Ju_subst (while J is still directly in the
    # main instruction and `r` is still the original iname).  Using e (unsplit)
    # as the sole outer iname places the load outside both j_tile and all
    # reduction loops — i.e. once per element, as in the pseudocode comment.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        J,
        sweep_inames=[x, r],
        precompute_inames=[jprcmpt_x, jprcmpt_r],
        precompute_outer_inames=frozenset({e}),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        default_tag="unr",
        within=within,
        compute_insn_id=J_fetch_id,
    )

    # }}}

    # {{{ Step 1: Split x inward and hoist D[r,i,j] (invariant in x) out
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

    # {{{ Step 2: Extract Ju(r, e, j) = sum_x Jprcmpt[x][r]*u[x,e,j] as a subst rule

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
        slabs=(0, 1)
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

    # {{{ Step 4: Precompute Ju = sum_x J[x,r,e]*u[x,e,j] in LOCAL memory
    #
    # tmp_Ju[r, e_inner, j_inner] loaded collaboratively per (e_outer, j_tile).
    # Shape: [ndim, n_e_per_wg, j_tile_len]

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        ju_subst_name,
        sweep_inames=[r, e_inner, j_inner_iname],
        precompute_inames=[juprcmpt_r, juprcmpt_e, juprcmpt_j],
        precompute_outer_inames=frozenset({e_outer, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=ju_tmp_name,
        within=within,
        compute_insn_id=ju_tmp_insn_id,
        default_tag=None,
    )
    t_unit = lp.tag_inames(t_unit, {juprcmpt_r: "unr", juprcmpt_e: "l.1"})
    if j_tile_len == i_tile_len:
        t_unit = lp.tag_inames(t_unit, {juprcmpt_j: "l.0"})
    else:
        t_unit = lp.split_iname(t_unit, juprcmpt_j, i_tile_len, inner_tag="l.0")

    # }}}

    # {{{ Step 5: relalize reduction on ju_tmp_id

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
        tags={juprcmpt_r: "unr"},
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        juprcmpt_r,
        within=lp_match.Id(acc_x_assign_id),
        tags={juprcmpt_r: "unr"},
    )

    # }}}

    # {{{ Step 6: Precompute D in LOCAL memory
    #
    # Dprcmpt[i_inner, j_inner] per (r, e_outer, i_tile, j_tile) when slicing,
    # or Dprcmpt[r, i_inner, j_inner] per (e_outer, i_tile, j_tile) otherwise.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        sweep_inames=[i_inner_iname, j_inner_iname]
        if precompute_slices_of_D
        else [r, i_inner_iname, j_inner_iname],
        precompute_inames=[None, iprftch_D, jprftch_D]
        if precompute_slices_of_D
        else [rprftch_D, iprftch_D, jprftch_D],
        precompute_outer_inames=frozenset({r, e_outer, i_tile_iname, j_tile_iname})
        if precompute_slices_of_D
        else frozenset({e_outer, i_tile_iname, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=D_fetch,
        compute_insn_id=D_fetch_id,
        default_tag=None,
        within=within,
    )
    t_unit = lp.tag_inames(t_unit, {iprftch_D: "l.0"})
    if not precompute_slices_of_D:
        t_unit = lp.tag_inames(t_unit, {rprftch_D: "unr"})
    t_unit = lp.split_iname(t_unit, jprftch_D, n_e_per_wg, inner_tag="l.1")

    # }}}

    # {{{ Step 7: Realize reduction over {r, j_inner} + privatize acc over i_tile

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

    t_unit = lp.duplicate_inames(
        t_unit,
        i_tile_iname,
        within=lp_match.Id(out_acc_init_id),
        tags={i_tile_iname: "unr"}
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        i_tile_iname,
        within=lp_match.Id(out_acc_assign_id),
        tags={i_tile_iname: "unr"}
    )

    # }}}

    t_unit = lp.add_inames_to_insn(t_unit, i_inner_iname, lp_match.Id(J_fetch_id))

    # x outer, r inner for both the J prefetch and the Ju update, matching the
    # pseudocode loop order (u[x][e][j] is fetched once per x across all r).
    t_unit = lp.prioritize_loops(t_unit, (jprcmpt_x, jprcmpt_r))
    t_unit = lp.prioritize_loops(t_unit, (x, juprcmpt_r))

    t_unit = lp.tag_inames(t_unit, {r: "unr", x: "unr"})

    return t_unit


if __name__ == "__main__":
    import numpy as np

    t_unit = lp.make_kernel(
        "{[x,r,e,i,j]: 0<=x,r<3 and 0<=i,j<20 and 0<=e<64}",
        """
        _J(_0, _1, _2) := J[_0, _1, _2]
        _D(_0, _1, _2) := D[_0, _1, _2]
        _u(_0, _1, _2) := u[_0, _1, _2]
        div_u[e, i] = sum([x,r,j], _J(x,r,e)*_D(r,i,j)*_u(x,e,j))
        """,
        [lp.GlobalArg("J,D,u", dtype=np.float64, shape=lp.auto), ...],
        lang_version=(2018, 2),
    )
    t_unit = transform(
        t_unit, ndim=3, ndof=20, n_e_per_wg_log2=2, i_tiles=2, j_tiles=2,
        precompute_slices_of_D=False,
    )

# vim: fdm=marker
