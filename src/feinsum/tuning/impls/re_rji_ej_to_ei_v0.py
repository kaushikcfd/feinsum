import math
from collections.abc import Iterable, Sequence
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


def _get_3d_arg(args: Sequence[fnsm.Array]) -> fnsm.Array:
    (arg,) = [arg for arg in args if arg.ndim == 3]
    return arg


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
#   double du_acc[R_e][i_tiles][3];
#
#   for (e_inner = 0; e_inner < R_e; e_inner++)
#     for (int i_tile = 0; i_tile < i_tiles; i_tile++)
#       for (int r =0; r < 3; r++)
#         du_acc[R_e][i_tile][r] = 0;
#
#   for (int j_tile = 0; j_tile < j_tiles; j_tile++) {
#     // Fetch uprcmpt[:,:] <- u[R_e * S_e *gid(0):R_e*S_e * gid(0)+R_e*S_e,
#                                j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#     // uprcmpt is a local variable
#     // Fetch uprcmpt_priv[:,:] <- uprcmpt[R_e*lid(1):R_e*lid(1)+R_e][0:j_tile_len]
#     for (int i_tile = 0; i_tile < i_tiles; i_tile++) {
#       for (int j = 0; j < j_tile_len; j++) {
#         for (int r = 0; r < 3; r++) {
#           double Dprcmpt = D[r, i_tile_len*i_tile+lid(0), j_tile_len*j_tile+j];
#           for (int e_inner = 0; e_inner < R_e; e_inner++)
#             du_acc[e_inner][i_tile][r] += Dprcmpt
#                                           * uprcmpt_priv[e_inner, j];
#         }
#       }
#     }
#   }
#
#   double du_dx_acc[R_e][i_tiles], du_dy_acc[R_e][i_tiles], du_dz_acc[R_e][i_tiles];
#
#   for (int e_inner  0; e_inner < R_e; e_inner++)
#     for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#       du_dx_acc[e_inner][i_tile]=0;
#       du_dy_acc[e_inner][i_tile]=0;
#       du_dz_acc[e_inner][i_tile]=0;
#     }
#
#   for (int e_inner  0; e_inner < R_e; e_inner++) {
#     int e = S_r*R_e*gid(0) + Re*lid(0) + e_inner;
#     for (int r = 0; r < 3; r++) {
#       double Jxr = Jx[r][e], Jyr = Jy[r][e], Jzr = Jz[r][e];
#       for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#         du_dx_acc[e_inner][i_tile] += Jxr * du_acc[e_inner][i_tile][r];
#         du_dy_acc[e_inner][i_tile] += Jyr * du_acc[e_inner][i_tile][r];
#         du_dz_acc[e_inner][i_tile] += Jzr * du_acc[e_inner][i_tile][r];
#       }
#     }
#   }
#
#   for (int e_inner  0; e_inner < R_e; e_inner++) {
#     int e = S_r*R_e*gid(0) + Re*lid(0) + e_inner;
#     for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#       du_dx[e][i_tile*i_tile_len + lid(0)] = du_dx_acc[e_inner][i_tile];
#       du_dy[e][i_tile*i_tile_len + lid(0)] = du_dy_acc[e_inner][i_tile];
#       du_dz[e][i_tile*i_tile_len + lid(0)] = du_dz_acc[e_inner][i_tile];
#     }
#   }
# }
@fnsm.tuning.einsum_arg("noutputs", lambda e: e.b)
@fnsm.tuning.einsum_arg("ndim", lambda e: _get_3d_arg(e.args[0]).shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("s_e_log2", lambda e: IntParameter(1, 5))
@fnsm.tuning.transform_param("r_e", lambda e: IntParameter(1, 8))
@fnsm.tuning.transform_param(
    "j_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
@fnsm.tuning.transform_param(
    "i_tiles", lambda e: IntParameter(1, math.ceil(e.shape[1] / 2))
)
def transform(
    t_unit: lp.TranslationUnit,
    noutputs: int,
    ndim: int,
    ndof: int,
    s_e_log2: int,
    r_e: int,
    i_tiles: int,
    j_tiles: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    s_e = 2**s_e_log2

    if s_e * r_e * math.ceil((ndof) / i_tiles) > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    if (ndof * s_e * r_e) * 8e-3 > 47:
        raise fnsm.InvalidParameterError("Shared memory limit exceeded")

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
        "re,rji,ej->ei",
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
        t_unit,
        ref_einsum,
        insn_match=within,
        kernel_name=kernel_name,
        long_dim_length=36,
    )
    i = sigma["i"]
    j = sigma["j"]
    u = sigma["u"]
    e = sigma["e"]
    D = sigma["D"]
    r = sigma["r"]
    Js = tuple(sigma[f"J{i}"] for i in range(noutputs))

    # e_inner: full inner range (s_e * r_e); e_s_e: l.1 (size s_e); e_r_e: unr
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
    e_s_e = vng(f"{e}_se")
    e_r_e = vng(f"{e}_re")

    i_tile_iname = vng(f"{i}_tile")
    i_inner_iname = vng(f"{i}_inner")
    i_tile_len = math.ceil(ndof / i_tiles)

    j_tile_iname = vng(f"{j}_tile")
    j_inner_iname = vng(f"{j}_inner")
    j_tile_len = math.ceil(ndof / j_tiles)

    # names for u LOCAL precompute inames
    uprcmpt_e = vng("uprcmpt_e")
    uprcmpt_e_se = vng("uprcmpt_e_se")
    uprcmpt_e_re = vng("uprcmpt_e_re")
    uprcmpt_j = vng("uprcmpt_j")
    u_local_name = vng("_tmp_u")

    # names for u PRIVATE prefetch (from LOCAL to PRIVATE)
    u_priv_name = vng("_tmp_u_priv")
    u_priv_fetch_id = ing("u_priv_fetch_id")

    # names for Du substitution and its private precompute
    du_subst_name = vng("_subst_Du")
    du_tmp_name = vng("_tmp_Du")
    prcmpt_r = vng("_prcmpt_r_Du")
    prcmpt_re = vng("_prcmpt_re_Du")  # sweep iname for e_r_e in Du precompute
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

    t_unit = lp.split_reduction_outward(t_unit, r, within=within)
    knl = t_unit[kernel_name]
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, j, within=within
    )
    t_unit = t_unit.with_kernel(knl)
    del knl

    # }}}

    # {{{ Step 2: Extract Du(r, e, i) = sum_j D[r,i,j]*u[e,j] as a substitution rule

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

    # {{{ Step 3: Split e (two stages), i, j inames
    #
    # Stage 1: e -> e_outer (g.0) + e_inner (size s_e * r_e, no tag yet)
    # Stage 2: e_inner -> e_s_e (l.1, size s_e) + e_r_e (unr, size r_e)
    # This gives: e = s_e*r_e*gid(0) + r_e*lid(1) + e_r_e

    t_unit = lp.split_iname(
        t_unit,
        e,
        s_e * r_e,
        outer_tag="g.0",
        inner_iname=e_inner,
        outer_iname=e_outer,
        within=within,
    )
    t_unit = lp.split_iname(
        t_unit,
        e_inner,
        r_e,
        outer_tag="l.1",
        inner_tag="unr",
        outer_iname=e_s_e,
        inner_iname=e_r_e,
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

    # {{{ Step 4: Precompute u in LOCAL memory
    #
    # u_local[uprcmpt_e, uprcmpt_j] covers [s_e*r_e, j_tile_len].
    # Loaded collaboratively per (e_outer, j_tile): uprcmpt_e_se maps to l.1,
    # uprcmpt_j inner maps to l.0.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        u,
        sweep_inames=[e_s_e, e_r_e, j_inner_iname],
        precompute_inames=[uprcmpt_e, uprcmpt_j],
        precompute_outer_inames=frozenset({e_outer, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.LOCAL,
        temporary_name=u_local_name,
        within=within,
        default_tag=None,
    )
    # uprcmpt_e covers [0, s_e*r_e); split into se (l.1) + re (inner, no tag).
    t_unit = lp.split_iname(
        t_unit,
        uprcmpt_e,
        r_e,
        outer_tag="l.1",
        inner_tag=None,
        outer_iname=uprcmpt_e_se,
        inner_iname=uprcmpt_e_re,
    )
    if i_tile_len == j_tile_len:
        t_unit = lp.tag_inames(t_unit, {uprcmpt_j: "l.0"})
    else:
        t_unit = lp.split_iname(t_unit, uprcmpt_j, i_tile_len, inner_tag="l.0")

    # }}}

    # (Step 4b — private u prefetch — happens after Step 7 once the Du
    # accumulation instruction exists and directly references _tmp_u.)
    t_unit = lp.add_prefetch(
        t_unit,
        u_local_name,
        sweep_inames=[e_r_e, j_inner_iname],
        fetch_outer_inames=frozenset({e_s_e, e_outer, i_inner_iname, j_tile_iname}),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_name=u_priv_name,
        default_tag="unr",
        prefetch_insn_id=u_priv_fetch_id,
    )

    # }}}

    # {{{ Step 6: Precompute Du = sum_{j_tile,j_inner} D*u in PRIVATE memory
    #
    # Sweep over both r and e_r_e so that e_r_e is represented by the fresh
    # iname prcmpt_re inside the j-reduction; this avoids any scheduling
    # conflict between the j_tile loop (D fetch) and the e_r_e loop (Du init/
    # assign) that would arise if e_r_e were an outer iname.
    #
    # du_tmp[prcmpt_r, prcmpt_re] is computed per (e_outer, e_s_e, i_tile,
    # i_inner), accumulating over (j_tile, j_inner).  After privatize over
    # i_tile this becomes du_tmp[r, e_r_e, i_tile], reproducing
    # du_acc[e_inner][i_tile][r] from kernel.c.

    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        du_subst_name,
        sweep_inames=[r, e_r_e],
        precompute_inames=[prcmpt_r, prcmpt_re],
        precompute_outer_inames=frozenset(
            {
                e_outer,
                e_s_e,
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
        lp_match.Id(prcmpt_Du_id),
        prcmpt_itile,
    )
    t_unit = lp.tag_inames(t_unit, {prcmpt_itile: "unr"})

    # }}}

    # {{{ Step 7: Realize Du reduction + privatize accumulator over (r,e_r_e,i_tile)

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_Du_id)

    (acc_name,) = (
        t_unit[kernel_name].id_to_insn[prcmpt_Du_id].read_dependency_names()
        - t_unit[kernel_name].all_inames()
    )

    inames_to_dup = (
        frozenset({prcmpt_r, prcmpt_re, prcmpt_itile})
        & t_unit[kernel_name].all_inames()
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
        tags={prcmpt_itile: "unr", prcmpt_r: "unr", prcmpt_re: "unr"},
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_dup,
        within=lp_match.Id(acc_assign_id),
        tags={prcmpt_itile: "unr", prcmpt_r: "unr", prcmpt_re: "unr"},
    )

    # }}}

    # {{{ Step 7b: Hoist D from LOCAL to PRIVATE scalar
    #
    # _D_fetch[prcmpt_r, i_inner, j_inner] does not depend on prcmpt_re, so
    # hoisting it to a private scalar outside the prcmpt_re loop saves r_e
    # redundant local-memory reads per (r, j).
    #
    # Note: after realize_reduction, prcmpt_Du_id is the *assign* instruction;
    # the actual update (acc += D_fetch * u_local) got a fresh id.

    (du_update_insn_id,) = [
        insn.id
        for insn in t_unit[kernel_name].instructions
        if u_priv_name in insn.read_dependency_names()
    ]
    D_priv_name = vng("_D_priv")
    t_unit = lp.precompute(  # type: ignore[no-untyped-call]
        t_unit,
        D,
        sweep_inames=[],
        precompute_outer_inames=frozenset(
            {
                e_outer,
                e_s_e,
                prcmpt_itile,
                i_inner_iname,
                j_tile_iname,
                j_inner_iname,
                prcmpt_r,
            }
        ),
        temporary_address_space=lp.AddressSpace.PRIVATE,
        temporary_name=D_priv_name,
        default_tag=None,
        within=lp_match.Id(du_update_insn_id),
    )

    # }}}

    # {{{ Step 8: Realize outer reduction over r + privatize accumulators over
    #            (i_tile, e_r_e)
    #
    # Duplicate both i_tile and e_r_e for the init and assign instructions so
    # they get fresh inames and don't conflict with the update loop structure.

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
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit, e_r_e, only_var_names=acc_names
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

    # Duplicate both i_tile and e_r_e for init and assign so they get fresh
    # inames independent from the update instruction (which uses the originals).
    e_r_e_tags = t_unit[kernel_name].iname_tags(e_r_e)
    e_r_e_init = vng(f"{e_r_e}")
    i_tile_init = vng(f"{i_tile_iname}")
    t_unit = lp.duplicate_inames(
        t_unit,
        (i_tile_iname, e_r_e),
        within=lp_match.Or(
            tuple(lp_match.Id(acc_init_id) for acc_init_id in acc_init_ids)
        ),
        new_inames=(i_tile_init, e_r_e_init),
        tags={i_tile_iname: "unr", e_r_e: e_r_e_tags},
    )
    e_r_e_assign = vng(f"{e_r_e}")
    i_tile_assign = vng(f"{i_tile_iname}")
    t_unit = lp.duplicate_inames(
        t_unit,
        (i_tile_iname, e_r_e),
        within=lp_match.Or(
            tuple(lp_match.Id(acc_assign_id) for acc_assign_id in acc_assign_ids)
        ),
        new_inames=(i_tile_assign, e_r_e_assign),
        tags={i_tile_iname: "unr", e_r_e: e_r_e_tags},
    )
    t_unit = lp.tag_inames(t_unit, {r: "unr"})

    # }}}

    # {{{ Step 9: Precompute Js (once per (r, e_r_e, e_s_e, e_outer))
    #
    # J[r, e] depends on both e_s_e and e_r_e, so both must appear in
    # precompute_outer_inames.

    for J in Js:
        t_unit = lp.precompute(  # type: ignore[no-untyped-call]
            t_unit,
            J,
            sweep_inames=[],
            precompute_outer_inames=frozenset(
                (r, i_inner_iname, e_s_e, e_r_e, e_outer)
            ),
            temporary_address_space=lp.AddressSpace.PRIVATE,
            default_tag=None,
            within=lp_match.And(
                (
                    lp_match.Iname(e_outer),
                    lp_match.Iname(e_s_e),
                    lp_match.Iname(e_r_e),
                    lp_match.Iname(r),
                )
            ),
        )

    # }}}

    t_unit = lp.prioritize_loops(t_unit, (r, i_tile_iname))

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
        "{[r,e,i,j]: 0<=r<3 and 0<=i,j<20 and 0<=e<1024}",
        """
        _Jx(_0, _1) := Jx[_0, _1]
        _Jy(_0, _1) := Jy[_0, _1]
        _Jz(_0, _1) := Jz[_0, _1]
        _D(_0, _1, _2) := D[_0, _1, _2]
        _u(_0, _1) := u[_0, _1]
        grad_x[e, i] = sum([r,j], _Jx(r,e)*_D(r,j,i)*_u(e,j))
        grad_y[e, i] = sum([r,j], _Jy(r,e)*_D(r,j,i)*_u(e,j))
        grad_z[e, i] = sum([r,j], _Jz(r,e)*_D(r,j,i)*_u(e,j))
        """,
        [lp.GlobalArg("Jx,Jy,Jz,D,u", dtype="float64", shape=lp.auto), ...],
        lang_version=(2018, 2),
    )
    t_unit = transform(
        t_unit,
        ndim=3,
        ndof=20,
        noutputs=3,
        i_tiles=2,
        j_tiles=2,
        s_e_log2=4,
        r_e=2,
    )
    print(lp.generate_code_v2(t_unit).device_code())  # type: ignore[no-untyped-call]

# vim: fdm=marker
