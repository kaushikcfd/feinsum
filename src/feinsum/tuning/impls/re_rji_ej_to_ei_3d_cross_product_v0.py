import itertools
import math
from collections.abc import Collection, Iterable, Sequence
from typing import Any, cast

import islpy as isl
import loopy as lp
import loopy.match as lp_match
from constantdict import constantdict
from pymbolic.typing import Expression

import feinsum as fnsm
from feinsum import loopy_utils as lp_utils
from feinsum.tuning import IntParameter


def _fset_union(fsets: Iterable[frozenset[str]]) -> frozenset[str]:
    result: frozenset[str] = frozenset()
    for fs in fsets:
        result |= fs
    return result


def _writes_any(outputs: Iterable[str]) -> lp_match.MatchExpressionBase:
    return lp_match.Or(tuple(lp_match.Writes(o) for o in outputs))


def _find_acc_init_assign_ids(knl: lp.LoopKernel, acc_name: str) -> tuple[str, str]:
    """Return (init_id, assign_id) for a realized-reduction accumulator."""
    (init_id,) = [
        insn.id
        for insn in knl.instructions
        if acc_name in insn.write_dependency_names()
        and acc_name not in insn.read_dependency_names()
    ]
    (assign_id,) = [
        insn.id
        for insn in knl.instructions
        if acc_name in insn.read_dependency_names()
        and acc_name not in insn.write_dependency_names()
    ]
    return init_id, assign_id


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


# Field components and which two J multipliers produce their outputs.
# e.g. ux produces d(ux)/dy (uses Jy) and d(ux)/dz (uses Jz).
_U_NAMES = ("ux", "uy", "uz", "vx", "vy", "vz")
_U_TO_J_PAIR: dict[str, tuple[str, str]] = {
    "ux": ("Jy", "Jz"),
    "uy": ("Jx", "Jz"),
    "uz": ("Jx", "Jy"),
    "vx": ("Jy", "Jz"),
    "vy": ("Jx", "Jz"),
    "vz": ("Jx", "Jy"),
}


# Transformation to the following kernel:
# void grad_components(double* Jx, double* Jy, double* Jz, double* u, double* D,
#                      double* du_dx, double* du_dy, double* du_dz) {
#   double dux_acc[R_e][i_tiles][3];
#   double duy_acc[R_e][i_tiles][3];
#   double duz_acc[R_e][i_tiles][3];
#
#   for (e_inner = 0; e_inner < R_e; e_inner++)
#     for (int i_tile = 0; i_tile < i_tiles; i_tile++)
#       for (int r =0; r < 3; r++) {
#         dux_acc[R_e][i_tile][r] = 0;
#         duy_acc[R_e][i_tile][r] = 0;
#         duy_acc[R_e][i_tile][r] = 0;
#       }
#
#   for (int j_tile = 0; j_tile < j_tiles; j_tile++) {
#     // Fetch uxprcmpt[:,:] <- ux[R_e * S_e *gid(0):R_e*S_e * gid(0)+R_e*S_e,
#                                  j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#     // Fetch uyprcmpt[:,:] <- uy[R_e * S_e *gid(0):R_e*S_e * gid(0)+R_e*S_e,
#                                  j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#     // Fetch uzprcmpt[:,:] <- uz[R_e * S_e *gid(0):R_e*S_e * gid(0)+R_e*S_e,
#                                  j_tile*j_tile_len:(j_tile+1)*j_tile_len]
#     // uprcmpt is a local variable
#     // Fetch uxprcmpt_priv[:,:] <- uxprcmpt[R_e*lid(1):R_e*lid(1)+R_e][:j_tile_len]
#     // Fetch uyprcmpt_priv[:,:] <- uyprcmpt[R_e*lid(1):R_e*lid(1)+R_e][:j_tile_len]
#     // Fetch uzprcmpt_priv[:,:] <- uzprcmpt[R_e*lid(1):R_e*lid(1)+R_e][:j_tile_len]
#     for (int i_tile = 0; i_tile < i_tiles; i_tile++) {
#       for (int j = 0; j < j_tile_len; j++) {
#         for (int r = 0; r < 3; r++) {
#           double Dprcmpt = D[r, i_tile_len*i_tile+lid(0), j_tile_len*j_tile+j];
#           for (int e_inner = 0; e_inner < R_e; e_inner++) {
#             dux_acc[e_inner][i_tile][r] += Dprcmpt
#                                           * uxprcmpt_priv[e_inner, j];
#             duy_acc[e_inner][i_tile][r] += Dprcmpt
#                                           * uyprcmpt_priv[e_inner, j];
#             duz_acc[e_inner][i_tile][r] += Dprcmpt
#                                           * uzprcmpt_priv[e_inner, j];
#           }
#         }
#       }
#     }
#   }
#
#   double dux_dy_acc[R_e][i_tiles], dux_dz_acc[R_e][i_tiles];
#   double duy_dx_acc[R_e][i_tiles], duy_dz_acc[R_e][i_tiles];
#   double duz_dx_acc[R_e][i_tiles], duz_dy_acc[R_e][i_tiles];
#
#   for (int e_inner  0; e_inner < R_e; e_inner++)
#     for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#       dux_dy_acc[e_inner][i_tile]=0;
#       dux_dz_acc[e_inner][i_tile]=0;
#       duy_dx_acc[e_inner][i_tile]=0;
#       duy_dz_acc[e_inner][i_tile]=0;
#       duz_dx_acc[e_inner][i_tile]=0;
#       duz_dy_acc[e_inner][i_tile]=0;
#     }
#
#   for (int e_inner  0; e_inner < R_e; e_inner++) {
#     int e = S_r*R_e*gid(0) + Re*lid(0) + e_inner;
#     for (int r = 0; r < 3; r++) {
#       double Jxr = Jx[r][e], Jyr = Jy[r][e], Jzr = Jz[r][e];
#       for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#         dux_dy_acc[e_inner][i_tile] += Jyr * dux_acc[e_inner][i_tile][r];
#         dux_dz_acc[e_inner][i_tile] += Jzr * dux_acc[e_inner][i_tile][r];
#         duy_dx_acc[e_inner][i_tile] += Jxr * dux_acc[e_inner][i_tile][r];
#         duy_dz_acc[e_inner][i_tile] += Jzr * dux_acc[e_inner][i_tile][r];
#         duz_dx_acc[e_inner][i_tile] += Jxr * dux_acc[e_inner][i_tile][r];
#         duz_dy_acc[e_inner][i_tile] += Jyr * dux_acc[e_inner][i_tile][r];
#       }
#     }
#   }
#
#   for (int e_inner  0; e_inner < R_e; e_inner++) {
#     int e = S_r*R_e*gid(0) + Re*lid(0) + e_inner;
#     for (int i_tile=0; i_tile < i_tiles; i_tile++) {
#       dux_dy[e][i_tile*i_tile_len + lid(0)] = dux_dy_acc[e_inner][i_tile];
#       dux_dz[e][i_tile*i_tile_len + lid(0)] = dux_dz_acc[e_inner][i_tile];
#       duy_dx[e][i_tile*i_tile_len + lid(0)] = duy_dx_acc[e_inner][i_tile];
#       duy_dz[e][i_tile*i_tile_len + lid(0)] = duy_dz_acc[e_inner][i_tile];
#       duz_dx[e][i_tile*i_tile_len + lid(0)] = duz_dx_acc[e_inner][i_tile];
#       duz_dy[e][i_tile*i_tile_len + lid(0)] = duz_dy_acc[e_inner][i_tile];
#     }
#   }
#   // ... (For n_fields_in_a_group = 1 ->)
#   // ... (Repeats the same for vx, vy, vz.)
#   // ... (However, if nfields_in_a_group = 2, the computations for vx,vy,vz
#   //      are fused in the earlier loops)
# }
@fnsm.tuning.einsum_arg("ndim", lambda e: _get_3d_arg(e.args[0]).shape[0])
@fnsm.tuning.einsum_arg("ndof", lambda e: e.shape[1])
@fnsm.tuning.transform_param("n_fields_in_a_group", lambda e: IntParameter(1, 2))
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
    n_fields_in_a_group: int,
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

    if s_e * math.ceil(ndof / i_tiles) > 600:
        raise fnsm.InvalidParameterError("Block dimension limit exceeded")

    if math.ceil(ndof / j_tiles) * s_e * r_e * (6 / n_fields_in_a_group) * 8e-3 > 47:
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
                fnsm.array(J, (ndim, "Nel")),
                fnsm.array("D", (ndim, ndof, ndof)),
                fnsm.array(u, ("Nel", ndof)),
            ]
            for u in _U_NAMES
            for J in _U_TO_J_PAIR[u]
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
    Js = (sigma["Jx"], sigma["Jy"], sigma["Jz"])
    us = tuple(sigma[u] for u in _U_NAMES)

    u_to_outputs = constantdict(
        {
            sigma[u]: (sigma[fe_out(2 * k)], sigma[fe_out(2 * k + 1)])
            for k, u in enumerate(_U_NAMES)
        }
    )

    if n_fields_in_a_group == 2:
        xyz_locals = {ax: vng(f"_tmp_{sigma['u' + ax]}") for ax in "xyz"}
        u_to_local_name = constantdict(
            {sigma[f"{f}{ax}"]: xyz_locals[ax] for f in "uv" for ax in "xyz"}
        )
        u_to_local_reuse_predecessor = constantdict(
            {sigma[f"v{ax}"]: sigma[f"u{ax}"] for ax in "xyz"}
        )
    else:
        u_to_local_name = constantdict({u: vng(f"_tmp_{u}") for u in us})
        u_to_local_reuse_predecessor = constantdict({})

    # }}}

    # {{{ Step 1: Split r outward and hoist J[r,e] out of the j-reductions

    t_unit = lp.split_reduction_outward(t_unit, r, within=within)
    knl = t_unit[kernel_name]
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, j, within=within
    )
    t_unit = t_unit.with_kernel(knl)
    del knl

    # }}}

    # {{{ Step 2: Extract each Du(r, e, i) = sum_j D[r,j,i]*u[e,j]

    u_to_du = constantdict({u: vng(f"D{u}") for u in us})
    for u, outputs in u_to_outputs.items():
        template = next(
            _get_reduction_expression_with_inames(insn.expression, frozenset({j}))
            for insn in t_unit[kernel_name].instructions
            if lp_match.And((within, lp_match.Writes(outputs[0])))(
                t_unit[kernel_name], insn
            )
        )
        t_unit = cast(
            "lp.TranslationUnit",
            lp.extract_subst(  # pyright: ignore[reportUnknownMemberType]
                t_unit,
                template=template,
                subst_name=u_to_du[u],
                parameters=(r, e, i),
                within=lp_match.And((within, _writes_any(outputs))),
            ),
        )

    # }}}

    u_to_upriv_fetch_id: dict[str, str] = {}
    for us_in_batch in itertools.batched(
        us, len(us) // n_fields_in_a_group, strict=True
    ):
        new_e, new_i, new_r, new_j = vng("e"), vng("i"), vng("r"), vng("j")
        batch_outputs = tuple(
            output for u in us_in_batch for output in u_to_outputs[u]
        )
        batch_match = lp_match.And((within, _writes_any(batch_outputs)))

        t_unit = lp.duplicate_inames(
            t_unit,
            (e, i, r, j),
            within=batch_match,
            new_inames=(new_e, new_i, new_r, new_j),
        )
        t_unit = t_unit.with_kernel(
            lp_utils.decouple_domain(
                t_unit[kernel_name],
                [new_e, new_i, new_r, new_j],
                parent_inames=cast(
                    "Collection[str]",
                    t_unit[kernel_name]
                    .get_inames_domain(e)
                    .get_var_names(isl.dim_type.param),
                ),
            )
        )

        batch_matched_insn_ids = tuple(
            insn.id
            for insn in t_unit[kernel_name].instructions
            if batch_match(t_unit[kernel_name], insn)
        )

        # e_outer: g.0 blocks; e_s_e: l.1 (size s_e); e_r_e: private unrolled lanes.
        e_inner, e_outer = vng(f"{new_e}_inner"), vng(f"{new_e}_outer")
        e_s_e = vng(f"{new_e}_se")
        e_r_e = vng(f"{new_e}_re")

        i_tile_iname = vng(f"{new_i}_tile")
        i_inner_iname = vng(f"{new_i}_inner")
        i_tile_len = math.ceil(ndof / i_tiles)

        j_tile_iname = vng(f"{new_j}_tile")
        j_inner_iname = vng(f"{new_j}_inner")
        j_tile_len = math.ceil(ndof / j_tiles)

        t_unit = lp.split_iname(
            t_unit,
            new_e,
            s_e * r_e,
            outer_tag="g.0",
            inner_iname=e_inner,
            outer_iname=e_outer,
            within=batch_match,
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
            new_i,
            i_tile_len,
            outer_iname=i_tile_iname,
            inner_iname=i_inner_iname,
            within=batch_match,
            inner_tag="l.0",
            outer_tag="unr",
        )
        t_unit = lp.split_iname(
            t_unit,
            new_j,
            j_tile_len,
            outer_iname=j_tile_iname,
            inner_iname=j_inner_iname,
            within=batch_match,
            inner_tag="unr",
        )

        u_to_priv: dict[str, str] = {}

        # {{{ Stage 1: tile all u's into shared memory

        uprcmpt_e = vng("uprcmpt_e")
        uprcmpt_j = vng("uprcmpt_j")

        for u in us_in_batch:
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                u,
                sweep_inames=[e_s_e, e_r_e, j_inner_iname],
                precompute_inames=[uprcmpt_e, uprcmpt_j],
                precompute_outer_inames=frozenset({e_outer, j_tile_iname}),
                temporary_address_space=lp.AddressSpace.LOCAL,
                temporary_name=u_to_local_name[u],
                within=batch_match,
                default_tag=None,
            )

        uprcmpt_e_se = vng("uprcmpt_e_se")
        uprcmpt_e_re = vng("uprcmpt_e_re")
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
        t_unit = lp.tag_inames(t_unit, {uprcmpt_e_re: "unr"})

        # }}}

        # {{{ Stage 2: prefetch all u's to private registers

        upriv_e = vng("upriv_e")
        upriv_j = vng("upriv_j")
        upriv_fetch_ids: list[str] = []
        upriv_e_inames: list[str] = []
        upriv_j_inames: list[str] = []

        for u in us_in_batch:
            u_priv_name = vng(f"_tmp_{u}_priv")
            u_priv_fetch_id = ing(f"{u}_priv_fetch")
            u_to_priv[u] = u_priv_name

            t_unit = lp.add_prefetch(
                t_unit,
                u_to_local_name[u],
                sweep_inames=[e_r_e, j_inner_iname],
                fetch_outer_inames=frozenset(
                    {e_s_e, e_outer, i_inner_iname, j_tile_iname}
                ),
                temporary_address_space=lp.AddressSpace.PRIVATE,
                temporary_name=u_priv_name,
                default_tag="unr",
                prefetch_insn_id=u_priv_fetch_id,
                within=lp_match.And((within, _writes_any(u_to_outputs[u]))),
            )
            u_to_upriv_fetch_id[u] = u_priv_fetch_id
            upriv_fetch_ids.append(u_priv_fetch_id)
            upriv_inames = tuple(
                iname
                for iname in sorted(
                    t_unit[kernel_name].id_to_insn[u_priv_fetch_id].within_inames
                    - frozenset({e_s_e, e_outer, i_inner_iname, j_tile_iname})
                )
            )
            # When r_e=1, the e-dimension has size 1 and loopy collapses it,
            # leaving only the j-dim iname.
            assert 1 <= len(upriv_inames) <= 2
            if len(upriv_inames) == 2:
                upriv_e_inames.append(upriv_inames[0])
            upriv_j_inames.append(upriv_inames[-1])

            if u in u_to_local_reuse_predecessor:
                t_unit = lp.add_dependency(
                    t_unit,
                    insn_match=lp_match.Id(u),
                    depends_on=lp_match.Id(
                        u_to_upriv_fetch_id[u_to_local_reuse_predecessor[u]]
                    ),
                )

        upriv_fetch_match = lp_match.Or(
            tuple(lp_match.Id(fid) for fid in upriv_fetch_ids)
        )
        for per_u, shared in [
            (upriv_e_inames, upriv_e),
            (upriv_j_inames, upriv_j),
        ]:
            if per_u:
                t_unit = lp.rename_inames(
                    t_unit, per_u, shared, existing_ok=True, within=upriv_fetch_match
                )

        # }}}

        # {{{ Stage 3: precompute Du = sum_j D * u_priv for all u's

        prcmpt_r = vng("_prcmpt_r_Du")
        prcmpt_re = vng("_prcmpt_re_Du")
        prcmpt_Du_ids: list[str] = []

        for u in us_in_batch:
            prcmpt_Du_id = ing(f"_compute_{u_to_du[u]}")
            du_tmp_name = vng(f"_tmp_{u_to_du[u]}")
            prcmpt_Du_ids.append(prcmpt_Du_id)

            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                u_to_du[u],
                sweep_inames=[new_r, e_r_e],
                precompute_inames=[prcmpt_r, prcmpt_re],
                precompute_outer_inames=frozenset(
                    {e_outer, e_s_e, i_tile_iname, i_inner_iname}
                ),
                temporary_address_space=lp.AddressSpace.PRIVATE,
                compute_insn_id=prcmpt_Du_id,
                default_tag="unr",
                temporary_name=du_tmp_name,
                within=lp_match.And((within, _writes_any(u_to_outputs[u]))),
            )
            t_unit = lp.privatize_temporaries_with_inames(
                t_unit, i_tile_iname, du_tmp_name
            )

        prcmpt_itile = vng("_prcmpt_itile_Du")
        t_unit = lp.duplicate_inames(
            t_unit,
            (i_tile_iname,),
            within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in prcmpt_Du_ids)),
            new_inames=(prcmpt_itile,),
            tags={prcmpt_itile: "unr"},
        )

        # }}}

        # {{{ Stage 4: realize reductions for all Du's

        du_update_insn_ids: list[str] = []
        du_init_insn_ids: list[str] = []
        du_assign_insn_ids: list[str] = []

        for prcmpt_Du_id, u in zip(prcmpt_Du_ids, us_in_batch, strict=True):
            t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_Du_id)
            (acc_name,) = (
                t_unit[kernel_name].id_to_insn[prcmpt_Du_id].read_dependency_names()
                - t_unit[kernel_name].all_inames()
            )

            # prcmpt_re disappears when r_e=1; prcmpt_itile when i_tiles=1.
            inames_to_dup = tuple(
                iname
                for iname in (prcmpt_itile, prcmpt_r, prcmpt_re)
                if iname in t_unit[kernel_name].all_inames()
            )
            t_unit = lp.privatize_temporaries_with_inames(
                t_unit,
                frozenset(inames_to_dup),
                only_var_names=frozenset({acc_name}),
            )

            acc_init_id, acc_assign_id = _find_acc_init_assign_ids(
                t_unit[kernel_name], acc_name
            )
            (du_update_insn_id,) = [
                insn.id
                for insn in t_unit[kernel_name].instructions
                if (
                    u_to_priv[u] in insn.read_dependency_names()
                    and acc_name in insn.read_dependency_names()
                    and acc_name in insn.write_dependency_names()
                )
            ]
            du_update_insn_ids.append(du_update_insn_id)
            du_init_insn_ids.append(acc_init_id)
            du_assign_insn_ids.append(acc_assign_id)

        # init and assign must be schedulable outside the j-tile loop independently
        # of the update instructions; give each role its own iname copies.
        # Only duplicate inames that actually exist (prcmpt_re absent when r_e=1,
        # prcmpt_itile absent when i_tiles=1).
        du_existing_base = tuple(
            iname
            for iname in (prcmpt_itile, prcmpt_r, prcmpt_re)
            if iname in t_unit[kernel_name].all_inames()
        )
        du_init_names = tuple(vng(f"{n}_init") for n in du_existing_base)
        du_assign_names = tuple(vng(f"{n}_assign") for n in du_existing_base)
        t_unit = lp.duplicate_inames(
            t_unit,
            du_existing_base,
            within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in du_init_insn_ids)),
            new_inames=du_init_names,
            tags=dict.fromkeys(du_init_names, "unr"),
        )
        t_unit = lp.duplicate_inames(
            t_unit,
            du_existing_base,
            within=lp_match.Or(
                tuple(lp_match.Id(id_) for id_ in du_assign_insn_ids)
            ),
            new_inames=du_assign_names,
            tags=dict.fromkeys(du_assign_names, "unr"),
        )
        t_unit = lp.tag_inames(
            t_unit, dict.fromkeys(du_existing_base, "unr")
        )

        # }}}

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
            within=lp_match.Or(
                tuple(lp_match.Id(id_) for id_ in du_update_insn_ids)
            ),
        )

        t_unit = lp.prioritize_loops(t_unit, (prcmpt_r, j_inner_iname, prcmpt_itile))

        t_unit = lp.realize_reduction(t_unit, insn_id_filter=batch_matched_insn_ids)
        acc_names = tuple(
            sorted(
                _fset_union(
                    t_unit[kernel_name].id_to_insn[id_].read_dependency_names()
                    - t_unit[kernel_name].all_inames()
                    for id_ in batch_matched_insn_ids
                )
            )
        )
        assert len(acc_names) == len(batch_outputs)
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, i_tile_iname, only_var_names=acc_names
        )
        t_unit = lp.privatize_temporaries_with_inames(
            t_unit, e_r_e, only_var_names=acc_names
        )

        acc_init_ids, acc_assign_ids = zip(
            *(
                _find_acc_init_assign_ids(t_unit[kernel_name], name)
                for name in acc_names
            ),
            strict=True,
        )

        e_r_e_tags = t_unit[kernel_name].iname_tags(e_r_e)
        dup_tags = {i_tile_iname: "unr", e_r_e: e_r_e_tags}
        e_r_e_init, i_tile_init = vng(f"{e_r_e}"), vng(f"{i_tile_iname}")
        t_unit = lp.duplicate_inames(
            t_unit,
            (i_tile_iname, e_r_e),
            within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in acc_init_ids)),
            new_inames=(i_tile_init, e_r_e_init),
            tags=dup_tags,
        )
        e_r_e_assign, i_tile_assign = vng(f"{e_r_e}"), vng(f"{i_tile_iname}")
        t_unit = lp.duplicate_inames(
            t_unit,
            (i_tile_iname, e_r_e),
            within=lp_match.Or(tuple(lp_match.Id(id_) for id_ in acc_assign_ids)),
            new_inames=(i_tile_assign, e_r_e_assign),
            tags=dup_tags,
        )
        t_unit = lp.tag_inames(t_unit, {new_r: "unr"})

        for J in Js:
            t_unit = lp.precompute(  # type: ignore[no-untyped-call]
                t_unit,
                J,
                sweep_inames=[],
                precompute_outer_inames=frozenset(
                    (new_r, i_inner_iname, e_s_e, e_r_e, e_outer)
                ),
                temporary_address_space=lp.AddressSpace.PRIVATE,
                default_tag=None,
                within=lp_match.And(
                    (
                        lp_match.Iname(e_outer),
                        lp_match.Iname(e_s_e),
                        lp_match.Iname(e_r_e),
                        lp_match.Iname(new_r),
                    )
                ),
            )

        t_unit = lp.prioritize_loops(t_unit, (new_r, i_tile_iname))

    t_unit = lp.remove_unused_inames(t_unit)

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
        _ux(_0, _1) := ux[_0, _1]
        _uy(_0, _1) := uy[_0, _1]
        _uz(_0, _1) := uz[_0, _1]
        _vx(_0, _1) := vx[_0, _1]
        _vy(_0, _1) := vy[_0, _1]
        _vz(_0, _1) := vz[_0, _1]

        dux_y[e, i] = sum([r,j], _Jy(r,e)*_D(r,j,i)*_ux(e,j))
        dux_z[e, i] = sum([r,j], _Jz(r,e)*_D(r,j,i)*_ux(e,j))
        duy_x[e, i] = sum([r,j], _Jx(r,e)*_D(r,j,i)*_uy(e,j))
        duy_z[e, i] = sum([r,j], _Jz(r,e)*_D(r,j,i)*_uy(e,j))
        duz_x[e, i] = sum([r,j], _Jx(r,e)*_D(r,j,i)*_uz(e,j))
        duz_y[e, i] = sum([r,j], _Jy(r,e)*_D(r,j,i)*_uz(e,j))

        dvx_y[e, i] = sum([r,j], _Jy(r,e)*_D(r,j,i)*_vx(e,j))
        dvx_z[e, i] = sum([r,j], _Jz(r,e)*_D(r,j,i)*_vx(e,j))
        dvy_x[e, i] = sum([r,j], _Jx(r,e)*_D(r,j,i)*_vy(e,j))
        dvy_z[e, i] = sum([r,j], _Jz(r,e)*_D(r,j,i)*_vy(e,j))
        dvz_x[e, i] = sum([r,j], _Jx(r,e)*_D(r,j,i)*_vz(e,j))
        dvz_y[e, i] = sum([r,j], _Jy(r,e)*_D(r,j,i)*_vz(e,j))
        """,
        [
            lp.GlobalArg(
                "Jx,Jy,Jz,D,ux,uy,uz,vx,vy,vz", dtype="float64", shape=lp.auto
            ),
            ...,
        ],
        lang_version=(2018, 2),
    )
    t_unit = transform(
        t_unit,
        ndim=3,
        ndof=20,
        n_fields_in_a_group=2,
        i_tiles=2,
        j_tiles=2,
        s_e_log2=4,
        r_e=2,
    )
    print(lp.generate_code_v2(t_unit).device_code())  # type: ignore[no-untyped-call]

# vim: fdm=marker
