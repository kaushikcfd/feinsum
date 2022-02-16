import numpy as np
import pyopencl as cl
import loopy as lp
import feinsum as fnsm
import logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def transform_div(t_unit, insn_match=None, kernel_name=None):
    ndim = 3
    ndofs = 35
    ref_einsum = fnsm.fused_einsum("es, sij, ej -> ei",
                                [(np.inf, ndim),
                                 (ndim, ndofs, ndofs),
                                 (np.inf, ndofs)],
                                dtypes="float64",
                                use_matrix=[
                                    [{"Jx"}, {"R"}, {"ux"}],
                                    [{"Jy"}, {"R"}, {"uy"}],
                                    [{"Jz"}, {"R"}, {"uz"}],
                                ])
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match)
    t_unit = lp.tag_inames(t_unit, {subst_map["s"]: "unr"})
    t_unit = lp.split_iname(t_unit, subst_map["e"], 8,
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, subst_map["i"], 4,
                            inner_tag="l.0", outer_tag="ilp")

    return t_unit


def transform_grad(t_unit, insn_match=None, kernel_name=None):
    from loopy.match import parse_match
    insn_match = parse_match(insn_match)

    # {{{ define ref_einsum; get subst_map

    ndim = 3
    ndofs = 35

    ref_einsum = fnsm.einsum("xer,rij,ej->xei",
                    fnsm.array((ndim, np.inf, ndim,),
                            "float64"),
                    fnsm.array((ndim, ndofs, ndofs),
                            "float64"),
                    fnsm.array((np.inf, ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])

    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match)

    # }}}

    # {{{ transform parameters

    ncells_per_workgroup = 9
    nworkitems_per_cell = 7
    j_tile_len = 9
    i_tile_len = 35

    # }}}

    # {{{ read variables

    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    R = subst_map["R"]
    J = subst_map["J"]
    u = subst_map["u"]
    x = subst_map["x"]
    r = subst_map["r"]
    out = subst_map["_fe_out"]
    e_inner, e_outer = f"{e}_inner", f"{e}_outer"
    i_inner, i_outer = f"{i}_inner", f"{i}_outer"
    j_inner = f"{j}_inner"
    i_tile, j_tile = f"{i}_tile", f"{j}_tile"
    J_prftch = f"{J}_prftch"
    e_prftch = f"{e}_prftch"
    j_prftch = f"{j}_prftch"
    i_prftch = f"{i}_prftch"
    r_prftch = f"{r}_prftch"
    r_prcmpt = f"{r}_prcmpt"

    # }}}

    # {{{ term hoisting to match the flop count of opt_einsum

    t_unit = lp.split_reduction_inward(t_unit, j)
    t_unit = fnsm.hoist_reduction_invariant_terms(t_unit, j)
    t_unit = fnsm.extract_einsum_terms_as_subst(
        t_unit,
        f"subst({r}, {e}, {i})",
        f"sum({j}, {R}[{r}, {i}, {j}]*{u}[{e}, {j}])")

    # }}}

    t_unit = lp.split_iname(t_unit, i, i_tile_len, outer_iname=i_tile)
    t_unit = lp.split_iname(t_unit, j, j_tile_len, outer_iname=j_tile)

    t_unit = lp.rename_iname(t_unit, i_inner, i)
    t_unit = lp.rename_iname(t_unit, j_inner, j)
    t_unit = lp.split_iname(t_unit, e, ncells_per_workgroup,
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, i, nworkitems_per_cell,
                            inner_tag="l.0")

    t_unit = lp.add_prefetch(t_unit,
                             J,
                             sweep_inames=[r, x],
                             fetch_outer_inames=frozenset([e_inner,
                                                           e_outer,
                                                           i_inner]),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name=J_prftch,
                             )

    # {{{ TODO: Make precompute smarter (should be a single precompute call)

    t_unit = lp.precompute(t_unit, "subst",
                           sweep_inames=[r],
                           precompute_outer_inames=frozenset({e_inner,
                                                              e_outer,
                                                              i_inner,
                                                              i_outer,
                                                              i_tile}),
                           precompute_inames=r_prcmpt,
                           temporary_name="tmp_hoist",
                           temporary_address_space=lp.AddressSpace.PRIVATE,
                           compute_insn_id="insn_hoist",
                           )
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  i_outer,
                                                  only_var_names=["tmp_hoist"])

    t_unit = lp.duplicate_inames(t_unit, i_outer, "id:insn_hoist",
                                 "i_outer_hoist")

    # }}}

    # {{{ Move 'u ' to shared.

    # Prefetch 'u' within the tile
    t_unit = lp.add_prefetch(t_unit, u,
                             sweep_inames=[e_inner, j],
                             fetch_outer_inames=frozenset([e_outer,
                                                           i_tile,
                                                           j_tile]),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             dim_arg_names=[e_prftch, j_prftch],
                             default_tag=None,
                             )
    t_unit = lp.split_iname(t_unit, e_prftch, ncells_per_workgroup,
                            inner_tag="l.1", outer_tag="unr")
    t_unit = lp.split_iname(t_unit, j_prftch, nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="unr")

    # }}}

    # {{{ Move 'R' to shared.

    t_unit = lp.add_prefetch(t_unit, R,
                             sweep_inames=[r_prcmpt, i_inner, "i_outer_hoist", j],
                             fetch_outer_inames=frozenset([e_outer,
                                                           i_tile,
                                                           j_tile]),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             dim_arg_names=[r_prftch, i_prftch, j_prftch],
                             default_tag=None,
                             within="id:insn_hoist",
                             )
    if 0:
        # This branch improves perf. by 20% but, something in loopy
        # non-deterministically leads to very ugly domains.
        t_unit = lp.join_inames(t_unit, [r_prftch, i_prftch, j_prftch],
                                "i_Rprftch")

        t_unit = lp.split_iname(t_unit, "i_Rprftch",
                                ncells_per_workgroup * nworkitems_per_cell,
                                outer_tag="unr")

        t_unit = lp.split_iname(t_unit, "i_Rprftch_inner",
                                nworkitems_per_cell,
                                inner_tag="l.0", outer_tag="l.1")
    else:
        t_unit = lp.split_iname(t_unit, i_prftch, ncells_per_workgroup,
                                inner_tag="l.1", outer_tag="unr")
        t_unit = lp.split_iname(t_unit, j_prftch, nworkitems_per_cell,
                                inner_tag="l.0", outer_tag="unr")

    # }}}

    # {{{ make buffer array smarter (should be a single call to buffer_array)

    t_unit = lp.buffer_array(t_unit, out, buffer_inames=[x],
                             init_expression="0",
                             default_tag=None, temporary_is_local=False)
    t_unit = lp.privatize_temporaries_with_inames(t_unit, i_outer,
                                                  only_var_names={f"{out}_buf"})

    t_unit = lp.duplicate_inames(t_unit,
                                 inames=[i_outer],
                                 within=f"id:init_{out}",
                                 new_inames=[f"{out}_init_1"])

    t_unit = lp.duplicate_inames(t_unit,
                                 inames=[i_outer],
                                 within=f"id:store_{out}",
                                 new_inames=[f"{out}_store_1"])

    # }}}

    # {{{ must be smarter way of doing this in loopy

    t_unit = lp.realize_reduction(t_unit, insn_id_filter="insn_hoist")
    t_unit = lp.privatize_temporaries_with_inames(
        t_unit,
        frozenset([r_prcmpt,
                   "i_outer_hoist"]),
        only_var_names={f"acc_{j_tile}_{j}"})
    t_unit = lp.duplicate_inames(t_unit,
                                 ["i_outer_hoist", r_prcmpt],
                                 within=f"id:insn_hoist_{j_tile}_{j}_init",
                                 new_inames=["i_outer_hoist_init",
                                             f"{r_prcmpt}_init"],
                                 )

    t_unit = lp.duplicate_inames(t_unit,
                                 ["i_outer_hoist", r_prcmpt],
                                 within="id:insn_hoist",
                                 new_inames=["i_outer_hoist_store",
                                             f"{r_prcmpt}_store"],
                                 )

    # }}}

    return t_unit


def transform_face_mass(t_unit, insn_match=None, kernel_name=None):
    # {{{ define ref_einsum; get subst_map

    nvoldofs = 35
    nfacedofs = 15
    nface = 4
    ref_einsum = fnsm.fused_einsum("ef, fij, fej -> ei",
                                   [(np.inf, nface),
                                    (nface, nvoldofs, nfacedofs),
                                    (nface, np.inf, nfacedofs)],
                                   dtypes="float64",
                                   use_matrix=[
                                       [{"J"}, {"R"}, {"v0"}],
                                       [{"J"}, {"R"}, {"v1"}],
                                       [{"J"}, {"R"}, {"v2"}],
                                       [{"J"}, {"R"}, {"v3"}],
                                   ])
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match)

    # }}}

    # {{{ read variables

    e = subst_map["e"]
    e_inner, e_outer = f"{e}_inner", f"{e}_outer"
    f = subst_map["f"]
    i = subst_map["i"]
    j = subst_map["j"]
    J = subst_map["J"]
    R = subst_map["R"]
    i_0, i_1 = f"{i}_0", f"{i}_1"
    j_0, j_1 = f"{j}_0", f"{j}_1"
    f_0, f_1 = f"{f}_0", f"{f}_1"
    out0, out1, out2, out3 = (subst_map["_fe_out"], subst_map["_fe_out_0"],
                              subst_map["_fe_out_1"], subst_map["_fe_out_2"])
    v = [subst_map[f"v{i}"] for i in range(4)]

    # }}}

    # {{{ transformation parameters

    ncells_per_group = 16
    nworkitems_per_cell = 12

    # }}}

    for k in range(4):
        t_unit = fnsm.extract_einsum_terms_as_subst(
            t_unit,
            f"subst{k}({f},{e},{j})",
            f"{v[k]}[{f},{e},{j}]*{J}[{e},{f}]")

    t_unit = lp.split_iname(t_unit, e, ncells_per_group,
                            outer_tag="g.0", inner_tag="l.1")

    # {{{ fetch 'R'

    t_unit = lp.add_prefetch(t_unit,
                             R,
                             [f, i, j],
                             fetch_outer_inames=frozenset([e_outer]),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             default_tag=None,
                             dim_arg_names=["f_Rprftch",
                                            "i_Rprftch",
                                            "j_Rprftch"],
                             )

    t_unit = lp.split_iname(t_unit, "i_Rprftch", ncells_per_group,
                            inner_tag="l.1", outer_tag="unr")
    t_unit = lp.split_iname(t_unit, "j_Rprftch", nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="unr")

    logger.info(f"Prefetched '{R}'")

    # }}}

    t_unit = lp.rename_iname(t_unit, i, i_0,
                             within=f"writes:{out0} or writes:{out1}")
    t_unit = lp.rename_iname(t_unit, f, f_0,
                             within=f"writes:{out0} or writes:{out1}")
    t_unit = lp.rename_iname(t_unit, j, j_0,
                             within=f"writes:{out0} or writes:{out1}")

    t_unit = lp.rename_iname(t_unit, i, i_1,
                             within=f"writes:{out2} or writes:{out3}")
    t_unit = lp.rename_iname(t_unit, f, f_1,
                             within=f"writes:{out2} or writes:{out3}")
    t_unit = lp.rename_iname(t_unit, j, j_1,
                             within=f"writes:{out2} or writes:{out3}")

    t_unit = lp.split_iname(t_unit, i_0, nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="unr")
    t_unit = lp.split_iname(t_unit, i_1, nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="unr")

    for igrp, fgrp, jgrp, prcmpt_tmp, subst in [(0, f_0, j_0,
                                                 "prcmpt_tmp_0", "subst0"),
                                                (0, f_0, j_0,
                                                 "prcmpt_tmp_1", "subst1"),
                                                (1, f_1, j_1,
                                                 "prcmpt_tmp_0", "subst2"),
                                                (1, f_1, j_1,
                                                 "prcmpt_tmp_1", "subst3")]:
        logger.info(f"Precomputing {subst}")
        t_unit = lp.precompute(t_unit,
                               subst,
                               sweep_inames=[fgrp,
                                             e_inner,
                                             jgrp],
                               precompute_outer_inames=frozenset([e_outer]),
                               precompute_inames=[f"f_prftch_{igrp}",
                                                  f"e_prftch_{igrp}",
                                                  f"j_prftch_{igrp}"],
                               default_tag=None,
                               temporary_name=prcmpt_tmp,
                               temporary_address_space=lp.AddressSpace.LOCAL)

    for igrp in [0, 1]:
        t_unit = lp.split_iname(t_unit,
                                f"e_prftch_{igrp}",
                                ncells_per_group,
                                inner_tag="l.1", outer_tag="unr")
        t_unit = lp.split_iname(t_unit,
                                f"j_prftch_{igrp}",
                                nworkitems_per_cell,
                                inner_tag="l.0", outer_tag="unr")

    t_unit = lp.add_dependency(t_unit,
                               "id:subst2 or id:subst3",
                               f"writes:{out0} or writes:{out1}")

    logger.info("Done with transformations.")
    return t_unit


def report_div_performance(cl_ctx):
    ndim = 3
    ndofs = 35
    expr = fnsm.fused_einsum("es, sij, ej -> ei",
                          [(np.inf, ndim),
                           (ndim, ndofs, ndofs),
                           (np.inf, ndofs)],
                          dtypes="float64",
                          use_matrix=[
                              [{"Jx"}, {"R"}, {"ux"}],
                              [{"Jy"}, {"R"}, {"uy"}],
                              [{"Jz"}, {"R"}, {"uz"}],
                          ])
    print(fnsm.stringify_comparison_vs_roofline(expr,
                                                cl_ctx=cl_ctx,
                                                transform=transform_div,
                                                long_dim_length=100_000,
                                                ))


def report_grad_performance(cl_ctx):
    ndim = 3
    ndofs = 35
    expr = fnsm.einsum("xer,rij,ej->xei",
                    fnsm.array((ndim, np.inf, ndim,),
                            "float64"),
                    fnsm.array((ndim, ndofs, ndofs),
                            "float64"),
                    fnsm.array((np.inf, ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])
    print(
        fnsm.stringify_comparison_vs_roofline(
            expr,
            cl_ctx=cl_ctx,
            transform=transform_grad))


def report_face_mass_performance(cl_ctx):
    nvoldofs = 35
    nfacedofs = 15
    nface = 4
    expr = fnsm.fused_einsum("ef, fij, fej -> ei",
                          [(np.inf, nface),
                           (nface, nvoldofs, nfacedofs),
                           (nface, np.inf, nfacedofs)],
                          dtypes="float64",
                          use_matrix=[
                              [{"J"}, {"R"}, {"v0"}],
                              [{"J"}, {"R"}, {"v1"}],
                              [{"J"}, {"R"}, {"v2"}],
                              [{"J"}, {"R"}, {"v3"}],
                          ])

    print(fnsm.stringify_comparison_vs_roofline(expr,
                                                cl_ctx=cl_ctx,
                                                transform=transform_face_mass))


def main():
    t_unit = lp.make_kernel(
        ["{[iel_0, idof_0, jdof_0, r_0]:"
         " 0<=iel_0<10000 and 0<=idof_0,jdof_0<35 and 0<=r_0<3}",
         "{[iel_1, idof_1, jdof_1, r_1, x_1]:"
         " 0<=iel_1<10000 and 0<=idof_1,jdof_1<35 and 0<=r_1,x_1<3}",
         "{[iel_2, idof_2, ifacedof, iface]:"
         " 0<=iel_2<10000 and 0<=idof_2<35 and 0<=ifacedof<15 and 0<=iface<4}"],
        """
        # ----- Div(v)
        div_out_x[iel_0,idof_0] = sum([jdof_0,r_0], \
                                      Jx[iel_0,r_0]*R[r_0,idof_0,jdof_0]*vx[iel_0,jdof_0])
        div_out_y[iel_0,idof_0] = sum([jdof_0,r_0], \
                                      Jy[iel_0,r_0]*R[r_0,idof_0,jdof_0]*vy[iel_0,jdof_0])
        div_out_z[iel_0,idof_0] = sum([jdof_0,r_0], \
                                      Jz[iel_0,r_0]*R[r_0,idof_0,jdof_0]*vz[iel_0,jdof_0])

        ... gbarrier {id=g_barrier_0, dep_query=(writes:div_out_*)}
        # ----- Grad(u)
        with {dep=g_barrier_0}
            grad_out[x_1, iel_1, idof_1] = sum([jdof_1, r_1], \
                                               J[x_1, iel_1, r_1]*R[r_1, idof_1, jdof_1]*u[iel_1, jdof_1])
        end

        ... gbarrier {id=g_barrier_1, dep_query=(writes:grad_out)}
        # ----- Lift(f*)
        with {dep=g_barrier_1}
            lift_0[iel_2, idof_2] = sum([iface, ifacedof], \
                                        Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_0[iface, iel_2, ifacedof])
            lift_1[iel_2, idof_2] = sum([iface, ifacedof], \
                                        Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_1[iface, iel_2, ifacedof])
            lift_2[iel_2, idof_2] = sum([iface, ifacedof], \
                                        Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_2[iface, iel_2, ifacedof])
            lift_3[iel_2, idof_2] = sum([iface, ifacedof], \
                                        Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_3[iface, iel_2, ifacedof])
        end
        """  # noqa: E501
    )

    t_unit = lp.add_dtypes(t_unit,
                           {arg.name: np.float64
                            for arg in t_unit.default_entrypoint.args
                            if arg.is_input})

    t_unit = transform_div(t_unit, "writes:div_out_*")
    t_unit = transform_grad(t_unit, "writes:grad_out")
    t_unit = transform_face_mass(t_unit, "writes:lift_*")

    # {{{ FIXME: These should've been added by loopy

    t_unit = lp.add_dependency(t_unit,
                               "id:Rlift_fetch_rule",
                               "id:g_barrier_1")
    t_unit = lp.add_dependency(t_unit,
                               "id:J_fetch_rule",
                               "id:g_barrier_0")
    t_unit = lp.add_dependency(t_unit,
                               "id:g_barrier_1",
                               "id:store_grad_out")

    # }}}

    return t_unit


if __name__ == "__main__":
    from feinsum.data.device_info import DEV_TO_PEAK_GFLOPS
    cl_ctx = cl.create_some_context()

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
    elif cl_ctx.devices[0].name not in DEV_TO_PEAK_GFLOPS:
        logger.info("Device not known.")
    else:
        # report_div_performance(cl_ctx)
        # report_grad_performance(cl_ctx)
        # report_face_mass_performance(cl_ctx)
        main()
