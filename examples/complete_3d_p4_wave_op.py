import numpy as np
import pyopencl as cl
import loopy as lp
import feinsum as f
import logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)

from pytools.tag import Tag


class DivInsnTag(Tag):
    """
    Tagged to divergence op statements.
    """


class GradInsnTag(Tag):
    """
    Tagged to grad statements.
    """


class LiftInsnTag(Tag):
    """
    Tagged to Lift operator statements.
    """


def transform_div(t_unit, insn_tag=None, kernel_name=None):
    ndim = 3
    ndofs = 35
    ref_einsum = f.fused_einsum("es, sij, ej -> ei",
                                [(np.inf, ndim),
                                 (ndim, ndofs, ndofs),
                                 (np.inf, ndofs)],
                                dtypes="float64",
                                use_matrix=[
                                    [{"Jx"}, {"R"}, {"ux"}],
                                    [{"Jy"}, {"R"}, {"uy"}],
                                    [{"Jz"}, {"R"}, {"uz"}],
                                ])
    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum, insn_tag)
    t_unit = lp.tag_inames(t_unit, {subst_map["s"]: "unr"})
    t_unit = lp.split_iname(t_unit, subst_map["e"], 8,
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, subst_map["i"], 4,
                            inner_tag="l.0", outer_tag="ilp")

    return t_unit


def transform_grad(t_unit, insn_tag=None, kernel_name=None):

    # {{{ define ref_einsum; get subst_map

    ndim = 3
    ndofs = 35

    ref_einsum = f.einsum("xer,rij,ej->xei",
                    f.array((ndim, np.inf, ndim,),
                            "float64"),
                    f.array((ndim, ndofs, ndofs),
                            "float64"),
                    f.array((np.inf, ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])

    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum, insn_tag)

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
    t_unit = f.hoist_reduction_invariant_terms(t_unit, j)
    t_unit = f.extract_einsum_terms_as_subst(
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

    t_unit = lp.join_inames(t_unit, [e_prftch, j_prftch], "i_uprftch")
    t_unit = lp.split_iname(t_unit, "i_uprftch",
                            ncells_per_workgroup * nworkitems_per_cell,
                            outer_tag="unr")

    t_unit = lp.split_iname(t_unit, "i_uprftch_inner",
                            nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="l.1")

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
                             )
    t_unit = lp.join_inames(t_unit, [r_prftch, i_prftch, j_prftch],
                            "i_Rprftch")

    t_unit = lp.split_iname(t_unit, "i_Rprftch",
                            ncells_per_workgroup * nworkitems_per_cell,
                            outer_tag="unr")

    t_unit = lp.split_iname(t_unit, "i_Rprftch_inner",
                            nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="l.1")

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
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  frozenset([r_prcmpt,
                                                             "i_outer_hoist"]),
                                                  only_var_names={"acc_j_tile_j"})
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


def transform_face_mass(t_unit, insn_tag=None, kernel_name=None):
    ...


def report_div_performance(cl_ctx):
    ndim = 3
    ndofs = 35
    expr = f.fused_einsum("es, sij, ej -> ei",
                          [(np.inf, ndim),
                           (ndim, ndofs, ndofs),
                           (np.inf, ndofs)],
                          dtypes="float64",
                          use_matrix=[
                              [{"Jx"}, {"R"}, {"ux"}],
                              [{"Jy"}, {"R"}, {"uy"}],
                              [{"Jz"}, {"R"}, {"uz"}],
                          ])
    print(f.stringify_comparison_vs_roofline(expr,
                                             cl_ctx=cl_ctx,
                                             transform=transform_div,
                                             long_dim_length=100_000,
                                             ))


def report_grad_performance(cl_ctx):
    ndim = 3
    ndofs = 35
    expr = f.einsum("xer,rij,ej->xei",
                    f.array((ndim, np.inf, ndim,),
                            "float64"),
                    f.array((ndim, ndofs, ndofs),
                            "float64"),
                    f.array((np.inf, ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])
    print(
        f.stringify_comparison_vs_roofline(
            expr,
            cl_ctx=cl_ctx,
            transform=transform_grad))


def report_face_mass_performance():
    ...


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

        .. gbarrier {dep_query: (writes:div_out_*)}
        # ----- Grad(u)
        grad_out[x_1, iel_1, idof_1] = sum([jdof_1, r_1], \
                                           J[x_1, iel_1, r_1]*R[r_1, idof_1, jdof_1]*u[iel_1, jdof_1])

        .. gbarrier {dep_query: (writes:grad_out)}
        # ----- Lift(f)
        lift_0[iel_2, idof_2] = sum([iface, ifacedof], \
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_0[iface, iel_2, ifacedof])
        lift_1[iel_2, idof_2] = sum([iface, ifacedof], \
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_1[iface, iel_2, ifacedof])
        lift_2[iel_2, idof_2] = sum([iface, ifacedof], \
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_2[iface, iel_2, ifacedof])
        lift_3[iel_2, idof_2] = sum([iface, ifacedof], \
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_3[iface, iel_2, ifacedof])
        """  # noqa: E501
    )

    t_unit = lp.add_dtypes(t_unit,
                           {arg.name: np.float64
                            for arg in t_unit.default_entrypoint.args
                            if arg.is_input})
    # TODO: Actually transform this kernel wrt to the above 3 transformations.

    print(t_unit)


if __name__ == "__main__":
    from feinsum.data.device_info import DEV_TO_PEAK_GFLOPS
    cl_ctx = cl.create_some_context()

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
    elif cl_ctx.devices[0].name not in DEV_TO_PEAK_GFLOPS:
        logger.info("Device not known.")
    else:
        # main()
        # report_div_performance(cl_ctx)
        report_grad_performance(cl_ctx)
