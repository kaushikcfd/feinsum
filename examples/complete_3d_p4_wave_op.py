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
    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum)
    t_unit = lp.tag_inames(t_unit, {subst_map["s"]: "unr"})
    t_unit = lp.split_iname(t_unit, subst_map["e"], 8,
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, subst_map["i"], 4,
                            inner_tag="l.0", outer_tag="ilp")

    return t_unit


def transform_grad(t_unit, insn_tag=None, kernel_name=None):
    ...


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


def report_grad_performance():
    ...


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
        report_div_performance(cl_ctx)
