import numpy as np
import pyopencl as cl
import loopy as lp
import feinsum as f
import logging

from loopy.match import Tagged
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def main(cl_ctx):
    t_unit = lp.make_kernel(
        ["{[iel_0, idof_0, jdof_0, r_0]:"
         " 0<=iel_0<10000 and 0<=idof_0,jdof_0<35 and 0<=r_0<3}",
         "{[iel_1, idof_1, jdof_1, r_1, x_1]:"
         " 0<=iel_1<10000 and 0<=idof_1,jdof_1<35 and 0<=r_1,x_1<3}",
         "{[iel_2, idof_2, ifacedof, iface]:"
         " 0<=iel_2<10000 and 0<=idof_2<35 and 0<=ifacedof<15 and 0<=iface<4}"],
        """
        # ----- Div(v)
        with {tags=div}
            div_out_x[iel_0,idof_0] = sum([jdof_0,r_0], \
                                          Jx[iel_0,r_0]*R[r_0,idof_0,jdof_0]*vx[iel_0,jdof_0])
            div_out_y[iel_0,idof_0] = sum([jdof_0,r_0], \
                                          Jy[iel_0,r_0]*R[r_0,idof_0,jdof_0]*vy[iel_0,jdof_0])
            div_out_z[iel_0,idof_0] = sum([jdof_0,r_0], \
                                          Jz[iel_0,r_0]*R[r_0,idof_0,jdof_0]*vz[iel_0,jdof_0])
        end

        ... gbarrier {id=g_barrier_0, dep_query=(writes:div_out_*)}
        # ----- Grad(u)
        with {dep=g_barrier_0, tags=grad}
            grad_out[x_1, iel_1, idof_1] = sum([jdof_1, r_1], \
                                               J[x_1, iel_1, r_1]*R[r_1, idof_1, jdof_1]*u[iel_1, jdof_1])
        end

        ... gbarrier {id=g_barrier_1, dep_query=(writes:grad_out)}
        # ----- Lift(f*)
        with {dep=g_barrier_1, tags=lift}
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

    # {{{ feinsum transformations

    grad_einsum = f.match_einsum(t_unit, insn_match=Tagged("grad"))
    div_einsum = f.match_einsum(t_unit, insn_match=Tagged("div"))
    lift_einsum = f.match_einsum(t_unit, insn_match=Tagged("lift"))

    fast_grad_einsum = max(
        f.query(grad_einsum, cl_ctx),
        key=lambda q: q.giga_op_info[np.dtype("float64")]/q.runtime_in_sec)
    fast_div_einsum = max(
        f.query(div_einsum, cl_ctx),
        key=lambda q: q.giga_op_info[np.dtype("float64")]/q.runtime_in_sec)
    fast_lift_einsum = max(
        f.query(lift_einsum, cl_ctx),
        key=lambda q: q.giga_op_info[np.dtype("float64")]/q.runtime_in_sec)

    t_unit = fast_grad_einsum.transform(t_unit, insn_match=Tagged("grad"))
    t_unit = fast_div_einsum.transform(t_unit, insn_match=Tagged("div"))
    t_unit = fast_lift_einsum.transform(t_unit, insn_match=Tagged("lift"))

    # }}}

    print(lp.generate_code_v2(t_unit).device_code())


if __name__ == "__main__":
    from feinsum.data.device_info import DEV_TO_PEAK_GFLOPS
    cl_ctx = cl.create_some_context()

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
    elif cl_ctx.devices[0].name not in DEV_TO_PEAK_GFLOPS:
        logger.info(f"Device {cl_ctx.devices[0]} not known to database.")
    else:
        main(cl_ctx)
