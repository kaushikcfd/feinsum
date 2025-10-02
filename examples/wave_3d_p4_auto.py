import logging
import os

import loopy as lp
import numpy as np
import pyopencl as cl
from loopy.match import Tagged

import feinsum as f

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def main(cl_ctx):
    t_unit = lp.make_kernel(
        [
            "{[iel_0, idof_0, jdof_0, r_0, x_0]:"
            " 0<=iel_0<1000 and 0<=idof_0,jdof_0<35 and 0<=x_0,r_0<3}",
            "{[iel_1, idof_1, jdof_1, r_1, x_1]:"
            " 0<=iel_1<1000 and 0<=idof_1,jdof_1<35 and 0<=r_1,x_1<3}",
            "{[iel_2, idof_2, ifacedof, iface]:"
            " 0<=iel_2<1000 and 0<=idof_2<35 and 0<=ifacedof<15 and 0<=iface<4}",
        ],
        """
        v_subst(_0, _1, _2) := v[_0, _1, _2]
        u_subst(_0, _1) := u[_0, _1]
        f_0_subst(_0, _1, _2) := F_0[_0, _1, _2]
        f_1_subst(_0, _1, _2) := F_1[_0, _1, _2]
        f_2_subst(_0, _1, _2) := F_2[_0, _1, _2]
        f_3_subst(_0, _1, _2) := F_3[_0, _1, _2]
        jac_subst(_0, _1, _2) := J[_0, _1, _2]
        D_subst(_0, _1, _2) := D[_0, _1, _2]
        jac_face_subst(_0, _1) := Jface[_0, _1]
        L_subst(_0, _1, _2) := L[_0, _1, _2]

        # ----- Div(v)
        with {tags=div}
            div_out[iel_0,idof_0] = sum([jdof_0,r_0,x_0], \
                                        jac_subst(x_0, r_0, iel_0)*D_subst(r_0,idof_0,jdof_0)*v_subst(x_0, iel_0,jdof_0))
        end

        ... gbarrier {id=g_barrier_0, dep_query=(writes:div_out)}
        # ----- Grad(u)
        with {dep=g_barrier_0, tags=grad}
            grad_out[x_1, iel_1, idof_1] = sum([jdof_1, r_1], \
                                               jac_subst(x_1, r_1, iel_1)*D_subst(r_1, idof_1, jdof_1)*u_subst(iel_1, jdof_1))
        end

        ... gbarrier {id=g_barrier_1, dep_query=(writes:grad_out)}
        # ----- Lift(f*)
        with {dep=g_barrier_1, tags=lift}
            lift_0[iel_2, idof_2] = sum([iface, ifacedof], \
                                        L_subst(idof_2, iface, ifacedof)*jac_face_subst(iface, iel_2)*f_0_subst(iface, iel_2, ifacedof))
            lift_1[iel_2, idof_2] = sum([iface, ifacedof], \
                                        L_subst(idof_2, iface, ifacedof)*jac_face_subst(iface, iel_2)*f_1_subst(iface, iel_2, ifacedof))
            lift_2[iel_2, idof_2] = sum([iface, ifacedof], \
                                        L_subst(idof_2, iface, ifacedof)*jac_face_subst(iface, iel_2)*f_2_subst(iface, iel_2, ifacedof))
            lift_3[iel_2, idof_2] = sum([iface, ifacedof], \
                                        L_subst(idof_2, iface, ifacedof)*jac_face_subst(iface, iel_2)*f_3_subst(iface, iel_2, ifacedof))
        end
        """,  # noqa: E501
    )

    t_unit = lp.add_dtypes(
        t_unit,
        {
            arg.name: np.float64
            for arg in t_unit.default_entrypoint.args
            if arg.is_input
        },
    )
    ref_t_unit = t_unit.copy()

    # {{{ feinsum transformations

    grad_einsum, _ = f.get_a_matched_einsum(t_unit, insn_match=Tagged("grad"))
    div_einsum, _ = f.get_a_matched_einsum(t_unit, insn_match=Tagged("div"))
    lift_einsum, _ = f.get_a_matched_einsum(t_unit, insn_match=Tagged("lift"))

    # {{{ auto-tune

    from feinsum.tuning.impls import (
        ifj_fe_fej_to_ei,
        xre_rij_ej_to_xei,
        xre_rij_xej_to_ei,
    )

    f.autotune(
        grad_einsum,
        os.path.abspath(xre_rij_ej_to_xei.__file__),
        cl_ctx,
        long_dim_length=1000,
        stop_after=3,
    )
    f.autotune(
        div_einsum,
        os.path.abspath(xre_rij_xej_to_ei.__file__),
        cl_ctx,
        long_dim_length=1000,
        stop_after=3,
    )
    f.autotune(
        lift_einsum,
        os.path.abspath(ifj_fe_fej_to_ei.__file__),
        cl_ctx,
        long_dim_length=1000,
        stop_after=3,
    )

    # }}}

    fast_grad_einsum = max(
        f.query(grad_einsum, cl_ctx), key=lambda q: q.giga_op_rate(np.float64)
    )
    fast_div_einsum = max(
        f.query(div_einsum, cl_ctx), key=lambda q: q.giga_op_rate(np.float64)
    )
    fast_lift_einsum = max(
        f.query(lift_einsum, cl_ctx), key=lambda q: q.giga_op_rate(np.float64)
    )

    t_unit = fast_grad_einsum.transform(t_unit, insn_match=Tagged("grad"))
    t_unit = fast_lift_einsum.transform(t_unit, insn_match=Tagged("lift"))
    t_unit = fast_div_einsum.transform(t_unit, insn_match=Tagged("div"))

    # }}}

    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit)


if __name__ == "__main__":
    cl_ctx = cl.create_some_context()
    main(cl_ctx)
