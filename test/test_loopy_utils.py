__copyright__ = """Copyright (C) 2022 Kaushik Kulkarni"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

import feinsum as f
import loopy as lp
import numpy as np
import feinsum.loopy_utils as lp_utils


def test_extract_subexpr_of_associative_op_as_subst(ctx_factory):
    nface = 4
    nvoldofs = 35
    nfacedofs = 15
    face_mass = f.batched_einsum(
        "ef, fij, fej -> ei",
        [
            [
                f.array("J", ("E", nface)),
                f.array("R", (nface, nvoldofs, nfacedofs)),
                f.array(f"v{i}", (nface, "E", nfacedofs)),
            ]
            for i in range(4)
        ],
    )
    t_unit = f.generate_loopy(face_mass)
    sigma = f.match_t_unit_to_einsum(t_unit, face_mass)

    output_names = ["_fe_out"] + [f"_fe_out_{i}" for i in range(3)]

    from loopy.symbolic import get_dependencies
    from pymbolic import variables

    knl = t_unit.default_entrypoint
    iname_f, iname_e, iname_j = sigma["f"], sigma["e"], sigma["j"]

    for i in range(4):
        knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
            knl,
            within=f"writes:{sigma[output_names[i]]}",
            subst_name=f"subst_{i}",
            arguments=variables(f"{iname_f} {iname_e} {iname_j}"),
            terms_filter=lambda x: (get_dependencies(x) & knl.all_inames())
            <= {iname_f, iname_e, iname_j},
        )

    t_unit = t_unit.with_kernel(knl)

    for i in range(4):
        t_unit = lp.precompute(
            t_unit,
            f"subst_{i}",
            sweep_inames=[iname_f, iname_j],
            precompute_outer_inames=frozenset({iname_e}),
            default_tag=None,
            temporary_address_space=lp.AddressSpace.PRIVATE,
        )

    opt_einsum_t_unit = f.generate_loopy_with_opt_einsum_schedule(face_mass)

    assert (lp.get_op_map(t_unit, subgroup_size=1).eval_and_sum({"E": 1})) == (
        lp.get_op_map(opt_einsum_t_unit, subgroup_size=1).eval_and_sum({"E": 1})
    )


def test_hoist_reduction_invariant_terms(ctx_factory):
    from loopy.symbolic import Reduction
    from pymbolic import variables

    cl_ctx = ctx_factory()
    nel = 1
    ndim = 3
    ndofs = 35
    expr = f.einsum(
        "xre, rij, ej->xei",
        f.array("J", (ndim, ndim, nel)),
        f.array("R", (ndim, ndofs, ndofs)),
        f.array("u", (nel, ndofs)),
    )
    t_unit = f.generate_loopy(expr)
    sigma = f.match_t_unit_to_einsum(t_unit, expr)

    # {{{ hoist the "j" redn-loop over "x" loop

    e, i, j, r = sigma["e"], sigma["i"], sigma["j"], sigma["r"]
    knl = t_unit.default_entrypoint
    knl = lp.split_reduction_inward(knl, j)

    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(knl, j)
    knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
        knl,
        within=None,
        subst_name="grad_without_jacobi_subst",
        arguments=variables(f"{r} {i} {e}"),
        terms_filter=lambda x: isinstance(x, Reduction),
    )

    hoisted_t_unit = t_unit.with_kernel(knl)

    hoisted_t_unit = lp.precompute(
        hoisted_t_unit,
        "grad_without_jacobi_subst",
        sweep_inames=[r, i],
        precompute_outer_inames=frozenset({e}),
    )

    # }}}

    trivial_ops = lp.get_op_map(t_unit, subgroup_size=1).eval_and_sum()
    opt_ops = lp.get_op_map(hoisted_t_unit, subgroup_size=1).eval_and_sum()
    assert trivial_ops > 4 * opt_ops
    lp.auto_test_vs_ref(t_unit, cl_ctx, hoisted_t_unit)


def test_wave_grad_transform_knowledge_transfer(ctx_factory):
    from testlib import transform_3d_p4_grad

    cl_ctx = ctx_factory()

    t_unit = lp.make_kernel(
        "{[iel_1, idof_1, jdof_1, r_1, x_1]:"
        " 0<=iel_1<Nel and 0<=idof_1,jdof_1<35 and 0<=r_1,x_1<3}",
        """
        jac_subst(_0, _1, _2) := J[_0, _1, _2]
        D_subst(_0, _1, _2) := D[_0, _1, _2]
        u_subst(_0, _1) := u[_0, _1]

        grad_out[x_1, iel_1, idof_1] = sum([jdof_1, r_1], \
                                           jac_subst(x_1, iel_1, r_1)*D_subst(r_1, idof_1, jdof_1)*u_subst(iel_1, jdof_1))
        """,  # noqa: E501
    )
    t_unit = lp.add_dtypes(
        t_unit,
        {
            arg.name: np.float64
            for arg in t_unit.default_entrypoint.args
            if arg.is_input and arg.name != "Nel"
        },
    )
    ref_t_unit = t_unit
    t_unit = transform_3d_p4_grad(t_unit, "writes:grad_out")
    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit, parameters={"Nel": 100})


def test_einsum_matching():
    t_unit = lp.make_kernel(
        "{[iel_2, idof_2, ifacedof, iface]:"
        " 0<=iel_2<10000 and 0<=idof_2<35 and 0<=ifacedof<15 and 0<=iface<4}",
        """
        flux_subst_0(_0, _1, _2) := F_0[_0, _1, _2]
        flux_subst_1(_0, _1, _2) := F_1[_0, _1, _2]
        flux_subst_2(_0, _1, _2) := F_2[_0, _1, _2]
        flux_subst_3(_0, _1, _2) := F_3[_0, _1, _2]
        lift_subst(_0, _1, _2) := Rlift[_0, _1, _2]
        face_jac_subst(_0, _1) := Jface[_0, _1]

        lift_0[iel_2, idof_2] = sum([iface, ifacedof],
                                    face_jac_subst(iel_2, iface)*lift_subst(iface, idof_2, ifacedof)*flux_subst_0(iface, iel_2, ifacedof))
        lift_1[iel_2, idof_2] = sum([iface, ifacedof],
                                    face_jac_subst(iel_2, iface)*lift_subst(iface, idof_2, ifacedof)*flux_subst_1(iface, iel_2, ifacedof))
        lift_2[iel_2, idof_2] = sum([iface, ifacedof],
                                    face_jac_subst(iel_2, iface)*lift_subst(iface, idof_2, ifacedof)*flux_subst_2(iface, iel_2, ifacedof))
        lift_3[iel_2, idof_2] = sum([iface, ifacedof],
                                    face_jac_subst(iel_2, iface)*lift_subst(iface, idof_2, ifacedof)*flux_subst_3(iface, iel_2, ifacedof))
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

    nvoldofs = 35
    nfacedofs = 15
    nface = 4

    ref_einsum = f.batched_einsum(
        "ef, fij, fej -> ei",
        [
            [
                f.array("J", ("E", nface)),
                f.array("R", (nface, nvoldofs, nfacedofs)),
                f.array(f"v{i}", (nface, "E", nfacedofs)),
            ]
            for i in range(4)
        ],
    )

    inferred_einsum, _ = f.get_a_matched_einsum(t_unit)
    assert f.canonicalize_einsum(inferred_einsum) == f.canonicalize_einsum(
        ref_einsum
    )


def test_sum_redn_algebraic_transforms(ctx_factory):
    from loopy.symbolic import Reduction
    from pymbolic import variables

    t_unit = lp.make_kernel(
        "{[e,i,j,x,r]: 0<=e<N_e and 0<=i,j<35 and 0<=x,r<3}",
        """
        y[i] = sum([r,j], J[x, r, e]*D[r,i,j]*u[e,j])
        """,
        [lp.GlobalArg("J,D,u", dtype=np.float64, shape=lp.auto), ...],
    )
    knl = t_unit.default_entrypoint

    knl = lp.split_reduction_inward(knl, "j")
    knl = lp_utils.hoist_invariant_multiplicative_terms_in_sum_reduction(
        knl, reduction_inames="j"
    )
    knl = lp_utils.extract_multiplicative_terms_in_sum_reduction_as_subst(
        knl,
        within=None,
        subst_name="grad_without_jacobi_subst",
        arguments=variables("r i e"),
        terms_filter=lambda x: isinstance(x, Reduction),
    )

    transformed_t_unit = t_unit.with_kernel(knl)
    transformed_t_unit = lp.precompute(
        transformed_t_unit,
        "grad_without_jacobi_subst",
        sweep_inames=["r", "i"],
        precompute_outer_inames=frozenset({"e"}),
        temporary_address_space=lp.AddressSpace.PRIVATE,
    )

    x1 = lp.get_op_map(t_unit, subgroup_size=1).eval_and_sum({"N_e": 1})
    x2 = lp.get_op_map(transformed_t_unit, subgroup_size=1).eval_and_sum({"N_e": 1})

    assert x1 == 33075
    assert x2 == 7980  # i.e. this demonstrates a 4.14x reduction in flops
