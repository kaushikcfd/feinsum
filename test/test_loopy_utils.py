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
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)

import feinsum as f
import loopy as lp
import numpy as np


def test_extract_subexpr_of_associative_op_as_subst(ctx_factory):
    from feinsum.loopy_utils import extract_subexpr_of_associative_op_as_subst

    nface = 4
    nvoldofs = 35
    nfacedofs = 15
    face_mass = f.fused_einsum("ef, fij, fej -> ei",
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
    t_unit = f.generate_loopy(face_mass)

    # {{{ prefetch 'J @ vec'

    knl = t_unit.default_entrypoint

    for i in range(4):
        knl = extract_subexpr_of_associative_op_as_subst(
            knl,
            f"subst_{i}(f, e, j)",
            f"J[e, f] * v{i}[f, e, j]")

    t_unit = t_unit.with_kernel(knl)

    for i in range(4):
        t_unit = lp.precompute(
            t_unit,
            f"subst_{i}",
            sweep_inames=["f", "j"],
            precompute_outer_inames=frozenset({"e"}),
            default_tag=None,
            temporary_address_space=lp.AddressSpace.PRIVATE)

    # }}}

    opt_einsum_t_unit = f.generate_loopy_with_opt_einsum_schedule(face_mass)

    assert ((lp.get_op_map(t_unit, subgroup_size=1)
             .eval_and_sum({"N_e": 1}))
            == (lp.get_op_map(opt_einsum_t_unit, subgroup_size=1)
                .eval_and_sum({"N_e": 1})))


def test_hoist_reduction_invariant_terms(ctx_factory):
    cl_ctx = ctx_factory()
    nel = 1
    ndim = 3
    ndofs = 35
    expr = f.fused_einsum("xre, rij, ej->xei",
                          ((ndim, ndim, nel),
                           (ndim, ndofs, ndofs),
                           (nel, ndofs)),
                          dtypes="float32",
                          use_matrix=[
                              [{"J"}, {"R"}, {"u"}]
                          ])
    t_unit = f.generate_loopy(expr)

    # {{{ hoist the "j" redn-loop over "x" loop

    hoisted_t_unit = lp.split_reduction_inward(t_unit, "j")
    hoisted_t_unit = f.hoist_reduction_invariant_terms(hoisted_t_unit, "j")
    hoisted_t_unit = f.extract_einsum_terms_as_subst(hoisted_t_unit,
                                                     "subst(r, e, i)",
                                                     "sum(j, R[r, i, j]*u[e, j])")

    hoisted_t_unit = lp.precompute(hoisted_t_unit,
                                   "subst",
                                   sweep_inames=["r", "i"],
                                   precompute_outer_inames=frozenset("e"))

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
        grad_out[x_1, iel_1, idof_1] = sum([jdof_1, r_1], \
                                           J[x_1, iel_1, r_1]*R[r_1, idof_1, jdof_1]*u[iel_1, jdof_1])
        """  # noqa: E501
    )
    t_unit = lp.add_dtypes(t_unit,
                           {arg.name: np.float64
                            for arg in t_unit.default_entrypoint.args
                            if arg.is_input and arg.name != "Nel"})
    ref_t_unit = t_unit
    t_unit = transform_3d_p4_grad(t_unit, "writes:grad_out")
    lp.auto_test_vs_ref(ref_t_unit, cl_ctx, t_unit, parameters={"Nel": 100})
    return t_unit


def test_match_einsum():
    t_unit = lp.make_kernel(
        "{[iel_2, idof_2, ifacedof, iface]:"
        " 0<=iel_2<10000 and 0<=idof_2<35 and 0<=ifacedof<15 and 0<=iface<4}",
        """
        lift_0[iel_2, idof_2] = sum([iface, ifacedof],
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_0[iface, iel_2, ifacedof])
        lift_1[iel_2, idof_2] = sum([iface, ifacedof],
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_1[iface, iel_2, ifacedof])
        lift_2[iel_2, idof_2] = sum([iface, ifacedof],
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_2[iface, iel_2, ifacedof])
        lift_3[iel_2, idof_2] = sum([iface, ifacedof],
                                    Jface[iel_2, iface]*Rlift[iface, idof_2, ifacedof]*F_3[iface, iel_2, ifacedof])
        """  # noqa: E501
    )

    t_unit = lp.add_dtypes(t_unit,
                        {arg.name: np.float64
                            for arg in t_unit.default_entrypoint.args
                            if arg.is_input})

    nvoldofs = 35
    nfacedofs = 15
    nface = 4
    ref_einsum = f.fused_einsum("ef, fij, fej -> ei",
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

    inferred_einsum = f.match_einsum(t_unit)
    assert f.normalize_einsum(inferred_einsum) == f.normalize_einsum(ref_einsum)
