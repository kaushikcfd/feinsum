__copyright__ = """Copyright (C) 2021 Kaushik Kulkarni"""

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


import loopy as lp
import numpy as np
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

import feinsum as f


def test_wave_div_components(ctx_factory):
    cl_ctx = ctx_factory()

    Ndofs = 35
    Ndim = 3
    wave_div_components = f.batched_einsum(
        "se, sij, ej -> ei",
        [
            (
                Ndim,
                np.inf,
            ),
            (Ndim, Ndofs, Ndofs),
            (np.inf, Ndofs),
        ],
        dtypes="float64",
        use_matrix=[
            [{"Jx"}, {"R"}, {"ux"}],
            [{"Jy"}, {"R"}, {"uy"}],
            [{"Jz"}, {"R"}, {"uz"}],
        ],
    )
    f.timeit(
        wave_div_components,
        transform=lambda t_unit, insn_match, kernel_name: t_unit,
        cl_ctx=cl_ctx,
        long_dim_length=300,
    )


def test_wave_face_mass(ctx_factory):
    cl_ctx = ctx_factory()

    Ndofs = 15
    Nface = 4

    wave_face_mass = f.batched_einsum(
        "se, sij, ej -> ei",
        [
            (
                Nface,
                np.inf,
            ),
            (Nface, Ndofs, Ndofs),
            (np.inf, Ndofs),
        ],
        dtypes="float64",
        use_matrix=[
            [{"J"}, {"R"}, {"v0"}],
            [{"J"}, {"R"}, {"v1"}],
            [{"J"}, {"R"}, {"v2"}],
            [{"J"}, {"R"}, {"v3"}],
        ],
    )

    f.timeit(
        wave_face_mass,
        transform=lambda t_unit, insn_match, kernel_name: t_unit,
        cl_ctx=cl_ctx,
        long_dim_length=300,
    )


def test_wave_grad(ctx_factory):
    cl_ctx = ctx_factory()
    Ndofs = 35
    Ndim = 3

    wave_grad = f.batched_einsum(
        "xre,rij,ej->xei",
        [
            (
                Ndim,
                Ndim,
                np.inf,
            ),
            (Ndim, Ndofs, Ndofs),
            (np.inf, Ndofs),
        ],
        dtypes="float64",
        use_matrix=[
            [{"J"}, {"R"}, {"u"}],
        ],
    )

    f.timeit(
        wave_grad,
        transform=lambda t_unit, insn_match, kernel_name: t_unit,
        cl_ctx=cl_ctx,
        long_dim_length=300,
    )


def test_opt_einsum_contract_schedule(ctx_factory):
    Ndofs = 35
    Ndim = 3
    cl_ctx = ctx_factory()

    expr = f.einsum(
        "xre,rij,ej->xei",
        f.array(
            (
                Ndim,
                Ndim,
                np.inf,
            ),
            "float64",
        ),
        f.array((Ndim, Ndofs, Ndofs), "float64"),
        f.array((np.inf, Ndofs), "float64"),
        arg_names=["J", "R", "u"],
    )
    knl1 = f.generate_loopy(expr)
    knl2 = f.generate_loopy(expr, f.get_opt_einsum_contraction_schedule(expr))
    assert len(knl1.default_entrypoint.instructions) == 1
    assert len(knl2.default_entrypoint.instructions) == 2
    lp.auto_test_vs_ref(knl1, cl_ctx, knl2, parameters={"N_e": 5})


def test_opt_einsum_contract_schedule_shorthand(ctx_factory):
    Ndofs = 35
    Ndim = 3
    cl_ctx = ctx_factory()

    expr = f.einsum(
        "xre,rij,ej->xei",
        f.array(
            (
                Ndim,
                Ndim,
                np.inf,
            ),
            "float64",
        ),
        f.array((Ndim, Ndofs, Ndofs), "float64"),
        f.array((np.inf, Ndofs), "float64"),
        arg_names=["J", "R", "u"],
    )
    knl1 = f.generate_loopy(expr)
    knl2 = f.generate_loopy_with_opt_einsum_schedule(expr)
    knl1_flops = (
        lp.get_op_map(knl1, subgroup_size=1)
        .filter_by(dtype=[np.float64])
        .eval_and_sum({"N_e": 500_000})
    )
    knl2_flops = (
        lp.get_op_map(knl2, subgroup_size=1)
        .filter_by(dtype=[np.float64])
        .eval_and_sum({"N_e": 500_000})
    )
    assert knl2_flops < knl1_flops
    lp.auto_test_vs_ref(knl1, cl_ctx, knl2, parameters={"N_e": 5})
