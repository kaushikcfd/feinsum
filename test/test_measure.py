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
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

import feinsum as f


def test_simple_matvec(ctx_factory):
    cl_ctx = ctx_factory()
    A = f.array("A", (10, 4), "float32")
    x = f.array("x", 4, "float32")
    y = f.array("y", 4, "float32")
    expr = f.batched_einsum("ij, j -> i", [[A, x], [A, y]])

    f.timeit(
        expr, transform=lambda t_unit, insn_match, kernel_name: t_unit, cl_ctx=cl_ctx
    )


def test_matvec_with_long_dim_result(ctx_factory):
    cl_ctx = ctx_factory()
    A = f.array("A", ("I", 4), "float32")
    x = f.array("x", 4, "float32")
    y = f.array("y", 4, "float32")
    expr = f.batched_einsum("ij, j -> i", [[A, x], [A, y]])

    f.timeit(
        expr, transform=lambda t_unit, insn_match, kernel_name: t_unit, cl_ctx=cl_ctx
    )


def test_pprint_roofline_comparison(ctx_factory):
    cl_ctx = ctx_factory()

    Ndim = 3
    Ndofs = 35
    expr = f.einsum(
        "xre,rij,ej->xei",
        f.array("J", (Ndim, Ndim, "E")),
        f.array("R", (Ndim, Ndofs, Ndofs)),
        f.array("u", ("E", Ndofs)),
    )

    f.stringify_comparison_vs_roofline(
        expr,
        cl_ctx=cl_ctx,
        transform=(
            lambda t_unit, insn_match, kernel_name: lp.split_iname(
                t_unit,
                f.match_t_unit_to_einsum(t_unit, expr)["e"],
                32,
                outer_tag="g.0",
                inner_tag="l.0",
            )
        ),
        long_dim_length=500,
        ignore_unknown_device=True,  # specs of CI machines is typically not known
    )
