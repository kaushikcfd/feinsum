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


import numpy as np
from pyopencl.tools import (  # noqa
        pytest_generate_tests_for_pyopencl as pytest_generate_tests)
import feinsum as f
import loopy as lp
import pytest


def test_simple_matvec(ctx_factory):
    cl_ctx = ctx_factory()
    expr = f.fused_einsum("ij, j -> i",
                        ((10, 4), (4,)),
                        dtypes="float32",
                        use_matrix=[[{"I0", "I1"}, {"I3", "I2"}],
                                    [{"I1", "I4"}, {"I2"}]]
                        )

    f.timeit(expr, transform=lambda x: x, cl_ctx=cl_ctx)


def test_matvec_with_long_dim_result(ctx_factory):
    cl_ctx = ctx_factory()
    expr = f.fused_einsum("ij, j -> i",
                        ((np.inf, 4), (4,)),
                        dtypes="float32",
                        use_matrix=[[{"I0", "I1"}, {"I3", "I2"}],
                                    [{"I1", "I4"}, {"I2"}]]
                        )

    f.timeit(expr, transform=lambda x: x, cl_ctx=cl_ctx)


def test_pprint_roofline_comparison(ctx_factory):
    from feinsum.data.device_info import DEV_TO_PEAK_F64_GFLOPS
    cl_ctx = ctx_factory()
    if len(cl_ctx.devices) != 1:
        pytest.skip("Multiple devices in the context")
    if cl_ctx.devices[0].name not in DEV_TO_PEAK_F64_GFLOPS:
        pytest.skip("Device not known.")

    Ndim = 3
    Ndofs = 35
    expr = f.einsum("xre,rij,ej->xei",
                    f.array((Ndim, Ndim, np.inf,),
                            "float64"),
                    f.array((Ndim, Ndofs, Ndofs),
                            "float64"),
                    f.array((np.inf, Ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])

    f.pprint_comparison_vs_roofline(
        expr,
        cl_ctx=cl_ctx,
        transform=lambda x: lp.split_iname(x, "e", 32,
                                           outer_tag="g.0", inner_tag="l.0"),
        long_dim_length=500)
