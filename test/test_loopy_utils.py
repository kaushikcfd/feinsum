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
