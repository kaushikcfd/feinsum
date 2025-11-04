from typing import Any

import loopy as lp
import numpy as np
from pyopencl.tools import (  # noqa
    pytest_generate_tests_for_pyopencl as pytest_generate_tests,
)

import feinsum as f
from feinsum.tuning import IntParameter, transform_param


@transform_param("wg_size", lambda ensm: (IntParameter(8, 16), IntParameter(8, 16)))
def transform(
    t_unit: lp.TranslationUnit,
    wg_size: tuple[int, int],
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    ref_einsum = f.einsum("ijk->ij", f.array("A", ("I", 72, 4), np.float64))
    subst_map = f.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match, kernel_name=kernel_name
    )
    i = subst_map["i"]
    j = subst_map["j"]

    l_0_size, l_1_size = wg_size
    assert isinstance(wg_size, tuple)
    assert 8 <= l_0_size <= 16 and 8 <= l_1_size <= 16

    t_unit = lp.split_iname(t_unit, i, l_1_size, inner_tag="l.1", outer_tag="g.0")
    t_unit = lp.split_iname(t_unit, j, l_0_size, inner_tag="l.0")

    return t_unit


def test_transform(ctx_factory):
    import os

    cl_ctx = ctx_factory()

    expr = f.einsum("ijk->ij", f.array("P", ("I", 72, 4), np.float64))
    f.autotune(expr, os.path.abspath(__file__), cl_ctx, stop_after=3,
               long_dim_length=100)
