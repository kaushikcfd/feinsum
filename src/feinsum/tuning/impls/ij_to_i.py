from typing import Any

import loopy as lp
import numpy as np

import feinsum as fnsm
from feinsum.tuning import IntParameter


@fnsm.tuning.einsum_arg(
    "j_len", lambda e: e.index_to_dim_length()[fnsm.SummationAxis(0)]
)
@fnsm.tuning.transform_param("l_0_size", lambda e: IntParameter(32, 32))
def transform(
    t_unit: lp.TranslationUnit,
    j_len: int,
    l_0_size: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    if l_0_size > 500:
        raise fnsm.InvalidParameterError(
            "Work-group size constraints for the" " hardware."
        )
    ref_einsum = fnsm.einsum("ij->i", fnsm.array((np.inf, j_len), np.float64))
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, kernel_name=kernel_name, insn_match=insn_match
    )
    i = subst_map["i"]
    t_unit = lp.split_iname(t_unit, i, l_0_size, inner_tag="l.0", outer_tag="g.0")

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    n_j = 4
    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum("ij->i", fnsm.array((np.inf, n_j), "float64"))

    fnsm.autotune(expr, os.path.abspath(__file__), cl_ctx, long_dim_length=200_000)
