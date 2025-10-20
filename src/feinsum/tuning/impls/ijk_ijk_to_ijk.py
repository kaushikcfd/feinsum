from typing import Any

import loopy as lp
import numpy as np

import feinsum as fnsm
from feinsum.tuning import IntParameter


@fnsm.tuning.einsum_arg("i_len", lambda e: e.shape[0])
@fnsm.tuning.einsum_arg("k_len", lambda e: e.shape[2])
@fnsm.tuning.transform_param("l_0_size", lambda e: IntParameter(1, e.shape[2]))
@fnsm.tuning.transform_param("l_1_size", lambda e: IntParameter(1, 32))
def transform(
    t_unit: lp.TranslationUnit,
    i_len: int,
    k_len: int,
    l_0_size: int,
    l_1_size: int,
    insn_match: Any | None = None,
    kernel_name: str | None = None,
) -> lp.TranslationUnit:
    if l_1_size * l_0_size > 550:
        raise fnsm.InvalidParameterError(
            "Work-group size constraints for the" " hardware."
        )
    ref_einsum = fnsm.einsum(
        "ijk,ijk->ijk",
        fnsm.array("A", (i_len, "Nj", k_len), np.float64),
        fnsm.array("B", (i_len, "Nj", k_len), np.float64),
    )
    subst_map = fnsm.match_t_unit_to_einsum(
        t_unit, ref_einsum, kernel_name=kernel_name, insn_match=insn_match
    )
    j, k = subst_map["j"], subst_map["k"]
    t_unit = lp.split_iname(t_unit, j, l_1_size, inner_tag="l.1", outer_tag="g.0")
    t_unit = lp.split_iname(t_unit, k, l_0_size, inner_tag="l.0", outer_tag="unr")

    return t_unit


if __name__ == "__main__":
    import os

    import pyopencl as cl

    n_i = 4
    n_k = 3
    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum(
        "ijk,ijk->ijk",
        fnsm.array("A", (n_i, "Nj", n_k), "float64"),
        fnsm.array("B", (n_i, "Nj", n_k), "float64"),
    )

    fnsm.autotune(expr, os.path.abspath(__file__), cl_ctx, long_dim_length=200_000)
