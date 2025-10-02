import logging

import loopy as lp
import numpy as np
import pyopencl as cl

import feinsum as f
from feinsum.tuning import IntParameter

logging.basicConfig(level="INFO")


@f.tuning.transform_param("block_size_div_32", lambda ensm: IntParameter(1, 5))
def transform(t_unit, block_size_div_32, insn_match=None, kernel_name=None):
    ref_einsum = f.batched_einsum(
        "ij,j->i",
        [(np.inf, 4), (4,)],
        dtypes="float64",
        use_matrix=[
            [{"a"}, {"c"}],
            [{"b"}, {"c"}],
        ],
    )
    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum)
    i = subst_map["i"]
    return lp.split_iname(
        t_unit, i, block_size_div_32 * 32, inner_tag="l.0", outer_tag="g.0"
    )


if __name__ == "__main__":
    import os

    cl_ctx = cl.create_some_context()

    expr = f.batched_einsum(
        "ij,j->i",
        [(np.inf, 4), (4,)],
        dtypes="float64",
        use_matrix=[
            [{"a"}, {"c"}],
            [{"b"}, {"c"}],
        ],
    )
    f.autotune(
        expr,
        os.path.abspath(__file__),
        cl_ctx,
        long_dim_length=500_000,
        stop_after=5,
    )
