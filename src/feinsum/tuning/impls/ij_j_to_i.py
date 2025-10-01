import feinsum as fnsm
import loopy as lp
import numpy as np

from feinsum.tuning import IntParameter
from typing import Any, Optional


def _get_scale_dtype(einsum: fnsm.BatchedEinsum) -> np.dtype[Any]:
    assert len(einsum.use_matrix) == 1
    for arg_shape, uses in zip(einsum.arg_shapes, einsum.use_matrix[0]):
        if len(arg_shape) == 1:
            use, = uses
            return einsum.value_to_dtype[use]

    raise AssertionError("Should not reach here")


@fnsm.tuning.einsum_arg(
    "j_len", lambda e: e.index_to_dim_length()[fnsm.SummationAxis(0)])
@fnsm.tuning.einsum_arg(
    "arg_2_dtype", lambda e: _get_scale_dtype(e))
@fnsm.tuning.transform_param("l_0_size", lambda e: IntParameter(32, 32))
def transform(t_unit: lp.TranslationUnit,
              j_len: int, arg_2_dtype: np.dtype[Any],
              l_0_size: int,
              insn_match: Optional[Any] = None,
              kernel_name: Optional[str] = None) -> lp.TranslationUnit:
    if l_0_size > 500:
        raise fnsm.InvalidParameterError("Work-group size constraints for the"
                                         " hardware.")
    ref_einsum = fnsm.einsum("ij,j->i",
                             fnsm.array((np.inf, j_len), np.float64),
                             fnsm.array(j_len, arg_2_dtype))
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum,
                                            kernel_name=kernel_name,
                                            insn_match=insn_match)
    i = subst_map["i"]
    t_unit = lp.split_iname(t_unit, i, l_0_size, inner_tag="l.0", outer_tag="g.0")

    return t_unit


if __name__ == "__main__":
    import pyopencl as cl
    import os

    n_j = 35
    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum("ij,j->i",
                       fnsm.array((np.inf, n_j), "float64"),
                       fnsm.array(n_j, "int64"))

    fnsm. autotune(expr, os.path.abspath(__file__), cl_ctx,
                   long_dim_length=200_000)
