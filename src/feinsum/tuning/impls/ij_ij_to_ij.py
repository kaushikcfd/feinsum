import feinsum as fnsm
import loopy as lp
import numpy as np

from feinsum.tuning import IntParameter
from typing import Any, Optional


@fnsm.tuning.einsum_arg("j_len", lambda e: e.shape[1])
@fnsm.tuning.transform_param("l_0_size", lambda e: IntParameter(1, e.shape[1]))
@fnsm.tuning.transform_param("l_1_size", lambda e: IntParameter(1, 32))
def transform(t_unit: lp.TranslationUnit,
              j_len: int, l_0_size: int, l_1_size: int,
              insn_match: Optional[Any] = None,
              kernel_name: Optional[str] = None) -> lp.TranslationUnit:
    if l_1_size * l_0_size > 550:
        raise fnsm.InvalidParameterError("Work-group size constraints for the"
                                         " hardware.")
    ref_einsum = fnsm.einsum("ij,ij->ij",
                             fnsm.array((np.inf, j_len), np.float64),
                             fnsm.array((np.inf, j_len), np.float64))
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum,
                                            kernel_name=kernel_name,
                                            insn_match=insn_match)
    i, j = subst_map["i"], subst_map["j"]
    t_unit = lp.split_iname(t_unit, i, l_1_size, inner_tag="l.1", outer_tag="g.0")
    t_unit = lp.split_iname(t_unit, j, l_0_size, inner_tag="l.0", outer_tag="unr")

    return t_unit


if __name__ == "__main__":
    import pyopencl as cl
    import os

    n_j = 4
    cl_ctx = cl.create_some_context()

    expr = fnsm.einsum("ij,ij->ij",
                       fnsm.array((np.inf, n_j), "float64"),
                       fnsm.array((np.inf, n_j), "float64"))

    fnsm. autotune(expr, os.path.abspath(__file__), cl_ctx,
                   long_dim_length=200_000)
