import feinsum as f
import pyopencl as cl
import numpy as np
import logging
logging.basicConfig(level="INFO")


cl_ctx = cl.create_some_context()

expr = f.fused_einsum("ij,j->i",
                      [(np.inf, 4), (4, )],
                      dtypes="float64",
                      use_matrix=[
                          [{"a"}, {"c"}],
                          [{"b"}, {"c"}],
                      ])


transform_src = """import loopy as lp
import feinsum as f
import numpy as np

def transform(t_unit, insn_match=None):
    ref_einsum = f.fused_einsum("ij,j->i",
                                [(np.inf, 4), (4, )],
                                dtypes="float64",
                                use_matrix=[
                                    [{"a"}, {"c"}],
                                    [{"b"}, {"c"}],
                                ])
    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum)
    i = subst_map["i"]
    return lp.split_iname(t_unit, i, 32, inner_tag='l.0', outer_tag='g.0')
"""


f.record(expr, cl_ctx, transform_str=transform_src, authors="kk",
         long_dim_length=500_000)
