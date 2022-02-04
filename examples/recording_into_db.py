import feinsum as f
import pyopencl as cl
import numpy as np


cl_ctx = cl.create_some_context()

expr = f.fused_einsum("ij,j->i",
                      [(np.inf, 4), (4, )],
                      dtypes="float64",
                      use_matrix=[
                          [{"a", "b"}, {"c"}],
                          [{"b"},      {"c", "d"}],
                      ])


transform_src = """
import loopy as lp

def transform(t_unit):
    return lp.split_iname(t_unit, 'i', 32, inner_tag='l.0', outer_tag='g.0')
"""


f.record(expr, cl_ctx, transform_str=transform_src, authors="kk")
