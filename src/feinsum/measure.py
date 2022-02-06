"""
.. currentmodule:: feinsum.measure

.. autofunction:: timeit
.. autofunction:: measure_giga_op_rate
.. autofunction:: stringify_comparison_vs_roofline
"""

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import pymbolic.primitives as prim
import loopy as lp
import pyopencl.array as cla

from typing import Callable, Dict, Any
from pyrsistent.typing import PMap as PMapT
from pyrsistent import pmap
from feinsum.einsum import FusedEinsum, INT_CLASSES, ShapeT, SizeParam
from more_itertools import zip_equal as zip
import logging
logger = logging.getLogger(__name__)


N_WARMUP_ROUNDS = 20
N_MIN_TIMING_ROUNDS = 500
N_MIN_SIM_SECS = 2


def generate_out_arrays(queue: cl.CommandQueue,
                        t_unit: "lp.TranslationUnit"
                        ) -> PMapT[str, cla.Array]:
    t_unit = lp.preprocess_kernel(t_unit)
    knl = t_unit.default_entrypoint

    out_buffers = {}

    for arg in knl.args:
        if arg.is_output:
            assert all(isinstance(dim, INT_CLASSES) for dim in arg.shape)
            out_buffers[arg.name] = cla.empty(queue,
                                              shape=arg.shape,
                                              dtype=arg.dtype)

    return pmap(out_buffers)


def generate_input_arrays(queue: cl.CommandQueue,
                          einsum: FusedEinsum,
                          long_dim_length: int,
                          np_seed: int = 0,
                          ) -> PMapT[str, cla.Array]:
    from numpy.random import default_rng

    # {{{ compute val_to_shape

    val_to_shape: Dict[str, ShapeT] = {}

    for use_row in einsum.use_matrix:
        for values, op_shape in zip(use_row, einsum.arg_shapes):
            op_shape = tuple(dim if isinstance(dim, INT_CLASSES) else long_dim_length
                             for dim in op_shape)
            for val in values:
                if val in val_to_shape:
                    assert val_to_shape[val] == op_shape
                else:
                    val_to_shape[val] = op_shape
    # }}}

    rng = default_rng(np_seed)

    return pmap({name: cla.to_device(queue,
                                     rng.random(size=val_to_shape[name],
                                                dtype=dtype)
                                     )
                 for name, dtype in einsum.value_to_dtype.items()
                 })


def timeit(einsum: FusedEinsum,
           *,
           transform: Callable[[lp.TranslationUnit],
                               lp.TranslationUnit],
           cl_ctx: cl.Context,
           long_dim_length: int = 100000
           ) -> float:
    """
    Returns the runtime in seconds for executing *einsum* on OpenCL context
    *cl_ctx*.

    :param transform: The transformation to be applied to
        :class:`loopy.TranslationUnit` lowered from *einsum*.
    """
    from time import time
    from feinsum.codegen.loopy import generate_loopy

    cq = cl.CommandQueue(cl_ctx)

    t_unit = generate_loopy(einsum)
    t_unit = lp.fix_parameters(t_unit, **{name: long_dim_length
                                          for name in (t_unit
                                                       .default_entrypoint
                                                       .all_params())})

    t_unit = lp.set_options(t_unit, "no_numpy")

    param_dict = generate_input_arrays(cq, einsum, long_dim_length)
    out_dict = generate_out_arrays(cq, t_unit)

    t_unit = transform(t_unit)

    arg_dict = param_dict.update(out_dict)

    # {{{ WARMUP

    for _ in range(N_WARMUP_ROUNDS):
        evt, _ = t_unit(cq, **arg_dict)

    cq.finish()

    # }}}

    total_sim_time = 0.0
    total_rounds = 0

    while ((total_rounds < N_MIN_TIMING_ROUNDS)
           or (total_sim_time < N_MIN_SIM_SECS)):

        evt.wait()

        clock_start = time()

        for _ in range(10):
            evt, _ = t_unit(cq, **arg_dict)

        evt.wait()
        clock_end = time()

        total_sim_time += (clock_end - clock_start)
        total_rounds += 10

    # TODO: verify the results in out_dict, with ref_solution

    return total_sim_time / total_rounds


def _get_giga_ops_from_einsum(expr: FusedEinsum) -> PMapT[np.dtype[Any],
                                                          prim.Expression]:
    from feinsum.codegen.loopy import generate_loopy_with_opt_einsum_schedule

    t_unit = generate_loopy_with_opt_einsum_schedule(expr,
                                                     use_blas=False,
                                                     optimize="optimal")

    kernel = t_unit.default_entrypoint
    kernel = kernel.copy(silenced_warnings=(kernel.silenced_warnings
                                            + ["insn_count_subgroups_upper_bound",
                                               "summing_if_branches_ops"]))
    t_unit = t_unit.with_kernel(kernel)
    op_map = lp.get_op_map(t_unit, subgroup_size=1)
    new_op_map: Dict[np.dtype[Any], prim.Expression] = {}

    for dtype in {op.dtype.numpy_dtype for op in op_map.keys()}:
        if dtype.kind == "c":
            c_ops = {op_type: op_map.filter_by(dtype=[dtype],
                                               name="add",
                                               kernel_name=kernel.name)
                     for op_type in ["add", "mul", "div"]}

            ops = (2 * c_ops["add"] + 6 * c_ops["mul"] + (6 + 3 + 2) * c_ops["div"])
            dtype = np.empty(0, dtype=dtype).real.dtype
        else:
            ops = op_map.filter_by(dtype=[dtype], kernel_name=kernel.name)

        (_, qpoly), = ops.sum().pwqpolynomial.get_pieces()

        from loopy.symbolic import qpolynomial_to_expr

        new_op_map.setdefault(dtype, 0)
        new_op_map[dtype] = (new_op_map[dtype] + qpolynomial_to_expr(qpoly) * 1e-9)

    return pmap(new_op_map)


def _get_footprint_gbytes(expr: FusedEinsum, long_dim_length: int) -> float:
    from feinsum.codegen.loopy import generate_loopy
    from pytools import product

    t_unit = generate_loopy(expr)
    t_unit = lp.fix_parameters(t_unit, **{name: long_dim_length
                                          for name in (t_unit
                                                       .default_entrypoint
                                                       .all_params())})
    t_unit = lp.infer_unknown_types(t_unit)
    kernel = t_unit.default_entrypoint

    return sum(product(arg.shape) * arg.dtype.itemsize
               for arg in kernel.args) * 1e-9


def measure_giga_op_rate(expr: FusedEinsum,
                         *,
                         transform: Callable[[lp.TranslationUnit],
                                             lp.TranslationUnit],
                         cl_ctx: cl.Context,
                         dtype: npt.DTypeLike = "float64",
                         long_dim_length: int = 100000,
                         ) -> PMapT[np.dtype[Any], float]:
    """
    Returns the arithmetic operations rate (in Giga Ops per second) for
    arithmetic operations involving *dtype*.
    """
    runtime = timeit(expr,
                     transform=transform,
                     cl_ctx=cl_ctx,
                     long_dim_length=long_dim_length)

    from pymbolic.mapper.evaluator import evaluate_to_float
    eval_context = {dim.name: long_dim_length
                    for dim in expr.index_to_dim_length().values()
                    if isinstance(dim, SizeParam)}
    return pmap({k: evaluate_to_float(v, eval_context)/runtime
                 for k, v in _get_giga_ops_from_einsum(expr).items()})


def stringify_comparison_vs_roofline(expr: FusedEinsum,
                                     *,
                                     transform: Callable[[lp.TranslationUnit],
                                                         lp.TranslationUnit],
                                     cl_ctx: cl.Context,
                                     long_dim_length: int = 100000) -> str:
    """
    Returns the prettified comparison of *expr* transformed with *transform*
    wrt roofline. The roofline model assumes that kernel's performance is
    limited by either the device's global memory bandwidth or the device's
    floating point units being saturated to their maximum throughput.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("tabulate is need for pretty printing."
                          " Install via `pip install tabulate`.")

    from feinsum.data.device_info import (DEV_TO_PEAK_GFLOPS,
                                          DEV_TO_PEAK_BW)
    from pymbolic.mapper.evaluator import evaluate_to_float

    dev, = cl_ctx.devices

    giga_op_map = _get_giga_ops_from_einsum(expr)

    if len(giga_op_map) > 1:
        raise ValueError("Cannot evaluate the FlOp-rate"
                         " for an einsum inolving multiple dtypes.")
    (dtype, giga_ops_expr), = giga_op_map.items()

    ngflops = evaluate_to_float(giga_ops_expr,
                                {dim.name: long_dim_length
                                 for dim in expr.index_to_dim_length().values()
                                 if isinstance(dim, SizeParam)})

    ngbs = _get_footprint_gbytes(expr, long_dim_length)
    roofline_flops = (ngflops
                      / max(ngflops/DEV_TO_PEAK_GFLOPS[dev.name][dtype.name],
                            ngbs/DEV_TO_PEAK_BW[dev.name]))

    measured_flops = measure_giga_op_rate(expr,
                                          transform=transform,
                                          cl_ctx=cl_ctx,
                                          dtype=dtype,
                                          long_dim_length=long_dim_length)

    table = [["Current Transform", "Roofline"],
             [f"{measured_flops[dtype]:.1f}", f"{roofline_flops:.1f}"]]
    return tabulate(table, tablefmt="fancy_grid")
