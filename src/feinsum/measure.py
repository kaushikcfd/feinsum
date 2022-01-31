"""
.. currentmodule:: feinsum.measure

.. autofunction:: timeit
.. autofunction:: measure_giga_op_rate
.. autofunction:: pprint_comparison_vs_roofline
"""

import numpy as np
import numpy.typing as npt
import pyopencl as cl
import loopy as lp
import pyopencl.array as cla

from typing import Callable, Dict, Any
from pyrsistent.typing import PMap as PMapT
from pyrsistent import pmap
from feinsum.einsum import FusedEinsum, INT_CLASSES, ShapeT
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


def _get_flops_from_complex_ops(op_map: "lp.ToCountMap",
                                complex_dtype: npt.DTypeLike,
                                kernel_name: str) -> int:

    complex_add = op_map.filter_by(dtype=[complex_dtype],
                                   name="add",
                                   kernel_name=kernel_name).eval_and_sum({})

    complex_mul = op_map.filter_by(dtype=[complex_dtype],
                                   name="mul",
                                   kernel_name=kernel_name).eval_and_sum({})

    complex_div = op_map.filter_by(dtype=[complex_dtype],
                                   name="div",
                                   kernel_name=kernel_name).eval_and_sum({})

    return (2 * complex_add  # type: ignore[no-any-return]
            + 6 * complex_mul
            + (6 + 3 + 2) * complex_div)


def _get_giga_ops_from_kernel(expr: FusedEinsum,
                              dtype: np.dtype[Any],
                              long_dim_length: int) -> float:
    from feinsum.codegen.loopy import generate_loopy
    from feinsum.einsum import contraction_schedule_from_opt_einsum, SizeParam
    from feinsum.make_einsum import array
    import opt_einsum

    _, path_info = opt_einsum.contract_path(expr.get_subscripts(),
                                            *[array([long_dim_length
                                                     if isinstance(dim, SizeParam)
                                                     else dim
                                                     for dim in arg_shape],
                                                    np.float64)
                                              for arg_shape in expr.arg_shapes],
                                            optimize="optimal",
                                            use_blas=False)

    t_unit = generate_loopy(expr,
                            contraction_schedule_from_opt_einsum(path_info))

    t_unit = lp.fix_parameters(t_unit, **{name: long_dim_length
                                          for name in (t_unit
                                                       .default_entrypoint
                                                       .all_params())})

    kernel = t_unit.default_entrypoint
    kernel = kernel.copy(silenced_warnings=(kernel.silenced_warnings
                                            + ["insn_count_subgroups_upper_bound",
                                               "summing_if_branches_ops"]))
    t_unit = t_unit.with_kernel(kernel)
    dtype = np.dtype(dtype)
    op_map = lp.get_op_map(t_unit, subgroup_size=1)
    extra_ops = 0

    if dtype == np.dtype(np.float32):
        extra_ops += _get_flops_from_complex_ops(op_map, np.complex64, kernel.name)
    elif dtype == np.dtype(np.float64):
        extra_ops += _get_flops_from_complex_ops(op_map, np.complex128, kernel.name)
    else:
        pass

    return ((op_map.filter_by(dtype=[dtype.type],  # type: ignore[no-any-return]
                              kernel_name=kernel.name).eval_and_sum({})
             + extra_ops) * 1e-9)


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
                         ) -> float:
    """
    Returns the arithmetic operations rate (in Giga Ops per second) for
    arithmetic operations involving *dtype*.
    """
    runtime = timeit(expr,
                     transform=transform,
                     cl_ctx=cl_ctx,
                     long_dim_length=long_dim_length)

    return _get_giga_ops_from_kernel(expr,
                                     dtype=np.dtype(dtype),
                                     long_dim_length=long_dim_length) / runtime


def pprint_comparison_vs_roofline(expr: FusedEinsum,
                                  *,
                                  transform: Callable[[lp.TranslationUnit],
                                                      lp.TranslationUnit],
                                  cl_ctx: cl.Context,
                                  dtype: npt.DTypeLike = "float64",
                                  long_dim_length: int = 100000,) -> None:
    """
    Pretty prints the comparison of *expr* transformed with *transform* wrt
    roofline. The roofline model assumes that kernel's performance is limited
    by either the device's global memory bandwidth or the device's floating
    point units being saturated to their maximum throughput.
    """
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("tabulate is need for pretty printing."
                          " Install via `pip install tabulate`.")

    from feinsum.data.device_info import (DEV_TO_PEAK_F64_GFLOPS,
                                          DEV_TO_PEAK_BW)
    dev, = cl_ctx.devices

    ngflops = _get_giga_ops_from_kernel(expr, np.dtype(dtype), long_dim_length)
    ngbs = _get_footprint_gbytes(expr, long_dim_length)
    roofline_flops = (ngflops
                      / max(ngflops/DEV_TO_PEAK_F64_GFLOPS[dev.name],
                            ngbs/DEV_TO_PEAK_BW[dev.name]))

    measured_flops = measure_giga_op_rate(expr,
                                          transform=transform,
                                          cl_ctx=cl_ctx,
                                          dtype=dtype,
                                          long_dim_length=long_dim_length)

    table = [["Current Transform", "Roofline"],
             [f"{measured_flops:.1f}", f"{roofline_flops:.1f}"]]
    print(tabulate(table, tablefmt="fancy_grid"))
