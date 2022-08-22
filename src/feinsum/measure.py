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

from typing import Callable, Dict, Any, Optional, Mapping, Tuple
from pyrsistent.typing import PMap as PMapT
from pyrsistent import pmap
from feinsum.einsum import (FusedEinsum, INT_CLASSES, SizeParam,
                            ContractionSchedule, IntegralT)
from more_itertools import zip_equal as zip
import logging
logger = logging.getLogger(__name__)


N_WARMUP_ROUNDS = 20
N_MIN_TIMING_ROUNDS = 500
N_MIN_SIM_SECS = 2


def get_real_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    return np.empty(0, dtype=dtype).real.dtype


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


def _generate_random_np_array(rng: "np.random._generator.Generator",
                              dtype: np.dtype[Any],
                              shape: Tuple[IntegralT, ...]
                              ) -> npt.NDArray[Any]:
    if dtype.kind == "c":
        real_dtype = get_real_dtype(dtype)
        # type-ignored because numpy addition types not quite precise
        return (rng.random(size=shape,  # type: ignore[no-any-return]
                           dtype=real_dtype)
                + dtype.type(1j) * rng.random(size=shape,
                                              dtype=real_dtype))
    else:
        return rng.random(size=shape, dtype=dtype)


def generate_input_arrays(queue: cl.CommandQueue,
                          einsum: FusedEinsum,
                          long_dim_length: int,
                          np_seed: int = 0,
                          ) -> PMapT[str, cla.Array]:
    from numpy.random import default_rng

    # {{{ compute val_to_shape

    val_to_shape: Dict[str, Tuple[IntegralT, ...]] = {}

    for use_row in einsum.use_matrix:
        for values, op_shape in zip(use_row, einsum.arg_shapes):
            # concrete_op_shape: shape after getting rid of SizeParams
            concrete_op_shape: Tuple[IntegralT, ...] = tuple(
                dim if isinstance(dim, INT_CLASSES) else long_dim_length
                for dim in op_shape)
            for val in values:
                if val in val_to_shape:
                    assert val_to_shape[val] == concrete_op_shape
                else:
                    val_to_shape[val] = concrete_op_shape
    # }}}

    rng = default_rng(np_seed)

    return pmap({name: cla.to_device(queue,
                                     _generate_random_np_array(rng,
                                                               dtype,
                                                               val_to_shape[name]))
                 for name, dtype in einsum.value_to_dtype.items()
                 })


def validate_fused_einsum_transform(einsum: FusedEinsum,
                                    cl_ctx: cl.Context,
                                    transform: Callable[[lp.TranslationUnit],
                                                         lp.TranslationUnit],
                                    schedule: Optional[ContractionSchedule] = None,
                                    ) -> None:
    """
    If the :class:`loopy.LoopKernel` generated from *einsum* does not replicate
    the results after being transformed with *transform*, then a
    :class:`RuntimeError` is raised.
    """

    from feinsum.codegen.loopy import generate_loopy

    cq = cl.CommandQueue(cl_ctx)

    ref_t_unit = generate_loopy(einsum, schedule=schedule)
    ref_t_unit = lp.set_options(ref_t_unit, no_numpy=True, return_dict=True)
    long_dim_length = 30

    param_dict = generate_input_arrays(cq, einsum, long_dim_length)
    out_dict = generate_out_arrays(
        cq,
        lp.fix_parameters(ref_t_unit, **{name: long_dim_length
                                         for name in (ref_t_unit
                                                      .default_entrypoint
                                                      .all_params())}))

    t_unit = transform(ref_t_unit)
    arg_dict = param_dict.update(out_dict)

    _, ref_outs = t_unit(cq, **arg_dict)
    _, transform_outs = t_unit(cq, **arg_dict)

    if frozenset(ref_outs.keys()) != frozenset(transform_outs.keys()):
        raise RuntimeError("Output names mismatch")

    for name, ref_out in sorted(ref_outs.items()):
        ref_out_np = ref_out.get()
        transform_out_np = transform_outs[name].get()
        dtype = ref_out_np.dtype
        if dtype != transform_out_np.dtype:
            raise RuntimeError(f"dtype mismatch for output '{name}'")

        real_dtype = get_real_dtype(dtype)

        if real_dtype == np.float32:
            atol = 1e-6
            rtol = 1e-6
        elif real_dtype == np.float64:
            atol = 1e-14
            rtol = 1e-14
        else:
            raise NotImplementedError(real_dtype)

        np.testing.assert_allclose(ref_out_np, transform_out_np,
                                   atol=atol, rtol=rtol)

    logger.info("Statistically verified the soundness of the transformation")


def timeit(einsum: FusedEinsum,
           *,
           transform: Callable[[lp.TranslationUnit],
                               lp.TranslationUnit],
           cl_ctx: cl.Context,
           long_dim_length: int = 100000,
           schedule: Optional[ContractionSchedule] = None
           ) -> float:
    """
    Returns the runtime in seconds for executing *einsum* on OpenCL context
    *cl_ctx*.

    :param transform: The transformation to be applied to
        :class:`loopy.TranslationUnit` lowered from *einsum*.
    """
    from time import time
    from feinsum.codegen.loopy import generate_loopy

    # Validate the transformation before fusing it
    validate_fused_einsum_transform(einsum, cl_ctx, transform, schedule)

    cq = cl.CommandQueue(cl_ctx)

    t_unit = generate_loopy(einsum, schedule=schedule)
    t_unit = lp.set_options(t_unit, no_numpy=True, return_dict=True)

    param_dict = generate_input_arrays(cq, einsum, long_dim_length)
    out_dict = generate_out_arrays(
        cq,
        lp.fix_parameters(t_unit, **{name: long_dim_length
                                     for name in (t_unit
                                                  .default_entrypoint
                                                  .all_params())}))

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

    return total_sim_time / total_rounds


def _get_giga_ops_from_einsum(expr: FusedEinsum) -> PMapT[np.dtype[Any],
                                                          prim.Expression]:
    from feinsum.codegen.loopy import generate_loopy_with_opt_einsum_schedule
    from loopy.symbolic import qpolynomial_to_expr

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
                                               name=op_type,
                                               kernel_name=kernel.name)
                     for op_type in ["add", "mul", "div"]}

            pwqpoly = (2 * c_ops["add"].sum()
                       + 6 * c_ops["mul"].sum()
                       + (6 + 3 + 2) * c_ops["div"].sum()).pwqpolynomial
            dtype = get_real_dtype(dtype)
        else:
            pwqpoly = op_map.filter_by(dtype=[dtype],
                                       kernel_name=kernel.name
                                       ).sum().pwqpolynomial

        new_op_map.setdefault(dtype, 0)

        if pwqpoly.n_piece() > 0:
            (_, qpoly), = pwqpoly.get_pieces()
            new_op_map[dtype] = (new_op_map[dtype]
                                 + qpolynomial_to_expr(qpoly) * 1e-9)

    return pmap(new_op_map)


def _get_footprint_gbytes(expr: FusedEinsum, long_dim_length: int) -> float:
    from feinsum.codegen.loopy import generate_loopy

    t_unit = generate_loopy(expr)
    t_unit = lp.fix_parameters(t_unit, **{name: long_dim_length
                                          for name in (t_unit
                                                       .default_entrypoint
                                                       .all_params())})
    t_unit = lp.infer_unknown_types(t_unit)
    kernel = t_unit.default_entrypoint

    # TODO: mypy is right arg.shape can be 'Any' expression
    return sum(  # type: ignore[no-any-return]
        np.product(arg.shape) * arg.dtype.itemsize
        for arg in kernel.args) * 1e-9


def measure_giga_op_rate(expr: FusedEinsum,
                         *,
                         transform: Callable[[lp.TranslationUnit],
                                             lp.TranslationUnit],
                         cl_ctx: cl.Context,
                         long_dim_length: int = 100000,
                         schedule: Optional[ContractionSchedule] = None
                         ) -> PMapT[np.dtype[Any], float]:
    """
    Returns the arithmetic operations rate (in Giga Ops per second) by
    arithmetic operation's result dtypes.
    """
    runtime = timeit(expr,
                     transform=transform,
                     cl_ctx=cl_ctx,
                     long_dim_length=long_dim_length,
                     schedule=schedule)

    from pymbolic.mapper.evaluator import evaluate_to_float
    eval_context = {dim.name: long_dim_length
                    for dim in expr.index_to_dim_length().values()
                    if isinstance(dim, SizeParam)}
    return pmap({k: evaluate_to_float(v, eval_context)/runtime
                 for k, v in _get_giga_ops_from_einsum(expr).items()})


def get_roofline_flop_rate(expr: FusedEinsum, dev_name: str,
                           long_dim_length: int = 100_000
                           ) -> PMapT[np.dtype[Any], float]:

    from feinsum.data.device_info import (DEV_TO_PEAK_GFLOPS,
                                          DEV_TO_PEAK_BW)
    from pymbolic.mapper.evaluator import evaluate_to_float

    dtype_to_gflops_expr = _get_giga_ops_from_einsum(expr)
    ngbs = _get_footprint_gbytes(expr, long_dim_length)

    dtype_to_gflops = {dtype: evaluate_to_float(giga_ops_aff,
                                                {dim.name: long_dim_length
                                                 for dim in (expr
                                                             .index_to_dim_length()
                                                             .values())
                                                 if isinstance(dim, SizeParam)})
                       for dtype, giga_ops_aff in dtype_to_gflops_expr.items()}
    roofline_time_due_to_flops = max(ngflops/DEV_TO_PEAK_GFLOPS[dev_name][dtype.name]
                                     for dtype, ngflops in dtype_to_gflops.items())
    roofline_time_due_to_global_bw = ngbs/DEV_TO_PEAK_BW[dev_name]
    roofline_time = max(roofline_time_due_to_flops, roofline_time_due_to_global_bw)

    return pmap({dtype: gflops/roofline_time
                 for dtype, gflops in dtype_to_gflops.items()})


def _strify_measured_vs_roofline(measured_flop_rate: Mapping[np.dtype[Any], float],
                                 roofline_flop_rate: Mapping[np.dtype[Any], float]
                                 ) -> str:
    try:
        from tabulate import tabulate
    except ImportError:
        raise ImportError("`tabulate` is need for pretty printing."
                          " Install via `pip install tabulate`.")
    assert measured_flop_rate.keys() == roofline_flop_rate.keys()
    perf_table = [["Dtype", "Measured GOps/s", "Roofline GOps/s"]]
    for dtype in sorted(measured_flop_rate.keys(),
                        key=lambda x: x.itemsize):
        perf_table.append([dtype.name,
                           f"{(measured_flop_rate[dtype]):.1f}",
                           f"{(roofline_flop_rate[dtype]):.1f}"])

    return tabulate(perf_table, tablefmt="fancy_grid")


def stringify_comparison_vs_roofline(expr: FusedEinsum,
                                     *,
                                     schedule: Optional[ContractionSchedule] = None,
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

    dev, = cl_ctx.devices

    roofline_flop_rate = get_roofline_flop_rate(expr, dev.name)

    measured_flop_rate = measure_giga_op_rate(expr,
                                              transform=transform,
                                              schedule=schedule,
                                              cl_ctx=cl_ctx,
                                              long_dim_length=long_dim_length)
    return _strify_measured_vs_roofline(measured_flop_rate, roofline_flop_rate)
