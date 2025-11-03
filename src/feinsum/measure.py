"""
.. currentmodule:: feinsum.measure

.. autofunction:: timeit
.. autofunction:: measure_giga_op_rate
.. autofunction:: stringify_comparison_vs_roofline
"""

import logging
from collections.abc import Mapping
from typing import Any, cast

import loopy as lp
import numpy as np
import numpy.typing as npt
import pymbolic.primitives as prim
import pyopencl as cl
import pyopencl.array as cla
from immutables import Map

from feinsum.diagnostics import NoDevicePeaksInfoError
from feinsum.einsum import (
    INT_CLASSES,
    BatchedEinsum,
    ContractionSchedule,
    IntegralT,
)
from feinsum.typing import ToStr, TransformT

logger = logging.getLogger(__name__)


N_WARMUP_ROUNDS = 20
N_MIN_TIMING_ROUNDS = 500
N_MIN_SIM_SECS = 2


def get_real_dtype(dtype: np.dtype[Any]) -> np.dtype[Any]:
    return np.empty(0, dtype=dtype).real.dtype


def generate_out_arrays(
    queue: cl.CommandQueue, t_unit: lp.TranslationUnit
) -> Map[str, cla.Array]:
    t_unit = lp.preprocess_kernel(t_unit)
    knl = t_unit.default_entrypoint

    out_buffers = {}

    for arg in knl.args:
        if arg.is_output:
            assert all(isinstance(dim, INT_CLASSES) for dim in arg.shape)
            assert arg.dtype is not None
            out_buffers[arg.name] = cla.zeros(
                queue, shape=arg.shape, dtype=arg.dtype.numpy_dtype
            )

    return Map(out_buffers)


def _generate_random_np_array(
    rng: "np.random._generator.Generator",
    dtype: np.dtype[Any],
    shape: tuple[IntegralT, ...],
) -> npt.NDArray[Any]:
    if dtype.kind == "c":
        real_dtype = get_real_dtype(dtype)
        # type-ignored because numpy addition types not quite precise
        return rng.random(  # type: ignore[no-any-return]
            size=shape, dtype=real_dtype
        ) + dtype.type(1j) * rng.random(size=shape, dtype=real_dtype)
    elif dtype.kind == "i":
        return rng.integers(low=-100, high=100, size=shape, dtype=dtype)
    else:
        return rng.random(size=shape, dtype=dtype)


def generate_input_arrays(
    queue: cl.CommandQueue,
    einsum: BatchedEinsum,
    long_dim_length: int,
    np_seed: int = 0,
) -> Map[str, cla.Array]:
    from numpy.random import default_rng

    # {{{ compute val_to_shape

    val_to_shape: dict[str, tuple[IntegralT, ...]] = {}

    for arg, shape in einsum.arg_to_shape.items():
        val_to_shape[arg] = tuple(
            dim if isinstance(dim, INT_CLASSES) else long_dim_length for dim in shape
        )

    # }}}

    rng = default_rng(np_seed)

    return Map(
        {
            name: cla.to_device(
                queue, _generate_random_np_array(rng, dtype, val_to_shape[name])
            )
            for name, dtype in einsum.arg_to_dtype.items()
        }
    )


def validate_batched_einsum_transform(
    einsum: BatchedEinsum,
    cl_ctx: cl.Context,
    transform: TransformT,
    schedule: ContractionSchedule | None = None,
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
    long_dim_length = 100

    arg_dict: dict[str, cla.Array | int] = dict(
        generate_input_arrays(cq, einsum, long_dim_length)
    )
    arg_dict.update(
        dict.fromkeys(ref_t_unit.default_entrypoint.all_params(), long_dim_length)
    )

    t_unit = transform(ref_t_unit, insn_match=None, kernel_name=None)

    # {{{ get output buffers

    ref_outs = generate_out_arrays(
        cq,
        lp.fix_parameters(
            t_unit,
            within=None,
            **dict.fromkeys(t_unit.default_entrypoint.all_params(), long_dim_length),
        ),
    )
    transform_outs = Map(
        {name: cla.zeros_like(ary) for name, ary in ref_outs.items()}
    )

    # }}}

    # pylint-disable-reason: for some reason pylint thinks ref_t_unit is not callable
    evt, ref_outs = ref_t_unit(
        cq, **arg_dict, **ref_outs  # pylint: disable=not-callable
    )
    evt.wait()
    evt, transform_outs = t_unit(cq, **arg_dict, **transform_outs)
    evt.wait()

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

        np.testing.assert_allclose(
            transform_out_np, ref_out_np, atol=atol, rtol=rtol
        )

    logger.info("Statistically verified the soundness of the transformation")


def timeit(
    einsum: BatchedEinsum,
    *,
    transform: TransformT,
    cl_ctx: cl.Context,
    long_dim_length: int = 100000,
    schedule: ContractionSchedule | None = None,
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
    validate_batched_einsum_transform(einsum, cl_ctx, transform, schedule)

    cq = cl.CommandQueue(cl_ctx)

    t_unit = generate_loopy(einsum, schedule=schedule)
    t_unit = lp.set_options(t_unit, no_numpy=True, return_dict=True)

    param_dict = generate_input_arrays(cq, einsum, long_dim_length)
    out_dict = generate_out_arrays(
        cq,
        lp.fix_parameters(
            t_unit,
            within=None,
            **dict.fromkeys(t_unit.default_entrypoint.all_params(), long_dim_length),
        ),
    )

    t_unit = transform(t_unit, insn_match=None, kernel_name=None)

    arg_dict = param_dict.update(out_dict)

    # {{{ WARMUP

    for _ in range(N_WARMUP_ROUNDS):
        evt, _ = t_unit(cq, **arg_dict)

    cq.finish()

    # }}}

    total_sim_time = 0.0
    total_rounds = 0

    while (total_rounds < N_MIN_TIMING_ROUNDS) or (total_sim_time < N_MIN_SIM_SECS):

        evt.wait()

        clock_start = time()

        for _ in range(10):
            evt, _ = t_unit(cq, **arg_dict)

        evt.wait()
        clock_end = time()

        total_sim_time += clock_end - clock_start
        total_rounds += 10

    return total_sim_time / total_rounds


def _get_giga_ops_from_einsum(
    expr: BatchedEinsum,
) -> Map[np.dtype[Any], prim.Expression]:
    from loopy.symbolic import qpolynomial_to_expr

    from feinsum.codegen.loopy import generate_loopy_with_opt_einsum_schedule

    t_unit = generate_loopy_with_opt_einsum_schedule(
        expr, use_blas=False, optimize="optimal"
    )

    kernel = t_unit.default_entrypoint
    kernel = kernel.copy(
        silenced_warnings=(
            [
                *kernel.silenced_warnings,
                "insn_count_subgroups_upper_bound",
                "summing_if_branches_ops",
            ]
        )
    )
    t_unit = t_unit.with_kernel(kernel)
    op_map = lp.get_op_map(t_unit, subgroup_size=1)
    new_op_map: dict[np.dtype[Any], prim.Expression] = {}

    for dtype in {op.dtype.numpy_dtype for op in op_map.keys()}:
        if dtype.kind == "c":
            c_ops = {
                op_type: op_map.filter_by(
                    dtype=[dtype], name=op_type, kernel_name=kernel.name
                )
                for op_type in ["add", "mul", "div"]
            }

            pwqpoly = (
                2 * c_ops["add"].sum()
                + 6 * c_ops["mul"].sum()
                + (6 + 3 + 2) * c_ops["div"].sum()
            ).pwqpolynomial
            dtype = get_real_dtype(dtype)
        else:
            pwqpoly = (
                op_map.filter_by(dtype=[dtype], kernel_name=kernel.name)
                .sum()
                .pwqpolynomial
            )

        new_op_map.setdefault(dtype, 0)

        if pwqpoly.n_piece() > 0:
            ((_, qpoly),) = pwqpoly.get_pieces()
            new_op_map[dtype] = new_op_map[dtype] + qpolynomial_to_expr(qpoly) * 1e-9  # type: ignore[no-untyped-call]

    return Map(new_op_map)


def _get_footprint_gbytes(expr: BatchedEinsum, long_dim_length: int) -> float:
    from feinsum.codegen.loopy import generate_loopy

    t_unit = generate_loopy(expr)
    t_unit = lp.fix_parameters(
        t_unit,
        within=None,
        **dict.fromkeys(t_unit.default_entrypoint.all_params(), long_dim_length),
    )
    t_unit = lp.infer_unknown_types(t_unit)
    kernel = t_unit.default_entrypoint

    result = (
        sum(
            np.prod(arg.shape) * cast("lp.LoopyType", arg.dtype).itemsize
            for arg in kernel.args
        )
        * 1e-9
    )
    assert isinstance(result, float)
    return result


def measure_giga_op_rate(
    expr: BatchedEinsum,
    *,
    transform: TransformT,
    cl_ctx: cl.Context,
    long_dim_length: int = 100000,
    schedule: ContractionSchedule | None = None,
) -> Map[np.dtype[Any], float]:
    """
    Returns the arithmetic operations rate (in Giga Ops per second) by
    arithmetic operation's result dtypes.
    """
    runtime = timeit(
        expr,
        transform=transform,
        cl_ctx=cl_ctx,
        long_dim_length=long_dim_length,
        schedule=schedule,
    )

    from pymbolic.mapper.evaluator import evaluate_to_float

    eval_context = {param.name: long_dim_length for param in expr.all_size_params}
    return Map(
        {
            k: evaluate_to_float(v, eval_context) / runtime
            for k, v in _get_giga_ops_from_einsum(expr).items()
        }
    )


def get_roofline_flop_rate(
    expr: BatchedEinsum, dev_name: str, long_dim_length: int = 100_000
) -> Map[np.dtype[Any], float]:

    from pymbolic.mapper.evaluator import evaluate_to_float

    from feinsum.data.device_info import DEV_TO_PEAK_BW, DEV_TO_PEAK_GFLOPS

    dtype_to_gflops_expr = _get_giga_ops_from_einsum(expr)
    ngbs = _get_footprint_gbytes(expr, long_dim_length)

    dtype_to_gflops = {
        dtype: evaluate_to_float(
            giga_ops_aff,
            {param.name: long_dim_length for param in expr.all_size_params},
        )
        for dtype, giga_ops_aff in dtype_to_gflops_expr.items()
    }
    try:
        roofline_time_due_to_flops = max(
            ngflops / DEV_TO_PEAK_GFLOPS[dev_name][dtype.name]
            for dtype, ngflops in dtype_to_gflops.items()
        )
        roofline_time_due_to_global_bw = ngbs / DEV_TO_PEAK_BW[dev_name]
    except KeyError as exc:
        raise NoDevicePeaksInfoError from exc
    roofline_time = max(roofline_time_due_to_flops, roofline_time_due_to_global_bw)

    return Map(
        {dtype: gflops / roofline_time for dtype, gflops in dtype_to_gflops.items()}
    )


def _strify_measured_vs_roofline(
    measured_flop_rate: Mapping[np.dtype[Any], ToStr],
    roofline_flop_rate: Mapping[np.dtype[Any], ToStr],
) -> str:
    try:
        from tabulate import tabulate
    except ImportError as exc:
        raise ImportError(
            "`tabulate` is need for pretty printing."
            " Install via `pip install tabulate`."
        ) from exc
    assert set(measured_flop_rate.keys()) == set(roofline_flop_rate.keys())
    perf_table = [["Dtype", "Measured GOps/s", "Roofline GOps/s"]]
    for dtype in sorted(measured_flop_rate.keys(), key=lambda x: x.itemsize):
        measured_flops = (
            f"{measured_flop_rate[dtype]:.1f}"
            if isinstance(measured_flop_rate[dtype], float)
            else str(measured_flop_rate[dtype])
        )
        roofline_flops = (
            f"{(roofline_flop_rate[dtype]):.1f}"
            if isinstance(roofline_flop_rate[dtype], float)
            else str(roofline_flop_rate[dtype])
        )

        perf_table.append([dtype.name, measured_flops, roofline_flops])

    return tabulate(perf_table, tablefmt="fancy_grid")


def stringify_comparison_vs_roofline(
    expr: BatchedEinsum,
    *,
    schedule: ContractionSchedule | None = None,
    transform: TransformT,
    cl_ctx: cl.Context,
    long_dim_length: int = 100000,
    ignore_unknown_device: bool = False,
) -> str:
    """
    Returns the prettified comparison of *expr* transformed with *transform*
    wrt roofline. The roofline model assumes that kernel's performance is
    limited by either the device's global memory bandwidth or the device's
    floating point units being saturated to their maximum throughput.

    :param ignore_unknown_device: If *False* raises an error if a roofline
        model is unknown for the device in *cl_ctx*. If *True*, no error is
        raised and the roofline performance is marked as "N/A" in the output.
    """

    (dev,) = cl_ctx.devices

    measured_flop_rate = measure_giga_op_rate(
        expr,
        transform=transform,
        schedule=schedule,
        cl_ctx=cl_ctx,
        long_dim_length=long_dim_length,
    )

    try:
        roofline_flop_rate = get_roofline_flop_rate(expr, dev.name)
    except NoDevicePeaksInfoError:
        return _strify_measured_vs_roofline(
            measured_flop_rate, dict.fromkeys(measured_flop_rate.keys(), "N/A")
        )
    else:
        return _strify_measured_vs_roofline(measured_flop_rate, roofline_flop_rate)
