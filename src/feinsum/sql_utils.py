from __future__ import annotations

__doc__ = """
.. autofunction:: query
.. autofunction:: get_timed_einsums_in_db
.. autofunction:: record_into_db

.. autoclass:: QueryInfo
"""

import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import numpy.typing as npt
from immutables import Map

from feinsum.einsum import (
    INT_CLASSES,
    BatchedEinsum,
    ShapeComponentT,
    SizeParam,
)

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    # avoid making pyopencl a hard dep.
    from collections.abc import Callable, Mapping, Sequence

    import loopy as lp
    import pyopencl as cl

    from feinsum.cl_utils import ContextT, DeviceT

    # transform: (t_unit, insn_match, kernel_name)
    TransformT = Callable[
        [lp.TranslationUnit, Any | None, str | None], lp.TranslationUnit
    ]


DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "data",
    "transform_archive_v6.sqlite",
)
TIMINGS_TABLENAME = "FEINSUM_TIMING_FACTS"


def dump_arg_to_dtype(einsum: BatchedEinsum) -> str:
    return json.dumps(
        {arg: dtype.name for arg, dtype in einsum.arg_to_dtype.items()},
        sort_keys=True,
    )


def dump_index_to_length(einsum: BatchedEinsum) -> str:
    return json.dumps(
        {
            k: v
            for k, v in einsum.index_to_dim_length.items()
            if isinstance(v, INT_CLASSES)
        },
        sort_keys=True,
    )


def dump_arg_names(einsum: BatchedEinsum) -> str:
    arg_names = [[arg.name for arg in arg_row] for arg_row in einsum.args]
    return json.dumps(arg_names)


def dump_cl_version(cl_device: cl.Device) -> str:
    # TODO: needs to consider more things into account
    return f"{cl_device.vendor}-{cl_device.driver_version}"


def dump_op_info(einsum: BatchedEinsum, long_dim_length: int) -> str:
    from pymbolic.mapper.evaluator import evaluate_to_float

    from feinsum.measure import _get_giga_ops_from_einsum

    eval_context = {
        dim.name: long_dim_length
        for dim in einsum.index_to_dim_length.values()
        if isinstance(dim, SizeParam)
    }
    dtype_to_ops = {
        k: evaluate_to_float(v, eval_context)
        for k, v in _get_giga_ops_from_einsum(einsum).items()
    }
    return json.dumps({k.name: v for k, v in dtype_to_ops.items()}, sort_keys=True)


def load_op_info(op_info: str) -> Map[np.dtype[Any], float]:
    return Map({np.dtype(k): v for k, v in json.loads(op_info).items()})


def _process_param(param: Any) -> Any:
    if isinstance(param, (int, bool)):
        return param
    elif isinstance(param, list):
        return tuple(_process_param(k) for k in param)
    else:
        raise NotImplementedError(type(param))


def load_transform_params(params_str: str) -> Map[str, Any]:
    preprocessed_params = json.loads(params_str)
    assert isinstance(preprocessed_params, dict)
    assert all(isinstance(k, str) for k in preprocessed_params)

    return Map({k: _process_param(v) for k, v in preprocessed_params.items()})


def dump_device_name(cl_device: DeviceT) -> str:
    dev_name = cl_device.name
    assert isinstance(dev_name, str)
    return (
        dev_name.replace(" ", "_")
        .replace("-", "_")
        .replace("@", "AT")
        .replace("(", "_")
        .replace(")", "_")
        .replace(".", "DOT")
    )


@dataclass(frozen=True)
class QueryInfo:
    transform_id: str
    transform_params: Map[str, Any]
    runtime_in_sec: float
    compiler_version: str
    giga_op_info: Map[np.dtype[Any], float]
    _einsum: BatchedEinsum

    def giga_op_rate(self, dtype: npt.DTypeLike) -> float:
        return self.giga_op_info[np.dtype(dtype)] / self.runtime_in_sec

    @cached_property
    def transform(self) -> TransformT:
        from feinsum.tuning import (
            _get_impls_path,
            get_transform_func_from_module_path,
        )

        module_path = os.path.join(_get_impls_path(), self.transform_id)
        return get_transform_func_from_module_path(module_path).bind_args(
            self._einsum, **self.transform_params
        )


def query(
    einsum: BatchedEinsum,
    cl_ctx: ContextT,
    *,
    database: str = DEFAULT_DB,
    err_if_no_results: bool = False,
) -> tuple[QueryInfo, ...]:
    """
    Returns facts of previous recorded runs of *einsum* on *cl_ctx*.

    :param err_if_no_results: If *True*, raises a :class:`RuntimeError` if no
        recorded runs corresponding to *einsum* are available in *database*.
        Defaults to *False*.
    """
    from feinsum.canonicalization import canonicalize_einsum

    einsum = canonicalize_einsum(einsum)
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    if len({dev.name for dev in cl_ctx.devices}) > 1:
        raise NotImplementedError(
            "CL contexts with multiple types of devices" " not supported."
        )

    cl_device = cl_ctx.devices[0]
    device_name = dump_device_name(cl_device)
    subscripts = einsum.get_subscripts()
    index_to_length = dump_index_to_length(einsum)
    arg_names = dump_arg_names(einsum)
    arg_to_dtype = dump_arg_to_dtype(einsum)

    cursor.execute(
        " SELECT name FROM sqlite_master" " WHERE (type='table' AND name=?);",
        (TIMINGS_TABLENAME,),
    )

    if not cursor.fetchall():
        raise RuntimeError(
            f"Database '{database}' does not" " contain the timing facts table."
        )

    cursor.execute(
        " SELECT"
        "     transform_id,"
        "     transform_params,"
        "     runtime_in_sec,"
        "     compiler_version,"
        "     giga_op_info"
        "  FROM "
        f"    {TIMINGS_TABLENAME}"
        " WHERE ("
        "    subscripts = ?"
        "    AND index_to_length = ?"
        "    AND arg_names = ?"
        "    AND arg_to_dtype = ?"
        "    AND device_name = ?"
        ");",
        (subscripts, index_to_length, arg_names, arg_to_dtype, device_name),
    )

    facts = cursor.fetchall()

    query_result = tuple(
        QueryInfo(
            transform_id=fact[0],
            transform_params=json.loads(fact[1]),
            runtime_in_sec=fact[2],
            compiler_version=fact[3],
            giga_op_info=load_op_info(fact[4]),
            _einsum=einsum,
        )
        for fact in facts
    )
    conn.close()

    if not query_result and err_if_no_results:
        str_idx_to_size = ", ".join(
            f"{idx}: {lngth}"
            for idx, lngth in (einsum.index_to_dim_length.items())
            if not isinstance(lngth, SizeParam)
        )
        stringified_einsum = (
            f"{einsum.get_subscripts()} [{str_idx_to_size}]"
            f" [#outputs={einsum.b}]"
        )
        raise RuntimeError(
            "No facts found for the einsum:" f" `{stringified_einsum}`."
        )

    return query_result


def _get_batched_einsum_from_sql_row(
    subscripts: str,
    index_to_length: Mapping[str, ShapeComponentT],
    arg_names: Sequence[Sequence[str]],
    arg_to_dtype: Mapping[str, str],
) -> BatchedEinsum:
    from functools import reduce
    from typing import cast

    from feinsum.einsum import SizeParam
    from feinsum.make_einsum import (
        _normalize_einsum_subscript,
        array,
        batched_einsum,
    )

    in_specs, _ = subscripts.split("->")
    index_to_length = dict(index_to_length)
    in_idx_sets = tuple(
        _normalize_einsum_subscript(in_spec, is_output=False)
        for in_spec in in_specs.split(",")
    )

    all_indices = reduce(
        frozenset.union,
        (frozenset(in_idx_set) for in_idx_set in in_idx_sets),
        cast("frozenset[str]", frozenset()),
    )
    for idx in all_indices:
        if idx not in index_to_length:
            assert idx.islower()
            index_to_length[idx] = SizeParam(idx.upper())

    arg_to_shape = {
        arg: [index_to_length[idx] for idx in in_idx_set]
        for arg_row in arg_names
        for in_idx_set, arg in zip(in_idx_sets, arg_row, strict=True)
    }
    args = [
        [array(arg, arg_to_shape[arg], arg_to_dtype[arg]) for arg in arg_row]
        for arg_row in arg_names
    ]
    return batched_einsum(subscripts, args)


def get_timed_einsums_in_db(
    cl_device: DeviceT, database: str = DEFAULT_DB
) -> tuple[BatchedEinsum, ...]:
    r"""
    Returns a :class:`tuple` of :class:`~feinsum.einsum.BatchedEinsum`\ s for
    which some timing data is available on the OpenCL device *device* in the
    database *database*.
    """

    device_name = dump_device_name(cl_device)

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        " SELECT"
        "     subscripts,"
        "     index_to_length,"
        "     args,"
        "     arg_to_dtype"
        "  FROM "
        f"    {TIMINGS_TABLENAME}"
        " WHERE "
        "    device_name = ?"
        ";",
        (device_name,),
    )

    facts = set(cursor.fetchall())
    seen_einsums: list[BatchedEinsum] = []
    conn.close()

    for subscripts, index_to_length_str, arg_names_str, arg_to_dtype_str in facts:
        index_to_length = json.loads(index_to_length_str)
        arg_to_dtype = json.loads(arg_to_dtype_str)
        arg_names = json.loads(arg_names_str)
        seen_einsums.append(
            _get_batched_einsum_from_sql_row(
                subscripts, index_to_length, arg_names, arg_to_dtype
            )
        )

    # Asserts that the canonicalization was sound.
    assert len(set(seen_einsums)) == len(seen_einsums)

    return tuple(seen_einsums)


def _create_timings_table_if_non_existent(conn: sqlite3.Connection) -> None:
    cursor = conn.cursor()
    cursor.execute(
        " SELECT name FROM sqlite_master" " WHERE (type='table' AND name=?);",
        (TIMINGS_TABLENAME,),
    )

    if not cursor.fetchall():
        # device table not available
        logger.info(f"Table {TIMINGS_TABLENAME} not in DB, creating one.")
        cursor.execute(
            f"CREATE TABLE {TIMINGS_TABLENAME} ("
            " ID INTEGER PRIMARY KEY AUTOINCREMENT,"
            " subscripts TEXT,"
            " index_to_length TEXT,"
            " args TEXT,"
            " arg_to_dtype TEXT,"
            " device_name TEXT,"
            " transform_id TEXT,"
            " transform_params TEXT,"
            " runtime_in_sec REAL,"
            " compiler_version TEXT,"
            " giga_op_info TEXT,"
            " timestamp TEXT"
            ")"
        )
    conn.commit()


def record_into_db(
    einsum: BatchedEinsum,
    cl_ctx: cl.Context,
    module_path: str,
    transform_params: Mapping[str, Any],
    database: str | sqlite3.Connection = DEFAULT_DB,
    long_dim_length: int = 100_000,
) -> None:
    """
    Records facts corresponding to the execution of *einsum* on *cl_ctx* with
    the transformation in *module_path* along with *transform_params* and
    records it in the SQL database *database*.
    """
    from feinsum.canonicalization import canonicalize_einsum
    from feinsum.measure import stringify_comparison_vs_roofline, timeit
    from feinsum.tuning import _get_impls_path, get_transform_func_from_module_path

    dirpath, transform_space_id = os.path.split(module_path)
    if dirpath != _get_impls_path():
        transform_space_id = module_path
    einsum = canonicalize_einsum(einsum)

    transform_func = get_transform_func_from_module_path(module_path).bind_args(
        einsum, **transform_params
    )
    logger.info(
        "\n"
        + stringify_comparison_vs_roofline(
            einsum,
            transform=transform_func,
            cl_ctx=cl_ctx,
        )
    )
    runtime = timeit(
        einsum,
        cl_ctx=cl_ctx,
        transform=transform_func,
        long_dim_length=long_dim_length,
    )
    if isinstance(database, str):
        conn = sqlite3.connect(database)
    else:
        assert isinstance(database, sqlite3.Connection)
        conn = database

    _create_timings_table_if_non_existent(conn)
    cursor = conn.cursor()
    subscripts = einsum.get_subscripts()
    index_to_length = dump_index_to_length(einsum)
    arg_names = dump_arg_names(einsum)
    arg_to_dtype = dump_arg_to_dtype(einsum)
    transform_params_str = json.dumps(transform_params, sort_keys=True)
    (cl_device,) = cl_ctx.devices
    device_name = dump_device_name(cl_device)
    compiler_version = dump_cl_version(cl_device)
    op_info = dump_op_info(einsum, long_dim_length=long_dim_length)

    # {{{ compute timestamp in Chicago

    from datetime import datetime

    import pytz

    timestamp = datetime.now(pytz.timezone("America/Chicago")).strftime(
        "%Y_%m_%d_%H%M%S"
    )

    # }}}

    cursor.execute(
        f"INSERT INTO {TIMINGS_TABLENAME}"
        " (subscripts, index_to_length, args,"
        "  arg_to_dtype, device_name, transform_id,"
        "  transform_params, runtime_in_sec,"
        "  compiler_version, giga_op_info, timestamp)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            subscripts,
            index_to_length,
            arg_names,
            arg_to_dtype,
            device_name,
            transform_space_id,
            transform_params_str,
            runtime,
            compiler_version,
            op_info,
            timestamp,
        ),
    )

    conn.commit()


# vim: foldmethod=marker
