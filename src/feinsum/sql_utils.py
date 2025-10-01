"""
.. autofunction:: query
.. autofunction:: get_timed_einsums_in_db
.. autofunction:: record_into_db

.. autoclass:: QueryInfo
"""

import sys
import os
import logging
import sqlite3
import numpy as np
import loopy as lp
import numpy.typing as npt
import json

from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Optional,
    Callable,
    Tuple,
    Any,
    List,
    Sequence,
    Mapping,
    Union,
)
from functools import cached_property
from immutables import Map
from feinsum.einsum import BatchedEinsum, INT_CLASSES, SizeParam
from feinsum.cl_utils import ContextT, DeviceT

logger = logging.getLogger(__name__)


if TYPE_CHECKING or getattr(sys, "FEINSUM_BUILDING_SPHINX_DOCS", False):
    # avoid making pyopencl a hard dep.
    import pyopencl as cl


# transform: (t_unit, insn_match, kernel_name)
TransformT = Callable[
    ["lp.TranslationUnit", Optional[Any], Optional[str]], "lp.TranslationUnit"
]


DEFAULT_DB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    os.path.pardir,
    os.path.pardir,
    "data",
    "transform_archive_v5.sqlite",
)
TIMINGS_TABLENAME = "FEINSUM_TIMING_FACTS"


def dump_value_to_dtype(einsum: BatchedEinsum) -> str:
    return json.dumps(
        {val: dtype.name for val, dtype in einsum.value_to_dtype.items()},
        sort_keys=True,
    )


def dump_index_to_length(einsum: BatchedEinsum) -> str:
    return json.dumps(
        {
            einsum.index_names[k]: v
            for k, v in einsum.index_to_dim_length().items()
            if isinstance(v, INT_CLASSES)
        },
        sort_keys=True,
    )


def dump_use_matrix(einsum: BatchedEinsum) -> str:
    use_matrix = [
        [sorted(values) for values in use_row] for use_row in einsum.use_matrix
    ]
    return json.dumps(use_matrix)


def dump_cl_version(cl_device: "cl.Device") -> str:
    # TODO: needs to consider more things into account
    return f"{cl_device.vendor}-{cl_device.driver_version}"


def dump_op_info(einsum: BatchedEinsum, long_dim_length: int) -> str:
    from feinsum.measure import _get_giga_ops_from_einsum
    from pymbolic.mapper.evaluator import evaluate_to_float

    eval_context = {
        dim.name: long_dim_length
        for dim in einsum.index_to_dim_length().values()
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


def dump_device_name(cl_device: "cl.Device") -> str:
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
            get_transform_func_from_module_path,
            _get_impls_path,
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
) -> Tuple[QueryInfo, ...]:
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
    use_matrix = dump_use_matrix(einsum)
    value_to_dtype = dump_value_to_dtype(einsum)

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
        "    AND use_matrix = ?"
        "    AND value_to_dtype = ?"
        "    AND device_name = ?"
        ");",
        (subscripts, index_to_length, use_matrix, value_to_dtype, device_name),
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
            f"{einsum.index_names[idx]}: {lngth}"
            for idx, lngth in (einsum.index_to_dim_length().items())
            if not isinstance(lngth, SizeParam)
        )
        stringified_einsum = (
            f"{einsum.get_subscripts()} [{str_idx_to_size}]"
            f" [#outputs={einsum.noutputs}]"
        )
        raise RuntimeError(
            "No facts found for the einsum:" f" `{stringified_einsum}`."
        )

    return query_result


def get_timed_einsums_in_db(
    cl_device: DeviceT, database: str = DEFAULT_DB
) -> Tuple[BatchedEinsum, ...]:
    r"""
    Returns a :class:`tuple` of :class:`~feinsum.einsum.BatchedEinsum`\ s for
    which some timing data is available on the OpenCL device *device* in the
    database *database*.
    """
    from feinsum.make_einsum import batched_einsum

    device_name = dump_device_name(cl_device)

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(
        " SELECT"
        "     subscripts,"
        "     index_to_length,"
        "     use_matrix,"
        "     value_to_dtype"
        "  FROM "
        f"    {TIMINGS_TABLENAME}"
        " WHERE "
        "    device_name = ?"
        ";",
        (device_name,),
    )

    facts = set(cursor.fetchall())
    seen_einsums: List[BatchedEinsum] = []
    conn.close()

    for subscripts, index_to_length_str, use_matrix, value_to_dtype in facts:
        input_subscripts, _ = subscripts.split("->")
        index_to_length: Mapping[str, int] = json.loads(index_to_length_str)
        arg_shapes: List[Sequence[Union[int, float]]] = []
        processed_use_matrix = [
            [frozenset(uses) for uses in use_row]
            for use_row in json.loads(use_matrix)
        ]
        for indexing_expr in input_subscripts.split(","):
            arg_shapes.append(
                [index_to_length.get(index, np.inf) for index in indexing_expr]
            )
        seen_einsums.append(
            batched_einsum(
                subscripts,
                arg_shapes,
                processed_use_matrix,
                value_to_dtype=json.loads(value_to_dtype),
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
            " use_matrix TEXT,"
            " value_to_dtype TEXT,"
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
    cl_ctx: ContextT,
    module_path: str,
    transform_params: Mapping[str, Any],
    database: str = DEFAULT_DB,
    long_dim_length: int = 100_000,
) -> None:
    """
    Records facts corresponding to the execution of *einsum* on *cl_ctx* with
    the transformation in *module_path* along with *transform_params* and
    records it in the SQL database *database*.
    """
    from feinsum.tuning import get_transform_func_from_module_path, _get_impls_path
    from feinsum.canonicalization import canonicalize_einsum
    from feinsum.measure import timeit, stringify_comparison_vs_roofline

    dirpath, transform_space_id = os.path.split(module_path)
    assert dirpath == _get_impls_path()
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
    conn = sqlite3.connect(database)
    _create_timings_table_if_non_existent(conn)
    cursor = conn.cursor()
    subscripts = einsum.get_subscripts()
    index_to_length = dump_index_to_length(einsum)
    use_matrix = dump_use_matrix(einsum)
    value_to_dtype = dump_value_to_dtype(einsum)
    transform_params_str = json.dumps(transform_params, sort_keys=True)
    (cl_device,) = cl_ctx.devices
    device_name = dump_device_name(cl_device)
    compiler_version = dump_cl_version(cl_device)
    op_info = dump_op_info(einsum, long_dim_length=long_dim_length)

    # {{{ compute timestamp in Chicago

    import pytz
    from datetime import datetime

    timestamp = datetime.now(pytz.timezone("America/Chicago")).strftime(
        "%Y_%m_%d_%H%M%S"
    )

    # }}}

    cursor.execute(
        f"INSERT INTO {TIMINGS_TABLENAME}"
        " (subscripts, index_to_length, use_matrix,"
        "  value_to_dtype, device_name, transform_id,"
        "  transform_params, runtime_in_sec,"
        "  compiler_version, giga_op_info, timestamp)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (
            subscripts,
            index_to_length,
            use_matrix,
            value_to_dtype,
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
