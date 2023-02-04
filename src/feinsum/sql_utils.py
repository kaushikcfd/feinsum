"""
.. autofunction:: query
.. autofunction:: get_timed_einsums_in_db
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
from typing import (TYPE_CHECKING, Optional, Callable,
                    Tuple, Any, List, Sequence, Mapping, Union)
from functools import cached_property
from immutables import Map
from feinsum.einsum import FusedEinsum, INT_CLASSES, SizeParam
from feinsum.cl_utils import ContextT, DeviceT

logger = logging.getLogger(__name__)


if TYPE_CHECKING or getattr(sys, "FEINSUM_BUILDING_SPHINX_DOCS", False):
    # avoid making pyopencl a hard dep.
    import pyopencl as cl


# transform: (t_unit, insn_match, kernel_name)
TransformT = Callable[["lp.TranslationUnit", Optional[Any], Optional[str]],
                      "lp.TranslationUnit"]


DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.path.pardir, os.path.pardir,
                          "data", "transform_archive_v4.sqlite")
TIMINGS_TABLENAME = "FEINSUM_TIMING_FACTS"


def dump_value_to_dtype(einsum: FusedEinsum) -> str:
    return json.dumps({val: dtype.name
                       for val, dtype in einsum.value_to_dtype.items()},
                      sort_keys=True
                      )


def dump_index_to_length(einsum: FusedEinsum) -> str:
    return json.dumps({einsum.index_names[k]: v
                       for k, v in einsum.index_to_dim_length().items()
                       if isinstance(v, INT_CLASSES)},
                      sort_keys=True
                      )


def dump_use_matrix(einsum: FusedEinsum) -> str:
    use_matrix = [
        [sorted(values) for values in use_row]
        for use_row in einsum.use_matrix]
    return json.dumps(use_matrix)


def dump_cl_version(cl_device: "cl.Device") -> str:
    # TODO: needs to consider more things into account
    return f"{cl_device.vendor}-{cl_device.driver_version}"


def dump_op_info(einsum: FusedEinsum, long_dim_length: int) -> str:
    from feinsum.measure import _get_giga_ops_from_einsum
    from pymbolic.mapper.evaluator import evaluate_to_float

    eval_context = {dim.name: long_dim_length
                    for dim in einsum.index_to_dim_length().values()
                    if isinstance(dim, SizeParam)}
    dtype_to_ops = {k: evaluate_to_float(v, eval_context)
                    for k, v in _get_giga_ops_from_einsum(einsum).items()}
    return json.dumps({k.name: v
                       for k, v in dtype_to_ops.items()},
                      sort_keys=True
                      )


def load_op_info(op_info: str) -> Map[np.dtype[Any], float]:
    return Map({np.dtype(k): v
                for k, v in json.loads(op_info).items()})


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

    return Map({k: _process_param(v)
                for k, v in preprocessed_params.items()})


def dump_device_name(cl_device: "cl.Device") -> str:
    dev_name = cl_device.name
    assert isinstance(dev_name, str)
    return (dev_name
            .replace(" ", "_")
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
    _einsum: FusedEinsum

    def giga_op_rate(self, dtype: npt.DTypeLike) -> float:
        return self.giga_op_info[np.dtype(dtype)]/self.runtime_in_sec

    @cached_property
    def transform(self) -> TransformT:
        from feinsum.tuning import (get_transform_func_from_module_path,
                                    _get_impls_path)

        module_path = os.path.join(_get_impls_path(), self.transform_id)
        return get_transform_func_from_module_path(
            module_path).bind_args(self._einsum, **self.transform_params)


def query(einsum: FusedEinsum,
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
        raise NotImplementedError("CL contexts with multiple types of devices"
                                  " not supported.")

    cl_device = cl_ctx.devices[0]
    device_name = dump_device_name(cl_device)
    subscripts = einsum.get_subscripts()
    index_to_length = dump_index_to_length(einsum)
    use_matrix = dump_use_matrix(einsum)
    value_to_dtype = dump_value_to_dtype(einsum)

    cursor.execute(" SELECT name FROM sqlite_master"
                   " WHERE (type='table' AND name=?);",
                   (TIMINGS_TABLENAME,))

    if not cursor.fetchall():
        raise RuntimeError(f"Database '{database}' does not"
                           " contain the timing facts table.")

    cursor.execute(" SELECT"
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
                   (subscripts, index_to_length,
                    use_matrix, value_to_dtype,
                    device_name))

    facts = cursor.fetchall()

    query_result = tuple(
        QueryInfo(
            transform_id=fact[0],
            transform_params=json.loads(fact[1]),
            runtime_in_sec=fact[2],
            compiler_version=fact[3],
            giga_op_info=load_op_info(fact[4]),
            _einsum=einsum)
        for fact in facts)
    conn.close()

    if not query_result and err_if_no_results:
        str_idx_to_size = ", ".join(f"{einsum.index_names[idx]}: {lngth}"
                                    for idx, lngth in (einsum
                                                       .index_to_dim_length()
                                                       .items())
                                    if not isinstance(lngth, SizeParam))
        stringified_einsum = (f"{einsum.get_subscripts()} [{str_idx_to_size}]"
                              f" [#outputs={einsum.noutputs}]")
        raise RuntimeError("No facts found for the einsum:"
                           f" `{stringified_einsum}`.")

    return query_result


def get_timed_einsums_in_db(cl_device: DeviceT,
                            database: str = DEFAULT_DB) -> Tuple[FusedEinsum,
                                                                 ...]:
    r"""
    Returns a :class:`tuple` of :class:`~feinsum.einsum.FusedEinsum`\ s for
    which some timing data is available on the OpenCL device *device* in the
    database *database*.
    """
    from feinsum.make_einsum import fused_einsum

    device_name = dump_device_name(cl_device)

    conn = sqlite3.connect(database)
    cursor = conn.cursor()
    cursor.execute(" SELECT"
                   "     subscripts,"
                   "     index_to_length,"
                   "     use_matrix,"
                   "     value_to_dtype"
                   "  FROM "
                   f"    {TIMINGS_TABLENAME}"
                   " WHERE "
                   "    device_name = ?"
                   ";", (device_name,))

    facts = set(cursor.fetchall())
    seen_einsums: List[FusedEinsum] = []
    conn.close()

    for (subscripts, index_to_length_str, use_matrix, value_to_dtype) in facts:
        input_subscripts, _ = subscripts.split("->")
        index_to_length: Mapping[str, int] = json.loads(index_to_length_str)
        arg_shapes: List[Sequence[Union[int, float]]] = []
        processed_use_matrix = [[frozenset(uses) for uses in use_row]
                                for use_row in json.loads(use_matrix)]
        for indexing_expr in input_subscripts.split(","):
            arg_shapes.append([index_to_length.get(index, np.inf)
                               for index in indexing_expr])
        seen_einsums.append(fused_einsum(subscripts,
                                         arg_shapes,
                                         processed_use_matrix,
                                         value_to_dtype=json.loads(value_to_dtype)))

    # Asserts that the canonicalization was sound.
    assert len(set(seen_einsums)) == len(seen_einsums)

    return tuple(seen_einsums)

# vim: foldmethod=marker
