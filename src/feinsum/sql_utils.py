"""
.. autofunction:: query
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
                    Tuple, Any)
from functools import cached_property
from immutables import Map
from feinsum.einsum import FusedEinsum, INT_CLASSES, SizeParam
from feinsum.cl_utils import ContextT

logger = logging.getLogger(__name__)


if TYPE_CHECKING or getattr(sys, "FEINSUM_BUILDING_SPHINX_DOCS", False):
    # avoid making pyopencl a hard dep.
    import pyopencl as cl


# transform: (t_unit, insn_match, kernel_name)
TransformT = Callable[["lp.TranslationUnit", Optional[Any], Optional[str]],
                      "lp.TranslationUnit"]


DEFAULT_DB = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          os.path.pardir, os.path.pardir,
                          "data", "transform_archive_v2.sqlite")


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


def dump_tablename(cl_device: "cl.Device") -> str:
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

    def giga_op_rate(self, dtype: npt.DTypeLike) -> float:
        return self.giga_op_info[np.dtype(dtype)]/self.runtime_in_sec

    @cached_property
    def transform(self) -> TransformT:
        raise NotImplementedError


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
    # TODO: This should  somehow solve the normalized FusedEinsum problem.
    from feinsum.normalization import normalize_einsum
    einsum = normalize_einsum(einsum)
    conn = sqlite3.connect(database)
    cursor = conn.cursor()

    if len({dev.name for dev in cl_ctx.devices}) > 1:
        raise NotImplementedError("CL contexts with multiple types of devices"
                                  " not supported.")

    cl_device = cl_ctx.devices[0]
    tablename = dump_tablename(cl_device)
    subscripts = einsum.get_subscripts()
    index_to_length = dump_index_to_length(einsum)
    use_matrix = dump_use_matrix(einsum)
    value_to_dtype = dump_value_to_dtype(einsum)

    cursor.execute(" SELECT name FROM sqlite_master"
                   " WHERE (type='table' AND name=?);",
                   (tablename,))

    if not cursor.fetchall():
        logger.warn(f"No entries for {cl_device}")
        return ()

    cursor.execute(" SELECT"
                   "     transform_id,"
                   "     transform_params,"
                   "     runtime_in_sec,"
                   "     compiler_version,"
                   "     giga_op_info"
                   "  FROM "
                   f"    {tablename}"
                   " WHERE ("
                   "    subscripts = ?"
                   "    AND index_to_length = ?"
                   "    AND use_matrix = ?"
                   "    AND value_to_dtype = ?"
                   ");",
                   (subscripts, index_to_length,
                    use_matrix, value_to_dtype))

    facts = cursor.fetchall()

    query_result = tuple(
        QueryInfo(
            transform_id=fact[0],
            transform_params=fact[1],
            runtime_in_sec=fact[2],
            compiler_version=fact[3],
            giga_op_info=load_op_info(fact[4]))
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


# vim: foldmethod=marker
