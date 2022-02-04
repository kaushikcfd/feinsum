"""
.. autofunction:: record
.. autofunction:: query_with_least_runtime
"""

import os
import logging
import sqlite3
import numpy as np
import loopy as lp

from dataclasses import dataclass
from typing import (TYPE_CHECKING, Optional, Union, Callable, Sequence,
                    Tuple, FrozenSet, Any, Dict, Mapping)
from pyrsistent import pmap
from pyrsistent.typing import PMap as PMapT
from feinsum.einsum import FusedEinsum, INT_CLASSES, SizeParam

logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    import pyopencl as cl


FallbackT = Union[str, Callable[["lp.TranslationUnit"], "lp.TranslationUnit"]]


DEFAULT_FALLBACKS = ()

# Normalized representation of feinsum?
# * Subscript
# * No assuming non-associative floating point operations.
# * Suggestion 1:
#    -

DEFAULT_TRANSFORM_ARCHIVE = os.path.join(os.path.dirname(__file__),
                                         "../../data/transform_archive_v0.db")


def _get_clbl_from_string(transform_src: str) -> Callable[["lp.TranslationUnit"],
                                                          "lp.TranslationUnit"]:

    result_dict: Dict[Any, Any] = {}
    exec(transform_src, result_dict)
    clbl = result_dict["transform"]
    if not callable(result_dict.get("transform")):
        raise ValueError("Provided transform source does not"
                         " define callable named 'transform'.")
    return clbl  # type: ignore[no-any-return]


def _get_normalized_value_name_mapping_for_db(einsum: FusedEinsum
                                              ) -> Mapping[str, str]:
    return pmap({val: f"arg_{i}"
                 for i, val in enumerate(einsum.value_to_dtype)})


def _get_value_to_dtype_for_db(einsum: FusedEinsum) -> str:
    normalized_value_map = _get_normalized_value_name_mapping_for_db(einsum)
    normalized_value_to_dtype = {normalized_value_map[k]: v
                                 for k, v in einsum.value_to_dtype.items()}
    return ("["
            + ", ".join(f"{val}: {dtype.name}"
                           for val, dtype in sorted(normalized_value_to_dtype
                                                    .items()))
            + "]")


def _get_index_to_length_for_db(einsum: FusedEinsum) -> str:
    return "[" + ", ".join(f"{einsum.index_names[k]}: {v}"
                           for k, v in einsum.index_to_dim_length().items()
                           if isinstance(v, INT_CLASSES)) + "]"


def _get_use_matrix_for_db(einsum: FusedEinsum) -> str:
    normalized_value_map = _get_normalized_value_name_mapping_for_db(einsum)

    def _stringify_use_row(use_row: Tuple[FrozenSet[str], ...]) -> str:
        normalized_use_row = [sorted({normalized_value_map[use]
                                      for use in uses})
                              for uses in use_row]
        return ("["
                + ", ".join("[" + ", ".join(uses) + "]"
                            for uses in normalized_use_row)
                + "]")

    return "[" + ",\n".join(_stringify_use_row(use_row)
                            for use_row in einsum.use_matrix) + "]"


def _get_cl_version_for_db(cl_device: "cl.Device") -> str:
    # TODO: needs to consider more things into account
    return f"{cl_device.vendor}-{cl_device.driver_version}"


def _get_op_info_for_db(einsum: FusedEinsum, long_dim_length: int) -> str:
    from feinsum.measure import _get_giga_ops_from_einsum
    from pymbolic.mapper.evaluator import evaluate_to_float

    eval_context = {dim.name: long_dim_length
                    for dim in einsum.index_to_dim_length().values()
                    if isinstance(dim, SizeParam)}
    dtype_to_ops = {k: evaluate_to_float(v, eval_context)
                    for k, v in _get_giga_ops_from_einsum(einsum).items()}
    return "\n".join(f"{k.name}: {v}"
                     for k, v in dtype_to_ops.items())


def record(einsum: FusedEinsum,
           cl_ctx: "cl.Context",
           *,
           transform_str: Optional[str] = None,
           transform_file_path: Optional[str] = None,
           authors: str,
           remarks: str = "",
           database: str = DEFAULT_TRANSFORM_ARCHIVE,
           long_dim_length: int = 50_000,
           ) -> None:

    from feinsum.measure import timeit
    from feinsum.codegen.loopy import generate_loopy

    # TODO: Instead of taking in a long_dim_length, should allow setting each
    # parameter its value.
    # How to record the FusedEinsum in sqlite? The
    # interaction matrix is more or less clear, but what permutation to apply
    # to t.
    if (transform_str is not None) and (transform_file_path is not None):
        raise ValueError("Cannot pass in both transform_str"
                         " and transform_file_path.")

    if transform_str is None and transform_file_path is None:
        raise ValueError("Must pass either transform_str"
                         " or transform_file_path.")

    if transform_str is None:
        assert transform_file_path is not None
        with open(transform_file_path, "r") as fp:
            transform_str = fp.read()

    assert transform_str is not None
    transform_clbl = _get_clbl_from_string(transform_str)

    runtime = timeit(einsum,
                     transform=transform_clbl,
                     cl_ctx=cl_ctx,
                     long_dim_length=long_dim_length)

    conn = sqlite3.connect(database)
    # TODO: How to handle multiple devices?:
    cl_device, = cl_ctx.devices
    device_name = (cl_device.name
                   .replace(" ", "_")
                   .replace("-", "_")
                   .replace("@", "AT")
                   .replace("(", "_")
                   .replace(")", "_")
                   .replace(".", "DOT")
                   )
    cursor = conn.cursor()

    # {{{ get available tables

    cursor.execute(" SELECT name FROM sqlite_master"
                   f" WHERE (type='table' AND name='{device_name}');")

    if not cursor.fetchall():
        # device table not available
        cursor.execute(f"CREATE TABLE {device_name} ("
                       " subscripts TEXT,"
                       " index_to_length TEXT,"
                       " use_matrix TEXT,"
                       " value_to_dtype TEXT,"
                       " loopy_transform TEXT,"
                       " runtime_in_sec REAL,"
                       " authors TEXT,"
                       " compiler_version TEXT,"
                       " cl_kernel TEXT,"
                       " giga_op_info TEXT,"
                       " remarks TEXT"
                       ")")

    # }}}

    subscripts = einsum.get_subscripts()
    index_to_length = _get_index_to_length_for_db(einsum)
    transform_str = transform_str.replace("\n", "\\n").replace("'", "''")
    use_matrix = _get_use_matrix_for_db(einsum).replace("\n", "\\n")
    value_to_dtype = _get_value_to_dtype_for_db(einsum)
    cl_kernel = (lp.generate_code_v2(transform_clbl(generate_loopy(einsum)))
                 .device_code()).replace("\n", "\\n")
    compiler_version = _get_cl_version_for_db(cl_device)
    op_info = _get_op_info_for_db(einsum, long_dim_length=long_dim_length)

    cursor.execute(f"INSERT INTO {device_name}"
                   " VALUES ("
                   f"'{subscripts}',"        # subscripts
                   f" '{index_to_length}',"   # index_to_length
                   f" '{use_matrix}',"        # use_matrix
                   f" '{value_to_dtype}',"    # value_to_dtype
                   f" '{transform_str}',"     # loopy_transform
                   f" {runtime},"             # runtime_in_sec
                   f" '{authors}',"           # authors
                   f" '{compiler_version}',"  # compiler_version
                   f" '{cl_kernel}',"         # cl_kernel
                   f" '{op_info}',"           # giga_op_info
                   f" '{remarks}'"           # remarks
                   ")")
    conn.commit()


def are_two_einsums_equivalent(e1: FusedEinsum, e2: FusedEinsum) -> bool:
    raise NotImplementedError


@dataclass(frozen=True, eq=True, repr=True)
class QueryInfo:
    loopy_transform: str
    runtime_in_sec: float
    authors: str
    compiler_version: str
    cl_kernel: str
    giga_op_info: PMapT[np.dtype[Any], float]
    remarks: str


def query_with_least_runtime(einsum: FusedEinsum,
                             device_name: str,
                             database_path: str,
                             fallbacks: Sequence[FallbackT] = DEFAULT_FALLBACKS,
                             ) -> QueryInfo:
    # TODO: This should  somehow solve the normalized FusedEinsum problem.
    raise NotImplementedError

# vim: foldmethod=marker
