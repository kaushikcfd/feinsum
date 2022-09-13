"""
.. autoclass:: IntParameter
.. autoclass:: BoolParameter

.. autofunction:: autotune
.. autofunction:: transform_param
.. autofunction:: einsum_arg
"""
import abc
import os
import sqlite3
import numpy as np
import pyopencl as cl
import loopy as lp
import opentuner

from typing import Callable, Any, Tuple, Mapping, Union
from dataclasses import dataclass
from functools import cached_property, cache
from feinsum.einsum import FusedEinsum
import logging
logger = logging.getLogger(__name__)


DB_FILENAME = os.path.join(os.path.dirname(__file__),
                           os.path.pardir,
                           os.path.pardir,
                           "data",
                           "transform_archive_v2.db")


# {{{ supported tuning parameters

class TuningParameter(abc.ABC):
    pass


@dataclass(frozen=True, repr=True)
class IntParameter(TuningParameter):
    """
    A parameter that takes values in the range ``[low, high)``.
    """
    low: int
    high: int


@dataclass(frozen=True, repr=True)
class BoolParameter(TuningParameter):
    """
    A parameter that can be either *True* or *False*.
    """

# }}}


# {{{

@dataclass(frozen=True, repr=True)
class EinsumDerivativeArg:
    var_name: str
    func: Callable[[FusedEinsum], Any]


@dataclass(frozen=True, repr=True)
class TransformParam:
    var_name: str
    func: Callable[[FusedEinsum], TuningParameter]


@dataclass(frozen=True, repr=True)
class ParametrizedTransform:
    transform: Callable
    einsum_derivative_args: Tuple[EinsumDerivativeArg, ...]
    transform_params: Tuple[TransformParam, ...]

    def __call__(self, *args, **kwargs) -> lp.TranslationUnit:
        return self.transform(*args, **kwargs)


def transform_param(fn,
                    arg_name: str,
                    param_getter: Callable[[FusedEinsum], TuningParameter]):
    """
    Decorate to a template transformation to inform
    :func:`autotune` about the parameter space.

    :param arg_name: Name of the argument in the template
        transformation to be parametrized.
    :param param_getter: A callable that expects an einsum and returns the
        parameter space for that einsum.
    """
    transform_param = TransformParam(arg_name, param_getter)
    if isinstance(fn, ParametrizedTransform):
        return ParametrizedTransform(
            fn.transform,
            fn.einsum_derivative_args,
            (transform_param,) + fn.transform_params)
    else:
        return ParametrizedTransform(fn, (), (transform_param,))


def einsum_arg(fn, arg_name, param_getter):
    """
    Decorate to a template transformation to inform
    :func:`autotune` about a static argument to the transformation
    implementation.

    :param arg_name: Name of the argument in the template transform
        that is to be statically set.
    :param param_getter: A callable that expects an einsum and returns the
        value of the static argument.
    """
    einsum_arg = EinsumDerivativeArg(arg_name, param_getter)
    if isinstance(fn, ParametrizedTransform):
        return ParametrizedTransform(
            fn.transform,
            (einsum_arg,) + fn.einsum_derivative_args,
            fn.transform_params)
    else:
        return ParametrizedTransform(fn, (einsum_arg,), ())

# }}}


@cache
def _get_impls_path() -> str:
    import importlib.util
    return os.path.abspath(
        os.path.join(importlib.util.find_spec("feinsum").origin,
                     os.path.pardir, "tuning", "impls"))


class ConfigurationNotInDBError(LookupError):
    pass


def bind_args(transform: ParametrizedTransform,
              einsum: FusedEinsum,
              **transform_args: Any) -> Callable[[lp.TranslationUnit],
                                                 lp.TranslationUnit]:
    """
    Binds *transform_args* to *transform* and returns a python callable
    to the corresponding instance in the transform space.
    """
    from functools import partial

    py_clbl = partial(transform.transform,
                      **{arg.name: arg.func(einsum)
                         for arg in transform.einsum_derivative_args},
                      **transform_args)
    return py_clbl


# {{{ Opentuner entrypoint

class OpentunerTuner(opentuner.MeasurementInterface):
    def __init__(self,
                 args: Any,
                 einsum: FusedEinsum,
                 cl_ctx: cl.Context,
                 module_path: str,
                 ):
        from feinsum.normalization import normalize_einsum
        super().__init__(args=args)

        self.einsum = normalize_einsum(einsum)
        self.cl_ctx = cl_ctx
        self.module_path = module_path

    @cached_property
    def sql_table_name(self) -> str:
        from feinsum.database import _get_cl_device_name_for_db
        dev, = self.cl_ctx.devices
        return _get_cl_device_name_for_db(dev)

    @cached_property
    def transform_space_id(self) -> str:
        dirpath, filepath = os.path.sep(self.module_path)

        if dirpath == _get_impls_path():
            return filepath
        else:
            return self.module_path

    @cached_property
    def conn(self) -> sqlite3.Connection:
        db = sqlite3.connect(DB_FILENAME)
        cursor = db.cursor()
        cursor.execute(" SELECT name FROM sqlite_master"
                       f" WHERE (type='table' AND name='{self.sql_table_name}');")

        if not cursor.fetchall():
            # device table not available
            logger.info(f"Table {self.sql_table_name} not in DB, creating one.")
            cursor.execute(f"CREATE TABLE {self.sql_table_name} ("
                           " ID INTEGER PRIMARY KEY AUTOINCREMENT,"
                           " subscripts TEXT,"
                           " index_to_length TEXT,"
                           " use_matrix TEXT,"
                           " value_to_dtype TEXT,"
                           " transform_id TEXT,"
                           " transform_params TEXT,"
                           " runtime_in_sec REAL,"
                           " compiler_version TEXT,"
                           " giga_op_info TEXT,"
                           " timestamp TEXT"
                           ")")
        return db

    @cached_property
    def transform_func(self) -> Union[Callable, ParametrizedTransform]:
        import importlib
        transform_module = importlib.import_module(self.module_path)
        return transform_module.transform

    def manipulator(self) -> None:
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        from opentuner import ConfigurationManipulator
        from opentuner.search.manipulator import BooleanParameter
        manipulator = ConfigurationManipulator()

        if isinstance(self.transform_func, ParametrizedTransform):
            for param in self.transform_func.transform_params:
                if isinstance(param, IntParameter):
                    manipulator.add_parameter(
                        opentuner.IntegerParameter(param.name,
                                                   param.low,
                                                   param.high-1))
                elif isinstance(param, BoolParameter):
                    manipulator.add_parameter(BooleanParameter())
                else:
                    raise NotImplementedError(f"Parameter: {param}.")
        else:
            assert callable(self.transform_params)

    def seed_configurations(self) -> None:
        import json
        from feinsum.database import (_get_index_to_length_for_db,
                                      _preprocess_string_for_sql,
                                      _get_use_matrix_for_db,
                                      _get_value_to_dtype_for_db)

        cursor = self.conn.cursor()
        subscripts = self.einsum.get_subscripts()
        index_to_length = _get_index_to_length_for_db(self.einsum)
        use_matrix = _preprocess_string_for_sql(
            _get_use_matrix_for_db(self.einsum))
        value_to_dtype = _get_value_to_dtype_for_db(self.einsum)

        cursor.execute(" SELECT"
                       "     transform_params"
                       "  FROM "
                       f"    {self.sql_table_name}"
                       " WHERE ("
                       f"    subscripts = '{subscripts}'"
                       f"    AND index_to_length = '{index_to_length}'"
                       f"    AND use_matrix = '{use_matrix}'"
                       f"    AND value_to_dtype = '{value_to_dtype}'"
                       ");")
        return [
            json.loads(transform_params)
            for transform_params in cursor.fetchall()
        ]

    def query_from_db(self, parameters) -> float:
        import json
        from feinsum.database import (_get_index_to_length_for_db,
                                      _preprocess_string_for_sql,
                                      _get_use_matrix_for_db,
                                      _get_value_to_dtype_for_db)

        cursor = self.conn.cursor()
        subscripts = self.einsum.get_subscripts()
        index_to_length = _get_index_to_length_for_db(self.einsum)
        use_matrix = _preprocess_string_for_sql(
            _get_use_matrix_for_db(self.einsum))
        value_to_dtype = _get_value_to_dtype_for_db(self.einsum)
        transform_params_str = json.dumps(parameters)

        cursor.execute(" SELECT"
                        "     runtime_in_sec"
                       "  FROM "
                       f"    {self.sql_table_name}"
                       " WHERE ("
                       f"    subscripts = '{subscripts}'"
                       f"    AND index_to_length = '{index_to_length}'"
                       f"    AND use_matrix = '{use_matrix}'"
                       f"    AND value_to_dtype = '{value_to_dtype}'"
                       f"    AND transform_params = '{transform_params_str}'"
                       ");")
        stored_results = cursor.fetchall()
        if not stored_results:
            raise ConfigurationNotInDBError
        else:
            return min(stored_result[0] for stored_result in stored_results)

    def record_into_db(self, runtime: float, parameters: Mapping[str, Any]) -> None:
        import json
        from feinsum.database import (_get_index_to_length_for_db,
                                      _preprocess_string_for_sql,
                                      _get_use_matrix_for_db,
                                      _get_value_to_dtype_for_db,
                                      _get_cl_version_for_db,
                                      _get_op_info_for_db
                                      )

        cursor = self.conn.cursor()
        subscripts = self.einsum.get_subscripts()
        index_to_length = _get_index_to_length_for_db(self.einsum)
        use_matrix = _preprocess_string_for_sql(
            _get_use_matrix_for_db(self.einsum))
        value_to_dtype = _get_value_to_dtype_for_db(self.einsum)
        transform_params_str = json.dumps(parameters)
        cl_device, = self.cl_ctx.devices
        compiler_version = _get_cl_version_for_db(cl_device)
        op_info = _preprocess_string_for_sql(
            _get_op_info_for_db(self.einsum, long_dim_length=self.long_dim_length))

        # {{{ compute timestamp in Chicago

        import pytz
        from datetime import datetime

        timestamp = (datetime
                     .now(pytz.timezone("America/Chicago"))
                     .strftime("%Y_%m_%d_%H%M%S"))

        # }}}

        cursor.execute(f"INSERT INTO {self.sql_table_name}"
                       " (subscripts, index_to_length, use_matrix,"
                       "  value_to_dtype, transform_id,"
                       "  transform_params, compiler_version,"
                       "  giga_op_info, timestamp)"
                       " VALUES (?,?,?,?,?,?,?,?,?,?)",
                       subscripts, index_to_length, use_matrix,
                       value_to_dtype, self.transform_space_id,
                       transform_params_str, compiler_version,
                       op_info, timestamp)

        self.conn.commit()

    def run(self, desired_result, input, limit):
        from feinsum.measurement import (timeit,
                                         stringify_comparison_vs_roofline)
        from feinsum.diagnostics import InvalidParameterError

        cfg = desired_result.configuration.data

        logger.info(cfg)

        # {{{ query from DB

        try:
            result = self.query_from_db(cfg)
        except ConfigurationNotInDBError:
            pass
        else:
            logger.info("DB Hit")
            return opentuner.Result(time=result)

        # }}}

        bound_transform = bind_args(self.transform_func,
                                    self.einsum,
                                    **cfg)

        try:
            logger.info(stringify_comparison_vs_roofline(
                self.einsum,
                transform=bound_transform,
                cl_ctx=self.cl_ctx,
            ))
        except InvalidParameterError as err:
            logger.info(f"Ignored configuration due to '{err}'.")
            return opentuner.Result(timer=np.inf)

        runtime = timeit(self.einsum,
                         cl_ctx=self.cl_ctx,
                         transform=bound_transform)
        self.record_into_db(runtime, cfg)

        return opentuner.Result(time=runtime)

# }}}


def autotune(einsum: FusedEinsum, module_path: str, cl_ctx: cl.Context) -> None:
    """
    TODO
    """
    if not os.path.isabs(module_path):
        raise ValueError("autotune expects an absolute path for the module")

    from collections import namedtuple

    kwargs = {
        "machine_class": None, "parallel_compile": False,
        "test_limit": None, "stop_after": None, "parallelism": 4,
        "pipelining": 0, "bail_threshold": 100, "no_dups": False,
        "seed_configuration": [], "results_log": None,
        "results_log_details": None, "quiet": False, "display_frequency": 10,
        "technique": None, "list_techniques": False,
        "generate_bandit_technique": False, "label": None,
        "print_search_space_size": False, "database": None,
        "print_params": False
    }

    args_t = namedtuple("MeasurementInterfaceArgs", sorted(kwargs.keys()))
    OpentunerTuner.main(args=args_t(**kwargs),
                        einsum=einsum, cl_ctx=cl_ctx, module_path=module_path)

# vim: fdm=marker
