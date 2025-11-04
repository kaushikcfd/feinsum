from __future__ import annotations

__doc__ = """
.. autoclass:: TuningParameter
.. autoclass:: IntParameter
.. autoclass:: BoolParameter
.. autoclass:: TupleParameter

.. autofunction:: autotune
.. autofunction:: transform_param
.. autofunction:: einsum_arg

.. class:: ConvertibleToTuningParamT

    A type alias for ``Union[TuningParameter, Tuple[Any, ...]]``.
    See :attr:`transform_param.func`.
"""
import abc
import logging
import os
import sqlite3
from dataclasses import dataclass
from functools import cache, cached_property
from typing import (
    TYPE_CHECKING,
    Any,
)

import numpy as np
import opentuner
from immutables import Map

from feinsum.einsum import INT_CLASSES, BatchedEinsum, IntegralT, ShapeComponentT
from feinsum.sql_utils import DEFAULT_DB, TIMINGS_TABLENAME

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import loopy as lp
    import pyopencl as cl

    from feinsum.typing import TransformT


logger = logging.getLogger(__name__)


# {{{ supported tuning parameters


class TuningParameter(abc.ABC):  # noqa: B024
    """
    Records the parameter space of a code-transformation.

    .. note::

        This is an abstract class.
    """


@dataclass(frozen=True, init=False)
class IntParameter(TuningParameter):
    """
    A parameter that takes values in the range ``[low, high]``.
    """

    low: IntegralT
    high: IntegralT

    def __init__(self, low: ShapeComponentT, high: ShapeComponentT):
        if not isinstance(low, INT_CLASSES):
            raise ValueError("low must be an integer")

        if not isinstance(high, INT_CLASSES):
            raise ValueError("high must be an integer")

        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)


@dataclass(frozen=True)
class BoolParameter(TuningParameter):
    """
    A parameter that can be either *True* or *False*.
    """


@dataclass(frozen=True)
class TupleParameter(TuningParameter):
    """
    A tuple of parameters. The resulting parameter space is a Cartesian product of
    the individual elements of the tuple.
    """

    _data: tuple[TuningParameter, ...]


# }}}


ConvertibleToTuningParamT = tuple[Any, ...] | TuningParameter


# {{{ einsum_arg/transform_param


@dataclass(frozen=True, repr=True)
class einsum_arg:  # noqa: N801
    """
    Decorate to a template transformation to inform
    :func:`autotune` about a static argument to the transformation
    implementation.

    :param arg_name: Name of the argument in the template transform
        that is to be statically set.
    :param param_getter: A callable that expects an einsum and returns the
        value of the static argument.
    """

    var_name: str
    func: Callable[[BatchedEinsum], Any]

    def __call__(self, fn: Callable[..., Any]) -> ParametrizedTransform:
        if isinstance(fn, ParametrizedTransform):
            return ParametrizedTransform(
                fn.transform,
                (self, *fn.einsum_derivative_args),
                fn.transform_params,
            )
        else:
            from functools import cache

            return ParametrizedTransform(cache(fn), (self,), ())


@dataclass(frozen=True, repr=True)
class transform_param:  # noqa: N801
    """
    Decorate to a template transformation to inform
    :func:`autotune` about the parameter space.

    :param arg_name: Name of the argument in the template
        transformation to be parametrized.
    :param func: A callable that expects an einsum and returns the
        parameter space for that einsum. The returned parameter space could be an
        instance of:
        - A :class:`TuningParameter`.
        - A :class:`tuple` of :class:`ConvertibleToTuningParamT` types that is
          (internally) mapped to a :class:`TupleParameter`.
    """

    var_name: str
    func: Callable[[BatchedEinsum], ConvertibleToTuningParamT]

    def __call__(
        self, fn: Callable[..., lp.TranslationUnit]
    ) -> ParametrizedTransform:
        if isinstance(fn, ParametrizedTransform):
            return ParametrizedTransform(
                fn.transform,
                fn.einsum_derivative_args,
                (self, *fn.transform_params),
            )
        else:
            from functools import cache

            return ParametrizedTransform(cache(fn), (), (self,))


@dataclass(frozen=True, repr=True)
class ParametrizedTransform:
    transform: Callable[..., lp.TranslationUnit]
    einsum_derivative_args: tuple[einsum_arg, ...]
    transform_params: tuple[transform_param, ...]

    def __call__(self, *args: Any, **kwargs: Any) -> lp.TranslationUnit:
        return self.transform(*args, **kwargs)

    def bind_args(self, einsum: BatchedEinsum, **transform_args: Any) -> TransformT:
        """
        Binds *transform_args* to *self* and returns a python callable
        to the corresponding instance in the self space.
        """
        from functools import partial

        py_clbl = partial(
            self.transform,
            **{
                arg.var_name: arg.func(einsum) for arg in self.einsum_derivative_args
            },
            **transform_args,
        )
        return py_clbl


# }}}


@cache
def _get_impls_path() -> str:
    import importlib.util

    feinsum_spec = importlib.util.find_spec("feinsum")
    assert feinsum_spec is not None
    assert feinsum_spec.origin is not None

    return os.path.abspath(
        os.path.join(feinsum_spec.origin, os.path.pardir, "tuning", "impls")
    )


class ConfigurationNotInDBError(LookupError):
    pass


def get_transform_func_from_module_path(module_path: str) -> ParametrizedTransform:
    from importlib import util

    _, filename = os.path.split(module_path)

    assert filename.endswith(".py")

    spec = util.spec_from_file_location(filename[:-3], module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            "Could not import 'transform' function" f" from {module_path}."
        )
    transform_module = util.module_from_spec(spec)
    spec.loader.exec_module(transform_module)
    transform_obj = transform_module.transform

    if isinstance(transform_obj, ParametrizedTransform):
        return transform_obj
    else:
        assert callable(transform_obj)
        return ParametrizedTransform(transform_obj, (), ())


def _convert_to_tuning_param(param: ConvertibleToTuningParamT) -> TuningParameter:
    if isinstance(param, TuningParameter):
        return param
    elif isinstance(param, tuple):
        return TupleParameter(tuple(_convert_to_tuning_param(el) for el in param))
    else:
        raise ValueError(
            "Only instances of ConvertibleToTuningParamT" " are supported."
        )


def _get_opentuner_param_name(key: tuple[str, ...]) -> str:
    return "_".join(key)


def _get_opentuner_params_from_tuning_param(
    key: tuple[str, ...], tuning_param: TuningParameter
) -> frozenset[opentuner.manipulator.Parameter]:
    if isinstance(tuning_param, IntParameter):
        from opentuner.search.manipulator import IntegerParameter

        return frozenset(
            [
                IntegerParameter(
                    _get_opentuner_param_name(key),
                    tuning_param.low,
                    tuning_param.high,
                )
            ]
        )
    elif isinstance(tuning_param, BoolParameter):
        from opentuner.search.manipulator import BooleanParameter

        return frozenset([BooleanParameter(_get_opentuner_param_name(key))])
    elif isinstance(tuning_param, TupleParameter):
        from functools import reduce

        return reduce(
            frozenset.union,
            (
                _get_opentuner_params_from_tuning_param((*key, f"_fetup_{i}"), k)
                for i, k in enumerate(tuning_param._data)
            ),
            frozenset(),
        )
    elif isinstance(tuning_param, TuningParameter):
        raise NotImplementedError(type(tuning_param))
    else:
        raise TypeError(type(tuning_param))


def _reconstruct_transform_params_from_opentuner_config(
    config: Mapping[str, Any],
    transform_params: tuple[transform_param, ...],
    ensm: BatchedEinsum,
) -> Mapping[str, Any]:

    def rec(key: tuple[str, ...], tuning_param: TuningParameter) -> Any:
        if isinstance(tuning_param, (IntParameter, BoolParameter)):
            param_name = _get_opentuner_param_name(key)
            return config[param_name]
        elif isinstance(tuning_param, TupleParameter):
            return tuple(
                rec((*key, f"_fetup_{isubparam}"), subparam)
                for isubparam, subparam in enumerate(tuning_param._data)
            )
        elif isinstance(tuning_param, TuningParameter):
            raise NotImplementedError(type(tuning_param))
        else:
            raise TypeError(type(tuning_param))

    return {
        param.var_name: rec(
            (param.var_name,), _convert_to_tuning_param(param.func(ensm))
        )
        for param in transform_params
    }


def _get_opentuner_config_from_transform_config(
    config: Mapping[str, Any],
) -> Map[str, Any]:
    result = {}

    def rec(key: tuple[str, ...], param: Any) -> None:
        if isinstance(param, (INT_CLASSES, bool)):
            result[_get_opentuner_param_name(key)] = param
            return
        elif isinstance(param, tuple):
            for ipar, par in enumerate(param):
                rec((*key, f"_fetup_{ipar}"), par)
        else:
            raise NotImplementedError(type(param))

    for k, v in config.items():
        rec((k,), v)

    return Map(result)


# {{{ Opentuner entrypoint


# type-ignored as we are sub-classing Any type.
class OpentunerTuner(opentuner.MeasurementInterface):  # type: ignore[misc]
    def __init__(
        self,
        args: Any,
        einsum: BatchedEinsum,
        cl_ctx: cl.Context,
        module_path: str,
        long_dim_length: int,
        db_path: str,
        *,
        # Args to super class ->
        project_name: str | None = None,
        program_name: str = "unknown",
        program_version: str = "unknown",
        manipulator: Any | None = None,
        objective: Any | None = None,
        input_manager: Any | None = None,
    ) -> None:
        from feinsum.canonicalization import canonicalize_einsum

        super().__init__(
            args=args,
            project_name=project_name,
            program_name=program_name,
            program_version=program_version,
            manipulator=manipulator,
            objective=objective,
            input_manager=input_manager,
        )

        self.einsum = canonicalize_einsum(einsum)
        self.cl_ctx = cl_ctx
        self.module_path = module_path
        self.long_dim_length = long_dim_length
        self.db_path = db_path

    @cached_property
    def transform_space_id(self) -> str:
        dirpath, filepath = os.path.split(self.module_path)

        if dirpath == _get_impls_path():
            return filepath
        else:
            return self.module_path

    @cached_property
    def conn(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path)
        from feinsum.sql_utils import _create_timings_table_if_non_existent

        _create_timings_table_if_non_existent(db)
        return db

    @cached_property
    def transform_func(self) -> ParametrizedTransform:
        return get_transform_func_from_module_path(self.module_path)

    def manipulator(self) -> opentuner.ConfigurationManipulator:
        from opentuner import ConfigurationManipulator

        manipulator = ConfigurationManipulator()

        for param in self.transform_func.transform_params:
            unprocessed_tuning_param = param.func(self.einsum)
            tuning_param = _convert_to_tuning_param(unprocessed_tuning_param)
            for opentuner_param in _get_opentuner_params_from_tuning_param(
                (param.var_name,), tuning_param
            ):
                manipulator.add_parameter(opentuner_param)

        return manipulator

    def seed_configurations(self) -> Sequence[Mapping[str, Any]]:
        from feinsum.sql_utils import (
            dump_arg_names,
            dump_arg_to_dtype,
            dump_device_name,
            dump_index_to_length,
            dump_op_info,
            load_transform_params,
        )

        cursor = self.conn.cursor()
        subscripts = self.einsum.get_subscripts()
        index_to_length = dump_index_to_length(self.einsum)
        arg_names = dump_arg_names(self.einsum)
        op_info = dump_op_info(self.einsum, self.long_dim_length)
        arg_to_dtype = dump_arg_to_dtype(self.einsum)
        (dev,) = self.cl_ctx.devices
        device_name = dump_device_name(dev)

        cursor.execute(
            " SELECT"
            "     transform_params"
            "  FROM "
            f"    {TIMINGS_TABLENAME}"
            " WHERE ("
            "    transform_id = ?"
            "    AND subscripts = ?"
            "    AND index_to_length = ?"
            "    AND args = ?"
            "    AND arg_to_dtype = ?"
            "    AND giga_op_info = ?"
            "    AND device_name = ?"
            ");",
            (
                self.transform_space_id,
                subscripts,
                index_to_length,
                arg_names,
                arg_to_dtype,
                op_info,
                device_name,
            ),
        )

        return [
            dict(
                _get_opentuner_config_from_transform_config(
                    load_transform_params(transform_params[0])
                )
            )
            for transform_params in cursor.fetchall()
        ]

    def query_from_db(self, parameters: Mapping[str, Any]) -> float:
        import json

        from feinsum.sql_utils import (
            dump_arg_names,
            dump_arg_to_dtype,
            dump_device_name,
            dump_index_to_length,
            dump_op_info,
        )

        cursor = self.conn.cursor()
        subscripts = self.einsum.get_subscripts()
        index_to_length = dump_index_to_length(self.einsum)
        arg_names = dump_arg_names(self.einsum)
        arg_to_dtype = dump_arg_to_dtype(self.einsum)
        transform_params_str = json.dumps(parameters, sort_keys=True)
        op_info = dump_op_info(self.einsum, self.long_dim_length)
        (dev,) = self.cl_ctx.devices
        device_name = dump_device_name(dev)

        cursor.execute(
            " SELECT"
            "     runtime_in_sec"
            "  FROM "
            f"    {TIMINGS_TABLENAME}"
            " WHERE ("
            "    subscripts = ?"
            "    AND index_to_length = ?"
            "    AND args = ?"
            "    AND arg_to_dtype = ?"
            "    AND transform_params = ?"
            "    AND giga_op_info = ?"
            "    AND device_name = ?"
            ");",
            (
                subscripts,
                index_to_length,
                arg_names,
                arg_to_dtype,
                transform_params_str,
                op_info,
                device_name,
            ),
        )
        stored_results: list[Sequence[float]] = cursor.fetchall()

        if not stored_results:
            raise ConfigurationNotInDBError
        else:
            return min(stored_result[0] for stored_result in stored_results)

    def record_into_db(self, runtime: float, parameters: Mapping[str, Any]) -> None:
        from feinsum.sql_utils import record_into_db

        record_into_db(
            self.einsum,
            self.cl_ctx,
            self.module_path,
            parameters,
            self.conn,
            self.long_dim_length,
        )

    def run(
        self, desired_result: opentuner.DesiredResult, input: Any, limit: Any
    ) -> opentuner.Result:
        from feinsum.diagnostics import InvalidParameterError
        from feinsum.measure import stringify_comparison_vs_roofline, timeit

        cfg = _reconstruct_transform_params_from_opentuner_config(
            desired_result.configuration.data,
            self.transform_func.transform_params,
            self.einsum,
        )

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

        bound_transform = self.transform_func.bind_args(self.einsum, **cfg)

        try:
            logger.info(
                "\n"
                + stringify_comparison_vs_roofline(
                    self.einsum,
                    transform=bound_transform,
                    cl_ctx=self.cl_ctx,
                    long_dim_length=self.long_dim_length,
                )
            )
        except InvalidParameterError as err:
            logger.info(f"Ignored configuration due to '{err}'.")
            return opentuner.Result(time=np.inf)

        runtime = timeit(
            self.einsum,
            cl_ctx=self.cl_ctx,
            transform=bound_transform,
            long_dim_length=self.long_dim_length,
        )
        self.record_into_db(runtime, cfg)

        return opentuner.Result(time=runtime)


# }}}


def autotune(
    einsum: BatchedEinsum,
    module_path: str,
    cl_ctx: cl.Context,
    *,
    db_path: str | None = None,
    long_dim_length: int = 100_000,
    stop_after: int | None = None,
) -> None:
    """
    For a transform space specified in *module_path*, searches the parameter
    space and records the timing results for each run in *db_path*.

    :param stop_after: After these many trials the routine exits. Pass *None*
        to go on indefinitely.
    """
    if not os.path.isabs(module_path):
        raise ValueError("autotune expects an absolute path for the module")

    if db_path is None:
        db_path = DEFAULT_DB

    from argparse import Namespace  # :puke: but required by opentuner. Big brain.

    kwargs: Mapping[str, Any] = {
        "machine_class": None,
        "parallel_compile": False,
        "test_limit": None,
        "stop_after": stop_after,
        "parallelism": 4,
        "pipelining": 0,
        "bail_threshold": 100,
        "no_dups": True,
        "seed_configuration": [],
        "results_log": None,
        "results_log_details": None,
        "quiet": False,
        "display_frequency": 10,
        "technique": None,
        "list_techniques": False,
        "generate_bandit_technique": False,
        "label": None,
        "print_search_space_size": False,
        "database": None,
        "print_params": False,
    }

    OpentunerTuner.main(
        args=Namespace(**kwargs),
        einsum=einsum,
        cl_ctx=cl_ctx,
        module_path=module_path,
        db_path=db_path,
        long_dim_length=long_dim_length,
    )


# vim: fdm=marker
