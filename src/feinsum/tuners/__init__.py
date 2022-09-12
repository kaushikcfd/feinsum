import numpy as np
import opentuner
import abc
import pyopencl as cl

from typing import Callable, Any, Tuple
from dataclasses import dataclass
from functools import cached_property
from feinsum.einsum import FusedEinsum, SizeParam
from feinsum.utils import (has_similar_subscript,
                           is_any_redn_dim_parametric)
from feinsum.einsum import FusedEinsum


class TuningParameter(abc.ABC):
    pass


class IntParameter(TuningParameter):
    """
    A parameter that takes values in ``[low, high)``.
    """
    low: int
    high: int


class BoolParameter(TuningParameter):
    """
    A parameter that can be either *True* or *False*.
    """


class EinsumDerivativeArg:
    var_name: str
    func: Callable[[FusedEinsum], Any]


class TransformArg:
    var_name: str
    func: Callable[[FusedEinsum], TuningParameter]


class ParametrizedTransform:
    transform: Callable
    einsum_derivative_args: Tuple[...]
    transform_args: Tuple[...]


@dataclass(frozen=True, init=False)
class BatchedEinsumTuner(MeasurementInterface):
    einsum: FusedEinsum
    template_transform: ParametrizedTransform
    db_name: str

    def __init__(self, einsum, template_transform, db_name):
        self.einsum = einsum
        self.template_transform = template_transform
        self.db_name = db_name
        super().__init__()

    def get_str(self,
                transform_parameters: Mapping[str, Any]):
        assert (set(transform_parameters.keys())
                == {arg.var_name
                    for arg in self.template_transform.transform_args})
    
        raise NotImplementedError



class OpentunerMeasurementInterface(opentuner.MeasurementInterface):
    def __init__(self,
                 einsum: FusedEinsum,
                 cl_ctx: cl.Context,
                 parametrized_transform: ParametrizedTransform,
                 db_filename: str,
                 db_tablename: str,
                 ):
        kwargs = {
            "machine_class": None, "parallel_compile": False, "test_limit": None,
            "stop_after": None, "parallelism": 4, "pipelining": 0,
            "bail_threshold": 500, "no_dups": False, "seed_configuration": [],
            "results_log": None, "results_log_details": None, "quiet": False,
            "display_frequency": 10, "technique": None,
            "list_techniques": False, "generate_bandit_technique": False, "label": None,
            "print_search_space_size": False, "database": None,
            "print_params": False}
        super().__init__(**kwargs)

        self.einsum = einsum
        self.cl_ctx = cl_ctx
        self.parametrized_transform = parametrized_transform
        self.db_filename = db_filename
        self.db_tablename = db_tablename

    @cached_property
    def sql_table_name(self):
        from feinsum.database import _get_cl_device_name_for_db
        dev, = self.cl_ctx.devices
        device_name = _get_cl_device_name_for_db(dev)
        ...

    @cached_property
    def conn(self):
        db = sqlite3.connect(DB_FILENAME)
        cursor = db.cursor()
        cursor.execute(" SELECT name FROM sqlite_master"
                       f" WHERE (type='table' AND name='{DB_TABLENAME}');")

        if not cursor.fetchall():
            # device table not available
            logger.info(f"Table {DB_TABLENAME} not in DB, creating one.")
            cursor.execute(f"CREATE TABLE {DB_TABLENAME} ("
                           " ID INTEGER PRIMARY KEY AUTOINCREMENT,"
                           "n_i_tile INT,"
                           "n_j_tile INT,"
                           "n_stmt_tile INT,"
                           "n_e_per_wg INT,"
                           "nwork_items_per_e INT,"
                           " runtime_in_sec REAL,"
                           " timestamp TEXT"
                           ")")
        return db





def tune(einsum: FusedEinsum, cl_cltx: cl.Context, online_tuning: bool = False):
    if (has_similar_subscript(einsum, "xre,rij,ej->xei")
            and np.isscalar(einsum.shape[0])
            and np.isscalar(einsum.shape[2])
            and einsum.arg_shapes[-1][-1] == einsum.shape[-1]
            and isinstance(einsum.shape[1], SizeParam)
            and not is_any_redn_dim_parametric(einsum)):
        from .impls.xre_rij_ej_to_xei import transform,
        db_name = "xre_rij_ej_to_xei.db"
        tuner = get_tuner(einsum, cl_ctx, transform, db_name)

        if online_tuning:
            # TODO: think of some way to express the stopping criteria
            tuner.tune()

        from feinsum.database import _get_clbl_from_string
        py_code = tuner.generate_code()
        return _get_clbl_from_string(py_code)
    else:
        raise NotImplementedError(f"Tuner for {einsum}")


# vim: fdm=marker
