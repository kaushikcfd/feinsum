import feinsum as fnsm
from pytools import memoize_on_first_arg
import pyopencl as cl
import numpy as np
import loopy as lp
import math
import opentuner
import sqlite3
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from functools import partial, cached_property
import logging

logger = logging.getLogger(__name__)

cl_ctx = cl.create_some_context()

DB_FILENAME = "dg_grad_p4_stage1_tuning.db"
DB_TABLENAME = "NVIDIA_TITAN_V"


@memoize_on_first_arg
def transform(t_unit, n_e_per_wg, nwork_items_per_e, i_tilelen, j_tilelen,
              prftch_u_to_local,
              insn_match=None, kernel_name=None):
    from loopy.match import parse_match

    within = parse_match(insn_match)
    knl = (t_unit[kernel_name]
           if kernel_name is not None
           else t_unit.default_entrypoint)
    insn_id, = [insn.id
                for insn in knl.instructions
                if within(knl, insn)]
    del knl

    expr = fnsm.einsum("ij,ej->ei",
                       fnsm.array((105, 35), "float64"),
                       fnsm.array((np.inf, 35), "float64"),
                       arg_names=["D", "u"])
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, expr, insn_match)
    vng = t_unit.default_entrypoint.get_var_name_generator()

    i = subst_map["i"]
    j = subst_map["j"]
    e = subst_map["e"]
    D = subst_map["D"]
    u = subst_map["u"]
    u_fetch = vng(u+"_prftch")
    e_inner, e_outer = vng(e+"_inner"), vng(e+"_outer")
    i_inner, i_tile = vng(i+"_inner"), vng(i+"_tile")
    j_inner, j_tile = vng(j+"_inner"), vng(j+"_tile")
    iprftch_D, jprftch_D = vng("iprftchD"), vng("jprftchD")
    i_inner_inner, i_inner_outer = vng(i_inner+"_inner"), vng(i_inner+"_outer")

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg,
                            inner_iname=e_inner, outer_iname=e_outer,
                            outer_tag="g.0", inner_tag="l.1")

    # {{{ tile to lower cache reuse distance of D/u

    t_unit = lp.split_iname(t_unit, i, i_tilelen,
                            inner_iname=i_inner, outer_iname=i_tile,
                            )
    t_unit = lp.split_iname(t_unit, j, j_tilelen,
                            inner_iname=j_inner, outer_iname=j_tile,
                            inner_tag="unr", outer_tag="unr"
                            )
    t_unit = lp.add_prefetch(t_unit, D, [i_inner, j_inner],
                             fetch_outer_inames=frozenset([e_outer,
                                                           i_tile,
                                                           j_tile]),
                             dim_arg_names=[iprftch_D, jprftch_D],
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             default_tag=None,
                             within=within)
    t_unit = lp.split_iname(t_unit, iprftch_D, n_e_per_wg,
                            inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, jprftch_D, nwork_items_per_e,
                            inner_tag="l.0")

    # }}}

    t_unit = lp.split_iname(t_unit, i_inner, nwork_items_per_e,
                            inner_tag="l.0", outer_tag="unr",
                            inner_iname=i_inner_inner,
                            outer_iname=i_inner_outer)

    if prftch_u_to_local:
        eprftch_u, jprftch_u = vng("eprftch_u"), vng("jprftch_u")
        t_unit = lp.add_prefetch(t_unit, u,
                                 sweep_inames=[e_inner, j_tile, j_inner],
                                 fetch_outer_inames=frozenset([e_outer]),
                                 temporary_address_space=lp.AddressSpace.LOCAL,
                                 temporary_name=u_fetch,
                                 dim_arg_names=[eprftch_u, jprftch_u],
                                 default_tag=None,
                                 )
        t_unit = lp.tag_inames(t_unit, {eprftch_u: "l.1"})
        t_unit = lp.split_iname(t_unit, jprftch_u, nwork_items_per_e,
                                inner_tag="l.0")
    else:
        t_unit = lp.add_prefetch(t_unit, u,
                                 sweep_inames=[j_tile, j_inner],
                                 fetch_outer_inames=frozenset([e_inner, e_outer]),
                                 temporary_address_space=lp.AddressSpace.PRIVATE,
                                 temporary_name=u_fetch,
                                 default_tag="unr",
                                 )
        # TODO: Yet another headache to ensure that the fetch instruction uses all
        # the hw axes.
        t_unit = lp.add_inames_to_insn(t_unit,
                                       i_inner_inner,
                                       f"writes:{u_fetch}")

    # {{{ TODO: remove once github.com/inducer/loopy/issues/666 is resolved.

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_id)

    acc_name = f"acc_{j_tile}_{j_inner}"
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  i_inner_outer,
                                                  only_var_names={acc_name})
    t_unit = lp.duplicate_inames(
        t_unit,
        i_inner_outer,
        within=f"writes:{acc_name} and not reads:{acc_name}")
    t_unit = lp.duplicate_inames(
        t_unit,
        i_inner_outer,
        within=f"reads:{acc_name} and not writes:{acc_name}")

    # }}}

    # t_unit = lp.set_options(t_unit, "write_code")

    return t_unit


class ConfigurationNotInDBError(LookupError):
    pass


def record_into_db(conn, i_tiles, j_tiles, n_e_per_wg,
                   nwork_items_per_e, prftch_u_to_local,
                   runtime):
    cursor = conn.cursor()
    # {{{ compute timestamp in Chicago

    import pytz
    from datetime import datetime

    timestamp = (datetime
                .now(pytz.timezone("America/Chicago")) .strftime("%Y_%m_%d_%H%M%S"))

    # }}}

    cursor.execute(f"INSERT INTO {DB_TABLENAME}"
                   " (i_tiles, j_tiles, n_e_per_wg,"
                   "  nwork_items_per_e, prftch_u_to_local,"
                   "  runtime_in_sec, timestamp)"
                   " VALUES ("
                   f"'{i_tiles}',"
                   f" '{j_tiles}',"
                   f" '{n_e_per_wg}',"
                   f" '{nwork_items_per_e}',"
                   f" '{prftch_u_to_local}',"
                   f" {runtime},"
                   f" '{timestamp}'"
                   ")")
    conn.commit()


def query_from_db(conn, i_tiles, j_tiles, n_e_per_wg,
                   nwork_items_per_e, prftch_u_to_local):
    cursor = conn.cursor()
    cursor.execute(" SELECT"
                   "     runtime_in_sec"
                   "  FROM "
                   f"    {DB_TABLENAME}"
                   f" WHERE ("
                   f"    i_tiles = {i_tiles}"
                   f"    AND j_tiles = {j_tiles}"
                   f"    AND n_e_per_wg = {n_e_per_wg}"
                   f"    AND nwork_items_per_e = {nwork_items_per_e}"
                   f"    AND prftch_u_to_local = {prftch_u_to_local}"
                   ");")
    prev_results = cursor.fetchall()
    if not prev_results:
        raise ConfigurationNotInDBError
    else:
        return min(prev_result[0] for prev_result in prev_results)


class TileSizesTuner(MeasurementInterface):

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
                           "i_tiles INT,"
                           "j_tiles INT,"
                           "n_e_per_wg INT,"
                           "nwork_items_per_e INT,"
                           "prftch_u_to_local INT,"
                           " runtime_in_sec REAL,"
                           " timestamp TEXT"
                           ")")
        return db

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """
        manipulator = ConfigurationManipulator()
        manipulator.add_parameter(
            IntegerParameter("i_tiles", 1, 20))
        manipulator.add_parameter(
            IntegerParameter("j_tiles", 1, 10))
        manipulator.add_parameter(
            IntegerParameter("prftch_u_to_local", 0, 0))
        manipulator.add_parameter(
            IntegerParameter("nwork_items_per_e", 1, 105))
        manipulator.add_parameter(
            IntegerParameter("n_e_per_wg", 1, 32))
        return manipulator

    def seed_configurations(self):
        cursor = self.conn.cursor()
        cursor.execute(" SELECT"
                       "     i_tiles,"
                       "     j_tiles,"
                       "     n_e_per_wg,"
                       "     nwork_items_per_e,"
                       "     prftch_u_to_local"
                       "  FROM "
                       f"    {DB_TABLENAME}"
                       ";")
        configs = cursor.fetchall()
        return [
            {"i_tiles": config[0], "j_tiles": config[1],
             "n_e_per_wg": config[2], "nwork_items_per_e": config[3],
             "prftch_u_to_local": config[4]}
            for config in configs
            if config[4] == 0
        ]

    def run(self, desired_result, input, limit):

        cfg = desired_result.configuration.data
        if cfg["n_e_per_wg"] * cfg["nwork_items_per_e"] > 500:
            return Result(time=np.inf)

        logger.info(cfg)

        # {{{ query from DB

        try:
            result = query_from_db(self.conn, **cfg)
        except ConfigurationNotInDBError:
            pass
        else:
            logger.info("DB Hit")
            return Result(time=result)

        # }}}

        specialized_transform = partial(transform,
                                        n_e_per_wg=cfg["n_e_per_wg"],
                                        nwork_items_per_e=cfg["nwork_items_per_e"],
                                        i_tilelen=math.ceil(105/cfg["i_tiles"]),
                                        j_tilelen=math.ceil(35/cfg["j_tiles"]),
                                        prftch_u_to_local=cfg["prftch_u_to_local"],
                                        )

        expr = fnsm.einsum("ij,ej->ei",
                           fnsm.array((105, 35), "float64"),
                           fnsm.array((np.inf, 35), "float64"),
                           arg_names=["D", "u"])
        print(fnsm.stringify_comparison_vs_roofline(
            expr,
            transform=specialized_transform,
            cl_ctx=cl_ctx,
        ))
        runtime = fnsm.timeit(expr,
                              cl_ctx=cl_ctx,
                              transform=specialized_transform)
        record_into_db(self.conn, cfg["i_tiles"], cfg["j_tiles"],
                       cfg["n_e_per_wg"],
                       cfg["nwork_items_per_e"], cfg["prftch_u_to_local"],
                       runtime)

        return Result(time=runtime)


if __name__ == "__main__":
    from feinsum.data.device_info import DEV_TO_PEAK_GFLOPS

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
    elif cl_ctx.devices[0].name not in DEV_TO_PEAK_GFLOPS:
        logger.info(f"Device {cl_ctx.devices[0]} not known to database.")
    else:
        if 1:
            argparser = opentuner.default_argparser()
            TileSizesTuner.main(argparser.parse_args())
        else:
            # enable for debugging
            specialized_transform = partial(transform,
                                            n_e_per_wg=25,
                                            nwork_items_per_e=18,
                                            i_tilelen=math.ceil(105/6),
                                            j_tilelen=math.ceil(35/1),
                                            prftch_u_to_local=0,
                                            )

            expr = fnsm.einsum("ij,ej->ei",
                               fnsm.array((105, 35), "float64"),
                               fnsm.array((np.inf, 35), "float64"),
                               arg_names=["D", "u"])
            print(fnsm.stringify_comparison_vs_roofline(
                expr,
                transform=specialized_transform,
                cl_ctx=cl_ctx,
                long_dim_length=100_000
            ))
