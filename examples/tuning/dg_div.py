import feinsum as fnsm
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
from functools import cache

logger = logging.getLogger(__name__)

cl_ctx = cl.create_some_context()

Ndim = 3
Ndof = 35

DB_FILENAME = "wave_div_3d_p4.db"
DB_TABLENAME = "NVIDIA_TITAN_V"


@cache
def transform(t_unit, n_e_per_wg, nwork_items_per_e,
              i_tiles, j_tiles,
              insn_match=None, kernel_name=None):
    from loopy.match import parse_match

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = parse_match(insn_match)
    knl = t_unit[kernel_name]
    insn_id, = [insn.id
                for insn in knl.instructions
                if within(knl, insn)]
    del knl

    ref_einsum = fnsm.einsum("xre,rij,xej->ei",
                             fnsm.array((Ndim, Ndim, np.inf), "float64"),
                             fnsm.array((Ndim, Ndof, Ndof), "float64"),
                             fnsm.array((Ndim, np.inf, Ndof), "float64"),
                             arg_names=["J", "D", "u"])

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    ing = t_unit.default_entrypoint.get_var_name_generator()
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match)
    i = subst_map["i"]
    j = subst_map["j"]
    J = subst_map["J"]
    e = subst_map["e"]
    D = subst_map["D"]
    u = subst_map["u"]
    x = subst_map["x"]
    r = subst_map["r"]

    j_inner, j_tile = vng(f"{j}_inner"), vng(f"{j}_tile")
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
    u_fetch = vng(f"{u}_fetch")
    i_inner, i_tile = vng(f"{i}_inner"), vng(f"{i}_tile")
    i_inner_inner, i_inner_outer = (vng(f"{i_inner}_inner"),
                                    vng(f"{i_inner}_outer"))
    rprftchD, iprftchD, jprftchD = (vng(f"{r}prftchD"),
                                    vng(f"{i}prftchD"),
                                    vng(f"{j}prftchD"))
    D_fetch = vng(f"{D}_fetch")
    prcmpt_x_redn = ing(f"prcmpt_{x}_redn")
    e_prcmpt_subst, r_prcmpt_subst, j_prcmpt_subst = (vng(f"{e}prcmpt_subst"),
                                                      vng(f"{r}prcmpt_subst"),
                                                      vng(f"{j}prcmpt_subst"))

    j_prcmpt_subst_inner, j_prcmpt_subst_outer = (vng(f"{j_prcmpt_subst}_inner"),
                                                  vng(f"{j_prcmpt_subst}_outer"))

    # }}}

    # {{{ term hoisting to match the flop count of opt_einsum

    t_unit = lp.split_reduction_inward(t_unit, x)
    t_unit = fnsm.hoist_reduction_invariant_terms(t_unit, x)
    t_unit = fnsm.extract_einsum_terms_as_subst(
        t_unit,
        f"subst({e}, {j}, {r})",
        f"sum({x}, {J}[{x}, {r}, {e}]*{u}[{x}, {e}, {j}])",
        insn_match=insn_match
    )

    t_unit = lp.split_iname(t_unit, j, math.ceil(Ndof/j_tiles),
                            outer_iname=j_tile, inner_iname=j_inner)

    # }}}

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg,
                            inner_iname=e_inner, outer_iname=e_outer,
                            inner_tag="l.1", outer_tag="g.0")

    t_unit = lp.add_prefetch(t_unit, J,
                             sweep_inames=[x, r],
                             fetch_outer_inames=frozenset({e_outer, e_inner,
                                                           i_inner_inner}),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             default_tag="unr",
                             within=within)

    # {{{ tile and prefetch D

    t_unit = lp.split_iname(t_unit, i, math.ceil(Ndof/i_tiles),
                            inner_iname=i_inner, outer_iname=i_tile,
                            outer_tag="unr"
                            )
    t_unit = lp.add_prefetch(t_unit, D, [i_inner, r, j_inner],
                             fetch_outer_inames=frozenset([e_outer,
                                                           i_tile,
                                                           j_tile]),
                             dim_arg_names=[rprftchD, iprftchD, jprftchD],
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             temporary_name=D_fetch,
                             default_tag=None)
    t_unit = lp.split_iname(t_unit, iprftchD, n_e_per_wg, inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, jprftchD, nwork_items_per_e, inner_tag="l.0")

    # }}}

    # {{{ precompute 'subst'

    t_unit = lp.precompute(t_unit, "subst",
                           sweep_inames=[r, j_inner, e_inner],
                           precompute_inames=[e_prcmpt_subst,
                                              j_prcmpt_subst,
                                              r_prcmpt_subst],
                           precompute_outer_inames=frozenset({e_outer,
                                                              i_tile,
                                                              j_tile}),
                           default_tag=None,
                           compute_insn_id=prcmpt_x_redn,
                           temporary_address_space=lp.AddressSpace.LOCAL)
    t_unit = lp.tag_inames(t_unit, {e_prcmpt_subst: "l.1"})

    # TODO: It might be worth exploring joining 'r_prcmpt_subst',
    # 'j_prcmpt_subst'.

    t_unit = lp.split_iname(t_unit, j_prcmpt_subst, nwork_items_per_e,
                            inner_iname=j_prcmpt_subst_inner,
                            outer_iname=j_prcmpt_subst_outer,
                            inner_tag="l.0",
                            outer_tag="unr"
                            )

    # }}}

    t_unit = lp.split_iname(t_unit, i_inner, nwork_items_per_e,
                            inner_iname=i_inner_inner,
                            outer_iname=i_inner_outer,
                            inner_tag="l.0",
                            outer_tag="unr",
                            )
    t_unit = lp.add_prefetch(t_unit, u,
                             sweep_inames=[x, j_prcmpt_subst_outer],
                             fetch_outer_inames=frozenset([j_prcmpt_subst_inner,
                                                           e_prcmpt_subst, e_outer,
                                                           j_tile]),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name=u_fetch,
                             # default_tag=None,
                             default_tag="unr",
                             )

    # {{{ TODO: remove once github.com/inducer/loopy/issues/666 is resolved.

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_id)
    inames_to_duplicate = (frozenset({i_tile, i_inner_outer})
                           & t_unit[kernel_name].all_inames())
    acc_name = f"acc_{r}_{j_tile}_{j_inner}"
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  inames_to_duplicate,
                                                  only_var_names={acc_name})

    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"writes:{acc_name} and not reads:{acc_name}")
    t_unit = lp.duplicate_inames(
        t_unit,
        inames_to_duplicate,
        within=f"reads:{acc_name} and not writes:{acc_name}")

    # }}}

    t_unit = lp.tag_inames(t_unit, {r: "unr", x: "unr"})

    return t_unit


class ConfigurationNotInDBError(LookupError):
    pass


def record_into_db(conn, i_tiles, j_tiles, n_e_per_wg,
                   nwork_items_per_e,
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
                   "  nwork_items_per_e,"
                   "  runtime_in_sec, timestamp)"
                   " VALUES ("
                   f"'{i_tiles}',"
                   f" '{j_tiles}',"
                   f" '{n_e_per_wg}',"
                   f" '{nwork_items_per_e}',"
                   f" {runtime},"
                   f" '{timestamp}'"
                   ")")
    conn.commit()


def query_from_db(conn, i_tiles, j_tiles, n_e_per_wg,
                   nwork_items_per_e):
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
            IntegerParameter("nwork_items_per_e", 1, Ndof))
        manipulator.add_parameter(
            IntegerParameter("n_e_per_wg", 2, 32))
        return manipulator

    def seed_configurations(self):
        cursor = self.conn.cursor()
        cursor.execute(" SELECT"
                       "     i_tiles,"
                       "     j_tiles,"
                       "     n_e_per_wg,"
                       "     nwork_items_per_e"
                       "  FROM "
                       f"    {DB_TABLENAME}"
                       ";")
        configs = cursor.fetchall()
        return [
            {"i_tiles": config[0], "j_tiles": config[1],
             "n_e_per_wg": config[2], "nwork_items_per_e": config[3]}
            for config in configs
        ]

    def run(self, desired_result, input, limit):

        cfg = desired_result.configuration.data

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

        if cfg["n_e_per_wg"] * cfg["nwork_items_per_e"] > 600:
            logger.info("Block dimension limit exceeded => ignored configuration.")
            return Result(time=np.inf)

        nkbs_in_local_mem = (
            Ndim * math.ceil(Ndof/cfg["i_tiles"]) * math.ceil(Ndof/cfg["j_tiles"])
            + Ndim * cfg["n_e_per_wg"] * math.ceil(Ndof/cfg["j_tiles"]))*8e-3

        if nkbs_in_local_mem > 47:
            logger.info("Shared memory limit exceeded => ignored configuration.")
            return Result(time=np.inf)

        specialized_transform = partial(transform,
                                        n_e_per_wg=cfg["n_e_per_wg"],
                                        nwork_items_per_e=cfg["nwork_items_per_e"],
                                        i_tiles=cfg["i_tiles"],
                                        j_tiles=cfg["j_tiles"],
                                        )

        expr = fnsm.einsum("xre,rij,xej->ei",
                           fnsm.array((Ndim, Ndim, np.inf), "float64"),
                           fnsm.array((Ndim, Ndof, Ndof), "float64"),
                           fnsm.array((Ndim, np.inf, Ndof), "float64"),
                           arg_names=["J", "D", "u"])
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
                       cfg["nwork_items_per_e"],
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
            # Enable for debugging
            expr = fnsm.einsum("xre,rij,xej->ei",
                               fnsm.array((Ndim, Ndim, np.inf), "float64"),
                               fnsm.array((Ndim, Ndof, Ndof), "float64"),
                               fnsm.array((Ndim, np.inf, Ndof), "float64"),
                               arg_names=["J", "D", "u"])

            specialized_transform = partial(transform,
                                            n_e_per_wg=32,
                                            nwork_items_per_e=4,
                                            i_tiles=1, j_tiles=4,
                                            )

            print(fnsm.stringify_comparison_vs_roofline(
                expr,
                transform=specialized_transform,
                cl_ctx=cl_ctx,
            ))
