import feinsum as fnsm
import pyopencl as cl
import numpy as np
import loopy as lp
import math
import opentuner
import sqlite3
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner.search.manipulator import BooleanParameter
from opentuner import MeasurementInterface
from opentuner import Result
from functools import partial, cached_property
import logging

logger = logging.getLogger(__name__)

cl_ctx = cl.create_some_context()

Ndim = 3
Ndof = 35

DB_FILENAME = "wave_grad_3d_p4.db"
DB_TABLENAME = "NVIDIA_TITAN_V"


def transform(t_unit, n_e_per_wg, nwork_items_per_e,
              prftch_u_to_local, i_tiles, j_tiles,
              insn_match=None, kernel_name=None):
    from loopy.match import parse_match

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = parse_match(insn_match)
    knl = t_unit[kernel_name]
    insn_id, = [insn.id
                for insn in knl.instructions
                if within(knl, insn)]
    del knl

    ref_einsum = fnsm.einsum("xre,rij,ej->xei",
                             fnsm.array((Ndim, Ndim, np.inf), "float64"),
                             fnsm.array((Ndim, Ndof, Ndof), "float64"),
                             fnsm.array((np.inf, Ndof), "float64"),
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
    e_inner, e_outer = f"{e}_inner", f"{e}_outer"
    u_fetch = vng(u+"_prftch")
    j_inner = vng(f"{j}_inner")
    j_tile = f"{j}_tile"
    i_tile, i_inner = f"{i}_tile", f"{i}_inner"
    r_prcmpt_subst = vng(f"{r}_prcmpt")
    i_prcmpt_subst = vng(f"{i}_prcmpt")
    rprftch_D, iprftch_D, jprftch_D = (vng("rprftchD"),
                                       vng("iprftchD"),
                                       vng("jprftchD"))

    prcmpt_j_redn = ing(f"prcmpt_{j}_redn")
    D_fetch = vng(f"{D}_fetch")
    J_fetch = vng(f"{J}_fetch")
    i_inner_inner, i_inner_outer = vng(f"{i_inner}_inner"), vng(f"{i_inner}_outer")
    # prftch_J = ing(f"prftch_{J}")

    # }}}

    # {{{ term hoisting to match the flop count of opt_einsum

    t_unit = lp.split_reduction_inward(t_unit, j)
    t_unit = fnsm.hoist_reduction_invariant_terms(t_unit, j)
    t_unit = fnsm.extract_einsum_terms_as_subst(
        t_unit,
        f"subst({e}, {i}, {r})",
        f"sum({j}, {D}[{r}, {i}, {j}]*{u}[{e}, {j}])",
        insn_match=insn_match
    )

    t_unit = lp.split_iname(t_unit, i, math.ceil(Ndof/i_tiles),
                            outer_iname=i_tile, inner_iname=i_inner)

    # }}}

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg,
                            inner_iname=e_inner, outer_iname=e_outer,
                            inner_tag="l.1", outer_tag="g.0")

    # {{{ prefetch 'u'

    if prftch_u_to_local:
        eprftch_u, jprftch_u = vng("eprftch_u"), vng("jprftch_u")
        t_unit = lp.add_prefetch(t_unit, u,
                                 sweep_inames=[e_inner, j],
                                 fetch_outer_inames=frozenset([e_outer]),
                                 temporary_address_space=lp.AddressSpace.LOCAL,
                                 temporary_name=u_fetch,
                                 dim_arg_names=[eprftch_u, jprftch_u],
                                 default_tag=None,
                                 within=within
                                 )
        t_unit = lp.tag_inames(t_unit, {eprftch_u: "l.1"})
        t_unit = lp.split_iname(t_unit, jprftch_u, nwork_items_per_e,
                                inner_tag="l.0")
    else:
        t_unit = lp.add_prefetch(t_unit, u,
                                 sweep_inames=[j],
                                 fetch_outer_inames=frozenset([e_inner,
                                                               e_outer]),
                                 temporary_address_space=lp.AddressSpace.PRIVATE,
                                 temporary_name=u_fetch,
                                 default_tag="unr",
                                 within=within
                                 )

    # }}}

    t_unit = lp.add_prefetch(t_unit, J,
                             sweep_inames=[x, r],
                             fetch_outer_inames=frozenset({e_outer, e_inner,
                                                           i_inner_inner}),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name=J_fetch,
                             default_tag="unr",
                             within=within)

    # {{{ tile and prefetch D

    t_unit = lp.split_iname(t_unit, j, math.ceil(Ndof/j_tiles),
                            inner_iname=j_inner, outer_iname=j_tile,
                            inner_tag="unr", outer_tag="unr")
    t_unit = lp.add_prefetch(t_unit, D, [i_inner, r, j_inner],
                             fetch_outer_inames=frozenset([e_outer,
                                                           i_tile,
                                                           j_tile]),
                             dim_arg_names=[rprftch_D, iprftch_D, jprftch_D],
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             temporary_name=D_fetch,
                             default_tag=None)
    t_unit = lp.split_iname(t_unit, iprftch_D, n_e_per_wg, inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, jprftch_D, nwork_items_per_e, inner_tag="l.0")

    # }}}

    # {{{ precompute 'subst'

    t_unit = lp.split_iname(t_unit, i_inner, nwork_items_per_e,
                            inner_iname=i_inner_inner,
                            outer_iname=i_inner_outer,
                            inner_tag="l.0", outer_tag="unr",
                            )

    # FIXME: There are certain cases in which we can avoid this to go to LOCAL.
    t_unit = lp.precompute(t_unit, "subst",
                           sweep_inames=[r, i_inner_outer],
                           precompute_inames=[i_prcmpt_subst,
                                              r_prcmpt_subst],
                           storage_axes=[i, r],
                           precompute_outer_inames=frozenset({e_inner,
                                                              e_outer,
                                                              i_tile,
                                                              i_inner_inner}),
                           default_tag="unr",
                           compute_insn_id=prcmpt_j_redn,
                           temporary_address_space=lp.AddressSpace.PRIVATE)

    # }}}

    # {{{ TODO: remove once github.com/inducer/loopy/issues/666 is resolved.

    t_unit = lp.realize_reduction(t_unit, insn_id_filter=prcmpt_j_redn)
    inames_to_duplicate = (frozenset({i_prcmpt_subst,
                                      r_prcmpt_subst})
                           & t_unit[kernel_name].all_inames())

    acc_name = f"acc_{j_tile}_{j_inner}"
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

    if not prftch_u_to_local:
        # TODO: Yet another headache to ensure that the fetch instruction uses all
        # the hw axes.
        t_unit = lp.add_inames_to_insn(t_unit,
                                       i_inner_inner,
                                       f"writes:{u_fetch}")

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
                   f" '{int(prftch_u_to_local)}',"
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
                   f"    AND prftch_u_to_local = {int(prftch_u_to_local)}"
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
            BooleanParameter("prftch_u_to_local"))
        manipulator.add_parameter(
            IntegerParameter("nwork_items_per_e", 1, 105))
        manipulator.add_parameter(
            IntegerParameter("n_e_per_wg", 2, 32))
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

        if ((math.ceil((Ndof*Ndim)/cfg["i_tiles"]) * math.ceil(Ndof/cfg["j_tiles"]))
                + int(cfg["prftch_u_to_local"]) * Ndof * cfg["n_e_per_wg"]
                + Ndim * Ndof * cfg["n_e_per_wg"])*8e-3 > 47:
            logger.info("Shared memory limit exceeded => ignored configuration.")
            return Result(time=np.inf)

        specialized_transform = partial(transform,
                                        n_e_per_wg=cfg["n_e_per_wg"],
                                        nwork_items_per_e=cfg["nwork_items_per_e"],
                                        i_tiles=cfg["i_tiles"],
                                        j_tiles=cfg["j_tiles"],
                                        prftch_u_to_local=cfg["prftch_u_to_local"],
                                        )

        expr = fnsm.einsum("xre,rij,ej->xei",
                           fnsm.array((Ndim, Ndim, np.inf), "float64"),
                           fnsm.array((Ndim, Ndof, Ndof), "float64"),
                           fnsm.array((np.inf, Ndof), "float64"),
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
        if 0:
            argparser = opentuner.default_argparser()
            TileSizesTuner.main(argparser.parse_args())
        else:
            # Enable for debugging
            expr = fnsm.einsum("xre,rij,ej->xei",
                               fnsm.array((Ndim, Ndim, np.inf), "float64"),
                               fnsm.array((Ndim, Ndof, Ndof), "float64"),
                               fnsm.array((np.inf, Ndof), "float64"),
                               arg_names=["J", "D", "u"])

            specialized_transform = partial(transform,
                                            n_e_per_wg=8,
                                            nwork_items_per_e=8,
                                            i_tiles=6, j_tiles=4,
                                            prftch_u_to_local=False,
                                            )

            print(fnsm.stringify_comparison_vs_roofline(
                expr,
                transform=specialized_transform,
                cl_ctx=cl_ctx,
                long_dim_length=200_000
            ))
