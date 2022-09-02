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
Nface = Ndim + 1
Nfields = 4
Nvoldof = 35
Nfacedof = 15

DB_FILENAME = "wave_facemass_3d_p4.db"
DB_TABLENAME = "NVIDIA_TITAN_V"


@cache
def transform(t_unit, n_e_per_wg, nwork_items_per_e,
              n_stmt_tile,
              n_i_tile, n_j_tile,
              insn_match=None, kernel_name=None):
    import loopy.match as lp_match
    from more_itertools import chunked

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    within = lp_match.parse_match(insn_match)

    ref_einsum = fnsm.fused_einsum("fe,fij,fej->ei",
                                   [(Nface, np.inf),
                                    (Nface, Nvoldof, Nfacedof),
                                    (Nface, np.inf, Nfacedof)],
                                   dtypes="float64",
                                   use_matrix=[
                                       [{"J"}, {"L"}, {f"v{i}"}]
                                       for i in range(Nfields)
                                   ])
    len_stmt_tile = math.ceil(Nfields/n_stmt_tile)
    len_j_tile = math.ceil(Nfacedof/n_j_tile)
    len_i_tile = math.ceil(Nvoldof/n_i_tile)

    # {{{ get corresponding variables in t_unit

    vng = t_unit.default_entrypoint.get_var_name_generator()
    subst_map = fnsm.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match)
    i = subst_map["i"]
    j = subst_map["j"]
    J = subst_map["J"]
    e = subst_map["e"]
    f = subst_map["f"]
    L = subst_map["L"]
    fields = [subst_map[f"v{i}"] for i in range(Nfields)]
    outputs = [subst_map["_fe_out"]] + [subst_map[f"_fe_out_{i}"]
                                        for i in range(Nfields-1)]
    subst_names = {field: vng("subst_hoist") for field in fields}
    e_s = [vng(e) for _ in range(n_stmt_tile)]
    i_s = [vng(i) for _ in range(n_stmt_tile)]
    j_s = [vng(j) for _ in range(n_stmt_tile)]
    f_s = [vng(f) for _ in range(n_stmt_tile)]
    j_tile_names = [vng(f"{j}_tile") for _ in range(n_stmt_tile)]
    j_inner_names = [vng(f"{j}_inner") for _ in range(n_stmt_tile)]
    i_tile_names = [vng(f"{i}_tile") for _ in range(n_stmt_tile)]
    i_inner_names = [vng(f"{i}_inner") for _ in range(n_stmt_tile)]
    i_inner_outer_names = [vng(f"{i}_inner_outer")
                           for _ in range(n_stmt_tile)]
    L_fetch = vng(f"{L}_fetch")
    prefetch_L_insns_ids = [vng(f"prftch_{L}") for _ in range(n_stmt_tile)]
    e_outer_names = [vng(f"{e}_outer") for _ in range(n_stmt_tile)]
    e_inner_names = [vng(f"{e}_inner") for _ in range(n_stmt_tile)]
    i_stmt_to_subst_prcmp_tmp = [vng("prcmpt_stage1")
                                 for _ in range(len_stmt_tile)]
    i_stmt_tile_to_itile_init = [vng(f"{i}_tile_init")
                                 for _ in range(n_stmt_tile)]
    i_stmt_tile_to_i_inner_outer_assign = [vng(f"{i}_inner_outer_assign")
                                           for _ in range(n_stmt_tile)]
    i_stmt_tile_to_itile_assign = [vng(f"{i}_tile_assign")
                                   for _ in range(n_stmt_tile)]
    i_stmt_tile_to_i_inner_outer_init = [vng(f"{i}_inner_outer_init")
                                         for _ in range(n_stmt_tile)]
    i_stmt_tile_to_e_prcmpt_stage1 = [vng(f"{e}_prcmpt_stage1")
                                      for _ in range(n_stmt_tile)]
    i_stmt_tile_to_f_prcmpt_stage1 = [vng(f"{f}_prcmpt_stage1")
                                      for _ in range(n_stmt_tile)]
    i_stmt_tile_to_j_prcmpt_stage1 = [vng(f"{j}_prcmpt_stage1")
                                      for _ in range(n_stmt_tile)]

    # }}}

    # {{{ splitting fields across outer_statement_tiles

    i_stmt_tile_to_fields = list(chunked(fields, len_stmt_tile))
    i_stmt_tile_to_outputs = list(chunked(outputs, len_stmt_tile))

    # }}}

    # {{{ split the kernel into disparate chunks

    for fields_in_tile, new_i, new_f, new_e, new_j in zip(
            i_stmt_tile_to_fields,
            i_s, f_s, e_s, j_s
    ):
        insn_match = lp_match.And(
            (within,
             lp_match.Or(tuple(lp_match.Reads(field_name)
                               for field_name in fields_in_tile))
             )
        )
        t_unit = lp.duplicate_inames(t_unit, (i, f, e, j),
                                     within=insn_match,
                                     new_inames=[new_i, new_f, new_e, new_j])

    # }}}

    for i_stmt_tile, fields_in_tile in enumerate(i_stmt_tile_to_fields):
        new_i = i_s[i_stmt_tile]
        new_j = j_s[i_stmt_tile]
        new_f = f_s[i_stmt_tile]
        new_e = e_s[i_stmt_tile]
        j_tile_name = j_tile_names[i_stmt_tile]
        j_inner_name = j_inner_names[i_stmt_tile]
        i_tile_name = i_tile_names[i_stmt_tile]
        i_inner_name = i_inner_names[i_stmt_tile]
        i_inner_outer_name = i_inner_outer_names[i_stmt_tile]
        f_prftchL, i_prftchL, j_prftchL = (vng(f"{f}prftch{L}"),
                                           vng(f"{i}prftch{L}"),
                                           vng(f"{j}prftch{L}"))

        # There's a problem here. The accumulator names are sort of random
        # here which is obnoxious. We probably need to use some metadata here.

        for field in fields_in_tile:
            subst_name = subst_names[field]
            # FIXME: use precompute inames based on which inner statement tile
            # does the field belong to.

            insn_match = lp_match.And((
                within,
                lp_match.Reads(field)
            ))

            t_unit = fnsm.extract_einsum_terms_as_subst(
                t_unit,
                f"{subst_name}({new_e}, {new_j}, {new_f})",
                f"{J}[{new_f}, {new_e}]*{field}[{new_f}, {new_e}, {new_j}]",
                insn_match=insn_match
            )

        t_unit = lp.split_iname(t_unit, new_j, len_j_tile,
                                outer_iname=j_tile_name,
                                inner_iname=j_inner_name)
        t_unit = lp.split_iname(t_unit, new_i, len_i_tile,
                                outer_iname=i_tile_name,
                                inner_iname=i_inner_name)
        t_unit = lp.split_iname(t_unit, new_e, n_e_per_wg,
                                inner_iname=e_inner_names[i_stmt_tile],
                                outer_iname=e_outer_names[i_stmt_tile],
                                outer_tag="g.0", inner_tag="l.1")
        t_unit = lp.add_prefetch(t_unit, L,
                                 sweep_inames=[j_inner_name, i_inner_name, new_f],
                                 fetch_outer_inames=frozenset(
                                     {j_tile_name,
                                      i_tile_name,
                                      e_outer_names[i_stmt_tile]}),
                                 dim_arg_names=[f_prftchL, i_prftchL, j_prftchL],
                                 temporary_name=L_fetch,
                                 temporary_address_space=lp.AddressSpace.LOCAL,
                                 default_tag=None,
                                 prefetch_insn_id=prefetch_L_insns_ids[i_stmt_tile],
                                 within=lp_match.Iname(i_inner_name),
                                 )
        t_unit = lp.split_iname(t_unit, i_prftchL, n_e_per_wg,
                                inner_tag="l.1")
        t_unit = lp.split_iname(t_unit, j_prftchL, nwork_items_per_e,
                                inner_tag="l.0")

        for istmt, field in enumerate(fields_in_tile):
            subst_name = subst_names[field]
            t_unit = lp.precompute(
                t_unit,
                subst_name,
                sweep_inames=[e_inner_names[i_stmt_tile],
                              j_inner_names[i_stmt_tile],
                              new_f],
                temporary_address_space=lp.AddressSpace.LOCAL,
                temporary_name=i_stmt_to_subst_prcmp_tmp[istmt],
                precompute_outer_inames=frozenset({e_outer_names[i_stmt_tile],
                                                   j_tile_names[i_stmt_tile],
                                                   }),
                precompute_inames=[
                    i_stmt_tile_to_e_prcmpt_stage1[i_stmt_tile],
                    i_stmt_tile_to_j_prcmpt_stage1[i_stmt_tile],
                    i_stmt_tile_to_f_prcmpt_stage1[i_stmt_tile]],
                default_tag=None)
            t_unit = lp.tag_inames(
                t_unit,
                {i_stmt_tile_to_e_prcmpt_stage1[i_stmt_tile]: "l.1"})
            t_unit = lp.split_iname(
                t_unit, i_stmt_tile_to_j_prcmpt_stage1[i_stmt_tile],
                nwork_items_per_e, inner_tag="l.0")

        t_unit = lp.split_iname(t_unit, i_inner_name, nwork_items_per_e,
                                outer_iname=i_inner_outer_name,
                                inner_tag="l.0",
                                # outer_tag="unr"
                                )

        outputs_insn_match = lp_match.And(
            (within,
             lp_match.Or(tuple(lp_match.Writes(output)
                               for output in i_stmt_tile_to_outputs[i_stmt_tile])))
        )
        insn_ids = [insn.id
                    for insn in t_unit[kernel_name].instructions
                    if outputs_insn_match(t_unit[kernel_name], insn)]

        t_unit = lp.realize_reduction(t_unit, insn_id_filter=insn_ids)
        inames_to_duplicate = sorted(frozenset({i_tile_name, i_inner_outer_name})
                                     & t_unit[kernel_name].all_inames())
        acc_names = {vng(f"acc_{new_f}_{j_tile_name}_{j_inner_name}")
                     for _ in fields_in_tile}
        t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                      set(inames_to_duplicate),
                                                      only_var_names=acc_names)

        new_iname_names_map = {
            i_tile_name: i_stmt_tile_to_itile_init[i_stmt_tile],
            i_inner_outer_name: i_stmt_tile_to_i_inner_outer_init[i_stmt_tile],
        }
        t_unit = lp.duplicate_inames(
            t_unit,
            inames_to_duplicate,
            within=lp_match.Or(
                tuple(lp_match.And((lp_match.Writes(acc_name),
                                    lp_match.Not(lp_match.Reads(acc_name))))
                      for acc_name in acc_names)),
            new_inames=[new_iname_names_map[iname] for iname in inames_to_duplicate]
        )

        new_iname_names_map = {
            i_tile_name: i_stmt_tile_to_itile_assign[i_stmt_tile],
            i_inner_outer_name: i_stmt_tile_to_i_inner_outer_assign[i_stmt_tile],
        }
        t_unit = lp.duplicate_inames(
            t_unit,
            inames_to_duplicate,
            within=lp_match.Or(
                tuple(lp_match.And((lp_match.Reads(acc_name),
                                    lp_match.Not(lp_match.Writes(acc_name))))
                      for acc_name in acc_names)),
            new_inames=[new_iname_names_map[iname] for iname in inames_to_duplicate]
        )

    for i_stmt_tile in range(1, n_stmt_tile):
        predecessor = lp_match.And((within,
                                    lp_match.Iname(j_tile_names[i_stmt_tile-1])))
        successor = lp_match.And((within,
                                  lp_match.Iname(j_tile_names[i_stmt_tile])))
        t_unit = lp.add_dependency(t_unit, successor, predecessor)

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
            IntegerParameter("nwork_items_per_e", 1, Nvoldof))
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

        # nkbs_in_local_mem = (
        #     (Nface
        #      * math.ceil(Nvoldof/cfg["i_tiles"])
        #      * math.ceil(Nfacedof/cfg["j_tiles"]))
        #     + Ndim * cfg["n_e_per_wg"] * math.ceil(Ndof/cfg["j_tiles"]))*8e-3

        # if nkbs_in_local_mem > 47:
        #     logger.info("Shared memory limit exceeded => ignored configuration.")
        #     return Result(time=np.inf)

        specialized_transform = partial(transform,
                                        n_e_per_wg=cfg["n_e_per_wg"],
                                        nwork_items_per_e=cfg["nwork_items_per_e"],
                                        i_tiles=cfg["i_tiles"],
                                        j_tiles=cfg["j_tiles"],
                                        )
        expr = fnsm.fused_einsum("ef,fij,fej->ei",
                                 [(np.inf, Nface),
                                  (Nface, Nvoldof, Nfacedof),
                                  (Nface, np.inf, Nfacedof)],
                                 dtypes="float64",
                                 use_matrix=[
                                     [{"J"}, {"L"}, {f"v{i}"}]
                                     for i in range(Nfields)
                                 ])

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
        if 0:
            argparser = opentuner.default_argparser()
            TileSizesTuner.main(argparser.parse_args())
        else:
            # Enable for debugging
            expr = fnsm.fused_einsum("fe,fij,fej->ei",
                                     [(Nface, np.inf),
                                      (Nface, Nvoldof, Nfacedof),
                                      (Nface, np.inf, Nfacedof)],
                                     dtypes="float64",
                                     use_matrix=[
                                         [{"J"}, {"L"}, {f"v{i}"}]
                                         for i in range(Nfields)
                                     ])
            # print(fnsm.generate_loopy_with_opt_einsum_schedule(expr))
            # 1/0

            specialized_transform = partial(transform,
                                            n_e_per_wg=32,
                                            n_stmt_tile=2,
                                            nwork_items_per_e=7,
                                            n_i_tile=1, n_j_tile=5,
                                            )

            print(fnsm.stringify_comparison_vs_roofline(
                expr,
                transform=specialized_transform,
                cl_ctx=cl_ctx,
            ))
