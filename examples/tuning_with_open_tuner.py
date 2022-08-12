import feinsum as fnsm
import pyopencl as cl
import numpy as np
import loopy as lp
import math
import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import MeasurementInterface
from opentuner import Result
from functools import partial
import logging
logger = logging.getLogger(__name__)

cl_ctx = cl.create_some_context()


def transform(t_unit, n_e_per_wg, nwork_items_per_e, i_tilelen, j_tilelen,
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
    i_inner_outer = vng(i_inner+"_outer")

    t_unit = lp.add_prefetch(t_unit, u,
                             sweep_inames=[j],
                             fetch_outer_inames=frozenset([e]),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name=u_fetch,
                             default_tag=None,
                             )

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg,
                            inner_iname=e_inner, outer_iname=e_outer,
                            outer_tag="g.0", inner_tag="l.1")

    # {{{ tile to lower cache reuse distance of D/u

    t_unit = lp.split_iname(t_unit, i, i_tilelen,
                            inner_iname=i_inner, outer_iname=i_tile,
                            )
    t_unit = lp.split_iname(t_unit, j, j_tilelen,
                            inner_iname=j_inner, outer_iname=j_tile,
                            outer_tag="unr"
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
                            outer_iname=i_inner_outer)

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
    t_unit = lp.add_inames_to_insn(t_unit,
                                   i_inner + "_inner",
                                   f"writes:{u_fetch}")

    # }}}

    # t_unit = lp.set_options(t_unit, "write_code")

    return t_unit


class TileSizesTuner(MeasurementInterface):

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
            IntegerParameter("nwork_items_per_e", 1, 105))
        manipulator.add_parameter(
            IntegerParameter("n_e_per_wg", 1, 32))
        return manipulator

    def run(self, desired_result, input, limit):

        cfg = desired_result.configuration.data
        if cfg["n_e_per_wg"] * cfg["nwork_items_per_e"] > 500:
            return Result(time=np.inf)

        print(cfg)

        specialized_transform = partial(transform,
                                        n_e_per_wg=cfg["n_e_per_wg"],
                                        nwork_items_per_e=cfg["nwork_items_per_e"],
                                        i_tilelen=math.ceil(105/cfg["i_tiles"]),
                                        j_tilelen=math.ceil(35/cfg["j_tiles"]))

        expr = fnsm.einsum("ij,ej->ei",
                           fnsm.array((105, 35), "float64"),
                           fnsm.array((np.inf, 35), "float64"),
                           arg_names=["D", "u"])
        print(fnsm.stringify_comparison_vs_roofline(
            expr,
            transform=specialized_transform,
            cl_ctx=cl_ctx,
            long_dim_length=200_000
        ))

        return Result(time=fnsm.timeit(expr,
                                       cl_ctx=cl_ctx,
                                       transform=specialized_transform,
                                       ))

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        self.manipulator().save_to_file(configuration.data,
                                        "mmm_final_config.json")


if __name__ == "__main__":
    from feinsum.data.device_info import DEV_TO_PEAK_GFLOPS

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
    elif cl_ctx.devices[0].name not in DEV_TO_PEAK_GFLOPS:
        logger.info(f"Device {cl_ctx.devices[0]} not known to database.")
    else:
        argparser = opentuner.default_argparser()
        TileSizesTuner.main(argparser.parse_args())
