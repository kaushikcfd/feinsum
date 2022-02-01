import feinsum as f
import numpy as np
import pyopencl as cl
import loopy as lp
import logging
logger = logging.getLogger(__name__)


def get_div_einsum(ndofs, ndim):
    return f.fused_einsum("es, sij, ej -> ei",
                          [(np.inf, ndim),
                           (ndim, ndofs, ndofs),
                           (np.inf, ndofs)],
                          dtypes="float64",
                          use_matrix=[
                              [{"Jx"}, {"R"}, {"ux"}],
                              [{"Jy"}, {"R"}, {"uy"}],
                              [{"Jz"}, {"R"}, {"uz"}],
                          ])


def variant_0(t_unit):
    """
    Un-fuse all the loops.
    """
    t_unit = lp.duplicate_inames(t_unit, ["e", "i", "j", "s"],
                                 within="id:insn",
                                 new_inames=["e_0", "i_0", "j_0", "s_0"])
    t_unit = lp.duplicate_inames(t_unit, ["e", "i", "j", "s"],
                                 within="id:insn_0",
                                 new_inames=["e_1", "i_1", "j_1", "s_1"])
    t_unit = lp.duplicate_inames(t_unit, ["e", "i", "j", "s"],
                                 within="id:insn_1",
                                 new_inames=["e_2", "i_2", "j_2", "s_2"])

    for iel, idof in [("e_0", "i_0"), ("e_1", "i_1"), ("e_2", "i_2")]:
        t_unit = lp.split_iname(t_unit, iel, 8,
                                outer_tag="g.0", inner_tag="l.1")
        t_unit = lp.split_iname(t_unit,
                                idof, 4, inner_tag="l.0")

    return t_unit


def variant_1(t_unit):
    """
    Simple work division strategy.
    """
    t_unit = lp.split_iname(t_unit, "e", 8,
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, "i", 4,
                            inner_tag="l.0")

    return t_unit


def main():
    from feinsum.data.device_info import DEV_TO_PEAK_F64_GFLOPS
    cl_ctx = cl.create_some_context()

    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
        return
    if cl_ctx.devices[0].name not in DEV_TO_PEAK_F64_GFLOPS:
        logger.info("Device not known.")
        return

    expr = get_div_einsum(ndofs=35, ndim=3)
    f.pprint_comparison_vs_roofline(expr,
                                    cl_ctx=cl_ctx,
                                    transform=variant_1,
                                    long_dim_length=100_000,
                                    )


if __name__ == "__main__":
    main()
