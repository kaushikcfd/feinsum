import feinsum as f
import numpy as np
import pyopencl as cl
import loopy as lp
import logging
logger = logging.getLogger(__name__)


def get_grad_einsum(ndofs, ndim):
    return f.einsum("xer,rij,ej->xei",
                    f.array((ndim, np.inf, ndim,),
                            "float64"),
                    f.array((ndim, ndofs, ndofs),
                            "float64"),
                    f.array((np.inf, ndofs),
                            "float64"),
                    arg_names=["J", "R", "u"])


def variant_0(t_unit):
    # The one in `main` right now
    return lp.split_iname(lp.split_iname(t_unit, "e", 8,
                                         outer_tag="g.0", inner_tag="l.1"),
                          "i", 4, inner_tag="l.0")


def variant_1(t_unit, nwork_items_per_cell=4, ncells_per_workgroup=8):
    t_unit = lp.split_reduction_outward(t_unit, "r")
    t_unit = lp.realize_reduction(t_unit)

    t_unit = lp.split_iname(t_unit, "i",
                            nwork_items_per_cell)
    t_unit = lp.split_iname(t_unit, "e",
                            ncells_per_workgroup)

    t_unit = lp.privatize_temporaries_with_inames(
        t_unit,
        "r",
        only_var_names={"acc_j"})
    t_unit = lp.duplicate_inames(t_unit,
                                 ("r",),
                                 within="id:insn_r_update_*",
                                 new_inames=["r_evaluate_at_point"])

    t_unit = lp.privatize_temporaries_with_inames(
        t_unit,
        "i_outer",
        only_var_names={"acc_r"})
    t_unit = lp.duplicate_inames(t_unit,
                                 inames=("i_outer",),
                                 within="id:insn_r_init",
                                 new_inames=["i_outer_init"])
    t_unit = lp.duplicate_inames(t_unit,
                                 inames=("i_outer",),
                                 within="id:insn",
                                 new_inames=["i_outer_store"])

    t_unit = lp.add_prefetch(t_unit,
                             "J",
                             sweep_inames={"r_evaluate_at_point"},
                             fetch_outer_inames=frozenset({"x", "i_inner", "e_outer",
                                                           "e_inner"}),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name="J_prftch")

    t_unit = lp.tag_inames(t_unit,
                           "e_outer:g.0,e_inner:l.1,i_inner:l.0")
    return t_unit


def variant_2(t_unit, nwork_items_per_cell=4, ncells_per_workgroup=8):
    t_unit = variant_1(t_unit,
                       nwork_items_per_cell=nwork_items_per_cell,
                       ncells_per_workgroup=ncells_per_workgroup)
    t_unit = lp.add_prefetch(t_unit,
                             "R",
                             sweep_inames={"r_evaluate_at_point", "i_inner",
                                           "i_outer", "j"},
                             fetch_outer_inames=frozenset({"e_outer"}),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             default_tag=None,
                             temporary_name="R_prftch")

    t_unit = lp.join_inames(t_unit, ["R_dim_0", "R_dim_1", "R_dim_2"], "iRprftch")
    t_unit = lp.split_iname(t_unit, "iRprftch",
                            nwork_items_per_cell * ncells_per_workgroup)
    t_unit = lp.split_iname(t_unit, "iRprftch_inner",
                            nwork_items_per_cell,
                            inner_tag="l.0", outer_tag="l.1")
    return t_unit


def main():
    from feinsum.data.device_info import DEV_TO_PEAK_F64_GFLOPS
    cl_ctx = cl.create_some_context()
    if len(cl_ctx.devices) != 1:
        logger.info("Multiple devices in the context")
    if cl_ctx.devices[0].name not in DEV_TO_PEAK_F64_GFLOPS:
        logger.info("Device not known.")

    expr = get_grad_einsum(ndofs=35, ndim=3)
    f.pprint_comparison_vs_roofline(expr,
                                    cl_ctx=cl_ctx,
                                    transform=variant_1,
                                    long_dim_length=50_000,
                                    )


if __name__ == "__main__":
    main()
