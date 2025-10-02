import logging

import loopy as lp
import numpy as np
import pyopencl as cl

import feinsum as f

logger = logging.getLogger(__name__)


def get_div_einsum(ndofs, ndim):
    return f.batched_einsum(
        "es, sij, ej -> ei",
        [(np.inf, ndim), (ndim, ndofs, ndofs), (np.inf, ndofs)],
        dtypes="float64",
        use_matrix=[
            [{"Jx"}, {"R"}, {"ux"}],
            [{"Jy"}, {"R"}, {"uy"}],
            [{"Jz"}, {"R"}, {"uz"}],
        ],
    )


def variant_0(t_unit):
    """
    Un-fuse all the loops.
    """
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "s"],
        within="id:insn",
        new_inames=["e_0", "i_0", "j_0", "s_0"],
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "s"],
        within="id:insn_0",
        new_inames=["e_1", "i_1", "j_1", "s_1"],
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "s"],
        within="id:insn_1",
        new_inames=["e_2", "i_2", "j_2", "s_2"],
    )

    for iel, _ in [("e_0", "i_0"), ("e_1", "i_1"), ("e_2", "i_2")]:
        t_unit = lp.split_iname(t_unit, iel, 32, outer_tag="g.0", inner_tag="l.0")

    return t_unit


def variant_1(t_unit, insn_match=None, kernel_name=None):
    """
    Simple work division strategy.
    """
    t_unit = lp.tag_inames(t_unit, {"s": "unr"})
    t_unit = lp.split_iname(t_unit, "e", 8, outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, "i", 4, inner_tag="l.0", outer_tag="ilp")

    return t_unit


def variant_3(t_unit):
    # Based on https://github.com/nchristensen/grudge/'s transformation
    # algorithm
    assert t_unit.default_entrypoint.arg_dict["R"].shape == (3, 35, 35)
    imatrix = "s"
    iel = "e"
    idof = "i"

    dof_vecs = [
        arg.name
        for arg in t_unit.default_entrypoint.args
        if arg.name in ["ux", "uy", "uz"]
    ]
    t_unit = lp.tag_inames(t_unit, {imatrix: "unr"})
    t_unit = lp.split_iname(t_unit, iel, 16, outer_tag="g.0")
    t_unit = lp.split_iname(
        t_unit, f"{iel}_inner", 8, outer_tag="ilp", inner_tag="l.0"
    )
    t_unit = lp.tag_inames(t_unit, {idof: "l.1"})
    for vec in dof_vecs:
        t_unit = lp.add_prefetch(
            t_unit,
            vec,
            f"j,{iel}_inner_outer,{iel}_inner_inner",
            temporary_name=f"{vec}_prftch",
            default_tag="l.auto",
        )
    t_unit = lp.add_inames_for_unused_hw_axes(t_unit)
    return t_unit


def main():
    cl_ctx = cl.create_some_context()

    expr = get_div_einsum(ndofs=35, ndim=3)
    print(
        f.stringify_comparison_vs_roofline(
            expr,
            cl_ctx=cl_ctx,
            transform=variant_1,
            long_dim_length=1000,
            ignore_unknown_device=True,  # For CI
        )
    )


if __name__ == "__main__":
    main()
