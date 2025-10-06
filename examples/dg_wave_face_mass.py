import logging

import loopy as lp
import numpy as np
import pyopencl as cl

import feinsum as f

logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def get_face_mass_einsum(nface, nfacedofs, nvoldofs):
    return f.batched_einsum(
        "ef, fij, fej -> ei",
        [(np.inf, nface), (nface, nvoldofs, nfacedofs), (nface, np.inf, nfacedofs)],
        dtypes="float64",
        use_matrix=[
            [{"J"}, {"R"}, {"v0"}],
            [{"J"}, {"R"}, {"v1"}],
            [{"J"}, {"R"}, {"v2"}],
            [{"J"}, {"R"}, {"v3"}],
        ],
    )


def variant_0(t_unit):
    """
    Un-fuse all the loops.
    """
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "f"],
        within="id:insn",
        new_inames=["e_0", "i_0", "j_0", "f_0"],
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "f"],
        within="id:insn_0",
        new_inames=["e_1", "i_1", "j_1", "f_1"],
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "f"],
        within="id:insn_1",
        new_inames=["e_2", "i_2", "j_2", "f_2"],
    )
    t_unit = lp.duplicate_inames(
        t_unit,
        ["e", "i", "j", "f"],
        within="id:insn_2",
        new_inames=["e_3", "i_3", "j_3", "f_3"],
    )

    for iel, idof in [
        ("e_0", "i_0"),
        ("e_1", "i_1"),
        ("e_2", "i_2"),
        ("e_3", "i_3"),
    ]:
        t_unit = lp.split_iname(t_unit, iel, 4, outer_tag="g.0", inner_tag="l.1")
        t_unit = lp.split_iname(t_unit, idof, 8, outer_tag="unr", inner_tag="l.0")

    return t_unit


def variant_1(t_unit):
    """
    Simple work division strategy.
    """
    ref_einsum = get_face_mass_einsum(nface=4, nvoldofs=35, nfacedofs=15)

    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum)
    iface = subst_map["f"]
    e = subst_map["e"]
    idof = subst_map["i"]

    t_unit = lp.tag_inames(t_unit, {iface: "unr"})
    t_unit = lp.split_iname(t_unit, e, 8, outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, idof, 4, inner_tag="l.0", outer_tag="ilp")

    return t_unit


def variant_2(t_unit, insn_match=None, kernel_name=None):
    from pymbolic import variables

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    ncells_per_group = 16
    nworkitems_per_cell = 12

    ref_einsum = get_face_mass_einsum(nface=4, nvoldofs=35, nfacedofs=15)

    subst_map = f.match_t_unit_to_einsum(
        t_unit, ref_einsum, insn_match=insn_match, kernel_name=kernel_name
    )
    e = subst_map["e"]
    iface = subst_map["f"]
    vs = [subst_map[f"v{i}"] for i in range(4)]
    idof = subst_map["i"]
    j = subst_map["j"]
    J = subst_map["J"]
    R = subst_map["R"]

    knl = t_unit[kernel_name]
    output_names = ["_fe_out"] + [f"_fe_out_{i}" for i in range(4)]

    for i in range(4):
        knl = lp.extract_multiplicative_terms_in_sum_reduction_as_subst(
            knl,
            within=f"writes:{output_names[i]}",
            subst_name=f"subst{i}",
            arguments=variables(f"{iface} {e} {j}"),
            terms_filter=lambda x: {J, vs[i]} & f.get_call_ids(x),
        )

    t_unit = t_unit.with_kernel(knl)

    t_unit = lp.split_iname(
        t_unit,
        e,
        ncells_per_group,
        inner_iname="e_inner",
        outer_iname="e_outer",
        outer_tag="g.0",
        inner_tag="l.1",
    )

    # {{{ fetch 'R'

    t_unit = lp.precompute(
        t_unit,
        R,
        [iface, idof, j],
        precompute_outer_inames=frozenset(["e_outer"]),
        temporary_address_space=lp.AddressSpace.LOCAL,
        default_tag=None,
        precompute_inames=["f_Rprftch", "i_Rprftch", "j_Rprftch"],
    )

    t_unit = lp.split_iname(
        t_unit, "i_Rprftch", ncells_per_group, inner_tag="l.1", outer_tag="unr"
    )
    t_unit = lp.split_iname(
        t_unit, "j_Rprftch", nworkitems_per_cell, inner_tag="l.0", outer_tag="unr"
    )

    logger.info("Prefetched 'R'")

    # }}}

    t_unit = lp.rename_iname(
        t_unit, idof, "i_0", within="writes:_fe_out or writes:_fe_out_0"
    )
    t_unit = lp.rename_iname(
        t_unit, iface, "f_0", within="writes:_fe_out or writes:_fe_out_0"
    )
    t_unit = lp.rename_iname(
        t_unit, j, "j_0", within="writes:_fe_out or writes:_fe_out_0"
    )

    t_unit = lp.rename_iname(
        t_unit, idof, "i_1", within="writes:_fe_out_1 or writes:_fe_out_2"
    )
    t_unit = lp.rename_iname(
        t_unit, iface, "f_1", within="writes:_fe_out_1 or writes:_fe_out_2"
    )
    t_unit = lp.rename_iname(
        t_unit, j, "j_1", within="writes:_fe_out_1 or writes:_fe_out_2"
    )

    t_unit = lp.split_iname(
        t_unit, "i_0", nworkitems_per_cell, inner_tag="l.0", outer_tag="unr"
    )
    t_unit = lp.split_iname(
        t_unit, "i_1", nworkitems_per_cell, inner_tag="l.0", outer_tag="unr"
    )

    for igrp, prcmpt_tmp, subst in [
        (0, "prcmpt_tmp_0", "subst0"),
        (0, "prcmpt_tmp_1", "subst1"),
        (1, "prcmpt_tmp_0", "subst2"),
        (1, "prcmpt_tmp_1", "subst3"),
    ]:
        logger.info(f"Precomputing {subst}")
        t_unit = lp.precompute(
            t_unit,
            subst,
            sweep_inames=[f"f_{igrp}", "e_inner", f"j_{igrp}"],
            precompute_outer_inames=frozenset(["e_outer"]),
            precompute_inames=[
                f"f_prftch_{igrp}",
                f"e_prftch_{igrp}",
                f"j_prftch_{igrp}",
            ],
            default_tag=None,
            temporary_name=prcmpt_tmp,
            temporary_address_space=lp.AddressSpace.LOCAL,
        )

    for igrp in [0, 1]:
        t_unit = lp.split_iname(
            t_unit,
            f"e_prftch_{igrp}",
            ncells_per_group,
            inner_tag="l.1",
            outer_tag="unr",
        )
        t_unit = lp.split_iname(
            t_unit,
            f"j_prftch_{igrp}",
            nworkitems_per_cell,
            inner_tag="l.0",
            outer_tag="unr",
        )

    t_unit = lp.add_dependency(
        t_unit, "id:subst2 or id:subst3", "writes:_fe_out or writes:_fe_out_0"
    )

    logger.info("Done with transformations.")
    return t_unit


def main():
    cl_ctx = cl.create_some_context()

    expr = get_face_mass_einsum(nface=4, nvoldofs=35, nfacedofs=15)
    print(
        f.stringify_comparison_vs_roofline(
            expr,
            cl_ctx=cl_ctx,
            transform=variant_2,
            long_dim_length=1000,
            ignore_unknown_device=True,  # For CI
        )
    )


if __name__ == "__main__":
    main()
