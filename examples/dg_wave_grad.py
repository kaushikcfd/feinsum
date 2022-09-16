import feinsum as f
import numpy as np
import pyopencl as cl
import loopy as lp
import logging
logging.basicConfig(level="INFO")
logger = logging.getLogger(__name__)


def get_grad_einsum(ndofs, ndim):
    return f.einsum("xre,rij,ej->xei",
                    f.array((ndim, ndim, np.inf),
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


def variant_3(t_unit):
    ncells_per_workgroup = 9
    nworkitems_per_cell = 7
    j_tile_len = 9
    i_tile_len = 35

    # {{{ term hoisting to match the flop count of opt_einsum

    t_unit = lp.split_reduction_inward(t_unit, "j")
    t_unit = f.hoist_reduction_invariant_terms(t_unit, "j")
    t_unit = f.extract_einsum_terms_as_subst(t_unit,
                                             "subst(r, e, i)",
                                             "sum(j, R[r, i, j]*u[e, j])")

    # }}}

    t_unit = lp.split_iname(t_unit, "i", i_tile_len, outer_iname="i_tile")
    t_unit = lp.split_iname(t_unit, "j", j_tile_len, outer_iname="j_tile")

    t_unit = lp.rename_iname(t_unit, "i_inner", "i")
    t_unit = lp.rename_iname(t_unit, "j_inner", "j")
    t_unit = lp.split_iname(t_unit, "e", ncells_per_workgroup,
                            inner_iname="e_inner", outer_iname="e_outer",
                            outer_tag="g.0", inner_tag="l.1")
    t_unit = lp.split_iname(t_unit, "i", nworkitems_per_cell,
                            inner_iname="i_inner", outer_iname="i_outer",
                            inner_tag="l.0")

    t_unit = lp.add_prefetch(t_unit,
                             "J",
                             sweep_inames=["r", "x"],
                             fetch_outer_inames=frozenset(["e_inner",
                                                           "e_outer",
                                                           "i_inner"]),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             temporary_name="J_prftch",
                             )

    # {{{ TODO: Make precompute smarter (should be a single precompute call)

    t_unit = lp.precompute(t_unit, "subst",
                           sweep_inames=["r"],
                           precompute_outer_inames=frozenset({"e_inner",
                                                              "e_outer",
                                                              "i_inner",
                                                              "i_outer",
                                                              "i_tile",
                                                              }),
                           temporary_name="tmp_hoist",
                           temporary_address_space=lp.AddressSpace.PRIVATE,
                           compute_insn_id="insn_hoist",
                           )
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  "i_outer",
                                                  only_var_names=["tmp_hoist"])

    t_unit = lp.duplicate_inames(t_unit, "i_outer", "id:insn_hoist",
                                 "i_outer_hoist")

    # }}}

    # {{{ Move 'u ' to shared.

    # Prefetch 'u' within the tile
    t_unit = lp.add_prefetch(t_unit, "u",
                             sweep_inames=["e_inner", "j"],
                             fetch_outer_inames=frozenset(["e_outer",
                                                           "i_tile",
                                                           "j_tile"]),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             dim_arg_names=["e_prftch", "j_prftch"],
                             default_tag=None,
                             )

    t_unit = lp.split_iname(t_unit, "e_prftch", ncells_per_workgroup,
                            inner_tag="l.1", outer_tag="unr")
    t_unit = lp.split_iname(t_unit, "j_prftch", nworkitems_per_cell,
                            inner_tag="l.0", outer_tag="unr")

    # }}}

    # {{{ Move 'R' to shared.

    t_unit = lp.add_prefetch(t_unit, "R",
                             sweep_inames=["r_0", "i_inner", "i_outer_hoist", "j"],
                             fetch_outer_inames=frozenset(["e_outer",
                                                           "i_tile",
                                                           "j_tile"]),
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             dim_arg_names=["r_prftch", "i_prftch", "j_prftch"],
                             default_tag=None,
                             )
    if 0:
        # This branch improves perf. by 20% but, something in loopy
        # non-deterministically leads to very ugly domains.
        t_unit = lp.join_inames(t_unit, ["r_prftch", "i_prftch", "j_prftch"],
                                "i_Rprftch")

        t_unit = lp.split_iname(t_unit, "i_Rprftch",
                                ncells_per_workgroup * nworkitems_per_cell,
                                outer_tag="unr")

        t_unit = lp.split_iname(t_unit, "i_Rprftch_inner",
                                nworkitems_per_cell,
                                inner_tag="l.0", outer_tag="l.1")
    else:
        t_unit = lp.tag_inames(t_unit, "r_prftch:unr")
        t_unit = lp.split_iname(t_unit, "i_prftch", ncells_per_workgroup,
                                inner_tag="l.1", outer_tag="unr")
        t_unit = lp.split_iname(t_unit, "j_prftch", nworkitems_per_cell,
                                inner_tag="l.0", outer_tag="unr")

    # }}}

    logger.info("Done with prefetching 'R'.")

    # {{{ make buffer array smarter (should be a single call to buffer_array)

    t_unit = lp.buffer_array(t_unit, "_fe_out", buffer_inames=["x"],
                             init_expression="0",
                             default_tag=None,
                             temporary_scope=lp.AddressSpace.PRIVATE)
    t_unit = lp.privatize_temporaries_with_inames(t_unit, "i_outer",
                                                  only_var_names={"_fe_out_buf"})

    t_unit = lp.duplicate_inames(t_unit,
                                 inames=["i_outer"],
                                 within="id:init__fe_out",
                                 new_inames=["_fe_out_init_1"])

    t_unit = lp.duplicate_inames(t_unit,
                                 inames=["i_outer"],
                                 within="id:store__fe_out",
                                 new_inames=["_fe_out_store_1"])

    # }}}

    # {{{ must be smarter way of doing this in loopy

    t_unit = lp.realize_reduction(t_unit, insn_id_filter="insn_hoist")
    t_unit = lp.privatize_temporaries_with_inames(t_unit,
                                                  frozenset(["r_0",
                                                             "i_outer_hoist"]),
                                                  only_var_names={"acc_j_tile_j"})
    t_unit = lp.duplicate_inames(t_unit,
                                 ["i_outer_hoist", "r_0"],
                                 within="id:insn_hoist_j_tile_j_init",
                                 new_inames=["i_outer_hoist_init", "r_0_init"],
                                 )

    t_unit = lp.duplicate_inames(t_unit,
                                 ["i_outer_hoist", "r_0"],
                                 within="id:insn_hoist",
                                 new_inames=["i_outer_hoist_store", "r_0_store"],
                                 )

    # }}}

    logger.info("Done with transformations.")

    return t_unit


def paranumal_transform(t_unit):

    t_unit = lp.split_reduction_inward(t_unit, "j")
    t_unit = f.hoist_reduction_invariant_terms(t_unit, "j")
    t_unit = f.extract_einsum_terms_as_subst(t_unit,
                                             "subst(r, e, i)",
                                             "sum(j, R[r, i, j]*u[e, j])")

    NblockV = 4

    t_unit = lp.split_iname(t_unit, "e", NblockV,
                            inner_tag="l.1", outer_tag="g.0")
    t_unit = lp.tag_inames(t_unit, {"i": "l.0"})
    t_unit = lp.add_prefetch(t_unit, "u", sweep_inames=["e_inner", "j"],
                             dim_arg_names=["eprftch_u", "jprftch_u"],
                             temporary_address_space=lp.AddressSpace.LOCAL,
                             default_tag=None)
    t_unit = lp.tag_inames(t_unit, {"eprftch_u": "l.1", "jprftch_u": "l.0"})
    t_unit = lp.precompute(t_unit, "subst", sweep_inames=["r"],
                           precompute_outer_inames=frozenset({"e_outer",
                                                              "e_inner",
                                                              "i"}),
                           temporary_address_space=lp.AddressSpace.PRIVATE,
                           default_tag="unr")

    t_unit = lp.add_prefetch(t_unit, "J", sweep_inames=["x", "r"],
                             fetch_outer_inames=frozenset({"e_outer",
                                                           "e_inner",
                                                           "i",
                                                           }),
                             temporary_address_space=lp.AddressSpace.PRIVATE,
                             default_tag="unr")
    t_unit = lp.tag_inames(t_unit, {"r": "unr", "j": "unr", "x": "unr"})

    return t_unit


def main():
    cl_ctx = cl.create_some_context()

    expr = get_grad_einsum(ndofs=35, ndim=3)
    print(f.stringify_comparison_vs_roofline(expr,
                                             cl_ctx=cl_ctx,
                                             transform=paranumal_transform,
                                             ignore_unknown_device=True,  # For CI
                                             ))


if __name__ == "__main__":
    main()
