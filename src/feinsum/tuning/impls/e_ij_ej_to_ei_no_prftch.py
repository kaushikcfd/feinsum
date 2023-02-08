import feinsum as fnsm
import loopy as lp
import numpy as np

from feinsum.tuning import IntParameter
from typing import Optional, Any


@fnsm.tuning.einsum_arg("ndofs", lambda e: e.shape[1])
@fnsm.tuning.transform_param(
    "nworkitems_per_e", lambda e: IntParameter(1, e.shape[1]))
@fnsm.tuning.transform_param(
    "n_e_per_wg", lambda e: IntParameter(1, 32))
def transform(t_unit: lp.TranslationUnit,
              ndofs: int,
              n_e_per_wg: int, nworkitems_per_e: int,
              insn_match: Optional[Any] = None,
              kernel_name: Optional[str] = None) -> lp.TranslationUnit:

    kernel_name = kernel_name or t_unit.default_entrypoint.name

    ref_einsum = fnsm.fused_einsum("e,ij,ej->ei",
                                   [(np.inf,),
                                    (ndofs, ndofs),
                                    (np.inf, ndofs)],
                                   dtypes=np.float64,
                                   use_matrix=[
                                       [{"J"}, {"D"}, {"u1"}],
                                       [{"J"}, {"D"}, {"u2"}],
                                       [{"J"}, {"D"}, {"u3"}],
                                       [{"J"}, {"D"}, {"u4"}],
                                   ])
    subst_map = fnsm.match_t_unit_to_einsum(t_unit,
                                            ref_einsum,
                                            kernel_name=kernel_name,
                                            insn_match=insn_match)
    vng = t_unit[kernel_name].get_var_name_generator()

    e = subst_map["e"]
    i = subst_map["i"]
    j = subst_map["j"]
    e_inner, e_outer = vng(f"{e}_inner"), vng(f"{e}_outer")
    i_inner, i_outer = vng(f"{i}_inner"), vng(f"{i}_outer")
    J = subst_map["J"]
    # out1 = subst_map["_fe_out"]
    out2 = subst_map["_fe_out_0"]
    out3 = subst_map["_fe_out_1"]
    out4 = subst_map["_fe_out_2"]

    t_unit = t_unit.with_kernel(
        lp.hoist_invariant_multiplicative_terms_in_sum_reduction(t_unit[kernel_name],
                                                                 j)
    )

    t_unit = lp.split_iname(t_unit, e, n_e_per_wg,
                            inner_tag="l.1", outer_tag="g.0")
    t_unit = lp.split_iname(t_unit, i, nworkitems_per_e,
                            inner_tag="l.0", outer_tag="unr")

    t_unit = lp.precompute(t_unit,
                           J,
                           sweep_inames=[],
                           precompute_outer_inames=frozenset([e_outer,
                                                              e_inner,
                                                              i_inner]),
                           default_tag="unr",
                           temporary_address_space=lp.AddressSpace.PRIVATE)

    t_unit = lp.duplicate_inames(t_unit, (i_outer, j,),
                                 within=f"writes:{out2}",
                                 tags={i_outer:
                                       t_unit[kernel_name].inames[i_outer].tags,
                                       j: t_unit[kernel_name].inames[j].tags})
    t_unit = lp.duplicate_inames(t_unit, (i_outer, j,),
                                 within=f"writes:{out3}",
                                 tags={i_outer:
                                       t_unit[kernel_name].inames[i_outer].tags,
                                       j: t_unit[kernel_name].inames[j].tags})
    t_unit = lp.duplicate_inames(t_unit, (i_outer, j,),
                                 within=f"writes:{out4}",
                                 tags={i_outer:
                                       t_unit[kernel_name].inames[i_outer].tags,
                                       j: t_unit[kernel_name].inames[j].tags})
    return t_unit


if __name__ == "__main__":
    import pyopencl as cl
    import os

    Ndof = 4

    cl_ctx = cl.create_some_context()

    expr = fnsm.fused_einsum("e,ij,ej->ei",
                             [(np.inf,),
                              (Ndof, Ndof),
                              (np.inf, Ndof)],
                             dtypes=np.float64,
                             use_matrix=[
                                 [{"J"}, {"D"}, {"u1"}],
                                 [{"J"}, {"D"}, {"u2"}],
                                 [{"J"}, {"D"}, {"u3"}],
                                 [{"J"}, {"D"}, {"u4"}],
                             ])

    fnsm. autotune(expr, os.path.abspath(__file__), cl_ctx)

# vim: fdm=marker
