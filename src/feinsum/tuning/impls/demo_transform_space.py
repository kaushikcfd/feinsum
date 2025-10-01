# NOTE: Keep this implementation short as this is a part of docs.
# Type-ignoring because this is part of the demo
# type: ignore
# flake8: noqa

# BEGINEXAMPLE
import feinsum as f
import loopy as lp
import numpy as np

from feinsum.tuning import IntParameter, einsum_arg, transform_param


@einsum_arg("N", lambda ensm: ensm.arg_shapes[0][1])
@transform_param("n_x_per_wg", lambda ensm: IntParameter(2, 32))
@transform_param("n_wi_per_x", lambda ensm: IntParameter(1, ensm.arg_shapes[0][1]))
def transform(t_unit, N, n_x_per_wg, n_wi_per_x, insn_match=None, kernel_name=None):
    ref_einsum = f.einsum(
        "xik,kj->xij",
        f.array((np.inf, N, N), np.float64),
        f.array((N, N), np.float64),
        arg_names=["A", "B"],
    )

    subst_map = f.match_t_unit_to_einsum(t_unit, ref_einsum, insn_match=insn_match)

    vng = t_unit.default_entrypoint.get_var_name_generator()
    x = subst_map["x"]
    i = subst_map["i"]
    j = subst_map["j"]
    k = subst_map["k"]
    B = subst_map["B"]

    Bprftch_k, Bprftch_j = vng("Bprftch_k"), vng("Bprftch_j")
    x_outer = vng(f"{x}_outer")

    # Notation:
    #    - The einsum computes 'out[x, i, j]'.
    #
    # Transform space:
    #   - Each work-group computes the slice:
    #     'out[group_id(0)*n_x_per_wg:(group_id(0)+1)*n_x_per_wg,:,:]'
    #   - Each work_item computes the slice:
    #     'out[group_id(0)+n_x_per_wg*local_id(1),local_id(0)::nwi_per_x,:]'

    # Loop Transformations: Work Division
    # -----------------------------------
    t_unit = lp.split_iname(
        t_unit, x, n_x_per_wg, outer_iname=x_outer, inner_tag="l.1", outer_tag="g.0"
    )

    t_unit = lp.split_iname(t_unit, i, n_wi_per_x, inner_tag="l.0")

    # Data Transformations: Precompute 'B'
    # ------------------------------------
    t_unit = lp.precompute(
        t_unit,
        B,
        sweep_inames={k, j},
        precompute_outer_inames=frozenset([x_outer]),
        precompute_inames=[Bprftch_k, Bprftch_j],
        default_tag="l.auto",
    )

    return t_unit


# ENDEXAMPLE


if __name__ == "__main__":
    import pyopencl as cl
    import os

    cl_ctx = cl.create_some_context()

    my_einsum = f.einsum(
        "xik,kj->xij",
        f.array((np.inf, 35, 35), np.float64),
        f.array((35, 35), np.float64),
    )
    f.autotune(
        my_einsum,
        os.path.abspath(__file__),
        cl_ctx,
        long_dim_length=1_000,
        stop_after=5,
    )
