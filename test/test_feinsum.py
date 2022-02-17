import feinsum as f
import numpy as np


def test_einsum_normalization_dg_einsums():

    # {{{ div components

    ndim = 3
    ndofs = 35

    einsum1 = f.fused_einsum("es, sij, ej -> ei",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes="float64",
                            use_matrix=[
                                [{"Jx"}, {"R"}, {"ux"}],
                                [{"Jy"}, {"R"}, {"uy"}],
                                [{"Jz"}, {"R"}, {"uz"}],
                            ])

    einsum2 = f.fused_einsum("td, dkl, tl -> tk",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes="float64",
                            use_matrix=[
                                [{"JacX"}, {"ref_mat"}, {"x_dofs"}],
                                [{"JacY"}, {"ref_mat"}, {"y_dofs"}],
                                [{"JacZ"}, {"ref_mat"}, {"z_dofs"}],
                            ])

    einsum3 = f.fused_einsum("td, dkl, tl -> tk",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes="float64",
                            use_matrix=[
                                [{"JacX"}, {"ref_mat"}, {"u"}],
                                [{"JacY"}, {"ref_mat"}, {"u"}],
                                [{"JacZ"}, {"ref_mat"}, {"u"}],
                            ])

    einsum4 = f.fused_einsum("es, sij, ej -> ei",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes="float32",
                            use_matrix=[
                                [{"Jx"}, {"R"}, {"ux"}],
                                [{"Jy"}, {"R"}, {"uy"}],
                                [{"Jz"}, {"R"}, {"uz"}],
                            ])

    assert (f.normalize_einsum(einsum1)
            == f.normalize_einsum(einsum2)
            == f.normalize_einsum(f.normalize_einsum(einsum1))
            == f.normalize_einsum(f.normalize_einsum(einsum2)))
    assert f.normalize_einsum(einsum2) != f.normalize_einsum(einsum3)
    assert f.normalize_einsum(einsum1) != f.normalize_einsum(einsum4)

    # }}}
