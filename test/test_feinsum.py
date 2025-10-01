__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import feinsum as f
import numpy as np


def are_einsums_isomorphic(e1: f.BatchedEinsum,
                           e2: f.BatchedEinsum) -> bool:
    e1_pr = f.canonicalize_einsum(e1)
    e2_pr = f.canonicalize_einsum(e2)
    if 0:
        # enable for debugging
        print(e1_pr.arg_shapes)
        print(e2_pr.arg_shapes)
        for use_row in e1_pr.use_matrix:
            print(use_row)
        for use_row in e2_pr.use_matrix:
            print(use_row)
    return e1_pr == e2_pr


def test_einsum_canonicalization_dg_einsums():

    # {{{ div components

    ndim = 3
    ndofs = 35

    einsum1 = f.batched_einsum("es, sij, ej -> ei",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes=np.float64,
                            use_matrix=[
                                [{"Jx"}, {"R"}, {"ux"}],
                                [{"Jy"}, {"R"}, {"uy"}],
                                [{"Jz"}, {"R"}, {"uz"}],
                            ])

    einsum2 = f.batched_einsum("td, dkl, tl -> tk",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes=np.float64,
                            use_matrix=[
                                [{"JacX"}, {"ref_mat"}, {"x_dofs"}],
                                [{"JacY"}, {"ref_mat"}, {"y_dofs"}],
                                [{"JacZ"}, {"ref_mat"}, {"z_dofs"}],
                            ])

    einsum3 = f.batched_einsum("td, dkl, tl -> tk",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes=np.float64,
                            use_matrix=[
                                [{"JacX"}, {"ref_mat"}, {"u"}],
                                [{"JacY"}, {"ref_mat"}, {"u"}],
                                [{"JacZ"}, {"ref_mat"}, {"u"}],
                            ])

    einsum4 = f.batched_einsum("es, sij, ej -> ei",
                            [(np.inf, ndim),
                            (ndim, ndofs, ndofs),
                            (np.inf, ndofs)],
                            dtypes=np.float32,
                            use_matrix=[
                                [{"Jx"}, {"R"}, {"ux"}],
                                [{"Jy"}, {"R"}, {"uy"}],
                                [{"Jz"}, {"R"}, {"uz"}],
                            ])
    assert are_einsums_isomorphic(einsum1, einsum2)
    assert are_einsums_isomorphic(f.canonicalize_einsum(einsum1),
                                  f.canonicalize_einsum(einsum2))
    assert not are_einsums_isomorphic(einsum2, einsum3)
    assert not are_einsums_isomorphic(einsum1, einsum4)

    # }}}


def test_canonicalization_with_automorphic_vertices():
    assert are_einsums_isomorphic(
        f.einsum("ij,ik->i",
                 f.array((np.inf, 10), np.float64),
                 f.array((np.inf, 10), np.float32)),

        f.einsum("ik,ij->i",
                 f.array((np.inf, 10), np.float32),
                 f.array((np.inf, 10), np.float64)),
    )

    assert not are_einsums_isomorphic(
        f.einsum("ijk,ij,ik->i",
                 f.array((np.inf, 10, 10), np.float64),
                 f.array((np.inf, 10), np.float64),
                 f.array((np.inf, 10), np.float32)),

        f.einsum("ijk,ij,ik->i",
                 f.array((np.inf, 10, 10), np.float64),
                 f.array((np.inf, 10), np.float32),
                 f.array((np.inf, 10), np.float64)),
    )

    assert are_einsums_isomorphic(
        f.einsum("ijk,ij,ik->i",
                 f.array((np.inf, 10, 10), np.float64),
                 f.array((np.inf, 10), np.float64),
                 f.array((np.inf, 10), np.float64)),

        f.einsum("ijk,ik,ij->i",
                 f.array((np.inf, 10, 10), np.float64),
                 f.array((np.inf, 10), np.float64),
                 f.array((np.inf, 10), np.float64)),
    )

    assert are_einsums_isomorphic(
        f.batched_einsum("ijk,ik,ij,ij->i",
                       [(np.inf, 10, 10),
                        (np.inf, 10),
                        (np.inf, 10),
                        (np.inf, 10)],
                       dtypes=np.float64,
                       use_matrix=[
                           [{"A"}, {"B"}, {"C"}, {"D"}]
                       ]),
        f.batched_einsum("ijk,ik,ij->i",
                       [(np.inf, 10, 10),
                        (np.inf, 10),
                        (np.inf, 10)],
                       dtypes=np.float64,
                       use_matrix=[
                           [{"P"}, {"Q"}, {"R", "S"}]
                       ]),
    )

    assert not are_einsums_isomorphic(
        f.batched_einsum("ijk,ik,ij,ij->i",
                       [(np.inf, 10, 10),
                        (np.inf, 10),
                        (np.inf, 10),
                        (np.inf, 10)],
                       dtypes=np.float64,
                       use_matrix=[
                           [{"A"}, {"B"}, {"C"}, {"D"}]
                       ]),
        f.batched_einsum("ijk,ij,ik->i",
                       [(np.inf, 10, 10),
                        (np.inf, 10),
                        (np.inf, 10)],
                       dtypes=np.float64,
                       use_matrix=[
                           [{"P"}, {"Q"}, {"R", "S"}]
                       ]),
    )

    assert are_einsums_isomorphic(
        f.batched_einsum("ijk,ik,ij,ij->i",
                       [(np.inf, 10, 10),
                        (np.inf, 10),
                        (np.inf, 10),
                        (np.inf, 10)],
                       dtypes=np.float64,
                       use_matrix=[
                           [{"A"}, {"B"}, {"C"}, {"D"}],
                           [{"A"}, {"B"}, {"C"}, {"B"}]
                       ]),
        f.batched_einsum("ijk,ik,ij->i",
                       [(np.inf, 10, 10),
                        (np.inf, 10),
                        (np.inf, 10)],
                       dtypes=np.float64,
                       use_matrix=[
                           [{"P"}, {"Q"}, {"R", "Q"}],
                           [{"P"}, {"Q"}, {"R", "S"}]
                       ]),
    )


def test_canonicalization_of_large_graphs():
    expr1 = f.batched_einsum("ij,ej->ei",
                           [(35, 35), (np.inf, 35)],
                           dtypes=np.float64,
                           use_matrix=[[{f"u{i}"}, {f"v{i}"}]
                                       for i in range(500)])
    expr2 = f.batched_einsum("et,st->es",
                          [(np.inf, 35), (35, 35)],
                          dtypes=np.float64,
                          use_matrix=[[{f"a{i}"}, {f"b{i}"}]
                                      for i in range(500)])

    assert are_einsums_isomorphic(expr1, expr2)


# vim: fdm=marker
