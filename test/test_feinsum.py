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

import numpy as np

import feinsum as f


def are_einsums_isomorphic(e1: f.BatchedEinsum, e2: f.BatchedEinsum) -> bool:
    e1_pr = f.canonicalize_einsum(e1)
    e2_pr = f.canonicalize_einsum(e2)
    return e1_pr == e2_pr


def test_einsum_canonicalization_dg_einsums():

    # {{{ div components

    ndim = 3
    ndofs = 35

    einsum1 = f.batched_einsum(
        "es, sij, ej -> ei",
        [
            [
                f.array("Jx", ("E", ndim)),
                f.array("R", (ndim, ndofs, ndofs)),
                f.array("ux", ("E", ndofs)),
            ],
            [
                f.array("Jy", ("E", ndim)),
                f.array("R", (ndim, ndofs, ndofs)),
                f.array("uy", ("E", ndofs)),
            ],
            [
                f.array("Jz", ("E", ndim)),
                f.array("R", (ndim, ndofs, ndofs)),
                f.array("uz", ("E", ndofs)),
            ],
        ],
    )

    einsum2 = f.batched_einsum(
        "td, dkl, tl -> tk",
        [
            [
                f.array("Jacx", ("E", ndim)),
                f.array("ref_mat", (ndim, ndofs, ndofs)),
                f.array("x_dofs", ("E", ndofs)),
            ],
            [
                f.array("Jacy", ("E", ndim)),
                f.array("ref_mat", (ndim, ndofs, ndofs)),
                f.array("y_dofs", ("E", ndofs)),
            ],
            [
                f.array("Jacz", ("E", ndim)),
                f.array("ref_mat", (ndim, ndofs, ndofs)),
                f.array("z_dofs", ("E", ndofs)),
            ],
        ],
    )

    einsum3 = f.batched_einsum(
        "td, dkl, tl -> tk",
        [
            [
                f.array("Jacx", ("E", ndim)),
                f.array("ref_mat", (ndim, ndofs, ndofs)),
                f.array("u", ("E", ndofs)),
            ],
            [
                f.array("Jacy", ("E", ndim)),
                f.array("ref_mat", (ndim, ndofs, ndofs)),
                f.array("u", ("E", ndofs)),
            ],
            [
                f.array("Jacz", ("E", ndim)),
                f.array("ref_mat", (ndim, ndofs, ndofs)),
                f.array("u", ("E", ndofs)),
            ],
        ],
    )

    einsum4 = f.batched_einsum(
        "es, sij, ej -> ei",
        [
            [
                f.array("Jx", ("E", ndim), "float32"),
                f.array("R", (ndim, ndofs, ndofs), "float32"),
                f.array("ux", ("E", ndofs), "float32"),
            ],
            [
                f.array("Jy", ("E", ndim), "float32"),
                f.array("R", (ndim, ndofs, ndofs), "float32"),
                f.array("uy", ("E", ndofs), "float32"),
            ],
            [
                f.array("Jz", ("E", ndim), "float32"),
                f.array("R", (ndim, ndofs, ndofs), "float32"),
                f.array("uz", ("E", ndofs), "float32"),
            ],
        ],
    )

    assert are_einsums_isomorphic(einsum1, einsum2)
    assert are_einsums_isomorphic(
        f.canonicalize_einsum(einsum1), f.canonicalize_einsum(einsum2)
    )
    assert not are_einsums_isomorphic(einsum2, einsum3)
    assert not are_einsums_isomorphic(einsum1, einsum4)

    # }}}


def test_canonicalization_with_automorphic_vertices():
    assert are_einsums_isomorphic(
        f.einsum(
            "ij,ik->i",
            f.array("A", ("I", 10), np.float64),
            f.array("B", ("I", 10), np.float32),
        ),
        f.einsum(
            "ik,ij->i",
            f.array("C", ("J", 10), np.float32),
            f.array("D", ("J", 10), np.float64),
        ),
    )

    assert not are_einsums_isomorphic(
        f.einsum(
            "ijk,ij,ik->i",
            f.array("A", ("I", 10, 10), np.float64),
            f.array("B", ("I", 10), np.float64),
            f.array("C", ("I", 10), np.float32),
        ),
        f.einsum(
            "ijk,ij,ik->i",
            f.array("A", ("I", 10, 10), np.float64),
            f.array("B", ("I", 10), np.float32),
            f.array("C", ("I", 10), np.float64),
        ),
    )

    assert are_einsums_isomorphic(
        f.einsum(
            "ijk,ij,ik->i",
            f.array("A", ("I", 10, 10), np.float64),
            f.array("B", ("I", 10), np.float64),
            f.array("C", ("I", 10), np.float64),
        ),
        f.einsum(
            "ijk,ik,ij->i",
            f.array("P", ("J", 10, 10), np.float64),
            f.array("Q", ("J", 10), np.float64),
            f.array("R", ("J", 10), np.float64),
        ),
    )

    assert not are_einsums_isomorphic(
        f.batched_einsum(
            "ijk,ik,ij,ij->i",
            [
                [
                    f.array("A", ("I", 10, 10)),
                    f.array("B", ("I", 10)),
                    f.array("C", ("I", 10)),
                    f.array("D", ("I", 10)),
                ]
            ],
        ),
        f.batched_einsum(
            "ijk,ik,ij,ik->i",
            [
                [
                    f.array("P", ("L", 10, 10)),
                    f.array("Q", ("L", 10)),
                    f.array("R", ("L", 10)),
                    f.array("S", ("L", 10)),
                ]
            ],
        ),
    )

    assert are_einsums_isomorphic(
        f.batched_einsum(
            "ijk,ik,ij,ij->i",
            [
                [
                    f.array("A", ("I", 10, 10)),
                    f.array("B", ("I", 10)),
                    f.array("C", ("I", 10)),
                    f.array("D", ("I", 10)),
                ]
            ],
        ),
        f.batched_einsum(
            "ikj,ik,ij,ik->i",
            [
                [
                    f.array("P", ("L", 10, 10)),
                    f.array("Q", ("L", 10)),
                    f.array("R", ("L", 10)),
                    f.array("S", ("L", 10)),
                ]
            ],
        ),
    )

    assert are_einsums_isomorphic(
        f.batched_einsum(
            "ijk,ik,ij,ij->i",
            [
                [
                    f.array("A", ("I", 10, 10)),
                    f.array("B", ("I", 10)),
                    f.array("C", ("I", 10)),
                    f.array("D", ("I", 10)),
                ],
                [
                    f.array("A", ("I", 10, 10)),
                    f.array("B", ("I", 10)),
                    f.array("C", ("I", 10)),
                    f.array("B", ("I", 10)),
                ],
            ],
        ),
        f.batched_einsum(
            "elm,em,el,el->e",
            [
                [
                    f.array("P", ("J", 10, 10)),
                    f.array("Q", ("J", 10)),
                    f.array("R", ("J", 10)),
                    f.array("Q", ("J", 10)),
                ],
                [
                    f.array("P", ("J", 10, 10)),
                    f.array("Q", ("J", 10)),
                    f.array("R", ("J", 10)),
                    f.array("S", ("J", 10)),
                ],
            ],
        ),
    )


def test_canonicalization_of_large_graphs():
    expr1 = f.batched_einsum(
        "ij,ej->ei",
        [
            [f.array(f"u{i}", (35, 35)), f.array(f"v{i}", ("E", 35))]
            for i in range(500)
        ],
    )
    expr2 = f.batched_einsum(
        "et,st->es",
        [
            [f.array(f"a{i}", ("E", 35)), f.array(f"b{i}", (35, 35))]
            for i in range(500)
        ],
    )

    assert are_einsums_isomorphic(expr1, expr2)


# vim: fdm=marker
