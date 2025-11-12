"""
.. autofunction:: is_any_redn_dim_parametric
.. autofunction:: get_n_redn_dim
.. autofunction:: get_tccg_benchmark

.. autoclass:: IndexNameGenerator
"""

import dataclasses as dc
from typing import Any

import numpy as np

from feinsum.einsum import BatchedEinsum, SizeParam, SummationAxis


def is_any_redn_dim_parametric(einsum: BatchedEinsum) -> bool:
    """
    .. testsetup::

        >>> from feinsum.utils import is_any_redn_dim_parametric
        >>> import numpy as np

    .. doctest::

        >>> import feinsum as f
        >>> ensm = f.einsum("ij,j->i",
        ...                 f.array((10, 4), "float64"),
        ...                 f.array(4, "float64"))
        >>> is_any_redn_dim_parametric(ensm)
        False
        >>> ensm = f.einsum("ij,jk->i",
        ...                 f.array((10, 4), "float64"),
        ...                 f.array((4, 20), "float64"))
        >>> is_any_redn_dim_parametric(ensm)
        False
        >>> ensm = f.einsum("i,j->i",
        ...                 f.array(125, "float64"),
        ...                 f.array(np.inf, "float64"))
        >>> is_any_redn_dim_parametric(ensm)
        True
    """
    return any(
        isinstance(dim_len, SizeParam)
        and isinstance(einsum.index_to_access_descr[idx], SummationAxis)
        for idx, dim_len in einsum.index_to_dim_length.items()
    )


def get_n_redn_dim(ensm: BatchedEinsum) -> int:
    """
    Returns the number of reduction indices in *ensm*.
    """
    return len(
        {
            idx
            for idx, access in ensm.index_to_access_descr.items()
            if isinstance(access, SummationAxis)
        }
    )


@dc.dataclass
class IndexNameGenerator:
    """
    Generates indices to be fed into an einstein summation.

    .. attribute:: banned_names

        A :class:`frozenset` of index names that will not be generated.

    .. doctest::

        >>> from feinsum import IndexNameGenerator
        >>> idx_name_gen = IndexNameGenerator(frozenset({'c'}))
        >>> idx_name_gen()
        'a'
        >>> idx_name_gen()
        'b'
        >>> idx_name_gen()
        'd'
    """

    banned_names: frozenset[str] = dc.field(default=frozenset())
    counter: int = dc.field(init=False, default=0)

    def __call__(self) -> str:
        if self.counter == 26:
            raise RuntimeError("All indices have been exhausted")

        new_name = chr(97 + self.counter)
        self.counter += 1

        if new_name in self.banned_names:
            return self()
        else:
            return new_name


# {{{ TCCG Benchmark suite


def _get_tccg_input_strings(i: int) -> tuple[str, str]:
    # See
    # <https://github.com/kimjsung/CGO2019-AE/tree/ff6283c1b5a85faaf7e44e89a0c7661373a5a27b/cogent/input_strings/tccg>.
    if i == 1:
        return ("abc-bda-dc", "312 312 24 312")
    elif i == 2:
        return ("abc-dca-bd", "312 24 296 312")
    elif i == 3:
        return ("abcd-dbea-ec", "72 72 24 72 72")
    elif i == 4:
        return ("abcd-deca-be", "72 24 72 72 72")
    elif i == 5:
        return ("abcd-ebad-ce", "72 72 24 72 72")
    elif i == 6:
        return ("abcde-efbad-cf", "48 32 24 32 48 32")
    elif i == 7:
        return ("abcde-ecbfa-fd", "48 32 32 24 48 48")
    elif i == 8:
        return ("abcde-efcad-bf", "48 24 32 32 48 32")
    elif i == 9:
        return ("abcd-ea-ebcd", "72 72 72 72 72")
    elif i == 10:
        return ("abcd-eb-aecd", "72 72 72 72 72")
    elif i == 11:
        return ("abcd-ec-abed", "72 72 72 72 72")
    elif i == 12:
        return ("ab-ac-cb", "5136 5120 5136")
    elif i == 13:
        return ("ab-acd-dbc", "312 296 296 312")
    elif i == 14:
        return ("ab-cad-dcb", "312 296 312 312")
    elif i == 15:
        return ("abc-acd-db", "312 296 296 312")
    elif i == 16:
        return ("abc-ad-bdc", "312 312 296 296")
    elif i == 17:
        return ("abc-adc-bd", "312 312 296 296")
    elif i == 18:
        return ("abc-adc-db", "312 296 296 312")
    elif i == 19:
        return ("abc-adec-ebd", "72 72 72 72 72")
    elif i == 20:
        return ("abcd-aebf-dfce", "72 72 72 72 72 72")
    elif i == 21:
        return ("abcd-aebf-fdec", "72 72 72 72 72 72")
    elif i == 22:
        return ("abcd-aecf-bfde", "72 72 72 72 72 72")
    elif i == 23:
        return ("abcd-aecf-fbed", "72 72 72 72 72 72")
    elif i == 24:
        return ("abcd-aedf-bfce", "72 72 72 72 72 72")
    elif i == 25:
        return ("abcd-aedf-fbec", "72 72 72 72 72 72")
    elif i == 26:
        return ("abcd-aefb-fdce", "72 72 72 72 72 72")
    elif i == 27:
        return ("abcd-aefc-fbed", "72 72 72 72 72 72")
    elif i == 28:
        return ("abcd-eafb-fdec", "72 72 72 72 72 72")
    elif i == 29:
        return ("abcd-eafc-bfde", "72 72 72 72 72 72")
    elif i == 30:
        return ("abcd-eafd-fbec", "72 72 72 72 72 72")
    elif i == 31:
        return ("abcdef-dega-gfbc", "24 16 16 24 16 16 24")
    elif i == 32:
        return ("abcdef-degb-gfac", "24 16 16 24 16 16 24")
    elif i == 33:
        return ("abcdef-degc-gfab", "24 16 16 24 16 16 24")
    elif i == 34:
        return ("abcdef-dfga-gebc", "24 16 16 24 16 16 24")
    elif i == 35:
        return ("abcdef-dfgb-geac", "24 16 16 24 16 16 24")
    elif i == 36:
        return ("abcdef-dfgc-geab", "24 16 16 24 16 16 24")
    elif i == 37:
        return ("abcdef-efga-gdbc", "24 16 16 16 24 16 24")
    elif i == 38:
        return ("abcdef-efgb-gdac", "24 16 16 16 24 16 24")
    elif i == 39:
        return ("abcdef-efgc-gdab", "24 16 16 16 24 16 24")
    elif i == 40:
        return ("abcdef-gdab-efgc", "24 16 16 16 24 16 24")
    elif i == 41:
        return ("abcdef-gdac-efgb", "24 16 16 16 24 16 24")
    elif i == 42:
        return ("abcdef-gdbc-efga", "24 16 16 16 24 16 24")
    elif i == 43:
        return ("abcdef-geab-dfgc", "24 16 16 24 16 16 24")
    elif i == 44:
        return ("abcdef-geac-dfgb", "24 16 16 24 16 16 24")
    elif i == 45:
        return ("abcdef-gebc-dfga", "24 16 16 24 16 16 24")
    elif i == 46:
        return ("abcdef-gfab-degc", "24 16 16 24 16 16 24")
    elif i == 47:
        return ("abcdef-gfac-degb", "24 16 16 24 16 16 24")
    elif i == 48:
        return ("abcdef-gfbc-dega", "24 16 16 24 16 16 24")
    else:
        raise ValueError("i must be in the set {1, 2, .., 48}." f" Got {i = }.")


def get_tccg_benchmark(i: int, dtype: Any = np.float64) -> BatchedEinsum:
    r"""
    Returns the *i*-th tensor contraction `TCCG benchamark suite
    <https://dl.acm.org/doi/abs/10.5555/3314872.3314885>`_ .

    TCCG benchmark suite contains 48 tensor contractions, consequently *i* must
    belong to :math:`\{1, 2, \ldots, 48\}`.

    :arg i: Serial number of the tensor contraction in the benchmark suite.
    :arg dtype: Dtype of the array operands in the constructed tensor contraction.
    """
    from feinsum.make_einsum import array, einsum

    subscript, shape = _get_tccg_input_strings(i)

    output, inA, inB = subscript.split("-")
    axis_lens = {
        chr(97 + i): int(axis_len) for i, axis_len in enumerate(shape.split(" "))
    }

    shapeA = [axis_lens[idx] for idx in inA]
    shapeB = [axis_lens[idx] for idx in inB]

    return einsum(
        f"{inA},{inB}->{output}",
        array("A", shapeA, dtype),
        array("B", shapeB, dtype),
    )


# }}}

# vim: fdm=marker
