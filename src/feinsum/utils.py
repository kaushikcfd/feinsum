"""
.. autofunction:: is_any_redn_dim_parametric
.. autofunction:: get_n_redn_dim

.. autoclass:: IndexNameGenerator
"""

import dataclasses as dc

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
