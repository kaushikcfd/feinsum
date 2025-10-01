"""
.. autofunction:: has_similar_subscript
.. autofunction:: is_any_redn_dim_parametric
.. autofunction:: get_n_redn_dim

.. autoclass:: IndexNameGenerator
"""


from feinsum.einsum import BatchedEinsum, SummationAxis, SizeParam
from feinsum import array
from feinsum.make_einsum import einsum as build_einsum
from typing import FrozenSet
import dataclasses as dc


def has_similar_subscript(einsum: BatchedEinsum,
                          subscript: str) -> bool:
    """
    Returns *True* only if *einsum*'s expression being applied
    to its operands is equivalently represented by *subscript*.

    .. testsetup::

        >>> from feinsum.utils import has_similar_subscript

    .. doctest::

        >>> import feinsum as f
        >>> ensm = f.einsum("ij,j->i",
        ...                 f.array((10, 4), "float64"),
        ...                 f.array(4, "float64"))
        >>> has_similar_subscript(ensm, "ij,j->i")
        True
        >>> has_similar_subscript(ensm, "ik,k->i")
        True
        >>> has_similar_subscript(ensm, "ik,kj->ij")
        False
        >>> ensm = f.einsum("ik,kj->ij",
        ...                 f.array((10, 4), "float64"),
        ...                 f.array((4, 100), "float64"))
        >>> has_similar_subscript(ensm, "ij,j->i")
        False
        >>> has_similar_subscript(ensm, "ik,kj->ij")
        True
        >>> has_similar_subscript(ensm, "pq,qr->pr")
        True
    """

    try:
        ref_einsum = build_einsum(subscript,
                                   *[array(shape, "float64")
                                     for shape in einsum.arg_shapes])
    except ValueError:
        return False
    else:
        return (ref_einsum.access_descriptors
                == einsum.access_descriptors)


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

    for arg_shape, access_descrs in zip(einsum.arg_shapes,
                                        einsum.access_descriptors):
        for access_descr, dim in zip(access_descrs, arg_shape):
            if (isinstance(access_descr, SummationAxis)
                    and isinstance(dim, SizeParam)):
                return True

    return False


def get_n_redn_dim(ensm: BatchedEinsum) -> int:
    """
    Returns the number of reduction indices in *ensm*.
    """
    return len({axis
                for axis in ensm.index_to_dim_length()
                if isinstance(axis, SummationAxis)})


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
    banned_names: FrozenSet[str] = dc.field(default=frozenset())
    counter: int = dc.field(init=False, default=0)

    def __call__(self) -> str:
        if self.counter == 26:
            raise RuntimeError("All indices have been exhausted")

        new_name = chr(97+self.counter)
        self.counter += 1

        if new_name in self.banned_names:
            return self()
        else:
            return new_name
