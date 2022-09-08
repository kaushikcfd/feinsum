"""
.. automethod:: has_similar_subscript
.. automethod:: is_any_redn_dim_parametric
"""


from feinsum.einsum import FusedEinsum, SummationAxis, SizeParam
from feinsum import array
from feinsum.make_einsum import einsum as build_einsum


def has_similar_subscript(einsum: FusedEinsum,
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


def is_any_redn_dim_parametric(einsum: FusedEinsum) -> bool:
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
