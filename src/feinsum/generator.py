"""
Generates Loopy kernels on which transformations could be applied.
"""

import loopy as lp

from feinsum.einsum import Einsum


def generate_loopy_kernel(einsum_decr: Einsum) -> lp.TranslationUnit:
    ...
