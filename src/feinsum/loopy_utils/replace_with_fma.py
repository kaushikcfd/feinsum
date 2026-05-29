from __future__ import annotations

__doc__ = """
.. autofunction:: replace_with_fma
"""

__copyright__ = "Copyright (C) 2026 Kaushik Kulkarni"

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

from typing import TYPE_CHECKING

import loopy as lp
import loopy.match as lp_match
import pymbolic.primitives as p
from constantdict import constantdict
from loopy.symbolic import (
    ExpansionState,
    RuleAwareIdentityMapper,
    SubstitutionRuleMappingContext,
)
from loopy.type_inference import TypeReader
from pymbolic import flattened_product
from pymbolic.primitives import is_arithmetic_expression

if TYPE_CHECKING:
    from loopy.library.reduction import ReductionOpFunction
    from loopy.translation_unit import CallablesTable
    from pymbolic.typing import Expression


def _try_replace_expr_with_fma(
    product: Expression,
    addend: Expression,
) -> p.Call | None:
    """
    If *product* is a :class:`~pymbolic.primitives.Product` that qualifies for
    FMA replacement, return :math:`\\operatorname{fma}(a, b, \\text{addend})`
    as a :class:`~pymbolic.primitives.Call`; otherwise return *None*.

    A product qualifies when, after flattening nested products via
    :func:`pymbolic.flattened_product`, it remains a
    :class:`~pymbolic.primitives.Product` with at least two factors and is not
    a bare negation (i.e. not exactly :math:`{-1} \\cdot x`).  The first
    non-:math:`{-1}` factor becomes *fma_a*; the product of the remaining
    factors becomes *fma_b*.
    """
    if not (
        isinstance(product, p.Product)
        and isinstance(
            flat_product := flattened_product(product.children), p.Product
        )
        and len(flat_product.children) > 1
        and not (
            len(flat_product.children) == 2
            and any(child == -1 for child in flat_product.children)
        )
    ):
        return None

    fma_a_idx = next(
        idx for idx, term in enumerate(flat_product.children) if term != -1
    )
    remaining_terms = tuple(
        term for idx, term in enumerate(flat_product.children) if idx != fma_a_idx
    )
    fma_a = flat_product.children[fma_a_idx]
    fma_b = (
        remaining_terms[0]
        if len(remaining_terms) == 1
        else p.Product(remaining_terms)
    )
    return p.Call(p.Variable("fma"), (fma_a, fma_b, addend))


class FMAReplacer(RuleAwareIdentityMapper[[]]):
    type_reader: TypeReader

    def __init__(
        self,
        rule_mapping_context: SubstitutionRuleMappingContext,
        type_reader: TypeReader,
    ) -> None:
        super().__init__(rule_mapping_context)
        self.type_reader = type_reader

    def map_sum(self, expr: p.Sum, expn_state: ExpansionState) -> Expression:
        dtypes = self.type_reader(expr, return_dtype_set=True)
        if len(dtypes) != 1:
            return super().map_sum(expr, expn_state)

        (dtype,) = dtypes

        if (
            dtype.numpy_dtype.kind != "f"
            or len(expr.children) != 2
            or any(child == 0 for child in expr.children)
        ):
            return super().map_sum(expr, expn_state)

        left, right = expr.children
        rec_left, rec_right = self.rec(left, expn_state), self.rec(right, expn_state)
        if (call := _try_replace_expr_with_fma(rec_left, rec_right)) is not None or (
            call := _try_replace_expr_with_fma(rec_right, rec_left)
        ) is not None:
            return call
        else:
            assert is_arithmetic_expression(rec_left)
            assert is_arithmetic_expression(rec_right)
            return p.Sum((rec_left, rec_right))


def _replace_with_fma_inner(
    kernel: lp.LoopKernel,
    within: lp_match.ToStackMatchConvertible,
    callables: CallablesTable,
) -> lp.LoopKernel:
    type_reader = TypeReader(kernel, callables)  # type: ignore[no-untyped-call]
    within = lp_match.parse_stack_match(within)
    vng = kernel.get_var_name_generator()
    rule_mapping_context = SubstitutionRuleMappingContext(kernel.substitutions, vng)
    fma_replacer = FMAReplacer(rule_mapping_context, type_reader)
    # map_args/map_tvs=False: arg decls and temp var decls carry no RHS to transform
    new_knl = fma_replacer.map_kernel(kernel, within, False, False)
    if new_knl != kernel:
        new_knl = new_knl.copy(state=lp.KernelState.INITIAL)
    return new_knl


def replace_with_fma(
    t_unit: lp.TranslationUnit, within: lp_match.ToStackMatchConvertible = None
) -> lp.TranslationUnit:
    r"""
    Returns a transformed version of *t_unit* where floating-point
    subexpressions of the form :math:`a \cdot b + c` are replaced with
    :math:`\operatorname{fma}(a, b, c)`.

    **Pattern matched.** A :class:`~pymbolic.primitives.Sum` with *exactly two
    children*, one of which is a :class:`~pymbolic.primitives.Product` with at
    least two factors.  For a product :math:`a \cdot b \cdots z`, the first
    factor becomes the first FMA argument and the product of the remaining
    factors becomes the second, i.e.
    :math:`\operatorname{fma}(a,\, b \cdots z,\, c)`.  Both orderings
    :math:`a \cdot b + c` and :math:`c + a \cdot b` are matched.  Sums of
    three or more terms are not directly matched; loopy's expression trees
    normally nest them as binary sums, so each binary pair is still reachable
    by the mapper.

    :arg t_unit: the translation unit to transform.
    :arg within: selects which instructions are eligible for rewriting.
        Accepts anything accepted by
        :func:`loopy.match.parse_stack_match` — a :class:`str` match
        expression, a :class:`~loopy.match.MatchExpressionBase`, a
        :class:`~loopy.match.ConcreteStackMatch`, or ``None`` (match
        everything).
    """

    from loopy.kernel.function_interface import CallableKernel
    t_unit = lp.infer_unknown_types(t_unit)

    clbls = t_unit.callables_table
    new_clbls: dict[str | ReductionOpFunction, lp.InKernelCallable] = {}
    for name, clbl in clbls.items():
        if isinstance(clbl, CallableKernel):
            new_clbls[name] = clbl.copy(
                subkernel=_replace_with_fma_inner(clbl.subkernel, within, clbls)
            )
        else:
            new_clbls[name] = clbl

    return t_unit.copy(callables_table=constantdict(new_clbls))
