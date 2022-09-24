import abc
import numpy as np

from loopy.symbolic import CombineMapper, Reduction
from typing import Sequence, Any, FrozenSet, Iterable, Mapping, Tuple
import pymbolic.primitives as p
from feinsum.einsum import FusedEinsum, SummationAxis, ScalarT, EinsumAxisAccess
from more_itertools import zip_equal as zip


class ReductionCollector(CombineMapper):
    def combine(self, values: Iterable[FrozenSet[p.Expression]]):
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_reduction(self, expr: Reduction) -> None:
        return self.combine([frozenset([expr]),
                             super().map_reduction(expr)])

    def map_algebraic_leaf(self, expr: Any) -> FrozenSet[p.Expression]:
        return frozenset()

    def map_constant(self, expr: Any) -> FrozenSet[p.Expression]:
        return frozenset()


class IsReductionSurroundedByIf(CombineMapper):
    def __init__(self, free_indices: FrozenSet[str]) -> None:
        self.free_indices = free_indices
        super().__init__()

    def combine(self, values: Iterable[bool]):
        return any(values)

    def map_reduction(self, expr: Reduction, nested_in_ifelse: bool) -> None:
        return nested_in_ifelse or super().map_reduction(expr,
                                                         nested_in_ifelse)

    def map_algebraic_leaf(self, expr: Any, nested_in_ifelse: bool) -> bool:
        return False

    def map_constant(self, expr: Any, nested_in_ifelse: bool) -> bool:
        return False

    def map_if(self, expr: p.If, nested_in_ifelse: bool) -> bool:
        from loopy.symbolic import get_dependencies
        new_nested_in_ifelse = (nested_in_ifelse
                                or bool(get_dependencies(expr.condition)
                                        & self.free_indices))
        # expr.condition will be always executed, do not worry about its
        # conditional execution
        return self.combine([self.rec(expr.condition, nested_in_ifelse),
                             self.rec(expr.then, new_nested_in_ifelse),
                             self.rec(expr.else_, new_nested_in_ifelse)])


def _strify_index_expr(access_descrs: Tuple[EinsumAxisAccess, ...],
                       access_descr_to_name: Mapping[EinsumAxisAccess, str]) -> str:
    return f"[{', '.join(access_descr_to_name[descr] for descr in access_descrs)}]"


def template_einsum_as_str(einsum: FusedEinsum, iexpr: int) -> str:
    """
    Stringifies (and returns) the einsum expression for *iexpr*-th output. Uses
    'ρ' to denote a scalar expression on its arguments.
    """
    from functools import reduce
    use_row = einsum.use_matrix[iexpr]
    inner_expr = " * ".join(
        f"ρ({','.join(use + _strify_index_expr(acc_descr, einsum.index_names) for use in uses)})"  # noqa: E501
        for acc_descr, uses in zip(einsum.access_descriptors, use_row)
    )

    all_access_descrs = reduce(frozenset.union,
                               [frozenset(access_descrs)
                                for access_descrs in einsum.access_descriptors],
                               frozenset())

    redn_index_names = sorted({einsum.index_names[acc_descr]
                               for acc_descr in all_access_descrs
                               if isinstance(acc_descr, SummationAxis)})

    return f"sum([{','.join(redn_index_names)}], {inner_expr})"


class UseExtractor(CombineMapper):
    """
    Collects all instances of :class:`Use` in an expression.
    """
    def __init__(self, inames):
        ...
    ...


class Match:
    # TODO: Not quite sure what's the correct state to store here?!
    ...


class Use(abc.ABC):
    """
    A global memory access per iteration point of an einsum. An iteration point
    is a point in the iteration domain of the :class:`feinsum.FusedEinsum`,
    where the iteration domain is the cross product of the domain spanned by
    the free indices and the reduction indices of an einsum.
    """


class PlainOldSubscriptUse(Use):
    """
    An unconditional global memory access per iteration point of the domain.
    """
    var_name: str
    indices: Tuple[p.Expression, ...]


class ConditionalUse(Use):
    cond: p.Expression
    then: Use
    else_: Use


def match(exprs: Sequence[p.Expression],
          einsum: FusedEinsum,
          free_indices: Sequence[str],
          dtypes: Mapping[str, np.dtype[Any]],
          shapes: Mapping[str, ScalarT]
          ) -> Match:
    if len(exprs) != einsum.noutputs:
        raise ValueError("The number of outputs do not match.")

    redn_collector = ReductionCollector()
    is_redn_surrounded_by_predicate = IsReductionSurroundedByIf(
        free_indices=frozenset(free_indices))

    for expr in exprs:
        redns_in_expr = redn_collector(expr)
        if not redns_in_expr:
            if any(any(isinstance(access_descr, SummationAxis)
                       for access_descr in access_descrs)
                   for access_descrs in einsum.access_descriptors):
                raise ValueError("No reductions in the current expression, but"
                                 " the reference einsum involves contractions.")

            # TODO: This case is tricky. What do we pick up as the expression
            # being matched over. Typically we look at the expression and then
            # see which expression is the reduction expression and then try to
            # infer the product expression from there. But.. but.. figuring out
            # the same in normal expression is pretty difficult. Maybe we can
            # require some more restrictions on such expressions. Anywho this
            # is no the biggest concern.
            raise NotImplementedError()

        if len(redns_in_expr) > 1:
            raise ValueError("An expression with multiple reduction nodes"
                             " cannot be inferred as an einsum-like"
                             " expression.")

        if is_redn_surrounded_by_predicate(expr, False):
            raise ValueError(f"Reduction in {expr} is nested"
                             " within an if-else arm => not allowed.")

        redn_in_expr, = redns_in_expr
        expr_to_match = redn_in_expr.expr

        if len(einsum.access_descriptors) > 1 and not isinstance(expr_to_match,
                                                                 p.Product):
            raise ValueError("")

        print(expr_to_match)
        1/0

# vim:fdm=marker
