import abc
import numpy as np
import pymbolic.primitives as p

from loopy.symbolic import CombineMapper, Reduction
from typing import Sequence, Any, FrozenSet, Iterable, Mapping, Tuple, Set, Dict
from feinsum.einsum import FusedEinsum, SummationAxis, ScalarT, EinsumAxisAccess
from more_itertools import zip_equal as zip
from functools import cached_property
from dataclasses import dataclass
from pytools import memoize_on_first_arg
from immutables import Map


class ReductionCollector(CombineMapper):
    """
    A mapper that collects all instances of :class:`loopy.symbolic.Reduction`
    in an expression.
    """
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
    """
    Mapper that returns *True* only if the expression contains a reduction
    expression that is conditionally executed depending on the values of
    *free_indices*.
    """
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
    return f"[{','.join(access_descr_to_name[descr] for descr in access_descrs)}]"


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


# {{{

class Use(abc.ABC):
    """
    A global memory access per iteration point of an einsum. An iteration point
    is a point in the iteration domain of the :class:`feinsum.FusedEinsum`,
    where the iteration domain is the cross product of the domain spanned by
    the free indices and the reduction indices of an einsum.
    """


@dataclass(frozen=True)
class PlainOldSubscriptUse(Use):
    """
    An unconditional global memory access per iteration point of the domain.
    """
    var_name: str
    indices: Tuple[p.Expression, ...]


@dataclass(frozen=True)
class ConditionalUse(Use):
    """
    A conditional global memory access per iteration point of the einsum's
    domain. The condition depends on dimensions spanning the iteration domain.

    .. note::

        - Since a :class:`Use` is associated with a fixed size global memory
          access the data type of both branches of a conditional use must be
          the same.
    """
    cond: p.Expression
    then: Use
    else_: Use

# }}}


class ConditionalUseDtypeMismatchError(ValueError):
    """
    Raised by :func:`_get_use_dtype` if it encounters a :class:`ConditionalUse`
    which uses with different dtypes in both of its branches.
    """


@memoize_on_first_arg
def _get_use_dtype(use: Use, var_name_to_dtype: Map[str, np.dtype[Any]]):
    """
    Returns the data type of *use*. By definition, *use* must be associated
    with a single dtype.
    """
    if isinstance(use, PlainOldSubscriptUse):
        return var_name_to_dtype[use.var_name]
    elif isinstance(use, ConditionalUse):
        then_dtype = _get_use_dtype(use.then, var_name_to_dtype)
        else_dtype = _get_use_dtype(use.else_, var_name_to_dtype)
        if then_dtype != else_dtype:
            raise ConditionalUseDtypeMismatchError
        else:
            return then_dtype
    else:
        raise NotImplementedError(type(use))


class UseExtractor(CombineMapper):
    """
    Collects all instances of :class:`Use` in an expression.

    .. attribute:: inames

        A :class:`frozenset` of :class:`str` containing the names by which the
        dimensions of einsum are referred to.
    """
    def __init__(self,
                 inames: FrozenSet[str],
                 var_to_dtype: Map[str, np.dtype[Any]]) -> None:
        super().__init__()
        self.inames = inames
        self.var_to_dtype = var_to_dtype

    def combine(self, values: FrozenSet[Use]) -> FrozenSet[Use]:
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    @cached_property
    def _dep_mapper(self):
        from loopy.symbolic import DependencyMapper
        return DependencyMapper()

    def get_dependencies(self, expr: p.Expression) -> FrozenSet[str]:
        return frozenset({x.name
                          for x in self._dep_mapper(expr)})

    def map_subscript(self, expr: p.Subscript) -> FrozenSet[Use]:
        if not isinstance(expr.aggregate, p.Variable):
            # TODO: Is this a case we even need to worry about?
            raise NotImplementedError
        if any(bool(self.get_dependencies(idx) & self.inames)
               for idx in expr.index_tuple):
            return frozenset([PlainOldSubscriptUse(
                expr.aggregate.name, expr.index_tuple)])
        else:
            return frozenset()

    def map_if(self, expr: p.If) -> FrozenSet[Use]:
        if self.rec(expr.condition):
            # the condition expression contains a use.
            raise NotImplementedError(
                "conditions that access variables in global"
                " memory aren't (yet) supported.")
        else:
            if (self.get_dependencies(expr.condition) & self.inames):
                then_uses = self.rec(expr.then)
                else_uses = self.rec(expr.else_)

                if len(then_uses) != len(else_uses):
                    raise ValueError(f"Branches of '{expr}' have different number "
                                     "of uses => disallowed as predicting its "
                                     "performance is quite challenging.")

                dtype_to_then_uses: Dict[np.dtype[Any], Set[Use]] = {}
                dtype_to_else_uses: Dict[np.dtype[Any], Set[Use]] = {}

                try:
                    for then_use, else_use in zip(then_uses, else_uses):
                        dtype_to_then_uses.setdefault(_get_use_dtype(
                                                        then_use,
                                                        self.var_to_dtype),
                                                      set()).add(then_use)
                        dtype_to_else_uses.setdefault(_get_use_dtype(
                                                        else_use,
                                                        self.var_to_dtype),
                                                      set()).add(else_use)
                except ConditionalUseDtypeMismatchError:
                    raise NotImplementedError("Matching with a weak-use match"
                                              " is not supported.")

                if (set(dtype_to_then_uses) != set(dtype_to_else_uses)
                        or any((len(dtype_to_then_uses[dtype])
                                != len(dtype_to_else_uses[dtype]))
                               for dtype in dtype_to_then_uses)):
                    raise NotImplementedError("matching with a weak-use match"
                                              " is not supported.")
                else:
                    new_uses = set()
                    for dtype in dtype_to_then_uses:
                        new_uses.update({ConditionalUse(expr.condition,
                                                        then_use,
                                                        else_use)
                                         for then_use, else_use in zip(
                                             dtype_to_then_uses[dtype],
                                             dtype_to_else_uses[dtype])})

                    return frozenset(new_uses)
            else:
                # static condition. (not yet implemented)
                raise NotImplementedError()

    def map_constant(self, expr: Any) -> FrozenSet[Use]:
        return frozenset()

    def map_algebraic_leaf(self, expr: Any) -> FrozenSet[Use]:
        return frozenset()


@dataclass(frozen=True)
class Match:
    einsum_use_to_loopy_use: Mapping[str, Use]
    loopy_operands: Tuple[Tuple[p.Expression, ...]]


def match(exprs: Sequence[p.Expression],
          einsum: FusedEinsum,
          free_indices: Sequence[str],
          dtypes: Mapping[str, Any],
          shapes: Mapping[str, Tuple[ScalarT, ...]]
          ) -> Match:
    if len(exprs) != einsum.noutputs:
        raise ValueError("The number of outputs do not match.")

    if len(free_indices) != einsum.ndim:
        raise ValueError(f"Expected {einsum.ndim} free indices,"
                         f" got {len(free_indices)}.")

    redn_collector = ReductionCollector()
    var_to_dtype = {var: np.dtype(dtype)
                    for var, dtype in dtypes.items()}
    is_redn_surrounded_by_predicate = IsReductionSurroundedByIf(
        free_indices=frozenset(free_indices))

    for iexpr, (expr, use_row) in enumerate(zip(exprs, einsum.use_matrix)):
        template_expr = template_einsum_as_str(einsum, iexpr)
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
            raise ValueError(
                f"Cannot infer operands the '{expr}' as '{template_expr}'"
                " since the reduction is not over a product expression."
            )

        if isinstance(expr_to_match, p.Product):
            from pymbolic.primitives import flattened_product
            einsum_terms: Tuple[p.Expression] = (flattened_product(expr_to_match
                                                                   .children)
                                                 .children)
        else:
            einsum_terms = (expr_to_match,)

        if len(einsum_terms) < len(einsum.access_descriptors):
            raise ValueError("Multiplicative terms in reduction of"
                             f"{expr} are not enough for {template_expr}"
                             " => matching unsuccessful.")

        extracted_uses = tuple(
            UseExtractor(
                frozenset(free_indices) | set(redn_in_expr.inames),
                Map(var_to_dtype))(operand_expr)
            for operand_expr in einsum_terms)

        # TODO:Now just run some tests for th

        print(extracted_uses)
        1/0

# vim:fdm=marker
