import abc
import numpy as np
import pymbolic.primitives as p
import loopy as lp

from loopy.symbolic import CombineMapper, Reduction
from typing import (Any, FrozenSet, Iterable, Mapping, Tuple, Set,
                    Dict, Optional, List)
from feinsum.einsum import FusedEinsum, SummationAxis, EinsumAxisAccess
from more_itertools import zip_equal as zip
from functools import cached_property
from dataclasses import dataclass
from pytools import memoize_on_first_arg
from immutables import Map
from loopy.types import LoopyType


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


# {{{ type of expressions being matched to a multi-dimensional array.

class MatchedArray(abc.ABC):
    """
    A global memory access per iteration point of an einsum. An iteration point
    is a point in the iteration domain of the :class:`feinsum.FusedEinsum`,
    where the iteration domain is the cross product of the domain spanned by
    the free indices and the reduction indices of an einsum.
    """


@dataclass(frozen=True)
class PlainOldSubscript(MatchedArray):
    """
    An unconditional global memory access per iteration point of the domain.
    """
    var_name: str
    indices: Tuple[p.Expression, ...]


@dataclass(frozen=True)
class ConditionalArray(MatchedArray):
    """
    A conditional global memory access per iteration point of the einsum's
    domain. The condition depends on dimensions spanning the iteration domain.

    .. note::

        - Since a :class:`MatchedArray` is associated with a fixed size global memory
          access the data type of both branches of a conditional use must be
          the same.
    """
    cond: p.Expression
    then: MatchedArray
    else_: MatchedArray

# }}}


class ConditionalArrayDtypeMismatchError(ValueError):
    """
    Raised by :func:`_get_use_dtype` if it encounters a :class:`ConditionalArray`
    which uses with different dtypes in both of its branches.
    """


@memoize_on_first_arg
def _get_use_dtype(use: MatchedArray, var_name_to_dtype: Map[str, np.dtype[Any]]):
    """
    Returns the data type of *use*. By definition, *use* must be associated
    with a single dtype.
    """
    if isinstance(use, PlainOldSubscript):
        return var_name_to_dtype[use.var_name]
    elif isinstance(use, ConditionalArray):
        then_dtype = _get_use_dtype(use.then, var_name_to_dtype)
        else_dtype = _get_use_dtype(use.else_, var_name_to_dtype)
        assert then_dtype == else_dtype
        return then_dtype
    else:
        raise NotImplementedError(type(use))


class UseExtractor(CombineMapper):
    """
    Collects all instances of :class:`MatchedArray` in an expression.

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

    def combine(self, values: FrozenSet[MatchedArray]) -> FrozenSet[MatchedArray]:
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    @cached_property
    def _dep_mapper(self):
        from loopy.symbolic import DependencyMapper
        return DependencyMapper()

    def get_dependencies(self, expr: p.Expression) -> FrozenSet[str]:
        return frozenset({x.name
                          for x in self._dep_mapper(expr)})

    def map_subscript(self, expr: p.Subscript) -> FrozenSet[MatchedArray]:
        if not isinstance(expr.aggregate, p.Variable):
            # TODO: Is this a case we even need to worry about?
            raise NotImplementedError
        if any(bool(self.get_dependencies(idx) & self.inames)
               for idx in expr.index_tuple):
            return frozenset([PlainOldSubscript(
                expr.aggregate.name, expr.index_tuple)])
        else:
            return frozenset()

    def map_if(self, expr: p.If) -> FrozenSet[MatchedArray]:
        if self.rec(expr.condition):
            # the condition expression contains a use.
            # This is the case of a weak use?
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

                dtype_to_then_uses: Dict[np.dtype[Any], Set[MatchedArray]] = {}
                dtype_to_else_uses: Dict[np.dtype[Any], Set[MatchedArray]] = {}

                for then_use, else_use in zip(then_uses, else_uses):
                    dtype_to_then_uses.setdefault(_get_use_dtype(
                                                    then_use,
                                                    self.var_to_dtype),
                                                  set()).add(then_use)
                    dtype_to_else_uses.setdefault(_get_use_dtype(
                                                    else_use,
                                                    self.var_to_dtype),
                                                  set()).add(else_use)

                if (set(dtype_to_then_uses) != set(dtype_to_else_uses)
                        or any((len(dtype_to_then_uses[dtype])
                                != len(dtype_to_else_uses[dtype]))
                               for dtype in dtype_to_then_uses)):
                    raise ValueError(f"Branches of '{expr}' have different number "
                                     "of uses => disallowed as predicting its "
                                     "performance is quite challenging.")
                else:
                    new_uses = set()
                    for dtype in dtype_to_then_uses:
                        new_uses.update({ConditionalArray(expr.condition,
                                                        then_use,
                                                        else_use)
                                         for then_use, else_use in zip(
                                             dtype_to_then_uses[dtype],
                                             dtype_to_else_uses[dtype])})

                    return frozenset(new_uses)
            else:
                # static condition. (not yet implemented)
                raise NotImplementedError()

    def map_constant(self, expr: Any) -> FrozenSet[MatchedArray]:
        return frozenset()

    def map_algebraic_leaf(self, expr: Any) -> FrozenSet[MatchedArray]:
        return frozenset()


@dataclass(frozen=True)
class MatchResult:
    """
    Records the result of :func:`match`.

    .. attribute:: index_mapping

        A mapping from the index name in the reference einsum to the

    .. attribute:: array_mapping

        A mapping from the array name in the reference einsum to the
        substitution name in the kernel.
    """
    index_mapping: Mapping[str, str]
    array_mapping: Mapping[str, str]


def match(t_unit: lp.TranslationUnit,
          *,
          kernel_name: Optional[str] = None,
          insn_match: Any = None,
          long_dim_threshold: int = 500
          ) -> Tuple[lp.TranslationUnit, FusedEinsum, MatchResult]:
    r"""
    Returns ``(transformed_t_unit, einsum, mapping)``, where:

    - ``transformed_t_unit`` is a copy of *t_unit* with all reads to arrays in
      *t_unit*\ s expressions spanning *insn_match* are replaced by
      substitutions.
    - ``einsum`` is an instance of :class:`feinsum.FusedEinsum` that can be
      interpreted from the *t_unit*\ 's statements within *insn_match*.
    - ``mapping`` is the substitution mapping as an instance of
      :class:`MatchResult` between the variables of ``transformed_t_unit`` and
      ``einsum``.

    :param long_dim_threshold: Axis length greater than or equal to this value
        is interpreted to be of :math:`\infty`-length. Defaults to 500.
    """

    from loopy.match import parse_match

    within = parse_match(insn_match)

    if kernel_name is None:
        if len(t_unit.entrypoints) != 1:
            raise ValueError("`kernel_name` is required when "
                             " translation unit has multiple entrypoints.")
        else:
            kernel_name = t_unit.default_entrypoint.name

    assert isinstance(kernel_name, str)

    knl = t_unit[kernel_name]

    insns = [insn
             for insn in knl.instructions
             if within(knl, insn)]

    if not all(len(insn.assignees) == 1
               for insn in insns):
        raise ValueError("Each instruction being matched for an einsum"
                         " must have a single assignee.")

    index_tuples: Set[p.Expression] = set()
    for insn in insns:
        assignee = insn.assignee
        if isinstance(assignee, p.Variable):
            index_tuples.add(())
        elif isinstance(assignee, p.Subscript):
            index_tuples.add(assignee.index_tuple)
        else:
            raise ValueError("Assignee can be either Subscript or Variable,"
                             f" got {type(assignee)}.")

    if len(index_tuples) != 1:
        raise ValueError("All instructions must write to expressions"
                         " with same indexing expression.")

    if len({insn.assignee_var_names for insn in insns}) != len(insns):
        raise ValueError("All the instructions being matched over must"
                         " write to a different output.")

    if len({insn.reduction_inames() for insn in insns}) != 1:
        raise ValueError("All the instructions being matched over must have the"
                         " same reduction inames.")

    index_tuple, = index_tuples
    free_inames: Tuple[str, ...] = tuple(idx.name for idx in index_tuple)

    # TODO: Check the strides for the assignees here.

    redn_collector = ReductionCollector()
    is_redn_surrounded_by_predicate = IsReductionSurroundedByIf(
        free_indices=frozenset(free_inames))

    arguments_rows: List[List[Set[str]]] = []
    matched_ary_to_argument_name: Dict[MatchedArray, str] = {}

    for insn in insns:
        arguments_row = []
        redns_in_expr = redn_collector(insn.expression)
        var_to_dtype: Dict[str, np.dtype[Any]] = {}

        for var in (insn.read_dependency_names()
                    & (set(knl.temporary_variables.keys())
                       | set(knl.arg_dict.keys()))):
            dtype = knl.get_var_descriptor(var).dtype
            if isinstance(dtype, LoopyType):
                var_to_dtype[var] = dtype.numpy_dtype
            else:
                raise ValueError(f"Got uninferred dtype for '{var}'"
                                 " -> not allowed.")

        if not redns_in_expr:
            # FIXME: Not an interesting case right now.
            raise NotImplementedError()

        if len(redns_in_expr) > 1:
            raise ValueError("An expression with multiple reduction nodes"
                             " cannot be inferred as an einsum-like"
                             " expression.")

        if is_redn_surrounded_by_predicate(insn.expression, nested_in_ifelse=False):
            raise ValueError(f"Reduction in '{insn}' is nested"
                             " within an if-else arm => not allowed.")

        if redns_in_expr:
            redn_in_expr, = redns_in_expr
            expr_to_match = redn_in_expr.expr
            redn_inames: Tuple[str, ...] = redn_in_expr.inames
            del redn_in_expr
        else:
            expr_to_match = insn.expression
            redn_inames = ()

        if isinstance(expr_to_match, p.Product):
            from pymbolic.primitives import flattened_product
            einsum_terms: Tuple[p.Expression, ...] = (flattened_product(expr_to_match
                                                                        .children)
                                                      .children)
        else:
            einsum_terms = (expr_to_match,)

        use_extractor = UseExtractor(frozenset(free_inames) | frozenset(redn_inames),
                                     Map(var_to_dtype))
        for term in einsum_terms:
            arguments = set()
            for matched_ary in use_extractor(term):
                if matched_ary not in matched_ary_to_argument_name:
                    matched_ary_to_argument_name[matched_ary] = (
                        f"arg_{len(matched_ary_to_argument_name)}")

                arguments.add(matched_ary_to_argument_name[matched_ary])

            arguments_row.append(arguments)

        arguments_rows.append(arguments_row)

    print(arguments_rows)

    # TODO.. Fill this ->
    # matched_ary_to_access_inames: Dict[MatchedArray, Tuple[str, ...]] = {}

# vim:fdm=marker
