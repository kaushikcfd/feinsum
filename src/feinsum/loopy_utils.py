"""
:mod:`loopy` helpers. The aim is to keep this module as lean as possible by
upstreaming the heavy lifting parts to :mod:`loopy` itself.

.. autofunction:: extract_subexpr_of_associative_op_as_subst
.. autofunction:: match_t_unit_to_einsum
.. autofunction:: match_einsum
"""

import numpy as np
import loopy as lp
import pymbolic.interop.matchpy as m
import pymbolic.primitives as p

from multiset import Multiset
from typing import (Union, ClassVar, Optional, Tuple, FrozenSet, Any, Dict,
                    List, Iterable, Set, Mapping)
from immutables import Map
from dataclasses import dataclass
from pymbolic.interop.matchpy.tofrom import (
    ToMatchpyExpressionMapper as BaseToMatchpyExpressionMapper,
    FromMatchpyExpressionMapper as BaseFromMatchpyExpressionMapper)
from matchpy import Arity
from feinsum.einsum import (FusedEinsum, FreeAxis, SizeParam, EinsumAxisAccess,
                            SummationAxis)
from feinsum.diagnostics import EinsumTunitMatchError
from loopy.symbolic import (pw_aff_to_expr, IdentityMapper as BaseIdentityMapper,
                            CombineMapper, Reduction)
from more_itertools import partition

PYMBOLIC_ASSOC_OPS = (p.Product, p.Sum, p.BitwiseOr, p.BitwiseXor,
                      p.BitwiseAnd, p.LogicalAnd, p.LogicalOr)


# {{{ loopy <-> matchpy interop

# type-ignore reason: cannot subclass from PymbolicOp (inferred to be of type Any)
@m.op_dataclass
class MatchableReductionOp(m.PymbolicOp):  # type: ignore[misc]
    inner_expr: m.ExprT
    operation: m.Id
    inames: m.TupleOp
    variable_name: Optional[str] = m.non_operand_field(default=None)

    arity: ClassVar[Arity] = Arity.ternary
    _mapper_method: ClassVar[str] = "map_reduction"


# type-ignore reason: cannot subclass from
# BaseToMatchpyExpressionMapper (inferred to be of type Any)
class ToMatchpyExpressionMapper(BaseToMatchpyExpressionMapper):  # type: ignore[misc]
    def map_reduction(self, expr: lp.Reduction) -> MatchableReductionOp:
        assert str(expr.operation) in ["sum", "product", "any", "all"]
        # pylint: disable=too-many-function-args
        return MatchableReductionOp(self.rec(expr.expr),
                                    m.Id(str(expr.operation)),
                                    m.TupleOp(tuple(m.Id(iname)
                                                    for iname in expr.inames))
                                    )


# type-ignore reason: cannot subclass from
# BaseFromMatchpyExpressionMapper (inferred to be of type Any)
class FromMatchpyExpressionMapper(
        BaseFromMatchpyExpressionMapper):  # type: ignore[misc]
    def map_reduction(self, expr: MatchableReductionOp) -> lp.Reduction:
        return lp.Reduction(expr.operation.value,
                            inames=tuple(iname.value
                                         for iname in expr.inames._operands),
                            expr=self.rec(expr.inner_expr),
                            )

# }}}


@dataclass
class TemplateReplacer:
    rule_lhs: p.Expression
    rule_rhs: p.Expression
    op_extra_arg_wildcard_name: str

    def __call__(self, **kwargs: Multiset) -> p.Expression:
        assert len(kwargs) == 1
        assert isinstance(self.rule_rhs, PYMBOLIC_ASSOC_OPS)
        star_wildcard_operands = kwargs[self.op_extra_arg_wildcard_name]
        leftover_args = []
        for subexpr, count in star_wildcard_operands.items():
            leftover_args.extend([subexpr] * count)

        return type(self.rule_rhs)(  # pylint: disable=abstract-class-instantiated
            (self.rule_lhs, *leftover_args))


# {{{ extract_subst

def extract_subexpr_of_associative_op_as_subst(
        kernel: lp.LoopKernel,
        rule_lhs: Union[p.Call, p.Variable, str],
        rule_rhs: Union[p.Expression, str],
        insn_match: Any = None,
) -> lp.LoopKernel:
    from loopy.symbolic import parse, get_dependencies
    from loopy.match import parse_match

    insn_match = parse_match(insn_match)
    vng = kernel.get_var_name_generator()
    to_matchpy_expr = ToMatchpyExpressionMapper()
    from_matchpy_expr = FromMatchpyExpressionMapper()

    # {{{ sanity checks

    if not isinstance(kernel, lp.LoopKernel):
        raise TypeError("`kernel` expected to be a `LoopKernel`, "
                        f"got '{type(kernel)}'")

    if isinstance(rule_lhs, str):
        rule_lhs = parse(rule_lhs)

    if isinstance(rule_rhs, str):
        rule_rhs = parse(rule_rhs)

    if isinstance(rule_lhs, p.Variable):
        rule_lhs = p.Call(rule_lhs, parameters=())

    if not isinstance(rule_lhs, p.Call):
        raise ValueError("rule_lhs must be either a Call or a Variable,"
                         f" got '{rule_lhs}'")

    if any(not isinstance(param, (p.Variable, p.DotWildcard))
           for param in rule_lhs.parameters):
        raise ValueError(f"Arguments to rule_lhs ({rule_lhs.parameters})"
                         " must be variables or wildcards.")

    if not (get_dependencies(rule_rhs) <= (get_dependencies(rule_lhs)
                                           | kernel.all_variable_names())):
        # FIXME: consider mangled symbols as well.
        raise ValueError(f"rule_rhs ({rule_rhs}) contains variables"
                         " not defined in the kernel's namespace.")

    if rule_lhs.function.name in kernel.substitutions:
        raise ValueError(f"Kernel {kernel.name} already has a substitution"
                         f" named '{rule_lhs.name}'")

    # }}}

    # Substitutions with wildcards: tricky, handle them when needed.
    if any(isinstance(param, p.DotWildcard) for param in rule_lhs.parameters):
        raise NotImplementedError("substitutions with wildcards not "
                                  "yet implemented")

    if not isinstance(rule_rhs, PYMBOLIC_ASSOC_OPS):
        raise TypeError(f"rule_rhs ({rule_rhs}) does not represent an"
                        " associative op.")

    extra_wildcard_name = vng("_lpy_w_star")

    replacement_rule = m.make_replacement_rule(
        pattern=type(rule_rhs)(rule_rhs.children +
                               (p.StarWildcard(extra_wildcard_name),)),
        replacement=TemplateReplacer(rule_lhs, rule_rhs, extra_wildcard_name),
        to_matchpy_expr=to_matchpy_expr,
        from_matchpy_expr=from_matchpy_expr,
    )

    new_insns = [
        insn.with_transformed_expressions(
            lambda expr: m.replace_all(expr, [replacement_rule],
                                       to_matchpy_expr, from_matchpy_expr))
        if insn_match(kernel, insn)
        else insn
        for insn in kernel.instructions
    ]

    if new_insns == kernel.instructions:
        raise RuntimeError(f"Did not find a match for `{rule_lhs} := {rule_rhs}`"
                           " in the kernel.")
    else:
        kernel = kernel.copy(instructions=new_insns)

    new_substitutions = kernel.substitutions.copy()
    new_substitutions[rule_lhs.function.name] = lp.SubstitutionRule(
        rule_lhs.function.name,
        tuple(param.name for param in rule_lhs.parameters),
        rule_rhs)

    kernel = kernel.copy(substitutions=new_substitutions)
    return kernel

# }}}


# {{{ matching loopy kernel against a ref. einsum

# {{{ mapper classes

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


class ContainsIf(CombineMapper):
    """
    Mapper that returns *True* only if the expression contains a conditional
    expression.
    """

    def combine(self, values: Iterable[bool]):
        return any(values)

    def map_algebraic_leaf(self, expr: Any) -> bool:
        return False

    def map_constant(self, expr: Any) -> bool:
        return False

    def map_if(self, expr: p.If) -> bool:
        return True

# }}}


def _get_indices_from_assignee(assignee: p.Expression) -> Tuple[str, ...]:
    if isinstance(assignee, p.Variable):
        return ()
    elif (isinstance(assignee, p.Subscript)
              and all(isinstance(idx, p.Variable)
                      for idx in assignee.index_tuple)):
        return tuple(idx.name for idx in assignee.index_tuple)
    else:
        raise EinsumTunitMatchError("Not all instructions assign to Variable or"
                                    " a Subscript with free indices --"
                                    " not allowed.")


def _check_if_t_unit_and_ref_einsum_have_the_same_axis_dim(ref_idx: EinsumAxisAccess,
                                                           physical_idx: str,
                                                           ref_einsum: FusedEinsum,
                                                           kernel: lp.LoopKernel,
                                                           long_dim_length: int,
                                                           ) -> None:

    idx_bounds = kernel.get_iname_bounds(physical_idx)
    if not idx_bounds.lower_bound_pw_aff.non_zero_set().is_empty():
        # TODO: Not sure whether we can also support lower bound with
        # offset
        raise NotImplementedError("Non-zero lower bounds not yet"
                                  " supported.")
    if not idx_bounds.upper_bound_pw_aff.is_cst():
        if not isinstance(ref_einsum.index_to_dim_length()[ref_idx], SizeParam):
            raise EinsumTunitMatchError(f"Index '{physical_idx}' in the"
                                        " t-unit is of parametric length,"
                                        " while not in the reference"
                                        " einsum.")
    else:
        knl_axis_size = pw_aff_to_expr(idx_bounds.upper_bound_pw_aff) + 1
        if (knl_axis_size == ref_einsum.index_to_dim_length()[ref_idx]
                or ((knl_axis_size > long_dim_length)
                    and isinstance(ref_einsum.index_to_dim_length()[ref_idx],
                                   SizeParam))):
            pass
        else:
            raise EinsumTunitMatchError(f"Index '{physical_idx}' in the"
                                        " t-unit does not have the same"
                                        " dimensionality as in einsum.")


def _get_iname_len(kernel, iname, long_dim_length) -> Union[np.floating, np.integer]:
    """
    Returns the iname bounds
    """
    from loopy.isl_helpers import static_min_of_pw_aff, static_max_of_pw_aff
    # TODO: Get the constant iname length.
    bounds = kernel.get_iname_bounds(iname)
    lbound_pwaff = bounds.lower_bound_pw_aff
    static_min_lbound_pwaff = static_min_of_pw_aff(lbound_pwaff,
                                                   constants_only=True)
    if static_min_lbound_pwaff.get_constant_val().to_python() != 0:
        raise EinsumTunitMatchError(f"Iname {iname} that appears as an einsum index"
                                    f" in {kernel.name} does not have '0' as its"
                                    " lower bound -> not allowed in feinsum"
                                    " grammar.")

    ubound_pw_aff = bounds.upper_bound_pw_aff

    ubound = static_max_of_pw_aff(ubound_pw_aff, constants_only=False)

    if ubound.is_cst():
        ubound_val = ubound.get_pieces()[0][1].get_constant_val().to_python()
        if ubound_val >= long_dim_length:
            return np.inf
        else:
            return ubound_val
    else:
        return np.inf


def _get_dtype_for_subst_argument(t_unit: lp.LoopKernel,
                                  kernel_name: str,
                                  rule_name: str) -> np.dtype[Any]:
    from loopy.type_inference import TypeReader
    from pymbolic.mapper.substitutor import make_subst_func
    from loopy.symbolic import (SubstitutionMapper,
                                SubstitutionRuleExpander)
    knl = t_unit.default_entrypoint
    # TODO: The loopy.TypeReader is flaky -> try to break it and fix it.
    type_reader = TypeReader(knl, t_unit.callables_table)
    subst = knl.substitutions[rule_name]
    subst_expr = subst.expression
    submap1 = SubstitutionRuleExpander(knl.substitutions)
    submap2 = SubstitutionMapper(
        make_subst_func({arg: np.iinfo(knl.index_dtype.numpy_dtype).max
                         for arg in subst.arguments})
    )

    subst_dtype = type_reader(submap2(submap1(subst_expr)))
    return subst_dtype.numpy_dtype


class SubstitutionInvocationGetter(CombineMapper):
    def __init__(self, argument_substs: FrozenSet[str]):
        self.argument_substs = argument_substs
        super().__init__()

    def combine(self, values: Iterable[FrozenSet[p.Call]]):
        from functools import reduce
        return reduce(frozenset.union, values, frozenset())

    def map_call(self, expr: p.Call) -> FrozenSet[p.Call]:
        if expr.function.name in self.argument_substs:
            return frozenset([expr])
        else:
            return super().map_call(expr)

    def map_algebraic_leaf(self, expr: Any) -> FrozenSet[p.Call]:
        return frozenset()

    def map_constant(self, expr: Any) -> FrozenSet[p.Call]:
        return frozenset()


def get_matched_einsum(t_unit: lp.TranslationUnit,
                       kernel_name: Optional[str] = None,
                       insn_match: Any = None,
                       argument_substitutions: Optional[FrozenSet[str]] = None,
                       long_dim_length: int = 500,
                       ) -> Tuple[FusedEinsum, Map[str, str]]:
    """
    Returns a tuple of the form ``(matched_einsum, subst_map)`` where,
    ``matched_einsum`` is the batched einsum having a memory access pattern similar
    to the instructions in *insn_match* of *t_unit*, and, ``subst_map`` is a mapping
    from the entities (i.e. indices, arguments) of *match_einsum* to the variables
    in *t_unit*.

    :param t_unit: The subject translation unit which is being matched against
        *ref_einsum*.
    :param insn_match: A match expression as understood by
        :func:`loopy.match.parse_match` representing the subset of the
        kernel's instructions raised to a batched einsum.
    :param argument_substitutions: A :class:`frozenset` of substitutions that are to
        be matched as an Einstein-summation's argument. If *None*, any substitution
        rule invocation is treated as accessing an Einstein Summation's argument.
    :param long_dim_length: Axis length above which can be assumed to be long
        enough to be invariant of having any noticeable effects on the kernel's
        performance.

    .. note::

         See xxx (TODO) for a description of a translation unit with a einsum-like
         memory access pattern.
    """
    if kernel_name is None:
        if len(t_unit.entrypoints) != 1:
            raise EinsumTunitMatchError("Must provide `kernel_name`, when"
                                        " the translation unit has more than"
                                        " 1 entrypoints.")
        else:
            kernel_name = t_unit.default_entrypoint.name

    assert isinstance(kernel_name, str)
    kernel = t_unit[kernel_name]

    argument_substitutions: FrozenSet[str] = (argument_substitutions
                                              or frozenset(kernel.substitutions))

    from loopy.match import parse_match
    insn_match = parse_match(insn_match)
    insns = [insn
             for insn in kernel.instructions
             if insn_match(kernel, insn)]

    if len({insn.within_inames for insn in insns}) != 1:
        raise EinsumTunitMatchError("Instructions forming the subject have"
                                    " more than 1 enclosing loop nest -- not"
                                    " allowed.")

    free_indices_set = {_get_indices_from_assignee(insn.assignees[0])
                        for insn in insns}
    if len(free_indices_set) != 1:
        raise EinsumTunitMatchError("Instructions have differing free indices."
                                    " This is not allowed as the expressions"
                                    " are not batched-einsums-like.")

    free_indices, = free_indices_set

    if frozenset(free_indices) != insns[0].within_inames:
        raise EinsumTunitMatchError("Einsum nested within an outer loop."
                                    " Such manipulation of batched einsums"
                                    " is not supported by feinsum.")

    get_reductions = ReductionCollector()
    batched_access_to_operands: List[Mapping[Tuple[str, ...], FrozenSet[str]]] = []
    subst_invokes_getter = SubstitutionInvocationGetter(argument_substitutions)

    for insn in insns:
        access_to_operands: Dict[Tuple[str, ...], Set[str]] = {}
        redns_in_expr = get_reductions((insn.expression, tuple(insn.predicates)))
        if len(redns_in_expr) == 0:
            inner_expr = insn.expr
            ensm_inames = insn.within_inames
            redn_inames = frozenset()
        elif len(redns_in_expr) == 1:
            redn_in_expr, = redns_in_expr
            inner_expr = redn_in_expr.expr
            redn_inames = frozenset(redn_in_expr.inames)
            ensm_inames = insn.within_inames | redn_inames
        else:
            raise EinsumTunitMatchError("More than one reductions found"
                                        " -> not within the grammar of expressions"
                                        " that can be matched by feinsum.")

        if isinstance(inner_expr, p.Product):
            from pymbolic.primitives import flattened_product
            einsum_terms: Tuple[p.Expression, ...] = (flattened_product(inner_expr
                                                                        .children)
                                                      .children)
        else:
            einsum_terms = (inner_expr,)

        for term in einsum_terms:
            subst_invokes = subst_invokes_getter(term)
            if not subst_invokes:
                continue
            assert all(isinstance(subst_invoke, p.Call)
                       for subst_invoke in subst_invokes)
            if not all(all(isinstance(arg, p.Variable) and arg.name in ensm_inames
                           for arg in subst_invoke.parameters)
                       for subst_invoke in subst_invokes):
                raise EinsumTunitMatchError("Invocation to a substitution in "
                                            f" '{term}' not called with the einstein"
                                            " summation's indices -> not a part of"
                                            " feinsum's grammar.")

            access_index_tuple_var: Tuple[p.Variable, ...] = next(
                iter(subst_invokes)).parameters

            if any(subst_invoke.parameters != access_index_tuple_var
                   for subst_invoke in subst_invokes):
                raise EinsumTunitMatchError("Not all substitution invocations in"
                                            f" '{term}' are called with the same"
                                            " arguments -> violates the feinsum's ."
                                            " grammar.")

            access_index_tuple: Tuple[str, ...] = tuple(
                idx.name for idx in access_index_tuple_var)
            subst_names = {subst_invoke.function.name
                           for subst_invoke in subst_invokes}
            (access_to_operands
             .setdefault(access_index_tuple, set())
             .update(subst_names))

        batched_access_to_operands.append(access_to_operands)

    # TODO: We need to do something about the einsum index dependencies on the
    # expression itself. It gets trickier if it comes as a multiplicative term and
    # we intend to hoist certain terms.

    # Step 1. Get a mapping from einsum inames to indices.
    ensm_inames = insns[0].within_inames | insn.reduction_inames()
    einsum_indices_generator = (chr(i) for i in range(ord("a"), ord("z") + 1))
    ensm_iname_to_index: Dict[str, str] = {iname: next(einsum_indices_generator)
                                           for iname in ensm_inames}

    # Step 2. Get the $\mathcal{L}(i)$ for each $i$ in the new indices.
    iname_to_len = {iname: _get_iname_len(kernel, iname, long_dim_length)
                    for iname in ensm_inames}

    # Step 3. Combine these accesses to get the einsum expression.
    from functools import reduce
    unioned_accesses: Tuple[Tuple[str, ...]] = tuple(
        reduce(frozenset.union,
               (frozenset(access_to_operands)
                for access_operands in batched_access_to_operands),
               frozenset())
    )
    einsum_subscripts = (",".join(["".join(ensm_iname_to_index[iname]
                                           for iname in accesses)
                                   for accesses in unioned_accesses])
                         + "->"
                         + "".join(ensm_iname_to_index[iname]
                                   for iname in free_indices))
    use_matrix: List[List[FrozenSet[str]]] = []
    for access_to_operands in batched_access_to_operands:
        use_row = [frozenset(access_to_operands.get(accesses, set()))
                   for accesses in unioned_accesses]
        use_matrix.append(use_row)

    # Step 4. Get the numeric data type for the substitution
    value_to_dtype: Dict[str, np.dtype[Any]] = {}

    for use_row in use_matrix:
        for uses in use_row:
            for use in uses:
                if use not in value_to_dtype:
                    value_to_dtype[use] = _get_dtype_for_subst_argument(
                        t_unit,
                        kernel_name,
                        use)

    # Step 5. Get arg_shapes from $L$'s.
    arg_shapes: List[Tuple[Union[np.floating, np.integer], ...]] = []
    for accesses in unioned_accesses:
        arg_shapes.append([iname_to_len[iname]
                           for iname in accesses])

    # Step 6. Construct the batched einsum.
    from feinsum.make_einsum import fused_einsum
    batched_einsum = fused_einsum(einsum_subscripts,
                                  arg_shapes,
                                  use_matrix=use_matrix,
                                  value_to_dtype=value_to_dtype)

    # FIXME: Verify that the kernel's domain is indeed a dense-hypercube.

    subst_map = Map(ensm_iname_to_index)
    return batched_einsum, subst_map


def extract_einsum_terms_as_subst(t_unit: lp.TranslationUnit,
                                  rule_lhs: Union[str, p.Expression],
                                  rule_rhs: Union[str, p.Expression],
                                  insn_match: Any = None,
                                  ) -> lp.TranslationUnit:
    from loopy.symbolic import parse
    knl = t_unit.default_entrypoint

    if isinstance(rule_rhs, str):
        rule_rhs = parse(rule_rhs)

    if not isinstance(rule_rhs, p.Product):
        rule_rhs = p.Product((rule_rhs,))

    return t_unit.with_kernel(extract_subexpr_of_associative_op_as_subst(knl,
                                                                         rule_lhs,
                                                                         rule_rhs,
                                                                         insn_match
                                                                         ))
# }}}


# {{{ pull_out_subproduct

# type-ignore-reason:  cannot subclass from BaseIdentityMapper (inferred as type Any)
class EinsumTermsHoister(BaseIdentityMapper):  # type: ignore[misc]
    """
    Mapper to hoist products out of a sum-reduction.
    """
    def __init__(self, reduction_inames: FrozenSet[str]):
        super().__init__()
        self.reduction_inames = reduction_inames

    def map_reduction(self, expr: lp.Reduction) -> p.Expression:
        if frozenset(expr.inames) != self.reduction_inames:
            return super().map_reduction(expr)

        from loopy.library.reduction import SumReductionOperation
        from loopy.symbolic import get_dependencies
        if isinstance(expr.expr, p.Product) and isinstance(expr.operation,
                                                           SumReductionOperation):
            from pymbolic.primitives import flattened_product
            inner_expr = flattened_product(self.rec(expr.expr).children)
            assert isinstance(inner_expr, p.Product)
            invariants, variants = partition(lambda x: (get_dependencies(x)
                                                        & self.reduction_inames),
                                             inner_expr.children)

            return p.Product(tuple(invariants)) * lp.Reduction(
                expr.operation,
                inames=expr.inames,
                expr=p.Product(tuple(variants)),
                allow_simultaneous=expr.allow_simultaneous)
        else:
            raise NotImplementedError(expr.expr)


def hoist_reduction_invariant_terms(t_unit: lp.TranslationUnit,
                                    reduction_inames: Union[str, FrozenSet[str]],
                                    ) -> lp.TranslationUnit:
    """
    Hoists loop-invariant terms in a sum reduction expression with a product
    inner expression.

    .. note::

        Placeholder until
        `Loopy-541 <https://github.com/inducer/loopy/issues/541>`_
        is fixed.
    """
    if isinstance(reduction_inames, str):
        reduction_inames = frozenset([reduction_inames])

    if not (reduction_inames <= t_unit.default_entrypoint.all_inames()):
        raise ValueError(f"Some inames in '{reduction_inames}' not a part of"
                         " the kernel")

    term_hoister = EinsumTermsHoister(reduction_inames)

    return t_unit.with_kernel(
        lp.map_instructions(t_unit.default_entrypoint,
                            insn_match=None,
                            f=lambda x: x.with_transformed_expressions(term_hoister)
                            ))

# }}}


# {{{ infer einsum

def match_einsum(t_unit: lp.TranslationUnit,
                 insn_match: Any = None,
                 kernel_name: Optional[str] = None,
                 long_dim_length: int = 1000,
                 ) -> FusedEinsum:
    """
    Returns the inferred :class:`~feinsum.einsum.FusedEinsum` for the
    instructions spanning *insn_match*.

    :param insn_match: A match expression as understood as
        :func:`loopy.match.parse_match`.
    """
    from loopy.match import parse_match
    from pymbolic.mapper.flattener import flatten
    from feinsum.make_einsum import fused_einsum

    if kernel_name is None:
        kernel_name = t_unit.default_entrypoint.name

    insn_match = parse_match(insn_match)
    kernel = t_unit[kernel_name]
    insns = [insn
             for insn in kernel.instructions
             if insn_match(kernel, insn)]

    if len({insn.within_inames for insn in insns}) != 1:
        raise EinsumTunitMatchError("Instructions forming the subject have"
                                    " more than 1 enclosing loop nest -- not"
                                    " allowed.")

    free_indices_set = {_get_indices_from_assignee(insn.assignees[0])
                        for insn in insns}
    if len(free_indices_set) != 1:
        raise EinsumTunitMatchError("Instructions have differing free indices"
                                    " -- not allowed.")
    free_indices, = free_indices_set

    for insn in insns:
        if not isinstance(insn.expression, lp.Reduction):
            raise ValueError(f"Instruction {insn} does not have a"
                             " reduction RHS.")
        if not isinstance(insn.expression.expr, p.Product):
            raise ValueError(f"Instruction {insn} does not have"
                             " reduction of a product")

        if insn.expression.inames != insns[0].expression.inames:
            raise ValueError("Instructions don't have the same reduction"
                            " inames, not supported.")

    redn_indices = insns[0].expression.inames

    iname_to_access_descr: Dict[str, EinsumAxisAccess] = {}

    for i, iname in enumerate(free_indices):
        iname_to_access_descr[iname] = FreeAxis(i)
    for i, iname in enumerate(redn_indices):
        iname_to_access_descr[iname] = SummationAxis(i)

    access_descrs_for_insns = {
        tuple(
            tuple(iname_to_access_descr[idx.name]
                  for idx in child.index_tuple)
            for child in flatten(insn.expression.expr).children)
        for insn in insns}
    if len(access_descrs_for_insns) != 1:
        raise ValueError("access pattern across instructions not uniform")

    access_descrs, = access_descrs_for_insns

    use_matrix = tuple(
        tuple(frozenset([child.aggregate.name])
              for child in flatten(insn.expression.expr).children)
        for insn in insns
    )
    all_values = {child.aggregate.name
                  for insn in insns
                  for child in flatten(insn.expression.expr).children}

    value_to_dtype = Map(
        {val: kernel.arg_dict.get(val, kernel.temporary_variables.get(val)).dtype
         for val in all_values}
    )

    arg_shape_of_all_insns = {
        tuple(
            tuple(dim
                  if (isinstance(dim, int) and (dim < long_dim_length))
                  else np.inf
                  for dim in kernel.get_var_descriptor(child.aggregate.name).shape)
            for child in flatten(insn.expression.expr).children)
        for insn in insns
    }

    if len(arg_shape_of_all_insns) != 1:
        raise ValueError("Arguments from instructions have different"
                         " shapes: not allowed in a fused einsum.")

    arg_shapes, = arg_shape_of_all_insns

    index_names = {}
    sorted_axes: List[EinsumAxisAccess] = ([FreeAxis(i)
                                            for i in range(len(free_indices))]
                                           + [SummationAxis(i)
                                              for i in range(len(redn_indices))])
    for idx, ichr in zip(sorted_axes, range(97, 123)):
        index_names[idx] = chr(ichr)

    subscripts = (",".join("".join(index_names[axis]
                                   for axis in axes)
                           for axes in access_descrs)
                  + "->"
                  + "".join(index_names[FreeAxis(i)]
                            for i in range(len(free_indices))))

    return fused_einsum(subscripts,
                        arg_shapes,
                        use_matrix,
                        value_to_dtype=value_to_dtype)

# }}}


def match_t_unit_to_einsum(*args, **kwargs):
    raise NotImplementedError

# vim: foldmethod=marker
