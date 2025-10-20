"""
:mod:`loopy` helpers. The aim is to keep this module as lean as possible by
upstreaming the heavy lifting parts to :mod:`loopy` itself.

.. autofunction:: get_a_matched_einsum
.. autofunction:: match_t_unit_to_einsum
.. autofunction:: get_call_ids
.. autofunction:: hoist_invariant_multiplicative_terms_in_sum_reduction
.. autofunction:: extract_multiplicative_terms_in_sum_reduction_as_subst
"""

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from typing import (
    Any,
    TypeVar,
    cast,
)

import islpy as isl
import loopy as lp
import numpy as np
import pymbolic.primitives as p
from bidict import frozenbidict
from immutables import Map
from loopy.diagnostic import LoopyError
from loopy.kernel import LoopKernel
from loopy.kernel.data import SubstitutionRule
from loopy.symbolic import CombineMapper, IdentityMapper, Reduction
from loopy.translation_unit import for_each_kernel
from more_itertools import zip_equal as szip
from pytools import memoize_on_first_arg

from feinsum.diagnostics import EinsumTunitMatchError
from feinsum.einsum import (
    BatchedEinsum,
    IntegralT,
    ShapeComponentT,
    ShapeT,
    SizeParam,
)

# {{{ matching loopy kernel against a ref. einsum


class ReductionCollector(CombineMapper[frozenset[p.Expression], []]):
    """
    A mapper that collects all instances of :class:`loopy.symbolic.Reduction`
    in an expression.
    """

    def combine(
        self, values: Iterable[frozenset[p.Expression]]
    ) -> frozenset[p.Expression]:
        from functools import reduce

        return reduce(frozenset.union, values, frozenset())

    def map_reduction(self, expr: Reduction) -> frozenset[p.Expression]:
        return self.combine([frozenset([expr]), super().map_reduction(expr)])

    def map_algebraic_leaf(self, expr: Any) -> frozenset[p.Expression]:
        return frozenset()

    def map_constant(self, expr: Any) -> frozenset[p.Expression]:
        return frozenset()


def _get_indices_from_assignee(assignee: p.Expression) -> tuple[str, ...]:
    r"""
    Returns the accessed indices in assignee. Expects the assignee to be seen in a
    batched-einsum matchable :class:`~loopy.LoopKernel`\ 's expression.
    """
    if isinstance(assignee, p.Variable):
        return ()
    elif isinstance(assignee, p.Subscript) and all(
        isinstance(idx, p.Variable) for idx in assignee.index_tuple
    ):
        return tuple(cast("p.Variable", idx).name for idx in assignee.index_tuple)
    else:
        raise EinsumTunitMatchError(
            "Not all instructions assign to Variable or"
            " a Subscript with free indices --"
            " not allowed."
        )


def _get_iname_length(
    kernel: lp.LoopKernel,
    iname: str,
    long_dim_length: IntegralT,
    iname_to_index: Mapping[str, str],
) -> ShapeComponentT:
    r"""
    Returns :math:`\mathcal{L}(\text{iname})`. In order to have the same iteration
    domain as that of an einsum, we enforce that *iname* has a zero lower bound
    (inclusive). In the case when :math:`\mathcal{L}(\text{iname}) \geq
    \text{long\_dim\_length}` or when the *iname*'s domain size is parametric we
    return :data:`numpy.inf` to denote an :math:`\infty`-long dimension.
    """
    from loopy.isl_helpers import static_max_of_pw_aff, static_min_of_pw_aff

    bounds = kernel.get_iname_bounds(iname)
    lbound_pwaff = bounds.lower_bound_pw_aff
    static_min_lbound_pwaff = static_min_of_pw_aff(lbound_pwaff, constants_only=True)
    if static_min_lbound_pwaff.get_constant_val().to_python() != 0:
        raise EinsumTunitMatchError(
            f"Iname {iname} that appears as an einsum index"
            f" in {kernel.name} does not have '0' as its"
            " lower bound -> not allowed in feinsum"
            " grammar."
        )

    ubound_pw_aff = bounds.upper_bound_pw_aff
    ubound = static_max_of_pw_aff(ubound_pw_aff, constants_only=False)

    if ubound.is_cst():
        ubound_val = ubound.get_pieces()[0][1].get_constant_val().to_python()
        assert isinstance(ubound_val, int)
        if ubound_val >= long_dim_length:
            assert iname_to_index[iname].islower()
            return SizeParam(iname_to_index[iname].upper())
        else:
            return ubound_val + 1
    else:
        assert iname_to_index[iname].islower()
        return SizeParam(iname_to_index[iname].upper())


@memoize_on_first_arg
def _infer_lp_types(t_unit: lp.TranslationUnit) -> lp.TranslationUnit:
    return lp.infer_unknown_types(t_unit)


def _get_dtype_for_subst_argument(
    t_unit: lp.TranslationUnit, kernel_name: str, rule_name: str
) -> np.dtype[Any]:
    """
    Returns the inferred data type for *rule_name*'s expression when all the
    arguments passed to it are equal to it kernel's
    :attr:`loopy.LoopKernel.index_dtype`.
    """
    from loopy.symbolic import SubstitutionMapper, SubstitutionRuleExpander
    from loopy.type_inference import TypeReader
    from pymbolic.mapper.substitutor import make_subst_func

    t_unit = _infer_lp_types(t_unit)
    knl = t_unit.default_entrypoint
    type_reader = TypeReader(knl, t_unit.callables_table)  # type: ignore[no-untyped-call]
    subst = knl.substitutions[rule_name]
    subst_expr = subst.expression
    # submap1: expand substitution invocations (expected by type inference mapper)
    submap1 = SubstitutionRuleExpander(knl.substitutions)
    # submap2: pass arguments (with appropriate dtypes) to the substitution of
    # interest
    submap2 = SubstitutionMapper(
        make_subst_func(
            {
                arg: np.iinfo(knl.index_dtype.numpy_dtype).max
                for arg in subst.arguments
            }
        )
    )

    subst_dtype = type_reader(submap2(submap1(subst_expr)))
    # type-ignore-reason: missing precise typing information in loopy
    return subst_dtype.numpy_dtype


class SubstitutionInvocationGetter(CombineMapper[frozenset[p.Call], []]):
    """
    Mapper to collect all the substitution invocations in an expression.

    .. attribute:: argument_substs

        The substitutions that are to be treated as arguments.
    """

    def __init__(self, argument_substs: frozenset[str]):
        self.argument_substs = argument_substs
        super().__init__()

    def combine(self, values: Iterable[frozenset[p.Call]]) -> frozenset[p.Call]:
        from functools import reduce

        return reduce(frozenset.union, values, frozenset())

    def map_call(self, expr: p.Call) -> frozenset[p.Call]:
        if cast("p.Variable", expr.function).name in self.argument_substs:
            return frozenset([expr])
        else:
            return super().map_call(expr)

    def map_algebraic_leaf(self, expr: Any) -> frozenset[p.Call]:
        return frozenset()

    def map_constant(self, expr: Any) -> frozenset[p.Call]:
        return frozenset()


def get_a_matched_einsum(
    t_unit: lp.TranslationUnit,
    kernel_name: str | None = None,
    insn_match: Any = None,
    argument_substitutions: frozenset[str] | None = None,
    long_dim_length: int = 500,
) -> tuple[BatchedEinsum, frozenbidict[str, str]]:
    """
    Returns a tuple of the form ``(matched_einsum, subst_map)`` where,
    ``matched_einsum`` is the batched einsum having a memory access pattern similar
    to the instructions in *insn_match* of *t_unit*, and, ``subst_map`` is a mapping
    from the variables in *t_unit* to the entities (i.e. indices, arguments) of
    *match_einsum*.

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
            raise EinsumTunitMatchError(
                "Must provide `kernel_name`, when"
                " the translation unit has more than"
                " 1 entrypoints."
            )
        else:
            kernel_name = t_unit.default_entrypoint.name

    assert isinstance(kernel_name, str)
    kernel = t_unit[kernel_name]

    argument_substitutions = argument_substitutions or frozenset(
        kernel.substitutions
    )
    assert isinstance(argument_substitutions, frozenset)

    from loopy.match import parse_match

    insn_match = parse_match(insn_match)
    insns = [insn for insn in kernel.instructions if insn_match(kernel, insn)]

    if len({insn.within_inames for insn in insns}) != 1:
        raise EinsumTunitMatchError(
            "Instructions forming the subject have"
            " more than 1 enclosing loop nest -- not"
            " allowed."
        )

    free_indices_set = {
        _get_indices_from_assignee(insn.assignees[0]) for insn in insns
    }
    if len(free_indices_set) != 1:
        raise EinsumTunitMatchError(
            "Instructions have differing free indices."
            " This is not allowed as the expressions"
            " are not batched-einsums-like."
        )

    (free_indices,) = free_indices_set

    if frozenset(free_indices) != insns[0].within_inames:
        raise EinsumTunitMatchError(
            "Einsum nested within an outer loop."
            " Such manipulation of batched einsums"
            " is not supported by feinsum."
        )

    get_reductions = ReductionCollector()
    batched_iname_set_to_operands: list[dict[tuple[str, ...], frozenset[str]]] = []
    subst_invokes_getter = SubstitutionInvocationGetter(argument_substitutions)

    # {{{ populate batched_access_to_operands

    for insn in insns:
        access_to_operands: dict[tuple[str, ...], set[str]] = {}
        redns_in_expr = get_reductions((insn.expression, tuple(insn.predicates)))
        if len(redns_in_expr) == 0:
            inner_expr = insn.expression
            ensm_inames = insn.within_inames
            redn_inames: frozenset[str] = frozenset()
        elif len(redns_in_expr) == 1:
            (redn_in_expr,) = redns_in_expr
            inner_expr = redn_in_expr.expr
            redn_inames = frozenset(redn_in_expr.inames)
            ensm_inames = insn.within_inames | redn_inames
        else:
            raise EinsumTunitMatchError(
                "More than one reductions found"
                " -> not within the grammar of expressions"
                " that can be matched by feinsum."
            )

        if isinstance(inner_expr, p.Product):
            from pymbolic.primitives import flattened_product

            flat_prod = flattened_product(inner_expr.children)
            if isinstance(flat_prod, p.Product):
                einsum_terms: tuple[p.Expression, ...] = flat_prod.children
            else:
                einsum_terms = (flat_prod,)
        else:
            einsum_terms = (inner_expr,)

        for term in einsum_terms:
            subst_invokes = subst_invokes_getter(term)
            if not subst_invokes:
                continue
            assert all(
                isinstance(subst_invoke, p.Call) for subst_invoke in subst_invokes
            )
            if not all(
                all(
                    isinstance(arg, p.Variable) and arg.name in ensm_inames
                    for arg in subst_invoke.parameters
                )
                for subst_invoke in subst_invokes
            ):
                raise EinsumTunitMatchError(
                    "Invocation to a substitution in "
                    f" '{term}' not called with the einstein"
                    " summation's indices -> not a part of"
                    " feinsum's grammar."
                )

            access_index_tuple_var: tuple[p.Variable, ...] = next(
                iter(subst_invokes)
            ).parameters  # type: ignore[assignment]

            if any(
                subst_invoke.parameters != access_index_tuple_var
                for subst_invoke in subst_invokes
            ):
                raise EinsumTunitMatchError(
                    "Not all substitution invocations in"
                    f" '{term}' are called with the same"
                    " arguments -> violates the feinsum's ."
                    " grammar."
                )

            access_index_tuple: tuple[str, ...] = tuple(
                idx.name for idx in access_index_tuple_var
            )
            subst_names = {
                cast("p.Variable", subst_invoke.function).name
                for subst_invoke in subst_invokes
            }
            access_to_operands.setdefault(access_index_tuple, set()).update(
                subst_names
            )

        batched_iname_set_to_operands.append(
            {k: frozenset(v) for k, v in access_to_operands.items()}
        )
    for iname_set_to_operands in batched_iname_set_to_operands:
        if len(iname_set_to_operands) != len(batched_iname_set_to_operands[0]):
            raise EinsumTunitMatchError(
                "Different einsums being performed by the instructions being"
                " matched against."
            )
        for iname_set, operands in iname_set_to_operands.items():
            if len(operands) != len(batched_iname_set_to_operands[0][iname_set]):
                raise EinsumTunitMatchError(
                    "Different einsums being performed by the instructions being"
                    " matched against."
                )

    # Step 1. Get a mapping from einsum inames to indices.
    ensm_inames = insns[0].within_inames | insn.reduction_inames()
    einsum_indices_generator = (chr(i) for i in range(ord("a"), ord("z") + 1))
    ensm_iname_to_index: dict[str, str] = {
        iname: next(einsum_indices_generator) for iname in ensm_inames
    }

    # Step 2. Get the $\mathcal{L}(i)$ for each $i$ in the new indices.
    iname_to_len = {
        iname: _get_iname_length(kernel, iname, long_dim_length, ensm_iname_to_index)
        for iname in ensm_inames
    }

    # Step 3. Combine these accesses to get the einsum expression.

    iname_sets: list[tuple[str, ...]] = []
    expanded_iname_sets: list[tuple[str, ...]] = []
    for iname_set, operands in sorted(
        batched_iname_set_to_operands[0].items(), key=lambda kxv: kxv[0]
    ):
        expanded_iname_sets.extend([iname_set] * len(operands))
        iname_sets.append(iname_set)
    einsum_subscripts = (
        ",".join(
            [
                "".join(ensm_iname_to_index[iname] for iname in in_iname_set)
                for in_iname_set in expanded_iname_sets
            ]
        )
        + "->"
        + "".join(ensm_iname_to_index[iname] for iname in free_indices)
    )
    arg_names: list[list[str]] = []
    for iname_set_to_operands in batched_iname_set_to_operands:
        arg_row = []
        for iname_set in iname_sets:
            arg_row.extend(sorted(iname_set_to_operands[iname_set]))
        assert len(arg_row) == len(expanded_iname_sets)
        arg_names.append(arg_row)

    # Step 4. Get the numeric data type for the substitution
    arg_to_dtype: dict[str, np.dtype[Any]] = {}

    for arg_row in arg_names:
        for arg in arg_row:
            if arg not in arg_to_dtype:
                arg_to_dtype[arg] = _get_dtype_for_subst_argument(
                    t_unit, kernel_name, arg
                )

    # Step 5. Get arg_shapes from $L$'s.
    arg_to_shape: dict[str, ShapeT] = {}
    for arg_row in arg_names:
        for arg, in_iname_set in zip(arg_row, expanded_iname_sets, strict=True):
            arg_shape = tuple(iname_to_len[iname] for iname in in_iname_set)
            if arg_to_shape.setdefault(arg, arg_shape) != arg_shape:
                raise EinsumTunitMatchError(
                    f"Substituion {arg} has multiple access maps in the instructions"
                    " being matched. This is not allowed."
                )

    # Step 6. Construct the batched einsum.
    from feinsum.make_einsum import array, batched_einsum

    beinsum = batched_einsum(
        einsum_subscripts,
        [
            [array(arg, arg_to_shape[arg], arg_to_dtype[arg]) for arg in arg_row]
            for arg_row in arg_names
        ],
    )

    # FIXME: Verify that the kernel's domain is indeed a dense-hypercube.
    output_names = ["_fe_out"] + [f"_fe_out_{i}" for i in range(len(insns) - 1)]

    subst_map = frozenbidict(
        {
            **ensm_iname_to_index,
            **{val: val for val in arg_to_dtype.keys()},
            **{
                insn.assignee_var_names()[0]: out_name
                for insn, out_name in szip(insns, output_names)
            },
        }
    )
    return beinsum, subst_map


def match_t_unit_to_einsum(
    t_unit: lp.TranslationUnit,
    einsum: BatchedEinsum,
    *,
    kernel_name: str | None = None,
    insn_match: Any = None,
    argument_substitutions: frozenset[str] | None = None,
    long_dim_length: int = 500,
) -> Mapping[str, str]:
    """
    Returns a mapping from the entities of *einsum* to the variables of the
    corresponding matched einsum in *t_unit*. See :func:`get_a_matched_einsum` for a
    subset of grammar of :mod:`loopy` kernels to be a matched as a batched einsum.
    """
    matched_einsum, var_in_tunit_to_var_in_matched_ensm = get_a_matched_einsum(
        t_unit, kernel_name, insn_match, argument_substitutions, long_dim_length
    )

    from feinsum.canonicalization import (
        get_substitution_mapping_between_isomorphic_batched_einsums,
    )

    isomorph_subst = get_substitution_mapping_between_isomorphic_batched_einsums(
        einsum, matched_einsum
    )
    matched_einsum_size_params = frozenset(
        {param.name for param in matched_einsum.all_size_params}
    )

    return Map(
        {
            var_in_ensm: var_in_tunit_to_var_in_matched_ensm.inv[var_in_matched_ensm]
            for var_in_ensm, var_in_matched_ensm in isomorph_subst.items()
            if var_in_matched_ensm not in matched_einsum_size_params
        }
    )


# }}}


# {{{ get_call_ids


class CallCollector(CombineMapper[frozenset[str], []]):
    def combine(self, values: Iterable[frozenset[str]]) -> frozenset[str]:
        from functools import reduce

        return reduce(frozenset.union, values, frozenset())

    def map_call(self, expr: p.Call) -> frozenset[str]:
        if isinstance(expr.function, p.Variable):
            return frozenset([expr.function.name]) | super().map_call(expr)
        else:
            return super().map_call(expr)

    def map_constant(self, expr: Any) -> frozenset[str]:
        return frozenset()

    def map_algebraic_leaf(self, expr: Any) -> frozenset[str]:
        return frozenset()


def get_call_ids(expr: p.Expression) -> frozenset[str]:
    """
    Returns the identifiers of the invoked functions in *expr*.
    """
    return CallCollector()(expr)


# }}}


# {{{ partition (copied from more-itertools)

Tpart = TypeVar("Tpart")


def partition(
    pred: Callable[[Tpart], bool], iterable: Iterable[Tpart]
) -> tuple[list[Tpart], list[Tpart]]:
    """
    Use a predicate to partition entries into false entries and true
    entries
    """
    # Inspired from https://docs.python.org/3/library/itertools.html
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    from itertools import filterfalse, tee

    t1, t2 = tee(iterable)
    return list(filterfalse(pred, t1)), list(filter(pred, t2))


# }}}


# {{{ hoist_reduction_invariant_terms


class EinsumTermsHoister(IdentityMapper[[]]):
    """
    Mapper to hoist products out of a sum-reduction.

    .. attribute:: reduction_inames

        Inames of the reduction expressions to perform the hoisting.
    """

    def __init__(self, reduction_inames: frozenset[str]):
        super().__init__()
        self.reduction_inames = reduction_inames

    def map_reduction(self, expr: Reduction) -> p.Expression:
        if frozenset(expr.inames) != self.reduction_inames:
            return super().map_reduction(expr)

        from loopy.library.reduction import SumReductionOperation
        from loopy.symbolic import get_dependencies

        if isinstance(expr.operation, SumReductionOperation):
            rec_inner = self.rec(expr.expr)
            if isinstance(rec_inner, p.Product):
                from pymbolic.primitives import flattened_product

                flattened = flattened_product(rec_inner.children)
                if isinstance(flattened, p.Product):
                    multiplicative_terms: tuple[p.Expression, ...] = (
                        flattened.children
                    )
                else:
                    multiplicative_terms = (flattened,)

            else:
                multiplicative_terms = (rec_inner,)

            invariants, variants = partition(
                lambda x: bool(get_dependencies(x) & self.reduction_inames),
                multiplicative_terms,
            )
            if not variants:
                # -> everything is invariant
                assert p.is_arithmetic_expression(rec_inner)
                return rec_inner * Reduction(
                    expr.operation,
                    inames=tuple(expr.inames),
                    expr=1,  # FIXME: invalid dtype (not sure how?)
                    allow_simultaneous=expr.allow_simultaneous,
                )
            if not invariants:
                # -> nothing to hoist
                return Reduction(
                    expr.operation,
                    inames=tuple(expr.inames),
                    expr=self.rec(expr.expr),
                    allow_simultaneous=expr.allow_simultaneous,
                )

            return p.Product(tuple(invariants)) * Reduction(
                expr.operation,
                inames=tuple(expr.inames),
                expr=p.Product(tuple(variants)),
                allow_simultaneous=expr.allow_simultaneous,
            )
        else:
            return super().map_reduction(expr)


def hoist_invariant_multiplicative_terms_in_sum_reduction(
    kernel: LoopKernel, reduction_inames: str | frozenset[str], within: Any = None
) -> LoopKernel:
    """
    Hoists loop-invariant multiplicative terms in a sum-reduction expression.

    :arg reduction_inames: The inames over which reduction is performed that defines
        the reduction expression that is to be transformed.
    :arg within: A match expression understood by :func:`loopy.match.parse_match`
        that specifies the instructions over which the transformation is to be
        performed.
    """
    from loopy.transform.instruction import map_instructions

    if isinstance(reduction_inames, str):
        reduction_inames = frozenset([reduction_inames])

    if not (reduction_inames <= kernel.all_inames()):
        raise ValueError(
            f"Some inames in '{reduction_inames}' not a part of" " the kernel."
        )

    term_hoister = EinsumTermsHoister(reduction_inames)

    kernel = map_instructions(  # type: ignore[no-untyped-call]
        kernel,
        insn_match=within,
        f=lambda x: x.with_transformed_expressions(term_hoister),
    )
    return kernel


# }}}


# {{{ extract_multiplicative_terms_in_sum_reduction_as_subst


class ContainsSumReduction(CombineMapper[bool, []]):
    """
    Returns *True* only if the mapper maps over an expression containing a
    SumReduction operation.
    """

    def combine(self, values: Iterable[bool]) -> bool:
        return any(values)

    def map_reduction(self, expr: Reduction) -> bool:
        from loopy.library.reduction import SumReductionOperation

        return isinstance(expr.operation, SumReductionOperation) or self.rec(
            expr.expr
        )

    def map_variable(self, expr: p.Variable) -> bool:
        return False

    def map_algebraic_leaf(self, expr: Any) -> bool:
        return False


class MultiplicativeTermReplacer(IdentityMapper[[]]):
    """
    Primary mapper of
    :func:`extract_multiplicative_terms_in_sum_reduction_as_subst`.
    """

    def __init__(
        self,
        *,
        terms_filter: Callable[[p.Expression], bool],
        subst_name: str,
        subst_arguments: tuple[p.Expression, ...],
    ) -> None:
        self.subst_name = subst_name
        self.subst_arguments = subst_arguments
        self.terms_filter = terms_filter
        super().__init__()

        # mutable state to record the expression collected by the terms_filter
        self.collected_subst_rule: SubstitutionRule | None = None

    def map_reduction(self, expr: Reduction) -> p.Expression:
        from loopy.library.reduction import SumReductionOperation
        from loopy.symbolic import SubstitutionMapper

        if isinstance(expr.operation, SumReductionOperation):
            if self.collected_subst_rule is not None:
                # => there was already a sum-reduction operation -> raise
                raise ValueError(
                    "Multiple sum reduction expressions found -> not" " allowed."
                )

            if isinstance(expr.expr, p.Product):
                from pymbolic.primitives import flattened_product

                flattened = flattened_product(expr.expr.children)
                if isinstance(flattened, p.Product):
                    terms: tuple[p.Expression, ...] = flattened.children
                else:
                    terms = (flattened,)
            else:
                terms = (expr.expr,)

            unfiltered_terms, filtered_terms = partition(self.terms_filter, terms)
            submap = SubstitutionMapper(
                {
                    argument_expr: p.Variable(f"arg{i}")
                    for i, argument_expr in enumerate(self.subst_arguments)
                }.get
            )
            self.collected_subst_rule = SubstitutionRule(
                name=self.subst_name,
                arguments=tuple(f"arg{i}" for i in range(len(self.subst_arguments))),
                expression=submap(
                    p.Product(tuple(filtered_terms)) if filtered_terms else 1
                ),
            )
            return Reduction(
                expr.operation,
                tuple(expr.inames),
                p.Product(
                    (
                        p.Variable(self.subst_name)(*self.subst_arguments),
                        *unfiltered_terms,
                    )
                ),
                expr.allow_simultaneous,
            )
        else:
            return super().map_reduction(expr)


def extract_multiplicative_terms_in_sum_reduction_as_subst(
    kernel: LoopKernel,
    within: Any,
    subst_name: str,
    arguments: Sequence[p.Expression],
    terms_filter: Callable[[p.Expression], bool],
) -> LoopKernel:
    """
    Returns a copy of *kernel* with a new substitution named *subst_name* and
    *arguments* as arguments for the aggregated multiplicative terms in a
    sum-reduction expression.

    :arg within: A match expression understood by :func:`loopy.match.parse_match`
        to specify the instructions over which the transformation is to be
        performed.
    :arg terms_filter: A callable to filter which terms of the sum-reduction
        comprise the body of substitution rule.
    :arg arguments: The sub-expressions of the product of the filtered terms that
        form the arguments of the extract substitution rule in the same order.

    .. note::

        A ``LoopyError`` is raised if none or more than 1 sum-reduction expression
        appear in *within*.
    """
    from loopy.match import parse_match

    within = parse_match(within)

    matched_insns = [
        insn
        for insn in kernel.instructions
        if within(kernel, insn)
        and ContainsSumReduction()((insn.expression, tuple(insn.predicates)))
    ]

    if len(matched_insns) == 0:
        raise LoopyError(
            f"No instructions found matching '{within}'"
            " with sum-reductions found."
        )
    if len(matched_insns) > 1:
        raise LoopyError(
            f"More than one instruction found matching '{within}'"
            " with sum-reductions found -> not allowed."
        )

    (insn,) = matched_insns
    replacer = MultiplicativeTermReplacer(
        subst_name=subst_name,
        subst_arguments=tuple(arguments),
        terms_filter=terms_filter,
    )
    new_insn = insn.with_transformed_expressions(replacer)
    new_rule = replacer.collected_subst_rule
    new_substitutions = dict(kernel.substitutions).copy()
    if subst_name in new_substitutions:
        raise LoopyError(
            f"Kernel '{kernel.name}' already contains a substitution"
            f" rule named '{subst_name}'."
        )
    assert new_rule is not None
    new_substitutions[subst_name] = new_rule

    return kernel.copy(
        instructions=[
            new_insn if insn.id == new_insn.id else insn
            for insn in kernel.instructions
        ],
        substitutions=new_substitutions,
    )


# }}}

# {{{ decouple_domain


@for_each_kernel
def decouple_domain(
    kernel: LoopKernel, inames: Collection[str], parent_inames: Collection[str]
) -> LoopKernel:
    r"""
    Returns a copy of *kernel* with altered domains. The home domain of
    *inames* i.e. :math:`\mathcal{D}^{\text{home}}({\text{inames}})` is
    replaced with two domains :math:`\mathcal{D}_1` and :math:`\mathcal{D}_2`.
    :math:`\mathcal{D}_1` is the domain with dimensions corresponding to *inames*
    projected out and :math:`\mathcal{D}_2` is the domain with all the dimensions
    other than the ones corresponding to *inames* projected out.

    :arg inames: The inamaes to be decouple from their home domain.
    :arg parent_inames: Inames in :math:`\mathcal{D}^{\text{home}}({\text{inames}})`
        that will be used as additional parametric dimensions during the
        construction of :math:`\mathcal{D}_1`.

    .. note::

        - An error is raised if all the *inames* do not correspond to the same home
          domain of *kernel*.
        - It is the caller's responsibility to ensure that :math:`\mathcal{D}_1
          \cup \mathcal{D}_2 = \mathcal{D}^{\text{home}}({\text{inames}})`. If this
          criterion is violated this transformation would violate dependencies.
    """

    # {{{ sanity checks

    if not inames:
        raise LoopyError(
            "No inames were provided to decouple into" " a different domain."
        )
    if frozenset(parent_inames) & frozenset(inames):
        raise LoopyError("Inames cannot be appear in `inames` and `parent_inames`.")

    # }}}

    hdi = kernel.get_home_domain_index(next(iter(inames)))
    for iname in inames:
        if kernel.get_home_domain_index(iname) != hdi:
            raise LoopyError("inames are not a part of the same home domain.")

    all_dims = frozenset(kernel.domains[hdi].get_var_dict())
    for parent_iname in parent_inames:
        if parent_iname not in all_dims:
            raise LoopyError(
                f"Parent iname '{parent_iname}' not a part of the"
                f" corresponding home domain '{kernel.domains[hdi]}'."
            )

    dom1 = kernel.domains[hdi]
    dom2 = kernel.domains[hdi]

    for iname in sorted(all_dims):
        if iname in inames:
            dt, pos = dom1.get_var_dict()[iname]
            dom1 = dom1.project_out(dt, pos, 1)
        elif iname in parent_inames:
            dt, pos = dom2.get_var_dict()[iname]
            if dt != isl.dim_type.param:
                n_params = dom2.dim(isl.dim_type.param)
                dom2 = dom2.move_dims(isl.dim_type.param, n_params, dt, pos, 1)
        else:
            dt, pos = dom2.get_var_dict()[iname]
            dom2 = dom2.project_out(dt, pos, 1)

    new_domains = list(kernel.domains)
    new_domains[hdi] = dom1
    new_domains.append(dom2)
    kernel = kernel.copy(domains=new_domains)
    return kernel


# }}}

# vim: foldmethod=marker
