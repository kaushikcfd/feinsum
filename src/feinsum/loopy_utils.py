"""
:mod:`loopy` helpers. The aim is to keep this module as lean as possible by
upstreaming the heavy lifting parts to :mod:`loopy` itself.

.. autofunction:: get_a_matched_einsum
.. autofunction:: match_t_unit_to_einsum
.. autofunction:: get_call_ids
"""

from collections.abc import Iterable, Mapping
from typing import (
    Any,
    cast,
)

import loopy as lp
import numpy as np
import pymbolic.primitives as p
from bidict import frozenbidict
from immutables import Map
from loopy.symbolic import CombineMapper, Reduction
from more_itertools import zip_equal as szip
from pytools import memoize_on_first_arg

from feinsum.diagnostics import EinsumTunitMatchError
from feinsum.einsum import BatchedEinsum, IntegralT

# {{{ matching loopy kernel against a ref. einsum


# type-ignore-reason: deriving from Any
class ReductionCollector(CombineMapper):  # type: ignore[misc]
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
        return tuple(idx.name for idx in assignee.index_tuple)
    else:
        raise EinsumTunitMatchError(
            "Not all instructions assign to Variable or"
            " a Subscript with free indices --"
            " not allowed."
        )


def _get_iname_length(
    kernel: lp.LoopKernel, iname: str, long_dim_length: IntegralT
) -> float | IntegralT:
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
            return np.inf
        else:
            return ubound_val + 1
    else:
        # Parametric upper bound => infty
        return np.inf


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
    type_reader = TypeReader(knl, t_unit.callables_table)
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
    return subst_dtype.numpy_dtype  # type: ignore[no-any-return]


# type-ignore-reason: deriving from Any
class SubstitutionInvocationGetter(CombineMapper):  # type: ignore[misc]
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
        if expr.function.name in self.argument_substs:
            return frozenset([expr])
        else:
            # type-ignore-reason: CombineMapper.map_call does not provide type
            # information
            return super().map_call(expr)  # type: ignore[no-any-return]

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
    batched_access_to_operands: list[Mapping[tuple[str, ...], frozenset[str]]] = []
    subst_invokes_getter = SubstitutionInvocationGetter(argument_substitutions)

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
            ).parameters

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
                subst_invoke.function.name for subst_invoke in subst_invokes
            }
            (
                access_to_operands.setdefault(access_index_tuple, set()).update(
                    subst_names
                )
            )

        batched_access_to_operands.append(
            Map({k: frozenset(v) for k, v in access_to_operands.items()})
        )

    # TODO: We need to do something about the einsum index dependencies on the
    # expression itself. It gets trickier if it comes as a multiplicative term and
    # we intend to hoist certain terms.

    # Step 1. Get a mapping from einsum inames to indices.
    ensm_inames = insns[0].within_inames | insn.reduction_inames()
    einsum_indices_generator = (chr(i) for i in range(ord("a"), ord("z") + 1))
    ensm_iname_to_index: dict[str, str] = {
        iname: next(einsum_indices_generator) for iname in ensm_inames
    }

    # Step 2. Get the $\mathcal{L}(i)$ for each $i$ in the new indices.
    iname_to_len = {
        iname: _get_iname_length(kernel, iname, long_dim_length)
        for iname in ensm_inames
    }

    # Step 3. Combine these accesses to get the einsum expression.
    from functools import reduce

    # type-ignore-reason: looks like mypy is not able to deduce that frozenset is an
    # iterable.
    # TODO: open an issue and link it here.
    unioned_accesses: tuple[tuple[str, ...], ...] = tuple(
        reduce(
            frozenset.union,  # type: ignore[arg-type]
            (
                frozenset(acc_to_operands)
                for acc_to_operands in batched_access_to_operands
            ),
            cast("frozenset[tuple[str, ...]]", frozenset()),
        )
    )
    einsum_subscripts = (
        ",".join(
            [
                "".join(ensm_iname_to_index[iname] for iname in accesses)
                for accesses in unioned_accesses
            ]
        )
        + "->"
        + "".join(ensm_iname_to_index[iname] for iname in free_indices)
    )
    use_matrix: list[list[frozenset[str]]] = []
    for acc_to_operands in batched_access_to_operands:
        use_row = [
            acc_to_operands.get(accesses, frozenset())
            for accesses in unioned_accesses
        ]
        use_matrix.append(use_row)

    # Step 4. Get the numeric data type for the substitution
    value_to_dtype: dict[str, np.dtype[Any]] = {}

    for use_row in use_matrix:
        for uses in use_row:
            for use in uses:
                if use not in value_to_dtype:
                    value_to_dtype[use] = _get_dtype_for_subst_argument(
                        t_unit, kernel_name, use
                    )

    # Step 5. Get arg_shapes from $L$'s.
    arg_shapes: list[tuple[float | IntegralT, ...]] = []
    for accesses in unioned_accesses:
        arg_shapes.append(tuple(iname_to_len[iname] for iname in accesses))

    # Step 6. Construct the batched einsum.
    from feinsum.make_einsum import batched_einsum

    batched_einsum = batched_einsum(
        einsum_subscripts,
        arg_shapes,
        use_matrix=use_matrix,
        value_to_dtype=value_to_dtype,
    )

    # FIXME: Verify that the kernel's domain is indeed a dense-hypercube.
    output_names = ["_fe_out"] + [f"_fe_out_{i}" for i in range(len(insns) - 1)]

    subst_map = frozenbidict(
        {
            **ensm_iname_to_index,
            **{val: val for val in value_to_dtype.keys()},
            **{
                insn.assignee_var_names()[0]: out_name
                for insn, out_name in szip(insns, output_names)
            },
        }
    )
    return batched_einsum, subst_map


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

    return Map(
        {
            var_in_ensm: var_in_tunit_to_var_in_matched_ensm.inv[var_in_matched_ensm]
            for var_in_ensm, var_in_matched_ensm in isomorph_subst.items()
        }
    )


# }}}


# {{{ get_call_ids


# type-ignore-reason: deriving from CombineMapper (i.e. Any)
class CallCollector(CombineMapper):  # type: ignore[misc]
    def combine(self, values: Iterable[frozenset[str]]) -> frozenset[str]:
        from functools import reduce

        return reduce(frozenset.union, values, frozenset())

    def map_call(self, expr: p.Call) -> frozenset[str]:
        if isinstance(expr.function, p.Variable):
            return frozenset(
                [expr.function.name]
            ) | super().map_call(  # type: ignore[no-any-return]
                expr
            )
        else:
            return super().map_call(expr)  # type: ignore[no-any-return]

    def map_constant(self, expr: Any) -> frozenset[str]:
        return frozenset()

    def map_algebraic_leaf(self, expr: Any) -> frozenset[str]:
        return frozenset()


def get_call_ids(expr: p.Expression) -> frozenset[str]:
    """
    Returns the identifiers of the invoked functions in *expr*.
    """
    return CallCollector()(expr)  # type: ignore[no-any-return]


# }}}

# vim: foldmethod=marker
