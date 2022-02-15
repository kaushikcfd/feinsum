"""
:mod:`loopy` helpers. The aim is to keep this module as lean as possible by
upstreaming the heavy lifting parts to :mod:`loopy` itself.

.. autofunction:: extract_subexpr_of_associative_op_as_subst
.. autofunction:: match_t_unit_to_einsum
"""

import loopy as lp
import pymbolic.interop.matchpy as m
import pymbolic.primitives as p

from multiset import Multiset
from typing import Union, ClassVar, Optional, Tuple
from pyrsistent.typing import PMap as PMapT
from pyrsistent import pmap
from dataclasses import dataclass
from more_itertools import zip_equal as szip
from pymbolic.interop.matchpy.tofrom import (
    ToMatchpyExpressionMapper as BaseToMatchpyExpressionMapper,
    FromMatchpyExpressionMapper as BaseFromMatchpyExpressionMapper)
from matchpy import Arity
from feinsum.einsum import (FusedEinsum, FreeAxis, SizeParam, EinsumAxisAccess,
                            SummationAxis)
from feinsum.diagnostics import EinsumTunitMatchError
from pytools.tag import Tag
from loopy.symbolic import pw_aff_to_expr

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

        return type(self.rule_rhs)((self.rule_lhs, *leftover_args))


# {{{ extract_subst

def extract_subexpr_of_associative_op_as_subst(
        kernel: lp.LoopKernel,
        rule_lhs: Union[p.Call, p.Variable, str],
        rule_rhs: p.Expression) -> lp.LoopKernel:
    from loopy.symbolic import parse, get_dependencies
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
    )

    new_insns = [
        insn.with_transformed_expressions(
            lambda expr: m.replace_all(expr, [replacement_rule],
                                       to_matchpy_expr, from_matchpy_expr))
        for insn in kernel.instructions]

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


def match_t_unit_to_einsum(t_unit: lp.TranslationUnit, ref_einsum: FusedEinsum,
                           insn_match_tag: Optional[Tag] = None,
                           long_dim_length: int = 5_000) -> PMapT[str, str]:
    """
    Returns a mapping from the variable names in *ref_einsum* to the variables
    in *t_unit*. The unification is done by matching the instructions tagged
    with *insn_match_tag* against *ref_einsum*.

    :param t_unit: The subject translation unit which is being matched against
        *ref_einsum*.
    :param ref_einsum: The pattern to match *t_unit* against.
    :param insn_match_tag: An optional tag representing the subset of the
        kernel's instructions that are to be considered as the subject of
        matching against the *ref_einsum* pattern.
    :param long_dim_length: Axis length above which can be assumed to be long
        enough to be invariant of having any noticeable effects on the kernel's
        performance.
    """
    subst_map = {}

    if len({clbl
            for clbl in t_unit.callables_table.values()
            if isinstance(clbl, lp.CallableKernel)}) > 1:
        # The logic for matching against multiple callables is more inolved,
        # skipping for now.
        raise NotImplementedError("Handling translation units with multiple"
                                  " kernels is currently not supported.")

    if insn_match_tag is None:
        insns = t_unit.default_entrypoint.instructions
    else:
        insns = [insn
                 for insn in t_unit.default_entrypoint.instructions
                 if insn.tags_of_type(insn_match_tag)]

    if len({insn.within_inames for insn in insns}) != 1:
        raise EinsumTunitMatchError("Instructions forming the subject have"
                                    " more than 1 enclosing loop nest -- not"
                                    " allowed.")

    if len(insns) != ref_einsum.noutputs:
        raise EinsumTunitMatchError("Number of outputs in ref_einsum, t_unit"
                                    " do not match.")

    free_indices_set = {_get_indices_from_assignee(insn.assignees[0])
                        for insn in insns}
    if len(free_indices_set) != 1:
        raise EinsumTunitMatchError("Instructions have differing free indices"
                                    " -- not allowed.")
    free_indices, = free_indices_set

    if len(free_indices) != ref_einsum.ndim:
        raise EinsumTunitMatchError("#free indices in the t_unit not equal"
                                    " to the #free indices in ref_einsum.")

    for i, physical_idx in enumerate(free_indices):
        if physical_idx in subst_map:
            raise EinsumTunitMatchError("Cannot have repeating free indices --"
                                        " not an einsum")
        else:
            _check_if_t_unit_and_ref_einsum_have_the_same_axis_dim(
                FreeAxis(i),
                physical_idx,
                ref_einsum,
                t_unit.default_entrypoint,
                long_dim_length=long_dim_length)
            subst_map[physical_idx] = ref_einsum.index_names[FreeAxis(i)]

    for insn, use_row in szip(insns, ref_einsum.use_matrix):
        if not isinstance(insn.expression, lp.Reduction):
            raise EinsumTunitMatchError(f"Instruction {insn} does not have a"
                                        " reduction RHS.")

        match_pattern = 1
        for access_descrs, values in szip(ref_einsum.access_descriptors,
                                         use_row):
            if len(values) > 1:
                raise NotImplementedError("Fused einsums with multiple"
                                          " values for a use -- not supported.")
            value, = values
            wc_indices = []
            for access_descr in access_descrs:
                ref_idx_name = ref_einsum.index_names[access_descr]
                if isinstance(access_descr, FreeAxis):
                    wc_indices.append(p.Variable(subst_map[ref_idx_name]))
                elif isinstance(access_descr, SummationAxis):
                    wc_indices.append(p.DotWildcard(ref_idx_name))
                else:
                    raise AssertionError()
            match_pattern *= p.DotWildcard(value)[tuple(wc_indices)]

        matches = list(m.match(insn.expression.expr, match_pattern))
        if len(matches) == 0:
            raise EinsumTunitMatchError(f"Expression matching for '{insn}' failed.")
        elif len(matches) > 1:
            raise NotImplementedError("Obtained more than 1 matches"
                                      " -- not yet supported.")
        else:
            for name_in_ref_einsum, name_in_t_unit in matches[0].items():
                # FIXME: Make sure to check the lengths of the reduction axes
                if name_in_ref_einsum in subst_map:
                    if subst_map[name_in_ref_einsum] != name_in_t_unit.name:
                        raise EinsumTunitMatchError("Non-unique values obtained for"
                                                    f" '{name_in_ref_einsum}'"
                                                    " variable of the einsum.")
                else:
                    subst_map[name_in_ref_einsum] = name_in_t_unit.name

    return pmap(subst_map)

# }}}

# vim: foldmethod=marker
