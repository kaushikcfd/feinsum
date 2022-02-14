"""
:mod:`loopy` helpers. The aim is to keep this module as lean as possible by
upstreaming the heavy lifting parts to :mod:`loopy` itself.

.. autofunction:: extract_subexpr_of_associative_op_as_subst
"""

import loopy as lp
import pymbolic.interop.matchpy as m
import pymbolic.primitives as p

from multiset import Multiset
from typing import Union, ClassVar, Optional
from dataclasses import dataclass
from pymbolic.interop.matchpy.tofrom import (
    ToMatchpyExpressionMapper as BaseToMatchpyExpressionMapper,
    FromMatchpyExpressionMapper as BaseFromMatchpyExpressionMapper)
from matchpy import Arity

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

# vim: foldmethod=marker
