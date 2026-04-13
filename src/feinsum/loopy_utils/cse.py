__copyright__ = """
Copyright (C) 2026 Kaushik Kulkarni
"""

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

from collections.abc import Collection
from functools import reduce
from typing import cast

import loopy as lp
import pymbolic.primitives as p
from constantdict import constantdict
from loopy.diagnostic import DependencyTypeInferenceFailure, LoopyError
from loopy.match import ToMatchConvertible
from loopy.symbolic import (
    IdentityMapper,
    ResolvedFunction,
    WalkMapper,
    get_dependencies,
)
from loopy.translation_unit import make_clbl_inf_ctx
from loopy.type_inference import TypeInferenceMapper
from pymbolic.typing import Expression


class NSuccRecorder(WalkMapper[[]]):
    """
    Records the number of successors of an expression in :attr:`nsuccs`.
    """

    def __init__(self) -> None:
        self.nsuccs: dict[Expression, int] = {}
        super().__init__()

    def _record_succ(self, expr: Expression) -> None:
        self.nsuccs[expr] = self.nsuccs.get(expr, 0) + 1

    def map_sum(self, expr: p.Sum) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_sum(expr)

    def map_product(self, expr: p.Product) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_product(expr)

    def map_bitwise_or(self, expr: p.BitwiseOr) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_bitwise_or(expr)

    def map_bitwise_xor(self, expr: p.BitwiseXor) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_bitwise_xor(expr)

    def map_bitwise_and(self, expr: p.BitwiseAnd) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_bitwise_and(expr)

    def map_logical_or(self, expr: p.LogicalOr) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_logical_or(expr)

    def map_logical_and(self, expr: p.LogicalAnd) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_logical_and(expr)

    def map_min(self, expr: p.Min) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_min(expr)

    def map_max(self, expr: p.Max) -> None:
        for child in expr.children:
            self._record_succ(child)
        super().map_max(expr)

    def map_subscript(self, expr: p.Subscript) -> None:
        self._record_succ(expr.index)
        super().map_subscript(expr)

    def map_lookup(self, expr: p.Lookup) -> None:
        self._record_succ(expr.aggregate)
        super().map_lookup(expr)

    def map_quotient(self, expr: p.Quotient) -> None:
        self._record_succ(expr.numerator)
        self._record_succ(expr.denominator)
        super().map_quotient(expr)

    def map_floor_div(self, expr: p.FloorDiv) -> None:
        self._record_succ(expr.numerator)
        self._record_succ(expr.denominator)
        super().map_floor_div(expr)

    def map_remainder(self, expr: p.Remainder) -> None:
        self._record_succ(expr.numerator)
        self._record_succ(expr.denominator)
        super().map_remainder(expr)

    def map_power(self, expr: p.Power) -> None:
        self._record_succ(expr.base)
        self._record_succ(expr.exponent)
        super().map_power(expr)

    def map_comparison(self, expr: p.Comparison) -> None:
        self._record_succ(expr.left)
        self._record_succ(expr.right)
        super().map_comparison(expr)

    def map_left_shift(self, expr: p.LeftShift) -> None:
        self._record_succ(expr.shiftee)
        self._record_succ(expr.shift)
        super().map_left_shift(expr)

    def map_right_shift(self, expr: p.RightShift) -> None:
        self._record_succ(expr.shiftee)
        self._record_succ(expr.shift)
        super().map_right_shift(expr)

    def map_bitwise_not(self, expr: p.BitwiseNot) -> None:
        self._record_succ(expr.child)
        super().map_bitwise_not(expr)

    def map_logical_not(self, expr: p.LogicalNot) -> None:
        self._record_succ(expr.child)
        super().map_logical_not(expr)

    def map_common_subexpression(self, expr: p.CommonSubexpression) -> None:
        raise NotImplementedError

    def map_if(self, expr: p.If) -> None:
        self._record_succ(expr.condition)
        self._record_succ(expr.then)
        self._record_succ(expr.else_)
        super().map_if(expr)

    def map_call(self, expr: p.Call) -> None:
        for child in expr.parameters:
            self._record_succ(child)
        super().map_call(expr)

    def map_reduction(self, expr: object) -> None:
        raise NotImplementedError

    def map_type_cast(self, expr: object) -> None:
        from loopy.symbolic import TypeCast

        assert isinstance(expr, TypeCast)
        self._record_succ(expr.child)
        super().map_type_cast(expr)

    def map_linear_subscript(self, expr: object) -> None:
        from loopy.symbolic import LinearSubscript

        assert isinstance(expr, LinearSubscript)
        self._record_succ(expr.aggregate)
        self._record_succ(expr.index)
        super().map_linear_subscript(expr)

    def map_sub_array_ref(self, expr: object) -> None:
        from loopy.symbolic import SubArrayRef

        assert isinstance(expr, SubArrayRef)
        self._record_succ(expr.swept_inames)
        self._record_succ(expr.subscript)
        super().map_sub_array_ref(expr)

    def map_resolved_function(self, expr: object) -> None:
        from loopy.symbolic import ResolvedFunction

        assert isinstance(expr, ResolvedFunction)
        self._record_succ(expr.function)
        super().map_resolved_function(expr)


class CSEMapper(IdentityMapper[[frozenset[Expression]]]):
    def __init__(
        self,
        cse_to_var_name: constantdict[Expression, str],
        cse_to_insn_id: constantdict[str, str],
        default_deps: frozenset[str],
        within_inames: frozenset[str],
        type_inf_mapper: TypeInferenceMapper,
    ) -> None:
        super().__init__()
        self.cse_to_var_name = cse_to_var_name
        self.cse_var_to_insn_id = cse_to_insn_id
        self.default_deps = default_deps
        self.within_inames = within_inames
        self.type_inf_mapper = type_inf_mapper

        # Mutable state:
        self.new_insns: list[lp.Assignment] = []
        self.initialized_cses: set[str] = set()

    def _get_cse_insn_deps_for_expr(
        self, exprs: Collection[Expression]
    ) -> frozenset[str]:
        deps: set[str] = set()
        for expr in exprs:
            deps |= set(get_dependencies(expr))
        return self.default_deps | frozenset(
            {
                self.cse_var_to_insn_id[dep]
                for dep in deps
                if dep in self.cse_var_to_insn_id
            }
        )

    def _maybe_replace_with_cse(
        self,
        orig_expr: Expression,
        new_expr: Expression,
        predicates: frozenset[Expression],
    ) -> Expression:
        """If *orig_expr* is a CSE candidate, emit its assignment and return a
        variable; otherwise return *new_expr* unchanged."""
        try:
            cse_var = self.cse_to_var_name[orig_expr]
        except KeyError:
            return new_expr

        if cse_var not in self.initialized_cses:

            try:
                orig_expr_dtype = self.type_inf_mapper(orig_expr)
            except DependencyTypeInferenceFailure:
                zero = 0
            else:
                np_dtype = orig_expr_dtype.numpy_dtype
                zero = np_dtype.type(0)

            assert cse_var in self.cse_var_to_insn_id
            new_insn = lp.Assignment(
                p.Variable(cse_var),
                (
                    p.If(
                        condition=p.LogicalAnd(tuple(pred for pred in predicates)),
                        then=new_expr,
                        else_=zero,
                    )
                    if predicates
                    else new_expr
                ),
                self.cse_var_to_insn_id[cse_var],
                depends_on=self._get_cse_insn_deps_for_expr(
                    frozenset({new_expr}) | predicates
                ),
                within_inames=self.within_inames,
            )
            self.new_insns.append(new_insn)
            assert new_insn.id not in new_insn.depends_on
            self.initialized_cses.add(cse_var)
        return p.Variable(cse_var)

    def rec(
        self, expr: Expression, predicates: frozenset[Expression]
    ) -> Expression:
        mapped_expr = super().rec(expr, predicates)
        return self._maybe_replace_with_cse(expr, mapped_expr, predicates)

    # {{{ if - special: predicates differ per branch

    def map_if(self, expr: p.If, predicates: frozenset[Expression]) -> Expression:
        new_cond = self.rec(expr.condition, predicates)
        new_then = self.rec(expr.then, predicates | frozenset({new_cond}))
        new_else = self.rec(
            expr.else_, predicates | frozenset({p.LogicalNot(new_cond)})
        )
        return p.If(new_cond, new_then, new_else)

    # }}}


def get_nsuccs(exprs: tuple[Expression, ...]) -> constantdict[Expression, int]:
    mapper = NSuccRecorder()
    for expr in exprs:
        mapper(expr)
    return constantdict(mapper.nsuccs)


@lp.for_each_kernel
def hoist_cses(
    kernel: lp.LoopKernel, within: ToMatchConvertible = None
) -> lp.LoopKernel:
    """
    Hoist repeated subexpressions in a loop nest into private temporaries.

    Scans the instructions that belong to all inames in *inames* and finds
    every compound subexpression that appears as a direct child of two or more
    parent nodes in the expression DAG (i.e. its *successor count* >= 2).  Each
    such subexpression is assigned to a fresh private scalar temporary and the
    original instructions are rewritten to reference that temporary instead.

    :arg within: Instructions nested in the same loop nest for which the CSEs
        are to be hoisted.
    :returns: A new kernel with CSE temporaries introduced and the original
        instructions updated to reference them.

    Example -- three CSEs are found: ``a[i]*b[i]`` is shared across four
    instructions, ``a[i]*b[i] + c[i]`` appears in two of them, and
    ``d[i] + e[i]`` in two others::

        >>> import loopy as lp
        >>> import numpy as np
        >>> t_unit = lp.make_kernel(
        ...     "{[i]: 0 <= i < n}",
        ...     '''
        ...     out1[i] = (a[i]*b[i] + c[i]) * (d[i] + e[i])
        ...     out2[i] = (a[i]*b[i]) * f[i]
        ...     out3[i] = (a[i]*b[i] + c[i]) * g[i]
        ...     out4[i] = (d[i] + e[i]) * h[i]
        ...     ''',
        ...     [lp.GlobalArg("a,b,c,d,e,f,g,h,out1,out2,out3,out4",
        ...                   np.float64, shape=("n",)),
        ...      lp.ValueArg("n")],
        ...     lang_version=(2018, 2),
        ... )
        >>> t_unit = hoist_cses(t_unit)
        >>> t_unit == lp.make_kernel(
        ...     "{[i]: 0 <= i < n}",
        ...     '''
        ...     _cse_1 = a[i]*b[i] {id=_cse_def_1}
        ...     _cse = _cse_1 + c[i] {id=_cse_def, dep=_cse_def_1}
        ...     _cse_0 = d[i] + e[i] {id=_cse_def_0}
        ...     out1[i] = _cse*_cse_0 {id=insn,dep=_cse_def:_cse_def_0:_cse_def_1}
        ...     out2[i] = _cse_1*f[i] {id=insn_0,dep=_cse_def:_cse_def_0:_cse_def_1}
        ...     out3[i] = _cse*g[i] {id=insn_1,dep=_cse_def:_cse_def_0:_cse_def_1}
        ...     out4[i] = _cse_0*h[i] {id=insn_2,dep=_cse_def:_cse_def_0:_cse_def_1}
        ...     ''',
        ...     [lp.GlobalArg("a,b,c,d,e,f,g,h,out1,out2,out3,out4",
        ...                   np.float64, shape=("n",)),
        ...      lp.ValueArg("n"),
        ...      lp.TemporaryVariable("_cse",   None, (), lp.AddressSpace.PRIVATE),
        ...      lp.TemporaryVariable("_cse_0", None, (), lp.AddressSpace.PRIVATE),
        ...      lp.TemporaryVariable("_cse_1", None, (), lp.AddressSpace.PRIVATE)],
        ...     lang_version=(2018, 2),
        ... )
        True
    """
    from loopy.match import parse_match
    within = parse_match(within)
    insn_ids = {insn.id for insn in kernel.instructions if within(kernel, insn)}
    if not insn_ids:
        raise LoopyError("No instructions found satisfying within.")

    if len({kernel.id_to_insn[id_].within_inames for id_ in insn_ids}) != 1:
        raise LoopyError(
            "hoist_cses requires all instructions to be nested in the same loop"
            " nest."
        )

    inames = kernel.id_to_insn[next(iter(insn_ids))].within_inames
    for iname in inames:
        insn_ids &= kernel.iname_to_insns()[iname]

    expr_to_nsucc = get_nsuccs(
        tuple(kernel.id_to_insn[id_].expression for id_ in sorted(insn_ids))
    )

    ing = kernel.get_instruction_id_generator()
    vng = kernel.get_var_name_generator()
    cse_to_var_name: dict[Expression, str] = {}
    cse_var_to_insn_id: dict[str, str] = {}

    for expr, nsucc in expr_to_nsucc.items():
        # consider subexpressions with repeat accesses that are not algebraic leafs.
        if nsucc >= 2 and (
            isinstance(expr, p.ExpressionNode)
            and not (
                isinstance(
                    expr, (p.Variable, p.NaN, p.FunctionSymbol, ResolvedFunction)
                )
            )
        ):
            var_name = vng("_cse")
            cse_to_var_name[expr] = var_name
            cse_var_to_insn_id[var_name] = ing("_cse_def")

    cse_mapper = CSEMapper(
        constantdict(cse_to_var_name),
        constantdict(cse_var_to_insn_id),
        reduce(
            lambda x, y: x | y,
            (kernel.id_to_insn[id_].depends_on for id_ in insn_ids),
            cast("frozenset[str]", frozenset()),
        )
        - insn_ids,
        inames,
        TypeInferenceMapper(kernel, make_clbl_inf_ctx(constantdict(), frozenset())),  # type: ignore[no-untyped-call]
    )
    new_insns: list[lp.InstructionBase] = []
    for id_ in sorted(insn_ids):
        insn = kernel.id_to_insn[id_]

        new_insns.append(
            insn.with_transformed_expressions(
                lambda expr, id_=id_: cse_mapper(
                    expr, kernel.id_to_insn[id_].predicates
                )
            ).copy(
                depends_on=insn.depends_on | frozenset(cse_var_to_insn_id.values())
            )
        )

    return kernel.copy(
        instructions=(
            cse_mapper.new_insns
            + new_insns
            + [insn for insn in kernel.instructions if insn.id not in insn_ids]
        ),
        temporary_variables=constantdict(
            {
                **kernel.temporary_variables,
                **{
                    cse_var_name: lp.TemporaryVariable(
                        cse_var_name, None, (), lp.AddressSpace.PRIVATE
                    )
                    for cse_var_name in cse_to_var_name.values()
                },
            }
        ),
    )
