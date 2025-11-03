from __future__ import annotations

__doc__ = """
.. autofunction:: canonicalize_einsum
"""

__copyright__ = "Copyright (C) 2022 Kaushik Kulkarni"

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

import abc
from dataclasses import dataclass
from functools import cached_property, reduce
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Mapping

import numpy as np
from bidict import frozenbidict
from immutables import Map
from pytools import UniqueNameGenerator, memoize_method

from feinsum.einsum import (
    INT_CLASSES,
    BatchedEinsum,
    IntegralT,
    ShapeComponentT,
    SizeParam,
)

# {{{ private interface to build an einsum graph


class EinsumGraphNode(abc.ABC):
    """
    A node in the einsum graph to canonicalize it.
    """

    @property
    @abc.abstractmethod
    def label(self) -> str:
        """
        Label used for graphviz visualization.
        """
        pass


@dataclass(frozen=True)
class ArgNode(EinsumGraphNode):
    name: str

    @property
    def label(self) -> str:
        return f"{self.name}"


@dataclass(frozen=True)
class IndexNode(EinsumGraphNode):
    name: str

    @property
    def label(self) -> str:
        return f"{self.name}"


@dataclass(frozen=True)
class AxisLengthNode(EinsumGraphNode):
    length: ShapeComponentT

    @property
    def label(self) -> str:
        name = (
            self.length.name
            if isinstance(self.length, SizeParam)
            else str(self.length)
        )
        return f"Length:{name}"


@dataclass(frozen=True)
class DimNode(EinsumGraphNode):
    dim: int

    @property
    def label(self) -> str:
        return f"dim-{self.dim + 1}"


@dataclass(frozen=True)
class DtypeNode(EinsumGraphNode):
    dtype: np.dtype[Any]

    @property
    def label(self) -> str:
        return f"{self.dtype}"


@dataclass(frozen=True)
class InputAccessNode(EinsumGraphNode):
    i: int
    j: int
    index: str
    d: int

    @property
    def label(self) -> str:
        return f"({self.i + 1}, {self.j + 1}, {self.index}, {self.d + 1})"


@dataclass(frozen=True)
class OutputAccessNode(EinsumGraphNode):
    index: str
    d: int

    @property
    def label(self) -> str:
        return f"({self.index}, {self.d + 1})"


@dataclass(frozen=True)
class IResultNode(EinsumGraphNode):
    i: int

    @property
    def label(self) -> str:
        return f"R-{self.i + 1}"


@dataclass(frozen=True)
class IPositionNode(EinsumGraphNode):
    j: int

    @property
    def label(self) -> str:
        return f"Pos-{self.j + 1}"


def to_bliss_color(v: EinsumGraphNode) -> int:
    if isinstance(v, ArgNode):
        return 1
    elif isinstance(v, IndexNode):
        return 2
    elif isinstance(v, InputAccessNode):
        return 3
    elif isinstance(v, OutputAccessNode):
        return 4
    elif isinstance(v, IResultNode):
        return 5
    elif isinstance(v, IPositionNode):
        return 6
    elif isinstance(v, DtypeNode):
        return 7
    elif isinstance(v, AxisLengthNode):
        return 8
    elif isinstance(v, DimNode):
        return 9
    else:
        raise NotImplementedError(f"Unknown node type '{type(v)}'.")


def to_graphviz_color(v: EinsumGraphNode) -> str:
    if isinstance(v, ArgNode):
        return "deepskyblue"
    elif isinstance(v, IndexNode):
        return "dodgerblue4"
    elif isinstance(v, InputAccessNode):
        return "darkseagreen1"
    elif isinstance(v, OutputAccessNode):
        return "forestgreen"
    elif isinstance(v, IResultNode):
        return "darksalmon"
    elif isinstance(v, IPositionNode):
        return "crimson"
    elif isinstance(v, DtypeNode):
        return "navajowhite"
    elif isinstance(v, AxisLengthNode):
        return "orange"
    elif isinstance(v, DimNode):
        return "thistle"
    else:
        raise NotImplementedError(f"Unknown node type '{type(v)}'.")


# }}}


def _axis_len_le(len1: ShapeComponentT, len2: ShapeComponentT) -> bool:
    """
    Returns true only if len1 < len2.
    """
    if isinstance(len1, SizeParam) and not isinstance(len2, SizeParam):
        return False
    elif isinstance(len2, SizeParam) and not isinstance(len1, SizeParam):
        return True
    elif isinstance(len1, SizeParam) and isinstance(len2, SizeParam):
        return False
    else:
        assert isinstance(len1, INT_CLASSES)
        assert isinstance(len2, INT_CLASSES)
        return bool(len1 < len2)


def visualize_einsum_graph(
    graph: Mapping[EinsumGraphNode, frozenset[EinsumGraphNode]],
) -> None:
    """
    Opens an Xwindow with the einsum graph connectivity
    as visualized by DOT.
    """
    from pytools.graphviz import show_dot

    node_to_name: dict[EinsumGraphNode, str] = {}
    vng = UniqueNameGenerator({"node"})

    all_nodes = frozenset(graph)
    assert all(val <= all_nodes for val in graph.values())
    graphviz_node_lines: list[str] = []
    graphviz_edge_lines: list[str] = []

    for node in all_nodes:
        name = vng("node")
        node_to_name[node] = name
        graphviz_node_lines.append(
            f'  {name}[label="{node.label}", color={to_graphviz_color(node)}];'
        )

    for node, successors in graph.items():
        pred = node_to_name[node]
        for successor in successors:
            succ = node_to_name[successor]
            graphviz_edge_lines.append(f"  {pred} -> {succ};")

    dot_lines = [
        "digraph {",
        "node[style=filled]",
        *graphviz_node_lines,
        "",
        "",
        *graphviz_edge_lines,
        "}",
    ]
    dot_code = "\n".join(dot_lines)

    show_dot(dot_code)


@dataclass(frozen=True)
class InducedDirectedGraph:
    edges: np.ndarray[tuple[int, int], np.dtype[np.uint64]]
    c: np.ndarray[tuple[int], np.dtype[np.uint8]]
    iota_dtype: Map[int, np.dtype[Any]]
    iota_length: Map[int, IntegralT]

    def __post_init__(self) -> None:
        assert self.edges.ndim == 2
        assert self.c.ndim == 1
        assert self.edges.shape[1] == 2
        n_idg = self.c.shape[0]
        assert all(0 <= n < n_idg for n in self.iota_dtype)
        assert all(0 <= n < n_idg for n in self.iota_length)
        assert len(self.iota_dtype) == len(set(self.iota_dtype.values()))
        assert len(self.iota_length) == len(set(self.iota_length.values()))

    @cached_property
    def _preds_and_succs(
        self,
    ) -> tuple[Mapping[int, frozenset[int]], Mapping[int, frozenset[int]]]:
        # TODO: Switch to numpy / scipy.sparse if dealing with huge graphs.
        preds: dict[int, set[int]] = {i: set() for i in range(self.n_idg)}
        succs: dict[int, set[int]] = {i: set() for i in range(self.n_idg)}
        for from_node, to_node in self.edges:
            succs[from_node].add(to_node)
            preds[to_node].add(from_node)

        frozen_preds = Map((k, frozenset(v)) for k, v in preds.items())
        frozen_succs = Map((k, frozenset(v)) for k, v in succs.items())

        return frozen_preds, frozen_succs

    @property
    def n_idg(self) -> int:
        return self.c.shape[0]

    @memoize_method
    def preds(self, i: int) -> frozenset[int]:
        """
        Returns all the predecessors of *i*-th node.
        """
        preds, _ = self._preds_and_succs
        return preds[i]

    @memoize_method
    def succs(self, i: int) -> frozenset[int]:
        """
        Returns all the successors of *i*-th node.
        """
        _, succs = self._preds_and_succs
        return succs[i]

    def visualize(self, output_to: str | None = None) -> None:
        graphviz_colors = {
            1: "deepskyblue",
            2: "dodgerblue4",
            3: "darkseagreen1",
            4: "forestgreen",
            5: "darksalmon",
            6: "crimson",
            7: "navajowhite",
            8: "orange",
            9: "thistle",
        }

        dot_code = "digraph {\n"
        dot_code += "node[style=filled]\n"
        for i, color in enumerate(self.c):
            dot_code += f"{i} [color={graphviz_colors[int(color)]}]\n"
        for i, j in self.edges:
            dot_code += f"{i} -> {j}\n"
        dot_code += "}"

        from pytools.graphviz import show_dot

        show_dot(dot_code, output_to)

    @memoize_method
    def to_bliss_canonicalized_induced_dag(
        self,
    ) -> tuple[InducedDirectedGraph, np.ndarray[tuple[int], np.dtype[np.uint64]]]:
        import pybliss as bliss

        self_digraph = bliss.digraph_from_numpy(self.n_idg, self.edges, self.c)
        bliss_stats = bliss.Stats()
        perm = self_digraph.get_permutation_to_canonical_form(bliss_stats)
        assert isinstance(perm, np.ndarray)

        new_edges, new_colors = perm[self.edges], self.c[perm]
        new_iota_dtype = Map(
            (perm[i], dtype) for i, dtype in self.iota_dtype.items()
        )
        new_iota_length = Map(
            (perm[i], dtype) for i, dtype in self.iota_length.items()
        )
        return (
            InducedDirectedGraph(
                new_edges, new_colors, new_iota_dtype, new_iota_length
            ),
            perm,
        )


def to_induced_dag(
    einsum: BatchedEinsum,
) -> tuple[
    InducedDirectedGraph,
    Mapping[int, str],
]:
    """
    Returns the induced directed graph for a batched einsum *einsum*.
    """
    # Implementation follows Appendix 'A' of <PAPER NAME>.

    # See Defn 16.
    n_dim = max([einsum.ndim, *[len(idx_set) for idx_set in einsum.in_idx_sets]])
    all_dims = frozenset(DimNode(i) for i in range(n_dim))

    # See Defn. 17
    input_accesses = frozenset(
        {
            InputAccessNode(i, j, idx, d)
            for i in range(einsum.b)
            for j, idx_set in enumerate(einsum.in_idx_sets)
            for d, idx in enumerate(idx_set)
        }
    )

    # See Defn. 18
    output_accesses = frozenset(
        {OutputAccessNode(idx, d) for d, idx in enumerate(einsum.out_idx_set)}
    )

    # See Defn 19
    dtype_nodes = frozenset(
        {DtypeNode(dtype) for dtype in einsum.arg_to_dtype.values()}
    )

    # See Defn 20
    axis_lengths = frozenset(
        {
            AxisLengthNode(axis_len)
            for axis_len in einsum.index_to_dim_length.values()
        }
    )

    # See Defn 9
    all_indices = frozenset({IndexNode(idx) for idx in einsum.all_indices})

    # See Defn 12
    all_args = frozenset({ArgNode(arg_name) for arg_name in einsum.all_args})

    n_arg = len(all_args)
    n_index = len(all_indices)
    n_length = len(axis_lengths)
    n_dtypes = len(dtype_nodes)
    n_access_in = len(input_accesses)
    n_access_out = len(output_accesses)

    # {{{ See Section A.1, define the iota mappings

    iota_arg = frozenbidict(
        (i, arg_node)
        for i, arg_node in enumerate(
            sorted(all_args, key=lambda arg_node: arg_node.name),
        )
    )
    iota_index = frozenbidict(
        (i, index_node)
        for i, index_node in enumerate(
            sorted(all_indices, key=lambda index_node: index_node.name),
            start=n_arg,
        )
    )

    iota_access_in = frozenbidict(
        (i, access_in_node)
        for i, access_in_node in enumerate(
            sorted(
                input_accesses,
                key=lambda access_in_node: (
                    access_in_node.i,
                    access_in_node.j,
                    access_in_node.d,
                    access_in_node.index,
                ),
            ),
            start=n_arg + n_index,
        )
    )

    iota_access_out = frozenbidict(
        (i, access_out_node)
        for i, access_out_node in enumerate(
            sorted(
                output_accesses,
                key=lambda access_out_node: (
                    access_out_node.d,
                    access_out_node.index,
                ),
            ),
            start=n_arg + n_index + n_access_in,
        )
    )

    iota_output = frozenbidict(
        (
            (i_result + n_arg + n_index + n_access_in + n_access_out),
            IResultNode(i_result),
        )
        for i_result in range(einsum.b)
    )

    iota_pos = frozenbidict(
        (
            (i_pos + n_arg + n_index + n_access_in + n_access_out + einsum.b),
            IPositionNode(i_pos),
        )
        for i_pos in range(einsum.n)
    )

    iota_dtype = frozenbidict(
        (i, dtype_node)
        for i, dtype_node in enumerate(
            sorted(
                dtype_nodes,
                key=lambda dtype_node: str(dtype_node.dtype),
            ),
            start=n_arg + n_index + n_access_in + n_access_out + einsum.b + einsum.n,
        )
    )

    iota_length = frozenbidict(
        (i, length_node)
        for i, length_node in enumerate(
            sorted(
                axis_lengths,
                key=lambda length_node: str(length_node),
            ),
            start=n_arg
            + n_index
            + n_access_in
            + n_access_out
            + einsum.b
            + einsum.n
            + n_dtypes,
        )
    )

    iota_dim = frozenbidict(
        (i, dim_node)
        for i, dim_node in enumerate(
            sorted(
                all_dims,
                key=lambda dim_node: str(dim_node),
            ),
            start=n_arg
            + n_index
            + n_access_in
            + n_access_out
            + einsum.b
            + einsum.n
            + n_dtypes
            + n_length,
        )
    )

    iotas: frozenbidict[int, EinsumGraphNode] = frozenbidict(
        {
            **iota_arg,
            **iota_index,
            **iota_access_in,
            **iota_access_out,
            **iota_output,
            **iota_pos,
            **iota_dtype,
            **iota_length,
            **iota_dim,
        }
    )
    assert (
        len(iotas)
        == n_arg
        + n_index
        + n_access_in
        + n_access_out
        + einsum.b
        + einsum.n
        + n_dtypes
        + n_length
        + n_dim
    )

    # }}}

    # {{{ record the edges (See (7) -- (17))

    e_access_in_to_arg = frozenset(
        {
            (InputAccessNode(i, j, idx, i_idx), ArgNode(arg.name))
            for i in range(einsum.b)
            for j, (idx_set, arg) in enumerate(
                zip(einsum.in_idx_sets, einsum.args[i], strict=True)
            )
            for i_idx, idx in enumerate(idx_set)
        }
    )

    e_arg_pos_to_access_in = frozenset(
        {
            (IPositionNode(access_in_node.j), access_in_node)
            for access_in_node in input_accesses
        }
    )
    e_out_to_access_in = frozenset(
        {
            (IResultNode(access_in_node.i), access_in_node)
            for access_in_node in input_accesses
        }
    )
    e_idx_to_access_in = frozenset(
        {
            (IndexNode(access_in_node.index), access_in_node)
            for access_in_node in input_accesses
        }
    )
    e_dim_to_access_in = frozenset(
        {
            (DimNode(access_in_node.d), access_in_node)
            for access_in_node in input_accesses
        }
    )
    e_idx_to_access_out = frozenset(
        {(IndexNode(access_out.index), access_out) for access_out in output_accesses}
    )
    e_dim_to_access_out = frozenset(
        {(DimNode(access_out.d), access_out) for access_out in output_accesses}
    )

    e_length = frozenset(
        {
            (AxisLengthNode(axis_len), IndexNode(idx))
            for idx, axis_len in einsum.index_to_dim_length.items()
        }
    )
    e_dtype = frozenset(
        {
            (DtypeNode(dtype), ArgNode(arg_name))
            for arg_name, dtype in einsum.arg_to_dtype.items()
        }
    )

    e_length_ranks_tmp = set()
    for length_i in axis_lengths:
        for length_j in axis_lengths:
            if _axis_len_le(length_i.length, length_j.length):
                e_length_ranks_tmp.add((length_i, length_j))
    e_length_ranks = frozenset(e_length_ranks_tmp)
    del e_length_ranks_tmp

    e_dtype_ranks_tmp = set()
    for dtype_i in dtype_nodes:
        for dtype_j in dtype_nodes:
            if str(dtype_i.dtype) < str(dtype_j.dtype):
                e_dtype_ranks_tmp.add((dtype_i, dtype_j))
    e_dtype_ranks = frozenset(e_dtype_ranks_tmp)
    del e_dtype_ranks_tmp

    e_dim_ranks_tmp = set()
    for dim_i in all_dims:
        for dim_j in all_dims:
            if dim_i.dim < dim_j.dim:
                e_dim_ranks_tmp.add((dim_i, dim_j))
    e_dim_ranks = frozenset(e_dim_ranks_tmp)
    del e_dim_ranks_tmp

    all_edges: frozenset[tuple[EinsumGraphNode, EinsumGraphNode]] = frozenset(
        {
            *e_access_in_to_arg,
            *e_arg_pos_to_access_in,
            *e_out_to_access_in,
            *e_idx_to_access_in,
            *e_arg_pos_to_access_in,
            *e_dim_to_access_in,
            *e_idx_to_access_out,
            *e_dim_to_access_out,
            *e_length,
            *e_dtype,
            *e_length_ranks,
            *e_dtype_ranks,
            *e_dim_ranks,
        }
    )

    # }}}

    if 0:
        graphviz_graph_tmp: dict[EinsumGraphNode, set[EinsumGraphNode]] = {
            node: set() for node in iotas.values()
        }
        for from_node, to_node in all_edges:
            graphviz_graph_tmp[from_node].add(to_node)
        graphviz_graph = Map(
            {
                from_node: frozenset(to_nodes)
                for from_node, to_nodes in graphviz_graph_tmp.items()
            }
        )
        del graphviz_graph_tmp

        visualize_einsum_graph(graphviz_graph)
        _ = 1 / 0

    # {{{ Instantiate the induced directed graph

    n_idg = len(iotas)
    bliss_colors = np.empty(n_idg, np.uint8)

    for i, node in iotas.items():
        bliss_colors[i] = to_bliss_color(node)

    bliss_edges = np.array(
        [
            [iotas.inv[from_node], iotas.inv[to_node]]
            for (from_node, to_node) in all_edges
        ],
        dtype=np.uint64,
    )

    # }}}

    return (
        InducedDirectedGraph(
            bliss_edges,
            bliss_colors,
            Map((i, dtype_node.dtype) for i, dtype_node in iota_dtype.items()),
            Map(
                (i, axis_len.length)
                for i, axis_len in iota_length.items()
                if isinstance(axis_len.length, INT_CLASSES)
            ),
        ),
        Map(
            {
                **{i: arg_node.name for i, arg_node in iota_arg.items()},
                **{i: index_node.name for i, index_node in iota_index.items()},
                **{
                    i: axis_len.length.name
                    for i, axis_len in iota_length.items()
                    if isinstance(axis_len.length, SizeParam)
                },
                **{
                    i: "_fe_out" if i_output < 1 else f"_fe_out_{i_output - 1}"
                    for i_output, i in enumerate(iota_output)
                },
            }
        ),
    )


def from_induced_dag(
    induced_dag: InducedDirectedGraph,
) -> tuple[BatchedEinsum, Mapping[int, str]]:
    # Follows the reconstruction from Appendix A.2

    # Equation (20)
    v_arg = frozenset(np.nonzero(induced_dag.c == 1)[0])
    v_index = frozenset(np.nonzero(induced_dag.c == 2)[0])
    v_access_in = frozenset(np.nonzero(induced_dag.c == 3)[0])
    v_access_out = frozenset(np.nonzero(induced_dag.c == 4)[0])
    v_output = frozenset(np.nonzero(induced_dag.c == 5)[0])
    v_arg_pos = frozenset(np.nonzero(induced_dag.c == 6)[0])
    v_dtype = frozenset(np.nonzero(induced_dag.c == 7)[0])
    v_length = frozenset(np.nonzero(induced_dag.c == 8)[0])
    v_dims = frozenset(np.nonzero(induced_dag.c == 9)[0])

    # {{{ Check the constaints (C1) -- (C30)

    # C1.
    assert np.all(np.logical_and(induced_dag.c >= 1, induced_dag.c <= 9))
    # C2.
    assert all(len(induced_dag.succs(i)) == 0 for i in v_arg)
    # C3.
    v_access_in_union_dtype = v_access_in | v_dtype
    assert all(induced_dag.preds(i) <= v_access_in_union_dtype for i in v_arg)
    # C4.
    assert all(len(induced_dag.preds(i) & v_dtype) == 1 for i in v_arg)
    # C5.
    assert all(induced_dag.preds(i) <= v_dtype for i in v_dtype)
    # C6.
    v_arg_u_v_dtype = v_arg | v_dtype
    assert all(induced_dag.succs(i) <= v_arg_u_v_dtype for i in v_dtype)
    # C7.
    assert all(induced_dag.succs(i) <= v_arg for i in v_access_in)
    # C8.
    assert all(
        (len(induced_dag.preds(i) & v_arg_pos) == 1)
        and (len(induced_dag.preds(i) & v_index) == 1)
        and (len(induced_dag.preds(i) & v_output) == 1)
        and (len(induced_dag.preds(i) & v_dims) == 1)
        and (len(induced_dag.preds(i)) == 4)
        for i in v_access_in
    )
    # C9.
    assert all(induced_dag.succs(i) <= v_access_in for i in v_output)
    # C10.
    assert all(len(induced_dag.preds(i)) == 0 for i in v_output)
    # C11.
    assert all(len(induced_dag.succs(i)) == 0 for i in v_access_out)
    # C12.
    assert all(
        (len(induced_dag.preds(i) & v_index) == 1)
        and (len(induced_dag.preds(i) & v_dims) == 1)
        and (len(induced_dag.preds(i)) == 2)
        for i in v_access_out
    )
    # C13.
    v_access_in_union_out = v_access_in | v_access_out
    assert all(induced_dag.succs(i) <= v_access_in_union_out for i in v_index)

    # C14.
    assert all(
        (len(induced_dag.preds(i) & v_length) == 1)
        and (len(induced_dag.preds(i)) == 1)
        for i in v_index
    )

    # C15.
    assert all(induced_dag.preds(i) <= v_length for i in v_length)
    # C16.
    v_index_union_length = v_index | v_length
    assert all(induced_dag.succs(i) <= v_index_union_length for i in v_length)

    # C17.
    assert all(induced_dag.preds(i) <= v_dims for i in v_dims)

    # C18.
    v_access_in_union_out_union_dim = v_access_in | v_access_out | v_dims
    assert all(
        induced_dag.succs(i) <= v_access_in_union_out_union_dim for i in v_dims
    )

    # C19.
    assert all(len(induced_dag.preds(i)) == 0 for i in v_arg_pos)

    # C20.
    assert all(induced_dag.succs(i) <= v_access_in for i in v_arg_pos)

    # C21. (FIXME). It is incorrect in the paper.

    # C22. (FIXME). It is incorrect in the paper.

    # C23. (FIXME). It is incorrect in the paper.

    # C24. (FIXME). Not sure what the paper says.

    # C25.
    for n in v_access_out:
        (i,) = induced_dag.preds(n) & v_index
        assert len(induced_dag.succs(i) & v_access_in) > 0

    # C26. (FIXME)

    # C27. (FIXME)

    # C28. (FIXME)

    # C29. (FIXME)

    # C30. (FIXME)

    # }}}

    # Step 1.
    def arg_name_gen(i: int) -> str:
        return f"arg_{i}"

    def idx_name_gen(i: int) -> str:
        assert i < 26
        return chr((ord("i") - ord("a") + i) % 26 + ord("a"))

    # Step 2.
    iota_inferred_arg_pos = Map(
        (arg_pos, i) for i, arg_pos in enumerate(sorted(v_arg_pos))
    )

    # Step 3.
    iota_inferred_output = Map(
        (output, i) for i, output in enumerate(sorted(v_output))
    )

    # Step 4.
    iota_inferred_index = Map(
        (i, idx_name_gen(i_idx)) for i_idx, i in enumerate(sorted(v_index))
    )

    # Step 5.
    iota_inferred_arg = Map(
        (i, arg_name_gen(i_arg)) for i_arg, i in enumerate(sorted(v_arg))
    )

    # Step 6.
    iota_inferred_dim = Map(
        (idim, len(induced_dag.preds(idim) & v_dims)) for idim in v_dims
    )
    assert sorted(iota_inferred_dim.values()) == list(range(len(v_dims)))

    # Step 7.
    i_inferred_dim_to_out_index_tmp = {}
    for i in v_access_out:
        (i_dim,) = induced_dag.preds(i) & v_dims
        (i_idx,) = induced_dag.preds(i) & v_index
        i_inferred_dim_to_out_index_tmp[iota_inferred_dim[i_dim]] = (
            iota_inferred_index[i_idx]
        )

    out_index_set = tuple(
        i_inferred_dim_to_out_index_tmp[i] for i in range(len(v_access_out))
    )
    del i_inferred_dim_to_out_index_tmp

    # Step 8, 9.
    i_inferred_pos_to_idx_set_tmp: dict[int, tuple[str, ...]] = {}
    i = next(iter(v_output))
    for j in v_arg_pos:
        i_accesses_in = induced_dag.succs(i) & induced_dag.succs(j) & v_access_in
        i_inferred_dim_to_idx: dict[int, str] = {}
        for i_access_in in i_accesses_in:
            (i_dim,) = induced_dag.preds(i_access_in) & v_dims
            (i_idx,) = induced_dag.preds(i_access_in) & v_index
            i_inferred_dim_to_idx[iota_inferred_dim[i_dim]] = iota_inferred_index[
                i_idx
            ]

        i_inferred_pos_to_idx_set_tmp[iota_inferred_arg_pos[j]] = tuple(
            i_inferred_dim_to_idx[i_dim] for i_dim in range(len(i_accesses_in))
        )

    in_index_sets = tuple(
        i_inferred_pos_to_idx_set_tmp[i] for i in range(len(v_arg_pos))
    )
    del i_inferred_pos_to_idx_set_tmp

    # Step 10.
    inferred_resultxpos_to_arg_tmp: dict[tuple[int, int], str] = {}

    for i in v_output:
        for j in v_arg_pos:
            i_access_ins = induced_dag.succs(i) & induced_dag.succs(j)
            (i_arg,) = reduce(
                frozenset.union,
                [
                    induced_dag.succs(i_access_in) & v_arg
                    for i_access_in in i_access_ins
                ],
                cast("frozenset[int]", frozenset()),
            )
            inferred_resultxpos_to_arg_tmp[
                iota_inferred_output[i], iota_inferred_arg_pos[j]
            ] = iota_inferred_arg[i_arg]

    arg_names = tuple(
        tuple(inferred_resultxpos_to_arg_tmp[i, j] for j in range(len(v_arg_pos)))
        for i in range(len(v_output))
    )
    del inferred_resultxpos_to_arg_tmp

    index_to_length_pred = Map(
        (j, i) for i in v_length for j in induced_dag.succs(i) & v_index
    )
    ilength_to_size_param_name = Map(
        (i, iota_inferred_index[j].upper())
        for i in v_length
        if i not in induced_dag.iota_length
        for j in induced_dag.succs(i) & v_index
    )

    inferred_index_to_length: Map[str, ShapeComponentT] = Map(
        (
            iota_inferred_index[i],
            induced_dag.iota_length.get(
                index_to_length_pred[i],
                SizeParam(iota_inferred_index[i].upper()),
            ),
        )
        for i in v_index
    )

    inferred_arg_to_dtype_tmp: dict[str, np.dtype[Any]] = {}
    for i in v_arg:
        (i_dtype,) = induced_dag.preds(i) & v_dtype
        inferred_arg_to_dtype_tmp[iota_inferred_arg[i]] = induced_dag.iota_dtype[
            i_dtype
        ]
    inferred_arg_to_dtype = Map(inferred_arg_to_dtype_tmp)
    del inferred_arg_to_dtype_tmp

    # Step 11.
    from feinsum.make_einsum import array, batched_einsum

    return batched_einsum(
        f"{','.join("".join(idx_set) for idx_set in in_index_sets)} -> "
        f"{''.join(out_index_set)}",
        [
            [
                array(
                    arg_name,
                    [inferred_index_to_length[idx] for idx in idx_set],
                    inferred_arg_to_dtype[arg_name],
                )
                for idx_set, arg_name in zip(in_index_sets, arg_row, strict=True)
            ]
            for arg_row in arg_names
        ],
    ), Map(
        {
            **iota_inferred_index,
            **iota_inferred_arg,
            **ilength_to_size_param_name,
            **{
                i: "_fe_out" if i_output < 1 else f"_fe_out_{i_output - 1}"
                for i, i_output in iota_inferred_output.items()
            },
        }
    )


def _get_canonicalized_einsum_with_subst_mapping(
    einsum: BatchedEinsum,
) -> tuple[BatchedEinsum, frozenbidict[str, str]]:
    """
    Returns a tuple of the form ``(canonicalized_einsum, subst_map)`` where
    *canonicalized_einsum* is an instance of :class:`BatchedEinsum` which is the
    canonicalized version of *einsum* and *subst_map* is the mapping from entities
    of *einsum* to the variables in `*canonicalized_einsum*.
    """
    induced_dag, iota_src = to_induced_dag(einsum)
    induced_dag, perm = induced_dag.to_bliss_canonicalized_induced_dag()
    einsum, iota_dst = from_induced_dag(induced_dag)
    assert len(iota_src) == len(iota_dst)
    return einsum, frozenbidict(
        (name_in_src, iota_dst[perm[i]]) for i, name_in_src in iota_src.items()
    )


def canonicalize_einsum(einsum: BatchedEinsum) -> BatchedEinsum:
    """
    Returns a canonicalized form of *einsum*.

    .. note::

        - Refer to (TODO PAPER) for a definition of isomorphism among
          fused einsums.
    """
    return _get_canonicalized_einsum_with_subst_mapping(einsum)[0]


def get_substitution_mapping_between_isomorphic_batched_einsums(
    batched_einsum_from: BatchedEinsum, batched_einsum_to: BatchedEinsum
) -> Mapping[str, str]:
    """
    Returns the isomorphism mapping from the entities of *batched_einsum_from* to
    the entities of *batched_einsum_to*.

    .. note::

        Raises a :class:`ValueError` if the two batched einsums are not isomorphic.
    """
    canon_batched_einsum_from, subst_map_from = (
        _get_canonicalized_einsum_with_subst_mapping(batched_einsum_from)
    )
    canon_batched_einsum_to, subst_map_to = (
        _get_canonicalized_einsum_with_subst_mapping(batched_einsum_to)
    )

    if canon_batched_einsum_from != canon_batched_einsum_to:
        raise ValueError("Einsums are not isomorphic.")

    return Map(
        {
            var_in_from_einsum: subst_map_to.inv[var_in_canon_einsum]
            for var_in_from_einsum, var_in_canon_einsum in subst_map_from.items()
        }
    )


# vim: foldmethod=marker
