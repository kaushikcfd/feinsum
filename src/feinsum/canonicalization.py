"""
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
import numpy as np
import functools
import pynauty as nauty

from typing import (Dict, Set, Mapping, List, Any, Union, Sequence,
                    Tuple, FrozenSet)
from feinsum.einsum import (FusedEinsum, ShapeT, ShapeComponentT,
                            EinsumAxisAccess, FreeAxis, IntegralT, SizeParam,
                            INT_CLASSES)
from feinsum.make_einsum import fused_einsum
from immutables import Map
from dataclasses import dataclass
from more_itertools import zip_equal as zip
from pytools import UniqueNameGenerator
from bidict import frozenbidict


# {{{ private interface to build an einsum graph

class EinsumGraphNode(abc.ABC):
    """
    A node in the einsum graph to canonicalize it.
    """
    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass


@dataclass(frozen=True)
class IndexNode(EinsumGraphNode):
    """
    A node in the einsum graph corresponding to an index in the einstein
    summation expression.
    """
    _name: str

    @property
    def name(self) -> str:
        return self._name


@dataclass(frozen=True)
class AxisNode(EinsumGraphNode):
    """
    A node in the einsum graph corresponding to the axis of the array accessed.
    """
    axis: int

    @property
    def name(self) -> str:
        return str(self.axis)


@dataclass(frozen=True)
class AccessNode(EinsumGraphNode):
    """
    Represents an access into an array i.e. the index accessing it along with
    the array's axis it is accessed into.
    """
    index: str
    axis: int

    @property
    def name(self) -> str:
        return f"({self.index}, {self.axis})"


@dataclass(frozen=True)
class ArrayNode(EinsumGraphNode):
    """
    An array (could be either input  or output) seen in a
    :class:`FusedEinsum`.
    """
    _name: str

    @property
    def name(self) -> str:
        return self._name


@dataclass(frozen=True)
class LiteralNode(EinsumGraphNode):
    dim: IntegralT

    @property
    def name(self) -> str:
        return str(self.dim)


@dataclass(frozen=True)
class DtypeNode(EinsumGraphNode):
    dtype: np.dtype[Any]

    @property
    def name(self) -> str:
        return self.dtype.name


@dataclass(frozen=True)
class SizeParamNode(EinsumGraphNode):
    _name: str

    @property
    def name(self) -> str:
        return self._name

# }}}


def visualize_einsum_graph(graph: Mapping[EinsumGraphNode,
                                          Set[EinsumGraphNode]]
                           ) -> None:
    """
    Opens an Xwindow with the einsum graph connectivity
    as visualized by DOT.
    """
    from pymbolic.imperative.utils import show_dot
    node_to_name: Dict[EinsumGraphNode, str] = {}
    vng = UniqueNameGenerator({"node"})

    all_nodes = frozenset(graph)
    assert all(val <= all_nodes for val in graph.values())
    graphviz_node_lines: List[str] = []
    graphviz_edge_lines: List[str] = []

    for node in all_nodes:
        name = vng("node")
        node_to_name[node] = name
        graphviz_node_lines.append(f'  {name}[label="{node.name}"];')

    for node, successors in graph.items():
        pred = node_to_name[node]
        for successor in successors:
            succ = node_to_name[successor]
            graphviz_edge_lines.append(f"  {pred} -> {succ};")

    dot_lines = (["digraph {"] + graphviz_node_lines
                 + ["", ""] + graphviz_edge_lines + ["}"])
    dot_code = "\n".join(dot_lines)

    show_dot(dot_code)


def _get_use_matrix_col_permutation(einsum: FusedEinsum,
                                    sorted_index_names: Sequence[str]
                                    ) -> Tuple[int, ...]:
    acc_descr_to_rank = {
        acc_descr: sorted_index_names.index(name)
        for acc_descr, name in einsum.index_names.items()}

    a = np.empty(len(einsum.access_descriptors), dtype=object)

    for i, acc_descrs in enumerate(einsum.access_descriptors):
        a[i] = tuple(acc_descr_to_rank[acc_descr] for acc_descr in acc_descrs)

    return tuple(np.argsort(a))


def _group_same_access_descriptors(einsum: FusedEinsum) -> FusedEinsum:

    new_arg_shapes: List[ShapeT] = []
    new_acc_descrs: List[Tuple[EinsumAxisAccess, ...]] = []
    acc_descrs_to_i_new_col: Dict[Tuple[EinsumAxisAccess, ...], int] = {}
    n_unique = 0

    for arg_shape, acc_descrs in zip(einsum.arg_shapes, einsum.access_descriptors):

        try:
            acc_descrs_to_i_new_col[acc_descrs]
        except KeyError:
            new_arg_shapes.append(arg_shape)
            new_acc_descrs.append(acc_descrs)
            acc_descrs_to_i_new_col[acc_descrs] = n_unique
            n_unique += 1

    assert set(acc_descrs_to_i_new_col) == set(einsum.access_descriptors)
    assert n_unique == len(acc_descrs_to_i_new_col)
    assert n_unique <= len(einsum.access_descriptors)

    new_use_matrix: List[List[Set[str]]] = [
        [
            set()
            for icol in range(n_unique)
        ]
        for irow in range(einsum.noutputs)
    ]

    for irow, use_row in enumerate(einsum.use_matrix):
        for acc_descrs, uses in zip(einsum.access_descriptors, use_row):
            icol = acc_descrs_to_i_new_col[acc_descrs]
            new_use_matrix[irow][icol].update(uses)

    return FusedEinsum(
        tuple(new_arg_shapes),
        einsum.value_to_dtype,
        tuple(new_acc_descrs),
        tuple(tuple(frozenset(uses)
                    for uses in use_row)
              for use_row in new_use_matrix),
        einsum.index_names)


def get_einsum_dag(einsum: FusedEinsum) -> Map[EinsumGraphNode,
                                               FrozenSet[EinsumGraphNode]]:

    output_names = ["_fe_out"] + [f"_fe_out_{i}"
                                  for i in range(einsum.noutputs-1)]
    einsum_dag: Dict[EinsumGraphNode, Set[EinsumGraphNode]] = {}

    # {{{ compute acc_descr_to_node, array_to_node

    acc_descr_to_node: Dict[EinsumAxisAccess, IndexNode] = {}
    access_to_node: Dict[Tuple[EinsumAxisAccess, int], AccessNode] = {}
    array_to_node: Dict[str, ArrayNode] = {}
    axis_to_node: Dict[int, AxisNode] = {}
    shape_dim_to_node: Dict[ShapeComponentT, Union[LiteralNode, SizeParamNode]] = {}
    dtype_to_node: Dict[np.dtype[Any], DtypeNode] = {}

    for acc_descrs in einsum.access_descriptors:
        for acc_descr in acc_descrs:
            acc_descr_to_node[acc_descr] = IndexNode(einsum.index_names[acc_descr])

    for ary_name in set(einsum.value_to_dtype) | set(output_names):
        array_to_node[ary_name] = ArrayNode(ary_name)

    for arg_shape in einsum.arg_shapes:
        for dim in arg_shape:
            if isinstance(dim, INT_CLASSES):
                shape_dim_to_node[dim] = LiteralNode(dim)
            else:
                assert isinstance(dim, SizeParam)
                shape_dim_to_node[dim] = SizeParamNode(dim.name)

    for idim in range(max((len(arg_shape)
                           for arg_shape in einsum.arg_shapes),
                          default=einsum.ndim)):
        axis_to_node[idim] = AxisNode(idim)

    for acc_descrs in einsum.access_descriptors:
        for idim, acc_descr in enumerate(acc_descrs):
            access_to_node[(acc_descr, idim)] = AccessNode(
                einsum.index_names[acc_descr], idim)

    for idim in range(einsum.ndim):
        access_to_node[(FreeAxis(idim), idim)] = AccessNode(
            einsum.index_names[FreeAxis(idim)], idim)

    for dtype in einsum.value_to_dtype.values():
        dtype_to_node[dtype] = DtypeNode(dtype)

    all_einsum_graph_nodes = (frozenset(array_to_node.values())
                              | frozenset(acc_descr_to_node.values())
                              | frozenset(shape_dim_to_node.values())
                              | frozenset(dtype_to_node.values())
                              | frozenset(axis_to_node.values())
                              | frozenset(access_to_node.values())
                              )
    # }}}

    # Initialize the DAG with EinsumGraphNodes
    for node in all_einsum_graph_nodes:
        einsum_dag[node] = set()

    # {{{ adding the edges

    # Add edges from values to outputs
    for output_name, use_row in zip(output_names, einsum.use_matrix):
        output_node = array_to_node[output_name]
        for uses in use_row:
            for use in uses:
                use_node = array_to_node[use]
                einsum_dag[use_node].add(output_node)

    # Add edges between the access axes ordering
    for idim in range(max((len(arg_shape)
                           for arg_shape in einsum.arg_shapes),
                          default=einsum.ndim)-1):
        node = axis_to_node[idim]
        succ_node = axis_to_node[idim+1]
        einsum_dag[node].add(succ_node)

    # add edges due to indexing the inputs
    for use_row in einsum.use_matrix:
        for uses, acc_descrs in zip(use_row, einsum.access_descriptors):
            for ary in uses:
                ary_node = array_to_node[ary]
                for idim, acc_descr in enumerate(acc_descrs):
                    access_node = access_to_node[(acc_descr, idim)]
                    einsum_dag[access_node].add(ary_node)

    # add edges due to indexing the outputs
    for output_name in output_names:
        ary_node = array_to_node[output_name]
        for idim in range(einsum.ndim):
            access_node = access_to_node[(FreeAxis(idim), idim)]
            einsum_dag[access_node].add(ary_node)

    # {{{ add edges defining an access

    for acc_descrs in einsum.access_descriptors:
        for idim, acc_descr in enumerate(acc_descrs):
            access_node = access_to_node[(acc_descr, idim)]
            axis_node = axis_to_node[idim]
            index_node = acc_descr_to_node[acc_descr]
            einsum_dag[axis_node].add(access_node)
            einsum_dag[index_node].add(access_node)

    for idim in range(einsum.ndim):
        acc_descr = FreeAxis(idim)
        access_node = access_to_node[(acc_descr, idim)]
        axis_node = axis_to_node[idim]
        index_node = acc_descr_to_node[acc_descr]
        einsum_dag[axis_node].add(access_node)
        einsum_dag[index_node].add(access_node)

    # }}}

    # add edges coming from index_to_length
    for acc_descr, dim in einsum.index_to_dim_length().items():
        einsum_dag[shape_dim_to_node[dim]].add(acc_descr_to_node[acc_descr])

    # add edges coming from dtypes
    for ary_name, dtype in einsum.value_to_dtype.items():
        einsum_dag[dtype_to_node[dtype]].add(array_to_node[ary_name])

    for output_name, use_row in zip(output_names, einsum.use_matrix):
        use_dtypes: FrozenSet[np.dtype[Any]] = functools.reduce(
            frozenset.union,
            (frozenset(einsum.value_to_dtype[use] for use in uses)
             for uses in use_row),
            frozenset())
        out_dtype = np.find_common_type(list(use_dtypes), [])
        einsum_dag[dtype_to_node[out_dtype]].add(array_to_node[output_name])

    all_dtypes = sorted(set(einsum.value_to_dtype.values()),
                        key=lambda x: x.itemsize)

    for prev_dtype, dtype in zip(all_dtypes[:-1], all_dtypes[1:]):
        if prev_dtype.itemsize != dtype.itemsize:
            einsum_dag[dtype_to_node[prev_dtype]].add(dtype_to_node[dtype])

    del all_dtypes

    all_size_param_nodes = {node
                            for dim, node in shape_dim_to_node.items()
                            if isinstance(dim, SizeParam)}
    literal_dims = sorted({dim
                           for dim, node in shape_dim_to_node.items()
                           if isinstance(dim, INT_CLASSES)})
    for prev_dim, dim in zip(literal_dims[:-1], literal_dims[1:]):
        einsum_dag[shape_dim_to_node[prev_dim]].add(shape_dim_to_node[dim])

    for dim in literal_dims:
        for size_param_node in all_size_param_nodes:
            einsum_dag[shape_dim_to_node[dim]].add(size_param_node)

    # }}}

    return Map({k: frozenset(v)
                for k, v in einsum_dag.items()})


def _get_canonicalized_einsum_with_subst_mapping(
        einsum: FusedEinsum) -> Tuple[FusedEinsum, frozenbidict[str, str]]:
    """
    Returns a tuple of the form ``(canonicalized_einsum, subst_map)`` where
    *canonicalized_einsum* is an instance of :class:`BatchedEinsum` which is the
    canonicalized version of *einsum* and *subst_map* is the mapping from variables
    in *einsum* to the variables in `*canonicalized_einsum*.
    """

    # collect all the uses with same desciptors together.
    einsum = _group_same_access_descriptors(einsum)

    output_names = ["_fe_out"] + [f"_fe_out_{i}"
                                  for i in range(einsum.noutputs-1)]

    einsum_dag = get_einsum_dag(einsum)

    # Now number the nodes to integers and get the canonical mapping. Done?
    # Then `pynauty` should give
    node_to_idx = {node: i
                   for i,  node in enumerate(einsum_dag.keys())}

    g = nauty.Graph(number_of_vertices=len(node_to_idx), directed=True,
                    adjacency_dict={node_to_idx[k]: {node_to_idx[v] for v in vs}
                                    for k, vs in einsum_dag.items()},
                    )

    reindex_map = {lbl: i
                   for i, lbl in enumerate(nauty.canon_label(g))}
    # recompute node_to_idx as per the mapping emitted by nauty.
    node_to_idx = {node: reindex_map[idx]
                   for node, idx in node_to_idx.items()}
    del reindex_map

    idx_to_new_idx: Dict[str, str] = {}
    input_ary_name_to_new_ary_name: Dict[str, str] = {}

    sorted_index_names: List[str] = []
    sorted_input_arys: List[str] = []
    sorted_output_arys: List[str] = []

    for node in sorted(einsum_dag.keys(), key=lambda node_: node_to_idx[node_]):
        if isinstance(node, IndexNode):
            sorted_index_names.append(node.name)
        elif isinstance(node, ArrayNode) and node.name in output_names:
            sorted_output_arys.append(node.name)
        elif isinstance(node, ArrayNode):
            sorted_input_arys.append(node.name)
        elif isinstance(node, (SizeParamNode, DtypeNode, LiteralNode,
                               AxisNode, AccessNode)):
            pass
        else:
            raise NotImplementedError(type(node))

    for i, old_idx in enumerate(sorted_index_names):
        idx_to_new_idx[old_idx] = chr(97+i)

    for i, old_input_ary in enumerate(sorted_input_arys):
        input_ary_name_to_new_ary_name[old_input_ary] = f"arg_{i}"

    use_matrix_col_permutation = _get_use_matrix_col_permutation(
        einsum, sorted_index_names)
    # TODO: O(m.n), but who cares, right?
    use_matrix_row_permutation = [output_names.index(name)
                                  for name in sorted_output_arys]

    new_use_matrix = tuple([
        tuple([
            frozenset(input_ary_name_to_new_ary_name[use]
                      for use in einsum.use_matrix[irow][icol])
            for icol in use_matrix_col_permutation
        ])
        for irow in use_matrix_row_permutation
    ])

    new_index_expr = "".join(idx if idx in [",", "-", ">"]
                             else idx_to_new_idx[idx]
                             for idx in einsum.get_subscripts())
    input_subscripts, output_subscript = new_index_expr.split("->")
    input_subscript_list = input_subscripts.split(",")
    new_input_subscripts = ",".join(input_subscript_list[icol]
                                    for icol in use_matrix_col_permutation)

    substitution_mapping = frozenbidict({
        **idx_to_new_idx,
        **input_ary_name_to_new_ary_name,
        **{old_output_name: output_names[use_matrix_row_permutation[irow]]
           for irow, old_output_name in enumerate(output_names)},
    })

    return fused_einsum(
        f"{new_input_subscripts}->{output_subscript}",
        operand_shapes=[[np.inf if isinstance(d, SizeParam) else d
                         for d in einsum.arg_shapes[icol]]
                        for icol in use_matrix_col_permutation],
        value_to_dtype={input_ary_name_to_new_ary_name[k]: v
                        for k, v in einsum.value_to_dtype.items()},
        use_matrix=new_use_matrix), substitution_mapping


def canonicalize_einsum(einsum: FusedEinsum) -> FusedEinsum:
    """
    Returns a canonicalized form of *einsum*.

    .. note::

        - Refer to (TODO PAPER) for a definition of isomorphism among
          fused einsums.
    """
    return _get_canonicalized_einsum_with_subst_mapping(einsum)[0]


def get_substitution_mapping_between_isomorphic_batched_einsums(
        batched_einsum_from, batched_einsum_to) -> Mapping[str, str]:
    """
    Returns the isomorphism mapping from *batched_einsum_from* to
    *batched_einsum_to*.

    .. note::

        Raises a :class:`ValueError` if the two batched einsums are not isomorphic.
    """
    canon_batched_einsum_from, subst_map_from = (
        _get_canonicalized_einsum_with_subst_mapping(batched_einsum_from))
    canon_batched_einsum_to, subst_map_to = (
        _get_canonicalized_einsum_with_subst_mapping(batched_einsum_to))

    if canon_batched_einsum_from != canon_batched_einsum_to:
        raise ValueError("Einsums are not isomorphic.")

    return Map({
        var_in_from_einsum: subst_map_to.inv[var_in_canon_einsum]
        for var_in_from_einsum, var_in_canon_einsum in subst_map_from.items()})

# vim: foldmethod=marker
