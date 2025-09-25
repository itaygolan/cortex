"""Helper file to visualize the compute graph."""

from __future__ import annotations

import collections
import html
from typing import TYPE_CHECKING, Any, Optional

from graphviz import Source

if TYPE_CHECKING:
    from cortex import Tensor
    from cortex.operations import Operation


def _tensor_label(tensor: Tensor) -> str:
    """Short label for a tensor node (shape, dtype, requires_grad, maybe grad)."""

    shape = tensor.shape
    dtype = tensor.dtype
    req = tensor.requires_grad

    op = tensor.operation
    op_str = repr(op)
    parts = [
        f"id={id(tensor)}",
        f"shape={shape}",
        f"dtype={dtype}",
        f"req_grad={req}",
        # f"op={op_str}",
    ]
    parts = [p for p in parts if p]
    label = "\\n".join([html.escape(str(p)) for p in parts])
    return label


def _op_label(op: Operation) -> str:
    """Short label for an operation node."""
    if op is None:
        return ""
    # prefer class name or repr
    try:
        name = getattr(op, "__class__", None)
        if name and hasattr(name, "__name__"):
            name = name.__name__
        else:
            name = repr(op)
    except Exception:
        name = repr(op)
    return html.escape(str(name))


def graph_to_dot_from_leaf(
    root: Tensor,
    max_nodes: Optional[int] = 1000,
    max_depth: Optional[int] = None,
    include_op_nodes: bool = True,
) -> str:
    """
    Build a DOT graph representing the compute graph upstream from `root`.

    - root: output tensor (e.g. z)
    - max_nodes: safety cap on number of tensor nodes visited
    - max_depth: maximum hops upstream (None => unlimited)
    - include_op_nodes: if True, create op nodes between parents and tensor nodes
    """
    visited: set[int] = set()  # visited tensor ids
    nodes = []
    edges = []
    q = collections.deque([(root, 0)])

    # Keep track of operation node ids to avoid duplication if include_op_nodes
    op_node_ids = {}

    while q and (max_nodes is None or len(visited) < max_nodes):
        tensor, depth = q.popleft()
        tid = id(tensor)
        if tid in visited:
            continue
        visited.add(tid)

        # Tensor node
        tname = f"t{tid}"
        tlabel = _tensor_label(tensor).replace('"', '\\"')
        nodes.append(f'  {tname} [label="{tlabel}", shape=box, fontsize=10];')

        # If tensor has an operation with parents, walk to parents
        op = getattr(tensor, "operation", None)
        if op is not None:
            parents: tuple[Any, ...] = getattr(op, "parents", tuple())
            # If including op nodes, create a single op node for this op instance
            if include_op_nodes:
                op_uid = id(op)
                opname = f"op{op_uid}"
                if op_uid not in op_node_ids:
                    oplabel = _op_label(op).replace('"', '\\"')
                    nodes.append(
                        f'  {opname} [label="{oplabel}", shape=ellipse, style=filled, fillcolor=lightgrey, fontsize=10];'
                    )
                    op_node_ids[op_uid] = opname
                else:
                    opname = op_node_ids[op_uid]

                # connect op -> tensor (op produces tensor)
                edges.append(f"  {opname} -> {tname};")

            for p in parents:
                if p is None:
                    continue
                pid = id(p)
                pname = f"t{pid}"
                # If including op nodes: parent -> op -> tensor
                if include_op_nodes:
                    edges.append(f"  {pname} -> {opname};")
                else:
                    # direct parent -> tensor
                    edges.append(f"  {pname} -> {tname};")

                # enqueue parent if we haven't visited it and depth allows
                if pid not in visited:
                    if max_depth is not None and depth + 1 > max_depth:
                        continue
                    q.append((p, depth + 1))
        else:
            # No operation: this is a leaf input (e.g., x or y). Nothing to do.
            pass

    dot = "digraph AutogradGraph {\n"
    dot += "  rankdir=LR;\n"
    dot += '  node [fontname="Helvetica"];\n'
    dot += "\n".join(nodes) + "\n" + "\n".join(edges) + "\n}\n"
    return dot


def visualize_graph(
    leaf: Tensor,
    max_nodes: Optional[int] = 1000,
    max_depth: Optional[int] = None,
    include_op_nodes: bool = True,
    notebook: bool = True,
):
    """
    Visualize the entire compute graph used to compute the given tensor
    `leaf`. Returns graphviz.Source in Jupyter if available, otherwise DOT text.
    """
    dot = graph_to_dot_from_leaf(
        leaf,
        max_nodes=max_nodes,
        max_depth=max_depth,
        include_op_nodes=include_op_nodes,
    )
    try:
        if notebook:
            return Source(dot)
        else:
            return dot
    except Exception:
        # graphviz not installed: return dot string so user can render with external `dot`
        return dot
