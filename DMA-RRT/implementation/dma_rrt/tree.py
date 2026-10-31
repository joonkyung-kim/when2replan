"""
    Tree of an RRT
"""

from __future__ import annotations

from typing import List, Optional
import numpy as np


class Node:
    def __init__(self, state: np.ndarray, parent: Optional[Node], cost: float) -> None:
        self.state = state
        self.parent = parent  # None only for the root
        self.cost = cost  # cost from the root to this node
        self.waypoint: Optional[np.ndarray] = None  # reference point of CL-RRT
        self.trajectory: Optional[np.ndarray] = None  # simulated states from the parent of CL-RRT


class Tree:
    def __init__(self, root_state: np.ndarray) -> None:
        root_state = np.asarray(root_state, dtype=float)
        self.nodes: List[Node] = [Node(root_state, None, 0.0)]
        self._positions: np.ndarray = root_state[:2][np.newaxis]  # size: (num_nodes, 2)

    def __len__(self) -> int:
        return len(self.nodes)

    @property
    def root(self) -> Node:
        return self.nodes[0]

    def add_node(self, state: np.ndarray, parent: Node, edge_cost: float) -> Node:
        """
        Add a new node as a child of parent
        :param state: state of the new node, its first two elements are the position
        :param parent: parent node, must be in this tree
        :param edge_cost: cost from parent to the new node
        :return: added node
        """
        node = Node(state, parent, parent.cost + edge_cost)
        self.nodes.append(node)
        self._positions = np.vstack((self._positions, state[:2]))
        return node

    def reroot(self, node: Node) -> None:
        """
        Make node the root and remove everything but its descendants. The costs of the nodes stay the same,
        so they are still comparable with each other
        """
        kept = self._get_subtree(node)
        node.parent = None
        self._set_nodes([n for n in self.nodes if n in kept])

    def remove_subtree(self, node: Node) -> None:
        """
        Remove node and all its descendants
        """
        removed = self._get_subtree(node)
        self._set_nodes([n for n in self.nodes if n not in removed])

    def _get_subtree(self, node: Node) -> set:
        # a parent is always before its children in self.nodes
        subtree = {node}
        for n in self.nodes:
            if n.parent in subtree:
                subtree.add(n)
        return subtree

    def _set_nodes(self, nodes: List[Node]) -> None:
        self.nodes = nodes
        self._positions = np.array([n.state[:2] for n in nodes])

    def nearest(self, position: np.ndarray) -> Node:
        """
        Get the node closest to position in euclidean distance
        """
        dists = np.linalg.norm(self._positions - position, axis=1)
        return self.nodes[np.argmin(dists)]

    def k_nearest(self, position: np.ndarray, k: int) -> List[Node]:
        """
        Get the k nodes closest to position in euclidean distance, the closest first
        """
        dists = np.linalg.norm(self._positions - position, axis=1)
        return [self.nodes[i] for i in np.argsort(dists)[:k]]

    def nodes_to(self, node: Node) -> List[Node]:
        """
        Get the nodes from the root to node
        """
        nodes = []
        current: Optional[Node] = node
        while current is not None:
            nodes.append(current)
            current = current.parent
        nodes.reverse()
        return nodes

    def path_to(self, node: Node) -> np.ndarray:
        """
        Get the states from the root to node
        :return: path size: (N, dim)
        """
        return np.array([n.state for n in self.nodes_to(node)])
