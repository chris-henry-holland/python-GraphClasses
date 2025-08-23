#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Set,
    Tuple,
    Union,
    Hashable,
    Any,
)

if TYPE_CHECKING:
    from graph_classes.limited_graph_types import (
        LimitedWeightedUndirectedGraph,
    )

import heapq

from graph_classes.utils import UnionFind, forestNodePairsTraversalStatistics

def kruskalIndex(self) -> Tuple[Union[float, int], Set[Tuple[int, int, int]]]:
    """
    Implementation of Kruskal's algorithm to find a minimum spanning
    tree or forest of a limited weighted undirected graph, returning the
    included edges in terms of the indices of the vertices they connect.
    For a connected graph a minimum spanning tree is the tree
    connecting all of the vertices of the graph (where the edges are
    considered to be undirected) such that the sum of weights of the
    edges is no larger than that of any other such tree.
    For a non-connected graph, a minimum spanning forest is a
    union of trees over the connected components of the graph, such
    that each such tree is a minimum spanning tree of the corresponding
    connected component.

    Args:
        None

    Returns:
    A 2-tuple whose index 0 contains the sum of the weights of edges
    of any minimum spanning tree or forest over the graph, and whose
    index 1 contains a set representing each edge in one of the
    minimum spanning tress or forests, in the form of a 3-tuple where
    indices 0 and 1 contain the indices of the vertices the edge
    connects and index 2 contains the weight of that edge.
    """
    uf = UnionFind(self.n)
    res = set()
    cost = 0
    edge_heap = []
    adj = getattr(self, self.adj_name)
    for idx1 in range(self.n):
        for idx2, min_weight in self.getAdjMinimumWeightsIndex(idx1).items():
            edge_heap.append((min_weight, idx1, idx2))
    heapq.heapify(edge_heap)
    while edge_heap:
        w, idx1, idx2 = heapq.heappop(edge_heap)
        if uf.connected(idx1, idx2): continue
        uf.union(idx1, idx2)
        res.add((idx1, idx2, w))
        cost += w
        if len(res) == self.n - 1: break
    return (cost, res)

def kruskal(self) -> Tuple[Union[float, int], Set[Tuple[Hashable, Hashable, int]]]:
    """
    Implementation of Kruskal's algorithm to find a minimum spanning
    tree or forest of an undirected weighted graph, returning the included
    edges in terms of the defined labels of the vertices they connect.
    For a connected graph a minimum spanning tree is the tree
    connecting all of the vertices of the graph (where the edges are
    considered to be undirected) such that the sum of weights of the
    edges is no larger than that of any other such tree.
    For a non-connected graph, a minimum spanning forest is a
    union of trees over the connected components of the graph, such
    that each such tree is a minimum spanning tree of the corresponding
    connected component.

    Args:
        None

    Returns:
    A 2-tuple whose index 0 contains the sum of the weights of edges
    of any minimum spanning tree or forest over the graph, and whose
    index 1 contains a set representing each edge in one of the
    minimum spanning tress or forests, in the form of a 3-tuple where
    indices 0 and 1 contain the defined labels of the vertices the edge
    connects and index 2 contains the weight of that edge.
    """
    cost, edges = self.kruskalIndex()
    return (cost, {(self.index2Vertex(e[0]),\
            self.index2Vertex(e[1]), e[2]) for e in edges})


def checkMinimumSpanningForest(
        graph: LimitedWeightedUndirectedGraph,
        cost: Union[int, float],
        forest_edges: Set[Tuple[Any, Any, Union[int, float]]],
        eps: float=10 ** -5) -> bool:
    # Checking the number of edges does not exceed the maximum possible
    # (i.e. a single tree, which has one fewer edge than the number of
    # vertices)
    if len(forest_edges) >= graph.n: return False
    if (sum(x[2] for x in forest_edges) - cost) > eps: return False
    uf = UnionFind(graph.n)

    # Checking there are no cycles and that the edges exist in the original
    # graph and have accurate weights
    for e in forest_edges:
        if not graph.vertexInGraph(e[0]) or not graph.vertexInGraph(e[1]):
            print("fail1")
            return False
        idx1, idx2 = graph.vertex2Index(e[0]), graph.vertex2Index(e[1])
        if uf.connected(idx1, idx2):
            print("fail2")
            return False
        d_dict = graph.getAdjMinimumWeightsIndex(idx1)
        if idx2 not in d_dict.keys():
            print("fail3")
            return False
        if abs(d_dict[idx2] - e[2]) > eps:
            print("fail4")
            return False
        uf.union(idx1, idx2)
    # Checking if the number of trees equals the number of connected
    # components of the original graph
    n_trees = graph.n - len(forest_edges)
    uf2 = UnionFind(graph.n)
    n_connected = graph.n
    for idx in range(graph.n):
        for idx2 in graph.getAdjMinimumWeightsIndex(idx).keys():
            if idx2 <= idx or uf2.connected(idx, idx2): continue
            n_connected -= 1
            uf2.union(idx, idx2)
    if n_trees != n_connected:
        print("fail5")
        return False
    #treeNodePairsTraversalStatistics(
    #    adj: List[Dict[int, Any]],
    #    op: Tuple[Callable[[Any, Any], Any], Any]=(lambda x, y: x + y, 0),
    #) -> List[Dict[int, Tuple[Any, int]]]

    # Checking if the tree can be improved by swapping out an included
    # edge for an excluded edge
    v2idx = lambda v: graph.vertex2Index(v)
    forest_adj = [{} for _ in range(graph.n)]
    for e in forest_edges:
        idx1, idx2 = v2idx(e[0]), v2idx(e[1])
        forest_adj[idx1][idx2] = e[2]
        forest_adj[idx2][idx1] = e[2]
    forest_path_stats = forestNodePairsTraversalStatistics(forest_adj, (lambda x, y: max(x, y), -float("inf")))
    for idx1, path_stats in enumerate(forest_path_stats):
        edge_weights = graph.getAdjMinimumWeightsIndex(idx1)
        for idx2, wt_tup in path_stats.items():
            if idx2 <= idx1 or idx2 not in edge_weights.keys() or idx2 == wt_tup[1]: continue
            max_weight = wt_tup[0]
            if max_weight > edge_weights[idx2]:
                return False
    return True