#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Hashable,
)

if TYPE_CHECKING:
    from graph_classes import (
        LimitedDirectedGraphTemplate,
    )

from collections import deque

from graph_classes.utils import containsDirectedCycle

### Variants of Kahn's algorithm for topological sorting ###

def kahnIndex(self) -> List[int]:
    """
    Method implementing Kahn's algorithm for topological sorting of
    directed graphs, giving a topological ordering of the indices of
    the vertices of the directed graph in a list if such an ordering
    exists, or an empty list if no such ordering exists (i.e. the
    directed graph contains a directed cycle).
    A topological ordering of the vertices in a directed graph is a
    linear ordering (not necessarily unique) of all the vertices in
    the graph such that for any vertex in the ordering, there does not
    exist a directed path through the directed graph to itself or any
    vertex that comes before the chosen vertex in the ordering.
    A topological ordering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        None
    
    Returns:
    List of ints, which if the directed graph contains a directed cycle
    is an empty list, while if it is acyclic contains a topological
    ordering of the vertices of the directed graph in terms of their
    graph indices.
    """
    qu = deque()
    in_degrees = [0] * self.n
    for idx in range(self.n):
        in_degrees[idx] = self.inDegreeIndex(idx)
        if not in_degrees[idx]: qu.append(idx)
    res = []
    while qu:
        idx = qu.popleft()
        res.append(idx)
        for idx2 in self.adjGeneratorIndex(idx):
            in_degrees[idx2] -= self.edgeCountIndex(idx, idx2)
            if not in_degrees[idx2]:
                qu.append(idx2)
    return res if len(res) == self.n else []

def kahn(self) -> List[Hashable]:
    """
    Method implementing Kahn's algorithm for topological sorting of
    directed graphs, giving a topological ordering of the vertices of
    the directed graph in a list if such an ordering exists, or an
    empty list if no such ordering exists (i.e. the directed graph
    contains a directed cycle).
    A topological ordering of the vertices in a directed graph is a
    linear ordering (not necessarily unique) of all the vertices in
    the graph such that for any vertex in the ordering, there does not
    exist a directed path through the directed graph to itself or any
    vertex that comes before the chosen vertex in the ordering.
    A topological ordering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        None
    
    Returns:
    List of hashable objects, which if the directed graph contains a
    directed cycle is an empty list, while if it is acyclic contains a
    topological ordering of the vertices of the directed graph.
    """
    return [self.index2Vertex(x) for x in self.kahnIndex()]

def checkTopologicalOrderingIndex(
    graph: LimitedDirectedGraphTemplate,
    ordering_idx: List[int],
) -> bool:
    """
    Function checking whether a list of vertex indices of a directed
    graph represents a topological ordering of the vertices of the
    graph if such a topological ordering exists, or is an empty list if
    such a topological ordering does not exist.
    A topological ordering of the vertices in a directed graph is a
    linear ordering (not necessarily unique) of all the vertices in
    the graph such that for any vertex in the ordering, there does not
    exist a directed path through the directed graph to itself or any
    vertex that comes before the chosen vertex in the ordering.
    A topological ordering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        Required positional:
        graph (class descending from LimitedDirectedGraphTemplate):
                The directed graph for which the topological ordering
                is being checked.
        ordering_idx (tuple of ints): Either an empty list indicating
                the status of graph as having no topological
                ordering of its vertices is to be checked (or the
                trivial case where the graph has no vertices), or a
                list containing indices of vertices of graph whose
                identity as representing a topological ordering of the
                vertices of graph is to be checked.
    
    Returns:
    Boolean (bool) with value True if graph has no topological ordering
    of its vertices and ordering_idx is an empty list or ordering_idx
    represents a topological ordering of the vertices of graph, and
    value False otherwise.
    """
    n = graph.n
    if not ordering_idx:
        return not n or containsDirectedCycle(graph)
    if len(ordering_idx) != n:
        return False
    seen = set()
    for idx in ordering_idx:
        if idx in seen or not 0 <= idx < n:
            return False
        seen.add(idx)
        for idx2 in graph.getAdjIndex(idx).keys():
            if idx2 in seen: return False
    return True

def checkTopologicalOrdering(
    graph: LimitedDirectedGraphTemplate,
    ordering: List[Hashable],
) -> bool:
    """
    Function checking whether a list of vertices of a directed graph
    represents a topological ordering of the vertices of the graph if
    such a topological ordering exists, or is an empty list if such
    a topological ordering does not exist.
    A topological ordering of the vertices in a directed graph is a
    linear ordering (not necessarily unique) of all the vertices in
    the graph such that for any vertex in the ordering, there does not
    exist a directed path through the directed graph to itself or any
    vertex that comes before the chosen vertex in the ordering.
    A topological ordering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        Required positional:
        graph (class descending from LimitedDirectedGraphTemplate):
                The directed graph for which the topological ordering
                is being checked.
        ordering (tuple of hashable objects): Either an empty list
                indicating the status of graph as having no topological
                ordering of its vertices is to be checked (or the
                trivial case where the graph has no vertices), or a
                list containing vertices of graph whose identity as a
                topological ordering of the vertices of graph is to
                be checked.
    
    Returns:
    Boolean (bool) with value True if graph has no topological ordering
    of its vertices and ordering is an empty list or ordering represents
    a topological ordering of the vertices of graph, and value False
    otherwise.
    """
    n = graph.n
    if not ordering:
        return not n or containsDirectedCycle(graph)
    if len(ordering) != n:
        return False
    seen = set()
    for v in ordering:
        if not graph.vertexInGraph(v):
            return False
        idx = graph.vertex2Index(v)
        if idx in seen:
            return False
        seen.add(idx)
        for idx2 in graph.getAdjIndex(idx).keys():
            if idx2 in seen: return False
    return True

def kahnLayeringIndex(self) -> List[List[int]]:
    """
    Method utilising a modified implementation of Kahn's algorithm for
    topological sorting of directed graphs to give a topological
    layering of the graph indices of the vertices of the graph if
    if such an layering exists, or an empty list if no such layering
    exists (i.e. the directed graph contains a directed cycle).
    A topological layering of the vertices in a directed graph is
    an ordered partition of the graph indices such that for a given
    partition, there does not exist an outgoing edge from any vertex
    in the partition to any other vertex in the partition or any
    vertex in any preceding partition, and each vertex is in the
    earliest partition possible subject to this constraint.
    A topological layering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        None
    
    Returns:
    List of list of ints, which if the directed graph contains a
    directed cycle is an empty list, while if the directed graph is
    acyclic contains the topological layering as a list of lists
    containing the ordered partitions of the vertices as represented
    by their graph indices, such that partitions appearing earlier in
    the list are considered to be earlier in the ordering (i.e. there
    does not exist an outgoing directed edge from any vertex to any
    other vertex whose graph index appears in a partition with an
    index in the returned list not exceeding that of the partition
    containing the graph index of the first vertex).
    """
    in_degrees = [0] * self.n
    qu = deque()
    for idx in range(self.n):
        in_degrees[idx] = self.inDegreeIndex(idx)
        if not in_degrees[idx]: qu.append(idx)
    res = []
    n_seen = len(qu)
    for depth in range(1, self.n + 1):
        res.append([])
        for _ in range(len(qu)):
            idx = qu.popleft()
            res[-1].append(idx)
            for idx2 in self.adjGeneratorIndex(idx):
                in_degrees[idx2] -= self.edgeCountIndex(idx, idx2)
                if not in_degrees[idx2]:
                    qu.append(idx2)
        if not qu: break
        n_seen += len(qu)
        if n_seen == self.n: break
    if qu: res.append(list(qu))
    return res if n_seen == self.n else []

def kahnLayering(self) -> List[List[Hashable]]:
    """
    Method utilising a modified implementation of Kahn's algorithm for
    topological sorting of directed graphs to give a topological
    layering of the vertices of the graph if if such an layering
    exists, or an empty list if no such layering exists (i.e. the
    directed graph contains a directed cycle).
    A topological layering of the vertices in a directed graph is
    an ordered partition of the graph indices such that for a given
    partition, there does not exist an outgoing edge from any vertex
    in the partition to any other vertex in the partition or any
    vertex in any preceding partition, and each vertex is in the
    earliest partition possible subject to this constraint.
    A topological layering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        None
    
    Returns:
    List of list of hashable objects, which if the directed graph
    contains a directed cycle is an empty list, while if the directed
    graph is acyclic contains the topological layering as a list of
    lists containing the ordered partitions of the vertices, such that
    partitions appearing earlier in the list are considered to be
    earlier in the ordering (i.e. there does not exist an outgoing
    directed edge from any vertex to any other vertex that appears in
    a partition with an index in the returned list not exceeding that
    of the partition containing the the first vertex).
    """
    res = self.kahnLayeringIndex()
    return [[self.index2Vertex(x) for x in lst] for lst in res]

def checkTopologicalLayeringIndex(
    graph: LimitedDirectedGraphTemplate,
    layering_idx: List[List[int]],
) -> bool:
    """
    Function checking whether a list of lists of vertex indices of a
    directed graph represents a topological layering of the vertices of
    the graph if such a topological layering exists, or is an empty
    list if such a topological layering does not exist.
    A topological layering of the vertices in a directed graph is
    an ordered partition of the graph indices such that for a given
    partition, there does not exist an outgoing edge from any vertex
    in the partition to any other vertex in the partition or any
    vertex in any preceding partition, and each vertex is in the
    earliest partition possible subject to this constraint.
    A topological layering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        Required positional:
        graph (class descending from LimitedDirectedGraphTemplate):
                The directed graph for which the topological layering
                is being checked.
        layering_idx (list of lists of ints): Either an empty list
                indicating the status of graph as having no topological
                layering of its vertices is to be checked (or the
                trivial case where the graph has no vertices), or a
                list of lists containing indices of vertices of graph
                whose identity as representing a topological layering
                of the vertices of graph is to be checked.
    
    Returns:
    Boolean (bool) with value True if graph has no topological layering
    of its vertices and layering_idx is an empty list or layering_idx
    represents a topological layering of the vertices of graph, and
    value False otherwise.
    """
    n = graph.n
    if not layering_idx:
        return not n or containsDirectedCycle(graph)
    if sum(len(x) for x in layering_idx) != n:
        return False
    nxt = set(layering_idx[0])
    if len(nxt) != len(layering_idx[0]):
        return False
    seen = set(nxt)
    for layer in layering_idx:
        if len(layer) > len(nxt):
            return False
        layer_set = set(layer)
        if len(layer_set) < len(layer) or not layer_set.issubset(nxt):
            return False
        seen |= layer_set
        nxt = set()
        for idx1 in layer_set:
            for idx2 in graph.getAdjIndex(idx1):
                if idx2 in seen:
                    return False
                nxt.add(idx2)
    return not nxt

def checkTopologicalLayering(
    graph: LimitedDirectedGraphTemplate,
    layering: List[List[Hashable]],
) -> bool:
    """
    Function checking whether a list of lists of vertices of a directed
    graph represents a topological layering of the vertices of the
    graph if such a topological layering exists, or is an empty
    list if such a topological layering does not exist.
    A topological layering of the vertices in a directed graph is
    an ordered partition of the graph indices such that for a given
    partition, there does not exist an outgoing edge from any vertex
    in the partition to any other vertex in the partition or any
    vertex in any preceding partition, and each vertex is in the
    earliest partition possible subject to this constraint.
    A topological layering exists for a directed graph if and only if
    the graph does not contain a directed cycle, or equivalently is
    acyclic (i.e. there does not exist a non-empty directed path from
    any vertex to itself).
    
    Args:
        Required positional:
        graph (class descending from LimitedDirectedGraphTemplate):
                The directed graph for which the topological layering
                is being checked.
        layering_idx (list of lists of ints): Either an empty list
                indicating the status of graph as having no topological
                layering of its vertices is to be checked (or the
                trivial case where the graph has no vertices), or a
                list of lists containing vertices of graph whose
                identity as representing a topological layering of the
                vertices of graph is to be checked.
    
    Returns:
    Boolean (bool) with value True if graph has no topological layering
    of its vertices and layering is an empty list or layering represents
    a topological layering of the vertices of graph, and value False
    otherwise.
    """
    n = graph.n
    if not layering:
        return not n or containsDirectedCycle(graph)
    if sum(len(x) for x in layering) != n:
        return False
    nxt = set()
    for v in layering[0]:
        if not graph.vertexInGraph(v):
            return False
        nxt.add(graph.vertex2Index(v))
    if len(nxt) != len(layering[0]):
        return False
    seen = set()
    for layer in layering:
        if len(layer) > len(nxt):
            return False
        layer_set = set()
        for v in layer:
            if not graph.vertexInGraph(v):
                return False
            layer_set.add(graph.vertex2Index(v))
        if len(layer_set) < len(layer) or not layer_set.issubset(nxt):
            return False
        seen |= layer_set
        nxt = set()
        for idx1 in layer_set:
            for idx2 in graph.getAdjIndex(idx1):
                if idx2 in seen:
                    return False
                nxt.add(idx2)
    return not nxt
