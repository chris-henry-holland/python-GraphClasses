#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Set,
    Tuple,
    Union,
    Hashable,
)

if TYPE_CHECKING:
    from graph_classes.limited_graph_types import (
        LimitedDirectedGraphTemplate,
    )

from collections import deque



from graph_classes.utils import containsDirectedCycle

from graph_classes.explicit_graph_types import (
    ExplicitUnweightedDirectedGraph,
)

### Algorithms for identifying strongly connected components in ###
### directed graphs                                             ###

def convertSCCReprIndex2Vertex(graph: LimitedDirectedGraphTemplate,\
        scc_repr_idx: List[int]) -> Dict[Hashable, Hashable]:
    return {graph.index2Vertex(idx1): graph.index2Vertex(idx2)\
            for idx1, idx2 in enumerate(scc_repr_idx)}

def convertSCCReprVertex2Index(graph: LimitedDirectedGraphTemplate,\
        scc_repr: Dict[Hashable, Hashable]) -> Dict[Hashable, Hashable]:
    return [graph.vertex2Index(scc_repr[graph.index2Vertex(idx1)])\
            for idx1 in range(graph.n)]

def kosarajuIndex(self) -> Tuple[dict]:
    """
    Kosaraju algorithm for finding strongly connected components
    (SCC) in a directed graph (with the directed graph given as a
    dictionary, where the keys are the vertices with the
    corresponding value being a set containing all the other
    vertices this vertex has a directed edge to- effectively the
    adjacency list representation of the outgoing edges).
    Each SCC is represented by one of its members. Returns a
    dictionary whose keys are the vertices of the original graph
    with the corresponding value being the representative vertex of
    the SCC to which it belongs.
    
    Can be used to solve Leetcode #1557 (when generalised to allow
    directed graphs with cycles- see below), #2101 (see below). In
    the examples included below, when input argument alg given as
    "kosaraju", kosarajuAdj() applied through condenseSCCAdj().
    """
    seen_bm = [0]
    scc_repr = [-1] * self.n
    top_sort_rev = []
    
    def visit(idx) -> None:
        bm = (1 << idx)
        if bm & seen_bm[0]: return
        seen_bm[0] |= bm
        for idx2 in self.adjGeneratorIndex(idx):
            visit(idx2)
        top_sort_rev.append(idx)
        return
        
    def assign(idx, idx0) -> None:
        if scc_repr[idx] != -1: return
        scc_repr[idx] = idx0
        for idx2 in self.inAdjGeneratorIndex(idx):
            assign(idx2, idx0)
        return
    
    for idx in range(self.n):
        visit(idx)
    for idx in reversed(top_sort_rev):
        assign(idx, idx)
    return scc_repr

def kosaraju(self) -> Dict[Hashable, Hashable]:
    return convertSCCReprIndex2Vertex(self, self.kosarajuIndex())

def tarjanSCCIndex(self) -> List[int]:
    """
    Tarjan algorithm for finding strongly connected components
    (SCC) in a directed graph (with the directed graph given as a
    dictionary, where the keys are the vertices with the
    corresponding value being a set containing all the other
    vertices this vertex has a directed edge to- effectively the
    adjacency list representation of the outgoing edges).
    Each SCC is represented by one of its members. Returns a
    dictionary whose keys are the vertices of the original graph
    with the corresponding value being the representative vertex of
    the SCC to which it belongs.
    
    Can be used to solve Leetcode #1557 (when generalised to allow
    directed graphs with cycles- see below), #2101 (see below) In
    the examples included below, when input argument alg given as
    "tarjan", tarjanSCCAdj() applied through condenseSCCAdj().
    """
    lo = {}
    stk = []
    in_stk = set()
    scc_repr = [-1] * self.n

    def recur(idx: int, t: int) -> int:
        if idx in lo.keys():
            return t
        t0 = t
        lo[idx] = t
        stk.append(idx)
        in_stk.add(idx)
        t += 1
        for idx2 in self.adjGeneratorIndex(idx):
            t = recur(idx2, t)
            if idx2 in in_stk:
                lo[idx] = min(lo[idx], lo[idx2])
        if lo[idx] != t0: return t
        while stk[-1] != idx:
            idx2 = stk.pop()
            in_stk.remove(idx2)
            scc_repr[idx2] = idx
        stk.pop()
        in_stk.remove(idx)
        scc_repr[idx] = idx
        return t
    t = 0
    for idx in range(self.n):
        t = recur(idx, t)
    return scc_repr

def tarjanSCC(self) -> Dict[int, int]:
    return convertSCCReprIndex2Vertex(self, self.tarjanSCCIndex())

def SCCReprEqualIndex(scc1: List[int], scc2: List[int]) -> bool:
    """
    """
    for idx, (idx1, idx2) in enumerate(zip(scc1, scc2)):
        if scc1[scc2[idx2]] != scc1[idx1] or\
                scc2[scc1[idx1]] != scc2[idx2]:
            return False
    return True

def SCCReprEqual(scc1: Dict[Hashable, Hashable],\
        scc2: Dict[Hashable, Hashable]) -> bool:
    """
    """
    for v, v1 in scc1.items():
        v2 = scc2[v]
        if scc1[scc2[v2]] != scc1[v1] or scc2[scc1[v1]] != scc2[v2]:
            return False
    return True

def checkSCCReprIndex(graph: LimitedDirectedGraphTemplate,\
        scc_repr: List[int]) -> bool:
    
    #scc_sets = {}
    #for idx1, idx2 in scc_repr.items():
    #    scc_sets.setdefault(idx2, set())
    #    scc_sets[idx2].add(idx1)
    for idx1, idx2 in enumerate(scc_repr):
        if scc_repr[idx2] != idx2:
            return False
    
    seen_scc = {}
    def bfs(idx: int) -> bool:
        res = [{idx}, set()]
        seen = {idx}
        qu = deque([idx])
        scc_idx = scc_repr[idx]
        while qu:
            idx2 = qu.popleft()
            in_orig_scc = (scc_repr[idx2] == scc_idx)
            for idx3 in graph.adjGeneratorIndex(idx2):
                if idx3 in seen: continue
                seen.add(idx3)
                scc_idx3 = scc_repr[idx3]
                if scc_idx3 == scc_idx:
                    if not in_orig_scc:
                        return False
                    res[0].add(idx3)
                else:
                    res[1].add(scc_idx3)
                    if scc_idx3 in seen_scc.keys():
                        if scc_idx in seen_scc[scc_idx3]:
                            return False
                        continue
                qu.append(idx3)
        if scc_idx in seen_scc.keys():
            if res[0] != seen_scc[scc_idx][0]:
                return False
            seen_scc[scc_idx] = res
        return True
    
    for idx in range(graph.n):
        if not bfs(idx):
            return False
    return True

def checkSCCRepr(graph: LimitedDirectedGraphTemplate,\
        scc_repr: Dict[Hashable, Hashable]) -> bool:
    scc_repr_idx = [-1] * graph.n
    for v1, v2 in scc_repr.items():
        scc_repr_idx[graph.vertex2Index(v1)] = graph.vertex2Index(v2)
    return checkSCCReprIndex(graph, scc_repr_idx)

### Algorithms for condensing the strongly connected components ###
### of a directed graph                                         ###

def condenseSCCIndex(self, alg: str="tarjan",\
        set_condensed_in_degrees: bool=False,\
        condensed_graph_from_idx: bool=False)\
        -> Tuple[Union[List[int], Dict[int, Set[Hashable]],\
        ExplicitUnweightedDirectedGraph]]:
    """
    Finds the strongly connected components (SCC) in a
    directed graph (with the directed graph given as a dictionary,
    where the keys are the vertices with the corresponding value
    being a set containing all the other vertices this vertex has
    a directed edge to- effectively the adjacency list
    representation of the outgoing edges) and generates a directed
    graph where each SCC is condensed down to one of its member
    vertices. Uses either Tarjan's algorithm for SCC (if alg given
    as "tarjan") or Kosaraju's algorithm (if alg given as
    "kosaraju").
    
    Each SCC is represented by one of its members. Returns a
    4-tuple whose 0th index is a dictionary whose keys are the
    vertices of the original graph with the corresponding value
    being the representative vertex of the SCC to which it belongs;
    the 1st index is a dictionary whose keys are the vertices
    chosen to represent each SCC with the corresponding value being
    a set containing all of the vertices in that SCC; the 2nd index
    is a dictionary representing the adjacency list representation
    of the directed graph (similar to the input) with each SCC
    condensed down to its representative member as a single vertex;
    and 3rd index is a dictionary whose keys are the vertices of
    the condensed directed graph of the 2nd index (i.e. the
    vertices representing each SCC) with the corresponding value
    being the in-degree of this vertex in the condensed graph.
    
    Note that the resulting directed graph is guaranteed to be
    acyclic.
    
    Can be used to solve Leetcode #1557 (when generalised to allow
    directed graphs with cycles- see below), #2101 (see below)
    """
    if alg not in {"tarjan", "kosaraju"}:
        raise ValueError("Input argument alg must be either "
                "'tarjan' or 'kosaraju'")
    conv_func = (lambda x: x) if condensed_graph_from_idx else\
            (lambda x: self.index2Vertex(x))
    condense_func = self.tarjanSCCIndex if alg == "tarjan" else\
            self.kosarajuIndex
    
    scc_repr = condense_func()
    scc_groups = {}
    for idx1, idx2 in enumerate(scc_repr):
        scc_groups.setdefault(idx2, set())
        scc_groups[idx2].add(idx1)
    
    edges = []
    vertices = []
    for idx1 in range(self.n):
        idx1_0 = scc_repr[idx1]
        if idx1_0 == idx1:
            vertices.append(conv_func(idx1))
        v1_0 = conv_func(idx1_0)
        for idx2 in self.adjGeneratorIndex(idx1):
            idx2_0 = scc_repr[idx2]
            if idx1_0 != idx2_0:
                v2_0 = conv_func(idx2_0)
                edges.append((v1_0, v2_0))
    scc_graph = ExplicitUnweightedDirectedGraph(vertices, edges,\
            set_condensed_in_degrees)
    return (scc_repr, scc_groups, scc_graph)
            
    """
    idx_to_group_repr = {}
    for v1, v2 in scc_repr.items():
        scc_groups.setdefault(v2, set())
        scc_groups[v2].add(v1)
        idx_to_group_repr[self.vertex2Index(v1)] = v2
    edges = []
    for v1 in range(self.n):
        i1 = idx_to_group_repr[v1]
        for v2 in self.adjGeneratorIndex(v1):
            i2 = idx_to_group_repr[v2]
            if i1 == i2: continue
            edges.append((i1, i2))
    return (scc_repr, scc_groups,\
            ExplicitUnweightedDirectedGraph(scc_groups.keys(), edges,\
            set_in_degrees=set_condensed_in_degrees))
    """

def condenseSCC(self, alg: str="tarjan",\
        set_condensed_in_degrees: bool=False)\
        -> Tuple[Union[Dict[Hashable, Hashable],\
        Dict[Hashable, Set[Hashable]],\
        ExplicitUnweightedDirectedGraph]]:
    #print(alg, set_condensed_in_degrees)
    if alg not in {"tarjan", "kosaraju"}:
        raise ValueError("Input argument alg must be either "
                "'tarjan' or 'kosaraju'")
    (scc_repr_idx, scc_groups_idx, scc_graph) =\
            self.condenseSCCIndex(alg=alg,\
            set_condensed_in_degrees=set_condensed_in_degrees,\
            condensed_graph_from_idx=False)
    scc_repr = convertSCCReprIndex2Vertex(self, scc_repr_idx)
    scc_groups = {self.index2Vertex(idx1):\
            {self.index2Vertex(idx2) for idx2 in idx2_set}\
            for idx1, idx2_set in scc_groups_idx.items()}
    return (scc_repr, scc_groups, scc_graph)

def checkCondensedSCCIndex(graph: LimitedDirectedGraphTemplate,\
        scc_repr: List[int], scc_groups: Dict[int, Set[int]],\
        scc_graph: ExplicitUnweightedDirectedGraph,\
        condensed_graph_from_idx: bool=False) -> bool:
    
    if len(scc_groups) != scc_graph.n:
        return False
    
    for idx1, idx2 in enumerate(scc_repr):
        if scc_repr[idx2] != idx2:
            return False
    
    if containsDirectedCycle(scc_graph):
        return False
    
    if condensed_graph_from_idx:
        for idx in scc_groups.keys():
            if not scc_graph.vertexInGraph(idx):
                return False
        adj_set_func = lambda idx: set(scc_graph.adjGenerator(idx))
    else:
        for idx in scc_groups.keys():
            if not scc_graph.vertexInGraph(graph.index2Vertex(idx)):
                return False
        adj_set_func = lambda idx: {graph.vertex2Index(v) for v in\
                scc_graph.adjGenerator(graph.index2Vertex(idx))}
    
    def bfs(idx: int) -> bool:
        res = [{idx}, set()]
        qu = deque([idx])
        scc_idx = scc_repr[idx]
        while qu:
            idx2 = qu.popleft()
            adj = list(graph.adjGeneratorIndex(idx2))
            if adj and min(adj) < idx:
                return True
            for idx3 in graph.adjGeneratorIndex(idx2):
                if idx3 in res[0]:
                    continue
                scc_idx3 = scc_repr[idx3]
                if scc_idx3 != scc_idx:
                    res[1].add(scc_idx3)
                    continue
                res[0].add(idx3)
                qu.append(idx3)
        return res[0] == scc_groups[scc_idx] and\
                res[1] == adj_set_func(scc_repr[idx])
    
    for idx in range(graph.n):
        if not bfs(idx):
            #print("hi3")
            return False
    return True

def checkCondensedSCC(graph: LimitedDirectedGraphTemplate,\
        scc_repr: Dict[Hashable, Hashable],\
        scc_groups: Dict[Hashable, Set[Hashable]],\
        scc_graph: ExplicitUnweightedDirectedGraph) -> bool:
    scc_groups_idx = {graph.vertex2Index(v1):\
            {graph.vertex2Index(v2) for v2 in v2_set}\
            for v1, v2_set in scc_groups.items()}
    #print(scc_repr)
    return checkCondensedSCCIndex(graph,\
            convertSCCReprVertex2Index(graph, scc_repr),\
            scc_groups_idx, scc_graph, condensed_graph_from_idx=False)

def condensedSCCEqualIndex(condensed1: List[int], condensed2: List[int]) -> bool:
    """
    """
    return False
    #for idx, (idx1, idx2) in enumerate(zip(scc1, scc2)):
    #    if scc1[scc2[idx2]] != scc1[idx1] or\
    #            scc2[scc1[idx1]] != scc2[idx2]:
    #        return False
    #return True

def condensedSCCEqual(condensed1: List[int], condensed2: List[int]) -> bool:
    """
    """
    return False
    #for idx, (idx1, idx2) in enumerate(zip(scc1, scc2)):
    #    if scc1[scc2[idx2]] != scc1[idx1] or\
    #            scc2[scc1[idx1]] != scc2[idx2]:
    #        return False
    #return True
