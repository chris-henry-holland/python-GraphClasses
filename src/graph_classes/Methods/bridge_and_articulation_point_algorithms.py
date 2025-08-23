#! /usr/bin/env python

from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Hashable,
)

from graph_classes.limited_graph_types import (
    LimitedUndirectedGraphTemplate,
)

### Algorithms that find bridges in undirected graphs ###

def tarjanBridgeIndex(self) -> Set[Tuple[int]]:
    """
    Method implementating the Tarjan bridge algorithm. For undirected
    graphs, identifies all bridges in the graph in terms of the vertex
    indices.
    A bridge is an edge of the graph which, if removed, increases
    the number of connected components of the graph (or
    equivalently, causes at least one pair of vertices which are
    connected in the original graph to become disconnected).
    
    Args:
        None
    
    Returns:
    Set of 2-tuples of integers, with each such tuple representing a
    bridge of the graph- the two items in a given 2-tuple are the
    indices of the vertices between which the bridge crosses in
    ascending order.
    """
    
    bridges = set()
    lo = {}

    def dfs(i: int, t: int, i0=None) -> int:
        if i in lo.keys(): return t
        t0 = t
        lo[i] = t
        t += 1
        for i2 in self.adjGeneratorIndex(i):
            if i2 == i0: continue
            t = dfs(i2, t, i)
            lo[i] = min(lo[i], lo[i2])
            if lo[i2] > t0 and self.edgeCountIndex(i, i2) == 1:
                bridges.add(tuple(sorted([i, i2])))
        return t
    
    t = 0
    for i in range(self.n):
        t = dfs(i, t)
    return bridges

def tarjanBridge(self) -> Set[Tuple[Hashable]]:
    """
    Method implementating the Tarjan bridge algorithm. For undirected
    graphs, identifies all bridges in the graph.
    A bridge is an edge of the graph which, if removed, increases
    the number of connected components of the graph (or
    equivalently, causes at least one pair of vertices which are
    connected in the original graph to become disconnected).
    
    Args:
        None
    
    Returns:
    Set of 2-tuples of hashable objects, with each such tuple
    representing a bridge of the graph- the two items in a given
    2-tuple are the vertices between which the bridge crosses in
    ascending order of their indices in the graph object.
    """
    
    return {tuple(map(self.index2Vertex, pair)) for pair in\
            self.tarjanBridgeIndex()}

def checkBridgesIndex(graph: "LimitedUndirectedGraphTemplate",\
        bridges_idx: Set[Tuple[int]], check_all: bool=True) -> bool:
    """
    Function checking whether the edges represented by the pairs of
    vertex indices in bridges_idx are the bridges of the undirected
    graph given by input argument graph.
    A bridge is an edge of the graph which, if removed, increases
    the number of connected components of the graph (or
    equivalently, causes at least one pair of vertices which are
    connected in the original graph to become disconnected).
    If check_all given as False, then only checks that the edges
    given are indeed bridges, otherwise also checks that all other
    edges are not bridges.
    
    Args:
        Required positional:
        graph (class descending from LimitedUndirectedGraphTemplate):
                The undirected graph for which the bridges are being
                ascertained.
        bridges_idx (set of 2-tuples of ints): The edges whose
                identity as bridges of graph are to be tested, in the
                form of a tuple of 2-tuples of ints, where each
                2-tuple represents an edge of graph by containing the
                indices in graph of the two vertices the edge connects.
        
        Optional named:
        check_all (bool): If False, only checks that the edges
                represented by bridges_idx are bridges of graph,
                otherwise also checks that the other edges of graph
                are not bridges.
            Default: True
    
    Returns:
    Boolean (bool) with value True and all of the edges represented by
    bridges_idx are bridges of graph, and if check_all is given as
    True all other edges of graph are not bridges, and value False
    otherwise.
    """
    def dfs(idx1: int, idx2: int) -> bool:
        edge_counts_dict = graph.getAdjEdgeCountsIndex(idx1)
        if edge_counts_dict.get(idx2, 0) != 1:
            return False
        seen = set(edge_counts_dict.keys())
        seen -= {idx1, idx2}
        stk = list(seen)
        seen.add(idx1)
        while stk:
            idx = stk.pop()
            #print(idx, stk)
            adj = graph.getAdjIndex(idx).keys()
            if idx2 in adj: return False
            for idx3 in adj:
                if idx3 in seen: continue
                seen.add(idx3)
                stk.append(idx3)
        return True
    
    if not check_all:
        for idx1, idx2 in bridges_idx:
            if not dfs(idx1, idx2): return False
        return True
    bridge_set = set(bridges_idx)
    for idx1 in range(graph.n):
        for idx2 in graph.getAdjIndex(idx1).keys():
            if idx1 >= idx2: continue
            if dfs(idx1, idx2) != ((idx1, idx2) in bridge_set):
                #print(getattr(graph, graph.adj_name))
                #print((idx1, idx2), (idx1, idx2) in bridge_set,\
                #        dfs(idx1, idx2), bridge_set)
                return False        
    return True

def checkBridges(graph: "LimitedUndirectedGraphTemplate",\
        bridges: Set[Tuple[Hashable]], check_all: bool=True) -> bool:
    """
    Function checking whether the edges represented by the pairs of
    vertices in bridges are the bridges of the undirected graph
    given by input argument graph.
    A bridge is an edge of the graph which, if removed, increases
    the number of connected components of the graph (or
    equivalently, causes at least one pair of vertices which are
    connected in the original graph to become disconnected).
    If check_all given as False, then only checks that the edges
    given are indeed bridges, otherwise also checks that all other
    edges are not bridges.
    
    Args:
        Required positional:
        graph (class descending from LimitedUndirectedGraphTemplate):
                The undirected graph for which the bridges are being
                ascertained.
        bridges (set of 2-tuples of hashable objects): The edges
                whose identity as bridges of graph are to be tested, in
                the form of a tuple of 2-tuples of hashable objects,
                where each 2-tuple represents an edge of graph by
                containing the two vertices the edge connects.
        
        Optional named:
        check_all (bool): If False, only checks that the edges
                represented by bridges are bridges of graph, otherwise
                also checks that the other edges of graph are not
                bridges.
            Default: True
    
    Returns:
    Boolean (bool) with value True and all of the edges represented by
    bridges are bridges of graph, and if check_all is given as
    True all other edges of graph are not bridges, and value False
    otherwise.
    """
    bridges_idx = {tuple(graph.vertex2Index(v) for v in bridge)\
            for bridge in bridges}
    return checkBridgesIndex(graph, bridges_idx, check_all=check_all)

### Algorithms that find articulation points in undirected graphs ###

def tarjanArticulationBasicIndex(self) -> Set[int]:
    """
    Implementation of the Tarjan articulation point algorithm for
    undirected graphs. Identifies all articulation points in an
    undirected graph in terms of their index in the graph.
    An articulation point is a vertex of the graph which, if removed
    along with all of its associated edges increases the number of
    connected components of the graph (or equivalently, causes at least
    one pair of vertices which are connected in the original graph to
    become disconnected).
    
    Args:
        None
    
    Returns:
    Set of ints containing precisely the indices of the vertices
    in the graph which are articulation points of graph.
    """
    # Need to check thoroughly
    # Gives correct answer for:
    #  {0: {1}, 1: {0, 2}, 2: {1}} - answer (1,)
    #  {0: {1,2}, 1: {0, 2}, 2: {0,1}} - answer ()
    #  {0: {1, 2, 3}, 1: {0, 2}, 2: {0, 1}, 3: {0, 4, 5},
    #        4: {3}, 5: {3}} - answer (3, 0)
    
    
    artic = set()
    lo = {}

    def dfs(idx, t: int, idx0: Optional[int]=None) -> int:
        if idx in lo.keys(): return t
        t0 = t
        lo[idx] = t
        curr_lo = t
        t += 1
        add = False
        for idx2 in self.adjGeneratorIndex(idx):
            if idx2 == idx0: continue
            if idx2 not in lo.keys():
                t = dfs(idx2, t, idx)
                if lo[idx2] >= t0:
                    add = True
            curr_lo = min(curr_lo, lo[idx2])
        if add: artic.add(idx)
        lo[idx] = curr_lo
        return t
    
    t = 0
    for idx in range(self.n):
        if idx in lo.keys(): continue
        lo[idx] = t
        child_count = 0
        for idx2 in self.adjGeneratorIndex(idx):
            if idx2 in lo.keys(): continue
            child_count += 1
            t = dfs(idx2, t + 1, idx0=idx)
        if child_count > 1: artic.add(idx)
    return artic

def tarjanArticulationBasic(self) -> Tuple[Hashable]:
    """
    Implementation of the Tarjan articulation point algorithm for
    undirected graphs. Identifies all articulation points in an
    undirected graph.
    An articulation point is a vertex of the graph which, if removed
    along with all of its associated edges increases the number of
    connected components of the graph (or equivalently, causes at least
    one pair of vertices which are connected in the original graph to
    become disconnected).
    
    Args:
        None
    
    Returns:
    Set of ints containing precisely the vertices in the graph which
    are articulation points of graph.
    """
    return set(map(self.index2Vertex,\
            self.tarjanArticulationBasicIndex()))

def checkArticulationBasicIndex(graph: "LimitedUndirectedGraphTemplate",\
        artic_idx: Set[int], check_all: bool=True) -> bool:
    """
    Function checking whether the vertices with graph indices
    contained in artic_idx are articulation points of graph.
    An articulation point is a vertex of the graph which, if removed
    along with all of its associated edges increases the number of
    connected components of the graph (or equivalently, causes at least
    one pair of vertices which are connected in the original graph to
    become disconnected).
    If check_all given as False, then only checks that the vertices
    given are indeed articulation points, otherwise also checks that
    all other vertices are not articulation points.
    
    Args:
        Required positional:
        graph (class descending from LimitedUndirectedGraphTemplate):
                The undirected graph for which the articulation points
                are being ascertained.
        artic_idx (set of ints): The indices of vertices in graph
                whose identity as articulation points of graph are to
                be tested.
        
        Optional named:
        check_all (bool): If False, only checks that the vertices
                represented by artic_idx are articulation points of
                graph, otherwise also checks that the other vertices
                of graph are not articulation points.
            Default: True
    
    Returns:
    Boolean (bool) with value True and all of the vertices with
    graph indices in artic_idx are articulation points of graph, and
    if check_all is given as True all other vertices of graph are not
    articulation points, and value False otherwise.
    """
    def dfs(idx1: int) -> bool:
        remain = set(graph.getAdjIndex(idx1).keys()).difference({idx1})
        if not remain: return False
        mn_adj_cnt = float("inf")
        for idx in remain:
            adj_cnt = len(graph.getAdjIndex(idx).keys())
            if adj_cnt < mn_adj_cnt:
                mn_adj_cnt = adj_cnt
                idx2 = idx
        remain.remove(idx2)
        seen = {idx1, idx2}
        stk = [idx2]
        while stk:
            idx = stk.pop()
            adj = graph.getAdjIndex(idx).keys()
            remain -= adj
            if not remain: return False
            for idx3 in adj:
                if idx3 in seen: continue
                seen.add(idx3)
                stk.append(idx3)
        return True
    
    if not check_all:
        for idx1 in artic_idx:
            if not dfs(idx1): return False
        return True
    for idx1 in range(graph.n):
        if dfs(idx1) != (idx1 in artic_idx):
            return False        
    return True

def checkArticulationBasic(graph: "LimitedUndirectedGraphTemplate",\
        artic: Set[Hashable], check_all: bool=True) -> bool:
    """
    Function checking whether the vertices of graph contained in
    artic are articulation points of graph.
    An articulation point is a vertex of the graph which, if removed
    along with all of its associated edges increases the number of
    connected components of the graph (or equivalently, causes at least
    one pair of vertices which are connected in the original graph to
    become disconnected).
    If check_all given as False, then only checks that the vertices
    given are indeed articulation points, otherwise also checks that
    all other vertices are not articulation points.
    
    Args:
        Required positional:
        graph (class descending from LimitedUndirectedGraphTemplate):
                The undirected graph for which the articulation points
                are being ascertained.
        artic (set of hashable objects): The vertices of graph whose
                identity as articulation points of graph are to be
                tested.
        
        Optional named:
        check_all (bool): If False, only checks that the vertices
                represented by artic are articulation points of graph,
                otherwise also checks that the other vertices of graph
                are not articulation points.
            Default: True
    
    Returns:
    Boolean (bool) with value True if all of the vertices of graph in
    artic are articulation points of graph, and if check_all is given
    as True all other vertices of graph are not articulation points,
    and value False otherwise.
    """
    artic_idx = {graph.vertex2Index(v) for v in artic}
    return checkArticulationBasicIndex(graph, artic_idx,\
            check_all=check_all)

def tarjanArticulationFullIndex(self)\
        -> Tuple[Union[Dict[int, List[Set[int]]], Dict[int, int],\
        int]]:
    """
    Modified Tarjan Algorithm for finding articulation points in
    the graph and for each articulation point identifying which
    adjacent vertices remain connected to each other after the
    removal of that articulation point. Also finds the vertices
    which are connected to each other.
    """
    
    artic = []
    lo = {}
    cc_dict = {} # Connected components
    artic = {} # Articulation points and the adjacent vertex groups
               # their removal disconnects
    lo = {}
    def dfs(idx: int, t: int, idx0: int, cc_i: int) -> int:
        if idx in lo.keys():
            return lo[idx]
        cc_dict[idx] = cc_i
        t0 = t
        lo[idx] = t
        curr_lo = t
        adj_remain = set(self.adjGeneratorIndex(idx))
        adj_remain.remove(idx0)
        adj_groups = [{idx0}]
        for idx2 in self.adjGeneratorIndex(idx):
            if idx2 not in adj_remain: continue
            elif idx2 in lo.keys():
                curr_lo = min(curr_lo, lo[idx2])
                adj_groups[0].add(idx2)
                adj_remain.remove(idx2)
                continue
            t = dfs(idx2, t + 1, idx, cc_i)
            curr_lo = min(curr_lo, lo[idx2])
            if idx2 not in adj_remain: continue
            if lo[idx2] < t0: j = 0
            else:
                j = -1
                adj_groups.append(set())
            rm_set = set()
            for idx3 in adj_remain:
                if idx3 in lo.keys():
                    adj_groups[j].add(idx3)
                    rm_set.add(idx3)
            adj_remain -= rm_set
        if len(adj_groups) > 1:
            artic[idx] = adj_groups
        lo[idx] = curr_lo
        return t
    
    t = 0
    cc_i = 0
    for idx in range(self.n):
        if idx in lo.keys(): continue
        cc_dict[idx] = cc_i
        lo[idx] = t
        adj_remain = set(self.adjGeneratorIndex(idx))
        adj_groups = []
        for idx2 in self.adjGeneratorIndex(idx):
            if idx2 not in adj_remain:
                continue
            t = dfs(idx2, t + 1, idx, cc_i)
            adj_groups.append(set())
            rm_set = set()
            for idx3 in adj_remain:
                if idx3 in lo.keys():
                    adj_groups[-1].add(idx3)
                    rm_set.add(idx3)
            adj_remain -= rm_set
        cc_i += 1
        if len(adj_groups) > 1:
            artic[idx] = adj_groups
    return (artic, cc_dict, cc_i)

def tarjanArticulationFull(self)\
        -> Tuple[Union[Dict[Hashable, List[Set[Hashable]]],\
        Dict[Hashable, int], int]]:
    artic_index, cc_dict_index, n_cc =\
            self.tarjanArticulationFullIndex()
    artic = {}
    for idx1, idx2_sets in artic_index.items():
        artic[self.index2Vertex(idx1)] = [{self.index2Vertex(idx2)\
                for idx2 in idx2_set} for idx2_set in idx2_sets]
    cc_dict = {self.index2Vertex(idx): cc_i\
            for idx, cc_i in cc_dict_index.items()}
    return artic, cc_dict, n_cc
