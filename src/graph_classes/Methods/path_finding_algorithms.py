#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Hashable,
    Callable,
)

if TYPE_CHECKING:
    from graph_classes import (
        LimitedGraphTemplate,
    )

import heapq
import inspect
import itertools

from abc import abstractmethod
from collections import deque
from sortedcontainers import SortedDict

# TODO- documentation

### Eulerian path and circuit finding algorithms ###

@abstractmethod
def _hierholzerIndex_abstract(self, start_idx: Optional[int],\
        end_idx: Optional[int], sort: bool=False,\
        reverse: bool=False) -> List[int]:
    # To be loaded as _hierholzerIndex()
    pass

def hierholzerIndex(self, start_idx: Optional[int]=None,\
        end_idx: Optional[int]=None, sort: bool=False,\
        reverse: bool=False) -> List[int]:
    """
    Implementation of the Hierholzer algorithm for finding an
    Eulerian path or cycle if one exists, with the option to
    specify either, both or neither of a start and end point.
    If such a path or cycle exists, it is expressed in terms
    of the indices of the vertices it traverses.
    An Eulerian path is a path through a graph (directed or
    undirected) that traverses every edge of the graph
    exactly once. An Eulerian cycle is an Eulerian path that
    starts and ends at the same vertex.

    This can be used to solve Leetcode #332, #1743 and #2097

    Args:
        Optional named:
        start_idx (int or None): The index of the vertex at
                which the path must begin, or None if the
                path may start at any vertex. 
            Default: None
        end_idx (int or None): The index of the vertex at
                which the path must end, or None if the
                path may end at any vertex.
            Default: None
        sort (bool): If given as True, the Eulerian path or
                cycle returned will at each step preferentially
                move to the lowest (if reverse is False) or
                highest (if reverse is True) index of the
                vertices reachable by an edge not yet traversed.
            Default: False
        reverse (bool): If sort is True, determines whether the
                each step in the Eulerian path or cycle should
                move to the lowest (if False) or highest (if True)
                index vertex still available.
            Default: False
    
    Returns:
    List of integers (int), giving the indices of the vertices
    encountered in a traversal of the identified Eulerian path or
    cycle in the order they are encountered if an Eulerian path
    or cycle fulfilling the specified restrictions exists,
    otherwise an empty list.
    The path given is an Eulerian cycle if and only if the
    first and last elements of the returned list are the same.
    """
    return self._hierholzerIndex(start_idx, end_idx, sort=sort,\
            reverse=reverse)

def hierholzer(self, start: Optional[Hashable]=None,\
        end: Optional[Hashable]=None, sort: bool=False,\
        reverse: bool=False) -> List[Hashable]:
    """
    Implementation of the Hierholzer algorithm for finding an
    Eulerian path or cycle if one exists, with the option to
    specify either, both or neither of a start and end point.
    If such a path or cycle exists, it is expressed in terms
    of the labels of the vertices it traverses.
    An Eulerian path is a path through a graph (directed or
    undirected) that traverses every edge of the graph
    exactly once. An Eulerian cycle is an Eulerian path that
    starts and ends at the same vertex.

    This can be used to solve Leetcode #1743

    Args:
        Optional named:
        start (hashable or None): The label of the vertex at
                which the path must begin, or None if the
                path may start at any vertex. 
            Default: None
        end (hashable or None): The label of the vertex at
                which the path must end, or None if the
                path may end at any vertex.
            Default: None
        sort (bool): If given as True, the Eulerian path or
                cycle returned will at each step preferentially
                move to the lowest (if reverse is False) or
                highest (if reverse is True) index of the
                vertices reachable by an edge not yet traversed.
            Default: False
        reverse (bool): If sort is True, determines whether the
                each step in the Eulerian path or cycle should
                move to the lowest (if False) or highest (if True)
                index vertex still available.
            Default: False
    
    Returns:
    List of hashable objects, giving the labels of the vertices
    encountered in a traversal of the identified Eulerian path or
    cycle in the order they are encountered if an Eulerian path
    or cycle fulfilling the specified restrictions exists,
    otherwise an empty list. 
    The path given is an Eulerian cycle if and only if the
    first and last elements of the returned list are the same.
    """
    start_idx = None if start is None else\
            self.vertex2Index(start)
    end_idx = None if end is None else\
            self.vertex2Index(end)
    res_index = self._hierholzerIndex(start_idx=start_idx,\
            end_idx=end_idx, sort=sort, reverse=reverse)
    return [self.index2Vertex(idx) for idx in res_index]

def _hierholzerIndex_directed(self, start_idx: Optional[int],\
            end_idx: Optional[int], sort: bool=False,\
            reverse: bool=False) -> List[int]:
    """
    Finding an Eulerian path for a digraph using Hierholzer's
    algorithm. If none exists, returns empty list
    Can be used to solve Leetcode #332, #2097
    """
    # To be loaded as _hierholzerIndex()
    
    # Checking if an Eulerian path is possible (if specified, starting
    # and start_idx and/or ending at end_idx) and if so whether it is
    # a cycle and if not where it must start/end
    extrema = [end_idx, start_idx]
    cycle = None if None in extrema else (extrema[0] == extrema[1])
    if self.store_in_degrees:
        diff_iter = (self.outDegreeIndex(idx) -\
                self.inDegreeIndex(idx) for idx in range(self.n))
    else:
        diff_iter = [0] * self.n
        for idx1 in range(self.n):
            for idx2, cnt in self.getAdjEdgeCountsIndex(idx1).items():
                diff_iter[idx1] += cnt
                diff_iter[idx2] -= cnt
    nonzero = [None, None]
    for idx, diff in enumerate(diff_iter):
        if not diff: continue
        elif cycle or abs(diff) > 1: return []
        i = (diff + 1) >> 1
        if nonzero[i] is not None or\
                (extrema[i] is not None and idx != extrema[i]):
            return []
        nonzero[i] = idx
    if all(x is None for x in nonzero):
        if cycle is None:
            if extrema[0] is not None: extrema[1] = extrema[0]
            elif extrema[1] is not None: extrema[0] = extrema[1]
            else: extrema = [self.n - 1 if reverse else 0] * 2
            cycle = True
        elif not cycle: return []
    else:
        extrema = nonzero
        cycle = False
    start_idx = extrema[1]
    
    out_edge_counts = []
    tot_edges = 0
    for idx in range(self.n):
        lst = []
        for idx2, cnt in self.getAdjEdgeCountsIndex(idx).items():
            lst.append([idx2, cnt])
            tot_edges += cnt
        if sort: lst.sort(reverse=not reverse)
        out_edge_counts.append(lst)
    
    res = []
    def dfs(idx: int) -> List[int]:
        while out_edge_counts[idx]:
            idx2 = out_edge_counts[idx][-1][0]
            out_edge_counts[idx][-1][1] -= 1
            if not out_edge_counts[idx][-1][1]:
                out_edge_counts[idx].pop()
            dfs(idx2)
        res.append(idx)
        return

    dfs(start_idx)
    if len(res) != tot_edges + 1:
        # Graph is not connected
        return []
    return res[::-1]

def _hierholzerIndex_undirected(self, start_idx: Optional[int],\
            end_idx: Optional[int], sort: bool=False,\
            reverse: bool=False) -> List[int]:
    """
    Finding an Eulerian path for an undirected graph using
    Hierholzer's algorithm. If none exists, returns empty list.
    
    Need to test properly- have used simple test cases from
    Leetcode #332 with undirected graph and seems to give the
    expected results
    """
    # To be loaded as _hierholzerIndex()
    
    # Checking if an Eulerian path is possible (if specified,
    # starting and start_idx and/or ending at end_idx) and if so
    # whether it is a cycle and if not where it must start/end
    extrema = [end_idx, start_idx]
    cycle = None if None in extrema else (extrema[0] == extrema[1])
    odd_set = {x for x in extrema if x is not None}
    for idx in range(self.n):
        deg = self.degreeIndex(idx)
        if not deg & 1:
            if cycle is False and idx in odd_set:
                return []
            continue
        elif cycle:
            return []
        odd_set.add(idx)
        if len(odd_set) > 2:
            return []
    if cycle is False and len(odd_set) < 2:
        return []
    elif cycle is None: cycle = (len(odd_set) < 2)
    if start_idx is None:
        if cycle: start_idx = self.n - 1 if reverse else 0
        elif sort:
            start_idx = max(odd_set) if reverse else min(odd_set)
        else: start_idx = next(iter(odd_set))
    edge_counts = []
    tot_edges = 0
    for idx in range(self.n):
        cnt_dict = SortedDict() if sort else {}
        for idx2, cnt in self.getAdjEdgeCountsIndex(idx).items():
            cnt_dict[idx2] = cnt
            tot_edges += cnt
        edge_counts.append(cnt_dict)
    tot_edges >>= 1
    res = []
    
    select_i = -1 if reverse else 0
    select_next = lambda idx: edge_counts[idx].peekitem(select_i)\
            if sort else\
            lambda idx: next(iter(edge_counts[idx].items()))
    
    def dfs(idx: int) -> List[int]:
        while edge_counts[idx]:
            idx2, cnt = select_next(idx)
            for j1, j2 in ((idx, idx2), (idx2, idx)):
                edge_counts[j1][j2] -= 1
                if not edge_counts[j1][j2]:
                    edge_counts[j1].pop(j2)
            dfs(idx2)
        res.append(idx)
        return

    dfs(start_idx)
    if len(res) != tot_edges + 1:
        # Graph is not connected
        return []
    return res[::-1]

### Shortest path finding algorithms ###
    
## Algorithms finding shortest paths between two sets of vertices ##

def _pathIndex2Vertex(self, path_inds: List[int])\
        -> List[Hashable]:
    return tuple(map(self.index2Vertex, path_inds))

def _transitionToBFS(self, arg_pair: Tuple[Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]]],\
        from_vertices: bool, bidirectional: bool)\
        -> Optional[Tuple[Union[int, float], Tuple[int]]]:
    if not hasattr(self, "bredthFirstSearch"):
        return None
    inds_sets = [None, None]
    mod_func = self._argVertices2IndexSet if from_vertices else\
            self._argIndices2IndexSet
    d1 = 0
    for i, arg in enumerate(arg_pair):
        inds_sets[i], d = mod_func(arg, check_dists=True)
        if d is None: return None
        d1 += d
    search_func = self._bredthFirstSearchBidirectionalIndex\
            if bidirectional\
            else self._bredthFirstSearchUnidirectionalIndex
    d2, path = search_func(*inds_sets)
    return (-1 if d2 == -1 else d1 + d2, path)

def _bredthFirstSearchUnidirectionalIndex(self,\
        start_inds: Set[int], end_inds: Set[int])\
        -> Tuple[Union[Optional[int], Tuple[Hashable]]]:
    # Assumes start_inds and end_inds are not empty
    print(f"Using {inspect.stack()[0][3]}()")
    #if not start_inds or not end_inds:
    #    return (None, ())
    intersect = start_inds.intersection(end_inds)
    if intersect:
        # If there is at least one vertex that is in both starts
        # and ends
        return (0, (next(iter(intersect)),))
    qu = deque()
    seen = {}
    for idx in start_inds:
        qu.append(idx)
        seen[idx] = -1
    for cnt in itertools.count(1):
        if not qu: break
        for _ in range(len(qu)):
            idx1 = qu.popleft()
            for idx2 in self.getAdjIndex(idx1).keys():
                if idx2 in seen.keys(): continue
                seen[idx2] = idx1
                if idx2 in end_inds:
                    path = []
                    j = idx2
                    while j != -1:
                        path.append(j)
                        j = seen[j]
                    return (cnt, tuple(path[::-1]))
                qu.append(idx2)
    return (None, ())

def _bredthFirstSearchBidirectionalIndex(self,\
        start_inds: Set[int], end_inds: Set[int])\
        -> Tuple[Union[Optional[int], Tuple[Hashable]]]:
    # Assumes start_inds and end_inds are not empty
    print(f"Using {inspect.stack()[0][3]}()")
    #if not start_inds or not end_inds:
    #    return (None, ())
    intersect = start_inds.intersection(end_inds)
    if intersect:
        # If there is at least one vertex that is in both starts
        # and ends
        return (0, (next(iter(intersect)),))
    adj_funcs = [self.getAdjIndex, self.getInAdjIndex]
    qus = [deque(), deque()]
    seen = [{}, {}]
    for qu_i, inds in enumerate((start_inds, end_inds)):
        for idx in inds:
            qus[qu_i].append(idx)
            seen[qu_i][idx] = -1
    for cnt in itertools.count(1):
        for qu_i, qu in enumerate(qus):
            qu_i2 = 1 - qu_i
            for _ in range(len(qu)):
                idx1 = qu.popleft()
                for idx2 in adj_funcs[qu_i](idx1).keys():
                    if idx2 in seen[qu_i].keys(): continue
                    seen[qu_i][idx2] = idx1
                    if idx2 in seen[qu_i2]:
                        idx_mid = idx2
                        curr_d = (cnt << 1) + qu_i - 1
                        break
                    qu.append(idx2)
                else: continue
                break
            else:
                if not qu:
                    return (None, ())
                continue
            break
        else: continue
        break
    path = []
    curr = idx_mid
    while curr != -1:
        path.append(curr)
        curr = seen[0][curr]
    path = path[::-1]
    curr = -1 if idx_mid == -1 else seen[1][idx_mid]
    while curr != -1:
        path.append(curr)
        curr = seen[1][curr]
    return (curr_d, tuple(path))

def bredthFirstSearchIndex(self, start_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        end_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        bidirectional: bool=True)\
        -> Tuple[Union[Optional[Union[int, float]], Tuple[int]]]:
    if not start_inds or not end_inds:
        return (None, ())
    func = self._bredthFirstSearchBidirectionalIndex\
            if bidirectional else\
            self._bredthFirstSearchUnidirectionalIndex
    inds_sets = [start_inds, end_inds]
    inds_names = ["start_inds", "end_inds"]
    d1 = 0
    for i, nm in enumerate(inds_names):
        inds_sets[i], d = self._argIndices2IndexSet(inds_sets[i],\
                check_dists=True)
        if d is None:
            raise ValueError(f"If {nm} is given as a dictionary, "\
                    "each of its values must be the same.")
        d1 += d
    d2, path = func(*inds_sets)
    return (-1 if d2 == -1 else d1 + d2, path)

def bredthFirstSearch(self, starts: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        ends: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        bidirectional: bool=True)\
        -> Tuple[Union[Optional[int], Tuple[Hashable]]]:
    """
    Performs a bredth first search for a minimum length path
    between any of the set of vertices given by starts and any of
    the set of vertices given by ends in the graph, if any paths
    between vertices in these two sets exists.
    The length of a path is the number of edges traversed in that
    path. A minimum length path between two sets of vertices is
    a path between a vertex in the first set and a vertex in the
    second set such that there does not exist any other such
    path whose length is smaller.
    
    Args:
        Required positional:
        starts (set of hashable objects): Set containing a subset
                of the elements of the attribute vertices,
                representing the vertices from which the paths
                considered may start.
        ends (set of hashable objects): Set containing a subset
                of the elements of the attribute vertices,
                representing the vertices from which the paths
                considered may end.
    
    Returns:
    2-tuple whose index 0 contains an integer (int) giving the
    shortest length of any path between any of the vertices in
    starts and any of the vertices in ends or -1 if no such path
    exists, and whose index 1 contains a tuple with the vertices on
    such a path with this length in the order they are encountered
    on the path, or an empty tuple if no such path exists.
    """
    if not all(self.vertexInGraph(x) for x in starts):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument starts must be "\
                "vertices of the graph.")
    elif not all(self.vertexInGraph(x) for x in ends):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument ends must be "\
                "vertices of the graph.")
    if not starts or not ends:
        return (None, ())
    func = self._bredthFirstSearchBidirectionalIndex\
            if bidirectional else\
            self._bredthFirstSearchUnidirectionalIndex
    v_sets = [starts, ends]
    v_names = ["starts", "ends"]
    d1 = 0
    for i, nm in enumerate(v_names):
        v_sets[i], d = self._argVertices2IndexSet(v_sets[i],\
                check_dists=True)
        if d is None:
            raise ValueError(f"If {nm} is given as a dictionary, "\
                    "each of its values must be the same.")
        d1 += d
    d2, path_inds = func(*v_sets)
    return (-1 if d2 == -1 else d1 + d2,\
            self._pathIndex2Vertex(path_inds))

def dijkstraApplicableIndex(self,\
        start_inds: Dict[int, Union[int, float]],\
        end_inds: Optional[Dict[int, Union[int, float]]]) -> bool:
    if not self.neg_weight_edge: return True
    if not any(self.negativeWeightAheadIndex(idx)\
            for idx in start_inds.keys()):
        return True
    elif end_inds is None: return False
    return not any (self.negativeWeightBeindIndex(idx)\
            for idx in end_inds.keys())

def _dijkstraUnidirectionalIndex(self,\
        start_inds: Dict[int, Union[int, float]],\
        end_inds: Dict[int, Union[int, float]])\
        -> Tuple[Union[Optional[Union[int, float]], Tuple[int]]]:
    # Assumes no negative weight edges and start_inds and end_inds
    # are not empty

    print(f"Using {inspect.stack()[0][3]}()")
    #if not start_inds or not end_inds:
    #    return (None, ())
    intersect = set(start_inds).intersection(end_inds)
    if intersect:
        # If there is at least one vertex that is in both
        # start_inds and end_inds
        return (0, (next(iter(intersect)),))
    heap = []
    for (idx, d) in start_inds.items():
        heap.append((d, idx, -1))
    heapq.heapify(heap)
    adj_min_wt_func = self.getAdjMinimumWeightsIndex
    seen = {}
    while heap:
        d1, idx1, idx0 = heapq.heappop(heap)
        if idx1 in seen.keys(): continue
        seen[idx1] = idx0
        if idx1 in end_inds.keys():
            path = []
            curr = idx1
            while curr != -1:
                path.append(curr)
                curr = seen[curr]
            return (d1, tuple(path[::-1]))
        for idx2, d2 in adj_min_wt_func(idx1).items():
            if idx2 in seen.keys(): continue
            heapq.heappush(heap, (d1 + d2 + end_inds.get(idx2, 0),\
                    idx2, idx1))
    return (None, ())

def _dijkstraBidirectionalIndex(self,\
        start_inds: Dict[int, Union[int, float]],\
        end_inds: Dict[int, Union[int, float]])\
        -> Tuple[Union[Optional[Union[int, float]], Tuple[int]]]:
    # Assumes no negative weight edges and start_inds and end_inds
    # are not empty
    
    print(f"Using {inspect.stack()[0][3]}()")
    #if not start_inds or not end_inds:
    #    return (None, ())
    intersect = set(start_inds).intersection(end_inds)
    if intersect:
        # If there is at least one vertex that is in both
        # start_inds and end_inds
        return (0, (next(iter(intersect)),))
    adj_min_wt_funcs = [self.getAdjMinimumWeightsIndex,\
            self.getInAdjMinimumWeightsIndex]
    heap = []
    for (idx, d) in start_inds.items():
        heap.append((d, idx, -1, 0))
    for (idx, d) in end_inds.items():
        heap.append((d, idx, -1, 1))
    heapq.heapify(heap)
    seen = [{}, {}]
    target_d = float("inf")
    curr_d = float("inf")
    found = False
    while heap:
        d1, idx1, idx0, j = heapq.heappop(heap)
        if idx1 in seen[j].keys(): continue 
        j2 = 1 - j
        if idx1 in seen[j2].keys() and\
                d1 + seen[j2][idx1][1] < curr_d:
            curr_d = d1 + seen[j2][idx1][1]
            target_d = (curr_d + 1) >> 1
            idx_mid = idx1
            found = True
        seen[j][idx1] = (idx0, d1)
        if d1 >= target_d: continue
        for idx2, d2 in adj_min_wt_funcs[j](idx1).items():
            if idx2 in seen[j].keys(): continue
            heapq.heappush(heap, (d1 + d2, idx2, idx1, j))
    if not found: return (None, ())
    path = []
    curr = idx_mid
    while curr != -1:
        path.append(curr)
        curr = seen[0][curr][0]
    path = path[::-1]
    curr = -1 if idx_mid == -1 else seen[1][idx_mid][0]
    while curr != -1:
        path.append(curr)
        curr = seen[1][curr][0]
    return (curr_d, tuple(path))

def dijkstraIndex(self, start_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        end_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        bidirectional: bool=True, use_bfs_if_poss: bool=True)\
        -> Tuple[Union[int, float], Tuple[int]]:
    if self.neg_weight_edge:
        raise ValueError("The method "\
                f"{inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight edges.")
    elif not all(0 <= idx < self.n for idx in start_inds):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the indices "\
                "given in input argument start_inds must be "\
                "non-negative integers strictly less than the "\
                "attribute n (the number of vertices in the "\
                "graph, which in this case has the value "\
                f"{self.n}).")
    elif not all(0 <= idx < self.n for idx in end_inds):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the indices "\
                "given in input argument end_inds must be "\
                "non-negative integers strictly less than the "\
                "attribute n (the number of vertices in the "\
                "graph, which in this case has the value "\
                f"{self.n}).")
    if use_bfs_if_poss:
        res = self._transitionToBFS((start_inds, end_inds),\
                from_vertices=False, bidirectional=bidirectional)
        if res is not None: return res
    """
    if use_bfs_if_poss and hasattr(self, "bredthFirstSearchIndex"):
        inds_sets = [start_inds, end_inds]
        inds_names = ["start_inds", "end_inds"]
        d1 = 0
        for i, nm in enumerate(inds_names):
            inds_sets[i], d =\
                    self._argIndices2IndexSet(inds_sets[i],\
                    check_dists=True)
            if d is None: break
            d1 += d
        else:
            func = self._bredthFirstSearchBidirectionalIndex\
                    if bidirectional\
                    else self._bredthFirstSearchUnidirectionalIndex
            d2, path = func(*inds_sets)
            return (-1 if d2 == -1 else d1 + d2, path)
    """
    func = self._dijkstraBidirectionalIndex if bidirectional else\
            self._dijkstraUnidirectionalIndex
    start_inds = self._argIndices2IndexDict(start_inds)
    end_inds = self._argIndices2IndexDict(end_inds)
    return func(start_inds, end_inds)

def dijkstra(self, starts: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        ends: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        bidirectional: bool=True, use_bfs_if_poss: bool=True)\
        -> Tuple[Union[int, float], Tuple[Hashable]]:
    """
    Performs a Dijkstra search for a minimum cost path between
    any of the set of vertices given by starts and any of the set
    of vertices given by ends in the graph. If argument
    bidirectional is given as True, a bidirectional Dijkstra search
    from both the vertices in starts and the vertices in ends
    vertices is performed, otherwise a unidirectional Dijkstra
    search from the vertices in starts is performed.
    The cost of a path is the sum of the weights of the (directed)
    edges that it traverses, plus the weight associated with the
    vertices at which the path begins and ends (as specified by
    the value corresponding to these vertices in the dictionaries
    starts and ends respectively).
    This may only be used if the graph contains no edges with
    negative weight (i.e. the attribute neg_weight_edge is False).
    If the graph contains at least one edge with negative weight
    (i.e. the attribute neg_weight_edge is True) and this method
    is called then an error will be raised.
    
    Args:
        Required positional:
        starts (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may start,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        ends (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may end,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        
        Optional named:
        bidirectional (bool): Whether the Dijkstra search should be
                bidirectional (True) or unidirectional (False).
            Default: True
        
    Returns:
    2-tuple whose index 0 contains a numeric value (int or float)
    giving the minimum cost for any path between any of the
    vertices in the keys of starts and any of the vertices in the
    keys of ends or -1 if no such path exists, and whose index
    1 contains a tuple with the vertices on such a path with this
    cost in the order they are encountered on the path, or an empty
    tuple if no such path exists.
    """
    if self.neg_weight_edge:
        raise ValueError("The method "\
                f"{inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight edges.")
    elif not all(self.vertexInGraph(x) for x in starts):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument starts must be "\
                "vertices of the graph.")
    elif not all(self.vertexInGraph(x) for x in ends):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument ends must be "\
                "vertices of the graph.")
    if use_bfs_if_poss:
        res = self._transitionToBFS((starts, ends),\
                from_vertices=True, bidirectional=bidirectional)
        if res is not None: return res
    func = self._dijkstraBidirectionalIndex if bidirectional else\
            self._dijkstraUnidirectionalIndex
    start_inds = self._argVertices2IndexDict(starts)
    end_inds = self._argVertices2IndexDict(ends)
    d, path_inds = func(start_inds, end_inds)
    return (d, self._pathIndex2Vertex(path_inds))

def _aStarUnidirectionalIndex(self,\
        start_inds: Dict[int, Tuple[Union[int, float]]],\
        end_inds: Dict[int, Tuple[Union[int, float]]],\
        heuristic: Callable[[int, int, Union[int, float]],\
        Union[int, float]])\
        -> Tuple[Union[Optional[Union[int, float]], Tuple[int]]]:
    # Assumes no negative weight edges and start_inds and end_inds
    # are not empty

    print(f"Using {inspect.stack()[0][3]}()")
    #if not start_inds or not end_inds:
    #    return (None, ())
    intersect = set(start_inds).intersection(end_inds)
    if intersect:
        # If there is at least one vertex that is in both
        # start_inds and end_inds
        return (0, (next(iter(intersect)),))
    adj_min_wt_func = self.getAdjMinimumWeightsIndex
    heur_func = lambda idx, neg_d: min(heuristic(idx, end_idx)\
            for end_idx in end_inds) - neg_d
    heap = []
    for idx, d in start_inds.items():
        heap.append((heur_func(idx, -d), -d, idx, -1))
    heapq.heapify(heap)
    
    seen = {}
    while heap:
        _, neg_d1, idx1, idx0 = heapq.heappop(heap)
        if idx1 in seen.keys(): continue
        seen[idx1] = idx0
        if idx1 in end_inds.keys():
            path = []
            curr = idx1
            while curr != -1:
                path.append(curr)
                curr = seen[curr]
            return (-neg_d1, tuple(path[::-1]))
        for idx2, d2_ in adj_min_wt_func(idx1).items():
            if idx2 in seen.keys(): continue
            neg_d2 = neg_d1 - d2_
            heapq.heappush(heap, (heur_func(idx2, neg_d2), neg_d2,\
                    idx2, idx1))
    return (None, ())

def _aStarBidirectionalIndex(self,\
        start_inds: Dict[int, Union[int, float]],\
        end_inds: Dict[int, Union[int, float]],
        heuristic: Callable[[int, int], Union[int, float]])\
        -> Tuple[Union[Optional[Union[int, float]], Tuple[int]]]:
    # Assumes no negative weight edges and start_inds and end_inds
    # are not empty

    print(f"Using {inspect.stack()[0][3]}()")
    #if not start_inds or not end_inds:
    #    return (None, ())
    intersect = set(start_inds).intersection(end_inds)
    if intersect:
        # If there is at least one vertex that is in both starts
        # and ends
        return (0, (next(iter(intersect)),))
    adj_min_wt_funcs = [self.getAdjMinimumWeightsIndex,\
            self.getInAdjMinimumWeightsIndex]
    heur_funcs = [lambda idx, neg_d: min(heuristic(idx, end_idx)\
            for end_idx in end_inds) - neg_d,\
            lambda idx, neg_d: min(heuristic(start_idx, idx)\
            for start_idx in start_inds) - neg_d]
    heap = []
    for j, inds in enumerate([start_inds, end_inds]):
        for idx, d in inds.items():
            heap.append((heur_funcs[j](idx, -d), -d, idx, -1, j))
    heapq.heapify(heap)
    seen = [{}, {}]
    target_d = float("inf")
    curr_d = float("inf")
    found = False
    while heap:
        h, neg_d1, idx1, idx0, j = heapq.heappop(heap)
        if idx1 in seen[j].keys(): continue
        j2 = 1 - j
        if idx1 in seen[j2].keys() and\
                seen[j2][idx1][1] - neg_d1 < curr_d:
            curr_d = seen[j2][idx1][1] - neg_d1
            target_d = (curr_d + 1) >> 1
            idx_mid = idx1
            found = True
        seen[j][idx1] = (idx0, -neg_d1)
        if h >= target_d: continue
        for idx2, d2_ in adj_min_wt_funcs[j](idx1).items():
            if idx2 in seen[j].keys(): continue
            neg_d2 = neg_d1 - d2_
            heapq.heappush(heap, (heur_funcs[j](idx2, neg_d2),\
                    neg_d2, idx2, idx1, j))
    if not found: return (None, ())
    path = []
    curr = idx_mid
    while curr != -1:
        path.append(curr)
        curr = seen[0][curr][0]
    path = path[::-1]
    curr = -1 if idx_mid == -1 else seen[1][idx_mid][0]
    while curr != -1:
        path.append(curr)
        curr = seen[1][curr][0]
    return (curr_d, tuple(path))

def aStarIndex(self, start_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        end_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        heuristic: Callable[[int, int], Union[int, float]],\
        bidirectional: bool=True)\
        -> Tuple[Union[int, float], Tuple[int]]:
    if self.neg_weight_edge:
        raise ValueError("The method "\
                f"{inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight edges.")
    elif not all(0 <= idx < self.n for idx in start_inds):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the indices "\
                "given in input argument start_inds must be "\
                "non-negative integers strictly less than the "\
                "attribute n (the number of vertices in the "\
                "graph, which in this case has the value "\
                f"{self.n}).")
    elif not all(0 <= idx < self.n for idx in end_inds):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the indices "\
                "given in input argument end_inds must be "\
                "non-negative integers strictly less than the "\
                "attribute n (the number of vertices in the "\
                "graph, which in this case has the value "\
                f"{self.n}).")
    func = self._aStarBidirectionalIndex if bidirectional else\
            self._aStarUnidirectionalIndex
    start_inds = self._argIndices2IndexDict(start_inds)
    end_inds = self._argIndices2IndexDict(end_inds)
    return func(start_inds, end_inds, heuristic)

def aStar(self, starts: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        ends: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        heuristic: Callable[[Hashable, Hashable],\
        Union[int, float]],\
        bidirectional: bool=True)\
        -> Tuple[Union[int, float], Tuple[Hashable]]:
    """
    Performs an A* search for a minimum cost path between
    any of the set of vertices given by starts and any of the set
    of vertices given by ends in the graph. If argument
    bidirectional is given as True, a bidirectional A* search
    from both the vertices in starts and the vertices in ends
    vertices is performed, otherwise a unidirectional A*
    search from the vertices in starts is performed.
    The cost of a path is the sum of the weights of the (directed)
    edges that it traverses, plus the weight associated with the
    vertices at which the path begins and ends (as specified by
    the value corresponding to these vertices in the dictionaries
    starts and ends respectively).
    This may only be used if the graph contains no edges with
    negative weight (i.e. the attribute neg_weight_edge is False).
    If the graph contains at least one edge with negative weight
    (i.e. the attribute neg_weight_edge is True) and this method
    is called then an error will be raised.
    
    Args:
        Required positional:
        starts (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may start,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        ends (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may end,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        heuristic (function): A heuristic function that takes
                two vertices as inputs and outputs an upper
                bound for the cost of any path between them.
        
        Optional named:
        bidirectional (bool): Whether the A* search should be
                bidirectional (True) or unidirectional (False).
            Default: True
    
    Returns:
    2-tuple whose index 0 contains a numeric value (int or float)
    giving the minimum cost for any path between any of the
    vertices in the keys of starts and any of the vertices in the
    keys of ends or -1 if no such path exists, and whose index
    1 contains a tuple with the vertices on such a path with this
    cost in the order they are encountered on the path, or an empty
    tuple if no such path exists.
    """
    if self.neg_weight_edge:
        raise ValueError("The method "\
                f"{inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight edges.")
    elif not all(self.vertexInGraph(x) for x in starts):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument starts must be "\
                "vertices of the graph.")
    elif not all(self.vertexInGraph(x) for x in ends):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument ends must be "\
                "vertices of the graph.")
    func = self._aStarBidirectionalIndex if bidirectional else\
            self._aStarUnidirectionalIndex
    heuristic_index = lambda idx1, idx2:\
            heuristic(*list(map(self.index2Vertex, (idx1, idx2))))
    start_inds = self._argVertices2IndexDict(starts)
    end_inds = self._argVertices2IndexDict(ends)
    d, path_inds = func(start_inds, end_inds, heuristic_index)
    return (d, self._pathIndex2Vertex(path_inds))

def _findShortestPathIndex(self, start_objs: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        end_objs: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        heuristic_idx: Optional[Callable[[int, int],\
        Union[int, float]]],\
        bidirectional: bool, from_vertices: bool)\
        -> Optional[Tuple[Union[Optional[Union[int, float]],\
        Tuple[int]]]]:
    
    arg_prep_func = self._argVertices2IndexDict if from_vertices\
            else self._argIndices2IndexDict
    if self.neg_weight_edge:
        # Consider putting this in its own method
        dist_prev_dict_idx = {x: (y, -1) for x, y in\
                arg_prep_func(start_objs).items()}
        dist_prev_dict_idx =\
                self._shortestPathFasterAlgorithmPathfinderIndex(\
                dist_prev_dict_idx)
        if not dist_prev_dict_idx: return None
        end_inds = arg_prep_func(end_objs)
        d_mn = float("inf")
        end_idx = None
        for idx, d_end in end_inds.items():
            if idx not in dist_prev_dict_idx.keys():
                continue
            d = dist_prev_dict_idx[idx][0] + d_end
            if d >= d_mn: continue
            d_mn = d
            end_idx = idx
        if end_idx is None: return (None, ())
        path = [end_idx]
        while path[-1] != -1:
            path.append(dist_prev_dict_idx[path[-1]][1])
        path.pop()
        return (d_mn, tuple(path[::-1]))
    kwargs = {}
    if heuristic_idx is not None:
        kwargs["heuristic"] = heuristic_idx
        search_func = self._aStarBidirectionalIndex\
                if bidirectional else\
                self._aStarUnidirectionalIndex
    else:
        res = self._transitionToBFS((start_objs, end_objs,),\
                from_vertices, bidirectional)
        if res is not None: return res
        search_func = self._dijkstraBidirectionalIndex\
                if bidirectional else\
                self._dijkstraUnidirectionalIndex
    start_inds, end_inds = list(map(arg_prep_func,\
            (start_objs, end_objs)))
    return search_func(start_inds, end_inds, **kwargs)

def findShortestPathIndex(self, start_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        end_inds: Union[Set[int],\
        Dict[int, Tuple[Union[int, float]]]],\
        heuristic: Optional[Callable[[int, int],\
        Union[int, float]]]=None,\
        bidirectional: bool=True)\
        -> Tuple[Union[Optional[Union[int, float]], Tuple[int]]]:

    if not all(0 <= idx < self.n for idx in start_inds):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the indices "\
                "given in input argument start_inds must be "\
                "non-negative integers strictly less than the "\
                "attribute n (the number of vertices in the "\
                "graph, which in this case has the value "\
                f"{self.n}).")
    elif not all(0 <= idx < self.n for idx in end_inds):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the indices "\
                "given in input argument end_inds must be "\
                "non-negative integers strictly less than the "\
                "attribute n (the number of vertices in the "\
                "graph, which in this case has the value "\
                f"{self.n}).")
    neg_cycle_err_msg = f"The method {inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight cycles."
    if self.neg_weight_cycle: raise ValueError(neg_cycle_err_msg)
    res = self._findShortestPathIndex(start_inds, end_inds,\
            heuristic, bidirectional=bidirectional)
    if res[0] is None and self.neg_weight_cycle:
        raise ValueError(neg_cycle_err_msg)
    return res
    
def findShortestPath(self, starts: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        ends: Union[Set[Hashable],\
        Dict[Hashable, Tuple[Union[int, float]]]],\
        heuristic: Optional[Callable[[Hashable, Hashable],\
        Union[int, float]]]=None,\
        bidirectional: bool=True)\
        -> Tuple[Union[Optional[Union[int, float]],\
        Tuple[Hashable]]]:
    """
    Performs an search for a minimum cost path between
    any of the set of vertices given by starts and any of the set
    of vertices given by ends in the graph, using either Dijkstra
    algorithm, A* algorithm or SPFA (Shortest Path Faster Algorithm)
    depending on whether the graph contains any negative weight
    edges and whether a heuristic is given.
    The SPFA will be chosen only if the graph contains any negative
    weight edges. If there are no negative weight edges, then the
    A* algorithm will be chosen if a heuristic function is given,
    otherwise the Dijkstra algorithm will be used.
    If argument bidirectional is given as True and the Dijkstra
    or A* algorithm is chosen (i.e. there are no negative weight
    edges in the graph), a bidirectional version of the chosen
    algorithm will be used, where a search from both the vertices
    in starts and the vertices in ends vertices is performed,
    otherwise a unidirectional A* search from the vertices in starts
    is performed.
    The cost of a path is the sum of the weights of the (directed)
    edges that it traverses, plus the weight associated with the
    vertices at which the path begins and ends (as specified by
    the value corresponding to these vertices in the dictionaries
    starts and ends respectively).
    This may only be used if the graph contains no negative weight
    cycles (i.e. a cycle whose sum of weights is negative). If the
    graph contains any negative weight cycle and this method is called
    then an error will be raised.
    
    Args:
        Required positional:
        starts (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may start,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        ends (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may end,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        
        
        Optional named:
        heuristic (function or None): If specified, a heuristic
                function that takes two vertices as inputs and
                outputs an upper bound for the cost of any path
                between them.
            Default: None
        bidirectional (bool): Whether any A* or Dijkstra search
                (if chosen) search should be bidirectional (True) or
                unidirectional (False).
            Default: True
    
    Returns:
    2-tuple whose index 0 contains a numeric value (int or float)
    giving the minimum cost for any path between any of the
    vertices in the keys of starts and any of the vertices in the
    keys of ends or -1 if no such path exists, and whose index
    1 contains a tuple with the vertices on such a path with this
    cost in the order they are encountered on the path, or an empty
    tuple if no such path exists.
    """
    if not all(self.vertexInGraph(x) for x in starts):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument starts must be "\
                "vertices of the graph.")
    elif not all(self.vertexInGraph(x) for x in ends):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), all the "\
                "vertices given in input argument ends must be "\
                "vertices of the graph.")
    neg_cycle_err_msg = f"The method {inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight cycles."
    if self.neg_weight_cycle: raise ValueError(neg_cycle_err_msg)
    heuristic_index = None if heuristic is None else\
            (lambda idx1, idx2:\
            heuristic(*list(map(self.index2Vertex,\
            (idx1, idx2)))))
    d, path_inds = self._findShortestPathIndex(starts, ends,\
            heuristic_index, bidirectional=bidirectional,\
            from_vertices=True)
    if d is None and self.neg_weight_cycle:
        raise ValueError(neg_cycle_err_msg)
    return (d, self._pathIndex2Vertex(path_inds))



## Algorithms finding shortest paths/distances from a single ##
## source or set of sources to all other vertices            ##

def _fromSourcesDistPathIndex2Vertex(self,\
        dist_prev_dict_idx: Dict[int, Tuple[Union[int, float]]])\
        -> Dict[Hashable, Tuple[Union[int, float, Hashable]]]:
    return {self.index2Vertex(idx1):\
            (tup[0], None if tup[1] == -1 else\
            self.index2Vertex(tup[1]))\
            for idx1, tup in dist_prev_dict_idx.items()}

def _fromSourcesDistIndex2Vertex(self,\
        dist_dict_idx: Dict[int, Union[int, float]])\
        -> Dict[Hashable, Union[int, float]]:
    return {self.index2Vertex(idx): d\
            for idx, d in dist_dict_idx.items()}

def _dijkstraFromSourcesPathfinderIndex(self,\
        source_inds: Dict[int, Union[int, float]],\
        _weight_mod_func: Optional[\
        Callable[[int, int, Union[int, float]],\
        Union[int, float]]]=None)\
        -> Dict[int, Tuple[Union[int, float]]]:
    # Assumes the graph (after applying the modification with
    # _weight_mod_func to the edge weights if applicable) has no
    # negative weight edges
    # _weight_mod_func is used to adapt the algorithm for use
    # as part of Johnson's algorithm (see methods johnsonIndex()
    # and johnson() below). In that case, the condition of no
    # negative weight edges (once _weight_mod_func has been applied
    # to all edge weights) is guaranteed to be satisfied
    adj_min_wt_func = self.getAdjMinimumWeightsIndex
    if _weight_mod_func is None:
        _weight_mod_func = lambda idx1, idx2, wt: wt
    heap = [(w, source_idx, -1)\
            for source_idx, w in source_inds.items()]
    heapq.heapify(heap)
    dist_prev_dict = {}
    while heap:
        d1, idx1, idx0 = heapq.heappop(heap)
        if idx1 in dist_prev_dict.keys(): continue
        dist_prev_dict[idx1] = (d1, idx0)
        for idx2, d2 in adj_min_wt_func(idx1).items():
            if idx2 in dist_prev_dict.keys(): continue
            heapq.heappush(heap,\
                    (d1 + _weight_mod_func(idx1, idx2, d2),\
                    idx2, idx1))
    return dist_prev_dict

def dijkstraFromSourcesPathfinderIndex(self,\
        source_inds: Dict[int, Union[int, float]])\
        -> Dict[int, Tuple[Union[int, float]]]:
    if self.neg_weight_edge:
        raise ValueError("The method "\
                f"{inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight edges.")
    elif source_inds and (min(source_inds.keys()) < 0 or\
            max(source_inds.keys()) >= self.n):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), input argument "\
                "source_inds must be a dictionary whose keys are "\
                "non-negative integers strictly  less than the "\
                "attribute n (the number  of vertices in the graph, "\
                f"which in this case has the value {self.n}).")
    return self._dijkstraFromSourcesPathfinderIndex(source_inds)

def dijkstraFromSourcesPathfinder(self,\
        sources: Dict[Hashable, Union[float, int]])\
        -> Dict[Hashable, Tuple[Union[int, float], Hashable]]:
    """
    Performs a Dijkstra search for a minimum cost path from the
    vertices given by sources and any vertex in the graph that
    is reachable from those vertices, giving the path cost and the
    vertex from which one such path came for each reachable vertex
    (allowing the user to identify a minimum weight path).
    The cost of a path is the sum of the weights of the (directed)
    edges that it traverses, plus the weight associated with the
    vertices at which the path begins (as specified by the value
    corresponding to these vertices in the dictionary sources).
    This may only be used if the graph contains no edges with
    negative weight (i.e. the attribute neg_weight_edge is False).
    If the graph contains at least one edge with negative weight
    (i.e. the attribute neg_weight_edge is True) and this method
    is called then an error will be raised.
    
    Args:
        Required positional:
        sources (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may start,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        
    Returns:
    Dictionary whose keys are all the labels of vertices in the graph
    that are reachable from the vertices given in sources and whose
    corresponding values are a 2-tuple whose index 0 contains a
    numeric value representing the minimum cost of any path from
    any vertex in sources to that vertex (including the start cost of
    the start vertex) and whose index 1 contains the label of a vertex
    that is the penultimate vertex on a such a path with that cost.
    """
    if self.neg_weight_edge:
        raise ValueError("The method "\
                f"{inspect.stack()[0][3]}() "\
                "may only be used when the graph contains no "\
                "negative weight edges.")
    for v in sources.keys():
        if not self.vertexInGraph(v):
            raise ValueError("For the method "\
                    f"{inspect.stack()[0][3]}(), input argument "\
                    "sources can only contain vertices of the graph.")
    
    source_inds = self._argVertices2IndexDict(sources)
    dist_prev_dict_idx = self._dijkstraFromSourcesPathfinderIndex(source_inds)
    return self._fromSourcesDistPathIndex2Vertex(\
            dist_prev_dict_idx)

def _shortestPathFasterAlgorithmDistancesIndex(self,\
        dists_dict: Dict[int, Union[int, float]],\
        in_qu: Optional[Set[int]]=None,\
        _default: Union[int, float]=float("inf"))\
        -> Dict[int, Tuple[Union[int, float]]]:
    """
    Giving in_qu as all indices signifies that there the source
    vertex is external to the graph, with a single 0 weight edge to
    each of the vertices in the graph and only returns the
    distances which are strictly less than zero. This is for use in
    Johnson's algorithm (see methods johnsonIndex() and johnson()
    below).
    
    Note that in_qu and dists_dict are typically modified by this
    method.
    
    Need to check
    """
    if in_qu is None: in_qu = set(dists_dict.keys())
    qu = deque(in_qu)
    adj_min_wt_func = self.getAdjMinimumWeightsIndex
    
    for _ in range(self.n):
        for _ in range(len(qu)):
            idx1 = qu.popleft()
            in_qu.remove(idx1)
            d1 = dists_dict.get(idx1, _default)
            for idx2, d2 in adj_min_wt_func(idx1).items():
                d3 = d1 + d2
                if d3 >= dists_dict.get(idx2, _default): continue
                dists_dict[idx2] = d3
                if idx2 in in_qu: continue
                qu.append(idx2)
                in_qu.add(idx2)
        if not qu: break
    else:
        self._neg_weight_cycle = True
        return {}
    self._neg_weight_cycle = False
    return dists_dict

def shortestPathFasterAlgorithmDistancesIndex(self,\
        source_inds: Set[int])\
        -> Dict[int, Tuple[Union[int, float]]]:
    """
    Shortest Path Faster Algorithm (SPFA) variant of Bellman-Ford
    Returns empty dict if negative weight cycle is detected
    
    Need to check
    """
    if source_inds and (min(source_inds.keys()) < 0 or\
            max(source_inds.keys()) >= self.n):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), input argument "\
                "source_inds must be a set of non-negative integers "\
                "strictly  less than the attribute n (the number "\
                "of vertices in the graph, which in this case "\
                f"has the value {self.n}).")
    if self.neg_weight_cycle: return {}
    return self._shortestPathFasterAlgorithmDistancesIndex(dict(source_inds))

def shortestPathFasterAlgorithmDistances(self,\
        sources: Dict[Hashable, Union[float, int]])\
        -> Dict[Hashable, Union[float, int]]:
    """
    Performs a SPFA (Shortest Path Faster Algorithm) search for
    a minimum cost path from the vertices given by sources and any
    vertex in the graph that is reachable from those vertices,
    giving the path cost only.
    The cost of a path is the sum of the weights of the (directed)
    edges that it traverses, plus the weight associated with the
    vertices at which the path begins (as specified by the value
    corresponding to these vertices in the dictionary sources).
    This may only be used if the graph contains no negative weight
    cycles (i.e. a cycle whose sum of weights is negative). If the
    graph contains any negative weight cycle and this method is called
    then an error will be raised.
    
    Args:
        Required positional:
        sources (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may start,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        
    Returns:
    Dictionary whose keys are all the labels of vertices in the graph
    that are reachable from the vertices given in sources and whose
    corresponding values are a numeric value representing the minimum
    cost of any path from any vertex in sources to that vertex (including
    the start cost of the start vertex).
    """
    for v in sources.keys():
        if not self.vertexInGraph(v):
            raise ValueError("For the method "\
                    f"{inspect.stack()[0][3]}(), input argument "\
                    "sources can only contain vertices of the graph.")
    if self.neg_weight_cycle: return {}
    source_inds = self._argVertices2IndexDict(sources)
    dist_dict_idx =\
            self._shortestPathFasterAlgorithmDistancesIndex(source_inds)
    return self._fromSourcesDistIndex2Vertex(dist_dict_idx)

def _shortestPathFasterAlgorithmPathfinderIndex(self,\
        dist_prev_dict_idx: Dict[int, Tuple[Union[int, float]]],\
        in_qu: Optional[Set[int]]=None,\
        _default: Union[int, float]=float("inf"))\
        -> Dict[int, Tuple[Union[int, float]]]:
    """
    Giving in_qu as all indices signifies that there the source
    vertex is external to the graph, with a single 0 weight edge to
    each of the vertices in the graph and only returns the
    distances which are strictly less than zero. This is for use in
    Johnson's algorithm (see methods johnsonIndex() and johnson()
    below).
    
    Note that dist_prev_dict_idx and in_qu are typically modified
    by this method.
    
    Need to check
    """
    if in_qu is None: in_qu = set(dist_prev_dict_idx.keys())
    qu = deque(in_qu)
    
    
    adj_min_wt_func = self.getAdjMinimumWeightsIndex
    
    for _ in range(self.n):
        for _ in range(len(qu)):
            idx1 = qu.popleft()
            in_qu.remove(idx1)
            d1 = dist_prev_dict_idx.get(idx1, (_default,))[0]
            for idx2, d2 in adj_min_wt_func(idx1).items():
                d3 = d1 + d2
                if d3 >= dist_prev_dict_idx.get(idx2,\
                        (_default,))[0]:
                    continue
                dist_prev_dict_idx[idx2] = (d3, idx1)
                if idx2 in in_qu: continue
                qu.append(idx2)
                in_qu.add(idx2)
        if not qu: break
    else:
        self._neg_weight_cycle = True
        return {}
    self._neg_weight_cycle = False
    return dist_prev_dict_idx

def shortestPathFasterAlgorithmPathfinderIndex(self,\
        source_inds: Dict[int, Union[int, float]])\
        -> Dict[int, Tuple[Union[int, float]]]:
    """
    Shortest Path Faster Algorithm (SPFA) variant of Bellman-Ford,
    giving the distances and the previous index for the shortest
    path to the current index
    Returns empty dict if negative weight cycle is detected
    
    Need to check
    """
    if source_inds and (min(source_inds.keys()) < 0 or\
            max(source_inds.keys()) >= self.n):
        raise ValueError("For the method "\
                f"{inspect.stack()[0][3]}(), input argument "\
                "source_inds must be a dictionary whose keys are "\
                "non-negative integers strictly  less than the "\
                "attribute n (the number  of vertices in the graph, "\
                f"which in this case has the value {self.n}).")
    # Checking whether any negative weight cycles have already
    # been detected (note that part of the getter neg_weight_cycle
    # negative weight 1-cycles are explicitly checked for if this
    # has not already been done)
    if self.neg_weight_cycle: return {}
    return self._shortestPathFasterAlgorithmPathfinderIndex(\
            {source_idx: (d, -1)\
            for source_idx, d in source_inds.items()})

def shortestPathFasterAlgorithmPathfinder(self,\
        sources: Dict[Hashable, Union[int, float]])\
        -> Dict[Hashable, Tuple[Union[int, float, Hashable]]]:
    """
    Performs a SPFA (Shortest Path Faster Algorithm) search for
    a minimum cost path from the vertices given by sources and any
    vertex in the graph that is reachable from those vertices,
    giving the path cost and the vertex from which one such path came
    for each reachable vertex (allowing the user to identify a
    minimum weight path).
    The cost of a path is the sum of the weights of the (directed)
    edges that it traverses, plus the weight associated with the
    vertices at which the path begins (as specified by the value
    corresponding to these vertices in the dictionary sources).
    This may only be used if the graph contains no negative weight
    cycles (i.e. a cycle whose sum of weights is negative). If the
    graph contains any negative weight cycle and this method is called
    then an error will be raised.
    
    Args:
        Required positional:
        sources (dict): Dictionary whose keys are the vertices of
                the graph at which the considered paths may start,
                whose corresponding values are numeric values
                (ints or floats) representing the weight that
                choice of start vertex contributes to the cost
                of paths starting from that vertex.
        
    Returns:
    Dictionary whose keys are all the labels of vertices in the graph
    that are reachable from the vertices given in sources and whose
    corresponding values are a 2-tuple whose index 0 contains a
    numeric value representing the minimum cost of any path from
    any vertex in sources to that vertex (including the start cost of
    the start vertex) and whose index 1 contains the label of a vertex
    that is the penultimate vertex on a such a path with that cost.
    """
    for v in sources.keys():
        if not self.vertexInGraph(v):
            raise ValueError("For the method "\
                    f"{inspect.stack()[0][3]}(), input argument "\
                    "sources can only contain vertices of the graph.")
    if self.neg_weight_cycle: return {}
    dist_prev_dict_idx =\
            self._shortestPathFasterAlgorithmPathfinderIndex(\
            {self.vertex2Index(v): (d, -1)\
            for v, d in sources.items()})
    return self._fromSourcesDistPathIndex2Vertex(\
            dist_prev_dict_idx)

def checkFromSourcesPathfinder(graph: LimitedGraphTemplate,\
        dist_prev_dict: Dict[Hashable,\
        Tuple[Union[int, float, Optional[Hashable]]]],\
        sources: Dict[Hashable, Union[int, float]],\
        eps: float=10 ** -5) -> bool:
    if graph.neg_weight_cycle:
        return not dist_prev_dict
    
    for v1, d in sources.items():
        if v1 not in dist_prev_dict.keys() or\
                dist_prev_dict[v1][0] > d + eps:
            print("hi1")
            return False
    
    for v1, (d1, v0) in dist_prev_dict.items():
        if not graph.vertexInGraph(v1):
            print("hi2")
            return False
        min_wts = graph.getAdjMinimumWeights(v1)
        for v2, w in min_wts.items():
            if v2 not in dist_prev_dict.keys() or\
                    dist_prev_dict[v2][0] > w + d1 + eps:
                print("hi3")
                return False
        if v0 is None:
            if v1 not in sources.keys() or d1 > sources[v1] + eps:
                print("hi4")
                return False
            continue
        if not graph.vertexInGraph(v0) or v0 == v1:
            print("hi5")
            return False
        min_wts0 = graph.getAdjMinimumWeights(v0)
        if v1 not in min_wts0.keys() or\
                v0 not in dist_prev_dict.keys() or\
                abs(dist_prev_dict[v0][0] + min_wts0[v1] - d1) > eps:
            print("hi6")
            return False
    return True

def checkFromSourcesDistances(graph: LimitedGraphTemplate,\
        dists_dict: Dict[int, Union[int, float]],\
        sources: Dict[Hashable, Union[int, float]],\
        eps: float=10 ** -5, check_pathfinder: bool=True) -> bool:
    # If check_pathfinder given as False, assumes that the pathfinder
    # method (in this case, dijkstraFromSourcesPathfinder() or
    # shortestPathFasterAlgorithmPathfinder()) is accurate
    pathfinder_method_nm = "shortestPathFasterAlgorithmPathfinder"\
            if graph.neg_weight_edge else\
            "dijkstraFromSourcesPathfinder"
    pathfinder = getattr(graph, pathfinder_method_nm)(sources)
    if check_pathfinder and not checkFromSourcesPathfinder(\
            graph, pathfinder, sources, eps=eps):
        raise ValueError("The pathfinder object calculated using "\
                f"the method {pathfinder_method_nm}() was itself "\
                "found to be incorrect, so check of the "
                "corresponding distance object cannot be performed")
    if pathfinder.keys() != dists_dict.keys():
        return False
    for v, d in dists_dict.items():
        if abs(pathfinder[v][0] - d) > eps:
            return False
    return True

## Algorithms finding shortest paths/distances between each ##
## pair of vertices in the graph                            ##

def _allPairsDistPathIndex2Vertex(self, dist_prev_dicts_ind:\
        List[Dict[int, Tuple[Union[int, float]]]])\
        -> Dict[Hashable,\
        Dict[Hashable, Tuple[Union[int, float, Optional[Hashable]]]]]:
    dist_prev_dicts = {}
    for idx1, dist_prev_dict_ind in enumerate(dist_prev_dicts_ind):
        v1 = self.index2Vertex(idx1)
        dist_prev_dicts[v1] = {self.index2Vertex(idx2):\
                (tup[0], None if tup[1] == -1 else\
                self.index2Vertex(tup[1]))\
                for idx2, tup in dist_prev_dict_ind.items()}
    return dist_prev_dicts
    """
    dist_prev_dicts = {}
    for vertex, dist_prev_dict_ind in\
            zip(self.vertexGenerator(), dist_prev_dicts_ind):
        if not dist_prev_dict_ind: continue
        dist_prev_dicts[vertex] = {self.index2Vertex(idx2):\
                (tup[0], self.index2Vertex(tup[1]))\
                for idx2, tup in dist_prev_dict_ind.items()}
    return dist_prev_dicts
    """

def floydWarshallDistancesIndex(self)\
        -> List[Dict[int, Union[int, float]]]:
    # Need to check
    # For weighted directed graph
    # Returns empty list if negative weight cycle detected
    
    # Checking whether any negative weight cycles have already
    # been detected (note that part of the getter neg_weight_cycle
    # negative weight 1-cycles are explicitly checked for if this
    # has not already been done)
    if self.neg_weight_cycle: return []
    
    dists = [self.getAdjMinimumWeightsIndex(idx)\
            for idx in range(self.n)]
    #if not hasattr(self, "_neg_weight_self_edge"):
    #    for idx in range(self.n):
    #        if dists[idx][idx] < 0:
    #            self._neg_weight_self_edge = True
    #            return []
    #        dists[idx][idx] = 0
    #    self._neg_weight_self_edge = False
    #else:
    #    for idx in range(self.n):
    #        dists[idx][idx] = 0
    
    for idx in range(self.n):
        for idx1 in range(self.n):
            if idx not in dists[idx1].keys(): continue
            d1 = dists[idx1][idx]
            for idx2 in dists[idx].keys():
                d = d1 + dists[idx][idx2]
                if d >= dists[idx1].get(idx2, float("inf")):
                    continue
                # Checking for negative weight cycle
                if idx1 == idx2 and d < 0:
                    self._neg_weight_cycle = True
                    return {}
                dists[idx1][idx2] = d
    for idx in range(self.n): dists[idx][idx] = 0
    self._neg_weight_cycle = False
    return dists

def floydWarshallDistances(self)\
        -> Dict[Hashable, Dict[Hashable, Union[int, float]]]:
    dists_ind = self.floydWarshallDistancesIndex()
    res = {}
    for vertex, dist_dict in zip(self.vertexGenerator(),\
            dists_ind):
        if not dist_dict: continue
        res[vertex] = {self.index2Vertex(idx2): d\
                for idx2, d in dist_dict.items()}
    return res

def floydWarshallPathfinderIndex(self)\
        -> List[Dict[int, Tuple[Union[int, float]]]]:
    # Need to check
    # For weighted directed graph
    # Returns empty list if negative weight cycle detected
    
    # Checking whether any negative weight cycles have already
    # been detected (note that part of the getter neg_weight_cycle
    # negative weight 1-cycles are explicitly checked for if this
    # has not already been done)
    if self.neg_weight_cycle: return []
    
    dist_prev_dicts = [{idx2: (d, idx1) for idx2, d in\
            self.getAdjMinimumWeightsIndex(idx1).items()}\
            for idx1 in range(self.n)]
    #print(dist_prev_dicts)
    for idx in range(self.n):
        for idx1 in range(self.n):
            if idx not in dist_prev_dicts[idx1].keys(): continue
            d1 = dist_prev_dicts[idx1][idx][0]
            for idx2 in dist_prev_dicts[idx].keys():
                d = d1 + dist_prev_dicts[idx][idx2][0]
                if d >= dist_prev_dicts[idx1].get(idx2,\
                        (float("inf"),))[0]:
                    continue
                # Checking for negative weight cycle
                if idx1 == idx2 and d < 0:
                    self._neg_weight_cycle = True
                    return {}
                dist_prev_dicts[idx1][idx2] =\
                        (d, dist_prev_dicts[idx][idx2][1])
        #print(idx)
        #print(dist_prev_dicts)
    for idx in range(self.n): dist_prev_dicts[idx][idx] = (0, -1)
    self._neg_weight_cycle = False
    return dist_prev_dicts

def floydWarshallPathfinder(self) ->  Dict[Hashable, Dict[Hashable,\
        Tuple[Union[int, float, Optional[Hashable]]]]]:
    dist_prev_dicts_ind = self.floydWarshallPathfinderIndex()
    return self._allPairsDistPathIndex2Vertex(dist_prev_dicts_ind)

def johnsonIndex(self)\
        -> List[Dict[int, Tuple[Union[int, float]]]]:
    
    # Checking whether any negative weight cycles have already
    # been detected (note that part of the getter neg_weight_cycle
    # negative weight 1-cycles are explicitly checked for if this
    # has not already been done)
    if self.neg_weight_cycle: return []
    elif not self.neg_weight_edge:
        return [self.dijkstraFromSourcesPathfinderIndex({idx: 0})\
                for idx in range(self.n)]
    h_values = self._shortestPathFasterAlgorithmDistancesIndex(\
            {}, in_qu=set(range(self.n)), _default=0)
    if self.neg_weight_cycle: return []
    #print(h_values)
    weight_mod_func = lambda idx1, idx2, wt:\
            wt + h_values.get(idx1, 0) - h_values.get(idx2, 0)
    res = []
    for idx1 in range(self.n):
        res.append({idx2: (tup[0] + h_values.get(idx2, 0) -\
                h_values.get(idx1, 0), tup[1])\
                for idx2, tup in\
                self._dijkstraFromSourcesPathfinderIndex({idx1: 0},\
                weight_mod_func).items()})
    #print(res)
    return res

def johnson(self) ->  Dict[Hashable, Dict[Hashable,\
        Tuple[Union[int, float, Optional[Hashable]]]]]:
    dist_prev_dicts_ind = self.johnsonIndex()
    return self._allPairsDistPathIndex2Vertex(dist_prev_dicts_ind)

def checkAllPairsPathfinder(graph: LimitedGraphTemplate,\
        dist_prev_dicts: Dict[Hashable, Dict[Hashable,\
        Tuple[Union[int, float, Optional[Hashable]]]]],\
        eps: float=10 ** -5) -> bool:
    if graph.neg_weight_cycle:
        return not dist_prev_dicts
    if len(dist_prev_dicts) != graph.n:
        print("hi1")
        return False
    for v in graph.vertexGenerator():
        if v not in dist_prev_dicts.keys():
            print("hi2")
            print(graph.grid.arr_flat)
            print(v)
            print(dist_prev_dicts.keys())
            return False
    
    for v1, v2_dict in dist_prev_dicts.items():
        if v1 not in v2_dict.keys() or v2_dict[v1][1] is not None or\
                abs(v2_dict[v1][0]) > eps:
            print("hi3")
            return False
        min_wts = graph.getAdjMinimumWeights(v1)
        
        for v3, w in min_wts.items():
            if v3 not in v2_dict.keys():
                print("hi4")
                return False
            for v2, (d, _) in dist_prev_dicts[v3].items():
                if v2 not in v2_dict.keys() or\
                        v2_dict[v2][0] > w + d + eps:
                    print("hi5")
                    return False
        for v2, (d, v3) in v2_dict.items():
            if not graph.vertexInGraph(v2):
                print("hi6")
                return False
            if v3 is None:
                if v1 != v2 or abs(d) > eps:
                    print("hi7")
                    return False
                continue
            elif not graph.vertexInGraph(v3) or v3 == v2:
                print("hi8")
                return False
            min_wts3 = graph.getAdjMinimumWeights(v3)
            if v2 not in min_wts3.keys() or\
                    v3 not in v2_dict.keys() or\
                    abs(v2_dict[v3][0] + min_wts3[v2] - d) > eps:
                print("hi9")
                return False
    return True

def checkAllPairsDistances(graph: LimitedGraphTemplate,\
        dist_prev_dicts: Dict[Hashable, Dict[Hashable,\
        Tuple[Union[int, float]]]],\
        eps: float=10 ** -5, check_pathfinder: bool=True) -> bool:
    # If check_pathfinder given as False, assumes that the pathfinder
    # method (in this case, johnson()) is accurate
    pathfinder = graph.johnson()
    if check_pathfinder and\
            not checkAllPairsPathfinder(graph, pathfinder, eps=eps):
        raise ValueError("The pathfinder object calculated using "\
                "the method johnson() was itself found to be "\
                "incorrect, so check of the corresponding distance "\
                "object cannot be performed")
    if pathfinder.keys() != dist_prev_dicts.keys():
        return False
    for v1, v2_dict1 in dist_prev_dicts.items():
        v2_dict2 = pathfinder[v1]
        if v2_dict1.keys() != v2_dict2.keys():
            return False
        for v2, d in v2_dict1.items():
            if abs(v2_dict2[v2][0] - d) > eps:
                return False
    return True
