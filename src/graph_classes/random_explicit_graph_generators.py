#!/usr/bin/env python
from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Generator,
    Callable,
)

import itertools
import math
import random

from sortedcontainers import SortedList


from graph_classes.utils import FenwickTree

from graph_classes.explicit_graph_types import (
    ExplicitGraphTemplate,
    ExplicitWeightedGraphTemplate,
    ExplicitUnweightedGraphTemplate,
    ExplicitWeightedDirectedGraph,
    ExplicitWeightedUndirectedGraph,
    ExplicitUnweightedDirectedGraph,
    ExplicitUnweightedUndirectedGraph,
)

def isqrt(num: int) -> int:
    res = num
    res2 = (num + 1) >> 1
    while res2 < res:
        res = res2
        res2 = (res + (num // res)) >> 1
    return res


def undirectedEdgeRepeatsGenerator(n_vertices: int, n_edges: int,\
        allow_self_edges: bool) -> Generator[Tuple[int], None, None]:
    #print("hello")
    n2 = n_vertices - (not allow_self_edges)
    n_opts = (n2 * (n2 + 1)) >> 1
    # Review
    for _ in range(n_edges):
        num = random.randrange(0, n_opts)
        
        idx1_rev = (isqrt(8 * num + 1) - 1) >> 1
        idx2_rev = num - ((idx1_rev * (idx1_rev + 1)) >> 1)
        #print(num, idx1_rev, idx2_rev)
        yield (n2 - 1 - idx1_rev, n_vertices - 1 - idx2_rev)
        #idx1 = random.randrange(0, n2)
        #idx2 = random.randrange(idx1 + (not allow_self_edges),\
        #        n_vertices)
        #yield (idx1, idx2)
    return

def directedEdgeRepeatsGenerator(n_vertices: int, n_edges: int,\
        allow_self_edges: bool) -> Generator[Tuple[int], None, None]:
    n2 = n_vertices - (not allow_self_edges)
    for _ in range(n_edges):
        idx1 = random.randrange(0, n_vertices)
        idx2 = random.randrange(0, n2)
        idx2 += (not allow_self_edges and idx2 >= idx1)
        yield (idx1, idx2)
    return

def findIndices(num: int, idx1_fenwick: FenwickTree,\
        starts: List[int], excl_lsts: List[SortedList]) -> Tuple[int]:
    #print(num)
    #print(excl_lsts)
    lft, rgt = 0, len(starts) - 1
    while lft < rgt:
        mid = lft + ((rgt - lft) >> 1)
        le_cnt = idx1_fenwick.query(mid)
        if le_cnt < num: lft = mid + 1
        else: rgt = mid
    idx1 = lft
    num -= idx1_fenwick.query(idx1 - 1)
    #print(idx1, num)
    idx2 = findNthNonExcluded(num, starts[idx1], excl_lsts[idx1])
    return (idx1, idx2)

def findNthNonExcluded(n: int, start: int, excl_lst: SortedList):
    lft, rgt = start + n - 1, start + len(excl_lst) + n - 1
    while lft < rgt:
        mid = lft + ((rgt - lft) >> 1)
        non_excl_le_cnt = mid - start - excl_lst.bisect_right(mid) + 1
        #print(mid, non_excl_le_cnt)
        if non_excl_le_cnt < n:
            lft = mid + 1
        elif non_excl_le_cnt > n:
            rgt = mid - 1
        else: rgt = mid
    return lft

def undirectedEdgeNoRepeatsGenerator(n_vertices: int, n_edges: int,\
        allow_self_edges: bool) -> Generator[Tuple[int], None, None]:
    n2 = n_vertices - (not allow_self_edges)
    idx1_fenwick = FenwickTree(n2, ((lambda x, y: x + y), 0))
    starts = [idx1 + (not allow_self_edges) for idx1 in range(n2)]
    excl_lsts = [SortedList() for _ in range(n2)]
    n_opts = (n2 * (n2 + 1)) >> 1
    #print(f"n_opts = {n_opts}")
    for idx1 in range(n2):
        idx1_fenwick.update(idx1, n2 - idx1)
    for _ in range(n_edges):
        num = random.randrange(1, n_opts + 1)
        idx1, idx2 = findIndices(num, idx1_fenwick,\
                starts, excl_lsts)
        excl_lsts[idx1].add(idx2)
        n_opts -= 1
        idx1_fenwick.update(idx1, -1)
        yield (idx1, idx2)
    return
    """
    rand_num = 
    idx1 = random.choice(list(seen.keys()))
    print(f"idx1 = {idx1}")
    idx2_prov = random.randrange(0, n - idx1 - len(seen[idx1]))
    print(f"idx2_prov = {idx2_prov}")
    idx2 = findNthNonExcluded(idx2_prov + 1, idx1, seen[idx1])
    if len(seen[idx1]) + 1 == n - idx1:
        print("hi")
        seen.pop(idx1)
    else: seen[idx1].add(idx2)
    return (idx1, idx2)
    """

def directedEdgeNoRepeatsGenerator(n_vertices: int, n_edges: int,\
        allow_self_edges: bool) -> Generator[Tuple[int], None, None]:
    n2 = n_vertices
    idx1_fenwick = FenwickTree(n2, ((lambda x, y: x + y), 0))
    starts = [0 for idx1 in range(n2)]
    excl_lsts = [SortedList() for _ in range(n2)] if allow_self_edges\
            else [SortedList([idx1]) for idx1 in range(n2)]
    n3 = n2 - (not allow_self_edges)
    n_opts = n3 * n2
    #print(f"n_opts = {n_opts}")
    for idx1 in range(n2):
        idx1_fenwick.update(idx1, n3)
    for _ in range(n_edges):
        num = random.randrange(1, n_opts + 1)
        idx1, idx2 = findIndices(num, idx1_fenwick,\
                starts, excl_lsts)
        excl_lsts[idx1].add(idx2)
        n_opts -= 1
        idx1_fenwick.update(idx1, -1)
        yield (idx1, idx2)
    return
    """
    print(seen)
    idx1 = random.choice(list(seen.keys()))
    idx2_prov = random.randrange(0, n - len(seen[idx1]))
    print(f"idx1 = {idx1}, idx2_prov = {idx2_prov}")
    idx2 = findNthNonExcluded(idx2_prov + 1, 0, seen[idx1])
    print(f"idx2 = {idx2}")
    if len(seen[idx1]) == n:
        seen.pop(idx1)
    else: seen[idx1].add(idx2)
    return (idx1, idx2)
    """

def checkFrequencies(n_samples: int, n_vertices: int,\
        n_edges: int, directed: bool, allow_self_edges: bool,\
        allow_rpt_edges: bool) -> Dict[int, int]:
    
    if directed:
        edge_func = directedEdgeRepeatsGenerator if allow_rpt_edges\
                else directedEdgeNoRepeatsGenerator
    else:
        edge_func = undirectedEdgeRepeatsGenerator if allow_rpt_edges\
                else undirectedEdgeNoRepeatsGenerator
    
    f_dict = {}
    if directed:
        for idx1 in range(n_vertices):
            for idx2 in range(n_vertices):
                f_dict[(idx1, idx2)] = 0
        if not allow_self_edges:
            for idx in range(n_vertices):
                f_dict.pop((idx, idx))
    else:
        for idx1 in range(n_vertices):
            for idx2 in range(idx1 + (not allow_self_edges),\
                    n_vertices):
                f_dict[(idx1, idx2)] = 0
    #print(f_dict)
    for _ in range(n_samples):
        for e in edge_func(n_vertices, n_edges, allow_self_edges):
            f_dict[e] += 1
    return f_dict
        


def randomWeightedEdges(n_vertices: int, n_edges: int, directed: bool,\
        wt_mn: Union[int, float], wt_mx: Union[int, float],\
        wt_typ: type, allow_self_edges: bool=True,\
        allow_rpt_edges: bool=True) -> List[List[Union[int, float]]]:
    
    if wt_typ == int:
        if not isinstance(wt_mn, int):
            wt_mn = math.ceil(wt_mn)
        if not isinstance(wt_mx, int):
            wt_mx = math.floor(wt_mx)
        wt_mx += 1
        wt_func = lambda: random.randrange(wt_mn, wt_mx)
    else:
        wt_func = lambda: random.uniform(wt_mn, wt_mx)
    
    if directed:
        edge_func = directedEdgeRepeatsGenerator if allow_rpt_edges\
                else directedEdgeNoRepeatsGenerator
    else:
        edge_func = undirectedEdgeRepeatsGenerator if allow_rpt_edges\
                else undirectedEdgeNoRepeatsGenerator
    
    return [(idx1, idx2, wt_func()) for idx1, idx2 in\
            edge_func(n_vertices, n_edges, allow_self_edges)]
    """
    res = []
    seen = {idx1: SortedList([idx1])\
            for idx1 in range(n_vertices - 1)} if not allow_self_edges\
            else {idx1: SortedList() for idx1 in range(n_vertices)}
    for _ in range(n_edges):
        
        idx1, idx2 = edge_func(seen, n_vertices, allow_self_edges)
        res.append([idx1, idx2, wt_func()])
    return res
    """

def randomUnweightedEdges(n_vertices: int, n_edges: int, directed: bool,\
        allow_self_edges: bool=True,\
        allow_rpt_edges: bool=True) -> List[List[Union[int, float]]]:
    
    if directed:
        edge_func = directedEdgeRepeatsGenerator if allow_rpt_edges\
                else directedEdgeNoRepeatsGenerator
    else:
        edge_func = undirectedEdgeRepeatsGenerator if allow_rpt_edges\
                else undirectedEdgeNoRepeatsGenerator
    
    return [(idx1, idx2) for idx1, idx2 in\
            edge_func(n_vertices, n_edges, allow_self_edges)]


def _randomExplicitGraphGenerator(n_vertices_rng: Tuple[int],\
        n_edges_rng_func: Callable[[int], Tuple[int]],\
        directed: bool, wt_props: tuple,\
        count: Optional[int], allow_self_edges: bool,\
        allow_rpt_edges: bool, randomise_indices: bool=False)\
        -> Generator["ExplicitGraphTemplate", None, None]:
    
    iter_obj = itertools.count(0) if count is None else range(count)
    mx_edge_func = (lambda n, n2: n * n2) if directed else\
            (lambda n, n2: (n2 * (n2 + 1)) >> 1)
    
    if wt_props:
        rand_edges_func = randomWeightedEdges
        graph_cls = ExplicitWeightedDirectedGraph if directed else\
                ExplicitWeightedUndirectedGraph
    else:
        rand_edges_func = randomUnweightedEdges
        graph_cls = ExplicitUnweightedDirectedGraph if directed else\
                ExplicitUnweightedUndirectedGraph
        
    for _ in iter_obj:
        n_vertices = random.randrange(n_vertices_rng[0], n_vertices_rng[1] + 1)
        n_edges_mn, n_edges_mx = n_edges_rng_func(n_vertices)
        if not allow_rpt_edges:
            n2 = n_vertices - (not allow_self_edges)
            n_edges_mx = min(n_edges_mx, mx_edge_func(n_vertices, n2))
            n_edges_mn = min(n_edges_mn, n_edges_mx)
        #print(f"n_vertices = {n_vertices}, "
        #        f"n_edges_mn = {n_edges_mn}, n_edges_mx = {n_edges_mx}")
        n_edges = random.randrange(n_edges_mn, n_edges_mx + 1)
        edges = rand_edges_func(n_vertices, n_edges, True,\
                *wt_props, allow_self_edges=allow_self_edges,\
                allow_rpt_edges=allow_rpt_edges)
        #print(f"n_edges = {len(edges)}")
        vertices = range(n_vertices)
        if randomise_indices:
            vertices = list(vertices)
            random.shuffle(vertices)
        yield graph_cls(vertices, edges)
    return

def randomExplicitWeightedDirectedGraphGenerator(\
        n_vertices_rng: Tuple[int],\
        n_edges_rng_func: Callable[[int], Tuple[int]],\
        wt_mn: Union[int, float],\
        wt_mx: Union[int, float],\
        wt_typ: type=int, count: Optional[int]=None,\
        allow_self_edges: bool=True,\
        allow_rpt_edges: bool=True,\
        randomise_indices: bool=True)\
        -> Generator["ExplicitWeightedDirectedGraph", None, None]:
    wt_props = (wt_mn, wt_mx, wt_typ)
    yield from _randomExplicitGraphGenerator(n_vertices_rng,\
            n_edges_rng_func, True, wt_props, count, allow_self_edges,\
            allow_rpt_edges, randomise_indices=randomise_indices)
    return

def randomExplicitWeightedUndirectedGraphGenerator(\
        n_vertices_rng: Tuple[int],\
        n_edges_rng_func: Callable[[int], Tuple[int]],\
        wt_mn: Union[int, float],\
        wt_mx: Union[int, float],\
        wt_typ: type=int, count: Optional[int]=None,\
        allow_self_edges: bool=True,\
        allow_rpt_edges: bool=True,\
        randomise_indices: bool=False)\
        -> Generator["ExplicitWeightedDirectedGraph", None, None]:
    wt_props = (wt_mn, wt_mx, wt_typ)
    yield from _randomExplicitGraphGenerator(n_vertices_rng,\
            n_edges_rng_func, False, wt_props, count,\
            allow_self_edges, allow_rpt_edges,\
            randomise_indices=randomise_indices)
    return

def randomExplicitUnweightedDirectedGraphGenerator(\
        n_vertices_rng: Tuple[int],\
        n_edges_rng_func: Callable[[int], Tuple[int]],\
        count: Optional[int]=None,\
        allow_self_edges: bool=True,\
        allow_rpt_edges: bool=True,\
        randomise_indices: bool=False)\
        -> Generator["ExplicitWeightedDirectedGraph", None, None]:
    yield from _randomExplicitGraphGenerator(n_vertices_rng,\
            n_edges_rng_func, True, (), count, allow_self_edges,\
            allow_rpt_edges, randomise_indices=randomise_indices)
    return

def randomExplicitUnweightedUndirectedGraphGenerator(\
        n_vertices_rng: Tuple[int],\
        n_edges_rng_func: Callable[[int], Tuple[int]],\
        count: Optional[int]=None,\
        allow_self_edges: bool=True,\
        allow_rpt_edges: bool=True,\
        randomise_indices: bool=False)\
        -> Generator["ExplicitWeightedDirectedGraph", None, None]:
    yield from _randomExplicitGraphGenerator(n_vertices_rng,\
            n_edges_rng_func, False, (), count, allow_self_edges,\
            allow_rpt_edges, randomise_indices=randomise_indices)
    return

def testRandomExplicitWeightedGraphsGenerator(n_graph_per_opt: int, directed: bool,\
        n_vertices_rng: Tuple[int], n_edges_rng_func: Callable[[int], int],\
        wt_rng: Tuple[Union[int, float]], wt_cls_opts: Tuple[type]=(int, float),\
        allow_self_edges_opts: Tuple[bool]=(True, False),\
        allow_rpt_edges_opts: Tuple[bool]=(True, False),\
        randomise_indices: bool=False)\
        -> Generator["ExplicitWeightedGraphTemplate", None, None]:
    graph_generator_cls = randomExplicitWeightedDirectedGraphGenerator\
            if directed else\
            randomExplicitWeightedUndirectedGraphGenerator
    
    for allow_self_edges in allow_self_edges_opts:
        for allow_rpt_edges in allow_rpt_edges_opts:
            for wt_cls in wt_cls_opts:
                for graph in graph_generator_cls(n_vertices_rng,\
                        n_edges_rng_func, *wt_rng, wt_cls,\
                        count=n_graph_per_opt,\
                        allow_self_edges=allow_self_edges,\
                        allow_rpt_edges=allow_rpt_edges,\
                        randomise_indices=randomise_indices):
                    yield graph

def testRandomExplicitUnweightedGraphsGenerator(n_graph_per_opt: int, directed: bool,\
        n_vertices_rng: Tuple[int], n_edges_rng_func: Callable[[int], int],\
        allow_self_edges_opts: Tuple[bool]=(True, False),\
        allow_rpt_edges_opts: Tuple[bool]=(True, False),\
        randomise_indices: bool=False)\
        -> Generator["ExplicitUnweightedGraphTemplate", None, None]:
    graph_generator_cls = randomExplicitUnweightedDirectedGraphGenerator\
            if directed else\
            randomExplicitUnweightedUndirectedGraphGenerator
    
    for allow_self_edges in allow_self_edges_opts:
        for allow_rpt_edges in allow_rpt_edges_opts:
            for graph in graph_generator_cls(n_vertices_rng,\
                    n_edges_rng_func, count=n_graph_per_opt,\
                    allow_self_edges=allow_self_edges,\
                    allow_rpt_edges=allow_rpt_edges,\
                    randomise_indices=randomise_indices):
                yield graph
