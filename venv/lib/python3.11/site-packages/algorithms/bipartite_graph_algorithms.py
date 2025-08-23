#!/usr/bin/env python

from typing import (
    Generator,
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
)

def hungarianAdjMatrix(
    adj_matrix: List[List[int]],
    eps: float=10 ** -5,
) -> Tuple[Union[int, Dict[int, int]]]:
    """
    
    """
    
    shape = (len(adj_matrix), len(adj_matrix[0]))
    n = max(shape)
    adj_matrix2 = [list(x) for x in adj_matrix]
    if shape[1] > shape[0]:
        adj_matrix2.extend([[0] * shape[1] for _ in range(shape[1] - shape[0])])
    elif shape[0] > shape[1]:
        for row in adj_matrix2:
            row.extend([0] * (shape[0] - shape[1]))

    for i1 in range(shape[0]):
        mn = min(adj_matrix2[i1])
        if not mn: continue
        for i2 in range(n):
            adj_matrix2[i1][i2] -= mn
    for i2 in range(n):
        mn = min(row[i2] for row in adj_matrix2)
        if not mn: continue
        for i1 in range(n):
            adj_matrix2[i1][i2] -= mn
    
    def coverRowsAndColumns(
        adj_matrix: List[List[int]],
    ) -> Tuple[Union[Set[int], Dict[int, int]]]:
        #print(adj_matrix)
        uncovered_cols = set(range(n))
        uncovered_rows = set(range(n))
        starred_rowcol_dict = {}
        starred_colrow_dict = {}
        for i1, row in enumerate(adj_matrix):
            for i2 in uncovered_cols:
                if row[i2] <= eps: break
            else: continue
            uncovered_cols.remove(i2)
            starred_rowcol_dict[i1] = i2
            starred_colrow_dict[i2] = i1
        primed_dict = {}
        while True:
            for i1 in uncovered_rows:
                for i2 in uncovered_cols:
                    if adj_matrix[i1][i2] <= eps: break
                else: continue
                break
            else: break
            if i1 in starred_rowcol_dict.keys():
                primed_dict[i1] = i2
                uncovered_rows.remove(i1)
                uncovered_cols.add(starred_rowcol_dict[i1])
                continue
            starred_rowcol_dict[i1] = i2
            while i2 in starred_colrow_dict.keys():
                i1_2 = starred_colrow_dict[i2]
                starred_colrow_dict[i2] = i1
                i1 = i1_2
                i2_2 = primed_dict[i1]
                primed_dict[i1] = i2
                i2 = i2_2
                starred_rowcol_dict[i1] = i2
            starred_colrow_dict[i2] = i1
            uncovered_rows = set(range(n))
            uncovered_cols = set(range(n)).difference(starred_colrow_dict.keys())
            primed_dict = {}
        return (uncovered_rows, uncovered_cols,\
                starred_rowcol_dict)
    
    while True:
        uncovered_rows, uncovered_cols, starred_rowcol_dict =\
                coverRowsAndColumns(adj_matrix2)
        if len(starred_rowcol_dict) == n:
            break
        mn = float("inf")
        for i1 in uncovered_rows:
            for i2 in uncovered_cols:
                mn = min(mn, adj_matrix2[i1][i2])
        for i1 in uncovered_rows:
            for i2 in uncovered_cols:
                adj_matrix2[i1][i2] -= mn
        covered_rows = set(range(n)).difference(uncovered_rows)
        covered_cols = set(range(n)).difference(uncovered_cols)
        for i1 in covered_rows:
            for i2 in covered_cols:
                adj_matrix2[i1][i2] += mn
    tot = sum(adj_matrix[i1][i2] for i1, i2 in\
            starred_rowcol_dict.items() if i1 < shape[0] and i2 < shape[1])
    return (tot, starred_rowcol_dict)

def binMatrix2UnweightedBipartiteAdj(
    bin_adj_matrix: List[List[int]],
) -> Tuple[Union[List[Set[int]], int]]:
    # Kuhn's for unweighted bipartite graphs when graph
    # expressed as a binary adjacency matrix
    n1, n2 = len(bin_adj_matrix), len(bin_adj_matrix[0])
    adj = [set() for _ in range(n1 + n2)]
    for i1, row in enumerate(bin_adj_matrix):
        for i2, v in enumerate(row, start=n1):
            if not v: continue
            adj[i1].add(i2)
            adj[i2].add(i1)
    return (adj, n1)

def kuhnAdj(
    adj: List[Set[int]],
    n1: int,
) -> Tuple[Union[List[int], int]]:
    # Kuhn's algorithm for unweighted bipartite graphs with graph
    # expressed as adjacency list.
    
    # Look into Hopcroft-Karp-Karzanov algorithm
    n = len(adj)
    n2 = n - n1
    matches = [-1] * n
    def dfs(idx: int) -> bool:
        for idx2 in adj[idx].difference(seen):
            seen.add(idx2)
            if matches[idx2] == -1 or dfs(matches[idx2]):
                matches[idx2] = idx
                matches[idx] = idx2
                return True
        return False
    
    out_rng = (0, n1) if n1 <= n2 else (n1, n)
    res = 0
    for idx in range(*out_rng):
        seen = set()
        res += (matches[idx] == -1 and dfs(idx))
    return res, matches

def kuhnBinMatrix(
    bin_adj_matrix: List[List[int]],
) -> Tuple[Union[int, List[int]]]:
    # Kuhn's algorithm for unweighted bipartite graphs when graph
    # expressed as a binary adjacency matrix
    return kuhnAdj(*binMatrix2UnweightedBipartiteAdj(bin_adj_matrix))

def hopcroftKarpAdj(
    adj: List[Set[int]],
    n1: int,
) -> int:
    # Hopcroft-Karp-Karzanov algorithm for unweighted bipartite graphs
    # with graph expressed as adjacency list.
    n = len(adj)
    n2 = n - n1
    matches = [-1] * n
    unmatched1 = set(range(n1))
    unmatched2 = set(range(n1, n))
    
    def createAugmentGraph(
        unmatched1: Set[int],
        unmatched2: Set[int],
    ) -> Tuple[Union[Dict[int, Set[int]], Set[int]]]:
        in_edges = {}
        out_edges = {}
        aug_found = False
        curr = set()
        ends = set()
        #print(unmatched1, unmatched2)
        for idx1 in unmatched1:
            out_edges[idx1] = set()
            for idx2 in adj[idx1]:
                if idx2 in unmatched2:
                    ends.add(idx2)
                else: curr.add(idx2)
                out_edges[idx1].add(idx2)
                in_edges.setdefault(idx2, set())
                in_edges[idx2].add(idx1)
        #print(ends)
        while not ends and curr:
            #print(curr)
            #prev = curr
            #curr = set()
            #for idx2 in prev:
            #    idx1 = matches[idx2]
            #    out_edges[idx2] = {idx1}
            #    in_edges[idx1] = {idx2}
            #    curr.add(idx1)
            if not curr: break
            prev = curr
            curr = set()
            for idx2_0 in prev:
                idx1 = matches[idx2_0]
                out_edges[idx2_0] = set()
                for idx2 in adj[idx1]:
                    if idx2 in in_edges.keys():
                        if idx2 not in curr:
                            continue
                    else:
                        in_edges[idx2] = set()
                        if idx2 in unmatched2:
                            ends.add(idx2)
                        else: curr.add(idx2)
                    out_edges[idx2_0].add(idx2)
                    in_edges[idx2].add(idx2_0)
    
        if not ends: return ({}, {}, set())
        return (out_edges, in_edges, ends)
    
    def pruneAugmentGraph(
        idx: int,
        out_edges: Dict[int, Set[int]],
        in_edges: Dict[int, Set[int]],
    ) -> None:
        #print(idx, out_edges, in_edges)
        def recur(idx: int) -> None:
            if idx in in_edges.keys():
                for idx_prev in in_edges.pop(idx):
                    out_edges[idx_prev].remove(idx)
            if idx in out_edges.keys():
                for idx_nxt in out_edges.pop(idx):
                    in_edges[idx_nxt].remove(idx)
                    if not in_edges[idx_nxt]:
                        recur(idx_nxt)
            return
        return recur(idx)
    
    def augmentMatches(
        ends: Set[int],
        out_edges: Dict[int, Set[int]],
        in_edges: Dict[int, Set[int]],
    ) -> None:
        
        def recur(idx: int, idx0: Optional[int]) -> None:
            if idx in unmatched1:
                matches[idx0] = idx
                matches[idx] = idx0
                pruneAugmentGraph(idx, out_edges, in_edges)
                unmatched1.remove(idx)
                return
            elif idx0 is not None:
                idx1 = matches[idx]
                matches[idx1] = idx0
                matches[idx0] = idx1
            #print(idx, in_edges)
            idx_nxt = next(iter(in_edges[idx]))
            pruneAugmentGraph(idx, out_edges, in_edges)
            recur(idx_nxt, idx)
            return
        
        for idx2 in ends:
            #print(idx2, ends, in_edges, unmatched2)
            if idx2 not in in_edges.keys(): continue
            unmatched2.remove(idx2)
            recur(idx2, None)
        return
    
    while unmatched1 and unmatched2:
        out_edges, in_edges, ends =\
                createAugmentGraph(unmatched1, unmatched2)
        #print(out_edges, in_edges, ends)
        if not ends: break
        augmentMatches(ends, out_edges, in_edges)
        #print(matches)
    return (matches, n1 - len(unmatched1))

def hopcroftKarpBinMatrix(
    bin_adj_matrix: List[List[int]]
) -> Tuple[Union[int, List[int]]]:
    # Hopcroft-Karp-Karzanov algorithm for unweighted bipartite graphs
    # when graph expressed as a binary adjacency matrix
    return hopcroftKarpAdj(\
            *binMatrix2UnweightedBipartiteAdj(bin_adj_matrix))

def fordFulkerson(
    graph: List[Dict[int, Union[int, float]]],
    start: int,
    end: int,
) -> Union[int, float]:
    eps = 10 ** -5

    graph = [dict(x) for x in graph] # deep copy of graph

    def dfs() -> Tuple[Union[int, float, List[int]]]:
        prev = {start: None}
        stk = [(start, float("inf"))]
        while stk:
            idx, flow = stk.pop()
            for idx2, mx_flow in graph[idx].items():
                if idx2 in prev.keys(): continue
                flow2 = min(flow, mx_flow)
                prev[idx2] = idx
                if idx2 == end: break
                stk.append((idx2, flow2))
            else: continue
            break
        else: return ()
        #print(prev)
        path = []
        while idx2 is not None:
            path.append(idx2)
            idx2 = prev[idx2]
        return (flow2, path[::-1])
    
    res = 0
    while True:
        pair = dfs()
        if not pair: break
        flow, path = pair
        res += flow
        #print(path)
        for i in range(len(path) - 1):
            idx1, idx2 = path[i], path[i + 1]
            #print(graph)
            #print(idx1, idx2)
            graph[idx1][idx2] -= flow
            if graph[idx1][idx2] < eps:
                graph[idx1].pop(idx2)
            graph[idx2][idx1] = graph[idx2].get(idx1, 0) + 1
    return res

def maximumInvitations(self, grid: List[List[int]]) -> int:
    """
    
    
    Solution to Leetcode #1820 (Premium)
    
    Original problem description (essentially the assignment problem)
    
    There are m boys and n girls in a class attending an upcoming
    party.

    You are given an m x n integer matrix grid, where grid[i][j] equals
    0 or 1. If grid[i][j] == 1, then that means the ith boy can invite
    the jth girl to the party. A boy can invite at most one girl, and a
    girl can accept at most one invitation from a boy.

    Return the maximum possible number of accepted invitations.
    """
    # Using Kuhn's algorithm for unweighted bipartite graphs
    kuhnBinMatrix(grid)[0]
    """
    # Using Hungarian algorithm
    grid2 = [[-x for x in row] for row in grid]
    res = hungarianAdjMatrix(grid2)
    #print(res)
    return -res[0]
    """
    """
    # Using Ford-Fulkerson algorithm
    n_boys, n_girls = len(grid), len(grid[0])
    n = n_boys + n_girls + 2
    graph = [{} for _ in range(n)]
    for i1 in range(1, n_boys + 1):
        graph[0][i1] = 1
    for i2 in range(n_boys + 1, n - 1):
        graph[i2][n - 1] = 1
    for i1, row in enumerate(grid, start=1):
        for i2, v in enumerate(row, start=n_boys + 1):
            if v: graph[i1][i2] = 1
    #print(graph)
    return fordFulkerson(graph, 0, n - 1)
    """

def minimumOperations(self, grid: List[List[int]]) -> int:
    """
    
    Solution to (Premium) Leetcode #2123
    
    Original problem description:
    
    You are given a 0-indexed binary matrix grid. In one operation,
    you can flip any 1 in grid to be 0.

    A binary matrix is well-isolated if there is no 1 in the matrix
    that is 4-directionally connected (i.e., horizontal and vertical)
    to another 1.

    Return the minimum number of operations to make grid
    well-isolated.
    """

    # By Konig's theorem, the minimum vertex cover (i.e. the smallest possible
    # set of vertices such that every edge of the graph is adjacent to to at
    # least one of the selected vertices) is the same as the size of the
    # maximum matching of the bipartite graphs (i.e. the largest possible
    # set of disjoint pairs of vertices in the graph where each pair of
    # vertices is connected by an edge).
    # The maximum matching can be found using Kuhn's algorithm
    shape = (len(grid), len(grid[0]))

    def move(pos: Tuple[int]) -> Generator[Tuple[int], None, None]:
        if pos[0] > 0: yield (pos[0] - 1, pos[1])
        if pos[0] + 1 < shape[0]: yield(pos[0] + 1, pos[1])
        if pos[1] > 0: yield (pos[0], pos[1] - 1)
        if pos[1] + 1 < shape[1]: yield(pos[0], pos[1] + 1)
        return
    
    vertices = []
    vertex_dict = {}
    adj = []
    for i1, row in enumerate(grid):
        for i2 in range(i1 & 1, len(row), 2):
            if not grid[i1][i2]: continue
            idx1 = len(vertices)
            vertex_dict[(i1, i2)] = idx1
            vertices.append((i1, i2))
            adj.append(set())
    n1 = len(vertices)
    for i1, row in enumerate(grid):
        for i2 in range(1 - (i1 & 1), len(row), 2):
            if not grid[i1][i2]: continue
            idx1 = len(vertices)
            vertex_dict[(i1, i2)] = idx1
            vertices.append((i1, i2))
            adj.append(set())
            for pos2 in move((i1, i2)):
                idx2 = vertex_dict.get(pos2, None)
                if idx2 is None: continue
                adj[idx1].add(idx2)
                adj[idx2].add(idx1)
    return kuhnAdj(adj, n1)[0]
