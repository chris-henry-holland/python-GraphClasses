#!/usr/bin/env python

from typing import (
    List,
    Tuple,
    Set,
    Dict,
    Optional,
    Union,
)

from collections import deque, Counter

def findCentralNodes(
    graph: List[Set[int]],
    preserve_graph: bool=True,
) -> Tuple[int]:
    # BFS
    
    # Creating deepcopy of graph as the algorithm deconstructs the
    # graph during the calculation
    if preserve_graph:
        graph = [set(x) for x in graph]
    n = len(graph)
    qu = deque()
    for idx in range(n):
        if len(graph[idx]) <= 1:
            qu.append(idx)
    n_remain = n
    while True:
        if n_remain <= 2: break
        n_remain -= len(qu)
        for _ in range(len(qu)):
            idx = qu.popleft()
            for idx2 in graph[idx]:
                graph[idx2].remove(idx)
                if len(graph[idx2]) == 1:
                    qu.append(idx2)
    return tuple(qu)

def createBinaryLift(parent: List[int]) -> List[List[int]]:
    res = [[x] for x in parent]
    remain = set(idx for idx in range(len(parent)) if parent[idx] != -1)
    for i in range(len(remain)):
        if not remain: break
        for idx in list(remain):
            idx2 = res[idx][i]
            res[idx].append(res[idx2][min(i, len(res[idx2]) - 1)])
            if res[idx][-1] == -1: remain.remove(idx)
    return res

class TreeLowestCommonAncestorFinder:
    def __init__(
        self,
        graph: List[Set[int]],
        root: Optional[int]=None
    ):
        self.graph = graph
        if root is None:
            root = findCentralNodes(graph)[0]
        self.n = len(graph)
        self.root = root
        self.parents, self.depth = self.getParentsAndDepth()
    
    @property
    def bin_lift(self) -> List[List[int]]:
        res = getattr(self, "_bin_lift", None)
        if res is None:
            res = createBinaryLift(self.parents)
            self._bin_lift = res
        return res
        
    def getParentsAndDepth(self) -> Tuple[List[int]]:
        depth = [0] * self.n
        parents = [-1] * self.n
        def dfs(idx: int, idx0: Optional[int]=None, dist: int=0) -> None:
            depth[idx] = dist
            for idx2 in self.graph[idx].difference({idx0}):
                parents[idx2] = idx
                dfs(idx2, idx0=idx, dist=dist + 1)
            return
        dfs(self.root)
        return parents, depth
    
    def __call__(self, idx1: int, idx2: int) -> int:
        if idx1 == idx2: return idx1
        # Raise the deeper of idx1 and idx2 in the tree
        # so that the starting points are at the same depth
        if self.depth[idx1] > self.depth[idx2]:
            idx1, idx2 = idx2, idx1
        d = self.depth[idx2] - self.depth[idx1]
        i = 0
        while d:
            if d & 1:
                idx2 = self.bin_lift[idx2][i]
            d >>= 1
            i += 1
        if idx1 == idx2: return idx1
        # Find the highest level in the tree for which the
        # ancestors of the two nodes in that level are different
        j = float("inf")
        
        while True:
            for j in reversed(range(min(j, len(self.bin_lift[idx1])))):
                if self.bin_lift[idx1][j] != self.bin_lift[idx2][j]:
                    break
            else: break
            idx1 = self.bin_lift[idx1][j]
            idx2 = self.bin_lift[idx2][j]
        # The answer is the parent of each of the ancestor nodes
        # in the level found
        return self.bin_lift[idx1][0]

def closestNode(
    n: int,
    edges: List[List[int]],
    query: List[List[int]]
) -> List[int]:
    """
    
    
    Solution to Leetcode #2277 (Premium)
    
    Original problem description:
    
    You are given a positive integer n representing the number of nodes
    in a tree, numbered from 0 to n - 1 (inclusive). You are also given
    a 2D integer array edges of length n - 1, where
    edges[i] = [node1i, node2i] denotes that there is a bidirectional
    edge connecting node1i and node2i in the tree.

    You are given a 0-indexed integer array query of length m where
    query[i] = [starti, endi, nodei] means that for the ith query, you
    are tasked with finding the node on the path from starti to endi
    that is closest to nodei.

    Return an integer array answer of length m, where answer[i] is the
    answer to the ith query.
    """
    graph = [set() for _ in range(n)]
    for e in edges:
        graph[e[0]].add(e[1])
        graph[e[1]].add(e[0])
    
    tlcaf = TreeLowestCommonAncestorFinder(graph)
    
    res = [0] * len(query)
    for i_q, (idx1, idx2, idx0) in enumerate(query):
        if idx0 == idx1 or idx0 == idx2:
            res[i_q] = idx0
            continue
        elif idx1 == idx2:
            res[i_q] = idx1
            continue
        lca1 = tlcaf(idx1, idx2)
        lca2 = tlcaf(idx1, idx0)
        lca3 = tlcaf(idx2, idx0)

        if lca2 == lca3: res[i_q] = lca1
        elif lca2 == lca1: res[i_q] = lca3
        elif lca3 == lca1: res[i_q] = lca2
    return res

def treeNodePairsTraversalStatistics(
    adj: List[Dict[int, Union[int, float]]],
) -> List[Dict[int, Tuple[Union[int, float], int]]]:
    """
    For each pair of nodes in a weighted undirected tree, finds the
    distance between each point and the adjacent node on the path
    between the two.
    """
    res = [{} for _ in range(len(adj))]
    def recur(idx: int, idx0: Optional[int], d0: Optional[int])-> None:
        """
        d0 is the distance from the node with index idx0 to the node
        with index idx
        """
        if idx0 is not None:
            for idx2, (d, idx3) in res[idx0].items():
                #print(idx2, (d, idx3))
                d2 = d + d0
                res[idx][idx2] = (d2, idx0)
                res[idx2][idx] = (d2, res[idx2][idx0][1])
            res[idx][idx0] = (d0, idx0)
            res[idx0][idx] = (d0, idx)
        for idx2, d2 in adj[idx].items():
            if idx2 == idx0: continue
            recur(idx2, idx, d2)
        return
    recur(0, None, None)
    return res

def countPairsOfConnectableServers(
    edges: List[List[int]],
    signalSpeed: int,
) -> List[int]:
    """
    
    Solution to Leetcode #3067
    
    Original problem description:
    
    You are given an unrooted weighted tree with n vertices
    representing servers numbered from 0 to n - 1, an array edges where
    edges[i] = [ai, bi, weighti] represents a bidirectional edge
    between vertices ai and bi of weight weighti. You are also given an
    integer signalSpeed.
    
    Two servers a and b are connectable through a server c if:

    a < b, a != c and b != c.
    The distance from c to a is divisible by signalSpeed.
    The distance from c to b is divisible by signalSpeed.
    The path from c to b and the path from c to a do not share any
    edges.
    Return an integer array count of length n where count[i] is the
    number of server pairs that are connectable through the server i.
    """
    n = len(edges) + 1
    adj = [{} for _ in range(n)]
    for e in edges:
        adj[e[0]][e[1]] = e[2]
        adj[e[1]][e[0]] = e[2]

    tree_pairs = treeNodePairsTraversalStatistics(adj)
    connect = [
        {
            x: y[1] for x, y in d.items() if not y[0] % signalSpeed
        } 
        for d in tree_pairs
    ]
    res = [0] * n
    for idx in range(n):
        f_dict = Counter(connect[idx].values())
        f_lst = list(f_dict.values())
        cnt = 0
        for f in f_lst:
            res[idx] += f * cnt
            cnt += f
    return res
