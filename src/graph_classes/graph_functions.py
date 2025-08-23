#! /usr/bin/env python
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
)


from graph_classes.explicit_graph_types import (
    ExplicitUndirectedUnweightedGraph,
    ExplicitUndirectedWeightedGraph,
    ExplicitDirectedUnweightedGraph,
    ExplicitDirectedWeightedGraph,
)

from graph_classes.grid_graph_types import (
    Grid,
    GridUndirectedUnweightedGraph,
    GridUndirectedWeightedGraph,
    GridDirectedUnweightedGraph,
    GridDirectedWeightedGraph,
)

# Example uses of the graph classes to solve problems

# Consider adding: #787 (Bellman-Ford within given number of moves), #1514 (Dijkstra for explicit graph), The Maze problems #490, #499, #505 (Dijkstra for grid graphs with non-standard moves), #1135, #1584, #1168 (MST), #1489 (MST and bridge finding)

def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    """
    Solution to Leetcode #207 (Course Schedule) using Kahn's algorithm
    """
    if not numCourses: return True
    graph = ExplicitDirectedUnweightedGraph(range(numCourses), prerequisites)
    res = graph.kahnIndex()
    return bool(res)

def findOrder(self, numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Solution to Leetcode #210 (Course Schedule II) using Kahn's algorithm
    """
    graph = ExplicitDirectedUnweightedGraph(range(numCourses), prerequisites)
    return graph.kahn()

def alienOrder(words: List[str]) -> str:
    """
    Solution to Leetcode #269 (Alien Dictionary) using Kahn's algorithm
    
    
    Original Leetcode problem description:
    
    There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.

You are given a list of strings words from the alien language's dictionary. Now it is claimed that the strings in words are 
sorted lexicographically
 by the rules of this new language.

If this claim is incorrect, and the given arrangement of string in words cannot correspond to any order of letters, return "".

Otherwise, return a string of the unique letters in the new alien language sorted in lexicographically increasing order by the new language's rules. If there are multiple solutions, return any of them.
    """
    # Check- Premium
    """
    n = len(words)
    out_edges = {}
    for i1, w1 in enumerate(words):
        for l1 in w1: out_edges.setdefault(l1, set())
        for i2 in range(i1 + 1, n):
            w2 = words[i2]
            for l1, l2 in zip(w1, w2):
                if l1 != l2:
                    out_edges[l1].add(l2)
                    break
            else:
                if len(w2) < len(w1): return ""
    """
    n = len(words)
    edges = []
    l_set = set()
    for i1, w1 in enumerate(words):
        for l1 in w1:
            l_set.add(l1)
        for i2 in range(i1 + 1, n):
            w2 = words[i2]
            for l1, l2 in zip(w1, w2):
                if l1 != l2:
                    edges.append((l1, l2))
                    break
            else:
                if len(w2) < len(w1): return ""
    graph = ExplicitDirectedUnweightedGraph(list(l_set), edges)
    res = graph.kahn()
    return "".join(res)

def minimumSemesters(n: int, relations: List[List[int]]) -> int:
    """
    Solution to Leetcode #1136 (Parallel Courses) using Kahn's
    algorithm
    """
    # Check- Premium
    if not n: return 0
    graph = ExplicitUndirectedUnweightedGraph(list(range(1, n + 1)),\
            relations)
    res = graph.kahnLayering()
    return len(res) if res else -1
    

def findSmallestSetOfVertices(n: int, edges: List[List[int]],\
        alg: str="tarjan") -> List[int]:
    """
    Modified Leetcode #1557 (demonstrates possible use of KosarauAdj()
    and TarjanSCCAdj() through condenseSCCAdj())
    
    Original Leetcode problem description:
    
    Given a directed acyclic graph, with n vertices numbered from 0 to
    n-1, and an array edges where edges[i] = [fromi, toi] represents a
    directed edge from node fromi to node toi.

    Find the smallest set of vertices from which all nodes in the graph
    are reachable. It's guaranteed that a unique solution exists.

    Notice that you can return the vertices in any order.
    
    Modification: Directed graph is not necessarily acyclic (much
    harder problem)- gives one of the possible solutions
    
    Input argument alg can be "tarjan" or "kosaraju", the former using
    TarjanSCCAdj() and the latter using KosarauAdj() (through
    condenseSCCAdj()).
    """
    # Note to get every possible solution, give all combinations of
    # possible representatives of SCCs with no external incoming edges
    graph = ExplicitDirectedUnweightedGraph(range(n), edges)
    scc_repr, scc_groups, condensed_graph =\
            graph.condenseSCC(alg=alg, set_condensed_in_degrees=True)
    res = [condensed_graph.index2Vertex(idx)\
            for idx in range(condensed_graph.n)\
            if not condensed_graph.inDegreeIndex(idx)]
    return res
    
def maximumDetonation(bombs: List[List[int]], alg: str="Tarjan") -> int:
    """
    Leetcode #2101 (demonstrates possible use of kosarau() and
    tarjanSCC() methods through the condenseSCC() method)
    
    Original Leetcode problem description:
    
    You are given a list of bombs. The range of a bomb is defined as the
    area where its effect can be felt. This area is in the shape of a
    circle with the center as the location of the bomb.

    The bombs are represented by a 0-indexed 2D integer array bombs where
    bombs[i] = [xi, yi, ri]. xi and yi denote the X-coordinate and
    Y-coordinate of the location of the ith bomb, whereas ri denotes the
    radius of its range.

    You may choose to detonate a single bomb. When a bomb is detonated,
    it will detonate all bombs that lie in its range. These bombs will
    further detonate the bombs that lie in their ranges.

    Given the list of bombs, return the maximum number of bombs that can
    be detonated if you are allowed to detonate only one bomb.
    
    Input argument alg can be "Tarjan" or "Kosaraju", the former using
    tarjanSCC() and the latter using kosaraju() (through condenseSCC()).
    """
    n = len(bombs)
    edges = []
    dist_sq = [0] * n
    for i1, (x1, y1, r1) in enumerate(bombs):
        dist_sq[i1] = r1 ** 2
        for i2 in range(i1):
            x2, y2, r2 = bombs[i2]
            ds = (x2 - x1) ** 2 + (y2 - y1) ** 2
            if ds <= dist_sq[i1]:
                edges.append((i1, i2))
            if ds <= dist_sq[i2]:
                edges.append((i2, i1))
    need_in_degrees = (alg == "kosaraju")
    graph = ExplicitDirectedUnweightedGraph(range(n), edges, store_in_degrees=need_in_degrees,\
            set_in_degrees=need_in_degrees)
    
    scc_repr, scc_groups, scc_graph = graph.condenseSCC(alg=alg)
    n2 = scc_graph.n
    
    sz_lst = [len(scc_groups[scc_graph.vertices[i]])\
            for i in range(n2)]
    memo = {}
    def dfs(i: int) -> Set[int]:
        args = i
        if args in memo.keys(): return memo[args]
        memo[args] = set()
        res = {i}
        for i2 in scc_graph.adjGeneratorIndex(i):
            res |= dfs(i2)
        memo[args] = res
        return res
    res = 0
    for i in range(n2):
        if scc_graph.inDegreeIndex(i): continue
        res = max(res, sum(sz_lst[i2] for i2 in dfs(i)))
    return res

def findItinerary(tickets: List[List[str]]) -> List[str]:
    """
    Solution to Leetcode #332 (demonstrates use of hierholzer() method)
    
    
    
    Original Leetcode problem description:
    
    You are given a list of airline tickets where
    tickets[i] = [fromi, toi] represent the departure and the arrival
    airports of one flight. Reconstruct the itinerary in order and
    return it.

    All of the tickets belong to a man who departs from "JFK", thus,
    the itinerary must begin with "JFK". If there are multiple valid
    itineraries, you should return the itinerary that has the smallest
    lexical order when read as a single string.
    - For example, the itinerary ["JFK", "LGA"] has a smaller lexical
      order than ["JFK", "LGB"].
    
    You may assume all tickets form at least one valid itinerary. You
    must use all the tickets once and only once.
    """
    start = "JFK"
    end = None

    vertices = set()
    for e in tickets:
        vertices.add(e[0])
        vertices.add(e[1])
    vertices = sorted(vertices)
    graph = ExplicitDirectedUnweightedGraph(vertices, tickets, store_in_degrees=True,\
            set_in_degrees=True)
    return graph.hierholzer(start=start, sort=True, reverse=False)

def validArrangement(pairs: List[List[int]]) -> List[List[int]]:
    """
    
    Solution to Leetcode #2097 (Valid Arrangement of Pairs)
    
    Original problem description:
    
    You are given a 0-indexed 2D integer array pairs where
    pairs[i] = [starti, endi]. An arrangement of pairs is valid if for
    every index i where 1 <= i < pairs.length, we have
    endi-1 == starti.

    Return any valid arrangement of pairs.

    Note: The inputs will be generated such that there exists a valid
    arrangement of pairs.
    """
    # TODO- Adapt to utilise Graph class
    out_edges = {}
    in_edges = {}
    for p in pairs:
        out_edges.setdefault(p[0], set())
        out_edges.setdefault(p[1], set())
        out_edges[p[0]].add(p[1])
        in_edges.setdefault(p[0], set())
        in_edges.setdefault(p[1], set())
        in_edges[p[1]].add(p[0])
    start = None
    for v, v_set in out_edges.items():
        l1, l2 = len(v_set), len(in_edges[v])
        if l1 == l2:
            continue
        elif l1 - l2 == 1:
            if start is not None: return []
            start = v
        elif l2 - l1 != 1: return []
    if start is None:
        start = next(iter(out_edges.keys()))
    #if start is not None:
    #    out_edges[end].add(start)
    #    in_edges[start].add(end)

    def HierholzerAlgorithm(out_edges: Dict[int, Set[int]], start: int) -> List[int]:
        res = []
        curr = [start]
        v = start
        while curr:
            v = curr[-1]
            if out_edges[v]:
                v2 = next(iter(out_edges[v]))
                out_edges[v].remove(v2)
                curr.append(v2)
            else:
                res.append(curr.pop())
        return res[::-1]
    
    path = HierholzerAlgorithm(out_edges, start)
    res = [[path[0]]]
    for i in range(1, len(path) - 1):
        res[-1].append(path[i])
        res.append([path[i]])
    res[-1].append(path[-1])
    return res

def criticalConnections(n: int, connections: List[List[int]]) -> List[List[int]]:
    """
    Leetcode #1192 (demonstrates possible use of tarjanBridge() method)
    
    Original Leetcode problem description:
    
    There are n servers numbered from 0 to n - 1 connected by
    undirected server-to-server connections forming a network
    where connections[i] = [ai, bi] represents a connection
    between servers ai and bi. Any server can reach other
    servers directly or indirectly through the network.

    A critical connection is a connection that, if removed,
    will make some servers unable to reach some other server.

    Return all critical connections in the network in any order.
    """
    graph = ExplicitUndirectedUnweightedGraph(range(n), connections)
    return graph.tarjanBridge()

def minPushBox(arr: List[List[str]], search_alg: str="astar") -> int:
    """
    Solution to Leetcode #1263
    
    Demonstrates the potential use of the UnweightedDirectedGridGraph
    and UnweightedUndirectedGridGraph classes, the search methods
    bredthFirstSearchIndex(), dijkstra() or aStarIndex() (depending
    on user choice, with unidirectional or bidirectional for each
    being avaliable) and the articulation point finding method
    tarjanArticulationFullIndex() (which uses Tarjan's algorithm
    for articulation points.
    
    
    
    Original Leetcode problem description:
    
    A storekeeper is a game in which the player pushes boxes around in
    a warehouse trying to get them to target locations.

    The game is represented by an m x n grid of characters grid where
    each element is a wall, floor, or box.

    Your task is to move the box 'B' to the target position 'T' under
    the following rules:
    - The character 'S' represents the player. The player can move up,
      down, left, right in grid if it is a floor (empty cell).
    - The character '.' represents the floor which means a free cell to
      walk.
    - The character '#' represents the wall which means an obstacle
      (impossible to walk there).
    - There is only one box 'B' and one target cell 'T' in the grid.
    - The box can be moved to an adjacent free cell by standing next to
      the box and then moving in the direction of the box. This is a
      push.
    - The player cannot walk through the box.
    
    Return the minimum number of pushes to move the box to the target.
    If there is no way to reach the target, return -1.
    
    """
    # The chosen search algorithm and whether or not to use bidirectional
    # search
    # Options: "bredthFirstSearch", "dijkstra", "aStar"
    search_alg = "aStar"
    bidirectional = True
    dijkstra_to_bfs_allowed = False

    search_alg_full = f"{search_alg}Index"
    def searchAlgorithmIndex(graph, search_kwargs: dict, alg: str=search_alg_full)\
            -> Tuple[Union[int, float], Tuple[int]]:
        if "heuristic" in search_kwargs and not alg.startswith("aStar"):
            search_kwargs.pop("heuristic")
        if alg.startswith("dijkstra"):
            search_kwargs["use_bfs_if_poss"] = dijkstra_to_bfs_allowed
        return getattr(graph, alg)(**search_kwargs)

    # Setting up the grid as a weighted undirected graph representing simple
    # movement between the empty spaces, one step at a time
    n_dim = 2
    grid = Grid(n_dim, arr)
    move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
            "restrict_direct_func": None}
    #weight_func = lambda grid, grid_idx1, state_idx1, grid_idx2, state_idx2, mv, n_step: n_step
    connected_func = lambda grid, grid_idx1, state_idx1, grid_idx2, state_idx2, mv, n_step: True
    n_state_func = lambda grid, grid_idx: int(grid.arr_flat[grid_idx] != "#")
    graph = GridUndirectedUnweightedGraph(grid, move_kwargs=move_kwargs,\
            connected_func=connected_func, n_state_func=n_state_func)
    shape = grid.shape
    n_dim = grid.n_dim
    
    # Identifying the articulation points of the graph, i.e. the vertices
    # which if removed increases the number of connected components of the
    # graph. These represent the positions in the grid for which, when the
    # box is at these positions, the player will not be able to freely
    # access the rest of the empty spaces in the graph without first moving
    # the box, and so when it comes to moving the box may restrict the
    # directions it may be pushed.
    # Also identifies the connected components of the graph (i.e. the
    # partitions of the grid that, ignoring the box, the player may move
    # freely within but may not move between, due to being separated
    # by walls with no traversable gaps)
    artic, cc_dict, n_cc = graph.tarjanArticulationFullIndex()
    artic_grid_index = {}
    n_artic_grps = {}
    for idx1, idx2_sets in artic.items():
        grid_idx1 = graph._index2GridIndex(idx1)[0]
        state_dict = {}
        for state_idx, idx2_set in enumerate(idx2_sets):
            for idx2 in idx2_set:
                state_dict[graph._index2GridIndex(idx2)[0]] = state_idx
        artic_grid_index[grid_idx1] = state_dict
        n_artic_grps[grid_idx1] = len(idx2_sets)
    
    # Finding the graph index of the positions of the player, the box and the target
    # and the grid index of the latter two
    found = set()
    err_msg = "The grid must contain one each of the characters "\
                "'S', 'B' and 'T'"
    l_set = set("SBT")
    for idx, (grid_idx, _) in enumerate(graph.vertexGridIndexGenerator()):
        l = graph.grid.arr_flat[grid_idx]
        if l not in l_set: continue
        if l in found:
            raise ValueError(err_msg)
        if l == "S": init_char_idx = idx
        elif l == "B":
            box_idx = idx
            box_grid_idx = grid_idx
        elif l == "T":
            target_idx = idx
            target_grid_idx = grid_idx
        else: continue
        found.add(l)
        if len(found) == 3: break
    else: raise ValueError(err_msg)

    # Checking the player, box and target are all in  the same
    # connected region (if not, pushing the box to the target is
    # impossible)
    if len(set(map(cc_dict.__getitem__, (init_char_idx, box_idx, target_idx)))) != 1:
        return -1
    
    # Setting up the grid as a directed weighted graph representing how
    # the box may be pushed around the grid by the player
    n_state_func = lambda grid, grid_idx:\
            n_artic_grps.get(grid_idx, int(grid.arr_flat[grid_idx] != "#"))

    def connectedFunction(grid: Grid, grid_idx1: int, state_idx1: int, grid_idx2: int,\
            state_idx2: int, mv: int, n_step: int) -> Optional[int]:
        poss_mv = {x[0] for x in grid.stepIndexGenerator(grid_idx1, n_diag=0) if x[1]}
        char_grid_idx = grid_idx1 - mv
        if -mv not in poss_mv or grid.arr_flat[char_grid_idx] == "#" or\
                (grid_idx1 in artic_grid_index.keys() and\
                artic_grid_index[grid_idx1][char_grid_idx] != state_idx1 or\
                (grid_idx2 in artic_grid_index.keys() and\
                artic_grid_index[grid_idx2][grid_idx1] != state_idx2)):
            return False
        return True
    
    graph2 = GridDirectedUnweightedGraph(grid, move_kwargs=move_kwargs,\
            connected_func=connectedFunction, n_state_func=n_state_func)

    # Finding the appropriate starting vertex in the graph representing the
    # possible movements of the box (graph2) based on the box's and player's
    # initial positions
    md_lst = [1]
    curr = 1
    for s in shape:
        curr *= s
        md_lst.append(curr)
    def heuristic1(idx1: int, idx2: int) -> int:
        (grid_idx1, _), (grid_idx2, _) = list(map(graph._index2GridIndex, (idx1, idx2)))
        return sum(abs(((grid_idx2 % md_lst[i + 1]) // md_lst[i]) -\
                ((grid_idx1 % md_lst[i + 1]) // md_lst[i])) for i in range(n_dim))
    if box_grid_idx in artic_grid_index.keys():
        #print("Searching for start vertex")
        search_kwargs = {"start_inds": {init_char_idx},\
                "end_inds": {box_idx}, "heuristic": heuristic1,\
                "bidirectional": bidirectional}
        _, path = searchAlgorithmIndex(graph, search_kwargs, alg=search_alg_full)
        char_grid_idx = graph._index2GridIndex(path[-2])[0]
        init_state_idx = artic_grid_index.get(box_grid_idx, {char_grid_idx: 0})[char_grid_idx]
    else:
        for char_grid_idx in grid.movesIndexGenerator(box_grid_idx, **graph.move_kwargs):
            break
        else: return -1
        init_state_idx = 0

    box_idx2 = graph2._gridIndex2Index(box_grid_idx, init_state_idx)

    # Finding indices of the vertices in graph2 that correspond to the box being
    # at the target (note that there may be more than one, due to the player
    # ending up in different locations)
    end_inds = {graph2._gridIndex2Index(target_grid_idx, i)\
            for i in range(graph2.gridIndexNState(target_grid_idx))}
    
    # Performing the search of graph2 for the shortest path (i.e. the path
    # that involves the fewest moves of the block) if such a path exists.
    def heuristic2(idx1: int, idx2: int) -> int:
        (grid_idx1, _), (grid_idx2, _) = list(map(graph2._index2GridIndex, (idx1, idx2)))
        return sum(abs(((grid_idx2 % md_lst[i + 1]) // md_lst[i]) -\
                ((grid_idx1 % md_lst[i + 1]) // md_lst[i])) for i in range(n_dim))
    search_kwargs = {"start_inds": {box_idx2},\
            "end_inds": end_inds, "heuristic": heuristic2,\
            "bidirectional": bidirectional}
    d, path = searchAlgorithmIndex(graph2, search_kwargs, alg=search_alg_full)
    return d
