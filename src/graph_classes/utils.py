#! /usr/bin/env python

from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Hashable,
    Generator,
    Any,
    Callable,
    Iterable,
)

import math
import random

from sortedcontainers import SortedSet



#def getPackageDirectory(curr_file: str, lvl: int, add: bool=True)\
#        -> str:
#    curr_dir = os.path.dirname(os.path.abspath(curr_file))
#    lvl_str = "../" * lvl
#    print(lvl_str, f"{curr_dir}/{lvl_str}")
#    pkg_dir = os.path.abspath(f"{curr_dir}/{lvl_str}")
#    if add: sys.path.append(pkg_dir)
#    return pkg_dir

#def addUnittestTemplateDirectory() -> None:
#    curr_dir = os.path.dirname(os.path.abspath(__file__))
#    unittest_template_dir = os.path.abspath(f"{curr_dir}/../unittest_templates")
#    sys.path.append(os.path.abspath(unittest_template_dir))
#    return

class UnionFind:
    def __init__(self, n: int):
        self.n = n
        self.root = list(range(n))
        self.rank = [1] * n
    
    def find(self, i: int) -> int:
        r = self.root[i]
        if r == i: return i
        res = self.find(r)
        self.root[i] = res
        return res
    
    def union(self, i1: int, i2: int) -> None:
        r1, r2 = list(map(self.find, (i1, i2)))
        if r1 == r2: return
        d = self.rank[r1] - self.rank[r2]
        if d < 0: r1, r2 = r2, r1
        elif not d: self.rank[r1] += 1
        self.root[r2] = r1
        return
    
    def connected(self, i1: int, i2: int) -> bool:
        return self.find(i1) == self.find(i2)

def forestNodePairsTraversalStatistics(
    adj: List[Dict[int, Any]],
    op: Tuple[Callable[[Any, Any], Any], Any]=(lambda x, y: x + y, 0),
) -> List[Dict[int, Tuple[Any, int]]]:
    """
    For each ordered pair of vertices in a weighted undirected forest, finds the
    the result of applying an associative (but not necessarily commultative)
    operation on all edges in the direct path between the vertices (in order)
    and the first vertex on the path from the first vertex to the second.

    Can be used to solve Leetcode #3067
    """
    n = len(adj)
    d_dict = [{} for _ in range(n)]
    def recur(idx: int, idx0: Optional[int])-> None:
        if idx0 is not None:
            d0 = adj[idx0][idx]
            for idx2, (d, idx3) in d_dict[idx0].items():
                #print(idx2, (d, idx3))
                d2 = d + d0
                d_dict[idx][idx2] = (op[0](d0, d), idx0)
                d_dict[idx2][idx] = (op[0](d, d0), d_dict[idx2][idx0][1])
            d_dict[idx][idx0] = (d0, idx0)
            d_dict[idx0][idx] = (d0, idx)
        for idx2 in adj[idx].keys():
            if idx2 == idx0: continue
            recur(idx2, idx)
        return
    recur(0, None)
    return d_dict


### Random k-tuples from first n natural numbers functions ###
### and generators                                         ###

def countFunctionNondecreasing(n: int, k: int) -> int:
    return math.comb(n + k - 1, k)
    
def countFunctionIncreasing(n: int, k: int) -> int:
    return math.comb(n, k)

def getIthNondecreasingKTuple(i: int, n: int, k: int,\
        allow_repeats: bool) -> Tuple[int]:
    
    count_func = countFunctionNondecreasing if allow_repeats else\
            countFunctionIncreasing
    
    if i < 0 or i >= count_func(n, k):
        raise ValueError("In the function "\
                "getIthNondecreasingKTuple(), the given value "\
                "of i was outside the valid range for the "\
                "given n and k.")
    
    res = []
    def recur(i: int, n: int, k: int, prev: int) -> None:
        if not k: return
        tot = count_func(n, k)
        target = tot - i
        lft, rgt = 0, n - 1
        while lft < rgt:
            mid = lft - ((lft - rgt) >> 1)
            #if tot - countFunction(n - mid, k) <= i:
            if count_func(n - mid, k) >= target:
                lft = mid
            else: rgt = mid - 1
        num = prev + lft
        res.append(num)
        lft2 = lft + (not allow_repeats)
        recur(count_func(n - lft, k) - target, n - lft2, k - 1,\
                num + (not allow_repeats))
        return
    
    recur(i, n, k, 0)
    return tuple(res)

def getIthSet(i: int, n: int, k: int) -> Set[int]:
    if k > n:
        raise ValueError("In the function getIthSet(), k must "\
                "be no larger than n")
    return set(getIthNondecreasingKTuple(i, n, k, allow_repeats=False))
    

def getIthMultiset(i: int, n: int, k: int) -> Dict[int, int]:
    res = {}
    for num in getIthNondecreasingKTuple(i, n, k, allow_repeats=True):
        res[num] = res.get(num, 0) + 1
    return res
    """
    def countFunction(n: int, k: int) -> int:
        return sp.comb(n + k - 1, k, exact=True)
    
    if i < 0 or i >= countFunction(n, k):
        raise ValueError("In the function getIthMultiset(), i must "\
                "be no less than 0 and strictly less than "\
                "(n + k - 1) choose k")
    
    res = {}
    def recur(i: int, n: int, k: int, prev: int) -> None:
        if not k: return
        tot = countFunction(n, k)
        target = tot - i
        lft, rgt = 0, n - 1
        while lft < rgt:
            mid = lft - ((lft - rgt) >> 1)
            #if tot - countFunction(n - mid, k) <= i:
            if countFunction(n - mid, k) >= target:
                lft = mid
            else: rgt = mid - 1
        num = prev + lft
        res[num] = res.get(num, 0) + 1
        recur(countFunction(n - lft, k) - target, n - lft, k - 1, num)
        return
    
    recur(i, n, k, 0)
    return res
    """

def numberedNondecreasingKTupleGenerator(inds: Iterable, n: int,\
        k: int, allow_repeats: bool, inds_sorted: bool=False)\
        -> Generator[Tuple[int], None, None]:
    if not inds_sorted:
        inds = sorted(inds)
    m = len(inds)
    
    count_func = countFunctionNondecreasing if allow_repeats else\
            countFunctionIncreasing
    
    inds_iter = iter(inds)
    
    ind_pair = [-1, next(inds_iter, float("inf"))]
    if not isinstance(ind_pair[1], int): return
    curr = []
    def recur(delta: int, n: int, k: int, prev: int)\
            -> Generator[Tuple[int], None, None]:
        if not k:
            yield_next = True
            res = tuple(curr)
            while yield_next:
                yield res
                ind_pair[0], ind_pair[1] =\
                        ind_pair[1], next(inds_iter, float("inf"))
                yield_next = (ind_pair[0] == ind_pair[1])
            return
        tot = count_func(n, k)
        tot2 = tot + delta
        lft = 0
        curr.append(0)
        while ind_pair[1] < tot2:
            target = tot2 - ind_pair[1]
            rgt = n - 1
            while lft < rgt:
                mid = lft - ((lft - rgt) >> 1)
                if count_func(n - mid, k) >= target:
                    lft = mid
                else: rgt = mid - 1
            num = prev + lft
            curr[-1] = num
            lft2 = lft + (not allow_repeats)
            yield from recur(delta + tot - count_func(n - lft, k),\
                    n - lft2, k - 1, num + (not allow_repeats))
            lft += 1
        curr.pop()
        return
    
    yield from recur(0, n, k, 0)
    return

def countFunctionAll(n: int, k: int) -> int:
    return n ** k
    
def countFunctionDistinct(n: int, k: int) -> int:
    return math.perm(n, k)

def findKthMissing(lst: SortedSet, k: int) -> int:
    # k starts at 0
    # Assumes lst contains only non-negative integers
    if not lst or k >= lst[-1]: return k + len(lst)
    
    def countLT(num: int) -> int:
        return num - lst.bisect_left(num)
    
    lft, rgt = k, k + len(lst)
    while lft < rgt:
        mid = lft - ((lft - rgt) >> 1)
        if countLT(mid) <= k: lft = mid
        else: rgt = mid - 1
    return lft

def getIthKTuple(i: int, n: int, k: int,\
        allow_repeats: bool) -> Tuple[int]:
    
    count_func = countFunctionAll if allow_repeats else\
            countFunctionDistinct
    
    if i < 0 or i >= count_func(n, k):
        raise ValueError("In the function "\
                "getIthKTuple(), the given value  of i was outside "\
                "the valid range for the given n and k.")
    
    count_func = countFunctionAll if allow_repeats else\
            countFunctionDistinct
    
    if allow_repeats:
        res = []
        for j in range(k):
            ans, i = divmod(i, count_func(n, k - j - 1))
            res.append(ans)
        return tuple(res)
    seen = SortedSet()
    res = []
    for j in range(k):
        ans, i = divmod(i, count_func(n - j - 1, k - j - 1))
        ans = findKthMissing(seen, ans)
        res.append(ans)
        seen.add(ans)
    return tuple(res)

def numberedKTupleGenerator(inds: Iterable, n: int,\
        k: int, allow_repeats: bool, inds_sorted: bool=False)\
        -> Generator[Tuple[int], None, None]:
    if not inds_sorted:
        inds = sorted(inds)
    m = len(inds)
    
    count_func = countFunctionAll if allow_repeats else\
            countFunctionDistinct
    
    def numProcessorAll(num: int) -> int:
        return num
    
    def numProcessorDistinct(num: int) -> int:
        num = findKthMissing(seen, num)
        seen.add(num)
        return num
    
    def seenProcessorAll(num: int) -> None:
        return
    
    def seenProcessorDistinct(num: int) -> None:
        seen.remove(num)
        return
    
    if allow_repeats:
        num_processor = numProcessorAll
        seen_processor = seenProcessorAll
    else:
        num_processor = numProcessorDistinct
        seen_processor = seenProcessorDistinct
        seen = SortedSet()
    
    inds_iter = iter(inds)
    
    ind_pair = [-1, next(inds_iter, float("inf"))]
    if not isinstance(ind_pair[1], int): return
    curr = []
    def recur(delta: int, n: int, k: int)\
            -> Generator[Tuple[int], None, None]:
        if not k:
            yield_next = True
            res = tuple(curr)
            while yield_next:
                yield res
                ind_pair[0], ind_pair[1] =\
                        ind_pair[1], next(inds_iter, float("inf"))
                yield_next = (ind_pair[0] == ind_pair[1])
            return
        n2 = n - (not allow_repeats)
        tot = count_func(n, k)
        tot2 = tot + delta
        lft = 0
        curr.append(0)
        div = count_func(n2, k - 1)
        while ind_pair[1] < tot2:
            
            q = (ind_pair[1] - delta) // div
            curr[-1] = num_processor(q)
            yield from recur(delta + q * div, n2, k - 1)
            seen_processor(curr[-1]) 
        curr.pop()
        return
    
    yield from recur(0, n, k)
    return

def randomSampleWithoutReplacement(n: int, k: int) -> List[int]:
    seen = SortedSet()
    res = []
    for i in range(k):
        num = findKthMissing(seen, random.randrange(0, n - i))
        res.append(num)
        seen.add(num)
    return res

def randomKTupleGenerator(n: int, k: int,\
        mx_n_samples: int, allow_index_repeats: bool,\
        allow_tuple_repeats: bool, nondecreasing: bool)\
        -> Generator[Tuple[int], None, None]:
    
    if nondecreasing:
        count_func = countFunctionNondecreasing if allow_index_repeats\
                else countFunctionIncreasing
        gen_func = numberedNondecreasingKTupleGenerator
    else:
        count_func = countFunctionAll if allow_index_repeats else\
                countFunctionDistinct
        gen_func = numberedKTupleGenerator
    
    tot = count_func(n, k)
    #print(mx_n_samples, tot)
    inds = [random.choice(range(tot)) for _ in range(mx_n_samples)]\
            if allow_tuple_repeats else\
            randomSampleWithoutReplacement(tot,\
            min(mx_n_samples, tot))
    yield from gen_func(inds, n,\
            k, allow_index_repeats, inds_sorted=False)

################

def verticesConnectedIndex(graph: Any,\
        idx1: int, idx2: int) -> bool:
    """
    Function finding whether there exists a path in the (directed)
    graph represented by input argument graph from the vertex with
    index idx1 to the vertex with index idx2.
    
    Args:
        Required positional:
        graph (class descending from LimitedGraphTemplate): The graph
                for which the existence of a path as described above
                is being assessed. This graph may be directed or
                undirected, weighted or unweighted.
        idx1 (int): The index of the vertex in the graph where the
                path (if it exists) starts
        idx2 (int): The index of the vertex in the graph where the
                path (if it exists) ends
    
    Returns:
    Boolean (bool), with value True if there exists a path from the
    vertex with index idx1 to the vertex with index idx2 in the graph
    and False otherwise.
    """
    if idx1 == idx2: return True
    
    stk = [idx1]
    seen = {idx1}
    while stk:
        idx = stk.pop()
        for idx3 in graph.getAdjIndex(idx).keys():
            if idx3 in seen: continue
            if idx3 == idx2: return True
            stk.append(idx3)
            seen.add(idx3)
    return False

def verticesConnected(graph: Any,\
        v1: Hashable, v2: Hashable) -> bool:
    """
    Function finding whether there exists a path in the (directed)
    graph represented by input argument graph from the vertex v1 to
    the vertex v2.
    
    Args:
        Required positional:
        graph (class descending from LimitedGraphTemplate): The graph
                for which the existence of a path as described above
                is being assessed. This graph may be directed or
                undirected, weighted or unweighted.
        v1 (hashable object): The vertex in the graph where the path
                (if it exists) starts
        v2 (hashable objext): The vertex in the graph where the path
                (if it exists) ends
    
    Returns:
    Boolean (bool), with value True if there exists a path from the
    vertex v1 to the vertex v2 in the graph and False otherwise.
    """
    idx1, idx2 = list(map(graph.vertex2Index, (v1, v2)))
    return verticesConnectedIndex(graph, idx1, idx2)

def containsDirectedCycle(graph: Any)\
        -> bool:
    """
    Function finding whether there exists a directed cycle in the
    directed graph represented by input argument graph.
    A directed cycle is a path that of at least one step that starts
    and ends at the same vertex, where each step in the path is a
    movement from one vertex to another (not necessarily different)
    vertex along a directed edge in the same direction as the edge's
    direction.
    
    Args:
        Required positional:
        graph (class descending from LimitedGraphTemplate): The
                directed graph for which the existence of a directed
                cycle is being assessed. This directed graph may be
                weighted or unweighted.
    
    Returns:
    Boolean (bool), with value True if there exists a directed cycle in
    the directed graph and False otherwise.
    """
    seen = set()
    curr_seen = set()
    def dfs(idx: int) -> bool:
        if idx in curr_seen:
            return True
        elif idx in seen:
            return False
        curr_seen.add(idx)
        seen.add(idx)
        for idx2 in graph.getAdjIndex(idx).keys():
            if dfs(idx2): return True
        curr_seen.remove(idx)
        return False
    
    for idx in range(graph.n):
        if dfs(idx): return True
    return False

class FenwickTree:
    """
    Creates a Fenwick tree for a sequence of elements of a commutative
    monoid. When first initialised, the every element of the sequence
    is set as the identity of the monoid.
    Also note that the sequence is zero-indexed
    
    Args:
        Required positional:
        n (int): the length of the sequence
        op (2-tuple of a function and an element of the monoid):
                the associative, commutative binary operation of the
                commutative monoid and its identity element
            Example: Addition of integers (lambda x, y: x + y, 0)
    
    Attributes:
        n (int): the length of the sequence
        arr (list of monoid elements): the Fenwick tree array
        op (2-tuple of a function and an element of the monoid):
                the associative, commutative binary operation of the
                commutative monoid and its identity element
             Example: Addition of integers (lambda x, y: x + y, 0)
            
    """
    def __init__(self, n: int, op: tuple):
        self.n = n
        self.arr = [op[1]] * (n + 1)
        self.op = op

    def query(self, i: int) -> int:
        """
        Returns the cumulative application of the commutative,
        associative binary operation of the monoid on all elements
        of the sequence with index no greater than i. This is
        referred to as the generalised summation up to the
        ith index
        
        Args:
            Required positional:
            i (int): the index at which the generalised summation
                    stops
        """
        if i < 0: return self.op[1]
        elif i >= self.n: i = self.n
        else: i += 1
        res = self.op[1]
        while i > 0:
            res = self.op[0](res, self.arr[i])
            i -= i & -i
        return res
    
    def update(self, i: int, v) -> None:
        """
        Increments the ith element of the sequence (recall the sequence
        is zero-indexed)- i.e. the ith element will be replaced by
        the operation self.op performed between the current ith
        element and v.
        
        Args:
            Required positional:
            i (int): the index of the sequence to be updated
            v (element of the monoid): the value to which the ith index
                    of the sequence is to be incremented.
        """
        i += 1
        while i <= self.n:
            self.arr[i] = self.op[0](self.arr[i], v)
            i += i & -i
        return
