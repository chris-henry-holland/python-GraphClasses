#!/usr/bin/env python

from typing import (
    Callable,
    List,
    Optional,
    Tuple,
    Union,
    Any,
)

import math
from sortedcontainers import SortedSet

class SegmentTree(object):
    """
    Creates a segment tree for a sequence consisting of elements
    of a set based around a specified associative (and possibly
    commutative) binary operation on that set (i.e. an operation
    that takes as input an ordered pair of elements of the set
    and outputs an element of the same set).

    On initialization, the sequence is entirely populated with the
    identity element of the given binary operation
    
    Initialization args:
        Required positional:
        start_idx (int): Lower bound for sequence index values
        end_idx (int): Upper bound for sequence index values
        
        Optional named:
        op (string or 3-tuple of a function on two variables, a
                numeric value and a boolean): Specifies the
                associative binary operation to be applied, its
                identity and whether the operation is commutative,
                either as a 3-tuple giving each of these in order or
                a string identifying a pre-implemented standard
                binary operation.
                The standard binary operations pre-implemented are:
                 "sum" (gives the interval sums for real numeric values-
                    identity 0)
                 "product" (gives the interval products for
                    real numeric values- identity 1)
                 "max" (gives the interval maxima for real numeric
                    values- identity -float("inf"))
                 "min" (gives the interval minima for real numeric
                    values- identity float("inf"))
                 "gcd" (gives the greatest common divisor over all integers
                    in the interval for integer values- identity 0)
                 "union" (gives the union of sets over the intervals
                    for sets- identity set(), the empty set)
                 "bitwise and" (gives the interval bitwise and for
                    integers- identity -1)
                 "bitwise or" (gives the interval bitwise or for
                    integers- identity 0)
                 "bitwise xor" (gives the interval bitwise excluisve
                    or for integers- identity 0)
            Default: "sum", or equivalently:
                    {"sum": (lambda x, y: x + y, 0, True)}
    
    Attributes:
        start_idx (int): The index of the first term in the sequence
        end_idx (int): The index of the final term in the sequence
        op (2-tuple): Contains at index 0 the binary assiciative
                operation as a function that takes two ordered
                inputs and gives a single output of the same
                kind, and at index 1 the identity element of that
                binary operation.
        size (int): The number of terms of the sequence (equal to
                end_idx - start_idx + 1).
        tree (list): A list of length twice the size representing
                the segment tree. The second half of this list is
                the same as the original sequence (in the same order).
        offset (int): The difference between the position of the first
                term in the sequence in the attribute tree (i.e. the
                attribute size) and index of the first term in the
                sequence (i.e. the attribute start_idx).
                Consequently this takes the value (size - start_idx).
        
    Methods:
        (See method documentation for specific details)
        query(): Finds the interval value for a specified subset
                of the binary operations.
        update(): Sets the number associated with a given sequence
                value for one of the binary operations.
        populate(): Sets the values for the sequence starting at a
                given index.
    """
    
    std_ops = {"sum": (lambda x, y: x + y, 0, True),
               "product": (lambda x, y: x * y, 1, True),
                "max": (max, -float("inf"), True),
                "min": (min, float("inf"), True),
                "gcd": (math.gcd, 0, True),
                "union": (lambda x, y: x.union(y), set(), True),
                "bitwise_and": (lambda x, y: x & y, -1, True),
                "bitwise_or": (lambda x, y: x | y, 0, True),
                "bitwise_xor": (lambda x, y: x ^ y, 0, True),
    }
    
    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        op: Union[str, Tuple[Callable[[Any, Any], Any], Any, bool]]="sum"
    ):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.size = end_idx - start_idx + 1
        self.op = self.std_ops[op] if isinstance(op, str) else op
        if not self.op[2]:
            n = 1
            while n < self.size:
                n <<= 1
        else: n = self.size
        self.offset = n - self.start_idx
        self.tree = [self.op[1] for _ in range(n << 1)]
    
    def __getitem__(self, i: int) -> Optional[Any]:
        """
        Returns the term in the sequence corresponding to the
        index i, if there is such a term, otherwise None.
        """
        if i < self.start_idx or i > self.end_idx:
            return None
        return self.tree[i + self.offset]

    def query(self, l: int, r: int) -> Any:
        """
        For the contiguous subsequence with indices no less than l
        and no greater than r, returns the result of repeatedly
        replacing the first and second terms of the subsequence with
        the result of the binary operation of the first term with the
        second term (in that order) until a single term remains.

        If the subsequence is empty (e.g. if l > r or the range [l, r]
        is outside the range of valid indices), then returns the
        identity of the binary operation.

        Args:
            Required positional:
            l (int): The smallest possible index of the subsequence.
            r (int): The largest possible index of the subsequence.
        
        Returns:
        Value of the same type as that of the terms in the sequence,
        representing result the application of the binary operation in
        the specified subsequence with indices between l and r inclusive
        as described above if the subsequence is not empty, otherwise
        the identity element of the binary operation.
        """
        l = max(l, self.start_idx) + self.offset
        r = min(r, self.end_idx) + self.offset + 1
        
        res_l = self.op[1] # The identity of the operation
        res_r = self.op[1]
        while l < r:
            if l & 1:
                res_l = self.op[0](res_l, self.tree[l])
                l += 1
            if r & 1:
                r -= 1
                res_r = self.op[0](self.tree[r], res_r)
            l >>= 1
            r >>= 1
        return self.op[0](res_l, res_r)
    
    def update(self, i: int, val: Any) -> None:
        """
        Changes the value of the term in the sequence with index i to
        the value val, updating the rest of the segment tree to
        reflect this change.

        Args:
            Required positional:
            i (int): The index of the sequence term to be changed.
                    In order for a change to be made, this must be
                    an integer between the attributes start_idx and end_idx
                    inclusive.
            val (any): A value of the same type as that of the terms
                    in the sequence (i.e. one that can be used as
                    either the first or second argument in the binary
                    operation alongside another term) that should
                    replace the existing term at index i in the
                    sequence.
        
        Returns:
        None
        """
        if i < self.start_idx or i > self.end_idx: return
        i += self.offset
        if self.tree[i] == val: return
        self.tree[i] = val
        while i > 1:
            j1, j2 = sorted([i, i ^ 1])
            self.tree[i >> 1] = self.op[0](self.tree[j1], self.tree[j2])
            i >>= 1
        return
    
    def populate(self, i0: int, arr: List[Any]) -> None:
        """
        Overwrites part or all of the sequence in a contiguous block
        starting at index i0 with the sequence represented by the array
        arr, in such a way that the number of terms in the sequence does
        not change.
        Any terms in arr that end up corresponding to indices outside
        of the range [start_idx, end_idx] will not be used.

        Args:
            Required positional:
            i (int): The index of the first term in the sequence to be
                    replaced.
                    In order for a change to be made, this must be
                    an integer between the attributes start_idx and end_idx
                    inclusive.
            arr (list of any type): An ordered list of values of
                    the same type as that of the terms in the sequence
                    (i.e. values that can be used as either the first
                    or second argument in the binary operation alongside
                    one another or another term) that should replace the
                    terms in the sequence as a contiguous block starting
                    from the term with index i0.
                    I.e. arr[0] replaces the term with index i0, arr[1]
                    replaces the term with index i0 + 1, ... arr[k]
                    replaces the term with index i0 + k for integer k
                    between 0 and len(arr) - 1 inclusive.
        
        Returns:
        None
        """
        if i0 < self.start_idx:
            arr = arr[self.start_idx - i0:]
        i0 += self.offset
        for i, val in enumerate(arr, start=i0):
            self.tree[i] = val
        l = i0
        r = l + len(arr)
        while l > 1:
            for i in reversed(range(l, r, 2)):
                j1, j2 = sorted([i, i ^ 1])
                self.tree[i >> 1] = self.op[0](self.tree[j1], self.tree[j2])
            l >>= 1
            r = ((r + 1) >> 1)
        return

# Review- consider making this a child class of SegmentTree
class SegmentTreeWithLazyPropogation(object):
    
    """
    Creates a segment tree for a sequence consisting of elements
    of a set based around a specified associative (and possibly
    commutative) binary operation on that set (i.e. an operation
    that takes as input an ordered pair of elements of the set
    and outputs an element of the same set), with lazy propogation.

    On initialization, the sequence is entirely populated with the
    identity element of the given binary operation
    
    Initialization args:
        Required positional:
        start_idx (int): Lower bound for sequence index values
        end_idx (int): Upper bound for sequence index values
        
        Optional named:
        op (string or 3-tuple of a function on two variables, a
                numeric value and a boolean): Specifies the
                associative binary operation to be applied, its
                identity and whether the operation is commutative,
                either as a 3-tuple giving each of these in order or
                a string identifying a pre-implemented standard
                binary operation.
                The standard binary operations pre-implemented are:
                The standard binary operations implemented are:
                 "sum" (gives the interval sums for real numeric values-
                    identity 0)
                 "product" (gives the interval products for
                    real numeric values- identity 1)
                 "max" (gives the interval maxima for real numeric
                    values- identity -float("inf"))
                 "min" (gives the interval minima for real numeric
                    values- identity float("inf"))
                 "gcd" (gives the greatest common divisor over all integers
                    in the interval for integer values- identity 0)
                 "union" (gives the union of sets over the intervals
                    for sets- identity set(), the empty set)
                 "bitwise and" (gives the interval bitwise and for
                    integers- identity -1)
                 "bitwise or" (gives the interval bitwise or for
                    integers- identity 0)
                 "bitwise xor" (gives the interval bitwise excluisve
                    or for integers- identity 0)
            Default: "sum", or equivalently:
                    {"sum": (lambda x, y: x + y, 0)}
    
    Attributes:
        start_idx (int): The index of the first term in the sequence
        end_idx (int): The index of the final term in the sequence
        op (2-tuple): Contains at index 0 the binary assiciative
                operation as a function that takes two ordered
                inputs and gives a single output of the same
                kind, and at index 1 the identity element of that
                binary operation.
        size (int): The number of terms of the sequence (equal to
                end_idx - start_idx + 1).
        tree (list): A list of length twice the size representing
                the segment tree. The second half of this list is
                the same as the original sequence (in the same order).
        offset (int): The difference between the position of the first
                term in the sequence in the attribute tree (i.e. the
                attribute size) and index of the first term in the
                sequence (i.e. the attribute start_idx).
                Consequently this takes the value (size - start_idx).
        
    Methods:
        (See method documentation for specific details)
        query(): Finds the interval value for a specified subset
                of the binary operations.
        update(): Sets the number associated with a given sequence
                value for one of the binary operations.
        populate(): Sets the values for the sequence starting at a
                given index.
    """
    
    std_ops = {"sum": (lambda x, y: x + y, 0, True),
               "product": (lambda x, y: x * y, 1, True),
                "max": (max, -float("inf"), True),
                "min": (min, float("inf"), True),
                "gcd": (math.gcd, 0, True),
                "union": (lambda x, y: x.union(y), set(), True),
                "bitwise_and": (lambda x, y: x & y, -1, True),
                "bitwise_or": (lambda x, y: x | y, 0, True),
                "bitwise_xor": (lambda x, y: x ^ y, 0, True),
    }
    
    def __init__(
        self,
        start_idx: int,
        end_idx: int,
        op: Union[str, Tuple[Callable[[Any, Any], Any]]]="sum",
        range_update_func: Optional[Callable[[Any, Any, int], Any]]=None,
    ):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.size = end_idx - start_idx + 1
        self.op = self.std_ops[op] if isinstance(op, str) else op
        if not self.op[2]:
            n = 1
            while n < self.size:
                n <<= 1
        else: n = self.size
        self.offset = n - self.start_idx
        self.tree = [self.op[1] for _ in range(n << 1)]
        self.lazy = [self.op[1] for _ in range(n << 1)]
        self.range_update_func = (
            lambda val, delta, range_size: self.op[0](val, self._scalarMultiple(delta, range_size))
        ) if range_update_func is None else range_update_func
    
    def _scalarMultiple(self, val: Any, mult: int) -> Any:
        # Note- mult must be a non-negative integer
        if val == self.op[1]: return self.op[1]
        res = self.op[1]
        curr = val
        while mult:
            if mult & 1:
                res = self.op[0](res, curr)
            mult >>= 1
            curr = self.op[0](curr, curr)
        return res
    
    def _calculateRepresentedRangeSize(self, i: int) -> int:
        if not i: return self.size
        l, r = i, i
        res = 0
        while l < self.size:
            if r >= self.size: res += r - self.size + 1
            #print(l)
            l <<= 1
            r = (r << 1) + 1
        return res + min(r, (self.size << 1) - 1) - l + 1
    
    def _passDownLazy(self, idx: int) -> None:
        # Changes the current tree node value based on lazy and
        # propogates the lazy value to the child nodes (if any)
        if self.lazy[idx] == self.op[1]: return
        ci1 = idx << 1
        if ci1 < len(self.lazy):
            self.lazy[ci1] = self.op[0](self.lazy[ci1], self.lazy[idx])
            if ci1 + 1 < len(self.lazy):
                self.lazy[ci1 + 1] = self.op[0](self.lazy[ci1 + 1], self.lazy[idx])

        self.tree[idx] = self.range_update_func(self.tree[idx], self.lazy[idx], self._calculateRepresentedRangeSize(idx))
        self.lazy[idx] = self.op[1]
        return

    def _resolveLazyNodes(self, node_inds: List[int]) -> None:
        
        node_ss = SortedSet(node_inds)
        idx = 0
        while idx < len(node_ss):
            i = node_ss[~idx]
            node_ss.add(i >> 1)
            idx += 1
        for i in node_ss:
            ci1 = i << 1
            if ci1 < len(self.lazy):
                self.lazy[ci1] = self.op[0](self.lazy[ci1], self.lazy[i])
                if ci1 + 1 < len(self.lazy):
                    self.lazy[ci1 + 1] = self.op[0](self.lazy[ci1 + 1], self.lazy[i])
            self.tree[i] = self.range_update_func(self.tree[i], self.lazy[i], self._calculateRepresentedRangeSize(i))
            self.lazy[i] = self.op[1]
        return
    
    def _updateNodeFromChildren(self, i: int) -> None:
        #print(f"Using _updateNodeFromChildren() with i = {i}")
        if i >= self.size: return
        self._passDownLazy(i)

        i2 = i << 1
        self.tree[i] = self.range_update_func(self.tree[i2], self.lazy[i2], self._calculateRepresentedRangeSize(i2))
        if i2 + 1 < len(self.tree):
            self.tree[i] = self.op[0](self.tree[i],
                                      self.range_update_func(self.tree[i2 + 1], self.lazy[i2 + 1], self._calculateRepresentedRangeSize(i2 + 1)))
        self.lazy[i] = self.op[1]
        return

    def modifyRange(
        self,
        update_start_idx: int,
        update_end_idx: int,
        delta: Any,
    ) -> None:
        l = max(update_start_idx, self.start_idx) + self.offset
        r = min(update_end_idx, self.end_idx) + self.offset + 1
        res = self.op[1] # The identity of the operation
        pass_up_set = set()
        while l < r:
            if r & 1:
                r -= 1
                self.lazy[r] = self.op[0](self.lazy[r], delta)
                if r > 1:
                    pass_up_set.add(r >> 1)

            if l & 1:
                self.lazy[l] = self.op[0](self.lazy[l], delta)
                if l > 1:
                    pass_up_set.add(l >> 1)
                l += 1
            pass_up_set2 = set()
            for i in pass_up_set:
                self._updateNodeFromChildren(i)
                if i > 1:
                    pass_up_set2.add(i >> 1)
            pass_up_set = pass_up_set2
            
            l >>= 1
            r >>= 1
        while pass_up_set:
            pass_up_set2 = set()
            for i in pass_up_set:
                self._updateNodeFromChildren(i)
                if i > 1:
                    pass_up_set2.add(i >> 1)
            pass_up_set = pass_up_set2
        return res
    
    def update(self, i: int, val: Any) -> None:
        """
        Changes the value of the term in the sequence with index i to
        the value val, updating the rest of the segment tree to
        reflect this change.

        Args:
            Required positional:
            i (int): The index of the sequence term to be changed.
                    In order for a change to be made, this must be
                    an integer between the attributes start_idx and end_idx
                    inclusive.
            val (any): A value of the same type as that of the terms
                    in the sequence (i.e. one that can be used as
                    either the first or second argument in the binary
                    operation alongside another term) that should
                    replace the existing term at index i in the
                    sequence.
        
        Returns:
        None
        """
        
        if i < self.start_idx or i > self.end_idx: return
        i += self.offset
        # Resolve any pending lazy updates in the affected nodes
        self._resolveLazyNodes([i])
        if self.tree[i] == val: return
        self.tree[i] = val
        while i > 1:
            j1, j2 = sorted([i, i ^ 1])
            self.tree[i >> 1] = self.op[0](self.tree[j1], self.tree[j2])
            i >>= 1
        return

    def query(self, l: int, r: int) -> Any:
        """
        For the contiguous subsequence with indices no less than l
        and no greater than r, returns the result of repeatedly
        replacing the first and second terms of the subsequence with
        the result of the binary operation of the first term with the
        second term (in that order) until a single term remains.

        If the subsequence is empty (e.g. if l > r or the range [l, r]
        is outside the range of valid indices), then returns the
        identity of the binary operation.

        Args:
            Required positional:
            l (int): The smallest possible index of the subsequence.
            r (int): The largest possible index of the subsequence.
        
        Returns:
        Value of the same type as that of the terms in the sequence,
        representing result the application of the binary operation in
        the specified subsequence with indices between l and r inclusive
        as described above if the subsequence is not empty, otherwise
        the identity element of the binary operation.
        """
        l = max(l, self.start_idx) + self.offset
        r = min(r, self.end_idx) + self.offset + 1

        l_nodes = []
        r_nodes = []
        while l < r:
            if l & 1:
                l_nodes.append(l)
                l += 1
            if r & 1:
                r -= 1
                r_nodes.append(r)
            l >>= 1
            r >>= 1
        self._resolveLazyNodes(l_nodes + r_nodes)
        res = self.op[1]
        for idx in l_nodes: res = self.op[0](res, self.tree[idx])
        for idx in reversed(r_nodes): res = self.op[0](res, self.tree[idx])

        return res
    
    def __getitem__(self, i: int) -> Optional[Any]:
        """
        Returns the term in the sequence corresponding to the
        index i, if there is such a term, otherwise None.
        """
        if i < self.start_idx or i > self.end_idx:
            return None
        return self.query(i, i)

    def populate(self, i0: int, arr: List[Any]) -> None:
        """
        Overwrites part or all of the sequence in a contiguous block
        starting at index i0 with the sequence represented by the array
        arr, in such a way that the number of terms in the sequence does
        not change.
        Any terms in arr that end up corresponding to indices outside
        of the range [start_idx, end_idx] will not be used.

        Args:
            Required positional:
            i (int): The index of the first term in the sequence to be
                    replaced.
                    In order for a change to be made, this must be
                    an integer between the attributes start_idx and end_idx
                    inclusive.
            arr (list of any type): An ordered list of values of
                    the same type as that of the terms in the sequence
                    (i.e. values that can be used as either the first
                    or second argument in the binary operation alongside
                    one another or another term) that should replace the
                    terms in the sequence as a contiguous block starting
                    from the term with index i0.
                    I.e. arr[0] replaces the term with index i0, arr[1]
                    replaces the term with index i0 + 1, ... arr[k]
                    replaces the term with index i0 + k for integer k
                    between 0 and len(arr) - 1 inclusive.
        
        Returns:
        None
        """
        if i0 < self.start_idx:
            arr = arr[self.start_idx - i0:]
            i0 = self.start_idx
        if not arr: return
        i0 += self.offset
        # Resolve any pending lazy updates in the affected nodes
        self._resolveLazyNodes(range(i0 + self.offset, min(i0 + len(arr), len(self.tree))))

        for i, val in enumerate(arr, start=i0):
            self.tree[i] = val
        l = i0
        r = l + len(arr)
        while l > 1:
            for i in reversed(range(l, r, 2)):
                i2 = i >> 1
                j1, j2 = sorted([i, i ^ 1])
                self.tree[i2] = self.op[0](self.tree[j1], self.tree[j2])
            l >>= 1
            r = ((r + 1) >> 1)
            #print(l, r)
        return

        
def lengthOfLIS(
    nums: List[int],
    k: Optional[int]=None,
) -> int:
    """
    Finds the length of the longest strictly increasing subsequence
    of the integer sequence nums for which no two successive elements
    in the subsequence differ by more than k (with no such restriction
    if k is given as None)
    
    Args:
        Required positional:
        nums (list/tuple of ints): the sequence
        k (int/None): the maximum difference between any two successive
                elements in the subsequences allowed to be considered
                (with None meaning there is no restriction on the
                difference between successive elements)
        
    Returns:
        Integer (int) giving the longest subsequence length for the
        specified restrictions. 
    
    Example:
        >>> lengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3)
        5
    
    Solution to Leetcode #2407: Longest Increasing Subsequence II
    and (for k=None) Leetcode #300: Longest Increasing Subsequence.
    
    Original problem 2407 description:

    You are given an integer array nums and an integer k.

    Find the longest subsequence of nums that meets the following
    requirements:

    The subsequence is strictly increasing and
    The difference between adjacent elements in the subsequence is at most
    k.
    Return the length of the longest subsequence that meets the
    requirements.

    A subsequence is an array that can be derived from another array by
    deleting some or no elements without changing the order of the remaining
    elements.
    
    """
    n = len(nums)
    if k is None: k = float("inf")
    if n <= 1: return n
    start_idx, end_idx = float("inf"), -float("inf")
    for num in nums:
        start_idx = min(start_idx, num)
        end_idx = max(end_idx, num)
    n_sum = 0
    st = SegmentTree(start_idx, end_idx, op=(lambda x, y: max(x, y), 0, True))
    
    res = 1
    for num in nums:
        max_prov = st.query(max(start_idx, num - k), num) + 1
        res = max(res, max_prov)
        st.update(num, max_prov)
    return res

def minimumValueSum(
    nums: List[int],
    andValues: List[int],
) -> int:
    """
    Consider all the ways in which to partition the sequence of non-negative
    integers nums into exactly len(andValues) contiguous subsequences, such
    that the bitwise and value in andValues. Assign to each such partitioning
    a value equal to the sum of the final element in each of the contiguous
    subsequences.

    This function computes and returns the minimum of these assigned values
    (among all possible such partitionings) if any exist, otherwise it
    returns -1

    Args:
        Required positional:
        nums (list/tuple of ints): the sequence of non-negative integers
                to be partitioned
        andValues (list/tuple of ints): the values of the bitwise and
                that the contiguous subsequences of nums (in order) should
                achieve.
        
    Returns:
        Integer (int) giving the smallest sum of the final elements in
        partitions possible to achieve for partitionings that fulfill
        the given restrictions, if any. If no such partitionings exist,
        returns -1.

    Example:
        >>> minimumValueSum([2,3,5,7,7,7,5], [0,7,5])
        17

    Solution to Leetcode #3117:  Minimum Sum of Values by Dividing Array.
    
    Original problem #3117 description:

    You are given two arrays nums and andValues of length n and m respectively.

    The value of an array is equal to the last element of that array.

    You have to divide nums into m disjoint contiguous 
    subarrays
    such that for the ith subarray [li, ri], the bitwise AND of the subarray
    elements is equal to andValues[i], in other words,
    nums[li] & nums[li + 1] & ... & nums[ri] == andValues[i] for all 1 <= i <= m,
    where & represents the bitwise AND operator.

    Return the minimum possible sum of the values of the m subarrays nums is
    divided into. If it is not possible to divide nums into m subarrays
    satisfying these conditions, return -1.
    """
    n = len(nums)
    m = len(andValues)
    if m > n: return -1

    if len(andValues) == 1:
        ba = -1
        for num in nums:
            if num & andValues[0] != andValues[0]: return -1
            ba &= num
        if ba != andValues[0]: return -1
        return nums[-1]

    row = []
    ba = -1
    idx0 = -1
    for i, num in enumerate(nums):
        if num & andValues[0] != andValues[0]: break
        ba &= num
        if ba == andValues[0]:
            if not row:
                idx0 = i
            row.append(num)
        elif row:
            row.append(float("inf"))
    if not row: return -1
    while row and not isinstance(row[-1], int):
        row.pop()
    mn_seg = SegmentTree(idx0, len(row) + idx0, "min")
    mn_seg.populate(idx0, row)
    for j in range(1, m - 1):
        rng = []
        idx0 += 1
        i0 = idx0
        excl_starts = {}
        row = []
        for i in range(idx0, n):
            av = andValues[j]
            if nums[i] & av != av:
                i0 = i + 1
                excl_starts = {}
                if row: row.append(float("inf"))
                continue
            num = nums[i]
            excl_starts2 = {}
            excl_starts_mn = i + 1
            b_i = 0
            while num:
                if num & 1 and not av & 1:
                    excl_starts2[b_i] = excl_starts.get(b_i, i)
                    excl_starts_mn = min(excl_starts_mn, excl_starts2[b_i])
                num >>= 1
                av >>= 1
                b_i += 1
            excl_starts = excl_starts2
            if excl_starts_mn <= i0 and row:
                row.append(float("inf"))
                continue
            val = mn_seg.query(i0 - 1, excl_starts_mn - 2)
            if not isinstance(val, int):
                if row: row.append(float("inf"))
                continue
            if not row:
                idx0 = i
            row.append(mn_seg.query(i0 - 1, excl_starts_mn - 2) + nums[i])
        if not row: return -1
        while row and not isinstance(row[-1], int):
            row.pop()
        mn_seg = SegmentTree(idx0, len(row) + idx0, "min")
        mn_seg.populate(idx0, row)
        
    res = float("inf")
    ba = -1
    for i in reversed(range(idx0 + 1, n)):
        ba &= nums[i]
        if ba & andValues[-1] != andValues[-1]: break
        elif ba != andValues[-1]: continue
        if i <= idx0 + len(row):
            res = min(res, row[i - 1 - idx0] + nums[-1])
    return res if isinstance(res, int) else -1

def maximumSumSubsequence(
    nums: List[int],
    queries: List[List[int]],
) -> int:
    """

    Examples:
        >>> maximumSumSubsequence([3,5,9], [[1,-2],[0,-3]])
        21

        >>> maximumSumSubsequence([4,0,-1,-2,3,1,-1], [[3,1],[0,-2],[1,-1],[0,-2],[5,4],[6,-3],[6,-2],[2,-1]])
        36
    
    Illustrates the use of segment tree with non-commutative binary
    operations.

    Solution to Leetcode #3165: Maximum Sum of Subsequence With Non-adjacent
    Elements

    Original problem #3165 description:

    You are given an array nums consisting of integers. You are also given a
    2D array queries, where queries[i] = [posi, xi].

    For query i, we first set nums[posi] equal to xi, then we calculate the
    answer to query i which is the maximum sum of a subsequence of nums where
    no two adjacent elements are selected.

    Return the sum of the answers to all queries.

    Since the final answer may be very large, return it modulo 10^9 + 7.

    A subsequence is an array that can be derived from another array by
    deleting some or no elements without changing the order of the remaining
    elements.

    
    """
    # index 0: neither left-most nor right-most subarray element selected
    # index 1: left-most but not right-most subarray element selected
    # index 2: right-most but not left-most subarray element selected
    # index 3: both left-most and right-most subarray element selected

    md = 10 ** 9 + 7

    n = len(nums)

    def combineFunction(vals1: Tuple[int, int, int, int], vals2: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        res = (
            max(vals1[0] + vals2[1], vals1[2] + vals2[0], vals1[0] + vals2[0], vals1[0], vals1[2], vals2[0], vals2[1]),
            max(vals1[1] + vals2[1], vals1[3] + vals2[0], vals1[1] + vals2[0], vals1[1], vals1[3]),
            max(vals1[0] + vals2[3], vals1[2] + vals2[2], vals1[0] + vals2[2], vals2[2], vals2[3]),
            max(vals1[1] + vals2[3], vals1[3] + vals2[2], vals1[1] + vals2[2]),
        )
        #print(vals1, vals2, res)
        return res

    st = SegmentTree(0, n - 1, op=(combineFunction, (-float("inf"), -float("inf"), -float("inf"), -float("inf")), False))
    arr = [(-float("inf"), -float("inf"), -float("inf"), num) for num in nums]
    st.populate(0, arr)
    #print(st.tree)
    res = 0
    for q in queries:
        st.update(q[0], (-float("inf"), -float("inf"), -float("inf"), q[1]))
        tup = st.query(0, n - 1)
        ans = max(tup)
        ans = max(ans, 0)
        res = (res + ans) % md
    return res

def maxSubarraySum(nums: List[int]) -> int:
    """
    
    Examples:
        >>> maxSubarraySum([-3,2,-2,-1,3,-2,3])
        7

        >>> maxSubarraySum([1,2,3,4])
        10

    Illustrates the use of segment tree with non-commutative binary
    operations.

    Solution to Leetcode #3410: Maximumize Subarray Sum After Removing
    All Occurrences of One Element

    Original problem #3410 description:

    You are given an integer array nums.

    You can do the following operation on the array at most once:

    - Choose any integer x such that nums remains non-empty on removing
      all occurrences of x.
    - Remove all occurrences of x from the array.
    
    Return the maximum subarray sum across all possible resulting
    arrays.

    A subarray is a contiguous non-empty sequence of elements within an
    array.
    """
    # index 0: maximum subarray sum
    # index 1: maximum subarray sum including the left-most subarray element
    # index 2: maximum subarray sum including the right-most subarray element
    # index 3: maximum subarray sum including both the left-most and
    #          right-most subarray elements

    n = len(nums)

    mx = max(nums)
    mn = min(nums)
    if mx <= 0: return mx
    elif mn >= 0: return sum(nums)

    def combineSubarraySums(tup1: Tuple[int, int, int, int], tup2: Tuple[int, int, int, int]):
        res1 = max(tup1[0], tup2[0], tup1[2] + tup2[1])
        res2 = max(tup1[1], tup1[3] + tup2[1])
        res3 = max(tup2[2], tup1[2] + tup2[3])
        res4 = tup1[3] + tup2[3]
        return (res1, res2, res3, res4)

    op = (combineSubarraySums, (-float("inf"), -float("inf"), -float("inf"), 0), False)
    st = SegmentTree(0, n - 1, op=op)
    st.populate(0, [(x, x, x, x) for x in nums])
    #print(st.tree)

    neg_inds = {}
    for i, num in enumerate(nums):
        if num >= 0: continue
        neg_inds.setdefault(num, [])
        neg_inds[num].append(i)
    
    res = mx
    for num, inds in neg_inds.items():
        #print(f"\nnum = {num}, inds = {inds}")
        rng_sum = st.query(0, inds[0] - 1)
        #print(0, inds[0] - 1, rng_sum)
        curr = rng_sum
        for i in range(1, len(inds)):
            if inds[i - 1] + 1 == inds[i]: continue
            rng_sum = st.query(inds[i - 1] + 1, inds[i] - 1)
            #print(inds[i - 1] + 1, inds[i] - 1, rng_sum)
            curr = combineSubarraySums(curr, rng_sum)
        rng_sum = st.query(inds[-1] + 1, n - 1)
        #print(inds[-1] + 1, n - 1, rng_sum)
        curr = combineSubarraySums(curr, st.query(inds[-1] + 1, n - 1))
        #print(num, curr)
        res = max(res, curr[0])
    return res

def handleQuery(
    nums1: List[int],
    nums2: List[int],
    queries: List[List[int]],
) -> List[int]:
    """
    

    Solution to Leetcode #2569:  Handling Sum Queries After Update.
    
    Original problem #2569 description:

    You are given two 0-indexed arrays nums1 and nums2 and a 2D array queries
    of queries. There are three types of queries:

    For a query of type 1, queries[i] = [1, l, r]. Flip the values from 0 to 1
    and from 1 to 0 in nums1 from index l to index r. Both l and r are
    0-indexed.
    For a query of type 2, queries[i] = [2, p, 0]. For every index 0 <= i < n,
    set nums2[i] = nums2[i] + nums1[i] * p.
    For a query of type 3, queries[i] = [3, 0, 0]. Find the sum of the elements
    in nums2.
    Return an array containing all the answers to the third type queries.

    Examples:
        >>> handleQuery([1,0,1], [0,0,0], [[1,1,1],[2,1,0],[3,0,0]])
        [668,758,1280]

        >>> handleQuery([1,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0],
                [4,33,4,8,19,48,21,9,23,33,36,43,47,48,18,30,38,1,47,19,21,31,19,24,3,41],
                [[1,9,19],[1,1,16],[3,0,0],[2,5,0],[3,0,0],[2,29,0],[3,0,0]])
        [[1,9,19],[1,1,16],[3,0,0],[2,5,0],[3,0,0],[2,29,0],[3,0,0]]
    """
    n = len(nums1)
    #m = 1
    #while m < n:
    #    m <<= 1
    m = n
    #print(f"m = {m}")
    st_nums1 = SegmentTreeWithLazyPropogation(0, m - 1, op="sum", range_update_func=(lambda val, delta, range_size: (range_size - val) if delta & 1 else val))
    st_nums1.populate(0, nums1)
    while queries and queries[-1][0] != 3:
        queries.pop()
    res = []
    curr = sum(nums2)
    for q in queries:
        if q[0] == 1:
            st_nums1.modifyRange(q[1], q[2], 1)
        elif q[0] == 2:
            curr += st_nums1.query(0, n) * q[1]
        else: res.append(curr)
    return res

if __name__ == "__main__":
    res = lengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3)
    print("\nlengthOfLIS([4,2,1,4,3,4,5,8,15], k = 3) = "
            f"{res}")
    
    res = minimumValueSum([2,3,5,7,7,7,5], [0,7,5])
    print("\nminimumValueSum([2,3,5,7,7,7,5], [0,7,5]) = "
            f"{res}")
    
    res = maximumSumSubsequence([3,5,9], [[1,-2],[0,-3]])
    print("\nmaximumSumSubsequence([3,5,9], [[1,-2],[0,-3]]) = "
            f"{res}")

    res = maximumSumSubsequence(
        [4,0,-1,-2,3,1,-1],
        [[3,1],[0,-2],[1,-1],[0,-2],[5,4],[6,-3],[6,-2],[2,-1]]
    )
    print("\nmaximumSumSubsequence([4,0,-1,-2,3,1,-1], [[3,1],[0,-2],[1,-1],[0,-2],[5,4],[6,-3],[6,-2],[2,-1]]) = "
            f"{res}")
    
    res = maxSubarraySum([-3,2,-2,-1,3,-2,3])
    print(f"\nmaxSubarraySum([-3,2,-2,-1,3,-2,3]) = {res}")

    res = maxSubarraySum([1,2,3,4])
    print(f"\nmaxSubarraySum([1,2,3,4]) = {res}")
    
    res = handleQuery([1,0,1], [0,0,0], [[1,1,1],[2,1,0],[3,0,0]])
    print("\nhandleQuery([1,0,1], [0,0,0], [[1,1,1],[2,1,0],[3,0,0]]) = "
            f"{res}")
    
    res = handleQuery([1,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0],
                [4,33,4,8,19,48,21,9,23,33,36,43,47,48,18,30,38,1,47,19,21,31,19,24,3,41],
                [[1,9,19],[1,1,16],[3,0,0],[2,5,0],[3,0,0],[2,29,0],[3,0,0]])
    print("\nhandleQuery([1,0,0,0,0,0,1,0,1,1,0,0,0,1,0,1,1,0,0,0,1,0,1,1,1,0], "
                "[4,33,4,8,19,48,21,9,23,33,36,43,47,48,18,30,38,1,47,19,21,31,19,24,3,41], "
                f"[[1,9,19],[1,1,16],[3,0,0],[2,5,0],[3,0,0],[2,29,0],[3,0,0]]) = {res}")
                
