#!/usr/bin/env python

from typing import List

import random

from sortedcontainers import SortedList


def uniformRandomDistinctIntegers(
    n: int,
    mn: int,
    mx: int
) -> List[int]:
    """
    Takes a uniform random sample of n distinct integers between mn and
    mx inclusive.

    Time complexity O(n * (log(n)) ** 2)

    Args:
        Required positional:
        n (int): The number of distinct integers to select in the sample.
                This must not exceed (mx - mn + 1)
        mn (int): The smallest an element of the sample should be
        mx (int): The largest an element of the sample should be

    Returns:
    List of n distinct integers between mn and mx inclusive representing
    a uniform random sample. The integers are sorted in strictly increasing
    order.

    Example:
        >>> uniformRandomDistinctIntegers(3, 5, 20)
        [8, 11, 13]

        Note that the returned values will by design vary with each
        execution, but for this input will always be exactly 3 distinct
        integers in strictly increasing order, each between 5 and 20
        inclusive.
    """
    sz = mx - mn + 1
    if sz < n:
        raise ValueError(f"Fewer than {n} integers between {mn} and {mx} inclusive")
    elif not n: return []
    elif sz == n: return list(range(mn, mx + 1))
    lst = SortedList()
    
    def countLT(num: int) -> int:
        return num - lst.bisect_left(num)
    
    def insertNum() -> None:
        num0 = random.randrange(0, sz - len(lst))
        lft, rgt = num0, num0 + len(lst)
        while lft < rgt:
            mid = lft - ((lft - rgt) >> 1)
            if countLT(mid) <= num0: lft = mid
            else: rgt = mid - 1
        lst.add(lft)
        return lft
    if 2 * n <= sz:
        for _ in range(n):
            insertNum()
        return [num + mn for num in lst]
    for _ in range(sz - n):
        insertNum()
    j = 0
    res = []
    for num in range(sz):
        if num == lst[j]:
            j += 1
            if j == len(lst): break
            continue
        res.append(num + mn)
    else: num = sz
    for num in range(num + 1, sz):
        res.append(num + mn)
    return res

if __name__ == "__main__":
    res = uniformRandomDistinctIntegers(3, 5, 20)

    print(f"uniformRandomDistinctIntegers(3, 5, 20) = {res}")