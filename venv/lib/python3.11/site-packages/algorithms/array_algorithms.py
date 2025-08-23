#!/usr/bin/env python

from typing import List

import heapq


def BoyerMoore(nums: List[int], n_maj: int=2, check: bool=True) -> List[int]:
    """
    Boyer-Moore voting algorithm to find which elements of
    the array nums occur more than floor(len(nums) / n_maj)
    times (referred to as majority elements) using O(1) additional space.
    If check is False, returns all candidates identified by the
    Boyer-Moore algorithm as potential majority elements without
    verifying that they truly are. Use of the option check = False
    has very few practical uses, the primary one being where 
    n_maj = 2 and it is already know there is a majority element
    (as in the problem statement of Leetcode #169).
    Solution to Leetcode #229 Majority Element II (with n_maj = 3 and
    check = True) and Leetcode #169 Majority Element (with n_maj = 2
    and check = False)
    """
    counts = [0] * (n_maj - 1)
    candidates = [None] * (n_maj - 1)
    candidate_dict = {}
    avail_heap = list(range(n_maj - 1))
    heapq.heapify(avail_heap)
    for num in nums:
        if num in candidate_dict.keys():
            counts[candidate_dict[num]] += 1
        elif avail_heap:
            j = heapq.heappop(avail_heap)
            candidates[j] = num
            candidate_dict[num] = j
            counts[j] = 1
        else:
            for j in range(n_maj - 1):
                counts[j] -= 1
                if not counts[j]:
                    heapq.heappush(avail_heap, j)
                    candidate_dict.pop(candidates[j])
    while candidates and candidates[-1] is None:
        candidates.pop()
    if not candidates or not check: return candidates
    
    # Check candidates
    thresh = (len(nums) // n_maj) + 1
    counts = [0] * len(candidates)
    res = []
    for num in nums:
        if num in candidate_dict.keys():
            i = candidate_dict[num]
            counts[i] += 1
            if counts[i] == thresh: res.append(num)
    
    return res
