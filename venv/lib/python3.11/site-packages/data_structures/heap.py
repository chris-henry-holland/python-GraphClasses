#!/usr/bin/env python

from typing import (
    List,
    Optional,
)

def maxHeapify(heap: List[int]) -> None:
    """
    Converts list heap into a max heap.
    
    Args:
        Required positional:
        heap (list of ints): the list to be converted into a max heap
       
    Returns:
        List of integers (int) organised as a heap. For an element
        at given index i of the array, its descendants are at
        indices 2 * i + 1 and 2 * i + 2 (if one or both of these
        indices is beyond the length of the array, the element at
        index i has just one or no descendants respectively).
    """
    n = len(heap)
    for i in reversed(range(n)):
        siftDown(heap, i, length=n)
    return
        
def siftDown(heap: List[int], i: int, length: int) -> None:
    while True:
        num = heap[i]
        i2 = i << 1
        lt, rt = i2 + 1, i2 + 2
        if rt < length:
            mx = max((heap[lt], lt), (heap[rt], rt))
        elif lt < length: mx = (heap[lt], lt)
        else: break
        if num >= mx[0]: break
        heap[i], heap[mx[1]] = heap[mx[1]], heap[i]
        i = mx[1]
    return

def heapSort(
    nums: List[int],
    in_place: bool=True
) -> Optional[List[int]]:
    n = len(nums)
    if not in_place:
        nums = list(nums)
    maxHeapify(nums)
    for i in reversed(range(1, n)):
        nums[0], nums[i] = nums[i], nums[0]
        siftDown(nums, 0, i)
    return None if in_place else nums
