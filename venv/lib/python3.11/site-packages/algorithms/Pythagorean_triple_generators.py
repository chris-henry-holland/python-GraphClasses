#! /usr/bin/env python

from typing import (
    Tuple,
    Generator,
    Optional,
)

import heapq
import itertools

from algorithms.number_theory_algorithms import isqrt, gcd

def pythagoreanTripleGeneratorByHypotenuse(
    primitive_only: bool=False,
    max_hypotenuse: Optional[int]=None,
) -> Generator[Tuple[Tuple[int, int, int], bool], None, None]:
    """
    Generator iterating over Pythagorean triples, yielding them
    in order of increasing size of the hypotenuse (i.e. the largest
    value in the Pythagorean triple).

    Args:
        Optional named:
        primitive_only (bool): Boolean specifying whether to
                iterate only over primitive Pythagorean triples
                (i.e. those whose side lengths are coprime) or
                all Pythagorean triples, with True specifying
                only primitive Pythagorean triples are to be
                iterated over.
            Default: False
        max_hypotenuse (None or int): If given, specifies the
                largest possible value of the hypotenuse of
                any Pythagorean triple yielded.
                If this is not given or given as None, the
                iterator will not self-terminate, so any loop
                based around this iterator must contain a
                mechanism to break the loop (e.g. break or
                return) to avoid an infinite loop.
            Default: None
    
    Yields:
    2-tuple whose index 0 contains a 3-tuple of integers
    specifying the corresponding Pythagorean triple, with
    the 3 items ordered in increasing size (so the hypotenuse
    is last) and whose index 1 contains a boolean denoting
    whether this Pythagorean triple is primitive (with True
    indicating that it is primitive).
    The triples are yielded in order of increasing size of
    hypotenuse, with triples with the same hypotenuse yielded
    in increasing order of their next longest side.
    """
    heap = []
    if max_hypotenuse is None: max_hypotenuse = float("inf")
    for m in itertools.count(1):
        m_odd = m & 1
        n_mn = 1 + m_odd
        m_sq = m ** 2
        min_hyp = m_sq + n_mn ** 2
        while heap and heap[0][0][0] < min_hyp:
            ans = heapq.heappop(heap) if primitive_only or heap[0][0][0] + heap[0][1][0] > max_hypotenuse else heapq.heappushpop(heap, (tuple(x + y for x, y in zip(*heap[0][:2])), heap[0][1], False))
            yield (tuple(ans[0][::-1]), ans[2])
        if min_hyp > max_hypotenuse: break
        n_mx = min(m - 1, isqrt(max_hypotenuse - m_sq)) if max_hypotenuse != float("inf") else m - 1
        # Note that since m and n are coprime and not both can be odd,
        # m and n must have different parity (as if they were both
        # even then they would not be coprime)
        for n in range(n_mn, n_mx+ 1, 2):
            if gcd(m, n) != 1: continue
            a, b, c = m_sq - n ** 2, 2 * m * n, m_sq + n ** 2
            if b < a: a, b = b, a
            heapq.heappush(heap, ((c, b, a), (c, b, a), True))
    return

def pythagoreanTripleGeneratorByPerimeter(
    primitive_only: bool=False,
    max_perimeter: Optional[int]=None,
) -> Generator[Tuple[Tuple[int, int, int], int, bool], None, None]:
    """
    Generator iterating over Pythagorean triples, yielding them
    in order of increasing size of the perimeter (i.e. the sum
    over the three values in the Pythagorean triple).

    Args:
        Optional named:
        primitive_only (bool): Boolean specifying whether to
                iterate only over primitive Pythagorean triples
                (i.e. those whose side lengths are coprime) or
                all Pythagorean triples, with True specifying
                only primitive Pythagorean triples are to be
                iterated over.
            Default: False
        max_perimeter (None or int): If given, specifies the
                largest possible value of the perimeter of
                any Pythagorean triple yielded.
                If this is not given or given as None, the
                iterator will not self-terminate, so any loop
                based around this iterator must contain a
                mechanism to break the loop (e.g. break or
                return) to avoid an infinite loop.
            Default: None
    
    Yields:
    3-tuple whose index 0 contains a 3-tuple of integers
    specifying the corresponding Pythagorean triple, with
    the 3 items ordered in increasing size (so the hypotenuse
    is last), whose index 1 contains an integer giving the
    perimeter of this Pythagorean triple and whose index 2
    contains a boolean denoting whether this Pythagorean
    triple is primitive (with True indicating that it is
    primitive).
    The triples are yielded in order of increasing size of
    perimeter, with triples with the same perimeter yielded
    in increasing order of hypotenuse, with triples with the
    same perimeter and hypotenuse yielded in increasing order
    of their next longest side.
    """
    heap = []
    if max_perimeter is None: max_perimeter = float("inf")
    for m in itertools.count(1):
        m_odd = m & 1
        n_mn = 1 + m_odd
        m_sq = m ** 2
        min_perim = m * (m + n_mn)
        while heap and heap[0][0] < min_perim:
            new_perim = heap[0][0] + sum(heap[0][2])
            ans = heapq.heappop(heap) if primitive_only or new_perim > max_perimeter else heapq.heappushpop(heap, (new_perim, tuple(x + y for x, y in zip(*heap[0][1:3])), heap[0][2], False))
            yield (tuple(ans[1][::-1]), ans[0], ans[3])
        if min_perim > max_perimeter: break
        n_mx = min(m - 1, max_perimeter // (2 * m) - m) if max_perimeter != float("inf") else m - 1
        for n in range(n_mn, n_mx + 1, 2):
            if gcd(m, n) != 1: continue
            a, b, c = m_sq - n ** 2, 2 * m * n, m_sq + n ** 2
            if b < a: a, b = b, a
            heapq.heappush(heap, ((a + b + c), (c, b, a), (c, b, a), True))
    return