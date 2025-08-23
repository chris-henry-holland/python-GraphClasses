#!/usr/bin/env python

from typing import (
    Tuple,
    Callable,
    Any,
)

class FenwickTree(object):
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
    def __init__(
        self,
        n: int,
        op: Tuple[Callable[[Any, Any], Any], Any]
    ):
        self.n = n
        self.arr = [op[1]] * (n + 1)
        self.op = op

    def query(self, i: int) -> Any:
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
    
    def update(self, i: int, v: Any) -> None:
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
