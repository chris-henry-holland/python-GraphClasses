#! /usr/bin/env python

from typing import (
    Tuple,
    Generator,
)

from collections import deque

def generalisedLaggedFibonacciGenerator(
    poly_coeffs: Tuple[int]=(100003, -200003, 0, 300007),
    lags: Tuple[int]=(24, 55),
    min_val: int=0,
    max_val: int=10 ** 6 - 1
) -> Generator[int, None, None]:
    """
    Generator iterating over the terms in a generalisation
    of a lagged Fibonacci generator sequence for given for a
    given initial polynomial and given lag lengths within n a
    given range.

    The generalisation of the lagged Fibonacci generator
    sequence for the given tuple of integers poly_coeffs, tuple
    of strictly positive integers lags, and the integers min_val
    and max_val is the sequence such that for integer i >= 1,
    the i:th term in the sequence is:
        t_i = (sum j from 0 to len(poly_coeffs) - 1) (poly_coeffs[j] * i ** j) % md + min_val
                for i <= max(lags)
              ((sum j fro 0 to len(lags) - 1) (t_(i - lags[i]))) % md + min_val
                otherwise
    where md is one greater than the difference between
    min_value and max_value and % signifies modular division
    (i.e. the remainder of the integer preceding that symbol
    by the integer succeeding it). This sequence contains integer
    values between min_value and max_value inclusive.

    The terms where i <= max(lags) are referred as the polynomial
    terms and the terms where i > max(lags) are referred to as the
    recursive terms.

    In the case that lags is length 2 with those two elements
    distinct, this is a traditional lagged Fibonacci generator
    sequence.

    For well chosen values of poly_coeffs and lags for given
    min_value and max_value, this can potentially be used as a
    generator of pseudo-random integers between min-value and
    max_value inclusive.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        poly_coeffs (tuple of ints): Tuple of integers giving
                the coefficients of the polynomial used to
                generate the polynomial terms.
            Default: (100003, -200003, 0, 300007)
        lags (tuple of ints): Strictly positive integers,
                which when generating the recursive terms,
                indicates how many steps back in the sequence
                the previous terms summed should each be
                from the position of the term being generated.
                Additionally, the maximum value determines
                at which term the transition from the polynomial
                terms to the recursive terms will occur.
            Default: (24, 55)
        min_value (int): Integer giving the smallest value
                possible for terms in the sequence.
        max_value (int): Integer giving the largest value
                possible for terms in the sequence. Must
                be no smaller than min_value.
    
    Yields:
    Integer (int) between min_value and max_value inclusive,
    with the i:th term yielded (for strictly positive integer
    i) representing the i:th term in the generalisation of
    the lagged Fibonacci generator defined above for the
    given parameters.
    """
    qu = deque()
    md = max_val - min_val + 1
    #print(md)
    lags = sorted(lags)
    max_lag = lags[-1]
    for k in range(1, max_lag + 1):
        num = (sum(c * k ** i for i, c in enumerate(poly_coeffs)) % md) + min_val
        #num = ((100003 - 200003 * k + 300007 * k ** 3) % md) - 5 * 10 ** 5
        #print(num)
        qu.append(num)
        yield num
    #cnt = 0
    while True:
        num = qu.popleft()
        for i in range(len(lags) - 1):
            num += qu[-lags[i]]
        num = (num % md) + min_val
        #if cnt < 10:
        #    print(num)
        #cnt += 1
        #num = ((qu[-24] + qu.popleft() + 10 ** 6) % md) - 5 * 10 ** 5
        qu.append(num)
        yield num
    return

def blumBlumShubPseudoRandomGenerator(
    s_0: int=290797,
    s_mod: int=50515093,
    t_min: int=0,
    t_max: int=499
) -> Generator[int, None, None]:
    """
    Generator iterating over the terms in a Blum Blum Shub sequence
    for a given seed value, modulus and value range.

    To find the terms in the Blum Blum Shub sequence for a given seed
    value, modulus and value range the following sequence s_i is used,
    calculated from the recurrence relation:
        s_(i + 1) = s_i * s_i (mod s_mod)
    where s_0 is the seed value and s_mod is the modulus for this Blum
    Blum Shub sequence.
    For strictly positive integers i, the i:th term in the Blum Blum Shub
    sequence, t_i is given by:
        t_n = s_i + t_min (mod t_max - t_min + 1)
    where t_min and t_max are the minimum and maximum values respectively
    of the value range.

    For well chosen values of s_0 and s_mod, this can potentially be used
    as a generator of pseudo-random integers between t_min and t_max
    inclusive.

    Note that the generator never terminates and thus any
    iterator over this generator must include provision to
    terminate (e.g. a break or return statement), otherwise
    it would result in an infinite loop.

    Args:
        Optional named:
        s_0 (int): Integer giving the seed value for this Blum Blum Shub
                sequence.
            Default: 290797
        s_mod (int): Integer strictly greater than 1 giving the modulus
                for this Blum Blum Shub sequence.
            Default: 50515093
        t_min (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence.
            Default: 0
        t_max (int): Integer giving the smallest value possible
                for terms in the Blum Blum Shub sequence. Must
                be no smaller than t_min.
            Default: 499
    
    Yields:
    Integer (int) between min_value and max_value inclusive,
    with the i:th term yielded (for strictly positive integer
    i) representing the i:th term in the Blum Blum Shub sequence
    for the given seed value (s_0), modulus (s_mod) and value range
    ([t_min, t_max]).
    """
    s = s_0
    t_mod = t_max - t_min + 1
    while True:
        s = s ** 2 % s_mod
        yield (s % t_mod) + t_min
    return