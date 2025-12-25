#! /usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Set,
    Union,
    Generator,
    Callable,
    Optional,
    Any,
    Hashable,
    Iterable,
)

import heapq
import itertools
import math

from algorithms.number_theory_algorithms import isqrt

def nthConvergent(
    n: int,
    cf_func: Callable[[int], int]
) -> Tuple[int]:
    """
    Finds the nth convergent of a given continued fraction
    representation of a non-negative number, with terms as
    given in cf_func()
    
    Args:
        Required positional:
        n (int): Strictly positive integer giving the
                convergent of the continued fraction is to
                be calculated
        cf_func (callable): A function accepting a single int
                as an argument. For given i, this should
                return the ith value in the 0-indexed continued
                fraction sequence. If the sequence has
                terminated before the ith index, it should
                return -1. Note that cf_func(0) must be a
                non-negative integer.
    
    Returns:
    2-tuple of ints where index 0 is the numerator and
    index 1 is the denominator
    """
    if n < 1: raise ValueError("n should be strictly positive")
    for i in reversed(range(n)):
        a = cf_func(i)
        if a != -1: break
    else: raise ValueError("The function cf_func returned -1 "
            "for 0, which is not allowed.")
    res = (a, 1)
    for i in reversed(range(i)):
        res = (res[1] + cf_func(i) * res[0], res[0])
    return res

def continuedFractionConvergentGenerator(
    continued_fraction_terms: Iterable[int]
) -> Generator[Tuple[int], None, None]:
    """
    Generates the convergents in order of a given continued fraction
    representation of a non-negative number with terms as given in
    the function cf_func() giving the terms in the continued fraction
    sequence under consideration.
    Note that if the continued fraction is infinite, this generator
    will not itself terminate, so any loop over this generator would
    in such a case need to contain a break or return statement.
    
    Args:
        Required positional:
        continued_fraction_terms (iterable): An ordered iterable
                object containing integers representing the terms
                in the continued fraction representation of the
                non-negative number in question. Note that the
                first term must be a non-negative integer
    
    Yields:
    Each convergent of the continued fraction with terms given by
    cf_func in turn.
    
    Outline of rationale:
    For continued fraction sequence, it is a known result (see
    https://pi.math.cornell.edu/~gautam/ContinuedFractions.pdf) that
    if p_i and q_i represent the numerator and denominator in lowest
    terms (i.e. gcd(p_i, q_i) = 1) of the ith convergent of a given
    continued fraction [a_0, a_1, a_2, ...] for any non-negative
    integer i:
        p_0 = a_0, q_0 = 1
        p_1 = a_1 * a_0 + 1, q_1 = a_1
        and for n >= 2:
        p_n = a_n * p_(n - 1) + p_(n - 2)
        q_n = a_n * q_(n - 1) + q_(n - 2)
    """
    it = iter(continued_fraction_terms)
    a_0 = next(it)
    curr = [(a_0, 1)]
    yield curr[0]
    a_1 = next(it)
    if a_1 == -1: return
    curr.append((a_1 * a_0 + 1, a_1))
    yield curr[-1]
    n = 2
    while True:
        a_n = next(it)
        if a_n == -1: break
        curr = [curr[1], (a_n * curr[-1][0] + curr[-2][0],\
                a_n * curr[-1][1] + curr[-2][1])]
        yield curr[-1]
        n += 1
    return

def eContinuedFractionSequenceValue(i: int) -> int:
    """
    Gives the ith index (0-indexed) value of the continued fraction
    sequence of e (Euler's number).
    
    Args:
        Required positional:
        i (int): The index of the continued fraction sequence to be
                returned
        
    Returns:
    The ith index value of the continued fraction sequence of e.
    """
    if i % 3 == 2:
        return ((i // 3) + 1) << 1
    return 1 + (i == 0)

def sqrtContinuedFractionRepresentation(
    num: int,
) -> Tuple[Union[Tuple[int], int]]:
    """
    Finds the continued fraction representation of the
    square root of num
    
    Args:
        Required positional:
        num (int): The number whose square root to be
                represented as a continued fraction
    
    Returns:
    2-tuple whose 0th index contains a tuple of ints which is
    the sequence of terms in the continued fraction
    representation up to the point where the sequence repeats
    and whose 1st index contains the index of the sequence
    the repetition goes back to (where 0-indexing is used).
    For any positive integer that is not an exact square,
    the sequence is guaranteed to repeat.
    If num is an exact square (e.g. 1, 4, 9, ...) then the
    1st index contains -1.
    """
    seen = {}
    res = []
    curr = (0, 1)
    rt = isqrt(num)
    if rt ** 2 == num: return ((rt,), -1)
    while True:
        if curr in seen.keys():
            return (tuple(res), seen[curr])
        seen[curr] = len(res)
        a = (rt + curr[0]) // curr[1]
        res.append(a)
        b = curr[0] - a * curr[1]
        curr = (-b, (num - b ** 2) // curr[1])
        prev = curr
    return ()

def sqrtContinuedFractionTermValue(i: int, num: int) -> int:
    """
    Gives the ith index (0-indexed) value of the continued fraction
    sequence of the square root of non-negative integer num.
    
    Args:
        Required positional:
        i (int): The index of the continued fraction sequence to be
                returned
        num (int): The non-negative integer whose square root the
                continued fraction sequence represents.
        
    Returns:
    The ith index value of the continued fraction sequence of the
    square root of num.
    If the sequence has terminated before the ith index (which is the
    case only for exact squares for i > 0), -1 is returned.
    
    Examples:
    >>> [sqrtContinuedFractionTermValue(i, 2) for i in range(5)]
    [1, 2, 2, 2, 2]
    >>> [sqrtContinuedFractionTermValue(i, 7) for i in range(10)]
    [2, 1, 1, 1, 4, 1, 1, 1, 4, 1]
    >>> [sqrtContinuedFractionTermValue(i, 4) for i in range(5)]
    [2, -1, -1, -1, -1]
    """
    num_cf = sqrtContinuedFractionRepresentation(num)
    if i < len(num_cf[0]): return num_cf[0][i]
    if num_cf[1] == -1:
        return -1
    j = num_cf[1]
    return num_cf[0][j + (i - j) % (len(num_cf[0]) - j)]

def sqrtContinuedFractionTermGenerator(
    num: int
) -> Generator[int, None, None]:
    """
    Generator yielding the terms of the continued fraction
    representation of the square root of num, for strictly
    positive integers.
    If num is a perfect square then yields exactly one value
    (num itself), otherwise the generator does not inherently
    terminate.
    
    Args:
        Required positional:
        num (int): The non-negative integer whose square root the
                continued fraction sequence represents.
        
    Yields:
    Integers giving the terms of the continued fraction sequence of the
    square root of num in order.
    """
    num_cf = sqrtContinuedFractionRepresentation(num)
    if num_cf[1] == -1:
        # Perfect square
        yield num_cf[0][0]
        return
    n = len(num_cf[0])
    i = 0
    while True:
        yield num_cf[0][i]
        i += 1
        if i == n: i = num_cf[1]
    return

def sqrtConvergentGenerator(
    num: int
) -> Generator[Tuple[int, int], None, None]:
    """
    Generator yielding the convergents of the square root of num,
    for strictly positive integers.
    If num is a perfect square then yields exactly one value,
    (num, 1), otherwise the generator does not inherently
    terminate.
    
    Args:
        Required positional:
        num (int): The non-negative integer whose square root the
                convergents are to be yielded
        
    Yields:
    2-tuples of non-negative integers giving the convergents of the
    square root of num in order as fractions in reduced form where
    index 0 contains the numerator and index 1 the demoninator of
    the corresponding convergent.
    """
    yield from continuedFractionConvergentGenerator(sqrtContinuedFractionTermGenerator(num))
    return

def sqrtBestRationalApproximation(
    denom_max: int,
    num: int
) -> Tuple[int]:
    """
    Finds the best rational approximation of the square root of
    a given non-negative number for the given maximum denominator
    denom_max.

    The best rational approximation of a non-negative number num
    for a given maximum denominator denom_max is the unique fraction
    p / q (where p and q are integers) such that q < denom_max and
    for any integers p2, q2:
        if abs(num - p2 / q2) < abs(num - p / q)
        then q2 > denom_max.
    
    Args:
        Required positional:
        denom_max (int): Strictly positive integer giving the maximum
                value of denominator the best rational approximation
                may have when expressed as a fraction in lowest terms.
        num (int): The non-negative integer whose square root the
                best rational approximation for the maximum denominator
                denom_max is to be found.
    
    Returns:
    2-tuple of integers (ints) giving the best rational approximation
    of num as a fraction in lowest terms, where index 0 contains the
    numerator and index 1 the denominator.

    Outline of rationale:
    See documentation for continuedFractionConvergentGenerator() and
    https://shreevatsa.wordpress.com/2011/01/10/not-all-best-rational-approximations-are-the-convergents-of-the-continued-fraction/
    """
    #return bestRationalApproximation(denom_max, sqrtContinuedFractionTermGenerator(num))
    it = iter(sqrtContinuedFractionTermGenerator(num))
    a_0 = next(it)
    curr = [(a_0, 1)]
    a_1 = next(it)
    if a_1 == -1: return (a_0, 1)
    curr.append((a_1 * a_0 + 1, a_1))
    a_n = a_1
    n = 2
    for a_n in it:
        if a_n == -1: break
        nxt = (a_n * curr[-1][0] + curr[-2][0],\
                a_n * curr[-1][1] + curr[-2][1])
        if nxt[1] > denom_max:
            break
        curr = [curr[1], nxt]
        n += 1
    else:
        return curr[-1]
    mult_max = (denom_max - curr[-2][1]) // curr[-1][1]
    #print(mult_max, a_n)
    if mult_max << 1 < a_n:
        return curr[-1]
    if mult_max << 1 == a_n:
        frac1 = (mult_max * curr[-1][0] + curr[-2][0],\
                mult_max * curr[-1][1] + curr[-2][1])
        frac2 = curr[-1]
        
        if frac1[0] * frac2[1] > frac2[0] * frac1[1]:
            frac1, frac2 = frac2, frac1
        #print(frac1, frac2)
        if frac1[0] ** 2 > frac1[1] ** 2 * num:
            #print("hi2")
            return frac1
        elif frac2[0] ** 2 < frac2[1] ** 2 * num:
            #print("hi3")
            return frac2
        #print(frac1, frac2, (frac1[0] * frac2[1]) ** 2 + (frac2[0] * frac1[1]) ** 2, 2 * num * (frac1[1] * frac2[1]) ** 2)
        return frac2 if (frac1[0] * frac2[1]) ** 2 + (frac2[0] * frac1[1]) ** 2 < 2 * num * (frac1[1] * frac2[1]) ** 2 else frac1
        
    return (mult_max * curr[-1][0] + curr[-2][0],\
            mult_max * curr[-1][1] + curr[-2][1])
    

def pellFundamentalSolution(D: int) -> Tuple[int]:
    """
    Finds the solution to Pell's equation and Pell's
    negative equation (if they exist):
        x ** 2 - D * y ** 2 = 1
        and
        x ** 2 - D * y ** 2 = -1
    respectively for given strictly positive integer D and
    strictly positive integers x and y such that there does
    not exist another such solution with smaller x (the
    so-called fundamental solution).
    Uses the standard method of solving Pell's equation using
    continued fractions.
    
    Args:
        Required positional:
        D (int): Strictly positive integer that is the value
                of D in the above equation
    
    Returns:
    2-tuple, giving the fundamental solution to Pell's equation
    (index 0) and Pell's negative equation (index 1). Each
    solution is either None (if no solution for strictly
    positive x and y exists for this D) or in the form of a
    2-tuple where indices 0 and 1 are the values of x and y
    for the fundamental solution.
    If D is a square, there is no solution with both strictly
    positive integers x and y, for either Pell's equation or
    Pell's negative equation so (None, None) is returned.
    Otherwise, there is always a solution to Pell's equation,
    so index 0 gives the value None if and only if D is a
    square.
    """
    #(since (1, 0) is the only possible solution to Pell's
    #equation with square D and integer x and y, but not
    #one satisfying the requirement that x and y are strictly
    #positive, and there is no solution not even a trivial one
    #to Pell's negative equation with square D and integer x and y).
    D_cf = sqrtContinuedFractionRepresentation(D)
    if D_cf[1] == -1:
        return (None, None)#(1, 0)
    def cf_func(i: int) -> int:
        if i < len(D_cf[0]): return D_cf[0][i]
        j = D_cf[1]
        return D_cf[0][j + (i - j) % (len(D_cf[0]) - j)]
        
    res = nthConvergent(len(D_cf[0]) - 1, cf_func)
    if (len(D_cf[0]) - D_cf[1]) & 1:
        # For continued fractions with odd cycle lengths
        # the solution found is the fundamental solution for:
        #  x ** 2 + D * y ** 2 = -1
        # The following converts this into the fundamental
        # solution for:
        #  x ** 2 + D * y ** 2 = 1
        x, y = res
        res2 = ((x ** 2 + D * y ** 2), (2 * x * y))
        return (res2, res)
    return (res, None)

def pellSolutionGenerator(
    D: int,
    negative: bool=False
) -> Generator[Tuple[int], None, None]:
    """
    Generator that yields the positive integer solutions to Pell's
    equation or Pell's negative equation:
        x ** 2 - D * y ** 2 = 1 (Pell's equation)
        or x ** 2 - D * y ** 2 = - 1 (Pell's negative equation)
    for given strictly positive integer D, in order of increasing
    size of x.
    Note that these equations either has no integer solutions or
    an infinite number of integer solutions. Pell's equation has
    and infinite number of integer solutions for any non-square
    positive integer value of D, while for square values of D
    it has no integer solutions, while Pell's negative equation
    some non-square positive integer values of D give an infinite
    number of integer solutions, while the other non-square positive
    integer values of D and all square values of D give no
    integer solutions.
    Given that for many values of D the generator does not by
    itself terminate, any loop over this generator should contain a
    break or return statement.
    
    Args:
        Required positional:
        D (int): The strictly positive integer number D used in
                Pell's equation or Pell's negative equation
        
        Optional named:
        negative (bool): If True, iterates over the solutions to
                Pell's equation for the given value of D, otherwise
                iterates over the solutions to Pell's negative
                equation for the given value of D.
            Default: False
    
    Yields:
    2-tuple of ints with the 0th index containing the value of x and
    1st index containing the value of y for the current solution to
    Pell's equation or Pell's negative equation (based on the input
    argument negative given).
    These solutions are yielded in increasing size of x (which by the
    form of Pell's equation and Pell's negative equation and the
    requirement that x and y are strictly positive implies the solutions
    are also yielded in increasing size of y), and it if the generator
    terminates, there are no solutions other than those yielded, and
    for any two consecutive solutions yielded (x1, y1) and (x2, y2), for
    any integer x where x1 < x < x2 there does not exist a positive
    integer y such that (x, y) is also a solution.
    """
    f_sol_pair = pellFundamentalSolution(D=D)
    f_sol = f_sol_pair[negative]
    if f_sol is None:
        # No solution
        return
    curr = f_sol
    if not negative:
        while True:
            yield curr
            curr = (curr[0] * f_sol[0] + D * curr[1] * f_sol[1],\
                    curr[1] * f_sol[0] + curr[0] * f_sol[1])
        return
    while True:
        yield curr
        curr = (curr[0] * f_sol[0] ** 2 + D * curr[0] * f_sol[1] ** 2\
                + 2 * D * curr[1] * f_sol[1] * f_sol[0],\
                curr[1] * f_sol[0] ** 2 + D * curr[1] * f_sol[1] ** 2\
                + 2 * curr[0] * f_sol[1] * f_sol[0])
    return

# Review- see if there exists a more efficient method, as for larger n
# this becomes very slow
# Additionally, consider using congruences to identify cases with no
# solution.
def generalisedPellFundamentalSolutions(
    D: int,
    n: int,
    pell_basic_sol: Optional[Tuple[int]]=None
) -> List[Tuple[int]]:
    """
    Finds a fundamental solution set to the generalised Pell's equation
    (if they exist):
        x ** 2 - D * y ** 2 = n
    for given strictly positive integer D, given non-zero integer n
    corresponding to the solution to Pell's equation with the same D
    value pell_basic_sol if this is given as not None or the
    fundamental solution to that Pell's equation otherwise (see
    documentation for pellFundamentalSolution() for definition of the
    fundamental solution for Pell's equation for a given value of D).
    A set S of ordered pairs of integers is fundamental solution set
    of the generalised Pell's equation for given D and n corresponding
    to a solution (x0, y0) to Pell's equation with x0 and y0 strictly
    positive integers for the same D value if and only if:
     - Any integer solution of the generalised Pell's equation
       in question (x, y) can be expressed as:
        x + sqrt(D) * y = (x1 + sqrt(D) * y1) * (x0 + sqrt(D) * y0) ** k
       where k is an integer and (x1, y1) an element of S
     - For any two distinct elements of S (x1, y1) and (x2, y2) there
       does not exist an integer k such that:
        x1 + sqrt(D) * y1 = (x2 + sqrt(D) * y2) * (x0 + sqrt(D) * y0) ** k
    Note that k may be negative and from the definition of Pell's
    equation, for any integer k:
        (x0 + sqrt(D) * y0) ** -k = (x0 - sqrt(D) * y0) ** k
    
    Args:
        Required positional:
        D (int): Strictly positive integer that is the value of D in
                the above equation
        n (int): Non-zero integer that is the value of n in the above
                equation
        
        Optional named:
        pell_basic_sol (2-tuple of ints or None): If given as not None,
                gives the integer solution to Pell's equation with the
                same D value the fundamental solution set returned
                corresponds to. Otherwise, the solution returned will
                correspond to the fundamental solution to Pell's
                equation for that D value (see documentation for
                pellFundamentalSolution() for definition of the
                fundamental solution for Pell's equation for a given
                value of D).
                Note that the solution to Pell's equation with the same
                D value used must have strictly positive x and y values
                but given that if (x, y) is a solution, then so are
                (-x, y), (x, -y) and (-x, -y), rather than throwing an
                error if one or both components is negative, the
                absolute value for both components is used.
            Default: None
    
    Returns:
    List of 2-tuples of ints, where each 2-tuple (x1, y1) represents
    exactly two distinct solutions to the generalised Pell's equation
    for the given D and n, namely:
        x = x1 and y = y1
        x = -x1 and y = -y1
    Collectively, the solutions represented by the elements of the list
    form a fundamental solution set of the generalised Pell's equation
    for the given D and n corresponding to the solution to Pell's
    equation with the same D value pell_basic_sol if this is given as
    not None or the fundamental solution to that Pell's equation
    otherwise.
    An empty list signifies that the generalised Pell's equation for
    the given D and n has no integer solutions.
    
    Outline of rationale/proof:
    See https://kconrad.math.uconn.edu/blurbs/ugradnumthy/pelleqn2.pdf
    In this paper, Theorem 3.3 implies that for any solution (x0, y0)
    of Pell's equation for given D there exists a corresponding
    fundamental solution set for the generalised Pell's equation with
    given D and n that is a subset of the solutions (x, y) of this
    generalised Pell's equation such that:
        abs(y) <= sqrt(abs(n)) * (sqrt(u) + 1 / sqrt(u)) / (2 * sqrt(D))
    where:
        u = x0 + sqrt(D) * y0
    We therefore check all y values within these bounds using the
    fundamental solution to Pell's equation for this D to find these
    solutions, the set of which contains as a subset a fundamental
    solution set for this generalised Pell's equation.
    """
    #print(D, n)
    if n == 1:
        #print("hi1")
        res = pellFundamentalSolution(D)[0]
        return [] if res is None else [res]
    elif n == -1:
        #print("hi2")
        res = pellFundamentalSolution(D)[1]
        return [] if res is None else [res]
    if pell_basic_sol is None:
        pell_basic_sol = pellFundamentalSolution(D)[0]
        if pell_basic_sol is None: return []
    #print(f"Pell fundamental solutions = {pellFundamentalSolution(D)}")
    # Checking congruences:
    for p, sq_congs in ((3, {0, 1}), (4, {0, 1}), (5, {0, 1, 4})):
        ysqD_congs = {(x * D) % p for x in sq_congs}
        n_cong = n % p
        for xsq_cong in sq_congs:
            if (xsq_cong - n_cong) % p in ysqD_congs:
                break
        else:
            #print(f"failed {p} congruence")
            return []
    
    sqrt_D = math.sqrt(D)
    abs_val_func = lambda x: abs(x[0] + x[1] * sqrt_D)
    x0, y0 = [abs(x) for x in pell_basic_sol]
    u = abs_val_func((x0, y0))
    u_sqrt = math.sqrt(u)
    r = (-n - 1) // D
    #print(f"r = {r}")
    y_lb = isqrt(r) + 1 if n < 0 else 0
    y_ub = math.floor(math.sqrt(abs(n)) * (u_sqrt + 1 / u_sqrt) /\
            (2 * math.sqrt(D)))
    #print(f"n = {n}, D = {D}, y lower bound = {y_lb}, y upper bound = {y_ub}")
    res = []
    for y in range(y_lb, y_ub + 1):
        x_sq = n + D * y ** 2
        x = isqrt(x_sq)
        if x ** 2 == x_sq:
            res.append((x, y))
            if y: res.append((x, -y))
    if not res: return []
    # Removing solutions that can be reached by another solution
    # via a power of a solution to Pell's equation with the same
    # value of D.
    res.sort(key=abs_val_func)
    res_set = {v for v in res}
    excl_inds = set()
    abs_val_mx = abs_val_func(res[-1])
    for i1, (x1, y1) in enumerate(res):
        #print(abs_val_mx, abs_val_func((x1, y1)))
        mx_u_pow = math.floor(math.log(abs_val_mx /\
                abs_val_func((x1, y1))))
        if mx_u_pow < 1: break
        x, y = (x1 * x0 + y1 * y0 * D, x1 * y0 + y1 * x0)
        for _ in range(mx_u_pow):
            if (x, y) in res_set:
                #print("removed degenerate")
                excl_inds.add(i1)
                break
            x, y = (x * x0 + y * y0 * D, x * y0 + y * x0)
    return [x for i, x in enumerate(res) if i not in excl_inds] if\
            excl_inds else res

def generalisedPellSolutionGenerator(
    D: int,
    n: int,
    excl_trivial: bool=True
) -> Generator[Tuple[int], None, None]:
    """
    Generator that yields the solutions to the generalised Pell's
    equation:
        x ** 2 - D * y ** 2 = n
    for given strictly positive integer D and integer n, with strictly
    positive (or if excl_trivial is False, non-negative) integers x and
    y in order of increasing size of x.
    Note that these equations either has no strictly positive integer
    solutions or an infinite number of strictly positive integer
    solutions.
    Given that for many values of D the generator does not by
    itself terminate, any loop over this generator should contain a
    break or return statement.
    
    Args:
        Required positional:
        D (int): The strictly positive integer D used in the
                generalised Pell's equation.
        n (int): The integer n used in the generalised Pell's equation.
        
        Optional named:
        excl_trivial (bool): If True, excludes solutions for which
                x or y is zero. Otherwise includes such solutions.
            Default: True
    
    Yields:
    2-tuple of ints with the 0th index containing the value of x and
    1st index containing the value of y for the current solution to
    the generalised Pell's equation for the given values of D and n.
    These solutions are yielded in increasing size of x, and if the
    generator terminates, there are no solutions for strictly positive
    (or if excl_trivial is given as True non-negative) x and y other
    than those yielded, and for any two consecutive solutions yielded
    (x1, y1) and (x2, y2), for any integer x where x1 < x < x2 there
    does not exist a positive integer y such that (x, y) is also a
    solution.
    """
    if not excl_trivial:
        if n >= 0:
            n_sqrt = isqrt(n)
            if n_sqrt ** 2 == n:
                yield (n_sqrt, 0)
        elif not n % D:
            n_div_D = -n // D
            n_div_D_sqrt = isqrt(n_div_D)
            if n_div_D_sqrt ** 2 == n_div_D:
                yield (0, n_div_D_sqrt)
    if n == 1:
        yield from pellSolutionGenerator(D, negative=False)
        return
    elif n == -1:
        yield from pellSolutionGenerator(D, negative=True)
        return
    elif not n:
        # If n = 0, has solutions if and only if D is a square
        D_sqrt = isqrt(D)
        if D_sqrt ** 2 != D: return
        for y in itertools.count(1):
            yield (y * D_sqrt, y)
        return
    pell_basic_sol = pellFundamentalSolution(D)[0]
    if pell_basic_sol is None: return
    x0, y0 = pell_basic_sol
    pell_heap = []
    for x, y in generalisedPellFundamentalSolutions(D, n,\
            pell_basic_sol=pell_basic_sol):
        #print(x, y)
        pell_heap.append((abs(x), abs(y), x, y, True))
        if y: pell_heap.append((abs(x), abs(y), x, y, False))
    heapq.heapify(pell_heap)
    prev_x = None
    #print(f"pell_heap = {pell_heap}")
    while pell_heap:
        x2, y2, x, y, b = heapq.heappop(pell_heap)
        #print(f"x = {x}, y = {y}")
        #x2, y2 = abs(x), abs(y)
        if x2 != prev_x and x2 and y2:
            yield (x2, y2)
        prev_x = x2
        if b:
            x_, y_ = (x * x0 + y * y0 * D, x * y0 + y * x0)
            heapq.heappush(pell_heap,\
                    (abs(x_), abs(y_), x_, y_, b))
        else:
            x_, y_ = (x * x0 - y * y0 * D, -x * y0 + y * x0)
            heapq.heappush(pell_heap,\
                    (abs(x_), abs(y_), x_, y_, b))
    return

if __name__ == "__main__":
    pass