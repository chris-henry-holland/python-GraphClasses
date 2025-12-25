
from typing import (
    Tuple,
    Generator,
)

from algorithms.number_theory_algorithms import (
    gcd,
    solveLinearNonHomogeneousDiophantineEquation,
)


def adjacentFarey(
    frac: Tuple[int],
    max_denom: int,
    nxt: bool=True,
    frac_farey: bool=False,
) -> Tuple[int, int]:
    """
    For any fraction frac between 0 and 1 inclusive with denominator in
    lowest terms no greater than max_denom, finds the largest fraction
    strictly less than frac and no less than 0 (if nxt given as False)
    or the smallest fraction strictly greater than frac and no greater
    than 1 (if nxt given as True) with denominator no greater than
    max_denom (or equivalently, finds the fraction that precedes/succeeds
    frac in the Farey sequence of order max_denom).
    
    Args:
        Required positional:
        frac (2-tuple of ints): The fraction strictly between 0 and 1,
                represented by a 2-tuple of ints whose 0th index contains
                the numerator and whose 1st index contains the denominator
        max_denom (int): The upper bound (inclusive) of the denominator
                for both frac and the solution in lowest terms
        
        Optional named:
        nxt (bool): If True then finds the next element in the Farey
                sequence of order max_denom (i.e. the smallest fraction
                strictly greater than frac with denominator no greater
                than max_denom), otherwise the previous element in the Farey
                sequence of order max_denom (i.e. the largest fraction
                strictly less than frac with denominator no greater
                than max_denom).
            Default: True
        frac_farey (bool): If True then the input frac is guaranteed
                to represent an element of the Farey sequence in lowest
                terms.
            Default: False
    
    Returns:
    2-tuple of ints representing the unique fraction in lowest terms that
    satisfies the given requirements, with index 0 and 1 containing the
    numerator and denominator respectively. If no such fraction exists
    (i.e frac = (1, 1) and nxt = True or frac = (0, 1) and nxt = False)
    then returns an empty tuple.
    
    Outline of rationale:
    This uses the fact that two consecutive elements of a Farey sequence
    p1 / q1 and p2 / q2 (with the former preceding the latter in the
    sequence) satisfy:
        p2 * q1 - p1 * q2 = 1
    and for the Farey sequence of order n, for an element fo that Farey
    sequence p2 / q2, if p2 / q2 < 1 then the next element, p3 / q3 is
    the integer solution to:
        p3 * q2 - p2 * q3 = 1
    where q3 <= n and there does not exist another solution with
    denominator larger than q3 and no greater than n, and if p2 / q2 > 0
    then the previous element p1 / p2 is the integer solution to:
        p2 * q1 - p1 * q2 = 1
    where q1 <= n and there does not exist another solution with
    denominator larger than q1.
    """
    if not frac_farey:
        if frac[0] < 0 and frac[1] < 0:
            frac = tuple(-x for x in frac)
        if frac[0] <= 0 or frac[0] >= frac[1]:
            raise ValueError("The input parameter frac must represent a "
                    "fraction that is strictly greater than zero and "
                    "strictly less than one.")
        g = gcd(*frac)
        frac = tuple(x // g for x in frac)
        if frac[1] > max_denom:
            raise ValueError("The input parameter frac must represent a "
                    "fraction which, when expressed in lowest terms, has a "
                    "denominator no greater thn max_denom")
    if (nxt and (frac[0] == frac[1])) or (not nxt and frac[0] == 0):
        return ()
    elif nxt and not frac[0]:
        return (1, max_denom)
    # General solution to Diophantine equation a * x - b * y = 1 where
    # frac = (b, a)
    args0 = (frac[0], -frac[1]) if nxt\
            else (-frac[0], frac[1])
    #print(args0)
    dx, x0, dy, y0 = solveLinearNonHomogeneousDiophantineEquation(*args0, 1)
    #print(x0, dx, y0, dy)
    neg = ((max_denom - x0) < 0) ^ (dx < 0)
    #print(neg)
    k = abs(max_denom - x0) // abs(dx)
    if neg: k = -k
    return (y0 + dy * k, x0 + dx * k)

def fareyNext(
    n: int,
    curr: Tuple[int]=(0, 1),
    lowest_terms: bool=False,
) -> Tuple[int]:
    """
    Finds the smallest element of the Farey sequence of order n which
    is strictly larger than the fraction curr.
    Note that this is distinct from adjacentFarey with nxt=True as this
    does not require that curr is itself in the Farey sequence of order
    n.
    
    Args:
        Required positional:
        n (int): The order of the Farey sequence in question
        
        Optional named:
        curr (2-tuple of ints): A fraction no less than 0 and strictly
                less than one, represented by a 2-tuple of ints whose
                0th index contains the numerator and whose 1st index
                contains the denominator, giving the fraction for
                which the smallest element of the Farey sequence with
                value strictly greater than this fraction is being
                sought.
            Default: (0, 1)
        lowest_terms (bool): Whether the fraction curr is guaranteed
                to be in lowest terms (i.e. gcd(curr[0], curr[1]) = 1
                and curr[1] > 0)
            Default: False
    
    Returns:
    2-tuple of ints representing smallest the fraction in the Farey
    sequence of order n whose value is strictly greater than the
    fraction curr whose 0th index contains the numerator and whose
    1st index contains the denominator. If no such element exists,
    returns an empty tuple.
    """
    if not lowest_terms:
        if curr[0] ^ curr[1]: return (0, 1)
        elif curr[0] < 0: curr = tuple(-x for x in curr)
        if curr[0] >= curr[1]: return ()
        g = gcd(*curr)
        curr = tuple(x // g for x in curr)
    elif curr[0] < 0: return (0, 1)
    elif curr[0] >= curr[1]: return ()
    if curr == (0, 1): return (1, n)
    elif curr[0] == 1 and (curr[1] << 1) > n:
        return (1, curr[1] - 1)
    elif curr[0] == curr[1] - 1 and ((curr[1] + 1) << 1) > n:
        if curr[1] >= n: return (1, 1)
        return (curr[0] + 1, curr[1] + 1)
    if curr[1] <= n:
        # curr is in the Farey sequence of order n
        return adjacentFarey(curr, n, nxt=True, frac_farey=True)
    res = (1, 1)
    for denom in range(2, n + 1):
        numer = ((curr[0] * denom) // curr[1]) + 1
        if numer * res[1] < denom * res[0]:
            res = (numer, denom)
    return res

def fareySequence(
    n: int,
    mn: Tuple[int]=(0, 1),
    mx: Tuple[int]=(1, 1)
) -> Generator[Tuple[int], None, None]:
    """
    Generator iterating over the elements of the Farey sequence of order
    n for elements with value between mn and mx (inclusive) in
    increasing order.
    
    Args:
        Required positional:
        n (int): The order of the Farey sequence being iterated over
        
        Optional named:
        mn (2-tuple of ints): A fraction between 0 and 1 inclusive,
                represented by a 2-tuple of ints whose 0th index
                contains the numerator and whose 1st index contains
                the denominator, giving the smallest value of
                fractions that may be yielded by the generator.
            Default: (0, 1)
        mx (2-tuple of ints): A fraction between 0 and 1 inclusive,
                represented by a 2-tuple of ints whose 0th index
                contains the numerator and whose 1st index contains
                the denominator, giving the largest value of
                fractions that may be yielded by the generator.
            Default: (1, 1)
    
    Yields:
    Each element of the Farey sequence of order n that is no less than
    the fraction mn and no greater than the fraction mx in order of
    increasing size and in lowest terms, represented by a 2-tuple of
    ints whose 0th index contains the numerator and whose 1st index
    contains the denominator.
    """
    # Using https://en.wikipedia.org/wiki/Farey_sequence
    if mx[0] > mx[1]: mx = (1, 0)
    if (mn[0] < 0) ^ (mn[1] < 0): mn = (0, 1)
    elif mn[0] < 0: mn = tuple(-x for x in mn)
    if mn[0] * mx[1] > mn[1] * mx[0] or mn[0] > mn[1]: return
    g = gcd(*mn)
    mn = tuple(x // g for x in mn)
    if mn[1] <= n: prev = mn
    else:
        prev = fareyNext(n, curr=mn, lowest_terms=True)
        if not prev or prev[0] * mx[1] > prev[1] * mx[0]: return
    yield prev
    curr = adjacentFarey(prev, n, nxt=True, frac_farey=True)
    while curr[0] * mx[1] <= curr[1] * mx[0]:
        yield curr
        mult = (n + prev[1]) // curr[1]
        prev, curr = curr, (mult * curr[0] - prev[0],\
                            mult * curr[1] - prev[1])
    return