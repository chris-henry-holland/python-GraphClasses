#! /usr/bin/env python

# src/algorithms/number_theory_algorithms.py

from typing import (
    List,
    Union,
    Tuple,
    Optional,
)

Real = Union[int, float]

def gcd(a: int, b: int) -> int:
    """
    For non-negative integers a and b (not both zero),
    calculates the greatest common divisor of the two, i.e.
    the largest positive integer that is an exact divisor
    of both a and b.

    Args:
        Required positional:
        a (int): Non-negative integer which is the first
                which the greatest common divisor must
                divide.
        b (int): Non-negative integer which is the second
                which the greatest common divisor must
                divide. Must be non-zero if a is zero.
    
    Returns:
    Strictly positive integer giving the greatest common
    divisor of a and b.
    """
    #return a if not b else gcd(b, a % b)
    while b != 0:
        a, b = b, a % b
    return a
    
def lcm(a: int, b: int) -> int:
    """
    For non-negative integers a and b (not both zero),
    calculates the lowest common multiple of the two, i.e.
    the smallest positive integer that is a multiple
    of both a and b.

    Args:
        Required positional:
        a (int): Non-negative integer which is the first
                which must divide the lowest common multiple.
        b (int): Non-negative integer which is the second
                which must divide the lowest common multiple.
                Must be non-zero if a is zero.
    
    Returns:
    Strictly positive integer giving the lowest common
    multiple of a and b.
    """

    return a * (b // gcd(a, b))

def extendedEuclideanAlgorithm(a: int, b: int) -> Tuple[int, Tuple[int, int]]:
    """
    Implementation of the extended Euclidean Algorithm to find the
    greatest common divisor (gcd) of integers a and b and finds an
    ordered pair of integers (m, n) such that:
        m * a + n * b = gcd(a, b)
    
    Args:
        Required positional:
        a (int): The first of the integers on which the extended
                Euclidean Algorithm is to be applied
        b (int): The second of the integers on which the extended
                Euclidan Algorithm is to be applied
    
    Returns:
    2-tuple whose index 0 contains a non-negative integer giving
    the greatest common divisor (gcd) of a and b, and whose
    index 1 contains a 2-tuple of integers, giving an ordered
    pair of integers (m, n) such that:
        m * a + n * b = gcd(a, b)
    """
    if b > a:
        swapped = True
        a, b = b, a
    else: swapped = False
    q_stk = []
    curr = [a, b]
    while True:
        q, r = divmod(*curr)
        if not r: break
        q_stk.append(q)
        curr = [curr[1], r]

    g = curr[1]
    mn_pair = [0, 1]
    while q_stk:
        q = q_stk.pop()
        mn_pair = [mn_pair[1], mn_pair[0] + mn_pair[1] * (-q)]
    if swapped: mn_pair = mn_pair[::-1]
    return (g, tuple(mn_pair))

def solveLinearCongruence(a: int, b: int, md: int) -> int:
    """
    Finds the smallest non-negative integer k such that solves
    the linear congruence:
        k * a = b (mod md)
    if such a value exists.

    A congruence relation for two integers m and n over a given
    modulus md:
        m = n (mod md)
    is a relation such that there exists an integer q such that:
        m + q * md = n
    
    Args:
        Required positional:
        a (int): Integer specifying the value of a in the above
                congruence to be solved for k.
        b (int): Integer specifying the value of b in the above
                linear congruence to be solved for k.
        md (int): Strictly positive integer specifying the
                modulus of the congruence (i.e. the value md in
                the linear congruence to be solved for k)
        
    Returns:
    Integer (int) giving the smallest non-negative integer value
    of k for which the linear congruence:
        k * a = b (mod md)
    is true if any such value exists, otherwise -1.

    Outline of method:
    Solves by first using the extended Euclidean algorithm to
    find the greatest common divisor (gcd) of a and md and
    an integer pair (m, n) for which:
        m * a + n * md = gcd(a, md)
    This implies the congruence:
        m * a = gcd(a, md) (mod md)
    If gcd(a, md) does not divide b then the linear congruence
    has no solution, as any linear combination of a and md with
    integer coefficients is a multiple of gcd(a, md). Otherwise,
    a solution to the linear congruence is:
        k = m * (b / gcd(a, md))
    A known property of linear congruences is that if there
    exists a solution, then any other integer is a solution
    if and only if it is congruent to the known solution under
    the chosen modulus.
    Therefore, to find the smallest non-negative such value,
    we take the smallest non-negative integer to which this
    value is congruent modulo md (which in Python can be found
    using k % md).
    """
    a %= md
    g, (m, n) = extendedEuclideanAlgorithm(a, md)
    b %= md
    q, r = divmod(b, g)
    return -1 if r else (q * m) % md

def nthRoot(a: Real, b: int, eps: Real=1e-5) -> Real:
    """
    Finds the non-negative real b:th root of a (a^(1/b)) to a given
    accuracy using the Newton-Raphson method.

    Args:
        Required positional:
        a (real numeric value): The number whose root is sought.
                This number should be real and non-negative.
        b (int): The root of a to be found. Must be non-zero.
                If this is negative, then a must be non-zero.
        
        Optional named:
        eps (small positive real numeric value): The maximum
                permitted error (i.e. the absolute difference
                between the actual value and the returned value
                must be no larger than this)
    
    Returns:
    Real numeric value giving a value within eps of the non-negative
    b:th root of a.

    Examples:
        >>> nthRoot(2, 2, eps=1e-5)
        1.4142156862745097

        This is indeed within 0.00001 of the square root of 2, which
        is 1.4142135623730950 (to 16 decimal places)

        >>> nthRoot(589, 5, eps=1e-5)
        3.5811555709280753

        >>> nthRoot(2, -2, eps=1e-5)
        0.7071078431372548

        Again, this is indeed within 0.00001 of the square root of a
        half, which is 0.7071067811865476 (to 16 decimal places)
    """
    if a < 0:
        raise ValueError("a should be non-negative")
    if not a: return 0
    elif b < 0:
        if not a:
            raise ValueError("If b is negative, a must be non-zero")
        b = -b
        a = 1 / a
    if b == 1:
        raise ValueError("b must be non-zero")
    x2 = float("inf")
    x1 = a
    while abs(x2 - x1) >= eps:
        x2 = x1
        x1 = ((b - 1) * x2 + a / x2 ** (b - 1)) / b
    return x2

def integerNthRoot(m: int, n: int) -> int:
    """
    For an integer m and a strictly positive integer n,
    finds the largest integer a such that a ** n <= m (or
    equivalently, the floor of the largest real n:th root
    of m. Note that for even n, m must be non-negative.
    Uses the Newton-Raphson method.
    
    Args:
        Required positional:
        m (int): Integer giving the number whose root is
                to be calculated. Must be non-negative
                if n is even.
        n (int): Strictly positive integer giving the
                root to be calculated.
    
    Returns:
    Integer (int) giving the largest integer a such that
    m ** n <= a.
    
    Examples:
    >>> integerNthRoot(4, 2)
    2
    >>> integerNthRoot(15, 2)
    3
    >>> integerNthRoot(27, 3)
    3
    >>> integerNthRoot(-26, 3)
    -3
    """
    if n < 1:
        raise ValueError("n must be strictly positive")
    if m < 0:
        if n & 1:
            neg = True
            m = -m
        else:
            raise ValueError("m can only be negative if n is odd")
    else: neg = False
    if not m: return 0
    x2 = float("inf")
    x1 = m
    while x1 < x2:
        x2 = x1
        x1 = ((n - 1) * x2 + m // x2 ** (n - 1)) // n
    if not neg: return x2
    if x2 ** n < m:
        x2 += 1
    return -x2

def isqrt(n: int) -> int:
    """
    For a non-negative integer n, finds the largest integer m
    such that m ** 2 <= n (or equivalently, the floor of the
    positive square root of n)
    
    Args:
        Required positional:
        n (int): The number for which the above process is
                performed.
    
    Returns:
    Integer (int) giving the largest integer m such that
    m ** 2 <= n.
    
    Examples:
    >>> isqrt(4)
    2
    >>> isqrt(15)
    3
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    return integerNthRoot(n, 2)

def IntegerPartitionGenerator(
    n: int,
    min_n_part: int=0,
    max_n_part: Optional[int]=None,
    min_part_size: int=1,
    balanced_first: bool=True
):
    mx = n if max_n_part is None else max_n_part
    mn = min_n_part
    if mx == 0 and n == 0:
        if mn == 0: yield []
        return
    if mx <= 0 or n < min_part_size: return
    if mx == 1:
        if mn <= 1: yield [n]
        return
    if mn <= 1 and balanced_first: yield [n]
    iter_obj = range(min_part_size, n - min_part_size + 1)
    if balanced_first: iter_obj = reversed(iter_obj)
    for i in iter_obj:
        for part in IntegerPartitionGenerator(n - i, max(mn - 1, 0), mx - 1, i):
            yield [i, *part]
    if mn <= 1 and not balanced_first: yield [n]
    return

def factorialPrimeFactorExponent(n: int, p: int) -> int:
    """
    For a positive integer n and a prime p, calculates the exponent
    of p in the prime factorisation of n! (n factorial).

    Args:
        n (int): The positive integer whose factorial is being
                assessed for the exponent of the chosen prime in
                the prime factorisation.
        p (int): The prime number whose exponent in n! is being
                calculated. This is assumed to be a prime and is
                not checked, so specifying a non-prime will give
                unexpected behaviour.
    
    Returns:
    Non-negative integer (int) giving the exponent of p in the prime
    factorisation of n! (n factorial).
    """
    res = 0
    while n:
        n //= p
        res += n
    return res

class PrimeModuloCalculator:
    """
    Class for making integer calculations modulo a given prime (i.e.
    calculations as a remainder when divided by p).

    All results modulo p are integers between 0 and (p - 1) inclusive.

    Intended to be extended to accommodate further common calculation
    types.

    Initialization args:
        Required positional:
        p (int): The prime to be used as the modulus for all calculations.
    
    Attributes:
        p (int): The prime to be used as the modulus for all calculations.
    
    Methods:
        (For more detail about a specific method, see that method's
        documentation)

        add(): Finds the sum of two integers modulo p.
        mult(): Finds the product of two integers modulo p.
        pow(): Finds a given integer to a given integer power modulo
                p. For numbers that are not multiples of p, allows
                negative exponents.
        multiplicativeInverse(): Finds the multiplicative inverse of a
                given integer modulo p (i.e. the integer between 1 and
                (p - 1) inclusive that multiplies with the chosen number
                modulo p to give 1). Requires that the chosen integer
                is not a multiple of p.
        factorial(): Finds the factorial of a non-negative integer
                modulo p.
        multiplicativeInverseFactorial(): Finds the multiplicative inverse
                of the factorial of a non-negative integer modulo p.
        binomial(): For an ordered pair of two non-negative integers
                finds the corresponding binomial coefficient modulo
                p.
        multinomial(): For a list of non-negative integers finds the
                multinomial coefficient modulo p.

    Can be used to solve Leetcode #2954
    """

    def __init__(self, p: int):
        self.p = p

    _factorials_div_ppow = [1, 1]

    def add(self, a: int, b: int) -> int:
        """
        Calculates the sum of the integers a and b modulo the attribute
        p (i.e. (a + b) % self.p).

        Args:
            Required positional:
            a (int): One of the two integers to sum modulo p.
            b (int): The other of the two integers to sum modulo p.
        
        Returns:
        An integer (int) between 0 and (p - 1) inclusive giving the
        sum of a and b modulo the attribute p.
        """
        return (a + b) % self.p

    def mult(self, a: int, b: int) -> int:
        """
        Calculates the product of the integers a and b modulo the attribute
        p (i.e. (a * b) % self.p).

        Args:
            Required positional:
            a (int): One of the two integers to multiply modulo p.
            b (int): The other of the two integers to multiply modulo p.
        
        Returns:
        An integer (int) between 0 and (p - 1) inclusive giving the
        product of a and b modulo the attribute p.
        """
        return (a * b) % self.p
    
    def pow(self, a: int, n: int) -> int:
        """
        Calculates the a to the power of n modulo the attribute p for
        integers a and n (i.e. (a ^ n) % self.p).
        For negative n, calculates the modulo p multiplicative inverse of
        a to the power of the absolute value of n (using Fermat's Little
        Theorem). This case requires that a is not a multiple of the
        attribute p.

        Args:
            Required positional:
            a (int): The integer whose exponent modulo p is to be calculated.
            n (int): The integer giving the exponent a is to be taken to.
        
        Returns:
        An integer (int) between 0 and (p - 1) inclusive giving a to the power
        of n modulo p for non-negative n or the modulo p multiplicative inverse
        of a to the power of (-n) modulo p for negative n.
        """
        if not n: return 1
        elif n > 0:
            return pow(a, n, self.p)
        if not a % self.p:
            raise ValueError("a may not be a multiple of p for negative exponents")
        return pow(a, self.p - n - 1, self.p)

    def multiplicativeInverse(self, a: int) -> int:
        """
        Calculates the modulo p multiplicative inverse of the integer a (where
        p is the attribute p), i.e. the integer b such that (a * b) = 1 mod p.
        a must not be a multiple of the attribute p.

        Args:
            Required positional:
            a (int): The integer whose modulo p multiplicative inverse is to be
                    calculated
        
        Returns:
        The modulo p multiplicative inverse of a, i.e. the unique integer b between
        1 and (p - 1) inclusive for which (a - b) = 1 mod p.
        """
        if not a % self.p: raise ValueError("a cannot be a multiple of the attribute p")
        return pow(a, self.p - 2, self.p)
    
    def _extendFactorialsDivPPow(self, a: int) -> None:
        a0 = len(self._factorials_div_ppow) - 1
        if a <= a0: return
        q0, r0 = divmod(a0, self.p)
        q1, r1 = divmod(a, self.p)
        if q0 == q1:
            for i in range(r0 + 1, r1 + 1):
                self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], i))
            return self._factorials_div_ppow[a]
        
        for i in range(r0 + 1, self.p):
            self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], i))
        
        def interPMultExtension(q: int, r_max: int) -> None:
            while not q % self.p:
                q //= self.p
            self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], q))
            for i in range(1, r_max):
                self._factorials_div_ppow.append(self.mult(self._factorials_div_ppow[-1], i))
            return
        
        for q in range(q0 + 1, q1):
            interPMultExtension(q, self.p)
        interPMultExtension(q1, r1 + 1)
        return
    
    def _factorialDivPPow(self, a: int) -> int:
        self._extendFactorialsDivPPow(a)
        return self._factorials_div_ppow[a]
    
    def factorial(self, a: int) -> int:
        """
        For a non-negative integer a calculates the factorial of a modulo
        the attribute p.

        Args:
            Required positional:
            a (int): Non-negative integer whose factorial modulo p is to
                    be calculated
        
        Returns:
        Integer between 0 and (p - 1) inclusive giving the factorial of
        a modulo p (i.e. a! mod p)
        """
        if a >= self.p:
            return 0
        return self._factorialDivPPow(a)
    
    def _multiplicativeInverseFactorialDivPPow(self, a: int) -> int:
        return self.multiplicativeInverse(self._factorialDivPPow(a))
    
    def multiplicativeInverseFactorial(self, a: int) -> int:
        """
        For non-negative a strictly less than the attribute p, calculates
        the modulo p multiplicative inverse of the factorial of a.

        Args:
            Required positional:
            a (int): Non-negative integer strictly less than p whose
                    factorial the modulo p multiplicative inverse is to
                    be calculated.

        Returns:
        Integer between 0 and (p - 1) inclusive giving the modulo p
        multiplicative inverse of the factorial of a (i.e. the integer
        b such that b * a! = 1 mod p)
        """
        if not a % self.p:
            raise ValueError("a may not be a multiple of p")
        return self._multiplicativeInverseFactorialDivPPow(a)

    def binomial(self, n: int, k: int) -> int:
        """
        For integers n and k, calculates the binomial coefficient
        n choose k modulo the attribute p.

        Args:
            Required positional:
            n (int): Integer giving the total number of objects
                    from which to choose.
            k (int): Integer giving the number of the n objects
                    to be selected.

        Returns:
        Integer (int) between 0 and (p - 1) inclusive giving the
        binomial coefficient n choose k modulo p.
        """
        if k > n or k < 0:
            return 0
        if n >= self.p:
            p_exp_numer = factorialPrimeFactorExponent(n, self.p)
            p_exp_denom = factorialPrimeFactorExponent(k, self.p) + factorialPrimeFactorExponent(n - k, self.p)
            if p_exp_numer > p_exp_denom:
                return 0
        return self.mult(
            self._factorialDivPPow(n),
            self.mult(self._multiplicativeInverseFactorialDivPPow(k),
            self._multiplicativeInverseFactorialDivPPow(n - k))
        )

    def multinomial(self, k_lst: List[int]) -> int:
        """
        For the list of non-negative integers k_lst, finds the multinomial
        coefficient modulo the attribute p.

        Args:
            Required positional:
            k_lst (list of ints): List of non-negative integers specifying
                    the multinomial coefficient to be calculated.

        Returns:
        The multinomial coefficient corresponding to k_lst (i.e. sum(k_lst)
        choose (k_lst[0], k_lst[1], ..., k_lst[-1])) modulo the attribute
        p.
        """
        n = sum(k_lst)
        if n >= self.p:
            p_exp_numer = factorialPrimeFactorExponent(n, self.p)
            p_exp_denom = sum(factorialPrimeFactorExponent(k, self.p) for k in k_lst)
            if p_exp_numer > p_exp_denom:
                return 0
        res = self._factorialDivPPow(n)
        for k in k_lst:
            res = self.mult(res, self._multiplicativeInverseFactorialDivPPow(k))
        return res

if __name__ == "__main__":
    res = nthRoot(2, 2, eps=1e-5)
    print(f"nthRoot(2, 2, eps=1e-5) = {res}")

    res = nthRoot(589, 5, eps=1e-5)
    print(f"nthRoot(589, 5, eps=1e-5) = {res}")

    res = nthRoot(2, -2, eps=1e-5)
    print(f"nthRoot(2, -2, eps=1e-5) = {res}")