#!/usr/bin/env python

from typing import (
    Generator,
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
)

import bisect
import functools
import itertools
import random
import time

from algorithms.number_theory_algorithms import isqrt

Real = Union[int, float]

def largestLEpowN(
    num: Union[int, float],
    base: int=10,
) -> int:
    if num <= base: return 1
    base_lst = [base]
    while base_lst[-1] < num:
        base_lst.append(base_lst[-1] ** 2)
    if num == base_lst[-1]: return 1 << (len(base_lst) - 1)
    base_lst.pop()
    i = len(base_lst) - 1
    num //= pow(base, 1 << i)
    res = 1 << i
    while num >= base:
        i = bisect.bisect_right(base_lst, num) - 1
        num //= pow(base, 1 << i)
        res |= 1 << i
    return res
    
class SimplePrimeSieve(object):
    """
    Simple prime sieve
    """
    def __init__(
        self,
        n_max: Optional[int]=None,
        use_p_lst: bool=True
    ):
        if n_max is None: n_max = 2
        self.sieve = [False, False, True, True]
        self.use_p_lst = use_p_lst
        if use_p_lst:
            self.p_lst = [2, 3]
        self.extendSieve(n_max)
    
    def extendSieve(self, n_max: int) -> None:
        if not self.use_p_lst:
            return self.extendSieveNoPLst(n_max)
        #since = time.time()
        n_orig = len(self.sieve) - 1
        if n_orig >= n_max: return
        sieve = self.sieve
        p_lst = self.p_lst
        sieve.extend([True for x in range(n_orig + 1, n_max + 1)])
        for p in p_lst:
            if p * p > n_max: break
            for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                i2 = p * i
                sieve[i2] = False
        
        def incorporatePrime(p: int) -> None:
            #print(f"p = {p}")
            if sieve[p]: p_lst.append(p)
            else: return
            for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                i2 = p * i
                sieve[i2] = False
            return
        
        # Using the fact that primes greater than 2 are 1 modulo 2
        start = n_orig + 1 + (n_orig & 1)
        end = isqrt(n_max)
        #print(start, end)
        p = start - 2
        for p in range(start, end + 1, 2):
            incorporatePrime(p)
        for p in range(p + 2, n_max + 1, 2):
            if sieve[p]: p_lst.append(p)
        return

    def extendSieveNoPLst(self, n_max: int):
        n_orig = len(self.sieve) - 1
        if n_orig >= n_max: return
        sieve = self.sieve
        sieve.extend([True for x in range(n_orig + 1, n_max + 1)])
        for i in range(max(2, (n_orig >> 1) + 1), (n_max >> 1) + 1):
            sieve[i << 1] = False
        for p in range(3, n_max + 1, 2):
            if p * p > n_max: break
            if not sieve[p]: continue
            for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                i2 = p * i
                sieve[i2] = False
        return

    @staticmethod
    def _millerRabinTrial(
        n: int,
        s: int,
        d: int,
        a: int,
    ) -> bool:
        x = pow(a, d, n)
        for _ in range(s):
            y = pow(x, 2, n)
            if y == 1 and x != 1 and x != n - 1:
                return False
            x = y
        if y != 1: return False
        return True
    
    def millerRabinPrimalityTest(
        self,
        n: int,
        a_vals: List[int],
        max_n_additional_trials: int=3
    ) -> bool:
        seen = {0}
        s = 1
        d = (n - 1) >> 1
        while not d & 1:
            d >>= 1
            s += 1
        for a in set(a_vals):
            seen.add(a)
            if not self._millerRabinTrial(n, s, d, a): return False
        for _ in range(min(max_n_additional_trials, n - len(seen) - 2)):
            a = 0
            while a in seen:
                a = random.randrange(2, n - 1)
            seen.add(a)
            if not self._millerRabinTrial(n, s, a): return False
        return True
    
    def millerRabinPrimalityTestWithKnownBounds(
        self,
        n: int,
        max_n_additional_trials_if_above_max: int=10,
    ) -> Tuple[bool, bool]:
        # List from https://en.wikipedia.org/wiki/Miller%E2%80%93Rabin_primality_test
        # index 0 signifies if the test shows n is a prime, index
        # 1 signifies whether that test definitely gave the correct
        # answer
        if n <= 1: return (False, True)
        elif n == 2: return (True, True)
        mx = 2 ** 64 - 1
        thresholds = [
            (0, [2]),
            (2047, [3]),
            (1373653, [5]),
            (25326001, [7]),
            (3215031751, [11]),
            (2152302898747, [13]),
            (3474749660383, [17]),
            (341550071728321, [19, 23]),
            (3825123056546413051, [31, 37]),
        ]
        seen = {0}
        s = 1
        d = (n - 1) >> 1
        while not d & 1:
            d >>= 1
            s += 1
        for m, a_lst in thresholds:
            if n < m: return (True, True)
            for a in a_lst:
                seen.add(a)
                if not self._millerRabinTrial(n, s, d, a): return (False, True)
        if n <= mx: return (True, True)
        for _ in range(min(max_n_additional_trials_if_above_max, n - len(seen) - 2)):
            a = 0
            while a in seen:
                a = random.randrange(2, n - 1)
            seen.add(a)
            if not self._millerRabinTrial(n, s, d, a): return (False, True)
        return (True, False)
    
    def isPrime(
        self,
        n: int,
        extend_sieve: bool=False,
        extend_sieve_sqrt: bool=False,
        use_miller_rabin_screening: bool=False,
        n_miller_rabin_trials: int=3,
    ) -> bool:
        """
        Checks if the strictly positive integer n is prime.
        
        Args:
            Required positional:
            n (int): The strictly positive integer whose status as a
                    prime number is to be assessed
            
            Optional named:
            extend_sieve (bool): Whether the prime sieve should be extended
                    up to n (if it currently does not extend that far) to
                    check for the prime.
                    Note that in general, in checking a single prime this
                    option makes the search significantly slower and uses
                    more memory, but is more efficient if checking multiple
                    primes of similar size.
                Default: False
            extend_sieve_sqrt (bool): If extend_sieve is given as False and
                    the sieve currently does not extend that far, whether
                    to extend the sieve up to the square root of n.
                    This is a compormise between fully extending the sieve
                    and not extending the sieve at all.
                Default: False
            use_miller_rabin_screening (bool): Uses Miller-Rabin primality
                    test to see if this test shows that n is definitely
                    composite before performing the full primality check.
                Default: False
            n_miller_rabin_trials (int): If use_miller_rabin_screening
                    is set to True, how many different trials of the
                    Miller-Rabin primality test (with different bases each
                    time) each return that n may be prime before performing
                    the full primality test.
                Default: 3
        
        Returns:
        Boolean (bool) giving whether or not n is a prime number.
        """
        if extend_sieve or len(self.sieve) > n:
            self.extendSieve(n)
            return self.sieve[n]
        if use_miller_rabin_screening and not\
                self.millerRabinPrimalityTest(n, n_trials=n_miller_rabin_trials):
            return False
        sqrt = isqrt(n)
        if extend_sieve_sqrt:
            self.extendSieve(sqrt)
        if self.use_p_lst:
            for p in self.p_lst:
                if p > sqrt: return True
                elif not n % p:
                    return False
        else:
            if not n & 1: return False
            for p in range(3, min(len(self.sieve), sqrt + 1), 2):
                if self.sieve[p] and not n % p:
                    return False
        p = len(self.sieve)
        p += not p & 1
        for p in range(p, sqrt + 1, 2):
            if not n % p:
                return False
        return True

    def endlessPrimeGenerator(self) -> Generator[int, None, None]:
        """
        Generates the primes in order of increasing size. Given that
        there are infinitely many prime numbers, this generator does
        not terminate by itself, so any loop utilising this generator
        must contain a break or return statement.
        Note that this generator can only be used if on instatiation
        of the SimplePrimeSieve object the option use_p_lst was set
        to True.
        
        Yields:
        Integer (int), with (for integer i) the ith item yielded
        being the ith prime
        """
        if not self.use_p_lst:
            return NotImplementedError("This method requires the SimplePrimeSieve "
                    "object to have use_p_lst set to True.")
        for i, p in enumerate(self.p_lst): yield p
        n_mx = 10 ** (largestLEpowN(len(self.sieve) - 1, base=10) + 1)
        #n_mx = 10 ** (math.ceil(math.log(len(self.sieve) - 1, 10)) + 1)
        while True:
            self.extendSieve(n_mx)
            for i in range(i + 1, len(self.p_lst)):
                yield self.p_lst[i]
            n_mx *= 10
        return
    
    def primeCountingFunction(self, n: Union[int, float]) -> int:
        """
        Gives the number of prime numbers less than or equal to a given
        number
        
        Args:
            Required positional:
            n (int/float): The number up to which the prime numbers are
                    to be counted
        
        Returns:
        Integer (int) giving the number of prime numbers no greater
        than n.
        """
        if n < 2: return 0
        if not self.use_p_lst:
            return NotImplementedError("This method requires the SimplePrimeSieve "
                    "object to have use_p_lst set to True.")
        n = int(n)
        self.extendSieve(n)
        return bisect.bisect_right(self.p_lst, n)

# Review- consider implementing as a child class of SimplePrimeSieve
class PrimeSPFsieve(object):
    """
    Finds the smallest prime factor for each number and the number of times
    it divides the number and (if use_p_lst) generates a list of primes.
    Methods primeFactors() and primeFactorisation() use this sieve to
    identify unique prime factors and find the factorisation of a given
    positive integer respectively
    """
    def __init__(self, n_max: Optional[int]=None, use_p_lst: bool=True, extend_mod6: bool=False):
        if n_max is None: n_max = 2
        self.sieve = [(-1, 0, 1)] * 2
        self.sieve.append((2, 1, 1))
        self.sieve.append((3, 1, 1))
        self.use_p_lst = use_p_lst
        if use_p_lst:
            self.p_lst = [2, 3]
        self.extend_mod6 = extend_mod6
        self.extendSieve(n_max)
    
    def extendSieve(self, n_max: int, mod6: Optional[bool]=None) -> None:
        if mod6 is None: mod6 = self.extend_mod6
        if not self.use_p_lst:
            return self.extendSieveNoPLst(n_max)
        since = time.time()
        n_orig = len(self.sieve) - 1
        if n_orig >= n_max: return
        sieve = self.sieve
        p_lst = self.p_lst
        sieve.extend([(x, 1, 1) for x in range(n_orig + 1, n_max + 1)])
        for p in p_lst:
            if p * p > n_max: break
            for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                i2 = p * i
                if p > sieve[i2][0]: continue
                sieve[i2] = (p, sieve[i][1] + 1, sieve[i][2]) if sieve[i][0] == p else (p, 1, i)
        
        def incorporatePrime(p: int) -> None:
            #print(f"p = {p}")
            if sieve[p][0] == p: p_lst.append(p)
            else: return
            for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                i2 = p * i
                if p > sieve[i2][0]: continue
                sieve[i2] = (p, sieve[i][1] + 1, sieve[i][2]) if sieve[i][0] == p else (p, 1, i)
            return
        
        if mod6:
            # Using the fact that primes greater than 3 are 1 or 5 modulo 6
            start = n_orig + 7 - (n_orig % 6)
            end = isqrt(n_max)
            if n_orig < start - 2 <= end:
                incorporatePrime(start - 2)
            p1 = start - 6
            for p1 in range(start, end - 3, 6):
                incorporatePrime(p1)
                incorporatePrime(p1 + 4)
            p1 += 6
            if p1 == start and max(n_orig, end) < start - 2 <= n_max:
                
                p2 = p1 - 2
                if sieve[p2][0] == p2:
                    p_lst.append(p2)
            if p1 <= end:
                incorporatePrime(p1)
                p2 = p1 + 4
                if sieve[p2][0] == p2:
                    p_lst.append(p2)
                p1 += 6
            #print(p1, n_max - 3)
            for p1 in range(p1, n_max - 3, 6):
                if sieve[p1][0] == p1:
                    p_lst.append(p1)
                p2 = p1 + 4
                if sieve[p2][0] == p2:
                    p_lst.append(p2)
            r = n_max % 6
            #print(r)
            if 1 <= r < 5:
                p = n_max - r + 1
                if p > n_orig and sieve[p][0] == p:
                    p_lst.append(p)
        else:
            # Using the fact that primes greater than 2 are 1 modulo 2
            start = n_orig + 1 + (n_orig & 1)
            end = isqrt(n_max)
            #print(start, end)
            p = start - 2
            for p in range(start, end + 1, 2):
                incorporatePrime(p)
                #if sieve[p][0] == p: p_lst.append(p)
                #if p * p > n_max: break
                #if sieve[p][0] != p: continue
                #for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                #    i2 = p * i
                #    if p > sieve[i2][0]: continue
                #    sieve[i2] = (p, sieve[i][1] + 1, sieve[i][2]) if sieve[i][0] == p else (p, 1, i)
            for p in range(p + 2, n_max + 1, 2):
                #print(p)
                if sieve[p][0] == p: p_lst.append(p)
        #print(f"Time taken = {time.time() - since:.4f} seconds")
        return

    def extendSieveNoPLst(self, n_max: int):
        n_orig = len(self.sieve) - 1
        if n_orig >= n_max: return
        sieve = self.sieve
        sieve.extend([(x, 1, 1) for x in range(n_orig + 1, n_max + 1)])
        for i in range(max(2, (n_orig >> 1) + 1), (n_max >> 1) + 1):
            prev_count = sieve[i][1] if sieve[i][0] == 2 else 0
            sieve[i << 1] = (2, prev_count + 1, sieve[i][2]) if sieve[i][0] == 2 else (2, 1, i)
        for p in range(3, n_max + 1, 2):
            if p * p > n_max: break
            if sieve[p][0] != p: continue
            for i in range(max(p, (n_orig // p) + 1), (n_max // p) + 1):
                i2 = p * i
                if p > sieve[i2][0]: continue
                sieve[i2] = (p, sieve[i][1] + 1, sieve[i][2]) if sieve[i][0] == p else (p, 1, i)
        return
    
    def millerRabinPrimalityTest(
        self,
        n: int,
        n_trials: int=3,
    ) -> bool:
        seen = {0}
        s = 1
        d = (n - 1) >> 1
        while not d & 1:
            d >>= 1
            s += 1
        for _ in range(min(n_trials, n - 3)):
            a = 0
            while a in seen:
                a = random.randrange(2, n - 1)
            seen.add(a)
            x = pow(a, d, n)
            for _ in range(s):
                y = pow(x, 2, n)
                if y == 1 and x != 1 and x != n - 1:
                    return False
                x = y
            if y != 1: return False
        return True
    
    def isPrime(
        self,
        n: int,
        extend_sieve: bool=False,
        extend_sieve_sqrt: bool=False,
        use_miller_rabin_screening: bool=False,
        n_miller_rabin_trials: int=3,
    ) -> bool:
        """
        Checks if the strictly positive integer n is prime.
        
        Args:
            Required positional:
            n (int): The strictly positive integer whose status as a
                    prime number is to be assessed
            
            Optional named:
            extend_sieve (bool): Whether the prime sieve should be extended
                    up to n (if it currently does not extend that far) to
                    check for the prime.
                    Note that in general, in checking a single prime this
                    option makes the search significantly slower and uses
                    more memory, but is more efficient if checking multiple
                    primes of similar size.
                Default: False
            extend_sieve_sqrt (bool): If extend_sieve is given as False and
                    the sieve currently does not extend that far, whether
                    to extend the sieve up to the square root of n.
                    This is a compormise between fully extending the sieve
                    and not extending the sieve at all.
                Default: False
            use_miller_rabin_screening (bool): Uses Miller-Rabin primality
                    test to see if this test shows that n is definitely
                    composite before performing the full primality check.
                Default: False
            n_miller_rabin_trials (int): If use_miller_rabin_screening
                    is set to True, how many different trials of the
                    Miller-Rabin primality test (with different bases each
                    time) each return that n may be prime before performing
                    the full primality test.
                Default: 3
        
        Returns:
        Boolean (bool) giving whether or not n is a prime number.
        """
        if extend_sieve or len(self.sieve) > n:
            self.extendSieve(n)
            return self.sieve[n][0] == n
        if use_miller_rabin_screening and not\
                self.millerRabinPrimalityTest(n, n_trials=n_miller_rabin_trials):
            return False
        sqrt = isqrt(n)
        if extend_sieve_sqrt:
            self.extendSieve(sqrt)
        if self.use_p_lst:
            for p in self.p_lst:
                if p > sqrt: return True
                elif not n % p:
                    return False
        else:
            if not n & 1: return False
            for p in range(3, min(len(self.sieve), sqrt + 1), 2):
                if self.sieve[p][0] == p and not n % p:
                    return False
        p = len(self.sieve)
        p += not p & 1
        for p in range(p, sqrt + 1, 2):
            if not n % p:
                return False
        return True

    def primeFactors(self, n: int) -> List[int]:
        """
        Gives the prime factors of a strictly positive integer
        
        Args:
            Required positional:
            n (int): The strictly positive integer of which the prime
                    factors are to be found
        
        Returns:
        Set of ints, representing the complete set of prime factors
        of n.
        """
        self.extendSieve(n)
        sieve = self.sieve
        #print(n, sieve)
        res = []
        while n != 1:
            res.append(sieve[n][0])
            n = sieve[n][2]
        return res

    def primeFactorisation(self, n: int) -> Dict[int, int]:
        """
        Gives the prime factorisation of a strictly positive integer.
        Note that by the Fundamental Theorem of Arithmetic this
        factorisation is guaranteed to be unique.
        
        Args:
            Required positional:
            n (int): The strictly positive integer of which the prime
                    factorisatio is to be found.
        
        Returns:
        Dictionary (dict) whose keys are the complete set of prime
        factors of n (as ints) and whose values are the corresponding
        power of that prime in the factorisation (also as ints).
        """
        self.extendSieve(n)
        sieve = self.sieve
        res = {}
        while n != 1:
            res[sieve[n][0]] = sieve[n][1]
            n = sieve[n][2]
        return res
    
    def factorCount(self, n: int) -> int:
        """
        Gives the number of positive factors of a strictly positive
        integer.
        
        Args:
            Required positional:
            n (int): The strictly positive integer for which the number
                    of positive factors is to be calculated
        
        Returns:
        Integer (int) giving the number of positive factors on n.
        """
        p_fact = self.primeFactorisation(n)
        res = 1
        for v in p_fact.values():
            res *= v + 1
        return res
    
    def factors(self, n: int) -> Set[int]:
        """
        Gives the complete set of positive factors of a strictly positive
        integer.
        
        Args:
            Required positional:
            n (int): The strictly positive integer whose factors are to
                    be found.
        
        Returns:
        Set of ints, representing the complete set of positive factors
        of n.
        """
        p_fact = self.primeFactorisation(n)
        p_lst = list(p_fact.keys())
        iter_lst = [[p ** x for x in range(p_fact[p] + 1)] for p in p_lst]
        res = set()
        for p_pow_tup in itertools.product(*iter_lst):
            res.add(functools.reduce(lambda x, y: x * y, p_pow_tup))
        return res
    
    def factorisationsGenerator(self, n: int,\
            max_factor: Union[int, float]=float("inf"),\
            max_n_factors: Union[int, float]=float("inf"),  proper: bool=False)\
            -> Generator[Tuple[int], None, None]:
        """
        Generator yielding every factorisation of a strictly positive
        integer into a product of integers greater than 1 subject to given
        constraints.
        
        Args:
            Required positional:
            n (int): The strictly positive integer whose factorisation
                    into integers greater than 1 is to be found.
            
            Optional named:
            max_factor (int/float): If given, gives the largest factor
                    that is allowed in any factorisation yielded.
                Default: float("inf") (i.e. no restriction on the size
                    of any factor in the factorisations yielded)
            max_n_factors (int/float): If given, gives the largest
                    number of factors (including multiplicity) allowed
                    in any factorisation yielded.
                Default: float("inf") (i.e. no restriction on the
                    number of factors in the factorisations yielded)
            proper (bool): Whether the factorisations yielded should
                    all be proper (i.e. whether the trivial factorisation
                    (n,) should be yielded, assuming it satisfies the
                    other constraints).
                Default: True
        
        Yields:
        Tuple of ints giving each of the factorisations of n in turn.
        In each factorisation, the factors comprising the factorisation
        are ordered in increasing size, with repeated factors appearing
        as many times as they are repeated in the factorisation.
        """
        if not max_n_factors: return
        elif max_n_factors == 1:
            if not proper and max_factor >= n:
                yield (n,)
            return
        facts = self.factors(n)
        facts -= {1, n}
        facts = sorted(facts)
        for fact in facts:
            if fact > max_factor: break
            n2 = n // fact
            for facts2 in self.factorisationsGenerator(n2,\
                    max_factor=fact, max_n_factors=max_n_factors - 1, proper=False):
                yield (*facts2, fact)
        else:
            if max_factor >= n and not proper:
                yield (n,)
        return
    
    def endlessPrimeGenerator(self) -> Generator[int, None, None]:
        """
        Generates the primes in order of increasing size. Given that
        there are infinitely many prime numbers, this generator does
        not terminate by itself, so any loop utilising this generator
        must contain a break or return statement.
        Note that this generator can only be used if on instatiation
        of the PrimeSPFsieve object the option use_p_lst was set
        to True.
        
        Yields:
        Integer (int), with (for integer i) the ith item yielded
        being the ith prime
        """
        if not self.use_p_lst:
            return NotImplementedError("This method requires the PrimeSPFsieve "
                    "object to have use_p_lst set to True.")
        for i, p in enumerate(self.p_lst): yield p
        n_mx = 10 ** (largestLEpowN(len(self.sieve) - 1, base=10) + 1)
        #n_mx = 10 ** (math.ceil(math.log(len(self.sieve) - 1, 10)) + 1)
        while True:
            self.extendSieve(n_mx)
            for i in range(i + 1, len(self.p_lst)):
                yield self.p_lst[i]
            n_mx *= 10
        return
    
    def primeCountingFunction(self, n: Real) -> int:
        """
        Gives the number of prime numbers less than or equal to a given
        number
        
        Args:
            Required positional:
            n (int/float): The number up to which the prime numbers are
                    to be counted
        
        Returns:
        Integer (int) giving the number of prime numbers no greater
        than n.
        """
        if n < 2: return 0
        if not self.use_p_lst:
            return NotImplementedError("This method requires the PrimeSPFsieve "
                    "object to have use_p_lst set to True.")
        n = int(n)
        self.extendSieve(n)
        return bisect.bisect_right(self.p_lst, n)
