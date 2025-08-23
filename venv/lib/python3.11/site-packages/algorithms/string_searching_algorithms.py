#!/usr/bin/env python

from typing import (
    Dict,
    Generator,
    List,
    Tuple,
    Optional,
    Union,
    Callable,
    Any,
    Hashable,
    Iterable,
)

import heapq
import itertools

from collections import deque

class KnuthMorrisPratt(object):
    """    
    Class implementing the Knuth-Morris-Pratt string searching
    algorithm, implemented to accept any finite ordered iterable
    object for both the pattern to be matched and the object(s)
    to be searched.

    Initialisation args:
        Required positional:
        pattern (iterable object): A finite ordered iterable
                object (e.g. a string) that is the pattern to be
                found as a contiguous subsequence in other similar
                iterable objects.
    
    Attributes:
        pattern_iter (iterable object): A finite ordered iterable
                object (e.g. a string) that is the pattern
                to be found as a contiguous subsequence in other
                similar iterable objects.
        lps (list of ints): A list with the same length as the number
                of elements in pattern_iter, representing the Longest
                Prefix Suffix (LPS) array for pattern_iter using the
                Knuth-Morris-Pratt (KMP) algorithm, in preparation for
                its use for finding the pattern in a given iterable
                object (see method matchStartGenerator()).
                For a string p of length n, the LPS array is a 1D integer
                array of length n, where the integer at a given index
                represents the longest non-prefix substring of p (i.e.
                a substring of p that does not begin at the start of p)
                ending at the corresponding index that matches a prefix
                of p. Alternatively, as the name suggests, the ith index
                represents the length of the longest proper prefix (i.e.
                a prefix that is not the whole string) of the string
                p[:i + 1] (i.e. the substring of p consisting of the
                first i + 1 characters of p) that is also a (proper)
                suffix of p[:i + 1].
    
    Methods:
        (For more detail see the documentation of the relevant method)
        
        constructLPS(): Constructs the Longest Prefix Suffix (LPS)
                array for attribute pattern_iter using the Knuth-Morris-
                Pratt algorithm in prepartion for use in identifying
                where pattern_iter occurs as a contiguous subsequence
                in some other similar finite ordered iterable objects.
        matchStartGenerator(): Creates a generator which yields the
                indices in a given finite ordered iterable object (where
                the first element of the object has index 0) which
                represent all the start indices of contiguous subsequences
                equal to pattern_iter, yielding these indices one
                at a time in strictly increasing order.
    """
    def __init__(self, pattern: Iterable[Any]):
        self._pattern_iter = pattern
    
    @property
    def pattern_iter(self):
        return self._pattern_iter
        
    @property
    def pattern(self):
        res = getattr(self, "_pattern", None)
        if res is None:
            p_iter = self._pattern_iter
            res = p_iter if hasattr(p_iter, "__getitem__") and\
                    hasattr(p_iter, "__len__") else list(p_iter)
            self._pattern = res
        return res
    
    @property
    def lps(self):
        res = getattr(self, "_lps", None)
        if res is None:
            res = self.constructLPS()
            self._lps = res
        return res
    
    def constructLPS(self) -> List[int]:
        """
        Constructs the Longest Prefix Suffix (LPS) array for the
        pattern represented by this KnuthMorrisPratt object (i.e. the
        attribute pattern) using the Knuth-Morris-Pratt (KMP)
        algorithm, in preparation for its use for finding the pattern
        in a given iterable object (see method matchStartGenerator()).
        For a string p of length n, the LPS array is a 1D integer array
        of length n, where the integer at a given index represents
        the longest non-prefix substring of p (i.e. a substring of p that
        does not begin at the start of p) ending at the corresponding
        index that matches a prefix of p. Alternatively, as the name
        suggests, the ith index represents the length of the longest
        proper prefix (i.e. a prefix that is not the whole string)
        of the string p[:i + 1] (i.e. the substring of p consisting of the
        first i + 1 characters of p) that is also a (proper) suffix of
        p[:i + 1].
        
        Returns:
            2-tuple whose index 0 contains the pattern iterable in a
            list with elements in the same order as in the pattern
            iterable object and whose index 1 contains a list of
            integers (int) representing the LPS array of pattern
           
        Example:
            >>> kmp = KnuthMorrisPratt("abacabcabacad")
            >>> kmp.generateLPS()
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 3, 4, 5, 0]
            
            This signifies for instance that the longest non-prefix
            substring ending at index 11 that matches a prefix of the
            string "abacabcabacad" is length 5 (namely the substring
            "abaca").
        """
        p = self.pattern
        lps = [0]
        j = 0
        p_iter = iter(p)
        next(p_iter)
        for l in p_iter:
            while l != p[j]:
                if not j: break
                j = lps[j - 1]
            else:
                j += 1
            lps.append(j)
        return lps
    
    def matchStartGenerator(self, s: Iterable[Any])\
            -> Generator[Any, None, None]:
        """
        Generator that yields each and every index in the finite
        ordered iterable object s at which a contiguous subsequence
        (for strings, a substring) matching the pattern represented by
        this KnuthMorrisPratt object (i.e. the attribute pattern)
        starts, using the Knuth-Morris-Pratt (KMP) Algorithm.
        
        Args:
            Required positional:
            s (iterable): The finite ordered iterable object in
                    which the start indices of occurences of the
                    pattern represented by this KnuthMorrisPratt object
                    are to be yielded.
        
        Returns:
            Generator yielding integers (int) giving the indices of s
            at which every contiguous subsequence of s matching the
            pattern represented by this KnuthMorrisPratt object (i.e.
            the attribute pattern) start, in increasing order.
        
        Example:
            >>> kmp = KnuthMorrisPratt("bb")
            >>> for i in kmp.matchStartGenerator("casbababbbbbabbceab"):
            >>>     print(i)
            7
            8
            9
            10
            13
            
            This signifies that a substrings of this string exactly
            matching "bb" begin at precisely the (0-indexed) indices 7,
            8, 9, 10 and 13 and nowhere else in the string
            "casbababbbbbabbceab".
        """
        p = self.pattern
        m = len(p)
        if not m:
            # An empty string is a substring of any given string starting
            # from any position in the given string
            yield from range(len(s))
            return
        elif m > len(s):
            # If the pattern is longer than the string, no matches
            # are possible
            return
        lps = self.lps
        j = 0
        for i, l in enumerate(s):
            while l != p[j]:
                if not j: break
                j = lps[j - 1]
            else:
                j += 1
                if j != m: continue
                yield i - m + 1
                j = lps[-1]
        return

class ZAlgorithm(object):
    """    
    Class implementing the Z algorithm for string searching,
    implemented to accept any finite ordered iterable object for both
    the pattern to be matched and the object(s) to be searched.

    Initialisation args:
        Required positional:
        pattern (ordered iterable container): A finite ordered
                iterable object (e.g. a string) that is the pattern to
                be found as a contiguous subsequence in other similar
                finite ordered iterable objects.
    
    Attributes:
        pattern_iter (iterable object): A finite ordered iterable
                object that is the pattern to be found as a contiguous
                subsequence in other similar objects.
    
    Methods:
        (For more detail see the documentation of the relevant method)
        
        constructZArray(): Constructs the Z array for attribute
                pattern_iter.
        matchStartGenerator(): Creates a generator which yields the
                indices in a given finite ordered iterable object (where
                the first element of the object has index 0) which
                represent all the start indices of contiguous subsequences
                equal to pattern_iter, yielding these indices one
                at a time in strictly increasing order.
    """
    def __init__(self, pattern: Iterable[Any]):
        self._pattern_iter = pattern
    
    @property
    def pattern_iter(self):
        return self._pattern_iter
    
    @property
    def pattern(self):
        res = getattr(self, "_pattern", None)
        if res is None:
            p_iter = self._pattern_iter
            res = p_iter if hasattr(p_iter, "__getitem__") and\
                    hasattr(p_iter, "__len__") else list(p_iter)
            self._pattern = res
        return res
    
    def constructZArray(self, s: Iterable[Any]) -> List[int]:
        """
        Constructs the Z array for the finite ordered iterable object
        s.
        For a string s of length n, the Z array is a 1D integer array of
        length n, where the integer at a given index represents
        the longest substring of s starting at the corresponding
        index that matches a prefix of s.
        
        Required positional:
            s (iterable): The finite ordered iterable object for
                    which the Z array is to be calculated.
        
        Returns:
            List of integers (int) representing the Z array
            of the finite ordered iterable object s.
        
        Example:
            >>> z_alg = ZAlgorithm("")
            >>> z_alg.constructZArray("abacabcabacad")
            [13, 0, 1, 0, 2, 0, 0, 5, 0, 1, 0, 1, 0]
            
            This signifies for instance that the longest substring
            starting at index 7 that matches a prefix of the chosen
            string (i.e. "abacabcabacad") is length 5 (namely the
            substring "abaca"- note that the next character of the
            substring starting at index 7 would be "d", which does not
            match the next character for any longer prefix, which would
            be "b").
        """
        n = len(s)
        res = [0] * n
        lft, rgt = 0, 1
        for i in range(1, n):
            if res[i - lft] < rgt - i:
                res[i] = res[i - lft]
                continue
            lft = i
            for rgt in range(max(rgt, i), n):
                if s[rgt] != s[rgt - lft]: break
            else: rgt = n
            res[i] = rgt - lft
        res[0] = n
        return res
        
    def matchStartGenerator(
        self,
        s: Iterable[Any],
        wild: Any="$",
    ) -> Generator[Any, None, None]:
        """
        Generator that yields each and every index in the finite
        ordered iterable object s at which a contiguous subsequence
        (for strings, a substring) matching the pattern represented by
        this ZAlgorithm object (i.e. the attribute pattern)
        starts, using the Z Algorithm.
        
        Args:
            Required positional:
            s (iterable): The finite ordered iterable object in
                    which the start indices of occurences of the
                    pattern represented by this ZAlgorithm object
                    are to be yielded.
        
            Optional named:
            wild_char (str): A string character which does not appear
                    in the string s (used as a separator placed between
                    p and s when they are combined in a single string
                    during the implementation of the Z algorithm to
                    ensure that the start of s cannot be inappropriately
                    treated as being part of the pattern.
                Default: "$"
        
        Returns:
            Generator yielding integers (int) giving the indices of s
            at which every contiguous subsequence of s matching the
            pattern represented by this ZAlgorithm object (i.e.
            the attribute pattern) start, in increasing order.
        
        Example:
            
            >>> z_alg = ZAlgorithm("bb")
            >>> for i in z_alg.matchStartGenerator("casbababbbbbabbceab"):
            >>>     print(i)
            7
            8
            9
            10
            13
            
            This signifies that a substrings of this string exactly
            matching "bb" begin at precisely the (0-indexed) indices 7,
            8, 9, 10 and 13 and nowhere else in the string
            "casbababbbbbabbceab".
        """
        s2 = list(self.pattern)
        m = len(s2)
        s2.append(wild)
        for l in s:
            s2.append(l)
        n2 = len(s2)
        n = n2 - m - 1
        z_arr = self.constructZArray(s2)
        res = []
        for i in range(m + 1, n):
            if z_arr[i] == m:
                yield i - m - 1
        return


def rollingHash(
    s: Iterable[Any],
    length: int,
    p_lst: Union[Tuple[int], List[int]]=(37, 53),
    md: int=10 ** 9 + 7,
    encoding_func: Optional[Callable[[Any], int]]=None,
) -> Generator[Tuple[int], None, None]:
    """
    Generator that yields the rolling hash values of each contiguous subsequence of
    the iterable s with length elements in order of their first element. The hash
    is polynomial-based around prime numbers as specified in p_lst modulo md.
    The elements of s are passed through the function encoding_func which transforms
    each possible input value into a distinct integer (by default, the identity
    if the elements of s are integers, and the ord() function if they are
    string characters).

    For a given length l, a prime p and a modulus md, the hash value of a
    0-indexed integer sequence of length l, arr, is calculated by the following
    formula:
        (arr[0] * p^(l - 1) + arr[1] * p^(l - 2) + ... + arr[l - 2] * p1 + arr[i + l - 1]) % md

    Thus, for strictly positive integer l, non-negative integer i, prime md,
    prime list of length n (p1, p2, ..., pn) and a 0-indexed string s with length
    of no less than (i + l), the (i + 1)th value yielded by the evaluation
    of:
        rollingHash(s, l, p_lst=(p1, p2), md=md, encoding_func=ord)
    is equal to the n-tuple of integers:
        ((ord(s[i]) * p1^(l - 1) + ord(s[i + 1]) * p1^(l - 2) + ... + ord(s[i + l - 2]) * p1 + ord(s[i + l - 1])) % md,
        (ord(s[i]) * p2^(l - 1) + ord(s[i + 1]) * p2^(l - 2) + ... + ord(s[i + l - 2]) * p2 + ord(s[i + l - 1])) % md,
        ...,
        (ord(s[i]) * pn^(l - 1) + ord(s[i + 1]) * pn^(l - 2) + ... + ord(s[i + l - 2]) * pn + ord(s[i + l - 1])) % md)
    
    Args:
        Required positional:
        s (iterable object): A finite ordered iterable object (e.g. a
                string) for which the rolling hash is to be generated.
        length (int): Strictly positive integer giving the lengths
                of the contiguous subsequences over which the rolling
                has is to be generated.
        
        Optional named:
        p_lst (List/tuple of ints): A collection of prime numbers for
                which the polynomial-based hashes are to be evaluated.
                The larger and more numerous the primes given, the
                less likely hash collisions are to occur (i.e. that
                two different sequences give rise to the same hash
                value), though each additional prime proportionally
                increases the evaluation time.
            Default: (37, 53)
        md (int): Strictly positive integer giving the modulus the hash
                values are to be taken (i.e. each hash is returned as
                its remainder to this modulus). For best results, this
                should be a prime number, and the larger the number
                the less likely hash collisions are to occur (see p_lst).
        encoding_func (callable): A function taking in an element of s and
                returning an integer, or for integer or string sequences
                None can be given (for integer sequences, the value is
                used directly, and for strings, the ASCII code number is
                used). To avoid hash collisions, this function should be
                injective (i.e. there should be no two distinct elements
                for which the result of applying the function is the
                same).
            Default: None
    
    Yields:
    An len(p_lst)-tuple of integers (int), each between 0 and (md - 1)
    inclusive, where index j (0 <= j < len(p_lst)) of the (i + 1)th yielded
    tuple is the rolling hash value for the input length, the prime
    p_lst[j] and the modulus md, as calculated above, for the contiguous
    subsequence of s between indices i and (i + length - 1) inclusive,
    where each element has been converted into an integer by the function
    encoding_func.
    The number of results yielded by the generator is one more than the
    number of elements in s minus length as long as this is positive,
    otherwise there are no yielded results
            
    Modified version of rolling hash can be used to solve Leetcode #1554
    (Premium).
    """
    if hasattr(s, "__len__") and len(s) < length:
        return
    if encoding_func is None:
        try: val = encoding_func(next(iter_obj))
        except StopIteration: return
        if isinstance(next(iter(s), str)):
            encoding_func = ord#lambda x: ord(x)
        else: encoding_func = lambda x: x
    iter_obj = iter(s)
    n_p = len(p_lst)
    hsh = [0] * n_p
    val_qu = deque()
    for i in range(length):
        try: val = encoding_func(next(iter_obj))
        except StopIteration: return
        val_qu.append(val)
        for j, p in enumerate(p_lst):
            hsh[j] = (hsh[j] * p + val) % md
    yield tuple(hsh)
    mults = [pow(p, length - 1, md) for p in p_lst]
    for i in itertools.count(length):
        try: val = encoding_func(next(iter_obj))
        except StopIteration: return
        val_qu.append(val)
        val2 = val_qu.popleft()
        for j, p in enumerate(p_lst):
            hsh[j] = ((hsh[j] - mults[j] * val2) * p + val) % md
        yield tuple(hsh)
    return

def rollingHashWithValue(
    s: Iterable[Any],
    length: int,
    p_lst: Union[Tuple[int], List[int]]=(37, 53),
    md: int=10 ** 9 + 7,
    encoding_func: Optional[Callable[[Any], int]]=None,
) -> Generator[Tuple[Any, Tuple[int]], None, None]:
    """
    Generator that yields the rolling hash values of each contiguous subsequence of
    the iterable s with length elements in order of their first element. The hash
    is polynomial-based around prime numbers as specified in p_lst modulo md.
    The elements of s are passed through the function encoding_func which transforms
    each possible input value into a distinct integer (by default, the identity
    if the elements of s are integers, and the ord() function if they are
    string characters).

    For a given length l, a prime p and a modulus md, the hash value of a
    0-indexed integer sequence of length l, arr, is calculated by the following
    formula:
        (arr[0] * p^(l - 1) + arr[1] * p^(l - 2) + ... + arr[l - 2] * p1 + arr[i + l - 1]) % md

    Thus, for strictly positive integer l, non-negative integer i, prime md,
    prime list of length n (p1, p2, ..., pn) and a 0-indexed string s with length
    of no less than (i + l), the (i + 1)th value yielded by the evaluation
    of:
        rollingHash(s, l, p_lst=(p1, p2), md=md, encoding_func=ord)
    is equal to the n-tuple of integers:
        ((ord(s[i]) * p1^(l - 1) + ord(s[i + 1]) * p1^(l - 2) + ... + ord(s[i + l - 2]) * p1 + ord(s[i + l - 1])) % md,
        (ord(s[i]) * p2^(l - 1) + ord(s[i + 1]) * p2^(l - 2) + ... + ord(s[i + l - 2]) * p2 + ord(s[i + l - 1])) % md,
        ...,
        (ord(s[i]) * pn^(l - 1) + ord(s[i + 1]) * pn^(l - 2) + ... + ord(s[i + l - 2]) * pn + ord(s[i + l - 1])) % md)
    
    Args:
        Required positional:
        s (iterable object): A finite ordered iterable object (e.g. a
                string) for which the rolling hash is to be generated.
        length (int): Strictly positive integer giving the lengths
                of the contiguous subsequences over which the rolling
                has is to be generated.
        
        Optional named:
        p_lst (List/tuple of ints): A collection of prime numbers for
                which the polynomial-based hashes are to be evaluated.
                The larger and more numerous the primes given, the
                less likely hash collisions are to occur (i.e. that
                two different sequences give rise to the same hash
                value), though each additional prime proportionally
                increases the evaluation time.
            Default: (37, 53)
        md (int): Strictly positive integer giving the modulus the hash
                values are to be taken (i.e. each hash is returned as
                its remainder to this modulus). For best results, this
                should be a prime number, and the larger the number
                the less likely hash collisions are to occur (see p_lst).
        encoding_func (callable): A function taking in an element of s and
                returning an integer, or for integer or string sequences
                None can be given (for integer sequences, the value is
                used directly, and for strings, the ASCII code number is
                used). To avoid hash collisions, this function should be
                injective (i.e. there should be no two distinct elements
                for which the result of applying the function is the
                same).
            Default: None
    
    Yields:
    An len(p_lst)-tuple of integers (int), each between 0 and (md - 1)
    inclusive, where index j (0 <= j < len(p_lst)) of the (i + 1)th yielded
    tuple is the rolling hash value for the input length, the prime
    p_lst[j] and the modulus md, as calculated above, for the contiguous
    subsequence of s between indices i and (i + length - 1) inclusive,
    where each element has been converted into an integer by the function
    encoding_func.
    The number of results yielded by the generator is one more than the
    number of elements in s minus length as long as this is positive,
    otherwise there are no yielded results
            
    Modified version of rolling hash can be used to solve Leetcode #1554
    (Premium).
    """
    if hasattr(s, "__len__") and len(s) < length:
        return
    if encoding_func is None:
        try: val = encoding_func(next(iter_obj))
        except StopIteration: return
        if isinstance(next(iter(s), str)):
            encoding_func = ord#lambda x: ord(x)
        else: encoding_func = lambda x: x
    iter_obj = iter(s)
    n_p = len(p_lst)
    hsh = [0] * n_p
    val_qu = deque()
    for i in range(length - 1):
        try: l = next(iter_obj)
        except StopIteration: return
        val = encoding_func(l)
        val_qu.append(val)
        for j, p in enumerate(p_lst):
            hsh[j] = (hsh[j] * p + val) % md
        yield (l, None)
    try: l = next(iter_obj)
    except StopIteration: return
    val = encoding_func(l)
    val_qu.append(val)
    for j, p in enumerate(p_lst):
        hsh[j] = (hsh[j] * p + val) % md
    yield (l, tuple(hsh))
    mults = [pow(p, length - 1, md) for p in p_lst]
    for i in itertools.count(length):
        try: l = next(iter_obj)
        except StopIteration: return
        val = encoding_func(l)
        val_qu.append(val)
        val2 = val_qu.popleft()
        for j, p in enumerate(p_lst):
            hsh[j] = ((hsh[j] - mults[j] * val2) * p + val) % md
        yield (l, tuple(hsh))
    return


def rollingHashSearch(
    s: Iterable[Any],
    patterns: List[Iterable[Any]],
    p_lst: Optional[Union[List[int], Tuple[int]]]=(31, 37),
    md: int=10 ** 9 + 7,
    check: bool=True,
) -> Dict[str, List[int]]:
    """
    Performs a rolling hash search on the ordered iterable object s to
    find the starting indices of all occurrences of contiguous ordered
    sub-iterable object that exactly match one of the finite ordered
    iterable objects in patterns (even if those occurrences overlap with
    each other in s).
    For details regarding the calculation of the rolling hash, see
    rollingHash().
    This is best suited for cases where many of the objects in
    patterns share lengths.

    Args:
        Required positional:
        s (iterable): The ordered iterable object to be searched
        patterns (list of str): List of finite ordered iterable objects
                whose starting indices in s are to be found.
        
        
        p_lst (list/tuple of ints): A list of distinct prime numbers to be
                used in the calculation of the rolling hash (see rollingHash()
                documentation for details). The larger and more numerous
                the primes given, the less likely hash collisions are to
                occur (i.e. that two different sequences give rise to the
                same hash value), though each additional prime proportionally
                increases the evaluation time.
            Default: (31, 37)
        md (int): The modulus used for the rolling hash (see rollingHash()
                documentation for details). This should be a large prime
                number.
            Default: 10 ** 9 + 7
        check (bool): Whether in the event of a hash match with a single
                matching hash for one of the elements of patterns of the
                correct length, a check is performed to rule out a
                hash collision, otherwise the match is assumed to be
                correct (note that by increasing the value of md, and
                using more and larger primes the chances that this could
                result in an error can be made arbitrarily small, but there
                is always a chance this could result in an error in the
                form of a false inclusion. On the other hand, for cases
                where a large numbers of pattern matches are present, not
                performing the check can speed up the execution
                significantly)
            Default: True
    
    Returns:
    Dictionary (dict) whose keys are the elements of patterns present as
    a contiguous sub-iterable object of s, and whose corresponding values are a
    list of the (0-indexed) indices of s at which the contiguous sub-iterable
    objects that element of patterns start, in increasing order.
    """
    #ord_A = ord("A")
    def char2num(l: str) -> int:
        return ord(l)# - ord_A
    
    pattern_dict = {}
    for pattern in patterns:
        length = len(pattern)
        pattern_dict.setdefault(length, {})
        hsh = next(rollingHash(pattern, length, p_lst=p_lst, md=md, func=char2num))
        pattern_dict[length].setdefault(hsh, set())
        pattern_dict[length][hsh].add(pattern)
    
    res = {}
    for length, hsh_dict in pattern_dict.items():
        for i, hsh in enumerate(rollingHash(s, length, p_lst=p_lst, md=md, func=char2num)):
            if hsh not in hsh_dict.keys(): continue
            if not check and len(hsh_dict[hsh]) == 1:
                pattern = next(iter(hsh_dict[hsh]))
            else:
                pattern = s[i: i + length]
                if pattern not in hsh_dict[hsh]: continue
            res.setdefault(pattern, [])
            res[pattern].append(i)
    return res

def findRepeatedDnaSequences(
    s: str,
    substring_length: int=10,
) -> List[str]:
    """
    For a string s representing a DNA sequence consisting exclusively of
    the characters "A", "C", "G" and "T", finds the set of contiguous
    substrings in s with length substring_length that appear more than
    once in the s (including if the matching substrings overlap with each
    other), including each such substring in the returned list exactly
    once. The returned substrings are not ordered.

    This illustrates the use of a rolling hash for string matching.

    Args:
        Required positional:
        s (str): String representing the DNA sequence consisting exclusively
                of the characters "A", "C", "G" and "T" that is to be
                searched for repeating contiguous substrings.
        
        Optional named:
        substring_length (int): Strictly positive integer giving the length
                of repeating contiguous substrings to be identified.
            Default: 10
    
    Returns:
    List of strings (str) containing all strings of length substring_length
    that appear as contiguous substrings of s more than once. This list is
    not ordered and has no repeated elements.

    Examples:
        >>> findRepeatedDnaSequences("AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT", substring_length=10)
        ["AAAAACCCCC","CCCCCAAAAA"]

        This signifies that the only contiguous substrings of length 10 that
        appear more than once in s are "AAAAACCCCC" (starting at indices 0
        and 10 in s, where 0-indexing is used) and "CCCCCAAAAA" (starting at
        indices 6 and 16 in s).

        >>> findRepeatedDnaSequences("AAAAAAAAAAAAA", substring_length=10)
        ["AAAAAAAAAA"]

        This signifies that the only contiguous substrings of length 10 that
        appear more than once in s are "AAAAAAAAAA" (starting at indices 0, 1,
        2 and 3 in s, where 0-indexing is used). Note that these count as
        repeated substrings even though they all overlap with each other.

    Solution to Leetcode #187 using rolling hash (with
    substring_length equal to 10)

    Original Leetcode #187 problem description

    The DNA sequence is composed of a series of nucleotides abbreviated
    as 'A', 'C', 'G', and 'T'.

    For example, "ACGAATTCCG" is a DNA sequence.
    When studying DNA, it is useful to identify repeated sequences within
    the DNA.

    Given a string s that represents a DNA sequence, return all the 10-letter-long
    sequences (substrings) that occur more than once in a DNA molecule. You may
    return the answer in any order.
    """

    # Note- hash guaranteed to fit inside unsigned int (4 bytes/32 bit) since
    # the maximum value it takes is (3 * (4 ** 11 - 1) / (4 - 1)) = (2 ** 22 - 1)
    # (corresponding to 10 "T"s in a row) which is strictly less than 2 ** 31,
    # as is required for an unsigned int

    alpha = "ACGT"
    alpha_dict = {x: i for i, x in enumerate(alpha)}
    alpha_sz = len(alpha)

    s_set = set(s)
    if not s_set.issubset(alpha_dict.keys()):
        raise ValueError("s can only contain the characters 'A', 'C', 'G' and 'T'")

    def char2num(l: str) -> int:
        return alpha_dict[l]
    
    md = 10 ** 9 + 7
    p_lst = (31, 37)
    hsh_dict = {}
    res = []
    for i, hsh in enumerate(rollingHash(s, substring_length, p_lst=p_lst, md=md, func=char2num)):
        if hsh in hsh_dict.keys():
            if not hsh_dict[hsh]:
                res.append(s[i: i + substring_length])
                hsh_dict[hsh] = True
        else: hsh_dict[hsh] = False
    return res

class AhoCorasick(object):
    """
    Data structure used for simultaneous matching of multiple
    patterns in an ordered finite iterable object consisting of
    hashable objects (e.g. a string), with time complexity
    O(n + m + z) where n is the length of (i.e. the number of
    elements in) the iterable object being searched, m is the sum
    of the lengths of the patterns and z is the total number of
    matches over all of the patterns in the string.

    This is structured the form of a trie, with additional connections
    between nodes in the trie for search failures.
    
    Initialization args:
        Required positional:
        words (list of ordered, finite iterables containing hashable
                objects): The patterns that are to be found in the
                samples given to this object.
        
    Attributes:
        words (list of ordered, finite iterables containing hashable
                objects): The patterns that are to be found in the
                samples given to this object.
        goto (list of dicts): Representation of the trie, with each
                entry representing a node in the trie. The dictionary
                has keys representing possible hashable objects that
                can appear in the object being searched, with the
                corresponding value being the index of goto to travel
                to if the next object encountered in the search
                equals the key.
        failure (list of ints): A list the same length as that of
                goto. In the event that during a search, the next
                object encountered does not appear as a key in the
                current node's entry in goto, the corresponding entry
                (i.e. the entry with the same index) in this list
                indicates which entry in goto to travel to next.
        out (list of ints): A list the same length as that of goto.
                This contains a bitmask indicating the indices in
                the attribute words of the patterns that end at
                the corresponding node in the trie.
        out_lens (list of ints): A list the same length as that of
                goto. This contains a bitmask indicating the lengths
                of the patterns that end at the corresponding node in
                the trie.
    
    Methods:
        (For more details about a specific method see the documentation
        for that method)
        buildAutomaton(): Constructs the modified trie (including the
                attributes goto, failure, out and out_lens) based on
                the attribute words.
        search(): Finds the starting index for every matching occurrence
                of every element in words for a given ordered, finite
                iterable object.
        searchEndIndices(): For a given ordered, finite iterable object,
                creates a generator that goes through the elements of
                the object in order, and for each index one or more
                occurrence of an element in words ends, yields the
                index in the object being searched and the indices in
                words of the elements with matches ending at that index.
        searchLengths(): For a given ordered, finite iterable object,
                creates a generator that goes through the elements of
                the object in order, and for each index one or more
                occurrence of an element in words ends, yields the
                index in the object being searched and the lengths of
                the elements of words elements with matches ending at
                that index.
                
    Can use for solution of Leetcode: #139, #140 and Premium Leetcode:
    #616 and #758 (basically the same problem) and #1065
    """

    def __init__(self, words: List[Iterable[Hashable]]):
        self.goto = [{}]
        self.failure = [-1]
        self.out = [0]
        self.out_lens = [0]
        self.words = words
        self.buildAutomaton()

    def buildAutomaton(self) -> None:
        """
        Constructs the trie and failure connections for the finite ordered
        iterable objects in the attribute words. This initialises the
        attributes goto, failure, out and out_lens.

        Args:
            None
        
        Returns:
            None
        """
        for i, w in enumerate(self.words):
            j = 0
            for l in w:
                if l not in self.goto[j].keys():
                    self.goto[j][l] = len(self.goto)
                    self.goto.append({})
                    self.failure.append(0)
                    self.out.append(0)
                    self.out_lens.append(0)
                j = self.goto[j][l]
            self.out[j] |= 1 << i
            self.out_lens[j] |= 1 << len(w)
        
        queue = deque(self.goto[0].values())
        
        while queue:
            j = queue.popleft()
            for l, j2 in self.goto[j].items():
                j_f = self.failure[j]
                while j_f and l not in self.goto[j_f].keys():
                    j_f = self.failure[j_f]
                j_f = self.goto[j_f].get(l, 0)
                self.failure[j2] = j_f
                self.out[j2] |= self.out[j_f]
                self.out_lens[j2] |= self.out_lens[j_f]
                queue.append(j2)
        return
    
    def _findNext(self, j: int, l: Hashable) -> int:
        while j and l not in self.goto[j].keys():
            j = self.failure[j]
        return self.goto[j].get(l, 0)
    
    def search(
        self,
        s: Iterable[Hashable],
    ) -> Dict[int, List[int]]:
        """
        Gives dictionary for the starting index of each occurrence
        of each element of the attribute words as a contiguous
        subsequence in the finite ordered iterable object s.

        Args:
            Required positional:
            s (iterable object): The finite ordered iterable object
                    with hashable elements being searched.
        
        Returns:
        Dictionary whose keys each give the index (0-indexed) of a
        pattern in the attribute words that appears as a contiguous
        subsequence in s with the corresponding value giving a list
        containing all the indices in s (0-indexed) at which all the
        matching contiguous subsequences in s, start in strictly
        increasing order.
        """
        j = 0
        res = {}
        for i, l in enumerate(s):
            j = self._findNext(j, l)
            bm = self.out[j]
            for idx, w in enumerate(self.words):
                if not bm: break
                if bm & 1:
                    res.setdefault(w, [])
                    res[i].append(i - len(w) + 1)
                bm >>= 1
        return res
    
    def searchEndIndices(
        self,
        s: Iterable[Hashable],
    ) -> Generator[Tuple[int, List[int]], None, None]:
        """
        Generator yielding a 2-tuple of each index of s (in ascending order)
        and a list of the corresponding indies of the patterns in self.words
        that have a match in s that ends exactly at that index of s.

        Args:
            Required positional:
            s (iterable object): The finite ordered iterable object
                    with hashable elements being searched.
        
        Yields:
        A 2-tuple whose index 0 contains an index (0-indexed) in s at which at
        least one contiguous substring of s is equal to a pattern in the
        attribute words and whose index 1 contains a list with the indices
        in the attribute words (0-indexed) in strictly increasing order of all
        the patterns that are equal to a contiguous subsequence of s ending at
        that index.
        These are yielded in strictly increasing order with respect to the
        indices (i.e. index 0 or the tuple) and all matches are yielded.
        """
        j = 0
        for i, l in enumerate(s):
            j = self._findNext(j, l)
            bm = self.out[j]
            idx = 0
            res = []
            while bm:
                if bm & 1: res.append(idx)
                idx += 1
                bm >>= 1
            yield (i, res)
        return

    def searchLengths(
        self,
        s: Iterable[Hashable],
    ) -> Generator[Tuple[int], None, None]:
        """
        Generator yielding a 2-tuple of each index of s (in ascending order)
        and a list of the lengths of the patterns in self.words that have a
        match in s that ends exactly at that index of s.

        Args:
            Required positional:
            s (iterable object): The finite ordered iterable object
                    with hashable elements being searched.
        
        Yields:
        A 2-tuple whose index 0 contains an index (0-indexed) in s at which at
        least one contiguous substring of s is equal to a pattern in the
        attribute words and whose index 1 contains a list with the lengths of
        all the patterns in the attribute words that are equal to a contiguous
        subsequence of s ending at that index in strictly increasing order.
        These are yielded in strictly increasing order with respect to the
        indices (i.e. index 0 or the tuple) and all matches are represented.
        """
        j = 0
        for i, l in enumerate(s):
            j = self._findNext(j, l)
            bm = self.out_lens[j]
            length = 0
            res = []
            while bm:
                if bm & 1: res.append(length)
                length += 1
                bm >>= 1
            yield res
        return
    
def wordBreak(
    s: str,
    wordDict: List[str],
) -> bool:
    """
    Function identifying whether the string s can be partitioned into
    contiguous substrings such that each substring appears in
    wordDict. There is no restriction on how many times an element
    of wordDict can appear in a partitioning.

    Args:
        Required positional:
        s (str): The string to be partitioned
        wordDict (list of strs): The strings from which each substring
                must come for a successful partitioning of s into
                contiguous substrings.
    
    Returns:
    Boolean (bool) giving True if such a partitioning of s into
    contiguous substrings is possible, otherwise False.

    Examples:
        >>> wordBreak("leetcode", wordDict = ["leet", "code"])
        True

        This signifies that "leetcode" can be partitioned into
        contiguous substrings such that each substring is in the
        list ["leet", "code"]. Such a partitioning (indeed the
        only possible partitioning) is "leet" "code".

        >>> wordBreak("applepenapple", ["apple", "pen"])
        True

        This signifies that "applepenapple" can be partitioned into
        contiguous substrings such that each substring is in the
        list ["apple","pen"]. Such a partitioning (indeed the only
        possible such  partitioning) is "apple" "pen" "apple". Note
        that "apple" appears twice despite only appearing once in
        the list, which is permitted.

        >>> wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"])
        False

        This signifies that there are no partitionings of "catsandog"
        such that each substring is in the list:
            ["cats", "dog", "sand", "and", "cat"]
        This is despite each character in the string being part of
        a contiguous substring present in the list. However, any choice
        of these substrings has at least one overlap or gap between the substrings
        (i.e. one character in the string that is either in more than one
        or none of the substrings respectively), meaning it is not a
        partitioning. 

    Solution to Leetcode #139 (Word Break) using Aho Corasick

    Original problem description for Leetcode #139:

    Given a string s and a dictionary of strings wordDict, return true if
    s can be segmented into a space-separated sequence of one or more
    dictionary words.

    Note that the same word in the dictionary may be reused multiple times
    in the segmentation.
    """
    ac = AhoCorasick(wordDict)
    arr = [False] * len(s)
    for i, lengths in enumerate(ac.searchLengths(s)):
        for j in lengths:
            if j == i + 1: arr[i] = True
            elif j > i + 1: break
            elif arr[i - j]: arr[i] = True
            if arr[i]: break
    return arr[-1]

def wordBreak2(
    s: str,
    wordDict: List[str],
) -> List[str]:
    """
    Function identifying all the ways the string s can be partitioned
    into contiguous substrings such that each substring appears in
    wordDict. There is no restriction on how many times an element
    of wordDict can appear as a substring in a partitioning.

    Args:
        Required positional:
        s (str): The string to be partitioned
        wordDict (list of strs): The strings from which each substring
                must come for a successful partitioning of s into
                contiguous substrings.
    
    Returns:
    List of strings (str), each giving a partitioning of s into contiguous
    substrings such that each substring is an element of wordDict, and
    between them giving all such possible partitionings. The borders between
    the partitions in a given partitioning is signified by a space. There
    order of in which the partitionings are given has no special
    significance.

    Examples:
        >>> wordBreak2("leetcode", ["leet", "code"])
        ['leet code']

        >>> wordBreak2("catsanddog", ["cat", "cats", "and", "sand", "dog"])
        ['cats and dog', 'cat sand dog']

        >>> wordBreak2("pineapplepenapple", ["apple", "pen", "applepen", "pine", "pineapple"])
        ['pine apple pen apple', 'pineapple pen apple', 'pine applepen apple'])

        >>> wordBreak2("catsandog", ["cats", "dog", "sand", "and", "cat"])
        []

        The returned empty list indicates that there are no partitionings of
        "catsandog" into contiguous substrings such that each substring is in
        the list:
            ["cats", "dog", "sand", "and", "cat"]

    Solution to Leetcode #140 (Word Break II) using Aho Corasick

    Original problem description for Leetcode #140:

    Given a string s and a dictionary of strings wordDict, add spaces in
    s to construct a sentence where each word is a valid dictionary word.
    Return all such possible sentences in any order.

    Note that the same word in the dictionary may be reused multiple
    times in the segmentation.
    """
    n = len(s)
    ac = AhoCorasick(wordDict)
    dp = [[] for _ in range(n)]
    for (i, inds) in ac.searchEndIndices(s):
        for j in inds:
            w = wordDict[j]
            if len(w) == i + 1:
                dp[i].append(w)
                continue
            for s2 in dp[i - len(w)]:
                dp[i].append(" ".join([s2, w]))
    return dp[-1]

def addBoldTag(s: str, words: List[str]) -> str:
    """
    Solution to Leetcode #616 and #758 (both Premium) using Aho Corasick
    """
    # Try to make faster
    
    # Using Aho-Corasick automaton

    n = len(s)
    if not n: return s
    ac = AhoCorasick(words)
    start_dict = ac.search(s)
    if not start_dict: return s
    rngs = []
    #print(start_dict)
    words = list(start_dict.keys())
    for i, w in enumerate(words):
        rngs.append([])
        length = len(w)
        for j in start_dict[i]:
            if not rngs[i] or j > rngs[i][-1][1]:
                rngs[i].append([j, j + length])
            else: rngs[i][-1][1] = j + length
    #print(rngs)
    heap = [[rng_lst[0][0], -rng_lst[0][1], 0, idx] for idx, rng_lst in enumerate(rngs)]
    heapq.heapify(heap)

    res = []
    i1, neg_i2, j, idx = heapq.heappop(heap)
    if i1: res.append(s[:i1])
    res.append("<b>")
    bs_i = i1 # index for start of current bold
    be_i = -neg_i2 # index for current end of current bold
    if j + 1 < len(rngs[idx]):
        heapq.heappush(heap, [rngs[idx][j + 1][0], -rngs[idx][j + 1][1], j + 1, idx])
    while heap:
        i1, neg_i2, j, idx = heapq.heappop(heap)
        i2 = -neg_i2
        if i1 <= be_i:
            be_i = max(be_i, i2)
            if be_i == n: break
        else:
            res.append(s[bs_i: be_i])
            res.append("</b>")
            res.append(s[be_i: i1])
            res.append("<b>")
            bs_i = i1
            be_i = i2
            if be_i == n: break
        if j + 1 < len(rngs[idx]):
            heapq.heappush(heap, [rngs[idx][j + 1][0], -rngs[idx][j + 1][1], j + 1, idx])
    res.append(s[bs_i: be_i])
    res.append("</b>")
    if be_i < n: res.append(s[be_i:])
    return "".join(res)

def manacherAlgorithm(s: Iterable[Any]) -> List[int]:
    """
    Implementation of Manacher's algorithm to find the 1D
    array or iterable (e.g. string) with the same size as s whose
    value for each index is half the length (rounded down) of the
    longest odd-length palindromic contiguous subsequence centred
    on the corresponding element of s.

    Args:
        Required positional:
        s (iterable): An ordered array to be processed

    Returns:
    A list of integers (int) with length equal to that of s, whose
    element at index i (0-indexed) gives half the length rounded
    down of the longest odd-length contiguous palindromic subsequence
    of s centred on that index.

    Examples:
        >>> manacherAlgorithm("ebabad")
        [0, 0, 1, 1, 0, 0]

        This signifies for instance that the longest odd-length
        palindromic contiguous subsequence centred on the character
        at index 2 (zero-indexed, so the character "a") has a rounded
        down half length of 1, i.e. a full length of 3, which
        corresponds to the substring from indices 1 to 3 inclusive,
        "bab".

        >>> manacherAlgorithm(['#', 'e', '#', 'b', '#', 'a', '#', 'b', '#', 'a', '#', 'a', '#', 'b', '#', 'd', '#'])
        [0, 1, 0, 1, 0, 3, 0, 3, 0, 1, 4, 1, 0, 1, 0, 1, 0]

        This signifies for instance that the longest odd-length
        palindromic contiguous subsequence centred on the character
        at index 5 (zero-indexed, so the element 'a') has a rounded
        down half length of 3, i.e. a full length of 7, which
        corresponds to the contiguous subsequence from indices 2 to
        8 inclusive, ['#', 'b', '#', 'a', '#', 'b', '#'], and the
        longest odd-length palindromic contiguous subsequence centred
        on the character at index 10 (zero-indexed, so the element '#'
        between the two 'a' elements) has a rounded down half length
        of 4, i.e. a full length of 9, which corresponds to the
        contiguous subsequence from indices 6 to 14 inclusive,
        ['#', 'b', '#', 'a', '#', 'a', '#', 'b', '#'].
        This example demonstrates how by preprocessing the original
        array with identical wild characters between each element
        and at each end, Manacher's algorithm can find the lengths
        of any palindromic contiguous subsequences (i.e. both odd
        and even length), with those centred on the wild characters
        being the even length palindromes. Indeed, the value actually
        gives the exact length of the corresponding palindrome in the
        original array.
    """
    n = len(s)
    curr_centre = 0
    curr_right = 0
    max_val = 0
    max_i = 0
    res = [0] * n
    for i in range(n):
        if i < curr_right:
            mirror = (curr_centre << 1) - i
            res[i] = min(res[mirror], curr_right - i)
        while i - res[i] - 1 >= 0 and i + res[i] + 1 < len(s) and\
                s[i + res[i] + 1] == s[i - res[i] - 1]:
            res[i] += 1
        if res[i] > max_val:
            max_val = res[i]
            max_i = i
        if i + res[i] > curr_right:
            curr_right = i + res[i]
            curr_centre = i
    return res

def longestPalindromicSubstrings(s: str) -> Tuple[int, List[int]]:
    """
    For a given string s, identifies the length such that there exists
    at least one palindromic string of that length that is a contiguous
    substring of s and there are no longer such substrings, and the
    start indices in s at which each such string appears as a substring.

    A string is palindromic if and only if it is equal to itself
    reversed.

    This demonstrates a use of Manacher's algorithm.

    Args:
        Required positional:
        s (str): The string to be searched for the palindromic
                contiguous substrings.

    Returns:
    A 2-tuple whose index 0 contains and integer (int) giving the length
    of the longest palindromic contiguous substring of s and whose index 1
    is a list of integers (int) giving the 0-indexed starting indices in s
    of all the longest palindromic contiguous substrings of that length,
    in strictly increasing order.

    Example:
        >>> longestPalindromicSubstrings("ebabadbab")
        (3, [1, 2, 6])

        This signifies that the longest a palindromic contiguous substring
        of "ebabadbab" can be is 3, of which there are three occurrences,
        starting at the 0-indexed indices 1 ("bab"), 2 ("aba") and 6
        ("bab"). Note that two of these substrings are the same string,
        and the indices of both occurrences are included.
    """
    if not s: return (0, [0])
    n = len(s)
    s2 = ["#"] * ((n << 1) + 1)
    for i, l in enumerate(s):
        s2[(i << 1) + 1] = l
    manacher_arr = manacherAlgorithm(s2)
    mx_len = -1
    mx_i = []
    for i, num in enumerate(manacher_arr):
        if num < mx_len: continue
        elif num > mx_len:
            mx_len = num
            mx_i = []
        mx_i.append(i)
    res = []
    for i in mx_i:
        j1 = (i - mx_len) >> 1
        res.append(j1)
    return (mx_len, res)

def longestPalindrome(s: str) -> str:
    """
    Uses Manacher's algorithm to find the unique palindromic
    substring of s for which there are no longer palindromic
    substrings in s and there are no palindromic substrings
    of s with the same length and a smaller starting index
    in s.

    Example:
        >>> longestPalindrome("ebabad")
        "bab"

    Solution to Leetcode #5- Longest Palindromic Substring
    
    Original problem description:
    
    Given a string s, return the longest palindromic substring in s.
    """
    return longestPalindromicSubstrings(s)[0]

def countPalindromicSubstrings(s: str) -> int:
    """
    
    Solution to Leetcode #647- Palindromic Substrings
    
    Original problem description:
    
    Given a string s, return the number of palindromic substrings in
    it.

    A string is a palindrome when it reads the same backward as
    forward.

    A substring is a contiguous sequence of characters within the
    string.
    """
    n = len(s)
    s2 = ["#"] * ((n << 1) + 1)
    for i, l in enumerate(s):
        s2[(i << 1) + 1] = l
    manacher_arr = manacherAlgorithm(s2)
    return sum((x + 1) >> 1 for x in manacher_arr)

if __name__ == "__main__":
    res = wordBreak("leetcode", wordDict = ["leet", "code"])
    print(f"\nwordBreak(\"leetcode\", wordDict = [\"leet\", \"code\"]) = {res}")

    res = wordBreak("applepenapple", ["apple", "pen"])
    print(f"\nwordBreak(\"applepenapple\", [\"apple\", \"pen\"])= {res}")

    res = wordBreak("catsandog", ["cats", "dog", "sand", "and", "cat"])
    print(f"\nwordBreak(\"catsandog\", [\"cats\", \"dog\", \"sand\", \"and\", \"cat\"])) = {res}")

    res = wordBreak2("leetcode", ["leet", "code"])
    print(f"\nwordBreak2(\"leetcode\", [\"leet\", \"code\"]) = {res}")

    res = wordBreak2("catsanddog", ["cat", "cats", "and", "sand", "dog"])
    print(f"\nwordBreak2(\"catsanddog\", [\"cat\", \"cats\", \"and\", \"sand\", \"dog\"]) = {res}")

    res = wordBreak2("pineapplepenapple", ["apple", "pen", "applepen", "pine", "pineapple"])
    print(f"\n\"pineapplepenapple\", [\"apple\", \"pen\", \"applepen\", \"pine\", \"pineapple\"]) = {res}")

    res = wordBreak2("catsandog", ["cats", "dog", "sand", "and", "cat"])
    print(f"\n\"catsandog\", [\"cats\", \"dog\", \"sand\", \"and\", \"cat \"]) = {res}")

    res = longestPalindrome("ebabad")
    print(f"\nlongestPalindrome(\"ebabad\") = {res}")

    res = longestPalindromicSubstrings("ebabadbab")
    print(f"\nlongestPalindromicSubstrings(\"ebabadbab\") = {res}")

    res = findRepeatedDnaSequences(
        "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT",
        substring_length=10
    )
    print("\nfindRepeatedDnaSequences(\"AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT\", substring_length=10) = "
            f"{res}")

    res = findRepeatedDnaSequences(
        "AAAAAAAAAAAAA",
        substring_length=10
    )
    print("\nfindRepeatedDnaSequences(\"AAAAAAAAAAAAA\", substring_length=10) = "
            f"{res}")