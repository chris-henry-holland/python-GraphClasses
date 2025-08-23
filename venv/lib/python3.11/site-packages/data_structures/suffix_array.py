#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Union,
)

import bisect

from sortedcontainers import SortedList
from collections import deque

class SuffixArray:
    # Using SA-IS algorithm constructed based on
    # https://zork.net/~st/jottings/sais.html
    """
    Class the constructs and utilises a suffix array for a given
    string, including creation of the longest common prefix (LCP)
    array and methods allowing it to be used to search the string.

    The suffix array for a given string is an array with length one
    greater than that of the string, storing the start indices of
    all the suffixes of the string (including the whole string and
    the empty suffix) in alphabetical order of the suffixes. If
    one suffix is equal to the prefix of a second, the first
    appears earlier in the suffix array (as such, the empty suffix
    is necessarily the first element in the suffix array).
    
    The longest common prefix (LCP) array for a given string and
    its suffix array is an array the same length as the suffix
    array denoting the longest common prefix between that suffix
    and the suffix that appears next in the suffix array.
    
    Construction of this array allows for efficient searching of
    the string for matching patterns.

    Initialization args:
        Required positional:
        s (str): The string for which the suffix array is to be
                constructed.
        
        Optional named:
        head_chars (str): a string containing characters that are
                considered to be alphabetically first, with
                the earlier the first occurrence of a character
                in head_chars appears the earlier it is considered
                to be alphabetically. This is included to
                facilitate the use of partition characters, for
                instance in the longest common substring problem.
            Default: "" (the empty string)
    
    Attributes:
        s (str): The string on which the suffix array is based.
        n (int): The length of the string s.
        suff_arr (list of ints): The suffix array of s, an array
                of length one greater than that of s (i.e. n + 1).
        lcp (list of ints): The longest common prefix array of
                s, an array of length equal to that of suff_arr.
        lcp_lr (list of ints): The LCP-LR array of s. An array whose
                length is the largest power of 2 not exceeding
                the length of s. This pre-computes the longest
                common prefix between suffixes that are not adjacent
                to each other in the suffix array, with the choice
                of the pairs matching that of a binary search of
                the suffix array, in order to facilitate that search.
    
    Methods:
        (For details of a specific method, see the documentation of
        that method)
        encodeChars(): Translates the characters included in s and
                head_chars into unique positive integers, such that
                the earlier alphabetically a character is, the
                smaller it is (accounting for the characters in
                head_chars being alphabetically first).
        compareCharacters(): Performs a comparison between two
                characters to determine their relative alphabetical
                ordering (accounding for the characters in
                head_chars being alphabetically first)
        buildFrequencyArrays(): TODO
        buildLSArrayAndLMS(): Creates the L-type S-type (LS)
                array and left-most S-type suffix (LMS) arrays,
                each used in the construction of the suffix array
        buildSuffixArray(): Builds the suffix array
        buildLongestCommonPrefixArray(): Builds the longest common
                prefix (LCP) array using the suffix array
        checkLCP(): Checks the correctness of the calculated
                LCP array
        buildLCPLR(): Builds the longest common prefix long range
                (LCP-LR) array
        search(): Uses the suffix, LCP and LCP-LR arrays to
                search for a particular contiguous substring in
                the attribute s, utilising a binary search
                technique.
    """
    def __init__(self, s: str, head_chars: str=""):
        self._s = s
        self._head_chars = head_chars
    
    @property
    def s(self):
        return self._s if self._s is not None else ""
    
    @property
    def n(self):
        return len(self.s)
    
    @property
    def head_chars(self):
        return self._head_chars if self._head_chars is not None else ""
    
    @property
    def head_char_ordering(self):
        res = getattr(self, "_head_char_ordering", None)
        if res is None:
            res = self._calculateHeadCharOrdering(self.head_chars)
            self._head_char_ordering = res
        return res
    
    @property
    def suff_arr(self):
        res = getattr(self, "_suff_arr", None)
        if res is None:
            res = self.buildSuffixArray()
            self._suff_arr = res
        return res
    
    @property
    def lcp(self):
        res = getattr(self, "_lcp", None)
        if res is None:
            res = self.buildLongestCommonPrefixArray()
            self._lcp = res
        return res
    
    @property
    def lcp_lr(self):
        res = getattr(self, "_lcp_lr", None)
        if res is None:
            res = self.buildLCPLR(self.lcp)
            self._lcp_lr = res
        return res

    def encodeChars(self, s: str, head_chars: str="") -> Dict[str, int]:
        chars = set(s)
        res = {}
        i = 1
        for l in head_chars:
            if l in res.keys(): continue
            res[l] = i
            i += 1
            chars.discard(l)
        for i, l in enumerate(sorted(chars), start=i):
            res[l] = i
        return res
    
    def _calculateHeadCharOrdering(self, head_chars: str) -> Dict[str, int]:
        res = {}
        i = 1
        for l in head_chars:
            if l in res.keys(): continue
            res[l] = i
            i += 1
        return res
    
    def compareCharacters(self, l1: str, l2: str) -> int:
        # Returns 0 if l1 and l2 are equal, 1 if l1 is lexicographically
        # smaller than l2 and -1 otherwise
        if l1 == l2: return 0
        hco = self.head_char_ordering
        if l1 in hco.keys():
            return 1 if self.hco[l1] < self.hco.get(l2, float("inf")) else -1
        elif l2 in hco.keys(): return -1
        return 1 if l1 < l2 else -1

    def buildFrequencyArrays(
        self,
        nums: List[int],
        nums_mx: int,
    ) -> Tuple[List[int], List[int]]:
        f_arr = [0] * (nums_mx + 1)
        for num in nums:
            f_arr[num] += 1
        cumu_arr = [0] * (len(f_arr) + 1)
        for i, f in enumerate(f_arr):
            cumu_arr[i + 1] = cumu_arr[i] + f
        return (f_arr, cumu_arr)
    
    def buildLSArrayAndLMS(
        self,
        nums: List[int],
    ) -> Tuple[Union[List[bool], List[int]]]:
        n = len(nums)
        arr = [True] * (n + 1)
        lms = []
        num1 = 0
        for i in reversed(range(n)):
            num1, num2 = nums[i], num1
            if num1 > num2 or (num1 == num2 and not arr[i + 1]):
                arr[i] = False
                if arr[i + 1]: lms.append(i + 1)
        return (arr, lms[::-1])
    
    def buildSuffixArray(self) -> List[int]:

        def induceSortLS(
            nums: List[int],
            suff_arr: List[int],
            ls_arr: List[bool],
            cumu_arr: List[int],
        ) -> None:
            for (curr_inds, iter_obj, incr) in\
                    (([cumu_arr[i] for i in range(len(cumu_arr) - 1)],\
                    range(len(suff_arr)), 1),\
                    ([cumu_arr[i] - 1 for i in range(1, len(cumu_arr))],\
                    reversed(range(len(suff_arr))), -1)):
                b = (incr == 1)
                for i in iter_obj:
                    suff_idx = suff_arr[i] - 1
                    if suff_idx < 0 or ls_arr[suff_idx] is b:
                        continue
                    char_idx = nums[suff_idx]
                    suff_arr[curr_inds[char_idx]] = suff_idx
                    curr_inds[char_idx] += incr
            return

        encoding = self.encodeChars(self.s, head_chars=self.head_chars)
        #print(encoding)
        nums = [encoding[l] for l in self.s]
        nums.append(0)

        def recur(nums: List[int], nums_mx: int) -> List[int]:
            n = len(nums) - 1
            if nums_mx == n:
                res = [n] * (n + 1)
                for idx, num in enumerate(nums):
                    res[num] = idx
                return res
            cumu_arr = self.buildFrequencyArrays(nums, nums_mx)[1]
            ls_arr, lms_inds = self.buildLSArrayAndLMS(nums)

            curr_inds = [cumu_arr[i] - 1 for i in range(1, len(cumu_arr))]
            suff_arr = [-1] * (n + 1)
            for i in lms_inds:
                char_idx = nums[i]
                suff_arr[curr_inds[char_idx]] = i
                curr_inds[char_idx] -= 1
            induceSortLS(nums, suff_arr, ls_arr, cumu_arr)
            lms_dict = {lms_inds[i]: i for i in range(len(lms_inds) - 1)}
            name_dict = {lms_inds[-1]: 1}
            curr_name = 1
            prev_inds = (n, n)
            for suff_arr_idx in range(1, n + 1):
                idx1 = suff_arr[suff_arr_idx]
                if idx1 not in lms_dict.keys(): continue
                j = lms_dict[idx1]
                inds = (idx1, lms_inds[j + 1])
                if inds[1] - inds[0] == prev_inds[1] - prev_inds[0]:
                    for (i1, i2) in zip(range(*prev_inds), range(*inds)):
                        if nums[i1] != nums[i2]:
                            curr_name += 1
                            break
                else:
                    curr_name += 1
                name_dict[idx1] = curr_name#len(name_lsts) - 1
                prev_inds = inds
            lms_summary = [name_dict[x] for x in lms_inds]
            lms_summary.append(0)
            lms_suff_arr = recur(lms_summary, curr_name)
            curr_inds = [cumu_arr[i] - 1 for i in range(1, len(cumu_arr))]
            suff_arr = [-1] * (n + 1)
            for j in reversed(lms_suff_arr[1:]):
                i = lms_inds[j]
                char_idx = nums[i]
                suff_arr[curr_inds[char_idx]] = i
                curr_inds[char_idx] -= 1
            induceSortLS(nums, suff_arr, ls_arr, cumu_arr)
            return suff_arr
        return recur(nums, len(encoding))
    
    def buildLongestCommonPrefixArray(self) -> List[int]:
        # Kasai algorithm
        # Based on 
        # https://leetcode.com/problems/number-of-distinct-substrings-in-a-string/solutions/1010936/python-suffix-array-lcp-o-n-logn/
        suff_arr_inv = [0] * (self.n + 1)
        for rnk, idx in enumerate(self.suff_arr):
            suff_arr_inv[idx] = rnk
        length = 0
        res = [0] * (self.n + 1)
        for idx, rnk in enumerate(suff_arr_inv):
            if rnk >= self.n:
                length = 0
                continue
            idx2 = self.suff_arr[rnk + 1]
            while idx + length < self.n and idx2 + length < self.n and\
                    self.s[idx + length] == self.s[idx2 + length]:
                length += 1
            res[rnk] = length
            length = max(length - 1, 0)
        return res
    
    def checkLCP(self) -> bool:
        nxt = self.s[self.suff_arr[0]:]
        for i in range(len(self.lcp) - 1):
            idx = self.suff_arr[i + 1]
            curr, nxt = nxt, self.s[idx:]
            mx_len = min(len(curr), len(nxt))
            for j in range(mx_len):
                if curr[j] != nxt[j]: break
            else: j = mx_len
            if j != self.lcp[i]:
                return False
        if self.lcp[-1]: return False
        return True
    
    def buildLCPLR(self, lcp: List[int]) -> List[int]:
        # Based on 
        # https://stackoverflow.com/questions/38128092/how-do-we-construct-lcp-lr-array-from-lcp-array

        # This array has length of the largest power of 2 that does not exceed
        # the length of s
        length = 1
        m = self.n + 1 # The length of self.lcp
        pow2 = 0
        while length <= m:
            length <<= 1
            pow2 += 1
        res = [0] * (length >> 1)
        idx0 = length >> 2
        for i in range(m >> 2):
            i2 = i << 2
            res[idx0 + i] = min(lcp[i2: i2 + 3])
        step2 = 4
        idx = idx0
        for _ in range(2, pow2):
            step, step2 = step2, step2 << 1
            for i in range(length - step - 1, -1, -step2):
                idx -= 1
                if i >= m: continue
                res[idx] = min(res[idx << 1], res[(idx << 1) + 1], lcp[i])
        return res

    def search(self, p: str) -> List[int]:
        lcp = self.lcp
        lcp_lr = self.lcp_lr
        suff_arr = self.suff_arr

        l_idx = 1
        lft, rgt = 0, (len(self.lcp_lr) << 1) - 1
        length = 0

        def maximiseCommonLength(
            i: int,
            length: int
        ) -> Tuple[Union[int, bool]]:
            end = min(self.n - i, len(p))
            is_ge_p = False
            for j in range(length, end):
                comp = self.compareCharacters(self.s[i + j], p[j])
                #print(self.s[i + j], p[j], comp)
                if comp == 0:#self.s[i + j] == p[j]:
                    continue
                elif comp == -1:#self.s[i + j] > p[j]:
                    is_ge_p = True
                break
            else:
                j = end
                is_ge_p = (end == len(p))
            return j, is_ge_p
        
        def binarySearchStep(
            lft: int,
            rgt: int,
            mid: int,
            length: int,
            length2: int,
        ) -> Tuple[Union[int, bool]]:
            if mid > self.n:
                return (lft, mid, length, True)
            length2 = lcp_lr[l_idx]
            last_lft = False
            if length > length2:
                last_lft = True
            elif length == length2:
                length3, last_lft = maximiseCommonLength(suff_arr[mid], length)
                if not last_lft: length = length3
            if last_lft:
                return (lft, mid, length, True)
            length2 = lcp[mid]
            if length != length2:
                length = min(length, length2)
            elif mid < self.n:
                length = maximiseCommonLength(suff_arr[mid + 1], length)[0]
            return (mid + 1, rgt, length, False)

        # Identifying a range of indices of length at most 4 in suffix array in which has the first
        # suffix of s in this array which has p as a prefix must be located if any such suffixes
        # exist
        while lft < rgt - 3 and length < len(p):
            mid = lft + ((rgt - lft) >> 1)
            l_idx <<= 1
            lft, rgt, length, last_lft = binarySearchStep(lft, rgt, mid, length, lcp_lr[l_idx])
            l_idx += (not last_lft)
        
        # Further restricting this range to length 2
        if lft == rgt - 3 and lft < self.n and length < len(p):
            lft, rgt, length, last_lft = binarySearchStep(lft, rgt, lft + 1, length, lcp[lft])
        
        # Finding the first index in the suffix array whose corresponding suffix of s has p as a
        # prefix if such a suffix exists, otherwise returning the empty list as this implies
        # that p is not a substring of s
        if length < len(p):
            if lft + 1 >= len(lcp) or lcp[lft] != length: return []
            i = suff_arr[lft + 1]
            if self.n - i < len(p): return []
            for j in range(length, len(p)):
                if self.s[i + j] != p[j]:
                    return []
            lft += 1
        # Finding all the suffixes in s which have p as a prefix, which by the definition
        # of the suffix array correspond to a contiguous subarray of the suffix array whose
        # first element is at the index identifies in the preceding steps. We use the LCP
        # array to identify when this contiguous subarray ends (which is when the LCP
        # entry is less than the length of p)
        res = []
        for sa_idx in range(lft, self.n + 1):
            res.append(suff_arr[sa_idx])
            if lcp[sa_idx] < len(p):
                break
        return sorted(res)

def strStr(haystack: str, needle: str) -> int:
    """
    Finds the smallest index (0-indexed) in the string haystack such
    that there is a contiguous substring of haystack starting at that
    index that is equal to needle. If no such index exists, returns
    -1.

    Args:
        Required positional:
        haystack (str): The string to be searched for a contiguous
                substring equal to needle
        needle (str): The string to be sought as a contiguous
                substring inside hastack
    
    Returns:
    Integer (int) denoting the smallest index (0-indexed) in
    haystack such that there is a contiguous substring of haystack
    starting at that index that is equal to needle, or -1 if no
    such index exists.
    
    This illustrates the use of the SuffixArray class for string
    searching

    An overkill solution to Leetcode #28 (Find the Index of the First
    Occurrence in a String) to illustrate and test the use of suffix
    array for pattern matching.
    
    Original problem description:
    
    Given two strings needle and haystack, return the index of the
    first occurrence of needle in haystack, or -1 if needle is not part
    of haystack.
    """
    sa = SuffixArray(haystack)
    #print(len(haystack), len(sa.suff_arr))
    #print(haystack)
    #print (sa.suff_arr, sa.lcp)
    ind_lst = sa.search(needle)
    return ind_lst[0] if ind_lst else -1

def countDistinct(s: str) -> int:
    """
    
    Solution to (Premium) Leetcode #1698 (Number of Distinct Substrings
    in a String) to illustrate and test a possible use of the LCP
    array
    
    Original problem description:
    
    Given a string s, return the number of distinct substrings of s.

    A substring of a string is obtained by deleting any number of
    characters (possibly zero) from the front of the string and any
    number (possibly zero) from the back of the string. 
    """
    n = len(s)
    sa = SuffixArray(s)
    return ((n * (n + 1)) >> 1) - sum(sa.lcp)

def longestCommonSubstring(
    s_lst: List[str],
    k: int,
    part_char_ascii_start: int=32,
) -> List[str]:
    """
    Finds the longest strings that each appear as contiguous
    substrings in at least k of the strings in s_lst.

    Solved using a suffix array and LCP array.

    Args:
        Required positional:
        s_lst (list of strs): The collection of strings for which
                the longest substrings common to at least k of them
                is sought.
        k (int): The minimum number of strings in s_lst for which
                the returned strings must be substrings.

        Optional named:
        part_char_ascii_start (int): Integer giving the smallest
                ASCII code considered for use as a partition
                character as part of the algorithm. This has
                a default value of 32 as this is the smallest
                code that displays on a console as a character,
                so avoiding the issues use of characters that
                do not show up on console when debugging. If
                the number of ASCII characters needed gets
                close to the maximum ASCII code (17 * 2^16 - 1)
                then this can be set to 0 (though in a case
                when so many characters are needed, adding
                an extra 32 is extremely unlikely to make a
                meaningful difference).
            Default: 32

    Returns:
    A list of strings (str) containing strings of the same length,
    consisting of all strings that appear as contiguous substrings
    in at least k of the strings in s_lst. If there are no such
    substrings (because k exceeds the number of strings in s_lst)
    then an empty list is returned.
    Note that since the empty string is a contiguous substring of
    any string, as long as k does not exceed the number of strings
    in s_lst then there is at least one returned string.
    """
    # Default value of part_char_start is set to 32 as this is
    # where the ASCII console readable characters start
    # (for purposes of easier debugging, though this can be
    # changed to 0 if the number of ASCII characters needed gets
    # close to the total number of ASCII characters)

    n = len(s_lst)
    if k > n: return []
    elif k <= 0:
        raise ValueError("k must be a strictly positive integer.")
    elif k == 1:
        length = max(len(x) for x in s_lst)
        return [x for x in s_lst if len(x) == length]
    incl_chars = set(s_lst[0])
    for i in range(1, n):
        incl_chars |= set(s_lst[i])
    partition_chars = []
    for i in range(len(incl_chars) + n - 1):
        try:  l = chr(i + part_char_ascii_start) 
        except ValueError:
            raise ValueError("There are not enough ASCII characters "
                    "to create the necessary partition characters.")
        if l in incl_chars: continue
        partition_chars.append(l)
        if len(partition_chars) >= n - 1:
            break
    for _ in range(len(partition_chars), n - 1):
        partition_chars.pop()
    s = s_lst[0]
    partition_char_inds = []
    partition_chars = list(partition_chars)
    for i in range(1, n):
        partition_char_inds.append(len(s))
        s = partition_chars[i - 1].join([s, s_lst[i]])
    
    sa = SuffixArray(s, head_chars=partition_chars)
    #print(s)
    #print(sa.suff_arr)
    #print(sa.lcp)
    # Sliding window
    curr_mn_incl_lcp_val = float("inf")
    incl_lcp_vals = SortedList()
    
    cnts = [0] * n
    n_nonzero_cnts = 0
    s_lst_idx_incl_qu = deque()
    curr_mx_len = 0
    res_inds = []
    prev_incl_i = -1
    # Start at n as we know the results do not include the empty substring
    # (at index 0 in the suffix array) or any of the substrings of s
    # beginning with the partition characters (which due to the partition
    # characters them being defined to not be in any of the strings in
    # s_lst and lexicographically smaller than any of the characters in
    # the strings in s_lst are at indices 1 to (n - 1) inclusive in the
    # suffix array)
    i1 = n
    #for i2, (suff_idx2, lcp_val2) in enumerate(zip(sa.suff_arr, sa.lcp)):
    for i2 in range(n, sa.n + 1):
        suff_idx2 = sa.suff_arr[i2]
        lcp_val2 = sa.lcp[i2]
        #print(i1, i2, incl_lcp_vals, curr_mn_incl_lcp_val, cnts, n_nonzero_cnts, suff_idx2, lcp_val2, s_lst_idx_incl_qu)
        s_lst_idx2 = bisect.bisect_left(partition_char_inds, suff_idx2)
        if not cnts[s_lst_idx2]:
            n_nonzero_cnts += 1
        cnts[s_lst_idx2] += 1
        
        if n_nonzero_cnts >= k:
            for i1 in range(i1, i2 + 1):
                s_lst_idx1 = s_lst_idx_incl_qu[0]
                if cnts[s_lst_idx1] == 1:
                    if n_nonzero_cnts == k:
                        break
                    n_nonzero_cnts -= 1
                s_lst_idx_incl_qu.popleft()
                cnts[s_lst_idx1] -= 1
                #print(cnts)
                lcp_val1 = sa.lcp[i1]
                incl_lcp_vals.remove(lcp_val1)
                if lcp_val1 == curr_mn_incl_lcp_val:
                    # for k > 1, incl_lcp_vals is guaranteed not to be empty here
                    curr_mn_incl_lcp_val = incl_lcp_vals[0]
            else: i1 = i2 + 1 # should not get here
        #print(f"i1 = {i1}, i2 = {i2}, n_nonzero_cnts = {n_nonzero_cnts}")
        if n_nonzero_cnts >= k and curr_mn_incl_lcp_val >= curr_mx_len:
            if curr_mn_incl_lcp_val > curr_mx_len:
                res_inds = []
                curr_mx_len = curr_mn_incl_lcp_val
                prev_incl_i = -1
            if prev_incl_i < i1: # Ensures no repetitions
                res_inds.append(suff_idx2)
                prev_incl_i = i2
                #print(f"res_inds = {res_inds}, curr_mx_len = {curr_mx_len}, res_inds = {res_inds}")
            
        if not incl_lcp_vals or lcp_val2 < incl_lcp_vals[0]:
            curr_mn_incl_lcp_val = lcp_val2
        incl_lcp_vals.add(lcp_val2)
        s_lst_idx_incl_qu.append(s_lst_idx2)

    if not curr_mx_len: return [""]
    return [s[suff_idx : suff_idx + curr_mx_len] for suff_idx in res_inds]

def longestRepeatedSubstrings(s: str) -> List[str]:
    """
    Finds all the strings that appear in the string s as contiguous
    substrings at least once (including overlapping occurrences)
    and there are no longer strings that satisfy these conditions.

    This illustrates a straightforward use of the longest common
    prefix (LCP) array.

    Args:
        Required positional:
        s (str): The string to be searched for the repeated
                longest contiguous substrings.

    Returns:
    A list of strings (str) containing all strings that appear as
    contiguous substrings in s at least once.
    Note that all strings will be the same length, and the only
    string that gives no result is the empty string, as for all
    other strings the empty string is a contiguous substring at
    either end and in between each character in the string.

    Examples:
        >>> longestRepeatedSubstrings("abracadabra")
        ['abra']

        This signifies that the longest string that is a
        contiguous substring of "abracadabra" more than once is
        "abra", with one occurrence starting at the 1st character
        and another at the 8th character. It further signifies that
        there are no other such strings of length 4 or greater.

        >>> longestRepeatedSubstrings("anbaban")
        ['an', 'ba']

        This signifies that the longest strings that appear more
        than once as contiguous substrings in "anbaban" are of
        length 2, and consist of "an" and "bn" only.

        >>> longestRepeatedSubstrings("abc")
        ['']

        This signifies that the longest (and consequently only)
        string that appears as a contiguous substring in "abc"
        more than once is the empty string (which appears at
        the beginning and end and between each character). This
        result occurs if and only if the string is non-empty with
        no repeated characters.

        >>> longestRepeatedSubstrings("")
        []

        This signifies that there are no strings (not even the
        empty string itself) that appear as a contiguous substring
        of the empty string. As observed above, the empty string
        is the only string for which this result occurs.
    """
    if not s: return []
    sa = SuffixArray(s)
    length = max(sa.lcp)
    res = []
    prev_idx_incl = False
    for suff_idx, lcp_val in zip(sa.suff_arr, sa.lcp):
        if lcp_val != length:
            prev_idx_incl = False
            continue
        elif prev_idx_incl:
            continue
        res.append(s[suff_idx:suff_idx + length])
        prev_idx_incl = True
    return res

def longestDupSubstring(s: str) -> str:
    """
    Finds one of the strings that appears in the string s at least
    once (including overlapping occurrences), and is at least as
    long as any other such string (referred to as a longest
    duplicate substring of s).

    This illustrates a straightforward use of the longest common
    prefix (LCP) array

    Examples:
        >>> longestDupSubstring("banana")
        'ana'
        
        This signifies that one of the longest duplicate substrings
        of "banana" is "ana" (with the two occurrences starting at
        the 2nd and 4th letter respectively).

        >>> longestDupSubstring("abcd")
        ''

        This signifies that "abcd" has no repeating substrings
        (which is due to there being no repeated characters in
        the string).

        >>> longestDupSubstring("aaaa")
        'aaa'
        
        This signifies that one of the longest duplicate substrings
        of "aaaa" is "aaa" (with the two occurrences starting at
        the 1st and 2nd letter respectively). Note that the
        two occurrences of "aaa" overlap with each other, which
        is permitted.
    
    Solution to Leetcode #1044: Longest Duplicate Substring

    Original problem description for Leetcode #1044:

    Given a string s, consider all duplicated substrings: (contiguous)
    substrings of s that occur 2 or more times. The occurrences may
    overlap.

    Return any duplicated substring that has the longest possible length.
    If s does not have a duplicated substring, the answer is "".
    """
    sa = SuffixArray(s)
    mx_len = max(sa.lcp)
    if not mx_len: return ""
    for i, lcp_val in enumerate(sa.lcp):
        if lcp_val == mx_len: break
    j = sa.suff_arr[i]
    return s[j: j + mx_len]

if __name__ == "__main__":
    res = strStr("abcbacba", "ba")
    print(f"strStr(\"abcbacba\", \"ba\") = {res}")

    # Example from https://www.youtube.com/watch?v=DTLjHSToxmo&list=PLDV1Zeh2NRsCQ_Educ7GCNs3mvzpXhHW5&index=5
    res = longestCommonSubstring(["aabb", "bcdc", "bcde", "cded"], 2)
    print(f"\nlongestCommonSubstring([\"aabb\", \"bcdc\", \"bcde\", \"cded\"], 2) = {res}")

    # Example from https://www.youtube.com/watch?v=OptoHwC3D-Y&list=PLDV1Zeh2NRsCQ_Educ7GCNs3mvzpXhHW5&index=6
    res = longestRepeatedSubstrings("abracadabra")
    print(f"\nlongestRepeatedSubstrings(\"abracadabra\") = {res}")

    res = longestRepeatedSubstrings("anbaban")
    print(f"\nlongestRepeatedSubstrings(\"anan\") = {res}")

    res = longestRepeatedSubstrings("abc")
    print(f"\nlongestRepeatedSubstrings(\"abc\") = {res}")

    res = longestRepeatedSubstrings("")
    print(f"\nlongestRepeatedSubstrings(\"\") = {res}")

    res = longestDupSubstring("banana")
    print(f"\nlongestDupSubstring(\"banana\") = \"{res}\"")

    res = longestDupSubstring("abcd")
    print(f"\nlongestDupSubstring(\"abcd\") = \"{res}\"")

    res = longestDupSubstring("aaaa")
    print(f"\nlongestDupSubstring(\"aaaa\") = \"{res}\"")