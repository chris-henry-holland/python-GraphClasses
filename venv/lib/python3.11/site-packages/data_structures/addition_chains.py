#!/usr/bin/env python

from typing import (
    Tuple,
    Optional,
    Any,
)

import bisect
import time

from algorithms.number_theory_algorithms import IntegerPartitionGenerator

class AdditionChainCalculator(object):

    def __init__(self):
        self.precursors_exact = tuple({"length": 0} for _ in range(2))
        self.precursors_Brauer = self.precursors_exact
        self.precursors_approx = self.precursors_exact

    def shortestAddChainsApprox(
        self,
        n: int,
        force_approx: bool=False
    ) -> Tuple[dict]:
        """"
        Identifies a series of steps for each positive integer up to n that
        approximately minimises the addition chain for that number.
        Essentially an approximate solution to the NP complete addition
        chain problem (inefficient for larger exponents)
        First occurrence of a chain calculated whose length is not the
        minimum is for n = 77
        """
        if not force_approx and len(self.precursors_exact) >= n + 1:
            # If exact precursors already calculated then unless there is
            # a reason not, use those rather than the approximate ones.
            return self.precursors_exact[:n + 1]
        elif len(self.precursors_approx) >= n + 1:
            return self.precursors_approx[:n + 1]
        
        bp = list(self.precursors_approx)
        for i in range(len(bp), n + 1):
            best_len = bp[-1]["length"] + 2
            for j in range((i + 1) // 2, i):
                precursors = set(bp[j].get("precursors", set())).union(\
                            set(bp[i - j].get("precursors", set())))
                precursors |= {j, i - j}
                length = len(precursors)
                #if i == n:
                #    print(f"{j}, {len(precursors)}, {precursors}")
                if length < best_len:
                    best_len = length
                    best = {"length": length, "prev_step": (i - j, j), "precursors": tuple(sorted(precursors))}
            #print(f"{i}: {best_len}, {best['precursors']}")
            bp.append(best)
        
        self.precursors_approx = tuple(bp)
        
        return self.precursors_approx
    
    def shortestAddPathApprox(
        self,
        n: int,
        force_approx: bool=False
    ) -> Tuple[int]:
        """
        Identifies the path with approximately minimum steps to successively
        add to get from 1 to the integer n.
        """
        if n == 1: return ((),)
        
        precursor_dicts = self.shortestAddChainsApprox(n, force_approx=force_approx)
        precursors = [*precursor_dicts[n]["precursors"], n]
        length = precursor_dicts[n]["length"]
        out_list = [()] * (length + 1)
        
        for i in range(1, length + 1):
            p = precursors[i]
            pair = []
            for j in precursor_dicts[p]["prev_step"]:
                pair.append(bisect.bisect_left(precursors, j, hi=i))
            out_list[i] = tuple(pair)
        return tuple(out_list)
        
    def shortestAddChainsBrauer(
        self,
        n: int,
        force_Brauer: bool=False
    ) -> Tuple[dict]:
        """
        Identifies a series of steps for each positive integer up to n that
        approximately minimises the addition chain for that number using
        Brauer chains (inefficient for larger exponents). Not exact but a
        better approximation than shortestAddChainsApprox()-  the length
        of the chain is minimal for all n < 12509
        """
        since = time.time()
        if not force_Brauer and len(self.precursors_exact) >= n + 1:
            # If exact precursors already calculated then unless there is a reason not,
            # use those rather than the approximate ones.
            return self.precursors_exact[:n + 1]
        elif len(self.precursors_Brauer) >= n + 1:
            return self.precursors_Brauer[:n + 1]
        
        bp = list(self.precursors_Brauer)
        
        for i in range(len(bp), n + 1):
            best_len = bp[-1]["length"] + 2
            best = None
            for j in range((i + 1) // 2, i):
                length = bp[j].get("length", 0) + 1
                if length > best_len: continue
                for p1 in bp[j].get("all_precursor_opts", {(): None}).keys():
                    if 2 * j != i:
                        i2 = bisect.bisect_left(p1, i - j)
                        if i2 == len(p1) or p1[i2] != i - j:
                            continue
                    precursors = (*p1, j)
                        
                    if length < best_len or best is None:
                        best_len = length
                        best = {"length": length, "prev_step": (i - j, j), "precursors": precursors,\
                                    "all_precursor_opts": {precursors: (i - j, j)}}
                    else:
                        best["all_precursor_opts"][precursors] = (i - j, j)
            #print(f"{i}: {best_len}, {best['precursors']}, {len(best['all_precursor_opts'])}")
            bp.append(best)
        
        #print(time.time() - since)
        self.precursors_Brauer = tuple(bp)
        return self.precursors_Brauer
    
    def shortestAddPathBrauer(
        self,
        n: int,
        force_Brauer: bool=False
    ) -> Tuple[int]:
        """
        Identifies the path with approximately minimum steps (from Brauer chain)
        to successively add to get from 1 to the integer n.
        """
        if n == 1: return ((),)
        precursor_dicts = self.shortestAddChainsBrauer(n, force_Brauer=force_Brauer)
        precursors = [*precursor_dicts[n]["precursors"], n]
        length = precursor_dicts[n]["length"]
        out_list = [()] * (length + 1)
        #print(precursors)
        
        for i in range(1, length + 1):
            out_list[i] = (bisect.bisect_left(precursors, precursors[i] - precursors[i - 1], hi=i), i - 1)
        return tuple(out_list)
    
    def shortestAddPathBinary(self, n: int) -> Tuple[int]:
        """
        Identifies the binary path (also known as Exponentiation by squaring)
        to successively add to get from 1 to the integer n.
        """
        if n == 1: return ((),)
        #length = math.floor(math.log(n, 2)) + 1
        out_list = [()]
        x_i = 0
        y_i = None
        while n > 1:
            n, rem = divmod(n, 2)
            if rem != 0:
                if y_i is None:
                    y_i = x_i
                else:
                    out_list.append((y_i, x_i))
                    y_i = len(out_list) - 1
            out_list.append((x_i, x_i))
            x_i = len(out_list) - 1
        if y_i is not None:
            out_list.append((y_i, x_i))
        return tuple(out_list)
                
        
    
    def shortestAddChainsExact(self, n: int) -> Tuple[dict]:
        """
        Identifies a series of steps for each positive integer up to n that
        minimises the addition chain for that number.
        Essentially a brute force solution to the NP complete addition
        chain problem (extremely inefficient for larger exponents)
        """
        since = time.time()
        
        if len(self.precursors_exact) >= n + 1:
            return self.precursors_exact[:n + 1]
        
        bp = list(self.precursors_exact)
        
        for i in range(len(bp), n + 1):
            best_len = bp[-1]["length"] + 2
            best = None
            for j in range((i + 1) // 2, i):
                for p1 in bp[j].get("all_precursor_opts", {(): None}).keys():
                    for p2 in bp[i - j].get("all_precursor_opts", {(): None}).keys():
                        #print(j)
                        #print("before:")
                        #print(bp[j].get("all_precursor_opts", {(): None}))
                        precursors = set(p1).union(set(p2))
                        precursors |= {j, i - j}
                        #print("after:")
                        #print(bp[j].get("all_precursor_opts", {(): None}))
                        length = len(precursors)
                        if length > best_len: continue
                        precursors = tuple(sorted(precursors))
                        
                        if length < best_len or best is None:
                            best_len = length
                            best = {"length": length, "prev_step": (i - j, j), "precursors": precursors,\
                                        "all_precursor_opts": {precursors: (i - j, j)}}
                            continue
                        best["all_precursor_opts"][precursors] = (i - j, j)
            #print(f"{i}: {best_len}, {best['precursors']}, {len(best['all_precursor_opts'])}")
            bp.append(best)
        
        #print(time.time() - since)
        self.precursors_exact = tuple(bp)
        return self.precursors_exact
    
    def shortestAddPathExact(self, n: int) -> Tuple[int]:
        """
        Identifies the path with minimum steps to successively add to get from 1 to the integer n.
        """
        if n == 1: return ((),)
        precursor_dicts = self.shortestAddChainsExact(n)
        precursors = [*precursor_dicts[n]["precursors"], n]
        #print(precursors)
        precursors_set = set(precursors)
        prec_map = {v: i for i, v in enumerate(precursors)}
        length = precursor_dicts[n]["length"]
        out_list = [()] * (length + 1)
        #print(precursors)
        
        for i in range(1, length + 1):
            for k, pair in precursor_dicts[precursors[i]]["all_precursor_opts"].items():
                if set(k).issubset(precursors_set): break
            #print(pair)
            out_list[i] = tuple(prec_map[j] for j in pair)
        return tuple(out_list)

    def pathValidityCheck(self, n_max: int, method: str="Brauer") -> str:
        kwargs = {}
        if method == "approx":
            func = self.shortestAddPathApprox
            kwargs = {"force_approx": True}
        elif method == "Brauer":
            func = self.shortestAddPathBrauer
            kwargs = {"force_Brauer": True}
        elif method == "exact":
            func = self.shortestAddPathExact
        elif method == "binary":
            func = self.shortestAddPathBinary
        else:
            raise ValueError(f"The value for the Exponentiator method "
                    f"pathValidityCheck() named argument method is '{method}', "
                    "which is not a valid value for this argument.")
        
        fail_list = ["Failed for the following values:"]
        for i in range(1, n_max + 1):
            path = func(i, **kwargs)
            vals = [1] * len(path)
            for j in range(1, len(path)):
                pair = path[j]
                vals[j] = vals[pair[0]] + vals[pair[1]]
            if vals[-1] != i:
                fail_list.append(f"{i}: value calculated {vals[-1]}")
            #else: print(f"Passed for {i}")
        if len(fail_list) == 1:
            return "All checks passed."
        return "\n".join(fail_list)

class Exponentiator(object):

    def __init__(
        self,
        binary_cutoff: int=200,
        default_method: str="Brauer"
    ):
        self.addition_chain_calculator = AdditionChainCalculator()
        self.binary_cutoff = binary_cutoff
        self.default_method = default_method
    
    def __call__(
        self,
        obj: Any,
        n: int, method: Optional[str]=None,
        binary_cutoff: Optional[int]=None
    ) -> Any:
        """
        Calculates obj^n for positive n
        """
        mult_identity = 1
           
        if n == 0: return mult_identity
        if n == 1: return obj
        
        if binary_cutoff is None:
            binary_cutoff = self.binary_cutoff
        
        if method is None:
            method = self.default_method
        
        
        if method == "approx" and (n <= binary_cutoff or len(self.precursors_approx) >= n + 1):
            func = self.addition_chain_calculator.shortestAddPathApprox
        elif method == "Brauer" and (n <= binary_cutoff or len(self.precursors_Brauer) >= n + 1):
            func = self.addition_chain_calculator.shortestAddPathBrauer
        elif method == "exact" and (n <= binary_cutoff or len(self.precursors_exact) >= n + 1):
            func = self.addition_chain_calculator.shortestAddPathExact
        elif method == "binary" or n > binary_cutoff:
            func = self.addition_chain_calculator.shortestAddPathBinary
        else:
            raise ValueError(f"The value for Exponentiator object function call "
                    f"named argument method is '{method}', which is not a valid "
                    "value for this argument.")
        path = func(n)
        vals = [1] * len(path)
        vals[0] = obj
        for j in range(1, len(path)):
            pair = path[j]
            vals[j] = vals[pair[0]] * vals[pair[1]]
        return vals[-1]

if __name__ == "__main__":

    addition_chain_calculator = AdditionChainCalculator()

    n = 127

    print(addition_chain_calculator.shortestAddPathBinary(n))
    print(addition_chain_calculator.shortestAddPathApprox(n, force_approx=True))
    print(addition_chain_calculator.shortestAddPathBrauer(n, force_Brauer=True))
    print(addition_chain_calculator.shortestAddPathExact(n))