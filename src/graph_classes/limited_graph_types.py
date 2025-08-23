#! /usr/bin/env python
from typing import (
    Hashable,
    Generator,
)

from abc import abstractmethod


from graph_classes.generic_graph_types import (
    GenericGraphTemplate,
    GenericWeightedGraphTemplate,
    GenericUnweightedGraphTemplate,
    GenericDirectedGraphTemplate,
    GenericUndirectedGraphTemplate,
    GenericUnweightedDirectedGraphTemplate,
    GenericWeightedDirectedGraphTemplate,
    GenericUnweightedUndirectedGraphTemplate,
    GenericWeightedUndirectedGraphTemplate,
)

class LimitedGraphTemplate(GenericGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def fullAdjIndex(self) -> str:
        return getattr(self, self.adj_name)
    
    def fullAdj(self) -> str:
        return {self.index2Vertex(idx1):\
                {self.index2Vertex(idx2): w for idx2, w in\
                self.getAdjIndex(idx1).items()}\
                for idx1 in range(self.n)}
    
    def getIndexLength(self) -> int:
        return self.n
    
    @abstractmethod
    def vertexGenerator(self) -> Generator[Hashable, None, None]:
        pass
    
    @abstractmethod
    def vertexInGraph(self, vertex: Hashable) -> bool:
        pass
    
    @abstractmethod
    def _negativeWeightAheadOrBehindIndex(self, idx: int,\
            ahead: bool) -> bool:
        pass
    
    def negativeWeightAheadIndex(self, idx: int) -> bool:
        return self._negativeWeightAheadOrBehindIndex(idx, True)
    
    def negativeWeightAhead(self, vertex: Hashable) -> bool:
        return self._negativeWeightAheadOrBehindIndex(\
                self.vertex2Index(vertex), True)
    
    def negativeWeightBehindIndex(self, idx: int) -> bool:
        return self._negativeWeightAheadOrBehindIndex(idx, False)
    
    def negativeWeightBehind(self, vertex: Hashable) -> bool:
        return self._negativeWeightAheadOrBehindIndex(\
                self.vertex2Index(vertex), False)

class LimitedWeightedGraphTemplate(LimitedGraphTemplate,\
        GenericWeightedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _hasNegativeEdgeWeights(self) -> bool:
        for idx in range(self.n):
            wts = self.getAdjIndex(idx).values()
            if not wts: continue
            if min(x[0] for x in wts) < 0:
                return True
        return False
    
    def _hasNegativeSelfEdgeWeights(self) -> bool:
        if not getattr(self, "_neg_weight_edge", True):
            return False
        if self.n <= 0: return False
        for idx in range(self.n):
            if self.getAdjIndex(idx).get(idx, [1])[0] < 0:
                return True
        return False
    
    def _negativeWeightAheadOrBehindIndex(self, idx: int,\
            ahead: bool) -> bool:
        if not self.neg_weight_edge:
            return False
        attr_name = "_neg_weight_ahead" if ahead\
                else "_neg_weight_behind"
        
        if not hasattr(self, attr_name):
            setattr(self, attr_name, [None] * self.n)
        record = getattr(self, attr_name)
        if record[idx] is not None:
            return record[idx]
        
        adj_wt_func = self.getAdjMinimumWeightsIndex if ahead else\
                self.getInAdjMinimumWeightsIndex
        
        seen = set()
        def dfs(idx: int) -> bool:
            if record[idx] is not None:
                return record[idx]
            seen.add(idx)
            out_min_wgts = adj_wt_func(idx)
            if min(out_min_wgts.values()) < 0 or\
                    any(record[idx2]\
                    for idx2 in out_min_wgts.keys()):
                record[idx] = False
                return False
            for idx2 in out_min_wgts.keys():
                if idx2 in seen or not dfs(idx2): continue
                record[idx] = True
                return True
            record[idx] = False
            return False
        return dfs(idx)

class LimitedUnweightedGraphTemplate(LimitedGraphTemplate,\
        GenericUnweightedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def _negativeWeightAheadOrBehindIndex(self, idx: int,\
            ahead: bool) -> bool:
        return False


class LimitedDirectedGraphTemplate(LimitedGraphTemplate,\
        GenericDirectedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def setInAdj(self) -> None:
        # Assumes that if this method has been called then the in
        # adjacencies are to be stored (as this is the whole purpose
        # of this method)
        
        if not hasattr(self, self.in_adj_name):
            self.resetInAdj()
        in_adj_lst = getattr(self, self.in_adj_name)
        unset_inds = {idx for idx in range(self.n) if\
                in_adj_lst[idx] is None}
        if not unset_inds: return
        elif len(unset_inds) == self.n:
            self.resetInAdj()
        if self.in_adj_directly_available:
            for idx in unset_inds:
                in_adj_lst[idx] = self.getInAdj(idx)
            return
        for idx in unset_inds:
            in_adj_lst[idx] = {}
        for idx1 in range(self.n):
            adj = self.getAdj(idx1)
            inds = set(adj.keys()).intersection(unset_inds)
            for idx2 in inds:
                in_adj_lst[idx2][idx1] = adj[idx2]
        # Note that since lists are mutable, the above changes made to
        # are automatically be applied to self._in_degrees_index, so it
        # is not necessary to reset self._in_degrees_index to in_degs
        return
    
    def resetInAdj(self) -> None:
        # Assumes that if this method has been called then the in
        # adjacencies are to be stored (as this is the whole purpose
        # of this method)
        res = [{} for _ in range(self.n)]
        for idx1 in range(self.n):
            for idx2, val in self.getAdjIndex(idx1).items():
                res[idx2][idx1] = val
        setattr(self, self.in_adj_name, res)
        return
    
    def getInAdjIndex(self, idx: int) -> int:
        in_adj_lst = getattr(self, self.in_adj_name, None)
        if in_adj_lst is not None:
            if in_adj_lst[idx] is not None: return in_adj_lst[idx]
        elif self.store_in_adj:
            in_adj_lst = [None for _ in range(self.n)]
            setattr(self, self.in_adj_name, in_adj_lst)
        if self.in_adj_directly_available:
            res = self._getInAdjIndex(idx)
            if self.store_in_adj:
                in_adj_lst[idx] = res
        elif self.store_in_adj:
            self.setInAdj()
            res = getattr(self, self.in_adj_name)[idx]
        else:
            res = {}
            for idx0 in range(self.n):
                adj_dict = self.getAdjIndex(idx0)
                if idx not in adj_dict.keys(): continue
                res[idx0] = adj_dict[idx]
        return res
    
    def setInDegrees(self) -> None:
        # Assumes that if this method has been called then the in
        # degrees are to be stored (as this is the whole purpose
        # of this method)
    
        #if hasattr(self, "_in_degrees_index"): return
        if not hasattr(self, "_in_degrees_index"):
            self.resetInDegrees()
        unset_inds = {idx for idx in range(self.n) if\
                self._in_degrees_index[idx] is None}
        if not unset_inds: return
        elif len(unset_inds) == self.n:
            self.resetInDegrees()
        in_degs = self._in_degrees_index
        in_adj = getattr(self, self.in_adj_name, None)
        if self.in_adj_directly_available or (in_adj is not None and\
                not any(in_adj[x] is None for x in unset_inds)):
            for idx in unset_inds:
                in_degs[idx] = sum(self.getInAdjEdgeCountsIndex(idx).values())
            return
        for idx in unset_inds:
            in_degs[idx] = 0
        for idx1 in range(self.n):
            adj = self.getAdjEdgeCountsIndex(idx1)
            inds = set(adj.keys()).intersection(unset_inds)
            for idx2 in inds:
                in_degs[idx2] += adj[idx2]
        # Note that since lists are mutable, the above changes made to
        # are automatically be applied to self._in_degrees_index, so it
        # is not necessary to reset self._in_degrees_index to in_degs
        return
    
    def resetInDegrees(self) -> None:
        # Assumes that if this method has been called then the in
        # degrees are to be stored (as this is the whole purpose
        # of this method)
        res = [0] * self.n
        for idx1 in range(self.n):
            for idx2, cnt in self.getAdjEdgeCountsIndex(idx1).items():
                res[idx2] += cnt
        self._in_degrees_index = res
        return
    
    def inDegreeIndex(self, idx: int) -> int:
        in_degs = getattr(self, "_in_degrees_index", None)
        if in_degs is not None and in_degs[idx] is not None:
            return in_degs[idx]
        in_adj = getattr(self, self.in_adj_name, None)
        if self.store_in_degrees and not self.in_adj_directly_available\
                and (in_adj is None or in_adj[idx] is None):
            self.setInDegrees()
            return self._in_degrees_index[idx]
        res = sum(self.getInAdjEdgeCountsIndex(idx).values())
        if self.store_in_degrees:
            if in_degs is None:
                self._in_degrees_index = [None] * self.n
            self._in_degrees_index[idx] = res
        return res

class LimitedUndirectedGraphTemplate(LimitedGraphTemplate,\
        GenericUndirectedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LimitedUnweightedDirectedGraphTemplate(\
        LimitedDirectedGraphTemplate,\
        LimitedUnweightedGraphTemplate,\
        GenericUnweightedDirectedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LimitedWeightedDirectedGraphTemplate(\
        LimitedDirectedGraphTemplate,\
        LimitedWeightedGraphTemplate,\
        GenericWeightedDirectedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LimitedUnweightedUndirectedGraphTemplate(\
        LimitedUndirectedGraphTemplate,\
        LimitedUnweightedGraphTemplate,\
        GenericUnweightedUndirectedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class LimitedWeightedUndirectedGraphTemplate(\
        LimitedUndirectedGraphTemplate,\
        LimitedWeightedGraphTemplate,\
        GenericWeightedUndirectedGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
