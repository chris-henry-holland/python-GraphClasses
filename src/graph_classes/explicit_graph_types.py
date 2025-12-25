#! /usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Hashable,
    Generator,
    Iterable,
)

import heapq

from abc import abstractmethod

from graph_classes.limited_graph_types import (
    LimitedGraphTemplate,
    LimitedWeightedGraphTemplate,
    LimitedUnweightedGraphTemplate,
    LimitedDirectedGraphTemplate,
    LimitedUndirectedGraphTemplate,
    LimitedUnweightedDirectedGraphTemplate,
    LimitedWeightedDirectedGraphTemplate,
    LimitedUnweightedUndirectedGraphTemplate,
    LimitedWeightedUndirectedGraphTemplate,
)

# Review- consider adding method to relabel the vertices

class ExplicitGraphTemplate(LimitedGraphTemplate):
    
    def __init__(
        self,
        vertices: Iterable[Hashable],
        edges: Iterable[Tuple[Hashable]],
        *args,
        **kwargs,
    ):
        self.n = len(vertices)
        self.vertices = list(vertices)
        self.vertex_dict = {v: i for i, v in enumerate(self.vertices)}
        setattr(self, self.adj_name, [{} for _ in range(self.n)])
        set_degrees = kwargs.get("set_degrees",\
                kwargs.get("set_out_degrees", False))
        store_degrees = kwargs.get("store_degrees",\
                kwargs.get("store_out_degrees", False))
        if set_degrees and store_degrees:
            self._degrees_index = [0] * self.n
        super().__init__(*args, **kwargs)
    
    def containsVertex(self, vertex: Hashable) -> bool:
        """
        Finds whether the hashable object vertex represents a
        vertex in the graph.

        Args:
            Required positional:
            vertex (hashable object): The hashable object whose
                    status as representing a vertex is to be
                    established.
        
        Returns:
        Boolean (bool) giving True if vertex represents a vertex
        in the graph and False otherwise.
        """
        return vertex in self.vertex_dict.keys()

    def vertex2Index(self, vertex: Hashable) -> int:
        """
        For a given vertex in the graph, returns the index of that
        vertex (i.e. its 0-indexed position in the list given by
        attribute vertices).
        
        Args:
            Required positional:
            vertex (hashable object): The vertex whose index is to be
                    returned.
        
        Returns:
        Integer (int) giving the index of vertex.
        """
        return self.vertex_dict[vertex]
    
    def index2Vertex(self, idx: int) -> Hashable:
        """
        For an integer between 0 and n - 1 (where n is the attribute
        n), returns the vertex whose index is that integer (i.e. the
        value contained in the list given by attribute vertices at
        that index).
        
        Args:
            Required positional:
            idx (int): The index of the vertex that is to be returned.
        
        Returns:
        Hashable object giving the vertex with index idx.
        """
        return self.vertices[idx]
    
    def vertexGenerator(self) -> Generator[Hashable, None, None]:
        for vertex in self.vertices:
            yield vertex
        return
    
    def vertexInGraph(self, vertex: Hashable) -> bool:
        return vertex in self.vertex_dict.keys()
    
    def _getAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        return getattr(self, self.adj_name)[idx]
    
    @abstractmethod
    def _addEdgeToAdjIndex(self, idx1: int, idx2: int,\
            weight: Optional[Union[int, float]]=1,\
            n_edges: int=1) -> None:
        pass
    
    @abstractmethod
    def addEdgeIndex(self, idx1: int, idx2: int,\
            weight: Union[int, float]) -> None:
        pass
    
    @abstractmethod
    def addEdge(self, v1: Hashable, v2: Hashable,\
            weight: Union[int, float]) -> None:
        pass
    
    @abstractmethod
    def _addEdgeIndex(self, idx1: int, idx2: int,\
            weight: Optional[Union[int, float]]=None,\
            n_edges: int=1) -> None:
        pass
    
    def addVertex(self, vertex: Hashable) -> bool:
        if not hasattr(self, "vertex_dict"):
            raise NotImplementedError("The method addVertex() is not "
                    "implemented for this class.")
        if self.vertexInGraph(vertex):
            return False
        self.vertex_dict[vertex] = self.n
        self.vertices.append(vertex)
        getattr(self, self.adj_name).append({})
        self.n += 1
        if hasattr(self, "in_adj_name") and\
                hasattr(self, self.in_adj_name):
            getattr(self, self.in_adj_name).append({})
        if hasattr(self, "_degrees_index"):
            self._degrees_index.append(0)
        if hasattr(self, "_in_degrees_index"):
            self._in_degrees_index.append(0)
        return True
        
class ExplicitWeightedGraphTemplate(
    ExplicitGraphTemplate,
    LimitedWeightedGraphTemplate
):
    
    def __init__(self, vertices: Iterable[Hashable],\
            edges: Iterable[Tuple[Hashable]], *args, **kwargs):
        self._neg_weight_edge = False
        self._neg_weight_self_edge = False
        super().__init__(vertices, edges, *args, **kwargs)
        for e in edges:
            idx1, idx2 = list(map(self.vertex_dict.__getitem__, e[:2]))
            self._addEdgeIndex(idx1, idx2, weight=e[2], n_edges=1)
    
    def _addEdgeToAdjIndex(self, idx1: int, idx2: int,\
            weight: Optional[Union[int, float]]=1,\
            n_edges: int=1) -> None:
        adj = getattr(self, self.adj_name)
        adj[idx1].setdefault(idx2, [])
        heapq.heappush(adj[idx1][idx2], weight)
        if weight < 0:
            self._neg_weight_edge = True
            if idx1 == idx2:
                self._neg_weight_self_edge = True
        return
    
    def addEdgeIndex(self, idx1: int, idx2: int,\
            weight: Union[int, float]) -> None:
        if not 0 <= idx1 < self.n or not  0 <= idx2 < self.n:
            raise ValueError("The indices given must be integers "
                    "between 0 and self.n - 1 = {self.n - 1} "
                    "inclusive.")
        self._addEdgeIndex(idx1, idx2, weight=weight, n_edges=1)
        return
    
    def addEdge(self, v1: Hashable, v2: Hashable,\
            weight: Union[int, float]) -> None:
        idx1, idx2 = list(map(self.__getitem__, (v1, v2)))
        if idx1 is None or idx2 is None:
            raise ValueError("The vertices given must be keys of "
                    "of the attribute vertices.")
        self._addEdgeIndex(idx1, idx2, weight=weight, n_edges=1)
        if hasattr(self, "_in_degrees_index"):
            self._in_degrees_index[idx2] += 1
        return

class ExplicitUnweightedGraphTemplate(
    ExplicitGraphTemplate,
    LimitedUnweightedGraphTemplate,
):
    
    def __init__(self, vertices: Iterable[Hashable],\
            edges: Iterable[Tuple[Hashable]], *args, **kwargs):
        super().__init__(vertices, edges, **kwargs)
        for e in edges:
            idx1, idx2 = list(map(self.vertex_dict.__getitem__, e[:2]))
            self._addEdgeIndex(idx1, idx2, n_edges=1)
    
    def _addEdgeToAdjIndex(self, idx1: int, idx2: int,\
            weight: Optional[Union[int, float]]=None,\
            n_edges: int=1) -> None:
        getattr(self, self.adj_name)[idx1][idx2] =\
                getattr(self, self.adj_name)[idx1].get(idx2, 0) +\
                n_edges
        return
    
    def addEdgeIndex(self, idx1: int, idx2: int,\
            n_edges: int) -> None:
        if not 0 <= idx1 < self.n or not  0 <= idx2 < self.n:
            raise ValueError("The indices given must be integers "
                    "between 0 and self.n - 1 = {self.n - 1} "
                    "inclusive.")
        self._addEdgeIndex(idx1, idx2, n_edges=n_edges)
        return
    
    def addEdge(self, v1: Hashable, v2: Hashable,\
            n_edges: int) -> None:
        idx1, idx2 = list(map(self.__getitem__, (v1, v2)))
        if idx1 is None or idx2 is None:
            raise ValueError("The vertices given must be keys of "
                    "of the attribute vertices.")
        self._addEdgeIndex(idx1, idx2, n_edges=n_edges)
        return

class ExplicitDirectedGraphTemplate(
    ExplicitGraphTemplate,
    LimitedDirectedGraphTemplate
):
    
    def __init__(self, vertices: List[Hashable],\
            edges: List[Tuple[Hashable]],\
            *args,\
            store_in_adj: bool=True,\
            set_in_adj: bool=False,\
            store_out_degrees: bool=False,\
            set_out_degrees: bool=False,\
            store_in_degrees: bool=True,\
            set_in_degrees: bool=False,\
            **kwargs):
        n = len(vertices)
        store_in_degrees = kwargs.get("store_in_degrees", True)
        set_in_degrees = kwargs.get("set_in_degrees", False)
        if set_in_degrees and store_in_degrees:
            self._in_degrees_index = [0] * self.n
        store_in_adj = kwargs.get("store_in_adj", True)
        set_in_adj = kwargs.get("set_in_adj", False)
        if set_in_adj and store_in_adj:
            setattr(self, self.in_adj_name, [{} for _ in range(n)])
        super().__init__(vertices, edges, *args, **kwargs)

    def _addEdgeIndex(self, idx1: int, idx2: int,\
            weight: Optional[Union[int, float]]=None,\
            n_edges: Optional[int]=None) -> None:
        self._addEdgeToAdjIndex(idx1, idx2, weight=weight,
                n_edges=n_edges)
        if hasattr(self, "in_adj_name") and\
                hasattr(self, self.in_adj_name) and\
                getattr(self, self.in_adj_name)[idx2] is not None:
            getattr(self, self.in_adj_name)[idx2][idx1] =\
                    getattr(self, self.adj_name)[idx1][idx2]
        if hasattr(self, "_degrees_index") and\
                self._degrees_index[idx2] is not None:
            self._degrees_index[idx1] += n_edges
        if hasattr(self, "_in_degrees_index") and\
                self._in_degrees_index[idx2] is not None:
            self._in_degrees_index[idx2] += n_edges
        return

class ExplicitUndirectedGraphTemplate(ExplicitGraphTemplate,\
        LimitedUndirectedGraphTemplate):
    
    def __init__(self, vertices: List[Hashable],\
            edges: List[Tuple[Hashable]], *args, **kwargs):
        super().__init__(vertices, edges, *args, **kwargs)
    
    def _addEdgeIndex(self, idx1: int, idx2: int,\
            weight: Optional[Union[int, float]]=None,\
            n_edges: int=1) -> None:
        self._addEdgeToAdjIndex(idx1, idx2, weight=weight,\
                n_edges=n_edges)
        self._addEdgeToAdjIndex(idx2, idx1, weight=weight,\
                n_edges=n_edges)
        if hasattr(self, "_degrees_index"):
            if self._degrees_index[idx1] is not None:
                self._degrees_index[idx1] += n_edges
            if self._degrees_index[idx2] is not None:
                self._degrees_index[idx2] += n_edges
        return

class ExplicitUnweightedDirectedGraph(\
        ExplicitDirectedGraphTemplate,\
        ExplicitUnweightedGraphTemplate,\
        LimitedUnweightedDirectedGraphTemplate):
    
    def __init__(self, vertices: Iterable[Hashable],\
            edges: Iterable[Tuple[Hashable]],\
            store_out_degrees: bool=False,\
            set_out_degrees: bool=False,\
            store_in_degrees: bool=True, set_in_degrees: bool=False,\
            store_in_adj: bool=True, set_in_adj: bool=False):
        super().__init__(vertices, edges,\
                store_out_adj=True,
                store_out_degrees=store_out_degrees,\
                set_out_degrees=set_out_degrees,\
                store_in_adj=store_in_adj, set_in_adj=set_in_adj,\
                store_in_degrees=store_in_degrees,\
                set_in_degrees=set_in_degrees)

class ExplicitWeightedDirectedGraph(\
        ExplicitDirectedGraphTemplate,\
        ExplicitWeightedGraphTemplate,\
        LimitedWeightedDirectedGraphTemplate):
    
    def __init__(self, vertices: Iterable[Hashable],\
            edges: Iterable[Tuple[Hashable]],\
            store_out_degrees: bool=False,\
            set_out_degrees: bool=False,\
            store_in_degrees: bool=True, set_in_degrees: bool=False,\
            store_in_adj: bool=True, set_in_adj: bool=False):
        super().__init__(vertices, edges,\
                store_out_adj=True,
                store_out_degrees=store_out_degrees,\
                set_out_degrees=set_out_degrees,\
                store_in_adj=store_in_adj, set_in_adj=set_in_adj,\
                store_in_degrees=store_in_degrees,\
                set_in_degrees=set_in_degrees)

class ExplicitUnweightedUndirectedGraph(\
        ExplicitUndirectedGraphTemplate,\
        ExplicitUnweightedGraphTemplate,\
        LimitedUnweightedUndirectedGraphTemplate):

    def __init__(self, vertices: Iterable[Hashable],\
            edges: Iterable[Tuple[Hashable]],\
            store_degrees: bool=False, set_degrees: bool=False):
        super().__init__(vertices, edges,\
                store_adj=True,\
                store_degrees=store_degrees,\
                set_degrees=set_degrees)

class ExplicitWeightedUndirectedGraph(\
        ExplicitUndirectedGraphTemplate,\
        ExplicitWeightedGraphTemplate,\
        LimitedWeightedUndirectedGraphTemplate):
    
    def __init__(self, vertices: Iterable[Hashable],\
            edges: Iterable[Tuple[Hashable]],\
            store_degrees: bool=False, set_degrees: bool=False):
        super().__init__(vertices, edges,\
                store_adj=True,\
                store_degrees=store_degrees,\
                set_degrees=set_degrees)
