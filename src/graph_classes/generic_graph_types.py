#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Hashable,
    Generator,
)

from abc import ABC, abstractmethod


# - Consider allowing negative indices to refer to indices close to
#   self.n

#Real = Union[int, float]

class GenericGraphTemplate(ABC):
    """
    Abstract class defining attributes and methods applicable to all
    types of graphs (weighted or unweighted, and directed or
    undirected), with attributes and methods specific to the different
    types of graph defined by the relevant descendants of this class.
    
    As an abstract class, graphs should not be instantiated as this
    class directly, but as one of its concrete descendant classes.
    The four principal concrete descendant classes of this class for
    which instances can be created are:
    - ExplicitUnweightedUndirectedGraph
    - ExplicitUnweightedDirectedGraph
    - ExplicitWeightedUndirectedGraph
    - ExplicitWeightedDirectedGraph
    
    Initialisation args:
        Required positional:
        
    
    Attributes:
    
    
    Methods:
        Defined:
        
        
        Abstract:
        
    """
    def __init__(self, *args, **kwargs):
        store_degrees = kwargs.get("store_degrees",\
                kwargs.get("store_out_degrees", False))
        self._store_degrees = store_degrees
        store_adj = kwargs.get("store_adj",\
                kwargs.get("store_out_adj", False))
        self._store_adj = store_adj
    
    @abstractmethod
    def vertex2Index(self, vertex: Hashable) -> int:
        pass
    
    @abstractmethod
    def index2Vertex(self, idx: int) -> Hashable:
        pass
    
    @property
    def idx_len(self) -> int:
        return self.getIndexLength()
    
    @abstractmethod
    def getIndexLength(self) -> int:
        pass
    
    @abstractmethod
    def _getAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        pass
    
    def getAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        """
        Finds the indices of the vertices adjacent to the vertex with
        index i, as well as the number of (directed) edges from the
        vertex with index i to each such adjacent vertex.
        For an undirected graph, a vertex v1 is adjacent to vertex v2
        if and only if there is at least one edge between v1 and v2.
        For a directed graph, a vertex v1 is adjacent to vertex v2 if
        and only if there is at least one directed edge from v1 to v2.
        
        Args:
            Required positional:
            idx (int): The index of the vertex for which the adjacent
                    vertices and number of of (directed) edges to thos
                    vertices are to be returned.
        
        Returns:
        Dictionary whose keys are the indices of the vertices to which
        the vertex with index idx is adjacent, and corresponding value
        is an integer (int) giving thenumber of (directed) edges from
        the vertex with index idx to the vertex whose index is the key.
        """
        if not self._store_adj:
            return self._getAdjIndex(idx)
        if not hasattr(self, self.adj_name):
            setattr(self, self.adj_name,\
                    [None] * self.idx_len)
        adj_lst = getattr(self, self.adj_name)
        if adj_lst[idx] is not None: return adj_lst[idx]
        res = self._getAdjIndex(idx)
        adj_lst[idx] = res
        return res
    
    def getAdj(self, vertex: Hashable)\
            -> Dict[Hashable, Union[int, List[Union[int, float]]]]:
        return {self.index2Vertex(idx2): val for idx2, val in\
                self.getAdjIndex(self.vertex2Index(vertex)).items()}
    
    def __getitem__(self, vertex: Hashable)\
            -> Dict[Hashable, Union[int, List[Union[int, float]]]]:
        return self.getAdj(vertex)
    
    @abstractmethod
    def getInAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
         pass
    
    def getInAdj(self, vertex: Hashable)\
            -> Dict[Hashable, Union[int, List[Union[int, float]]]]:
        return {self.index2Vertex(idx2): val for idx2, val in\
                self.getInAdjIndex(self.vertex2Index(vertex)).items()}
    
    @abstractmethod
    def getAdjEdgeCountsIndex(self, idx: int) -> Dict[int, int]:
        pass
    
    def getAdjEdgeCounts(self, vertex: Hashable)\
            -> Dict[Hashable, int]:
        return {self.index2Vertex(idx2): cnt for idx2, cnt in\
                self.getAdjEdgeCountsIndex(\
                self.vertex2Index(vertex)).items()}
    
    @abstractmethod
    def getInAdjEdgeCountsIndex(self, idx: int) -> Dict[int, int]:
        pass
    
    def getInAdjEdgeCounts(self, vertex: Hashable)\
            -> Dict[Hashable, int]:
        return {self.index2Vertex(idx2): cnt for idx2, cnt in\
                self.getInAdjEdgeCountsIndex(\
                self.vertex2Index(vertex)).items()}
    
    @abstractmethod
    def getAdjMinimumWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        pass
    
    def getAdjMinimumWeights(self, vertex: Hashable)\
            -> Dict[int, Union[int, float]]:
        return {self.index2Vertex(idx2): wt for idx2, wt in\
                self.getAdjMinimumWeightsIndex(\
                self.vertex2Index(vertex)).items()}
    
    @abstractmethod
    def getInAdjMinimumWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        pass
    
    def getInAdjMinimumWeights(self, vertex: Hashable)\
            -> Dict[int, Union[int, float]]:
        return {self.index2Vertex(idx2): wt for idx2, wt in\
                self.getInAdjMinimumWeightsIndex(\
                self.vertex2Index(vertex)).items()}
    
    @abstractmethod
    def getAdjTotalWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        pass
    
    def getAdjTotalWeights(self, vertex: Hashable)\
            -> Dict[int, Union[int, float]]:
        return {self.index2Vertex(idx2): wt for idx2, wt in\
                self.getAdjTotalIndex(\
                self.vertex2Index(vertex)).items()}
    
    @abstractmethod
    def getInAdjTotalWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        pass
    
    def getInAdjTotalWeights(self, vertex: Hashable)\
            -> Dict[int, Union[int, float]]:
        return {self.index2Vertex(idx2): wt for idx2, wt in\
                self.getInAdjTotalWeightsIndex(\
                self.vertex2Index(vertex)).items()}
    
    def adjGeneratorIndex(self, idx: int)\
            -> Generator[int, None, None]:
        #print("Using adjGeneratorIndex")
        for idx2 in self.getAdjIndex(idx).keys():
            yield idx2
        return
    
    def adjGenerator(self, vertex: Hashable)\
            -> Generator[Hashable, None, None]:
        for idx2 in self.adjGeneratorIndex(self.vertex2Index(vertex)):
            yield self.index2Vertex(idx2)
        return
    
    def inAdjGeneratorIndex(self, idx: int)\
            -> Generator[int, None, None]:
        #print("Using inAdjGeneratorIndex()")
        for idx2 in self.getInAdjIndex(idx).keys():
            yield idx2
        return
    
    def inAdjGenerator(self, vertex: Hashable)\
            -> Generator[Hashable, None, None]:
        for idx2 in\
                self.inAdjGeneratorIndex(self.vertex2Index(vertex)):
            yield self.index2Vertex(idx2)
        return
    
    @abstractmethod
    def edgeCountIndex(self, idx1: int, idx2: int) -> int:
        pass
    
    def edgeCount(self, vertex1: Hashable, vertex2: Hashable) -> int:
        return self.edgeCountIndex(self.vertex2Index(vertex1),\
                self.vertex2Index(vertex2))
    
    @abstractmethod
    def edgeMinimumWeightIndex(self, idx1: int, idx2: int)\
            -> Optional[Union[int, float]]:
        pass
    
    def edgeMinimumWeight(self, vertex1: Hashable, vertex2: Hashable)\
            -> Optional[Union[int, float]]:
        return self.edgeMinimumWeight(self.vertex2Index(vertex1),\
                self.vertex2Index(vertex2))
    
    def _degreeIndex(self, idx: int) -> int:
        degs = getattr(self, "_degrees_index", None)
        if degs is not None and degs[idx] is not None:
            return degs[idx]
        adj = getattr(self, self.adj_name, None)
        res = sum(self.getAdjEdgeCountsIndex(idx).values())
        if self._store_degrees:
            if degs is None:
                self._degrees_index = [None] * self.n
            self._degrees_index[idx] = res
        return res
    
    def _argIndices2IndexDict(self,\
            inds: Union[Set[int],Dict[int, Tuple[Union[int, float]]]])\
            -> Dict[int, Tuple[Union[int, float]]]:
        if isinstance(inds, dict): return inds
        return {x: 0 for x in inds}
    
    def _argVertices2IndexDict(self, vertices: Union[Set[Hashable],\
            Dict[Hashable, Tuple[Union[int, float]]]])\
            -> Dict[int, Tuple[Union[int, float]]]:
        if isinstance(vertices, dict):
            return {self.vertex2Index(x): y\
                    for x, y in vertices.items()}
        return {self.vertex2Index(x): 0 for x in vertices}

    def _argIndices2IndexSet(self, inds: Union[Set[int],\
            Dict[int, Tuple[Union[int, float]]]],\
            check_dists: bool=True, eps: float=10 ** -5)\
            -> Tuple[Union[Set[int], Optional[Union[int, float]]]]:
        if isinstance(inds, set): return inds, 0
        inds_set = set(inds.keys())
        if not check_dists or not inds:
            return (inds_set, 0)
        elif len(inds) == 1:
            return (inds_set, next(iter(inds.values())))
        rng = [float("inf"), -float("inf")]
        has_float = False
        for d in inds.values():
            if isinstance(d, float): has_float = True
            rng = [min(rng[0], d), max(rng[1], d)]
            if rng[1] - rng[0] > eps: break
        else:
            if not has_float: return (inds_set, rng[0])
            return (inds_set, (rng[0] / 2) + (rng[1] / 2))
        return (inds_set, None)
    
    def _argVertices2IndexSet(self, vertices: Union[Set[Hashable],\
            Dict[Hashable, Tuple[Union[int, float]]]],\
            check_dists: bool=True, eps: float=10 ** -5)\
            -> Tuple[Union[Set[Hashable],\
            Optional[Union[int, float]]]]:
        if isinstance(vertices, set): return vertices, 0
        inds_set = {self.vertex2Index(x) for x in vertices.keys()}
        if not check_dists or not vertices:
            return (inds_set, 0)
        elif len(vertices) == 1:
            return (inds_set, next(iter(vertices.values())))
        rng = [float("inf"), -float("inf")]
        has_float = False
        for d in vertices.values():
            if isinstance(d, float): has_float = True
            rng = [min(rng[0], d), max(rng[0], d)]
            if rng[1] - rng[0] > eps: break
        else:
            if not has_float: return (inds_set, rng[0])
            return (inds_set, (rng[0] / 2) + (rng[1] / 2))
        return (inds_set, None)
    

class GenericWeightedGraphTemplate(GenericGraphTemplate):
    adj_name = "adj_weights"
    
    def __init__(self, *args, neg_weight_edge: Optional[bool]=None,\
            neg_weight_self_edge: Optional[bool]=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._neg_weight_edge = neg_weight_edge
        self._neg_weight_self_edge = neg_weight_self_edge
    
    @property
    def neg_weight_edge(self) -> bool:
        res = getattr(self, "_neg_weight_edge", None)
        if res is not None: return res
        res = self._hasNegativeEdgeWeights() if\
                hasattr(self, "_hasNegativeEdgeWeights") else None
        self._neg_weight_edge = res
        if res is False: self._neg_weight_self_edge = False
        return res
    
    @property
    def neg_weight_self_edge(self) -> bool:
        res = getattr(self, "_neg_weight_self_edge", None)
        if res is not None: return res
        res = self._hasNegativeSelfEdgeWeights() if\
                hasattr(self, "_hasNegativeSelfEdgeWeights") else None
        self._neg_weight_self_edge = res
        return res
    
    def getAdjEdgeCountsIndex(self, idx: int) -> Dict[int, int]:
        return {idx2: len(lst) for idx2, lst in\
                self.getAdjIndex(idx).items()}
    
    def getInAdjEdgeCountsIndex(self, idx: int) -> Dict[int, int]:
        return {idx2: len(lst) for idx2, lst in\
                self.getInAdjIndex(idx).items()}
    
    def getAdjMinimumWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        return {idx2: lst[0] for idx2, lst in\
                self.getAdjIndex(idx).items()}

    def getInAdjMinimumWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        return {idx2: lst[0] for idx2, lst in\
                self.getInAdjIndex(idx).items()}
    
    def getAdjTotalWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        return {idx2: sum(lst) for idx2, lst in\
                self.getAdjIndex(idx).items()}

    def getInAdjTotalWeightsIndex(self, idx: int)\
            -> Dict[int, Union[int, float]]:
        return {idx2: sum(lst) for idx2, lst in\
                self.getInAdjIndex(idx).items()}
    
    def edgeCountIndex(self, idx1: int, idx2: int) -> int:
        return len(self.getAdjIndex(idx1).get(idx2, []))
    
    def edgeMinimumWeightIndex(self, idx1: int, idx2: int)\
            -> Optional[Union[int, float]]:
        return self.getAdjIndex(idx1).get(idx2, [None])[0]
    
    def edgeTotalWeightIndex(self, idx1: int, idx2: int)\
            -> Optional[Union[int, float]]:
        adj = self.getAdjIndex(idx1)
        return sum(adj[idx2]) if idx2 in adj.keys() else None

class GenericUnweightedGraphTemplate(GenericGraphTemplate):
    adj_name = "adj_counts"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    # Unweighted graph has no negative weight edges (all have implicit
    # weight of 1)
    @property
    def neg_weight_edge(self) -> bool:
        return False
    
    @property
    def neg_weight_self_edge(self) -> bool:
        return False
    
    @property
    def neg_weight_cycle(self) -> bool:
        return False
    
    def getAdjEdgeCountsIndex(self, idx: int) -> Dict[int, int]:
        return self.getAdjIndex(idx)
    
    def getInAdjEdgeCountsIndex(self, idx: int) -> Dict[int, int]:
        return self.getInAdjIndex(idx)

    def getAdjMinimumWeightsIndex(self, idx: int)\
            -> Dict[int, int]:
        return {idx2: 1 for idx2 in self.getAdjIndex(idx).keys()}
    
    def getInAdjMinimumWeightsIndex(self, idx: int)\
            -> Dict[int, int]:
        return {idx2: 1 for idx2 in self.getInAdjIndex(idx).keys()}
    
    def getAdjTotalWeightsIndex(self, idx: int)\
            -> Dict[int, int]:
        return self.getAdjEdgeCountsIndex(idx)
    
    def getInAdjTotalWeightsIndex(self, idx: int)\
            -> Dict[int, int]:
        return self.getInAdjEdgeCountsIndex(idx)
    
    def edgeCountIndex(self, idx1: int, idx2: int) -> int:
        return self.getAdjIndex(idx1).get(idx2, 0)
    
    def edgeMinimumWeightIndex(self, idx1: int, idx2: int)\
            -> Optional[int]:
        return 1 if idx2 in self.getAdjIndex(idx1).keys() else None
    
    def edgeTotalWeightIndex(self, idx1: int, idx2: int)\
            -> Optional[int]:
        return self.getAdjIndex(idx1).get(idx2, None)

class GenericDirectedGraphTemplate(GenericGraphTemplate):
    
    def __init__(self, *args,\
            store_in_adj: bool=True,\
            store_in_degrees: bool=True,\
            **kwargs):
        self.store_in_degrees = store_in_degrees
        self.store_in_adj = store_in_adj
        super().__init__(*args, store_in_adj=store_in_adj,\
                store_in_degrees=store_in_degrees, **kwargs)
    
    @property
    def in_adj_directly_available(self) -> bool:
        return hasattr(self, "_getInAdjIndex")
    
    @property
    def store_out_degrees(self) -> bool:
        return self._store_degrees
    
    @store_out_degrees.setter
    def store_out_degrees(self, b: bool) -> None:
        self._store_degrees = b
        return
    
    @abstractmethod
    def setInAdj(self) -> None:
        pass
    
    @abstractmethod
    def resetInAdj(self) -> None:
        pass
    
    def outDegreeIndex(self, idx: int) -> int:
        return self._degreeIndex(idx)
    
    def outDegree(self, vertex: Hashable) -> int:
        return self.outDegreeIndex(self.vertex2Index(vertex))
    
    @abstractmethod
    def setInDegrees(self) -> None:
        pass
    
    @abstractmethod
    def resetInDegrees(self) -> None:
        pass
    
    @abstractmethod
    def inDegreeIndex(self, idx: int) -> int:
        pass
    
    def inDegree(self, vertex: Hashable) -> int:
        return self.inDegreeIndex(self.vertex2Index(vertex))
    
class GenericUndirectedGraphTemplate(GenericGraphTemplate):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def in_adj_directly_available(self) -> bool:
        return True
    
    @property
    def store_degrees(self):
        return self._store_degrees
    
    @store_degrees.setter
    def store_degrees(self, b: bool) -> None:
         self._store_degrees = b
         return
    
    def getInAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
         return self.getAdjIndex(idx)
    
    def degreeIndex(self, idx: int) -> int:
        return self._degreeIndex(idx)
    
    def degree(self, vertex: Hashable) -> int:
        return self.degreeIndex(self.vertex2Index(vertex))
    
class GenericUnweightedDirectedGraphTemplate(\
        GenericDirectedGraphTemplate,\
        GenericUnweightedGraphTemplate):
    graph_type_name = "unweighted directed graph"
    in_adj_name = "in_adj_counts"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GenericWeightedDirectedGraphTemplate(\
        GenericDirectedGraphTemplate,\
        GenericWeightedGraphTemplate):
    """
    Class whose instances represent weighted graphs (either directed
    or undirected) and allows for graph algorithms such as Dijkstra
    to be performed through methods.
    
    Initialisation args:
        Required positional:
        vertices (list of hashable objects): A list of the objects
                representing the vertices of the graph.
        edges (List of 3-tuples): A list of tuples representing the
                edges of the graph, with for each tuple in the list,
                index 0 containing the object represenging the vertex
                (as per vertices) from which the edge originates,
                index 2 containing the object representing the
                vertex to which the edge terminates and index 2
                containing the weight of the edge (as a number).
                Note that if the graph is undirected (as signified
                by the argument directed being given as False), the
                order of the two vertices in the tuple does not
                matter.
        
        Optional named:
        directed (bool): If True, the graph is directed, otherwise
                undirected.
            Default: False
    
    Attributes:
        n (int): Non-negative integer giving the number of vertices
                in the graph
        vertices (list of hashable objects): A list of objects
                of length n representing the vertices of the graph.
                Each vertex may also be referenced by an index, which
                is the integer between 0 and n - 1 inclusive giving
                its (0-indexed) position in this list.
        vertex_dict (dict): A dictionary whose keys and values are
                the elements of the vertices attribute list and the
                index of that vertex (i.e. the 0-indexed position of
                the vertex in the vertices attribute) respectively.
                This is effectively the inverse of the list attribute
                vertices.
        directed (bool): If True, this represents a directed graph,
                otherwise an undirected graph
        adj (list of dicts): Adjacency list of the graph, represented
                by a list of dicts of length n, where each entry is
                a dictionary representing the edges (in the case of
                directed graph, outgoing edges) of the vertex in
                the equivalent position in the attribute vertices
                list, whose keys and values are the indices in the
                attribute vertices list of the vertices that vertex
                has (directed) edges to, and the corresponding
                weight of that edge respectively.
                In the case of repeated (directed) edges, the weight
                is taken to be the minimum of the weights of those
                (directed edges).
                In the case of undirected graphs, each edge is
                included in the dictionary for both of the vertices
                associated with this edge, while in the case of
                directed graphs, it is only the vertex from which
                the directed edge originates from for which it
                is included.
        neg_weight_edge (bool): Whether the (directed) graph contains
                any negative weight (directed) edges.
                The presence of such an edge prevents certain
                algorithms from being used, for example Dijkstra.
    
    Indexing:
        Accessing an index corresponding to an element of the
        vertices attribute list, returns a dictionary whose keys
        and values are all the vertices with which the vertex used
        as the index shares an edge (or, for directed graphs, all
        the vertices where there exists a directed edge going from
        the index vertex to that vertex) and the weight of that
        (directed) edge respectively.
    
    Methods:
            (see documentation of the method in question for more
            detail)
        getAdjIndex(): For a given vertex, gives a dictionary giving
                the vertices the chosen vertex has an (outgoing) edge
                to with the corresponding weight, with all vertices
                referred to by their position in the attribute
                vertices (i.e. by an integer from 0 to n - 1
                inclusive).
        vertex2Index(): For a given vertex, returns the position of
                that vertex in the attribute vertices.
        index2Vertex(): Given an integer from 0 to n - 1 inclusive,
                gives the vertex to which that integer corresponds
                (based on which vertex is at that position in 
                the attribute vertices).
        dijkstra(): Performs a Dijkstra search for the minimum cost
                path between one set of vertices and another set of
                vertices in the graph. Note that this will return
                an error if the graph contains any (directed)
                edges with negative weight.
    """
    in_adj_name = "in_adj_weights"
    graph_type_name = "weighted directed graph"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def neg_weight_cycle(self) -> Optional[bool]:
        # None signifies not known for sure
        res = getattr(self, "_neg_weight_cycle", None)
        if res is not None: return res
        if not self.neg_weight_edge:
            res = False
        elif self.neg_weight_self_edge:
            res = True
        if res is not None:
            self._neg_weight_cycle = res
        return res

class GenericUnweightedUndirectedGraphTemplate(\
        GenericUndirectedGraphTemplate,\
        GenericUnweightedGraphTemplate):
    graph_type_name = "unweighted undirected graph"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GenericWeightedUndirectedGraphTemplate(\
        GenericUndirectedGraphTemplate,\
        GenericWeightedGraphTemplate):
    graph_type_name = "weighted undirected graph"
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def neg_weight_cycle(self) -> Optional[bool]:
        # A weighted undirected graph has a negative weight
        # cycle if and only if it contains a negative weight
        # edge (as traversing a negative weight edge in one
        # direction then the other forms a negative weight
        # cycle)
        return self.neg_weight_edge
