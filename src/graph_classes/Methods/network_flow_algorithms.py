#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
    Optional,
    Union,
    Hashable,
)

if TYPE_CHECKING:
    from graph_classes.limited_graph_types import (
        LimitedGraphTemplate,
    )

from graph_classes.explicit_graph_types import (
    ExplicitWeightedDirectedGraph,
)

### Ford-Fulkerson algorithm for maximum flow through a network ###

def fordFulkersonIndex(
        self,
        start_idx: int,
        end_idx: int,
        eps: float=10 ** -5,
        return_poss_distribution: bool=True
) -> Tuple[Union[int, float, Optional[ExplicitWeightedDirectedGraph]]]:
    """
    Method implementing the Ford-Fulkerson algorithm to find the
    maximum flow from the vertex with index start_idx to the vertex
    with index end_idx for the network represented by this graph
    object, where each edge of the graph can support a
    flow capacity up to its weight (for weighted graphs) or 1 (for
    unweighted graphs) in the direction of the edge (for directed
    graphs) or in either direction (for undirected graphs). Finds
    the maximum flow and (if return_poss_distribution is given as
    True) the direct flow between adjacent vertices for a possible
    distribution of a flow of that size.
    
    Args:
        Required positional:
        start_idx (int): The index of the vertex of the graph from
                which the flow starts (i.e. the source).
                Should be integer between 0 and (self.n - 1) inclusive
        end_idx (int): The index of the vertex of the graph from
                which the flow ends (i.e. the sink).
                Should be integer between 0 and (self.n - 1) inclusive
                and different to start_idx
        
        Optional named:
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
        return_poss_distribution (bool): If True, then finds a possible
                flow through the network that has the maximum flow from
                the vertex with index start_idx to the vertex with
                index end_idx, representing this flow by the total
                direct flow between each adjacent pair of vertices
                for which there is non-zero direct flow from the first
                to the second.
            Default: True
    
    Returns:
    2-tuple whose index 0 contains the size of the maximum possible
    flow from the vertex with index start_idx to the vertex with index
    end_idx in the network represented by graph, and whose index 1
    contains None if return_poss_distribution is given as False or an
    ExplicitWeightedDirectedGraph object representing a possible flow
    distribution through the network with that flow, where the vertices
    are the same as for the graph and with each vertex having the
    same index as in the graph, and the directed edge between a pair
    of vertices represents that there is direct flow from the first
    vertex of the pair to the second (i.e. flow straight from the
    first vertex to the second with no intervening vertices), with the
    weight of the edge representing the size of this direct flow.
    """
    if self.neg_weight_edge:
        raise NotImplementedError("The method fordFulkersonIndex() "
                "cannot be used for graphs with negative weight "
                "edges.")
    elif start_idx == end_idx:
        raise ValueError("The values of input arguments start_idx and "
                "end_idx in the method fordFulkersonIndex() must be "
                "different") 
    
    # Direct flow capacities between each pair of vertices that
    # share at least one edge
    capacities = [dict(self.getAdjTotalWeightsIndex(idx))\
            for idx in range(self.n)]
    #print(capacities)
    def dfs() -> Tuple[Union[int, float, List[int]]]:
        prev = {start_idx: None}
        stk = [(start_idx, float("inf"))]
        while stk:
            idx, flow = stk.pop()
            for idx2, mx_flow in capacities[idx].items():
                if idx2 in prev.keys(): continue
                flow2 = min(flow, mx_flow)
                prev[idx2] = idx
                if idx2 == end_idx: break
                stk.append((idx2, flow2))
            else: continue
            break
        else: return ()
        #print(prev)
        path = []
        while idx2 is not None:
            path.append(idx2)
            idx2 = prev[idx2]
        return (flow2, path[::-1])
    
    tot_flow = 0
    while True:
        pair = dfs()
        if not pair: break
        flow, path = pair
        tot_flow += flow
        #print(self.n, path)
        for i in range(len(path) - 1):
            idx1, idx2 = path[i], path[i + 1]
            capacities[idx1][idx2] -= flow
            if capacities[idx1][idx2] < eps:
                capacities[idx1].pop(idx2)
            #print(self.n, idx2)
            capacities[idx2][idx1] = capacities[idx2].get(idx1, 0) + flow
    if not return_poss_distribution:
        return tot_flow, None
    flow_edges = []
    vertices = []
    for idx1 in range(self.n):
        v1 = self.index2Vertex(idx1)
        vertices.append(v1)
        orig_capacity_dict = self.getAdjTotalWeightsIndex(idx1)
        for idx2, orig_capacity in orig_capacity_dict.items():
            v2 = self.index2Vertex(idx2)
            net_flow = orig_capacity - capacities[idx1].get(idx2, 0)
            if net_flow >= eps:
                flow_edges.append((v1, v2, net_flow))
    flow_graph = ExplicitWeightedDirectedGraph(vertices, flow_edges)
    #print(flows)
    return tot_flow, flow_graph

def fordFulkerson(self,\
        start: Hashable, end: Hashable, eps: float=10 ** -5,\
        return_poss_distribution: bool=True)\
        -> Tuple[Union[int, float, Optional[ExplicitWeightedDirectedGraph]]]:
    """
    Method implementing the Ford-Fulkerson algorithm to find the
    maximum flow from the vertex start to the vertex end for the
    network represented by this graph object, where each edge of
    the graph can support a flow capacity up to its weight (for
    weighted graphs) or 1 (for unweighted graphs) in the direction
    of the edge (for directed graphs) or in either direction (for
    undirected graphs). Finds the maximum flow and (if
    return_poss_distribution is given as True) the direct flow between
    adjacent vertices for a possible distribution of a flow of that
    size.
    
    Args:
        Required positional:
        start (hashable object): The vertex of the graph from which the
                flow starts (i.e. the source).
                Should be one of the vertices of the graph
        end (hashable object): The vertex of the graph from which the
                flow ends (i.e. the sink).
                Should be one of the vertices of the graph and distinct
                from start
        
        Optional named:
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
        return_poss_distribution (bool): If True, then finds a possible
                flow through the network that has the maximum flow from
                the vertex start to the vertex end, representing this
                flow by the total direct flow between each adjacent
                pair of vertices for which there is non-zero direct
                flow from the first to the second.
            Default: True
    
    Returns:
    2-tuple whose index 0 contains the size of the maximum possible
    flow from the vertex with index start_idx to the vertex with index
    end_idx in the network represented by graph, and whose index 1
    contains None if return_poss_distribution is given as False or an
    ExplicitWeightedDirectedGraph object representing a possible flow
    distribution through the network with that flow, where the vertices
    are the same as for the graph and with each vertex having the
    same index as in the graph, and the directed edge between a pair
    of vertices represents that there is direct flow from the first
    vertex of the pair to the second (i.e. flow straight from the
    first vertex to the second with no intervening vertices), with the
    weight of the edge representing the size of this direct flow.
    """
    #print(self.shape, self.grid.arr_flat)
    #print(self.n_state_runs_graph_starts)
    #print(self.n_state_runs_grid_starts)
    #print(start, end, self.vertex2Index(start), self.vertex2Index(end))
    tot_flow, flow_graph = self.fordFulkersonIndex(\
            self.vertex2Index(start), self.vertex2Index(end),\
            eps=eps, return_poss_distribution=return_poss_distribution)
    return (tot_flow, flow_graph)

def checkFlowValidIndex(graph: LimitedGraphTemplate,\
        tot_flow: Union[int, float],\
        flow_graph: ExplicitWeightedDirectedGraph,\
        start_idx: int, end_idx: int,\
        eps: float=10 ** -5, req_max_flow: bool=True,\
        allow_cycles: bool=True, indices_match: bool=False) -> bool:
    """
    Function assessing for a network reresented by the graph object
    graph whether the flow distribution given by input argument flows
    (in terms of the vertex indices) with total flow tot_flow from the
    vertex with index start_idx (the source) to the vertex at index
    end_idx (the sink) is a valid flow distribution satisfying
    the specified requirements.
    
    Args:
        Required positional:
        graph (class descending from ExplicitGraphTemplate): The graph
                representing the network for which the flow
                distribution is being assessed
        tot_flow (int or float): The total flow from the vertex with
                index start_idx to the vertex with index end_idx
        flows (ExplicitWeightedDirectedGraph): Weighted directed graph
                representing the flow distribution through the network
                represented by graph being assessed.
                The vertices of this graph must be exactly the same
                as those of graph, and the directed edge between a pair
                of vertices represents that there is direct flow from
                the first vertex of the pair to the second (i.e. flow
                straight from the first vertex to the second with no
                intervening vertices), with the weight of the edge
                representing the size of this direct flow.
        start_idx (int): The index of the vertex of the graph from
                which the flow starts (i.e. the source).
                Should be integer between 0 and (graph.n - 1) inclusive
        end_idx (int): The index of the vertex of the graph from
                which the flow ends (i.e. the sink).
                Should be integer between 0 and (graph.n - 1) inclusive
                and different to start_idx
        
        Optional named:
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
        req_max_flow (bool): If True then the flow must be the largest
                possible size flow from the vertex with index start_idx
                to the vertex with index end_idx
                This is assessed using the max flow/min cut theorem for
                networks
            Default: True
        allow_cycles (bool): If True, then the flow is permitted to
                contain cycles (i.e. there exists at least one
                path of one or more steps in the flow leading from a
                vertex to itself where each step in the path has
                non-zero flow), while if False then the flow is
                required not to contain cycles.
            Default: True
        indices_match (bool): If True, then each vertex in graph and
                flow_graph is guaranteed to have the same index in
                both.
            Default: False
    
    Returns:
    Boolean (bool) giving the value True if flow_graph represents a
    valid flow distribution in the network represented by graph from
    the vertex with index start_idx in graph to the vetex with index
    end_idx in graph satisfying the specified requirements (e.g. is a
    maximum flow and/or contains no cycles) or the value False
    otherwise (i.e. is not a valid flow or does not satisfy all of the
    requirements).
    """
    if tot_flow < eps or start_idx: return True
    net_flux = {start_idx: tot_flow, end_idx: -tot_flow}
    
    flow_to_graph_inds = (lambda idx: idx) if indices_match else\
            (lambda idx:\
            graph.vertex2Index(flow_graph.index2Vertex(idx)))
    
    idx2_cap_func = (lambda idx1: graph.getAdjTotalWeightsIndex(idx1))\
            if indices_match else\
            (lambda idx1: graph.getAdjTotalWeightsIndex(\
            graph.vertex2Index(flow_graph.index2Vertex(idx1))))
    
    for idx1 in range(flow_graph.n):
        idx2_flow_dict = flow_graph.getAdjTotalWeightsIndex(idx1)
        if not idx2_flow_dict: continue
        idx2_capacity_dict =\
                graph.getAdjTotalWeightsIndex(flow_to_graph_inds(idx1))
        for idx2, flow in idx2_flow_dict.items():
            if flow <= -eps or flow >= idx2_capacity_dict.get(idx2, 0) + eps:
                return False
            if flow < eps: continue
            net_flux[idx1] = net_flux.get(idx1, 0) - flow
            net_flux[idx2] = net_flux.get(idx2, 0) + flow
            for idx in (idx1, idx2):
                if abs(net_flux[idx]) < eps:
                    net_flux.pop(idx)
    
    if net_flux:
        #print("Inconsistent flux")
        #print(getattr(graph, graph.adj_name))
        #print()
        #print(start_idx, end_idx)
        #print()
        #print(flows)
        #print()
        #print(tot_flow)
        #print()
        #print(net_flux)
        return False
    if req_max_flow:
        graph_to_flow_inds = (lambda idx: idx) if indices_match else\
            (lambda idx:\
            flow_graph.vertex2Index(graph.index2Vertex(idx)))
        # Checking is a maximum flow using the min cut/max flow theorem
        # (i.e. when excluding edges for which the flow equals the
        # capacity, the source and sink should be disconnected)
        stk = [start_idx]
        seen = {start_idx}
        while stk:
            idx = stk.pop()
            flow_dict = flow_graph.getAdjTotalWeightsIndex(\
                    graph_to_flow_inds(idx))
            for idx2, w in graph.getAdjTotalWeightsIndex(idx).items():
                if idx2 in seen: continue
                w2 = w - flow_dict.get(graph_to_flow_inds(idx2), 0)
                if w2 < eps: continue
                if idx2 == end_idx: return False
                seen.add(idx2)
                stk.append(idx2)
    if allow_cycles: return True
    # Checking whether the directed graph represented by flows is
    # acyclic
    #edges = []
    #for idx1, idx2_dict in enumerate(flows):
    #    for idx2, flow in idx2_dict.items():
    #        if abs(flow) < eps: continue
    #        edges.append([idx1, idx2])
    #graph2 = ExplicitUnweightedDirectedGraph(range(graph.n), edges)
    top_sort = flow_graph.kahnIndex()#graph2.kahnIndex()
    #if not top_sort:
    #    print("Found cycle")
    #    print(start_idx, end_idx)
    #    print(flows)
    #print(top_sort)
    return bool(top_sort)

def checkFlowValid(graph: LimitedGraphTemplate,\
        tot_flow: Union[int, float],\
        flow_graph: ExplicitWeightedDirectedGraph,\
        start: Hashable, end: Hashable,\
        eps: float=10 ** -5, req_max_flow: bool=True,\
        allow_cycles: bool=False, indices_match: bool=False) -> bool:
    """
    Function assessing for a network reresented by the graph object
    graph whether the flow distribution given by input argument flows
    (in terms of the vertex indices) with total flow tot_flow from the
    vertex start (the source) to the vertex end (the sink) is a valid
    flow distribution satisfying the specified requirements.
    
    Args:
        Required positional:
        graph (class descending from ExplicitGraphTemplate): The graph
                representing the network for which the flow
                distribution is being assessed
        tot_flow (int or float): The total flow from the vertex start
                to the vertex end
        flows (ExplicitWeightedDirectedGraph): Weighted directed graph
                representing the flow distribution through the network
                represented by graph being assessed.
                The vertices of this graph must be exactly the same
                as those of graph, and the directed edge between a pair
                of vertices represents that there is direct flow from
                the first vertex of the pair to the second (i.e. flow
                straight from the first vertex to the second with no
                intervening vertices), with the weight of the edge
                representing the size of this direct flow.
        start (hashable object): The vertex of the graph from which the
                flow starts (i.e. the source).
                Should be one of the vertices of graph and flow_graph
        end (hashable object): The vertex of the graph from which the
                flow ends (i.e. the sink).
                Should be one of the vertices of graph and flow_graph
                and distinct from start
        
        Optional named:
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
        req_max_flow (bool): If True then the flow must be the largest
                possible size flow from the vertex start to the vertex
                end.
                This is assessed using the max flow/min cut theorem for
                networks.
            Default: True
        allow_cycles (bool): If True, then the flow is permitted to
                contain cycles (i.e. there exists at least one
                path of one or more steps in the flow leading from a
                vertex to itself where each step in the path has
                non-zero flow), while if False then the flow is
                required not to contain cycles.
            Default: True
        indices_match (bool): If True, then each vertex in graph and
                flow_graph is guaranteed to have the same index in
                both.
            Default: False
    
    Returns:
    Boolean (bool) giving the value True if flow_graph represents a
    valid flow distribution in the network represented by graph from
    the vertex start to the vetex end satisfying the specified
    requirements (e.g. is a maximum flow and/or contains no cycles) or
    the value False otherwise (i.e. is not a valid flow or does not
    satisfy all of the requirements).
    """
    #flows_idx = [{} for _ in range(graph.n)]
    #for v1, v2_dict in flows.items():
    #    idx1 = graph.vertex2Index(v1)
    #    for v2, flow in v2_dict.items():
    #        idx2 = graph.vertex2Index(v2)
    #        flows_idx[idx1][idx2] = flow
    start_idx = graph.vertex2Index(start)
    end_idx = graph.vertex2Index(end)
    return checkFlowValidIndex(graph, tot_flow, flow_graph, start_idx,\
            end_idx, eps=eps, req_max_flow=req_max_flow,\
            allow_cycles=allow_cycles, indices_match=indices_match)
