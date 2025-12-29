#! /usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    List,
    Tuple,
    Dict,
    Optional,
    Union,
    Hashable,
    Callable,
)

if TYPE_CHECKING:
    from graph_classes.limited_graph_types import (
        LimitedGraphTemplate,
        LimitedDirectedGraphTemplate,
    )

from graph_classes.explicit_graph_types import (
    ExplicitWeightedDirectedGraph,
)

# TODO:
#  - Add documentation to the new functions findPossibleNetworkFlowWithVertexBalancesIndex()
#    and findPossibleNetworkFlowWithVertexBalancesAndEdgeFlowLowerBoundIndex()
#  - Add functions findPossibleNetworkFlowWithVertexBalances() and
#    findPossibleNetworkFlowWithVertexBalancesAndEdgeFlowLowerBound()
#  - Add unit tests that check circulation flows found by the new functions
#    and fordFulkerson for flows for multiple sources or sinks, with or without
#    source/sink flow restrictions (in test/test_network_flow_algorithms.py)
#  - Add functions that calculate network circulation flows with vertex balances
#    with or without edge flow lower bounds to get a flow with minimum cost,
#    for a costs per unit flow for the different edges

### Ford-Fulkerson algorithm for maximum flow through a network ###

def fordFulkersonIndex(
    self,
    start_inds: Dict[int, Union[int, float]],
    end_inds: Dict[int, Union[int, float]],
    edge_capacities_function_index: Optional[Callable[[LimitedGraphTemplate, int, int], Union[int, float]]]=None,
    eps: float=10 ** -5,
    return_poss_distribution: bool=True,
) -> Tuple[Union[int, float], Optional[ExplicitWeightedDirectedGraph]]:
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
        start_inds (dict): Dictionary whose keys are the indices of
                the vertices of the graph from which the flow starts
                (i.e. the sources), with corresponding value being
                the maximum total allowed flow from that vertex (or
                None if there is no such limit).
                The keys should be integers between 0 and (self.n - 1)
                inclusive
        end_inds (dict): Dictionary whose keys are the indices of
                the vertices of the graph from which the flow ends
                (i.e. the sinkss), with corresponding value being
                the maximum total flow into that vertex (or None if
                there is no such limit).
                The keys should be integers between 0 and (self.n - 1)
                inclusive, and none of the keys may also be keys
                in start_inds.
                
        Optional named:
        edge_capacities_function_index (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two integers (representing the indices of vertex 1
                and vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the maximum direct
                flow permitted from vertex 1 to vertex 2 for the given
                graph. Note that the result of this function may only
                be non-zero if there exists an edge in the graph from
                vertex 1 to vertex 2.
                If not specified (or given as None) the function using
                the graph total weight of edges from vertex 1 to vertex
                2 (or for unweighted graphs the number of edges from
                vertex 1 to vertex 2) is returned.
            Default: None
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
        return_poss_distribution (bool): If True, then finds a possible
                flow through the network that maximises the flow amount,
                representing this flow by a weighted directed graph
                whose vertices are the vertices with non-zero flow
                in and/or out and whose edges are those with non-zero
                flow, weighted according to the flow quantity through
                that edge.
            Default: True
    
    Returns:
    2-tuple whose index 0 contains the size of the maximum possible
    flow through the given network for the start and end vertices
    and maximum flows given by start_inds and end_inds, and whose index 1
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
    #print(start_inds, end_inds)
    if self.neg_weight_edge:
        raise NotImplementedError("The method fordFulkersonIndex() "
                "cannot be used for graphs with negative weight "
                "edges.")
    elif not set(start_inds.keys()).isdisjoint(end_inds.keys()):
        raise ValueError(
            "The keys of input argument dictionaries start_inds and "
            "end_inds in the method fordFulkersonIndex() cannot share "
            "any values.") 
    
    source_outgoing_edge_capacities = {x: (float("inf") if y is None else y) for x, y in start_inds.items()}
    sink_incoming_edge_capacities = {x: (float("inf") if y is None else y) for x, y in end_inds.items()}
    #print(f"start_inds = {start_inds}")
    #print(f"end_inds = {end_inds}")

    if edge_capacities_function_index is None:
        edge_capacities_function_index = lambda graph, v1_idx, v2_idx: graph.getAdjTotalWeightsIndex(v1_idx).get(v2_idx, 0)

    # Direct flow capacities between each pair of vertices that
    # share at least one edge
    
    capacities = [{} for _ in range(self.n)]
    for v1_idx in range(self.n):
        for v2_idx in self.getAdjIndex(v1_idx).keys():
            c = edge_capacities_function_index(self, v1_idx, v2_idx)
            if c: capacities[v1_idx][v2_idx] = c
    if return_poss_distribution:
        # storing the original capacities for use later
        orig_capacities = [dict(x) for x in capacities]
    #print(capacities)
    def findAugmentingPath() -> Tuple[Union[int, float, List[int]]]:
        for v0_idx in source_outgoing_edge_capacities.keys():
            prev = {v0_idx: None}
            stk = [(v0_idx, source_outgoing_edge_capacities[v0_idx])]
            while stk:
                idx, flow = stk.pop()
                for v2_idx, mx_flow in capacities[idx].items():
                    if v2_idx in prev.keys(): continue
                    flow2 = min(flow, mx_flow)
                    prev[v2_idx] = idx
                    if v2_idx in sink_incoming_edge_capacities.keys():
                        flow2 = min(flow2, sink_incoming_edge_capacities[v2_idx])
                        break
                    stk.append((v2_idx, flow2))
                else: continue
                break
            else: continue
            break
        else: return ()
        #print(prev)
        path = []
        while v2_idx is not None:
            path.append(v2_idx)
            v2_idx = prev[v2_idx]
        return (flow2, path[::-1])
    
    tot_flow = 0
    while True:
        pair = findAugmentingPath()
        if not pair: break
        flow, path = pair
        tot_flow += flow
        #print(self.n, path)
        for i in range(len(path) - 1):
            v1_idx, v2_idx = path[i], path[i + 1]
            capacities[v1_idx][v2_idx] -= flow
            if capacities[v1_idx][v2_idx] < eps:
                capacities[v1_idx].pop(v2_idx)
            #print(self.n, v2_idx)
            capacities[v2_idx][v1_idx] = capacities[v2_idx].get(v1_idx, 0) + flow
    if not return_poss_distribution:
        return tot_flow, None
    flow_edges = []
    vertices = []
    for v1_idx in range(self.n):
        v1 = self.index2Vertex(v1_idx)
        vertices.append(v1)
        for v2_idx, orig_capacity in orig_capacities[v1_idx].items():
            if v2_idx >= self.n: continue
            v2 = self.index2Vertex(v2_idx)
            net_flow = orig_capacity - capacities[v1_idx].get(v2_idx, 0)
            if net_flow >= eps:
                flow_edges.append((v1, v2, net_flow))
    flow_graph = ExplicitWeightedDirectedGraph(vertices, flow_edges)
    #print(flows)
    return tot_flow, flow_graph

def fordFulkerson(
    self,
    starts: Dict[Hashable, Optional[Union[int, float]]],
    ends: Dict[Hashable, Optional[Union[int, float]]],
    edge_capacities_function: Optional[Callable[[LimitedGraphTemplate, Hashable, Hashable], Union[int, float]]]=None,
    eps: float=10 ** -5,
    return_poss_distribution: bool=True,
) -> Tuple[Union[int, float, Optional[ExplicitWeightedDirectedGraph]]]:
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
        starts (dict): Dictionary whose keys are the vertices of
                the graph from which the flow starts (i.e. the sources),
                with corresponding value being the maximum total allowed
                flow from that vertex (or None if there is no such limit).
        ends (dict): Dictionary whose keys are the vertices of
                the graph from which the flow ends (i.e. the sinks),
                with corresponding value being the maximum total allowed
                flow from that vertex (or None if there is no such limit).
                The vertices should all be distinct from the vertices in
                starts.
        
        Optional named:
        edge_capacities_function (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two hashable objects (representing vertex 1 and
                vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the maximum direct
                flow permitted from vertex 1 to vertex 2 for the given
                graph.
                If not specified (or given as None) the function using
                the graph total weight of edges from vertex 1 to vertex
                2 (or for unweighted graphs the number of edges from
                vertex 1 to vertex 2) is returned.
            Default: None
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
    flow through the given network for the start and end vertices
    and maximum flows given by start_inds and end_inds, and whose index 1
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
    # review- should relabel the vertices of flow_graph to match those
    # of the original graph

    #print(self.shape, self.grid.arr_flat)
    #print(self.n_state_runs_graph_starts)
    #print(self.n_state_runs_grid_starts)
    #print(start, end, self.vertex2Index(start), self.vertex2Index(end))
    edge_capacities_function_index = None if edge_capacities_function is None else lambda graph, v1_idx, v2_idx: edge_capacities_function(graph.index2Vertex(v1_idx), graph.index2Vertex(v2_idx))
    #print(f"starts = {starts}")
    #print(f"ends = {ends}")
    tot_flow, flow_graph = self.fordFulkersonIndex(
        {self.vertex2Index(x): c for x, c in starts.items()},
        {self.vertex2Index(x): c for x, c in ends.items()},
        edge_capacities_function_index=edge_capacities_function_index,
        eps=eps,
        return_poss_distribution=return_poss_distribution,
    )
    
    return (tot_flow, flow_graph)

def checkFlowValidIndex(
    graph: LimitedGraphTemplate,
    tot_flow: Union[int, float],
    flow_graph: ExplicitWeightedDirectedGraph,
    start_inds: Dict[int, Union[int, float]],
    end_inds: Dict[int, Union[int, float]],
    edge_capacities_function_index: Optional[Callable[[LimitedGraphTemplate, int, int], Union[int, float]]]=None,
    edge_flow_lower_bound_function_index: Optional[Callable[[LimitedGraphTemplate, int, int], Union[int, float]]]=None,
    eps: float=10 ** -5,
    req_max_flow: bool=True,
    allow_cycles: bool=True,
    indices_match: bool=False,
) -> bool:
    """
    Function assessing for a network represented by the graph object
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
        start_inds (dict): Dictionary whose keys are the indices of
                the vertices of the graph from which the flow starts
                (i.e. the sources), with corresponding value being
                the maximum total allowed flow from that vertex (or
                None if there is no such limit).
                The keys should be integers between 0 and (self.n - 1)
                inclusive
        end_inds (dict): Dictionary whose keys are the indices of
                the vertices of the graph from which the flow ends
                (i.e. the sinkss), with corresponding value being
                the maximum total flow into that vertex (or None if
                there is no such limit).
                The keys should be integers between 0 and (self.n - 1)
                inclusive, and none of the keys may also be keys
                in start_inds.
        
        Optional named:
        edge_capacities_function_index (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two integers (representing the indices of vertex 1
                and vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the maximum direct
                flow permitted from vertex 1 to vertex 2 for the given
                graph. Note that the result of this function may only
                be non-zero if there exists an edge in the graph from
                vertex 1 to vertex 2.
                If not specified (or given as None) the function using
                the graph total weight of edges from vertex 1 to vertex
                2 (or for unweighted graphs the number of edges from
                vertex 1 to vertex 2) is returned.
            Default: None
        edge_flow_lower_bound_function_index (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two integers (representing the indices of vertex 1
                and vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the minimum flow
                from vertex 1 to vertex 2 that must be achieved for the
                flow to be for the given graph. Note that the result of this
                function may only be non-zero if there exists an edge in
                the graph from vertex 1 to vertex 2.
                If not specified (or given as None) the function returning
                zero for every pair of vertices is used (signifying that
                none of the edges has a lower bound for its flow).
            Default: None
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
    # Review- need to make sure the handling of negative flows is
    # appropriate (esp. should negative flows ever be allowed?)

    if tot_flow < eps: return True
    net_vertex_flow_balance = {}#{start_idx: tot_flow, end_idx: -tot_flow}
    
    flow_to_graph_inds = (lambda idx: idx) if indices_match else\
            (lambda idx:\
            graph.vertex2Index(flow_graph.index2Vertex(idx)))
    
    if edge_capacities_function_index is None:
        edge_capacities_function_index = lambda graph, v1_idx, v2_idx: graph.getAdjTotalWeightsIndex(v1_idx).get(v2_idx, 0)

    #idx2_cap_func = (lambda idx1: graph.getAdjTotalWeightsIndex(idx1))\
    #        if indices_match else\
    #        (lambda idx1: graph.getAdjTotalWeightsIndex(\
    #        graph.vertex2Index(flow_graph.index2Vertex(idx1))))
    
    # Checking that no flow exceeds the capacity
    for idx1 in range(flow_graph.n):
        idx2_flow_dict = flow_graph.getAdjTotalWeightsIndex(idx1)
        if not idx2_flow_dict: continue
        #idx2_capacity_dict = graph.getAdjTotalWeightsIndex(flow_to_graph_inds(idx1))
        for idx2, flow in idx2_flow_dict.items():
            #if flow <= -eps or flow >= idx2_capacity_dict.get(idx2, 0) + eps:
            #    return False
            if flow < -eps or flow > edge_capacities_function_index(graph, flow_to_graph_inds(idx1), flow_to_graph_inds(idx2)) + eps:
                #print("the flow through an edge is either negative or exceeds the capacity of the edge")
                #print(flow, edge_capacities_function_index(graph, flow_to_graph_inds(idx1), flow_to_graph_inds(idx2)))
                return False
            if flow < eps: continue
            net_vertex_flow_balance[idx1] = net_vertex_flow_balance.get(idx1, 0) - flow
            net_vertex_flow_balance[idx2] = net_vertex_flow_balance.get(idx2, 0) + flow
            for idx in (idx1, idx2):
                if abs(net_vertex_flow_balance[idx]) < eps:
                    net_vertex_flow_balance.pop(idx)
    
    # Checking that any edge with a flow lower bound has at least that much flow
    if edge_flow_lower_bound_function_index is not None:
        for v1_idx in range(graph.n):
            v1_in_graph = flow_graph.containsVertex(v1_idx)
            for v2_idx in graph.getAdjIndex(v1_idx):
                lb_flow = edge_flow_lower_bound_function_index(graph, v1_idx, v2_idx) - eps
                if lb_flow <= 0:
                    continue
                if not v1_in_graph:
                    #print(f"vertex index {v1_idx} is not in the graph")
                    return False
                if flow_graph.getAdj(v1_idx).get(v2_idx, 0) < lb_flow:
                    #print("the lower bound of flow through an edge is not met")
                    return False
        edge_flow_lower_bound_function_index = lambda graph, v1_idx, v2_idx: 0

    # Checking that each source and sink vertex does not
    # is indeed a source or sink respectively, that it
    # does not exceed its capacity, all other vertices
    # have zero net balance and that the total flow from
    # sources to sinks is consistent between the two
    # and matches the given total flow.
    # Also identifies the sources and sinks that are
    # saturated (i.e. their net inflow or outflow is
    # equal to its capacity) for use in establishing
    # whether the calculated flow is maximal.
    end_inds_total_flow = 0
    start_inds_total_flow = 0
    unsaturated_end_inds = set()
    unsaturated_start_inds = set()
    for idx, bal in net_vertex_flow_balance.items():
        if not bal: continue # should not happen
        if bal > 0:
            if idx not in end_inds.keys():
                # Positive flux for a non-sink vertex
                #print("a non-sink vertex has positive in-flux")
                return False
            end_inds_total_flow += bal
            if end_inds[idx] is not None:
                if bal > end_inds[idx]:
                    # Net flow exceeds the capacity of the vertex
                    #print("the net flow out of a sink vertex exceeds its capacity")
                    return False
                elif bal < end_inds[idx]:
                    unsaturated_end_inds.add(idx)
            continue
        if idx not in start_inds.keys():
            # Negative flux for a non-source vertex
            #print("a non-source vertex has negative in-flux")
            return False
        start_inds_total_flow -= bal
        if start_inds[idx] is not None:
            if -bal > start_inds[idx]:
                # Net flow exceeds the capacity of the vertex
                #print("the net flow into a source vertex exceeds its capacity")
                return False
            elif -bal < start_inds[idx]:
                unsaturated_start_inds.add(idx)

    if abs(start_inds_total_flow - end_inds_total_flow) > eps or abs(end_inds_total_flow - tot_flow) > eps:
        # Total flow inconsistent with either of the calculated end flows
        #print("total flow calculated is inconsistent with either the total source inflow or sink outflow")
        #print(f"calculated total flow = {tot_flow}, source total inflow = {start_inds_total_flow}, sink total outflow = {end_inds_total_flow}")
        return False

    #if net_flux:
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
    #    return False
    if req_max_flow and unsaturated_start_inds and unsaturated_end_inds:
        graph_to_flow_inds = (lambda idx: idx) if indices_match else\
            (lambda idx:\
            flow_graph.vertex2Index(graph.index2Vertex(idx)))
        # Checking is a maximum flow using the min cut/max flow theorem
        # (i.e. when excluding edges for which the flow equals the
        # capacity, the source and sink should be disconnected)
        stk = list(unsaturated_start_inds)
        seen = set(unsaturated_start_inds)
        while stk:
            idx = stk.pop()
            flow_dict = flow_graph.getAdjTotalWeightsIndex(\
                    graph_to_flow_inds(idx))
            for idx2, w in graph.getAdjTotalWeightsIndex(idx).items():
                if idx2 in seen: continue
                w2 = w - flow_dict.get(graph_to_flow_inds(idx2), 0)
                if w2 < eps: continue
                if idx2 in unsaturated_end_inds:
                    #print("the flow is not maximal, as a path containing only unsaturated edges from an unsaturated source vertex to an unsaturated sink has been found")
                    return False
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
    if not top_sort:
        #print("the flow contains flow cycles, which are not allowed")
        return False
    #if not top_sort:
    #    print("Found cycle")
    #    print(start_idx, end_idx)
    #    print(flows)
    #print(top_sort)
    return True

def checkFlowValid(
    graph: LimitedGraphTemplate,
    tot_flow: Union[int, float],
    flow_graph: ExplicitWeightedDirectedGraph,
    starts: Hashable,
    ends: Hashable,
    edge_capacities_function: Optional[Callable[[LimitedGraphTemplate, Hashable, Hashable], Union[int, float]]]=None,
    edge_flow_lower_bound_function: Optional[Callable[[LimitedGraphTemplate, Hashable, Hashable], Union[int, float]]]=None,
    eps: float=10 ** -5,
    req_max_flow: bool=True,
    allow_cycles: bool=False,
    indices_match: bool=False,
) -> bool:
    """
    Function assessing for a network reresented by the graph object
    graph whether the flow distribution given by input argument flows
    (in terms of the vertex indices) with total flow tot_flow from the
    vertices in starts (the sources) to the vertices in ends (the
    sinks) is a valid flow distribution satisfying the specified
    requirements.
    
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
        starts (dict): Dictionary whose keys are the vertices of
                the graph from which the flow starts (i.e. the sources),
                with corresponding value being the maximum total allowed
                flow from that vertex (or None if there is no such limit).
        ends (dict): Dictionary whose keys are the vertices of
                the graph from which the flow ends (i.e. the sinks),
                with corresponding value being the maximum total allowed
                flow from that vertex (or None if there is no such limit).
                The vertices should all be distinct from the vertices in
                starts.
        
        Optional named:
        edge_capacities_function (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two hashable objects (representing vertex 1 and
                vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the maximum direct
                flow permitted from vertex 1 to vertex 2 for the given
                graph.
                If not specified (or given as None) the function using
                the graph total weight of edges from vertex 1 to vertex
                2 (or for unweighted graphs the number of edges from
                vertex 1 to vertex 2) is returned.
            Default: None
        edge_flow_lower_bound_function (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two hashable objects (representing vertex 1 and vertex
                2 respectively in the graph), with the returned value of
                the function being a non-negative real number (int or float)
                representing the minimum flow from vertex 1 to vertex 2 that
                must be achieved for the flow to be for the given graph.
                If not specified (or given as None) the function returning
                zero for every pair of vertices is used (signifying that
                none of the edges has a lower bound for its flow).
            Default: None
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
    the vertices starts to the vetices ends satisfying the specified
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
    start_inds = {graph.vertex2Index(x): c for x, c in starts.items()}
    end_inds = {graph.vertex2Index(x): c for x, c in ends.items()}
    edge_capacities_function_index = None if edge_capacities_function is None else lambda graph, v1_idx, v2_idx: edge_capacities_function(graph.index2Vertex(v1_idx), graph.index2Vertex(v2_idx))
    edge_flow_lower_bound_function_index = None if edge_flow_lower_bound_function is None else lambda graph, v1_idx, v2_idx: edge_flow_lower_bound_function(graph.index2Vertex(v1_idx), graph.index2Vertex(v2_idx))
    return checkFlowValidIndex(
        graph,
        tot_flow,
        flow_graph,
        start_inds,
        end_inds,
        edge_capacities_function_index=edge_capacities_function_index,
        edge_flow_lower_bound_function_index=edge_flow_lower_bound_function_index,
        eps=eps,
        req_max_flow=req_max_flow,
        allow_cycles=allow_cycles,
        indices_match=indices_match
    )

def findPossibleNetworkFlowWithVertexBalancesIndex(
    self,
    vertex_balance_function_index: Callable[[LimitedGraphTemplate, int], Union[int, float]],
    edge_capacities_function_index: Optional[Callable[[LimitedGraphTemplate, int, int], Union[int, float]]]=None,
    eps: float=10 ** -5,
) -> Optional[ExplicitWeightedDirectedGraph]:
    """
    Method finding a possible flow through a network for which
    any ordered pair of vertices has a flow capacity specified by
    the function edge_capacities_function_index() and whose
    vertices have flow balances specified by the function
    edge_capacities_function_index(), if any such flow exists.

    The flow capacity of an ordered pair of vertices in the network
    is the maximum amount of direct flow (i.e. flow that does not
    pass through any other vertices) from the first of the pair to
    the second that is allowed.

    The flow balance of a vertex in the network is the total amount
    of flow from other vertices into this vertex minus the total
    amount of flow from this vertex to another vertex.

    This function adapts this problem into a maximum network flow
    problem for a network with sources and sinks with maximum
    capacities corresponding to the vertex balances, which is then
    solved by the function fordFulkersonIndex(). This result is
    a solution to the original problem if and only if the sources
    and sinks are fully saturated (i.e. the flow from the sources
    and into the sinks is equal to its capacity), otherwise no
    flow solving the original problem exists.
    
    Args:
        Required positional:
        vertex_balance_function_index (callable or None): A function
                taking as inputs a graph with finite vertices
                and an integer (representing the index of a vertex
                in the graph), with the returned value of the function
                being a real number (int or float) representing the
                flow balance requirement of that vertex (i.e. the total
                direct flow from other vertices into this vertex minus
                the total direct flow from this vertex to any other
                vertex that must be achieved by the returned flow).
                
        Optional named:
        edge_capacities_function_index (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two integers (representing the indices of vertex 1
                and vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the maximum direct
                flow permitted from vertex 1 to vertex 2 for the given
                graph. Note that the result of this function may only
                be non-zero if there exists an edge in the graph from
                vertex 1 to vertex 2.
                If not specified (or given as None) the function using
                the graph total weight of edges from vertex 1 to vertex
                2 (or for unweighted graphs the number of edges from
                vertex 1 to vertex 2) is returned.
            Default: None
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
    
    Returns:
    If such a flow exists, an ExplicitWeightedDirectedGraph object
    representing a possible flow distribution through the network
    satisfying the flow requirements, where the vertices are the same
    as for the graph and with each vertex having the same index as in
    the graph, and the directed edge between a pair of vertices
    represents that there is direct flow from the first vertex of the
    pair to the second (i.e. flow straight from the first vertex to
    the second with no intervening vertices), with the weight of the
    edge representing the size of this direct flow.
    If no such flow exists, returns None.
    """

    # Positive balance means the vertex is a sink, negative
    # a source
    start_inds = {}
    end_inds = {}
    
    for v_idx in range(self.n):
        bal = vertex_balance_function_index(v_idx)
        if not bal: continue
        elif bal > 0: end_inds[v_idx] = bal
        else: start_inds[v_idx] = -bal
    starts_capacity = sum(start_inds.values())
    ends_capacity = sum(end_inds.values())
    if starts_capacity != ends_capacity:
        # Overall net vertex balance must be zero
        return None
    flow, res = self.fordFulkersonIndex(
        start_inds,
        end_inds,
        edge_capacities_function_index=edge_capacities_function_index,
        eps=eps,
        return_poss_distribution=True,
    )
    if flow != starts_capacity:
        # for a valid network circulation, the source and
        # sink vertices must be fully saturated
        return None
    return res

def findPossibleNetworkFlowWithVertexBalancesAndEdgeFlowLowerBoundIndex(
    self,
    vertex_balance_function_index: Callable[[LimitedGraphTemplate, int], Union[int, float]]=None,
    edge_flow_lower_bound_function_index: Callable[[LimitedDirectedGraphTemplate, int, int], Union[int, float]]=None,
    edge_capacities_function_index: Optional[Callable[[LimitedGraphTemplate, int, int], Union[int, float]]]=None,
    eps: float=10 ** -5,
) -> Optional[ExplicitWeightedDirectedGraph]:
    """
    Method finding a possible flow through a network for which
    any ordered pair of vertices has a flow capacity specified by
    the function edge_capacities_function_index(), whose
    vertices have flow balances specified by the function
    edge_capacities_function_index() and for which the minimum
    direct flow between any pair of vertices is specified by the
    function edge_flow_lower_bound_function_index(), if any such
    flow exists.

    The flow capacity of an ordered pair of vertices in the network
    is the maximum amount of direct flow (i.e. flow that does not
    pass through any other vertices) from the first of the pair to
    the second that is allowed.

    The flow balance of a vertex in the network is the total amount
    of flow from other vertices into this vertex minus the total
    amount of flow from this vertex to another vertex.

    This function adapts this problem into a maximum network flow
    problem for a network with sources and sinks with maximum
    capacities corresponding to the vertex balances, which is then
    solved by the function fordFulkersonIndex(). This result is
    a solution to the original problem if and only if the sources
    and sinks are fully saturated (i.e. the flow from the sources
    and into the sinks is equal to its capacity), otherwise no
    flow solving the original problem exists.
    
    Args:
        Optional named:
        vertex_balance_function_index (callable or None): A function
                taking as inputs a graph with finite vertices
                and an integer (representing the index of a vertex
                in the graph), with the returned value of the function
                being a real number (int or float) representing the
                flow balance requirement of that vertex (i.e. the total
                direct flow from other vertices into this vertex minus
                the total direct flow from this vertex to any other
                vertex that must be achieved by the returned flow).
                If not specified (or given as None) the function returning
                0 for all vertex indices is used (i.e. all vertices have
                zero net flow balance).
            Default: None
        edge_flow_lower_bound_function_index (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two integers (representing the indices of vertex 1
                and vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the minimum direct
                flow from vertex 1 to vertex 2 for the returned flow. Note
                that the result of this function may only be non-zero if
                there exists an edge in the graph from vertex 1 to vertex 2.
                If not specified (or given as None) the function using
                returning zero for every ordered pair of graph vertex
                indices is returned (i.e. there is no lower bound on the
                direct flow between any pair of vertices in the graph).
            Default: None
        edge_capacities_function_index (callable or None): If specified,
                a function taking as inputs a graph with finite vertices
                and two integers (representing the indices of vertex 1
                and vertex 2 respectively in the graph), with the
                returned value of the function being a non-negative
                real number (int or float) representing the maximum direct
                flow permitted from vertex 1 to vertex 2 for the given
                graph. Note that the result of this function may only
                be non-zero if there exists an edge in the graph from
                vertex 1 to vertex 2.
                If not specified (or given as None) the function using
                the graph total weight of edges from vertex 1 to vertex
                2 (or for unweighted graphs the number of edges from
                vertex 1 to vertex 2) is returned.
            Default: None
        eps (float): Small number representing the tolerance for
                float equality (i.e. two numbers that differ by
                less than this number are considered to be equal).
            Default: 10 ** -5
    
    Returns:
    If such a flow exists, an ExplicitWeightedDirectedGraph object
    representing a possible flow distribution through the network
    satisfying the flow requirements, where the vertices are the same
    as for the graph and with each vertex having the same index as in
    the graph, and the directed edge between a pair of vertices
    represents that there is direct flow from the first vertex of the
    pair to the second (i.e. flow straight from the first vertex to
    the second with no intervening vertices), with the weight of the
    edge representing the size of this direct flow.
    If no such flow exists, returns None.
    """
    # edge_lower_bound_function cannot return negative values

    if edge_flow_lower_bound_function_index is None:
        return self.findPossibleNetworkFlowWithVertexBalancesIndex(
            vertex_balance_function_index,
            edge_capacities_function_index=edge_capacities_function_index,
            eps=eps,
        )

    vertex_bal_adjust = {}

    for v1_idx in range(self.n):
        for v2_idx in self.getAdjIndex(v1_idx).keys():
            lb = edge_flow_lower_bound_function_index(v1_idx, v2_idx)
            if not lb: continue
            vertex_bal_adjust[v1_idx] = vertex_bal_adjust.get(v1_idx, 0) + lb
            if not vertex_bal_adjust[v1_idx]: vertex_bal_adjust.pop(v1_idx)
            vertex_bal_adjust[v2_idx] = vertex_bal_adjust.get(v1_idx, 0) - lb
            if not vertex_bal_adjust[v2_idx]: vertex_bal_adjust.pop(v2_idx)

    if vertex_balance_function_index is None:
        vertex_balance_function_index2 = lambda graph, v_idx: vertex_bal_adjust.get(v_idx, 0)
    else:
        vertex_balance_function_index2 = lambda graph, v_idx: vertex_balance_function_index(graph, v_idx) + vertex_bal_adjust.get(v_idx, 0)
    
    if edge_capacities_function_index is None:
        edge_capacities_function_index = lambda graph, v1_idx, v2_idx: graph.getAdjTotalWeightsIndex(v1_idx).get(v2_idx, 0)
    
    edge_capacities_function_index2 = lambda graph, v1_idx, v2_idx: edge_capacities_function_index(graph, v1_idx, v2_idx) - edge_flow_lower_bound_function_index(graph, v1_idx, v2_idx)

    flow_graph0 = self.findPossibleNetworkFlowWithVertexBalancesIndex(
        vertex_balance_function_index2,
        edge_capacities_function_index=edge_capacities_function_index2,
        eps=eps,
    )
    if flow_graph0 is None: return None # No flow that fits the constraints is possible

    # Review- try to implement mechanism for changing weights of existing
    # weighted graphs rather than creating a new graph

    
    flow_edge_dict = {}
    #vertices = []
    for v1_idx0 in range(flow_graph0.n):
        v1_idx = flow_graph0.index2Vertex(v1_idx0)
        #vertices.append(v1)
        for v2_idx0, flow in flow_graph0.getAdjTotalWeightsIndex(v1_idx).items():
            if not (0 <= v2_idx0 < flow_graph0.n): continue
            if abs(flow) <= eps: continue
            v2_idx = self.index2Vertex(v2_idx)
            flow_edge_dict.setdefault(v1_idx, {})
            flow_edge_dict[v1_idx][v2_idx] = flow_edge_dict[v1_idx].at(v2_idx) + flow
    for v1_idx in range(self.n):
        for v2_idx in self.getAdjIndex(v1_idx).keys():
            lb_flow = edge_flow_lower_bound_function_index(self, v1_idx, v2_idx)
            if not lb_flow: continue
            flow_edge_dict.setdefault(v1_idx, {})
            flow_edge_dict[v1_idx].setdefault(v2_idx, 0)
            flow_edge_dict[v1_idx][v2_idx] = flow_edge_dict[v1_idx].get(v2_idx, 0) + lb_flow
            if not flow_edge_dict[v1_idx][v2_idx]:
                flow_edge_dict[v1_idx].pop(v2_idx)
                if not flow_edge_dict[v1_idx]: flow_edge_dict.pop(v1_idx)
    vertices = []
    flow_edges = []
    seen_vertices = set()
    for v1 in flow_edge_dict.keys():
        if not flow_edge_dict[v1_idx]: continue
        if v1 not in seen_vertices:
            seen_vertices.add(v1)
            vertices.append(v1)
        for v2, flow in flow_edge_dict[v1_idx].items():
            if not abs(flow) <= eps: continue
            if v2 not in seen_vertices:
                seen_vertices.add(v2)
                vertices.append(v2)
            flow_edges.append((v1, v2, flow) if flow > 0 else (v2, v1, -flow))

    flow_graph = ExplicitWeightedDirectedGraph(vertices, flow_edges)
    return flow_graph