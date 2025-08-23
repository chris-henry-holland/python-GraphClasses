#! /usr/bin/env python

import importlib
import os
import sys

from .method_loader import loadMethodsMultipleModules

pkg_name = "Graph_classes"
path = os.path.dirname(os.path.realpath(__file__))
path_lst = path.split("/")
while path_lst and path_lst[-1] != pkg_name:
    path_lst.pop()
if path_lst: path_lst.pop()
pkg_path = "/".join(path_lst)
sys.path.append(pkg_path)

from graph_classes.generic_graph_types import GenericGraphTemplate,\
        GenericUnweightedGraphTemplate
from graph_classes.limited_graph_types import LimitedGraphTemplate,\
        LimitedUndirectedGraphTemplate,\
        LimitedDirectedGraphTemplate,\
        LimitedWeightedUndirectedGraphTemplate
        

methodname_dicts = {}

methodname_dicts["path_finding_algorithms"] = {
    LimitedGraphTemplate: [
        ("_hierholzerIndex", "_hierholzerIndex_abstract"),
        "hierholzerIndex",
        "hierholzer",
    ],
    LimitedDirectedGraphTemplate: [
        ("_hierholzerIndex", "_hierholzerIndex_directed"),
    ],
    LimitedUndirectedGraphTemplate: [
        ("_hierholzerIndex", "_hierholzerIndex_undirected"),
    ],
    GenericGraphTemplate: [
        "_pathIndex2Vertex",
        "_transitionToBFS",
        "_dijkstraUnidirectionalIndex",
        "_dijkstraBidirectionalIndex",
        "dijkstraIndex",
        "dijkstra",
        "_aStarUnidirectionalIndex",
        "_aStarBidirectionalIndex",
        "aStarIndex",
        "aStar",
        "_findShortestPathIndex",
        "findShortestPathIndex",
        "findShortestPath",
    ],
    GenericUnweightedGraphTemplate: [
        "_bredthFirstSearchUnidirectionalIndex",
        "_bredthFirstSearchBidirectionalIndex",
        "bredthFirstSearchIndex",
        "bredthFirstSearch",
    ],
    LimitedGraphTemplate: [
        "_fromSourcesDistPathIndex2Vertex",
        "_fromSourcesDistIndex2Vertex",
        "_dijkstraFromSourcesPathfinderIndex",
        "dijkstraFromSourcesPathfinderIndex",
        "dijkstraFromSourcesPathfinder",
        "_shortestPathFasterAlgorithmDistancesIndex",
        "shortestPathFasterAlgorithmDistancesIndex",
        "shortestPathFasterAlgorithmDistances",
        "_shortestPathFasterAlgorithmPathfinderIndex",
        "shortestPathFasterAlgorithmPathfinderIndex",
        "shortestPathFasterAlgorithmPathfinder",
        "_allPairsDistPathIndex2Vertex",
        "floydWarshallDistancesIndex",
        "floydWarshallDistances",
        "floydWarshallPathfinderIndex",
        "floydWarshallPathfinder",
        "johnsonIndex",
        "johnson",
    ],
}

methodname_dicts["bridge_and_articulation_point_algorithms"] = {
    LimitedUndirectedGraphTemplate: [
        "tarjanBridgeIndex",
        "tarjanBridge",
        "tarjanArticulationBasicIndex",
        "tarjanArticulationBasic",
        "tarjanArticulationFullIndex",
        "tarjanArticulationFull",
    ]
}

methodname_dicts["minimum_spanning_tree_algorithms"] = {
    LimitedWeightedUndirectedGraphTemplate: [
        "kruskalIndex",
        "kruskal",
    ]
}

methodname_dicts["topological_sort_algorithms"] = {
    LimitedDirectedGraphTemplate: [
        "kahnIndex",
        "kahn",
        "kahnLayeringIndex",
        "kahnLayering",
    ]
}

methodname_dicts["strongly_connected_component_algorithms"] = {
    LimitedDirectedGraphTemplate: [
        "kosarajuIndex",
        "kosaraju",
        "tarjanSCCIndex",
        "tarjanSCC",
        "condenseSCCIndex",
        "condenseSCC",
    ]
}

methodname_dicts["network_flow_algorithms"] = {
    LimitedGraphTemplate: [
        "fordFulkersonIndex",
        "fordFulkerson",
    ]
}


# Loading the methods themselves into a dictionary
method_dicts = loadMethodsMultipleModules(methodname_dicts)
        

"""
for module_name in method_module_names:
    module = importlib.import_module(
        ".{}".format(module_name), package=__name__
    )
    method_dict_update = getattr(module, method_import_dict_name)
    for k, v in method_dict_update.items():
        method_dicts.setdefault(k, {})
        if k != "global_dicts":
            method_dicts[k].update(v)
            continue
        # Below, k = "global_dicts"
        for k2, v2 in v.items():
            method_dicts[k].setdefault(k2, {})
            method_dicts[k][k2].update(v2)

method_import_dict = methodImportPrep(
    methodname_dict=methodname_dict, convmethodname_dict=convmethodname_dict
)
"""
