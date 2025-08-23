#! /usr/bin/env python

import sys
import os

from .limited_graph_types import LimitedGraphTemplate,\
        LimitedUnweightedGraphTemplate,\
        LimitedWeightedGraphTemplate,\
        LimitedUndirectedGraphTemplate,\
        LimitedDirectedGraphTemplate,\
        LimitedUnweightedUndirectedGraphTemplate,\
        LimitedWeightedUndirectedGraphTemplate,\
        LimitedUnweightedDirectedGraphTemplate,\
        LimitedWeightedDirectedGraphTemplate

from .explicit_graph_types import ExplicitGraphTemplate,\
        ExplicitUnweightedGraphTemplate,\
        ExplicitWeightedGraphTemplate,\
        ExplicitUndirectedGraphTemplate,\
        ExplicitDirectedGraphTemplate,\
        ExplicitUnweightedUndirectedGraph,\
        ExplicitWeightedUndirectedGraph,\
        ExplicitUnweightedDirectedGraph,\
        ExplicitWeightedDirectedGraph

from .grid_graph_types import Grid,\
        GridGraphTemplate,\
        GridUnweightedGraphTemplate,\
        GridWeightedGraphTemplate,\
        GridUndirectedGraphTemplate,\
        GridDirectedGraphTemplate,\
        GridUnweightedUndirectedGraph,\
        GridWeightedUndirectedGraph,\
        GridUnweightedDirectedGraph,\
        GridWeightedDirectedGraph

from .Methods import method_dicts

from .random_explicit_graph_generators import\
        randomExplicitWeightedDirectedGraphGenerator,\
        randomExplicitWeightedUndirectedGraphGenerator,\
        randomExplicitUnweightedDirectedGraphGenerator,\
        randomExplicitUnweightedUndirectedGraphGenerator

from .random_grid_graph_generators import\
        randomBinaryGridUnweightedUndirectedGraphGenerator

# Adding methods
for cls, method_dict in method_dicts.items():
    # print(f"Method/attribute: {k}")
    for method_nm, method in method_dict.items():
        setattr(cls, method_nm, method)
