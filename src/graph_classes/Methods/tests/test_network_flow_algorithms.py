#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Hashable,
    Any,
)

import math
import unittest


"""
pkg_name = "Graph_classes"
path = os.path.dirname(os.path.realpath(__file__))
path_lst = path.split("/")
while path_lst and path_lst[-1] != pkg_name:
    path_lst.pop()
if path_lst: path_lst.pop()
pkg_path = "/".join(path_lst)
sys.path.append(pkg_path)
"""

from graph_classes.Methods.tests.graph_method_test_templates import (
    toString,
    TestGraphMethodTemplate,
)
        

from graph_classes.Methods.network_flow_algorithms import (
    checkFlowValid,
)
        
from graph_classes.utils import (
    randomKTupleGenerator,
)

from graph_classes.explicit_graph_types import (
    ExplicitWeightedDirectedGraph,
)

from graph_classes.grid_graph_types import (
    Grid,
    GridUnweightedUndirectedGraph,
)

pkg_name = "graph_classes"
module_name = "network_flow_algorithms"

def runAllTests() -> None:
    unittest.main()
    return

class TestFordFulkerson(TestGraphMethodTemplate):
    method_name = "fordFulkerson"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitWeightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ())),\
                        "opts": [{"args": (0, 1), "result": 0}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1, 5),))),\
                        "opts": [{"args": (0, 1), "result": 5}]},
                # From https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
                {"obj_func": (lambda cls2: cls2(range(6), ((0, 1, 16),\
                        (0, 2, 13), (1, 2, 10), (1, 3, 12), (2, 1, 4),\
                        (2, 4, 14), (3, 2, 9), (3, 5, 20), (4, 3, 7),\
                        (4, 5, 4)))),\
                        "opts": [{"args": (0, 5), "result": 23}]},
            ],
            GridUnweightedUndirectedGraph: [
                {"obj_func": (lambda cls2: cls2(Grid(2,\
                        [[False, False], [False, False]]),\
                        **cls.binaryGridGraphKwargs(0))),\
                        "opts": [{"args": (((0, 0), 0), ((1, 1), 0)), "result": 2},\
                        {"args": (((0, 0), 0), ((1, 0), 0)), "result": 2}]},
                {"obj_func": (lambda cls2: cls2(Grid(2,\
                        [[False, False], [False, False]]),\
                        **cls.binaryGridGraphKwargs(1))),\
                        "opts": [{"args": (((0, 0), 0), ((1, 1), 0)), "result": 3},\
                        {"args": (((0, 0), 0), ((1, 0), 0)), "result": 3}]},
                {"obj_func": (lambda cls2: cls2(Grid(2,\
                        [[False, True], [False, False]]),\
                        **cls.binaryGridGraphKwargs(0))),\
                        "opts": [{"args": (((0, 0), 0), ((1, 1), 0)), "result": 1},\
                        {"args": (((0, 0), 0), ((1, 0), 0)), "result": 1}]},
                {"obj_func": (lambda cls2: cls2(Grid(2,\
                        [[False, True], [True, False]]),\
                        **cls.binaryGridGraphKwargs(0))),\
                        "opts": [{"args": (((0, 0), 0), ((1, 1), 0)), "result": 0}]},
                {"obj_func": (lambda cls2: cls2(Grid(2,\
                        [[False, True], [True, False]]),\
                        **cls.binaryGridGraphKwargs(1))),\
                        "opts": [{"args": (((0, 0), 0), ((1, 1), 0)), "result": 1}]},
            ]
        }
        return known_good_results
    
    @classmethod
    def knownError(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_err = {
            ExplicitWeightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1, -1),))),\
                        "opts": [{"args": (0, 1), "err": NotImplementedError}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1, 1),))),\
                        "opts": [{"args": (0, 2), "err": KeyError}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1, 1),))),\
                        "opts": [{"args": (-1, 1), "err": KeyError}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1, 1),))),\
                        "opts": [{"args": (0, 0), "err": ValueError}]},
            ]
        }
        return known_err
    
    res_type_alias_lst = [Tuple[Union[int, float,\
            ExplicitWeightedDirectedGraph]], Union[int, float]]
    res_type_alias_lst.append(Union[res_type_alias_lst[0], res_type_alias_lst[1]])
    
    def resultEqualityFunction(self, res1: Union[int, float],\
            res2: Union[int, float]) -> bool:
        if hasattr(res1, "__getitem__"):
            res1 = res1[0]
        if hasattr(res2, "__getitem__"):
            res2 = res2[0]
        return abs(res1 - res2) < self.eps
    
    def resultString(self, res: Union[int, float],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        v1, v2 = method_args
        stem = "a maximum total flow through the network from vertex "\
                f"{toString(v1)} to vertex {toString(v2)} of "
        if not hasattr(res, "__getitem__"):
            return f"{stem}{res}"
        return f"{stem}{res[0]} with a possible flow "\
                "distribution represented by the "\
                f"{res[1].graph_type_name} object with adjacency "\
                f"dictionary:\n{self.graphAdjString(res[1])}\n"
    
    #def testFunction(self, res: Any, args: Optional[Tuple[Any]]=None,\
    #        kwargs: Optional[Dict[str, Any]]=None) -> Tuple[bool, str]:
    
    def methodResultTest(self, obj: Any,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[Union[int, float]]=None)\
            -> bool:
        
        v1, v2 = method_args
        eps = method_kwargs.get("eps", self.eps)\
                if method_kwargs is not None else self.eps
            
        test_func = lambda res: (checkFlowValid(obj, *res, v1, v2,\
                eps=eps, req_max_flow=True, allow_cycles=True,\
                indices_match=True),\
                "which are inconsistent with each ther and/or the "\
                "graph, or is not a maximum flow between these two "\
                "vertices")
        
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=True)
        return res[0]#abs(res[0]) >= self.eps
    
    def test_fordFulkerson_known_good_flows(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_fordFulkerson_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_fordFulkerson_weighted_random(self) -> None:
        n_graph_per_opt = 10
        n_pairs_per_graph = 10
        randomise_indices = True
    
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (1, 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        method_args_opts = (0, 1)
        rand_generator_func = lambda graph:\
                (tuple(graph.index2Vertex(idx) for idx in inds)\
                for inds in\
                randomKTupleGenerator(graph.n, 2,\
                mx_n_samples=n_pairs_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=False))
        
        #method_vertex_count = 2
        #method_vertex_opts_kwargs = {\
        #        "mx_n_samples": n_pairs_per_graph,\
        #        "allow_index_repeats": False,\
        #        "allow_tuple_repeats": False, "nondecreasing": False}
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        res = self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                wt_rng=wt_rng,\
                wt_cls_opts=wt_cls_opts,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices,\
                method_args_opts=method_args_opts,\
                rand_generator_func=rand_generator_func,\
                rand_first=False)
        return
    
    def test_fordFulkerson_unweighted_random(self) -> None:
        n_graph_per_opt = 10
        n_pairs_per_graph = 10
        randomise_indices = True
    
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        
        method_args_opts = (0, 1)
        rand_generator_func = lambda graph:\
                (tuple(graph.index2Vertex(idx) for idx in inds)\
                for inds in\
                randomKTupleGenerator(graph.n, 2,\
                mx_n_samples=n_pairs_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=False))
        
        #method_vertex_count = 2
        #method_vertex_opts_kwargs = {\
        #        "mx_n_samples": n_pairs_per_graph,\
        #        "allow_index_repeats": False,\
        #        "allow_tuple_repeats": False, "nondecreasing": False}
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        res = self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices,\
                method_args_opts=method_args_opts,\
                rand_generator_func=rand_generator_func,\
                rand_first=False)
        return
    
    def test_fordFulkerson_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 5
        n_pairs_per_graph = 10
        
        n_dim_rng = (1, 3)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(400 ** (1 / n_dim)))] * n_dim)
        p_wall_rng = (0.2, 0.8)
        n_diag_rng_func = lambda n_dim: (0, n_dim - 1)
        
        method_args_opts = (0, 1)
        rand_generator_func = lambda graph:\
                (tuple(graph.index2Vertex(idx) for idx in inds)\
                for inds in\
                randomKTupleGenerator(graph.n, 2,\
                mx_n_samples=n_pairs_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=False))
        
        res = self.randomBinaryGridUnweightedGraphMethodTestsTemplate(\
                n_graph_per_opt=n_graph_per_opt,\
                n_dim_rng=n_dim_rng,\
                shape_rng_func=shape_rng_func,\
                p_wall_rng=p_wall_rng,\
                n_diag_rng_func=n_diag_rng_func,\
                method_args_opts=method_args_opts,\
                rand_generator_func=rand_generator_func,\
                rand_first=False)
        #print(res)
        return
    
if __name__ == "__main__":
    #print("testing network_flow_algorithms")
    runAllTests()

