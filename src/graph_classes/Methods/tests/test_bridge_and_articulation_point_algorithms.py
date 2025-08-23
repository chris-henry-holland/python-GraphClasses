#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
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
    TestGraphMethodTemplate,
)

from graph_classes.Methods.bridge_and_articulation_point_algorithms import (
    checkBridges,
    checkArticulationBasic,
)

from graph_classes import (
    ExplicitUnweightedUndirectedGraph,
)

# TODO- Implement TestTarjanArticulationFull

pkg_name = "graph_classes"
module_name = "bridge_and_articulation_point_algorithms"

def runAllTests() -> None:
    unittest.main()
    return

class TestTarjanBridge(TestGraphMethodTemplate):
    method_name = "tarjanBridge"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedUndirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ())),\
                        "opts": [{"result": set()}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),))),\
                        "opts": [{"result": {(0, 1)}}]},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0)))),\
                        "opts": [{"result": set()}]},
                # The following 3 examples are from
                # https://www.geeksforgeeks.org/bridge-in-a-graph/
                {"obj_func": (lambda cls2: cls2(range(5),\
                        ((0, 1), (0, 2), (0, 3), (1, 2), (3, 4)))),\
                        "opts": [{"result": {(0, 3), (3, 4)}}]},
                {"obj_func": (lambda cls2: cls2(range(7), ((0, 1), (0, 2),\
                        (1, 2), (1, 3), (1, 4), (1, 6), (3, 5), (4, 5)))),\
                        "opts": [{"result": {(1, 6)}}]},
                {"obj_func": (lambda cls2: cls2(range(4),\
                        ((0, 1), (1, 2), (2, 3)))),\
                        "opts": [{"result": {(0, 1), (1, 2), (2, 3)}}]},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Set[Tuple[Hashable]]]
    
    def resultEqualityFunction(self, res1: Set[Tuple[Hashable]],\
            res2: Set[Tuple[Hashable]]) -> bool:
        # Assumes no repeated edges in either res1 or res2, including
        # the same edge in reverse (i.e. for any two distinct vertices
        # v1 and v2, res1 and res2 are assumed never to contain both
        # (v1, v2) and (v2, v1) or either of these more than once).
        if len(res1) != len(res2): return False
        bridge_dict1 = {}
        for v1, v2 in res1:
            bridge_dict1.setdefault(v1, set())
            bridge_dict1[v1].add(v2)
            bridge_dict1.setdefault(v2, set())
            bridge_dict1[v2].add(v1)
        for v1, v2 in res2:
            if v2 not in bridge_dict1.get(v1, set()) or\
                    v1 not in bridge_dict1.get(v2, set()):
                return False
        return True
    
    def resultString(self, res: Set[Tuple[Hashable]],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return f"the bridges of the graph of:\n{res}\n"
    
    def methodResultTest(self, obj: Any,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[Set[Tuple[Hashable]]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkBridges(obj, res, check_all=True),\
                "which is not the set of bridges of the graph")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def test_tarjanBridge_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_tarjanBridge_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_tarjanBridge_weighted_random(self) -> None:
        n_graph_per_opt = 50
    
        directed_opts = (False,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (1, 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                wt_rng=wt_rng,\
                wt_cls_opts=wt_cls_opts,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts)
        return
    
    def test_tarjanBridge_unweighted_random(self) -> None:
        n_graph_per_opt = 50
    
        directed_opts = (False,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts)
        return
    
    def test_tarjanBridge_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 10
        
        n_dim_rng = (1, 3)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(400 ** (1 / n_dim)))] * n_dim)
        p_wall_rng = (0.2, 0.8)
        n_diag_rng_func = lambda n_dim: (0, n_dim - 1)
        
        res = self.randomBinaryGridUnweightedGraphMethodTestsTemplate(\
                n_graph_per_opt=n_graph_per_opt,\
                n_dim_rng=n_dim_rng,\
                shape_rng_func=shape_rng_func,\
                p_wall_rng=p_wall_rng,\
                n_diag_rng_func=n_diag_rng_func)
        #print(res)
        return

class TestTarjanArticulationBasic(TestGraphMethodTemplate):
    method_name = "tarjanArticulationBasic"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedUndirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(3), ())),\
                        "opts": [{"result": set()}]},
                {"obj_func": (lambda cls2: cls2(range(3), ((0, 1), (1, 2)))),\
                        "opts": [{"result": {1}}]},
                {"obj_func": (lambda cls2: cls2(range(6),\
                        ((0, 1), (0, 2), (0, 3), (1, 2), (3, 4),\
                        (3, 5)))),\
                        "opts": [{"result": {0, 3}}]},
                # The following 3 examples are from
                # https://www.geeksforgeeks.org/bridge-in-a-graph/
                {"obj_func": (lambda cls2: cls2(range(5),\
                        ((0, 1), (0, 2), (0, 3), (1, 2), (3, 4)))),\
                        "opts": [{"result": {0, 3}}]},
                {"obj_func": (lambda cls2: cls2(range(4),\
                        ((0, 1), (1, 2), (2, 3)))),\
                        "opts": [{"result": {1, 2}}]},
                {"obj_func": (lambda cls2: cls2(range(7),\
                        ((0, 1), (0, 2), (1, 2), (1, 3), (1, 4),\
                        (1, 6), (3, 5), (4, 5)))),\
                        "opts": [{"result": {1}}]},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Set[Hashable]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[0],\
    #        res2: res_type_alias_lst[0]) -> bool:
    #    return res1 == res2
    
    def resultString(self, res: Set[Hashable],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return f"the articulation points of the graph of:\n{res}\n"
    
    def methodResultTest(self, obj: Any,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[Set[Hashable]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkArticulationBasic(obj, res, check_all=True),\
                "which is not the set of articulation points of the "\
                "graph")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def test_tarjanArticulationBasic_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_tarjanArticulationBasic_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_tarjanArticulationBasic_weighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (False,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (1, 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                wt_rng=wt_rng,\
                wt_cls_opts=wt_cls_opts,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices)
        return
    
    def test_tarjanArticulationBasic_unweighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (False,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices)
        return
    
    def test_tarjanArticulationBasic_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 10
        
        n_dim_rng = (1, 3)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(400 ** (1 / n_dim)))] * n_dim)
        p_wall_rng = (0.2, 0.8)
        n_diag_rng_func = lambda n_dim: (0, n_dim - 1)
        
        res = self.randomBinaryGridUnweightedGraphMethodTestsTemplate(\
                n_graph_per_opt=n_graph_per_opt,\
                n_dim_rng=n_dim_rng,\
                shape_rng_func=shape_rng_func,\
                p_wall_rng=p_wall_rng,\
                n_diag_rng_func=n_diag_rng_func)
        #print(res)
        return

        
if __name__ == "__main__":
    #print("testing bridge and articulation point algorithms")
    runAllTests()

