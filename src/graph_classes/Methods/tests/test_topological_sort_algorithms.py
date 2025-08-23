#!/usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Tuple,
    Optional,
    Hashable,
    Any,
)

if TYPE_CHECKING:
    from graph_classes.explicit_graph_types import (
        ExplicitGraphTemplate,
    )

import unittest

#sys.path.append(os.path.abspath('../tests'))

#from graph_method_test_templates import\
#        toString, TestGraphMethodTemplate

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

from graph_classes.Methods.topological_sort_algorithms import (
    checkTopologicalOrdering,
    checkTopologicalLayering,
)

from graph_classes import (
    ExplicitUnweightedDirectedGraph,
)

pkg_name = "graph_classes"
module_name = "topological_sort_algorithms"

def runAllTests() -> None:
    unittest.main()
    return

class TestKahn(TestGraphMethodTemplate):
    method_name = "kahn"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ()))},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),)))},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0))))},
                # The following 3 examples are from
                # https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
                {"obj_func": (lambda cls2: cls2(range(6), ((2, 3), (3, 1),\
                        (4, 0), (4, 1), (5, 0), (5, 2))))},
                {"obj_func": (lambda cls2: cls2(range(5), ((0, 1), (1, 2),\
                        (2, 3), (3, 4))))},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [List[int]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[-1],\
    #        res2: res_type_alias_lst[-1]) -> bool:
    #    res1 == res2
    
    def resultString(self, res: res_type_alias_lst[-1],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a topological ordering of the vertices of the graph "\
                f"of:\n{res}\n"
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res: (checkTopologicalOrdering(obj, res),\
                "which is not a valid topological ordering of the "\
                "vertices of the graph")
        
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def test_kahn_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_kahn_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_kahn_weighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (1, 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 6, (x * (x - 1)) >> 4)
        
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
    
    def test_kahn_unweighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 6, (x * (x - 1)) >> 4)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices)
        return

class TestKahnLayering(TestGraphMethodTemplate):
    method_name = "kahnLayering"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ())),\
                        "opts": [{"result": [[1, 0]]}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),))),\
                        "opts": [{"result": [[0], [1]]}]},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0)))), "opts": [{"result": []}]},
                # The following 3 examples are from
                # https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/
                {"obj_func": (lambda cls2: cls2(range(6), ((2, 3), (3, 1),\
                        (4, 0), (4, 1), (5, 0), (5, 2)))),\
                        "opts": [{"result": [[4, 5], [0, 2], [3], [1]]}]},
                {"obj_func": (lambda cls2: cls2(range(5), ((0, 1), (1, 2),\
                        (2, 3), (3, 4)))),\
                        "opts": [{"result": [[0], [1], [2], [3], [4]]}]},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [List[List[int]]]
    
    def resultEqualityFunction(self, res1: res_type_alias_lst[-1],\
            res2: res_type_alias_lst[-1]) -> bool:
        return [set(x) for x in res1] == [set(x) for x in res2]
    
    def resultString(self, res: res_type_alias_lst[0],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a topological layering of the vertices of the graph "\
                f"of:\n{res}\n"
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res: (checkTopologicalLayering(obj, res),\
                "which is not a valid topological layering of the "\
                "vertices of the graph")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def test_kahnLayering_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_kahnLayering_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_kahnLayering_weighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (1, 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 6, (x * (x - 1)) >> 4)
        
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
    
    def test_kahnLayering_unweighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 6, (x * (x - 1)) >> 4)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices)
        return

if __name__ == "__main__":
    #print("testing bridge and articulation point algorithms")
    runAllTests()

