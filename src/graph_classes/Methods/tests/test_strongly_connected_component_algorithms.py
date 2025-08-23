#!/usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Hashable,
    Any,
)

if TYPE_CHECKING:
    from graph_classes.explicit_graph_types import (
        ExplicitGraphTemplate,
    )

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

from graph_classes.Methods.strongly_connected_component_algorithms import\
        checkSCCRepr,\
        SCCReprEqual,\
        checkCondensedSCC,\
        condensedSCCEqual

from graph_classes import (
    ExplicitUnweightedDirectedGraph,
)

pkg_name = "graph_classes"
module_name = "strongly_connected_component_algorithms"

def runAllTests() -> None:
    unittest.main()
    return

class TestSCCAlgorithmsTemplate(TestGraphMethodTemplate):
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ())),\
                        "opts": [{"result": {0: 0, 1: 1}}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),))),\
                        "opts": [{"result": {0: 0, 1: 1}}]},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0)))),\
                        "opts": [{"result": {0: 1, 1: 1}}]},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Dict[Hashable, Hashable]]
    
    def resultEqualityFunction(self, res1: res_type_alias_lst[0],\
            res2: res_type_alias_lst[0]) -> bool:
        return SCCReprEqual(res1, res2)
    
    def resultString(self, res: res_type_alias_lst[0],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a strongly connected component representative "\
                f"dictionary:\n{res}\n"
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkSCCRepr(obj, res),\
                "which is not a valid strongly connected component "\
                "representative dictionary")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def _test_SCCAlgorithm_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def _test_SCCAlgorithm_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def _test_SCCAlgorithm_weighted_random(self) -> None:
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
    
    def _test_SCCAlgorithm_unweighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
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

class TestKosaraju(TestSCCAlgorithmsTemplate):
    method_name = "kosaraju"
    
    def test_kosaraju_known_good(self) -> None:
        return self._test_SCCAlgorithm_known_good()
    
    def test_kosaraju_known_err(self) -> None:
        return self._test_SCCAlgorithm_known_err()
    
    def test_kosaraju_weighted_random(self) -> None:
        return self._test_SCCAlgorithm_weighted_random()
    
    def test_kosaraju_unweighted_random(self) -> None:
        return self._test_SCCAlgorithm_unweighted_random()

class TestTarjanSCC(TestSCCAlgorithmsTemplate):
    method_name = "tarjanSCC"
    
    def test_tarjanSCC_known_good(self) -> None:
        return self._test_SCCAlgorithm_known_good()
    
    def test_tarjanSCC_known_err(self) -> None:
        return self._test_SCCAlgorithm_known_err()
    
    def test_tarjanSCC_weighted_random(self) -> None:
        return self._test_SCCAlgorithm_weighted_random()
    
    def test_tarjanSCC_unweighted_random(self) -> None:
        return self._test_SCCAlgorithm_unweighted_random()


class TestCondenseSCC(TestGraphMethodTemplate):
    method_name = "condenseSCC"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ()))},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),)))},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0))))},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Tuple[Union[Dict[Hashable, Hashable],\
            Dict[Hashable, Set[Hashable]],\
            ExplicitUnweightedDirectedGraph]]]
    
    def resultEqualityFunction(self, res1: res_type_alias_lst[-1],\
            res2: res_type_alias_lst[-1]) -> bool:
        return condensedSCCEqual(res1, res2)
    
    def resultString(self, res: res_type_alias_lst[-1],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a condensed representation of the graph into its "\
                "strongly connected components, with "\
                "a strongly connected component representative "\
                f"dictionary:\n{res[0]}\nwith the vertices of the "\
                "represented by each representative vertex of:\n"\
                f"{res[1]}\nand condensed graph in "\
                "where each strongly connected component is treated "\
                "as a single vertex (with value equal to that of the "\
                "representative of that component) with adjacency "\
                f"dictionary:\n{self.graphAdjString(res[2])}\n"
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> int:
        test_func = lambda res:\
                (checkCondensedSCC(obj, *res),\
                "which is not a valid condensed representation of "\
                "the graph into its strongly connected components")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return len(res[1])
    
    def test_condensedSCC_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_condensedSCC_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_condensedSCC_weighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (1, 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        method_kwargs_opts = {"alg": ("kosaraju", "tarjan"),\
                "set_condensed_in_degrees": (True, False)}
        
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
                randomise_indices=randomise_indices,\
                method_kwargs_opts=method_kwargs_opts)
        return
    
    def test_condensedSCC_unweighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (True,)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        
        method_kwargs_opts = {"alg": ("kosaraju", "tarjan"),\
                "set_condensed_in_degrees": (True, False)}
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 6, (x * (x - 1)) >> 4)
        
        self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices,\
                method_kwargs_opts=method_kwargs_opts)
        return

if __name__ == "__main__":
    #print("testing strongly connected component algorithms")
    runAllTests()

