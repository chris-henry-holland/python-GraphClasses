#!/usr/bin/env python


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
    from graph_classes.limited_graph_types import (
        LimitedWeightedUndirectedGraphTemplate,
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

from graph_classes.Methods.minimum_spanning_tree_algorithms import (
    checkMinimumSpanningForest,
)

from graph_classes.explicit_graph_types import (
    ExplicitWeightedUndirectedGraph,
)
        

pkg_name = "graph_classes"
module_name = "minimum_spanning_tree_algorithms"

def runAllTests() -> None:
    unittest.main()
    return

class TestKruskal(TestGraphMethodTemplate):
    method_name = "kruskal"
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitWeightedUndirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ()))},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1, 1),)))},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1, 2), (1, 0, 3))))},
                # The following 2 examples are from
                # https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
                {"obj_func": (lambda cls2: cls2(range(4), ((0, 1, 10), (1, 3, 15),\
                        (2, 3, 4), (2, 0, 6), (0, 3, 5))))},
                {"obj_func": (lambda cls2: cls2(range(9), ((7, 6, 1), (8, 2, 2),\
                        (6, 5, 2), (0, 1, 4), (2, 5, 4), (8, 6, 6), (2, 3, 7),\
                        (7, 8, 7), (0, 7, 8), (1, 2, 8), (3, 4, 9), (5, 4, 10),\
                        (1, 7, 11), (3, 5, 14))))},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Tuple[List[Tuple[int, int, int]], Any]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[-1],\
    #        res2: res_type_alias_lst[-1]) -> bool:
    #    res1 == res2
    
    def resultString(self, res: Tuple[List[Tuple[int, int, int]], Any],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a minimum spanning forest with the edges (given as a "\
                "3-tuple of the labels of the two vertices the edge "\
                "connects followed by the weight of the edge) of:\n"\
                f"{res[1]}\nwith a total cost of {res[0]}"
    
    def methodResultTest(self, obj: "LimitedWeightedUndirectedGraphTemplate",\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[Tuple[List[Tuple[int, int, int]], Any]]=None)\
            -> bool:
        test_func = lambda res: (checkMinimumSpanningForest(obj, *res, eps=self.eps),\
                " which is not a minimum spanning forest of the graph "\
                "with that cost.")
        
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def test_kruskal_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def test_kruskal_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def test_kruskal_weighted_random(self) -> None:
        n_graph_per_opt = 50
        randomise_indices = True
    
        directed_opts = (False,)
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


if __name__ == "__main__":
    #print("testing minimum spanning tree algorithms")
    runAllTests()

