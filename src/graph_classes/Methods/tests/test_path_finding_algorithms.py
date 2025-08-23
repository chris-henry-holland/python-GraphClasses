#!/usr/bin/env python

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Dict,
    List,
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

import math
import random
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

from graph_classes.Methods.path_finding_algorithms import (
    checkAllPairsPathfinder,
    checkAllPairsDistances,
    checkFromSourcesPathfinder,
    checkFromSourcesDistances,
)

from graph_classes.utils import (
    randomKTupleGenerator,
)
        
from graph_classes.explicit_graph_types import (
    ExplicitUnweightedDirectedGraph,
    ExplicitWeightedDirectedGraph,
)

pkg_name = "graph_classes"
module_name = "path_finding_algorithms"

def runAllTests() -> None:
    unittest.main()
    return

def randSourceSinkWeightFunction(wt_mn: Union[int, float],\
        wt_mx: Union[int, float],\
        wt_typ: type=int)\
        -> Union[int, float]:
    #print(wt_typ, wt_typ == int)
    if wt_typ == int:
        if not isinstance(wt_mn, int):
            wt_mn = math.ceil(wt_mn)
        if not isinstance(wt_mx, int):
            wt_mx = math.floor(wt_mx)
        wt_mx += 1
        res = random.randrange(wt_mn, wt_mx)
    else: res = random.uniform(wt_mn, wt_mx)
    #print(res)
    return res

class TestAllPairsPathfinderAlgorithmsTemplate(\
        TestGraphMethodTemplate):
    eps = 10 ** -5
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ()))},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),)))},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0))))},
            ],
            ExplicitWeightedDirectedGraph: [
                # The following 2 examples from https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/
                {"obj_func": (lambda cls2: cls2(list("ABCDE"),\
                        (("A", "B", 4), ("A", "D", 5), ("B", "C", 1),\
                        ("B", "E", 6), ("C", "A", 2), ("C", "D", 3),\
                        ("D", "C", 1), ("D", "E", 2), ("E", "A", 1),\
                        ("E", "D", 4))))},
                {"obj_func": (lambda cls2: cls2(range(4),\
                        ((0, 1, 5), (0, 2, 10), (1, 2, 3), (2, 3, 1))))},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Dict[Hashable, Dict[Hashable,\
            Tuple[Union[int, float, Optional[Hashable]]]]]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[0],\
    #        res2: res_type_alias_lst[0]) -> bool:
    #    res1 == res2
    
    def resultString(self, res: res_type_alias_lst[0],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a minimum distances array represented by the "\
                f"dictionary of dictionaries:\n{res}\n(where, "\
                "for vertices v1 and v2, the value corresponding "\
                "to keys v1 and v2 is a 2-tuple whose index 0 "\
                "contains the minimum total distance of paths from "\
                "v1 to v2 and whose index 1 contains the vertex "\
                "preceding v2 on one path of that distance if v1 and "\
                "v2 are distinct and None otherwise, while v2 not "\
                "appearing  in the inner dictionary corresponding to "\
                "v1 signifies that there is no path from v1 to v2) "
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkAllPairsPathfinder(obj, res, eps=self.eps),\
                "which is not the correct all pairs pathfinder array")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def _test_allPairsPathfinder_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def _test_allPairsPathfinder_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def _test_allPairsPathfinder_weighted_random(self) -> None:
        n_graph_per_opt = 20
        randomise_indices = True
    
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (-2, 200)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        
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
                randomise_indices=randomise_indices)
        #print(res)
        return
    
    def _test_allPairsPathfinder_unweighted_random(self) -> None:
        n_graph_per_opt = 40
        randomise_indices = True
    
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        res = self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices)
        #print(res)
        return
    
    def _test_allPairsPathfinder_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 5
        n_pairs_per_graph = 10
        
        n_dim_rng = (1, 2)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(200 ** (1 / n_dim)))] * n_dim)
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

class TestFloydWarshallPathfinder(\
        TestAllPairsPathfinderAlgorithmsTemplate):
    method_name = "floydWarshallPathfinder"
    
    def test_floydWarshallPathfinder_known_good(self) -> None:
        return self._test_allPairsPathfinder_known_good()
    
    def test_floydWarshallPathfinder_known_err(self) -> None:
        return self._test_allPairsPathfinder_known_err()
    
    def test_floydWarshallPathfinder_weighted_random(self) -> None:
        return self._test_allPairsPathfinder_weighted_random()
    
    def test_floydWarshallPathfinder_unweighted_random(self) -> None:
        return self._test_allPairsPathfinder_unweighted_random()
    
    def test_floydWarshallPathfinder_binaryGridGraph_random(self) -> None:
        return self._test_allPairsPathfinder_binaryGridGraph_random()

class TestJohnson(TestAllPairsPathfinderAlgorithmsTemplate):
    method_name = "johnson"
    
    def test_johnson_known_good(self) -> None:
        return self._test_allPairsPathfinder_known_good()
    
    def test_johnson_known_err(self) -> None:
        return self._test_allPairsPathfinder_known_err()
    
    def test_johnson_weighted_random(self) -> None:
        return self._test_allPairsPathfinder_weighted_random()
    
    def test_johnson_unweighted_random(self) -> None:
        return self._test_allPairsPathfinder_unweighted_random()

    def test_johnson_binaryGridGraph_random(self) -> None:
        return self._test_allPairsPathfinder_binaryGridGraph_random()

class TestAllPairsDistancesAlgorithmsTemplate(\
        TestGraphMethodTemplate):
    eps = 10 ** -5
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ()))},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),)))},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0))))},
            ],
            ExplicitWeightedDirectedGraph: [
                # The following 2 examples from https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/
                {"obj_func": (lambda cls2: cls2(list("ABCDE"),\
                        (("A", "B", 4), ("A", "D", 5), ("B", "C", 1),\
                        ("B", "E", 6), ("C", "A", 2), ("C", "D", 3),\
                        ("D", "C", 1), ("D", "E", 2), ("E", "A", 1),\
                        ("E", "D", 4)))),\
                        "opts": [{"result":\
                        {"A": {"A": 0, "B": 4, "C": 5, "D": 5, "E": 7},\
                        "B": {"A": 3, "B": 0, "C": 1, "D": 4, "E": 6},\
                        "C": {"A": 2, "B": 6, "C": 0, "D": 3, "E": 5},\
                        "D": {"A": 3, "B": 7, "C": 1, "D": 0, "E": 2},\
                        "E": {"A": 1, "B": 5, "C": 5, "D": 4, "E": 0}}}]},
                {"obj_func": (lambda cls2: cls2(range(4),\
                        ((0, 1, 5), (0, 2, 10), (1, 2, 3), (2, 3, 1)))),\
                        "opts": [{"result":\
                        {0: {0: 0, 1: 5, 2: 8, 3: 9},\
                        1: {1: 0, 2: 3, 3: 4}, 2: {2: 0, 3: 1},\
                        3: {3: 0}}}]},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Dict[Hashable, Dict[Hashable,\
            Tuple[Union[int, float, Optional[Hashable]]]]]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[0],\
    #        res2: res_type_alias_lst[0]) -> bool:
    #    res1 == res2
    
    def resultString(self, res: res_type_alias_lst[0],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a minimum distances array represented by the "\
                f"dictionary of dictionaries:\n{res}\n(where, "\
                "for vertices v1 and v2, the value corresponding "\
                "to keys v1 and v2 is the minimum total distance "\
                "of paths from v1 to v2, while v2 not appearing "\
                "in the inner dictionary corresponding to v1 "\
                "signifies that there is no path from v1 to v2) "
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkAllPairsDistances(obj, res, eps=self.eps,\
                check_pathfinder=True),\
                "which is not the correct minimum distances array")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def _test_allPairsDistances_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def _test_allPairsDistances_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def _test_allPairsDistances_weighted_random(self) -> None:
        n_graph_per_opt = 20
        randomise_indices = True
    
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (-2, 200)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        
        
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
                randomise_indices=randomise_indices)
        #print(res)
        return
    
    def _test_allPairsDistances_unweighted_random(self) -> None:
        n_graph_per_opt = 40
        randomise_indices = True
    
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        
        n_edges_rng_func =\
                lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        
        res = self.randomGraphMethodTestsTemplate(\
                directed_opts=directed_opts,\
                n_graph_per_opt=n_graph_per_opt,\
                n_vertices_rng=n_vertices_rng,\
                n_edges_rng_func=n_edges_rng_func,\
                allow_self_edges_opts=allow_self_edges_opts,\
                allow_rpt_edges_opts=allow_rpt_edges_opts,\
                randomise_indices=randomise_indices)
        #print(res)
        return
    
    def _test_allPairsDistances_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 5
        n_pairs_per_graph = 10
        
        n_dim_rng = (1, 3)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(200 ** (1 / n_dim)))] * n_dim)
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

class TestFloydWarshallDistances(\
        TestAllPairsDistancesAlgorithmsTemplate):
    method_name = "floydWarshallDistances"
    
    def test_floydWarshallDistances_known_good(self) -> None:
        return self._test_allPairsDistances_known_good()
    
    def test_floydWarshallDistances_known_err(self) -> None:
        return self._test_allPairsDistances_known_err()
    
    def test_floydWarshallDistances_weighted_random(self) -> None:
        return self._test_allPairsDistances_weighted_random()
    
    def test_floydWarshallDistances_unweighted_random(self) -> None:
        return self._test_allPairsDistances_unweighted_random()
    
    def test_floydWarshallDistances_binaryGridGraph_random(self) -> None:
        return self._test_allPairsDistances_binaryGridGraph_random()

class TestFromSourcesPathfinderAlgorithmsTemplate(\
        TestGraphMethodTemplate):
    eps = 10 ** -5
    
    @classmethod
    def knownGoodResults(cls) -> Dict[Any, List[Dict[str, Any]]]:
        known_good_results = {
            ExplicitUnweightedDirectedGraph: [
                {"obj_func": (lambda cls2: cls2(range(2), ())),\
                        "opts": [{"args": ({0: 0},)}, {"args": ({1: 0},)},\
                        {"args": ({1: 0, 0: 0},)}, {"args": ({0: -2},)},\
                        {"args": ({1: 5.3},)},\
                        {"args": ({0: 5.3, 0: -2},)}]},
                {"obj_func": (lambda cls2: cls2(range(2), ((0, 1),))),\
                        "opts": [{"args": ({0: 0},)}, {"args": ({1: 0},)},\
                        {"args": ({1: 0, 0: 0},)}, {"args": ({0: -2},)},\
                        {"args": ({1: 5.3},)},\
                        {"args": ({0: 5.3, 0: -2},)}]},
                {"obj_func": (lambda cls2: cls2(range(2),\
                        ((0, 1), (1, 0)))),\
                        "opts": [{"args": ({0: 0},)}, {"args": ({1: 0},)},\
                        {"args": ({1: 0, 0: 0},)}, {"args": ({0: -2},)},\
                        {"args": ({1: 5.3},)},\
                        {"args": ({0: 5.3, 0: -2},)}]},
            ]
        }
        return known_good_results
    
    res_type_alias_lst = [Dict[Hashable,\
            Tuple[Union[int, float, Optional[Hashable]]]]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[0],\
    #        res2: res_type_alias_lst[0]) -> bool:
    #    res1 == res2
    
    def resultString(self, res: res_type_alias_lst[0],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a minimum distance from source array represented "\
                f"by the dictionary:\n{res}\n(where, "\
                "for the vertex v, the value corresponding "\
                "to key v is a 2-tuple whose index 0 contains the "\
                "minimum total distance of paths from a vertex in "\
                "sources plus that source vertex's initial distance "\
                "to v and whose index 1 contains the vertex "\
                "preceding v on one such path if that path contains "\
                "more than one vertex and None otherwise, while v "\
                "not appearing in the dictionary signifies that "\
                "there is no path from any vertex in sources to v) "
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkFromSourcesPathfinder(obj, res, *method_args,\
                eps=self.eps),\
                "which is not the correct from sources pathfinder "\
                "dictionary")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def _test_fromSourcesPathfinder_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def _test_fromSourcesPathfinder_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def _test_fromSourcesPathfinder_weighted_random(self,\
            min_wt: Union[int, float]=0) -> None:
        n_graph_per_opt = 20
        n_source_set_per_graph = 10
        randomise_indices = True
        
        n_source_rng_func = lambda graph: (1, (graph.n >> 1) + 1)
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (min_wt, min_wt + 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        source_wt_rng = (-20, 100)
        source_wt_typ_opts = (int, float)
        
        source_wt_func = lambda wt_typ:\
                randSourceSinkWeightFunction(\
                *source_wt_rng, wt_typ)
        #source_wt_func(wt_typ)\
        method_args_opts = (0,)
        rand_generator_func = lambda graph:\
                (({graph.index2Vertex(idx): source_wt_func(wt_typ)\
                for idx in inds},) for inds in\
                randomKTupleGenerator(graph.n,\
                random.randrange(*n_source_rng_func(graph)),\
                mx_n_samples=n_source_set_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=True) for wt_typ in\
                (random.choice(source_wt_typ_opts),))
    
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
        #print(res)
        return
    
    def _test_fromSourcesPathfinder_unweighted_random(self) -> None:
        n_graph_per_opt = 40
        n_source_set_per_graph = 10
        randomise_indices = True
        
        n_source_rng_func = lambda graph: (1, (graph.n >> 1) + 1)
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        source_wt_rng = (-20, 100)
        source_wt_typ_opts = (int, float)
        
        source_wt_func = lambda wt_typ:\
                randSourceSinkWeightFunction(\
                *source_wt_rng, wt_typ)
        
        method_args_opts = (0,)
        rand_generator_func = lambda graph:\
                (({graph.index2Vertex(idx): source_wt_func(wt_typ)\
                for idx in inds},) for inds in\
                randomKTupleGenerator(graph.n,\
                random.randrange(*n_source_rng_func(graph)),\
                mx_n_samples=n_source_set_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=True) for wt_typ in\
                (random.choice(source_wt_typ_opts),))
        
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
        #print(res)
        return
    
    def _test_fromSourcesPathfinder_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 5
        n_source_set_per_graph = 10
        
        n_source_rng_func = lambda graph: (0, (graph.n >> 1) + 1)
        
        n_dim_rng = (1, 2)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(200 ** (1 / n_dim)))] * n_dim)
        p_wall_rng = (0.2, 0.8)
        n_diag_rng_func = lambda n_dim: (0, n_dim - 1)
        
        source_wt_rng = (-20, 100)
        source_wt_typ_opts = (int, float)
        source_wt_func = lambda wt_typ:\
                randSourceSinkWeightFunction(\
                *source_wt_rng, wt_typ)
        
        method_args_opts = (0,)
        rand_generator_func = lambda graph:\
                (({graph.index2Vertex(idx): source_wt_func(wt_typ)\
                for idx in inds},) for inds in\
                randomKTupleGenerator(graph.n,\
                random.randrange(*n_source_rng_func(graph)),\
                mx_n_samples=n_source_set_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=True) for wt_typ in\
                (random.choice(source_wt_typ_opts),))
        
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

class TestDijkstraFromSourcesPathfinder(\
        TestFromSourcesPathfinderAlgorithmsTemplate):
    method_name = "dijkstraFromSourcesPathfinder"
    
    #def test_dijkstraFromSourcesPathfinder_known_good(self) -> None:
    #    return self._test_fromSourcesPathfinder_known_good()
    
    #def test_dijkstraFromSourcesPathfinder_known_err(self) -> None:
    #    return self._test_fromSourcesPathfinder_known_err()
    
    #def test_dijkstraFromSourcesPathfinder_weighted_random(self) -> None:
    #    return self._test_fromSourcesPathfinder_weighted_random(min_wt=0)
    
    #def test_dijkstraFromSourcesPathfinder_unweighted_random(self) -> None:
    #    return self._test_fromSourcesPathfinder_unweighted_random()
    
    def test_dijkstraFromSourcesPathfinder_binaryGridGraph_random(self) -> None:
        return self._test_fromSourcesPathfinder_binaryGridGraph_random()

class TestShortestPathFasterAlgorithmPathfinder(\
        TestFromSourcesPathfinderAlgorithmsTemplate):
    method_name = "shortestPathFasterAlgorithmPathfinder"
    
    def test_shortestPathFasterAlgorithmPathfinder_known_good(self) -> None:
        return self._test_fromSourcesPathfinder_known_good()
    
    def test_shortestPathFasterAlgorithmPathfinder_known_err(self) -> None:
        return self._test_fromSourcesPathfinder_known_err()
    
    def test_shortestPathFasterAlgorithmPathfinder_weighted_random(self) -> None:
        return self._test_fromSourcesPathfinder_weighted_random(min_wt=-2)
    
    def test_shortestPathFasterAlgorithmPathfinder_unweighted_random(self) -> None:
        return self._test_fromSourcesPathfinder_unweighted_random()
    
    def test_shortestPathFasterAlgorithmPathfinder_binaryGridGraph_random(self) -> None:
        return self._test_fromSourcesPathfinder_binaryGridGraph_random()

class TestFromSourcesDistancesAlgorithmsTemplate(\
        TestGraphMethodTemplate):
    eps = 10 ** -5
    
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
    
    res_type_alias_lst = [Dict[Hashable, Tuple[Union[int, float]]]]
    
    #def resultEqualityFunction(self, res1: res_type_alias_lst[0],\
    #        res2: res_type_alias_lst[0]) -> bool:
    #    res1 == res2
    
    def resultString(self, res: res_type_alias_lst[0],\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None) -> str:
        return "a minimum distance from sources array represented by "\
                "the dictionary:\n{res}\n(where, for vertex v, the "\
                "value corresponding to keys v is the minimum total "\
                "distance of paths from a vertex in sources plus "\
                "that source vertex's initial distance "\
                "to v, while v not appearing in the dictionary "\
                "signifies that there is no path from any vertex in "\
                "sources to v) "
    
    def methodResultTest(self, obj: ExplicitGraphTemplate,\
            method_args: Optional[Tuple[Hashable]]=None,\
            method_kwargs: Optional[Dict[str, Any]]=None,\
            known_result: Optional[res_type_alias_lst[0]]=None)\
            -> bool:
        test_func = lambda res:\
                (checkFromSourcesDistances(obj, res, *method_args,\
                eps=self.eps, check_pathfinder=True),\
                "which is not the correct distance from sources "\
                "dictionary")
        res = self._methodResultTest(obj, test_func,\
                method_args=method_args, method_kwargs=method_kwargs,\
                known_result=known_result, full_chk_if_known=False)
        return bool(res)
    
    def _test_fromSourcesDistances_known_good(self) -> None:
        return self.knownGoodResultTestTemplate()
    
    def _test_fromSourcesDistances_known_err(self) -> None:
        return self.knownErrorTestTemplate()
    
    def _test_fromSourcesDistances_weighted_random(self,\
            min_wt: Union[int, float]=0) -> None:
        n_graph_per_opt = 20
        n_source_set_per_graph = 10
        randomise_indices = True
        
        n_source_rng_func = lambda graph: (1, (graph.n >> 1) + 1)
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        wt_rng = (min_wt, min_wt + 100)
        wt_cls_opts = (int, float)
        n_vertices_rng = (4, 30)
        source_wt_rng = (-20, 100)
        source_wt_typ_opts = (int, float)
        
        source_wt_func = lambda wt_typ:\
                randSourceSinkWeightFunction(\
                *source_wt_rng, wt_typ)
        #source_wt_func(wt_typ)\
        method_args_opts = (0,)
        rand_generator_func = lambda graph:\
                (({graph.index2Vertex(idx): source_wt_func(wt_typ)\
                for idx in inds},) for inds in\
                randomKTupleGenerator(graph.n,\
                random.randrange(*n_source_rng_func(graph)),\
                mx_n_samples=n_source_set_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=True) for wt_typ in\
                (random.choice(source_wt_typ_opts),))
    
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
        #print(res)
        return
    
    def _test_fromSourcesDistances_unweighted_random(self) -> None:
        n_graph_per_opt = 40
        n_source_set_per_graph = 10
        randomise_indices = True
        
        n_source_rng_func = lambda graph: (1, (graph.n >> 1) + 1)
        directed_opts = (True, False)
        allow_self_edges_opts = (True, False)
        allow_rpt_edges_opts = (True, False)
        n_vertices_rng = (4, 30)
        source_wt_rng = (-20, 100)
        source_wt_typ_opts = (int, float)
        
        source_wt_func = lambda wt_typ:\
                randSourceSinkWeightFunction(\
                *source_wt_rng, wt_typ)
        
        method_args_opts = (0,)
        rand_generator_func = lambda graph:\
                (({graph.index2Vertex(idx): source_wt_func(wt_typ)\
                for idx in inds},) for inds in\
                randomKTupleGenerator(graph.n,\
                random.randrange(*n_source_rng_func(graph)),\
                mx_n_samples=n_source_set_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=True) for wt_typ in\
                (random.choice(source_wt_typ_opts),))
        
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
        #print(res)
        return
    
    def _test_fromSourcesDistances_binaryGridGraph_random(self) -> None:
        n_graph_per_opt = 5
        n_source_set_per_graph = 10
        
        n_source_rng_func = lambda graph: (0, (graph.n >> 1) + 1)
        
        n_dim_rng = (1, 2)
        shape_rng_func = lambda n_dim: tuple([(1, math.ceil(200 ** (1 / n_dim)))] * n_dim)
        p_wall_rng = (0.2, 0.8)
        n_diag_rng_func = lambda n_dim: (0, n_dim - 1)
        
        source_wt_rng = (-20, 100)
        source_wt_typ_opts = (int, float)
        source_wt_func = lambda wt_typ:\
                randSourceSinkWeightFunction(\
                *source_wt_rng, wt_typ)
        
        method_args_opts = (0,)
        rand_generator_func = lambda graph:\
                (({graph.index2Vertex(idx): source_wt_func(wt_typ)\
                for idx in inds},) for inds in\
                randomKTupleGenerator(graph.n,\
                random.randrange(*n_source_rng_func(graph)),\
                mx_n_samples=n_source_set_per_graph,\
                allow_index_repeats=False,\
                allow_tuple_repeats=False,\
                nondecreasing=True) for wt_typ in\
                (random.choice(source_wt_typ_opts),))
        
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

class TestShortestPathFasterAlgorithmDistances(\
        TestFromSourcesDistancesAlgorithmsTemplate):
    method_name = "shortestPathFasterAlgorithmDistances"
    
    def test_shortestPathFasterAlgorithmDistances_known_good(self) -> None:
        return self._test_fromSourcesDistances_known_good()
    
    def test_shortestPathFasterAlgorithmDistances_known_err(self) -> None:
        return self._test_fromSourcesDistances_known_err()
    
    def test_shortestPathFasterAlgorithmDistances_weighted_random(self) -> None:
        return self._test_fromSourcesDistances_weighted_random(min_wt=-2)
    
    def test_shortestPathFasterAlgorithmDistances_unweighted_random(self) -> None:
        return self._test_fromSourcesDistances_unweighted_random()
    
    def test_shortestPathFasterAlgorithmDistances_binaryGridGraph_random(self) -> None:
        return self._test_fromSourcesDistances_binaryGridGraph_random()

if __name__ == "__main__":
    #print("testing path finding algorithms")
    runAllTests()

