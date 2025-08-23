#!/usr/bin/env python

from typing import (
    Dict,
    List,
    Tuple,
    Optional,
    Union,
    Generator,
    Any,
    Callable,
)

import itertools
import math

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

#from graph_classes.utils import (
#    addUnittestTemplateDirectory,
#)

#pkg_dir = getPackageDirectory(__file__, 3, add=True)
#print(pkg_dir)
#print(type(__file__))
#curr_dir = os.path.dirname(os.path.abspath(__file__))
#pkg_dir = os.path.abspath(f"{curr_dir}/../../../")
#unittest_template_dir = os.abspath(f"{pkg_dir}/../unittest_templates")
#print(os.path.abspath(f"{pkg_dir}/../unittest_templates"))
#sys.path.append(os.path.abspath(f"{curr_dir}/../../../../unittest_templates"))

#addUnittestTemplateDirectory()

from unittest_templates.method_test_templates import TestMethodTemplate

#sys.path.append(os.path.abspath(f"{curr_dir}/../../../"))
#print(os.path.abspath(f"{curr_dir}/../../../"))

from graph_classes.random_explicit_graph_generators import (
    testRandomExplicitWeightedGraphsGenerator,
    testRandomExplicitUnweightedGraphsGenerator,
)

from graph_classes.random_grid_graph_generators import (
    testRandomBinaryGridUnweightedUndirectedGraphsGenerator,
)

def deconstructArgsKwargsOptions(args_opts: Tuple[Tuple[Any]],\
        kwargs_opts: Dict[str, Tuple[Any]])\
        -> Tuple[Tuple[List[Union[int, str]], List[Tuple[Any]]]]:
    
    err_msg_rand = "The integer values in args_opts and kwargs_opts "\
            "(if any) must be collectively distinct and consecutive "\
            "non-negative integers starting at 0"
        
    args_deconstr = ([], [], [], [])
    rand_seen = set()
    for idx, opts in enumerate(args_opts):
        if isinstance(opts, int):
            if opts in rand_seen:
                raise ValueError(err_msg_rand)
            rand_seen.add(opts)
            args_deconstr[2].append(idx)
            args_deconstr[3].append(opts)
            continue
        args_deconstr[0].append(idx)
        args_deconstr[1].append(opts)
    kwargs_deconstr = ([], [], [], [])
    for nm, opts in kwargs_opts.items():
        if isinstance(opts, int):
            if opts in rand_seen:
                raise ValueError(err_msg_rand)
            rand_seen.add(opts)
            kwargs_deconstr[2].append(nm)
            kwargs_deconstr[3].append(opts)
            continue
        kwargs_deconstr[0].append(nm)
        kwargs_deconstr[1].append(opts)
    if rand_seen != set(range(len(rand_seen))):
        raise ValueError(err_msg_rand)
    return (args_deconstr, kwargs_deconstr)

def argsKwargsReconstructGenerator(args: List[Any],\
        kwargs: Dict[str, Any],\
        args_opts_deconstr: Tuple[Union[List[int], List[Any]]],\
        kwargs_opts_deconstr: Tuple[Union[List[str], List[Any]]])\
        -> Generator[None, None, None]:
    # Relies on args and kwargs being mutable
    for kwargs_vals in itertools.product(*kwargs_opts_deconstr[1]):
        for kwargs_nm, val in zip(kwargs_opts_deconstr[0],\
                kwargs_vals):
            kwargs[kwargs_nm] = val
        for args_vals in itertools.product(*args_opts_deconstr[1]):
            for args_idx, val in zip(args_opts_deconstr[0],\
                    args_vals):
                args[args_idx] = val
            yield None
    return

def argsKwargsRandFirstGenerator(\
        args_opts_deconstr: Tuple[Union[List[int], List[Any]]],\
        kwargs_opts_deconstr: Tuple[Union[List[str], List[Any]]],\
        rand_generator_func: Callable[[int], Tuple[Any]],\
        test_rand_count: bool=True)\
        -> Generator[Tuple[Union[List[Any], Dict[str, Any]]],\
        None, None]:
    # Note- the values yielded are mutable, so they should either
    # be hard copied or be no longer needed before the next value
    # of the generator is yielded
    
    if test_rand_count:
        try:
            trial_rand = next(iter(rand_generator_func()))
        except StopIteration:
            pass
        else:
            if len(trial_rand) != len(args_opts_deconstr[2])\
                    + len(kwargs_opts_deconstr[2]):
                print(trial_rand, len(trial_rand), len(args_opts_deconstr[2])\
                    + len(kwargs_opts_deconstr[2]))
                raise ValueError("The output of rand_generator_func() "\
                        "does not yield the correct number of random "\
                        "arguments")
    args = [None] * (len(args_opts_deconstr[0]) +\
            len(args_opts_deconstr[2]))
    kwargs = {}
    reconstruct_gen = lambda: argsKwargsReconstructGenerator(args,\
            kwargs, args_opts_deconstr, kwargs_opts_deconstr)
    
    for rand_lst in rand_generator_func():
        for args_idx, rand_idx in zip(*args_opts_deconstr[2:]):
            args[args_idx] = rand_lst[rand_idx]
        for kwargs_nm, rand_idx in zip(*kwargs_opts_deconstr[2:]):
            kwargs[kwargs_nm] = rand_lst[rand_idx]
        for _ in reconstruct_gen():
            yield (args, kwargs)
    return

def argsKwargsRandLastGenerator(\
        args_opts_deconstr: Tuple[Union[List[int], List[Any]]],\
        kwargs_opts_deconstr: Tuple[Union[List[str], List[Any]]],\
        rand_generator_func: Callable[[int], Tuple[Any]],\
        test_rand_count: bool=True)\
        -> Generator[Tuple[Union[List[Any], Dict[str, Any]]],\
        None, None]:
    # Note- the values yielded are mutable, so they should either
    # be hard copied or be no longer needed before the next value
    # of the generator is yielded
    
    if test_rand_count:
        try:
            trial_rand = next(iter(rand_generator_func()))
        except StopIteration:
            pass
        else:
            if len(trial_rand) != len(args_opts_deconstr[2])\
                    + len(kwargs_opts_deconstr[2]):
                print(trial_rand, len(trial_rand), len(args_opts_deconstr[2])\
                    + len(kwargs_opts_deconstr[2]))
                raise ValueError("The output of rand_generator_func() "\
                        "does not yield the correct number of random "\
                        "arguments")
    args = [None] * (len(args_opts_deconstr[0]) +\
            len(args_opts_deconstr[2]))
    kwargs = {}
    reconstruct_gen = lambda: argsKwargsReconstructGenerator(args,\
            kwargs, args_opts_deconstr, kwargs_opts_deconstr)
    
    for _ in reconstruct_gen():
        for rand_lst in rand_generator_func():
            for args_idx, rand_idx in zip(*args_opts_deconstr[2:]):
                args[args_idx] = rand_lst[rand_idx]
            for kwargs_nm, rand_idx in zip(*kwargs_opts_deconstr[2:]):
                kwargs[kwargs_nm] = rand_lst[rand_idx]
            yield (args, kwargs)
    return

class TestGraphMethodTemplate(TestMethodTemplate):
    eps = 10 ** -5
    
    def graphAdjString(self, graph: Any) -> str:
        return str(graph.fullAdj().items())
        #return str({v1: {v2 for v2 in v2_set}\
        #        for v1, v2_dict in graph.fullAdj().items()})
    
    def objectDescriptionFunction(self, obj: Any) -> str:
        return f"{obj.graph_type_name} object "\
                f"with adjacency dictionary:\n{self.graphAdjString(obj)}\n"
    
    @staticmethod
    def binaryGridGraphKwargs(n_diag: int) -> Dict[str, Any]:
        return {
            "move_kwargs": {"n_diag": n_diag, "n_step_cap": 1},
            "n_state_func": lambda grid, grid_idx:\
                    1 - grid.arr_flat[grid_idx],
            "connected_func": lambda grid, grid_idx1, state_idx1,\
                    grid_idx2, state_idx2, mv, n_step: True,
        }
    
    def randomGraphMethodTestsTemplate(self,\
            directed_opts: Tuple[bool],\
            n_graph_per_opt: int=10,\
            n_vertices_rng: Tuple[int]=(4, 30),\
            n_edges_rng_func: Optional[Callable[[int], int]]=None,\
            wt_rng: Optional[Tuple[Union[int, float]]]=None,\
            wt_cls_opts: Tuple[type]=(int, float),\
            allow_self_edges_opts: Tuple[bool]=(True, False),\
            allow_rpt_edges_opts: Tuple[bool]=(True, False),\
            randomise_indices: bool=True,\
            method_args_opts: Optional[Tuple[Union[Tuple[Any], int]]]=None,\
            method_kwargs_opts: Optional[Dict[str, Union[Tuple[Any], int]]]=None,\
            rand_generator_func: Optional[Callable[[Any],\
            Tuple[Any]]]=None,\
            rand_first: bool=True)\
            -> Dict[Any, int]:
        
        default_method_vertex_opts_kwargs = {"mx_n_samples": 10,\
                "allow_index_repeats": True,\
                "allow_tuple_repeats": False, "nondecreasing": False}
        
        if method_args_opts is None:
            method_args_opts = ()
        if method_kwargs_opts is None:
            method_kwargs_opts = {}
        
        """
        if method_vertex_opts_kwargs is None:
            method_vertex_opts_kwargs = {}
        for kwarg_nm, default_val in\
                default_method_vertex_opts_kwargs.items():
            method_vertex_opts_kwargs.setdefault(kwarg_nm, default_val)
        
        def methodVertexGenerator(graph: "GraphTemplate")\
                -> Generator[Tuple[Any], None, None]:
            for inds in randomKTupleGenerator(\
                    graph.n, method_vertex_count,\
                    **method_vertex_opts_kwargs):
                yield tuple(graph.index2Vertex(idx) for idx in inds)
            return
        """
        if rand_generator_func is None:
            rand_generator_func = lambda graph: ((),)
        
        opts_deconstr =\
                deconstructArgsKwargsOptions(method_args_opts,\
                method_kwargs_opts)
        
        args_kwargs_gen0 = argsKwargsRandFirstGenerator\
                if rand_first else argsKwargsRandLastGenerator
        
        args_kwargs_gen = lambda rand_gen_func:\
                args_kwargs_gen0(*opts_deconstr, rand_gen_func,\
                test_rand_count=True)
        
        if wt_rng is None:
            graph_generator_func = lambda directed:\
                    testRandomExplicitUnweightedGraphsGenerator(\
                    n_graph_per_opt, directed, n_vertices_rng,\
                    n_edges_rng_func,\
                    allow_self_edges_opts=allow_self_edges_opts,\
                    allow_rpt_edges_opts=allow_rpt_edges_opts,\
                    randomise_indices=randomise_indices)
        else:
            graph_generator_func = lambda directed:\
                    testRandomExplicitWeightedGraphsGenerator(\
                    n_graph_per_opt, directed, n_vertices_rng,\
                    n_edges_rng_func, wt_rng, wt_cls_opts=wt_cls_opts,\
                    allow_self_edges_opts=allow_self_edges_opts,\
                    allow_rpt_edges_opts=allow_rpt_edges_opts,\
                    randomise_indices=randomise_indices)
        if n_edges_rng_func is None:
            n_edges_rng_func =\
                    lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        res = {}
        for directed in directed_opts:
            for graph in graph_generator_func(directed):
                for (args, kwargs) in args_kwargs_gen(\
                        lambda: rand_generator_func(graph)):
                    #print(graph.n, args, kwargs)
                    ans = self.methodResultTest(graph,\
                            method_args=args, method_kwargs=kwargs)
                    res[ans] = res.get(ans, 0) + 1
        #print(res)
        return res
    """
    def randomTestsTemplate(self,\
            directed_opts: Tuple[bool],\
            n_graph_per_opt: int=10,\
            n_vertices_rng: Tuple[int]=(4, 30),\
            n_edges_rng_func: Optional[Callable[[int], int]]=None,\
            wt_rng: Optional[Tuple[Union[int, float]]]=None,\
            wt_cls_opts: Tuple[type]=(int, float),\
            allow_self_edges_opts: Tuple[bool]=(True, False),\
            allow_rpt_edges_opts: Tuple[bool]=(True, False),\
            randomise_indices: bool=True,\
            method_args_opts: Optional[Tuple[Tuple[Any]]]=None,\
            method_kwargs_opts: Optional[Dict[str, Tuple[Any]]]=None)\
            -> None:
        
        if method_args_opts is None:
            method_args_opts = ()
        if method_kwargs_opts is None:
            method_kwargs_opts = {}
        
        kwargs_nms, kwargs_opt_vals =\
                deconstructKwargsOptions(method_kwargs_opts)
        
        if wt_rng is None:
            graph_generator_func = lambda directed:\
                    testRandomExplicitUnweightedGraphsGenerator(\
                    n_graph_per_opt, directed, n_vertices_rng,\
                    n_edges_rng_func,\
                    allow_self_edges_opts=allow_self_edges_opts,\
                    allow_rpt_edges_opts=allow_rpt_edges_opts,\
                    randomise_indices=randomise_indices)
        else:
            graph_generator_func = lambda directed:\
                    testRandomExplicitWeightedGraphsGenerator(\
                    n_graph_per_opt, directed, n_vertices_rng,\
                    n_edges_rng_func, wt_rng, wt_cls_opts=wt_cls_opts,\
                    allow_self_edges_opts=allow_self_edges_opts,\
                    allow_rpt_edges_opts=allow_rpt_edges_opts,\
                    randomise_indices=randomise_indices)
        if n_edges_rng_func is None:
            n_edges_rng_func =\
                    lambda x: ((x * (x  - 1)) >> 3, (x * (x - 1)) >> 1)
        for directed in directed_opts:
            for graph in graph_generator_func(directed):
                for kwargs_vals in itertools.product(*kwargs_opt_vals):
                    kwargs = reconstructKwargs(kwargs_nms, kwargs_vals)
                    for args in itertools.product(*method_args_opts):
                        self.methodResultTest(graph, method_args=args,\
                                method_kwargs=kwargs)
        #print(f"Total connected = {connected_cnt} of {tot_cnt}")
        return
    """
    def randomBinaryGridUnweightedGraphMethodTestsTemplate(self,\
            n_graph_per_opt: int=10,\
            n_dim_rng: Tuple[int]=(1, 3),\
            shape_rng_func: Optional[Callable[[int], Tuple[Tuple[int]]]]=None,\
            p_wall_rng: Tuple[float]=(0.2, 0.8),\
            n_diag_rng_func: Optional[Callable[[int], int]]=None,\
            method_args_opts: Optional[Tuple[Union[Tuple[Any], int]]]=None,\
            method_kwargs_opts: Optional[Dict[str, Union[Tuple[Any], int]]]=None,\
            rand_generator_func: Optional[Callable[[Any],\
            Tuple[Any]]]=None,\
            rand_first: bool=True)\
            -> Dict[Any, int]:
        
        if shape_rng_func is None:
            shape_rng_func = lambda n_dim: tuple([(1, math.ceil(400 ** (1 / n_dim)))] * n_dim)
        if n_diag_rng_func is None:
            n_diag_rng_func = lambda n_dim: (0, n_dim - 1)
        
        if method_args_opts is None:
            method_args_opts = ()
        if method_kwargs_opts is None:
            method_kwargs_opts = {}
        
        if rand_generator_func is None:
            rand_generator_func = lambda graph: ((),)
        
        opts_deconstr =\
                deconstructArgsKwargsOptions(method_args_opts,\
                method_kwargs_opts)
        
        args_kwargs_gen0 = argsKwargsRandFirstGenerator\
                if rand_first else argsKwargsRandLastGenerator
        
        args_kwargs_gen = lambda rand_gen_func:\
                args_kwargs_gen0(*opts_deconstr, rand_gen_func,\
                test_rand_count=True)
        
        graph_generator_func =\
                lambda: testRandomBinaryGridUnweightedUndirectedGraphsGenerator(\
                n_graph_per_opt, n_dim_rng, shape_rng_func,\
                p_wall_rng, n_diag_rng_func)
        res = {}
        for graph in graph_generator_func():
            #print(graph.n, graph.shape)
            for (args, kwargs) in args_kwargs_gen(\
                    lambda: rand_generator_func(graph)):
                #print(graph.n, args, kwargs)
                ans = self.methodResultTest(graph,\
                        method_args=args, method_kwargs=kwargs)
                res[ans] = res.get(ans, 0) + 1
        return res
