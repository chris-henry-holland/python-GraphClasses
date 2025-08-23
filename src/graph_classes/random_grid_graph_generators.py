#!/usr/bin/env python

from typing import (
    Dict,
    Tuple,
    Optional,
    Generator,
    Callable,
)

import functools
import itertools
import random


#from .utils import FenwickTree

from graph_classes.grid_graph_types import (
    Grid,
    GridUnweightedUndirectedGraph,
)

def randomBinaryGrid(shape: Tuple[int], p_true: float) -> Grid:
    if not 0 <= p_true <= 1:
        raise ValueError("p_true must be between 0 and 1 inclusive")
    length = functools.reduce(lambda x, y: x * y, shape, 1)
    arr_flat = [random.random() < p_true for _ in range(length)]
    return Grid.createGridFromFlat(shape, arr_flat)

def randomBinaryGridUnweightedUndirectedGraph(shape: Tuple[int],\
        p_wall: float, n_diag: int=0)\
        -> "GridUnweightedUndirectedGraph":
    grid = randomBinaryGrid(shape, p_wall)
    move_kwargs = {"n_diag": n_diag, "n_step_cap": 1,\
            "allowed_direct_idx_func": None}
    n_state_func = lambda grid, grid_idx: 1 - grid.arr_flat[grid_idx]
    connected_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: True
    return GridUnweightedUndirectedGraph(grid, move_kwargs,\
            n_state_func, connected_func)

def randomBinaryGridUnweightedUndirectedGraphGenerator(\
        n_dim_rng: Tuple[int],\
        shape_rng_func: Callable[[int], Tuple[Tuple[int]]],\
        p_wall_rng: Tuple[float],\
        n_diag_rng_func: Callable[[int], Tuple[int]],\
        count: Optional[int]=None)\
        -> Generator["GridUnweightedUndirectedGraph", None, None]:
    
    iter_obj = itertools.count(0) if count is None else range(count)
    move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
            "allowed_direct_idx_func": None,\
            "directed_axis_restrict": None}
    n_state_func = lambda grid, grid_idx: 1 - grid.arr_flat[grid_idx]
    connected_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: True
    
    for _ in iter_obj:
        n_dim = random.randrange(n_dim_rng[0], n_dim_rng[1] + 1)
        print(n_dim_rng, n_dim)
        shape = [random.randrange(rng[0], rng[1] + 1)\
                for rng in shape_rng_func(n_dim)]
        print(shape)
        n_diag_rng = n_diag_rng_func(n_dim)
        move_kwargs["n_diag"] =\
                random.randrange(n_diag_rng[0], n_diag_rng[1] + 1)
        p_wall = random.uniform(*p_wall_rng)
        grid = randomBinaryGrid(shape, p_wall)
        yield GridUnweightedUndirectedGraph(grid, move_kwargs,\
                n_state_func, connected_func)
    
    return

def testRandomBinaryGridUnweightedUndirectedGraphsGenerator(\
        n_graph_per_opt: int,\
        n_dim_rng: Tuple[int],\
        shape_rng_func: Callable[[int], Tuple[Tuple[int]]],\
        p_wall_rng: Tuple[float],\
        n_diag_rng_func: Callable[[int], Tuple[int]],\
        count: Optional[int]=None)\
        -> Generator["GridUnweightedUndirectedGraph", None, None]:
    
    move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
            "allowed_direct_idx_func": None,\
            "directed_axis_restrict": None}
    n_state_func = lambda grid, grid_idx: 1 - grid.arr_flat[grid_idx]
    connected_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: True
    
    for n_dim in range(n_dim_rng[0], n_dim_rng[1] + 1):
        shape_rng = shape_rng_func(n_dim)
        #print(n_dim, shape_rng)
        n_diag_rng = n_diag_rng_func(n_dim)
        for n_diag in range(n_diag_rng[0], n_diag_rng[1] + 1):
            move_kwargs["n_diag"] = n_diag
            for _ in range(n_graph_per_opt):
                shape = [random.randrange(rng[0], rng[1] + 1)\
                        for rng in shape_rng]
                #print(n_dim, shape)
                p_wall = random.uniform(*p_wall_rng)
                grid = randomBinaryGrid(shape, p_wall)
                yield GridUnweightedUndirectedGraph(grid, move_kwargs,\
                        n_state_func, connected_func)
    return

def randomRestrictedBinaryGridUnweightedDirectedGraphGenerator(\
        n_dim_rng: Tuple[int],\
        shape_rng_func: Callable[[int], Tuple[Tuple[int]]],\
        p_wall_rng: Tuple[float],\
        n_diag_rng_func: Callable[[int], Tuple[int]],\
        axis_restricts: Dict[int, Tuple[bool]],\
        count: Optional[int]=None)\
        -> Generator["GridUnweightedUndirectedGraph", None, None]:
    
    iter_obj = itertools.count(0) if count is None else range(count)
    move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
            "allowed_direct_idx_func": None,\
            "directed_axis_restrict": None}
    n_state_func = lambda grid, grid_idx: 1 - grid.arr_flat[grid_idx]
    connected_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: True
    
    for _ in iter_obj:
        n_dim = random.randrange(n_dim_rng[0], n_dim_rng[1] + 1)
        #print(n_dim_rng, n_dim)
        shape = [random.randrange(rng[0], rng[1] + 1)\
                for rng in shape_rng_func(n_dim)]
        #print(shape)
        n_diag_rng = n_diag_rng_func(n_dim)
        move_kwargs["n_diag"] =\
                random.randrange(n_diag_rng[0], n_diag_rng[1] + 1)
        p_wall = random.uniform(*p_wall_rng)
        grid = randomBinaryGrid(shape, p_wall)
        yield GridUnweightedUndirectedGraph(grid, move_kwargs,\
                n_state_func, connected_func)
    
    return

def testRandomRestrictedBinaryGridUnweightedDirectedGraphsGenerator(\
        n_graph_per_opt: int,\
        n_dim_rng: Tuple[int],\
        shape_rng_func: Callable[[int], Tuple[Tuple[int]]],\
        p_wall_rng: Tuple[float],\
        n_diag_rng_func: Callable[[int], Tuple[int]],\
        count: Optional[int]=None)\
        -> Generator["GridUnweightedUndirectedGraph", None, None]:
    
    move_kwargs = {"n_diag": 0, "n_step_cap": 1,\
            "allowed_direct_idx_func": None}
    n_state_func = lambda grid, grid_idx: 1 - grid.arr_flat[grid_idx]
    connected_func = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: True
    
    for n_dim in range(n_dim_rng[0], n_dim_rng[1] + 1):
        shape_rng = shape_rng_func(n_dim)
        #print(n_dim, shape_rng)
        n_diag_rng = n_diag_rng_func(n_dim)
        for n_diag in range(n_diag_rng[0], n_diag_rng[1] + 1):
            move_kwargs["n_diag"] = n_diag
            for _ in range(n_graph_per_opt):
                shape = [random.randrange(rng[0], rng[1] + 1)\
                        for rng in shape_rng]
                #print(n_dim, shape)
                p_wall = random.uniform(*p_wall_rng)
                grid = randomBinaryGrid(shape, p_wall)
                yield GridUnweightedUndirectedGraph(grid, move_kwargs,\
                        n_state_func, connected_func)
    return
