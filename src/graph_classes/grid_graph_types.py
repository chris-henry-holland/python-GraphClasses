#! /usr/bin/env python
import bisect
import functools

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

from graph_classes.limited_graph_types import (
    LimitedGraphTemplate,
    LimitedWeightedGraphTemplate,
    LimitedUnweightedGraphTemplate,
    LimitedDirectedGraphTemplate,
    LimitedUndirectedGraphTemplate,
    LimitedUnweightedDirectedGraphTemplate,
    LimitedWeightedDirectedGraphTemplate,
    LimitedUnweightedUndirectedGraphTemplate,
    LimitedWeightedUndirectedGraphTemplate,
)

class Grid(object):
    """
    Class whose instances represent a grid with an arbitrary number of
    dimensions.
    
    Initialisation args:
        Required positional:
        n_dim (int): Non-negative integer giving the number of
                dimensions of the grid.
        arr (list): Representation of the grid of objects as nested
                lists, where the degree of nesting is at least
                n_dim and each list at the same level of nesting
                is the same length.
    
    Attributes:
        n_dim (int): Non-negative integer giving the number of
                dimensions of the grid.
        shape (n_dim-tuple of ints): Tuple of strictly positive ints
                giving the shape of the grid (i.e. the length of
                lists at each level of nesting in order of increasing
                levels of nesting.
        length (int): Strictly positive integer giving the total number
                of elements in the grid (equal to the product of all
                elements of attribute shape, or 1 if n_dim is 0)
        arr_flat (list of objects): List containing the flattened
                version of the grid (constructed by recursively
                concatenating the nested lists in the input argument
                arr from inner to outer layers of nesting).
                For example, if arr is a 3-dimensional grid:
                    [[[1, 2], [3, 4], [5, 6]],
                    [[7, 8], [9, 10], [11, 12]]]
                then arr_flat will be:
                    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    
    Indexing:
        Accessing an in
    
    Methods:
        
    """
    @classmethod
    def createGridFromFlat(cls, shape: Tuple[int], arr_flat: List[Any])\
            -> "Grid":
        if functools.reduce(lambda x, y: x * y, shape, 1) !=\
                len(arr_flat):
            raise ValueError("The product of the elements of shape "\
                    "(where an empty tuple has the product 1) must "\
                    "equal the length of arr_flat")
        res = cls(0, [1])
        res.n_dim = len(shape)
        res.shape = tuple(shape)
        res.length = len(arr_flat)
        res.arr_flat = list(arr_flat)
        return res
    
    def __init__(self, n_dim: int, arr: Any):
        self.n_dim = n_dim
        if not n_dim:
            self.length = 1
            self.shape = ()
            self.arr_flat = [arr]
        else:
            shape = [0] * n_dim
            length = 1
            curr = arr
            for i in range(n_dim):
                shape[i] = len(curr)
                length *= shape[i]
                if not len(curr): break
                curr = curr[0]
            self.length = length
            self.shape = tuple(shape)
            self.arr_flat = self._flattenArray(arr)
    
    def _flattenArray(self, arr: List[Any]) -> List[Any]:
        res = [None] * self.length
        def recur(i: int=0, idx: int=0, arr2: Any=arr) -> None:
            if i == self.n_dim:
                res[idx] = arr2
                return
            idx *= self.shape[i]
            for j in range(self.shape[i]):
                recur(i=i + 1, idx=idx + j, arr2=arr2[j])
            return
        recur()
        return res
    
    def index2Coordinates(self, idx: int) -> Tuple[int]:
        if not 0 <= idx < self.length:
            raise ValueError(f"The index must be an integer between "
                    f"0 and {self.length - 1} inclusive.")
        coords = [0] * self.n_dim
        for i in reversed(range(self.n_dim)):
            idx, coords[i] = divmod(idx, self.shape[i])
        return tuple(coords)
    
    def coordinates2Index(self, coords: Tuple[int]) -> int:
        idx = 0
        for i, (j, s) in enumerate(zip(coords, self.shape)):
            if not -s <= j < s:
                raise ValueError(f"The index for dimension {i} must "
                        f"be between {-s} and {s - 1} inclusive.")
            idx = idx * s + (j if j >= 0 else s + j)
        return idx
    
    def coordGenerator(self) -> Generator[Tuple[int], None, None]:
        """
        Generators that yield the coordinates in the grid in the order
        they appear in the attribute arr_flat
        """
        curr = [0] * self.n_dim
        def recur(i_d: int) -> Generator[Tuple[int], None, None]:
            if i_d == self.n_dim:
                yield tuple(curr)
                return
            for i in range(self.shape[i_d]):
                curr[i_d] = i
                yield from recur(i_d + 1)
            return
        yield from recur(0)
        return
    
    def coordValueGenerator(self)\
            -> Generator[Tuple[Union[Tuple[int], Any]], None, None]:
        yield from zip(self.coordGenerator, self.arr_flat)
        return
    
    def stepIndexGenerator(self, idx: int,\
            n_diag: int=0,\
            directed_axis_restrict: Optional[Dict[int, Tuple[bool]]]=None,\
            allowed_direct_idx_func: Optional[Callable[[int, int],\
            bool]]=None) -> Generator[Tuple[int], None, None]:
        
        mx_axis_steps = [[0, 0] for _ in range(self.n_dim)]
        idx2 = idx
        for axis_idx in reversed(range(self.n_dim)):
            axis_len =  self.shape[axis_idx]
            idx2, j = divmod(idx2, axis_len)
            axis_restrict = directed_axis_restrict.get(axis_idx, (False, False))
            mx_axis_steps[axis_idx] = [(not axis_restrict[0]) * j,\
                    (not axis_restrict[1]) * (axis_len - j - 1)]
        #print(mx_axis_steps)
        if directed_axis_restrict is None:
            directed_axis_restrict = {}
        if allowed_direct_idx_func is None:
            allowed_direct_idx_func = (lambda x, y: True)
        
        if not hasattr(self, "idx_tail_increments"):
            self.idx_tail_increments = [1] * self.n_dim
            curr_incr_idx = 1
            for axis_idx in reversed(range(self.n_dim)):
                self.idx_tail_increments[axis_idx] = curr_incr_idx
                curr_incr_idx *= self.shape[axis_idx]
        #mx_n_chng = n_diag + 1
        
        def recur(axis_idx: int, curr_step_idx: int,\
                mx_n_step: Union[int, float]=float("inf"),\
                n_chng_allowed: int=n_diag + 1)\
                -> Generator[Tuple[int], None, None]:
            if axis_idx == self.n_dim:
                if curr_step_idx and allowed_direct_idx_func(idx, curr_step_idx):
                    yield (curr_step_idx, mx_n_step)
                return
            curr_step_idx *= self.shape[axis_idx]
            if not n_chng_allowed:
                curr_step_idx *= self.idx_tail_increments[axis_idx]
                if allowed_direct_idx_func(idx, curr_step_idx):
                    yield (curr_step_idx, mx_n_step)
                return
            
            mx_steps = mx_axis_steps[axis_idx]
            if mx_steps[0]:
                yield from recur(axis_idx + 1, curr_step_idx - 1,\
                        min(mx_n_step, mx_steps[0]), n_chng_allowed - 1)
            yield from recur(axis_idx + 1, curr_step_idx, mx_n_step, n_chng_allowed)
            if mx_steps[1]:
                yield from recur(axis_idx + 1, curr_step_idx + 1,\
                        min(mx_n_step, mx_steps[1]), n_chng_allowed - 1)
        yield from recur(0, 0)
        return
        
    """
    def stepIndexGenerator(self, idx: int, n_diag: int=0,\
            allowed_direct_idx_func: Optional[Callable[[int, int],\
            bool]]=None) -> Generator[Tuple[int], None, None]:
        mx_n_steps = [[0, 0] for _ in range(self.n_dim)]
        idx2 = idx
        for i in reversed(range(self.n_dim)):
            idx2, j = divmod(idx2, self.shape[i])
            mx_n_steps[i] = [j, self.shape[i] - j - 1]
        if allowed_direct_idx_func is None:
            allowed_direct_idx_func = (lambda x, y: True)
        if not hasattr(self, "increments"):
            self.increments = [1] * self.n_dim
            curr = 1
            for i in reversed(range(self.n_dim)):
                self.increments[i] = curr
                curr *= self.shape[i]
        mx_n_chng = n_diag + 1
        
        def recur(i: int, curr: int,\
                n_step: Union[int, float]=float("inf"),\
                n_chng: int=0) -> Generator[Tuple[int], None, None]:
            if not n_step: return
            elif i == self.n_dim:
                if curr and allowed_direct_idx_func(idx, curr):
                    yield (curr, n_step)
                return
            curr *= self.shape[i]
            if n_chng == mx_n_chng:
                curr *= self.increments[i]
                if allowed_direct_idx_func(idx, curr):
                    yield (curr, n_step)
                return
            
            yield from recur(i + 1, curr - 1,\
                    min(n_step, mx_n_steps[i][0]), n_chng + 1)
            yield from recur(i + 1, curr, n_step, n_chng)
            yield from recur(i + 1, curr + 1,\
                    min(n_step, mx_n_steps[i][1]), n_chng + 1)
        yield from recur(0, 0, n_step=float("inf"), n_chng=0)
        return
    """
    
    def movesIndexGenerator(self, idx: int, n_diag: int=0,\
            n_step_cap: Union[int, float]=float("inf"),\
            directed_axis_restrict: Optional[Dict[int, Tuple[bool]]]=None,\
            allowed_direct_idx_func: Optional[Dict[int,\
            Tuple[int]]]=None,\
            block_func: Optional[Callable[[int], bool]]=None)\
            -> Generator[Tuple[int], None, None]:
        if directed_axis_restrict is None:
            directed_axis_restrict = {}
        if allowed_direct_idx_func is None:
            allowed_direct_idx_func = (lambda x, y: True)
        if block_func is None:# or not block_vals:
            for mv, n_step_mx in\
                    self.stepIndexGenerator(idx, n_diag=n_diag,\
                    directed_axis_restrict=directed_axis_restrict,\
                    allowed_direct_idx_func=allowed_direct_idx_func):
                idx2 = idx
                for n_step in range(1, min(n_step_mx, n_step_cap) + 1):
                    idx2 += mv
                    yield (idx2, mv, n_step)
            return
        for mv, n_step_mx in\
                self.stepIndexGenerator(idx, n_diag=n_diag,\
                directed_axis_restrict=directed_axis_restrict,\
                allowed_direct_idx_func=allowed_direct_idx_func):
            idx2 = idx
            for n_step in range(1, min(n_step_mx, n_step_cap) + 1):
                idx2 += mv
                if block_func(idx2):#self.arr_flat[idx2] in block_vals:
                    break
                yield (idx2, mv, n_step)
        return
    
    def __getitem__(self, coords: Tuple[int]) -> Any:
        return self.arr_flat[self.coordinates2Index(coords)]

class GridGraphTemplate(LimitedGraphTemplate):
    """
    Class whose instances represent a grid of an arbitrary number
    of dimensions as a weighted graph or directed graph where the
    elements of the grid are vertices, and the edges and their
    weights are defined by rules for moving between two given
    elements of the grid based on their values and relative
    positions in the grid.
    
    TODO
    """
    move_kwargs_def = {"n_diag": 0, "n_step_cap": 1}
    
    def __init__(self, grid: Grid,\
            move_kwargs: Optional[Dict[str, Any]]=None,\
            n_state_func: Optional[Callable[[Grid, int], int]]=None,\
            **kwargs):
        self.grid = grid
        self.n_dim = grid.n_dim
        self.shape = grid.shape
        if move_kwargs is None:
            self.move_kwargs = self.move_kwargs_def
        else:
            for k, v in self.move_kwargs_def.items():
                move_kwargs.setdefault(k, v)
            self.move_kwargs = move_kwargs
        # n_state_func uses grid index for its second argument (i.e.
        # position in flattened grid)
        has_blocks = False
        if n_state_func is None:
            self.n = grid.length
            self.n_state_func = lambda grid, grid_idx: 1
            self.n_state_runs_graph_starts = [0, self.n]
            self.n_state_runs_grid_starts = [0, self.n]
        else:
            self.n_state_func = n_state_func
            prev_n_state = 0
            graph_idx = 0
            n_state_runs_graph_starts = []
            n_state_runs_grid_starts = []
            for grid_idx, val in enumerate(grid.arr_flat):
                #print(grid_idx, graph_idx)
                n_state = n_state_func(grid, grid_idx)
                if n_state != prev_n_state:
                    if n_state:
                        n_state_runs_graph_starts.append(graph_idx)
                        n_state_runs_grid_starts.append(grid_idx)
                    else: has_blocks = True
                    prev_n_state = n_state
                graph_idx += n_state
            n_state_runs_graph_starts.append(graph_idx)
            n_state_runs_grid_starts.append(self.grid.length)
            self.n_state_runs_graph_starts = n_state_runs_graph_starts
            self.n_state_runs_grid_starts = n_state_runs_grid_starts
            self.n = graph_idx
            #print(graph_idx)
        """
        if n_state_func is not None:
            non_one_found = False
            vgi = []
            vgid = {}
            for grid_idx, val in enumerate(grid.arr_flat):
                n_state = n_state_func(grid, grid_idx)
                if n_state != 1: non_one_found = True
                if not n_state:
                    has_blocks = True
                    continue
                vgid[grid_idx] = (len(vgi), n_state)
                for i in range(n_state): vgi.append((grid_idx, i))
            if not non_one_found:
                self.n_state_func = lambda grid, grid_idx: 1
            else:
                self.n = len(vgi)
                self._vertices_grid_index = vgi
                self._vertices_grid_index_dict = vgid
        else: self.n_state_func = lambda grid, grid_idx: 1
        """
        #print(self.n)
        #print(self.n_state_runs_graph_starts)
        #print(self.n_state_runs_grid_starts)
        block_func = lambda grid_idx:\
                not self.gridIndexNState(grid_idx)\
                if has_blocks else None
        self.move_kwargs["block_func"] = block_func
        super().__init__(**kwargs)
    
    def index2Vertex(self, idx: int) -> Tuple[int]:
        grid_idx, state_idx = self._index2GridIndex(idx)
        return self._gridIndex2Vertex(grid_idx, state_idx)
    
    def gridIndexNState(self, grid_idx: int) -> int:
        return self.n_state_func(self.grid, grid_idx) if\
                0 <= grid_idx < self.grid.length else 0
        #if not hasattr(self, "_vertices_grid_index_dict"):
        #    return int(0 <= grid_idx < self.grid.length)
        #return self._vertices_grid_index_dict.get(grid_idx, (0, 0))[1]
    
    def _vertex2GridIndex(self, vertex: Tuple[Union[Tuple[int], int]])\
            -> Tuple[int]:
        return (self.grid.coordinates2Index(vertex[0]), vertex[1])
    
    def _gridIndex2Vertex(self, grid_idx: int, state_idx: int=0)\
            -> Tuple[int]:
        return (self.grid.index2Coordinates(grid_idx), state_idx)

    def _gridIndex2Index(self, grid_idx: int, state_idx: int=0) -> int:
        #if grid_idx < 0 or grid_idx >= self.grid.length:
        #    raise ValueError("grid_idx must be between 0 and "\
        #            f"{self.grid.length - 1} inclusive")
        i = bisect.bisect_right(self.n_state_runs_grid_starts,\
                grid_idx) - 1
        graph_idx1 = self.n_state_runs_graph_starts[i]
        grid_idx1 = self.n_state_runs_grid_starts[i]
        n_state1 = self.n_state_func(self.grid, grid_idx1)
        #if state_idx >= n_state1:
        #    raise ValueError("state_idx is too great for the given "\
        #            "grid_idx"
        
        res = graph_idx1 + (grid_idx - grid_idx1) * n_state1 + state_idx
        #grid_idx2 = self.n_state_runs[i + 1][1]
        #if res >= grid_idx2:
        #    raise ValueError("This grid_idx does not represent any "\
        #            "vertices in the graph")
        return res
        
        #return getattr(self, "_vertices_grid_index_dict",\
        #        {grid_idx: (grid_idx, 1)})[grid_idx][0] + state_idx
    
    def _index2GridIndex(self, idx: int) -> Tuple[int]:
        i = bisect.bisect_right(self.n_state_runs_graph_starts, idx)\
                - 1
        graph_idx1 = self.n_state_runs_graph_starts[i]
        grid_idx1 = self.n_state_runs_grid_starts[i]
        #print(self.n, grid_idx1, self.n_state_runs_graph_starts, i)
        q, r = divmod(idx - graph_idx1,\
                self.n_state_func(self.grid, grid_idx1))
        return (grid_idx1 + q, r)
        #return self._vertices_grid_index[idx]\
        #        if hasattr(self, "_vertices_grid_index") else (idx, 0)
    
    def vertex2Index(self, vertex: Tuple[Union[Tuple[int], int]])\
            -> int:
        grid_idx, state_idx = self._vertex2GridIndex(vertex)
        return self._gridIndex2Index(grid_idx, state_idx)
    
    def _gridIndexInGraph(self, grid_idx: int) -> bool:
        return bool(self.gridIndexNState(grid_idx))
        #if not hasattr(self, "_vertices_grid_index_dict"):
        #    return 0 <= grid_idx < self.grid.length
        #return grid_idx in self._vertices_grid_index_dict.keys()
    
    def coordInGraph(self, coord: Tuple[int]) -> bool:
        return self._gridIndexInGraph(\
                self.grid.coordinates2Index(coord))
        #if not hasattr(self, "_vertices_grid_index"):
        #    if len(coord) != len(self.shape): return False
        #    return all(isinstance(c, int) and 0 <= c < s\
        #            for c, s in zip(coord, self.shape))
        #return self._vertex2GridIndex(coord) in\
        #        self._vertices_grid_index_dict.keys()
    
    def vertexInGraph(self, vertex: Tuple[Union[Tuple[int]], int])\
            -> bool:
        if not hasattr(vertex, "__len__") or len(vertex) != 2:
            return False
        grid_coord, state_idx = vertex
        return isinstance(state_idx, int) and state_idx >= 0 and\
                self.coordInGraph(grid_coord) and\
                state_idx < self.gridIndexNState(\
                self.grid.coordinates2Index(grid_coord))
    
    def _getAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        grid_idx, state_idx = self._index2GridIndex(idx)
        res_grid_idx = self._getAdjGridIndex(grid_idx, state_idx)
        return {self._gridIndex2Index(grid_idx2, state_idx2): w_lst\
                for (grid_idx2, state_idx2), w_lst in\
                res_grid_idx.items()}
    
    def vertexGridIndexGenerator(self)\
            -> Generator[Tuple[int], None, None]:
        graph_idx2 = self.n_state_runs_graph_starts[0]
        #grid_idx2 = self.n_state_runs_grid_starts[0]
        for i in range(len(self.n_state_runs_graph_starts) - 1):
            grid_idx1 = self.n_state_runs_grid_starts[i]
            graph_idx1 = graph_idx2
            n_state = self.n_state_func(self.grid, grid_idx1)
            graph_idx2 = self.n_state_runs_graph_starts[i + 1]
            for grid_idx, _ in enumerate(range(graph_idx1,\
                    graph_idx2, n_state), start=grid_idx1):
                for state_idx in range(n_state):
                    yield (grid_idx, state_idx)
        return
            
        
        #if hasattr(self, "_vertices_grid_index"):
        #    for grid_idx, state_idx in self._vertices_grid_index:
        #        yield grid_idx, state_idx
        #else:
        #    for grid_idx in range(self.n):
        #        yield (grid_idx, 0)
        #return

    def vertexGenerator(self)\
            -> Generator[Tuple[Union[Tuple[int], int]], None, None]:
        for grid_idx, state_idx in self.vertexGridIndexGenerator():
            yield self._gridIndex2Vertex(grid_idx, state_idx)
        return
        #if hasattr(self, "_vertices_grid_index"):
        #    for grid_idx, state_idx in self._vertices_grid_index:
        #        yield self._gridIndex2Vertex(grid_idx, state_idx)
        #else:
        #    for coord in self.grid.coordGenerator():
        #        yield (coord, 0)
        #return

class GridWeightedGraphTemplate(GridGraphTemplate,\
        LimitedWeightedGraphTemplate):
    weight_func_def = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: grid.arr_flat[grid_idx1]

    def __init__(self, grid: Grid,\
            weight_func: Optional[\
                Callable[["GridWeightedGraphTemplate", int, int, int,\
                int, int, int], Optional[Union[int, float]]]]=None,\
            neg_weight_edge: Optional[bool]=None, **kwargs):
        super().__init__(grid=grid, **kwargs)
        self.weight_func = self.weight_func_def if weight_func is None\
                            else weight_func
        if neg_weight_edge is not None:
            # Already known whether or not has negative weight edges
            self._neg_weight_edge = neg_weight_edge

    def _getAdjGridIndex(self, grid_idx: int, state_idx: int=0)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        res = {}
        for (grid_idx2, mv, n_step) in\
                self.grid.movesIndexGenerator(grid_idx,\
                **self.move_kwargs):
            for state_idx2 in range(self.gridIndexNState(grid_idx2)):
                w = self.weight_func(self.grid, grid_idx, state_idx,\
                        grid_idx2, state_idx2, mv, n_step)
                if w is not None: res[(grid_idx2, state_idx2)] = [w]
        return res

class GridUnweightedGraphTemplate(GridGraphTemplate,\
        LimitedUnweightedGraphTemplate):
    connected_func_def = lambda grid, grid_idx1, state_idx1, grid_idx2,\
            state_idx2, mv, n_step: True

    def __init__(self, grid: Grid,\
            connected_func: Optional[\
                Callable[["GridUnweightedGraphTemplate", int, int,\
                int, int, int, int], bool]]=None,\
            **kwargs):
        super().__init__(grid, **kwargs)
        self.connected_func = self.connected_func_def if\
                connected_func is None else connected_func
        self._neg_weight_edge = False
    
    def _getAdjGridIndex(self, grid_idx: int, state_idx: int=0)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        res = {}
        for (grid_idx2, mv, n_step) in\
                self.grid.movesIndexGenerator(grid_idx,\
                **self.move_kwargs):
            for state_idx2 in range(self.gridIndexNState(grid_idx2)):
                if self.connected_func(self.grid, grid_idx, state_idx,\
                        grid_idx2, state_idx2, mv, n_step):
                    res[(grid_idx2, state_idx2)] = 1
        return res

class GridUndirectedGraphTemplate(GridGraphTemplate,\
        LimitedUndirectedGraphTemplate):

    def __init__(self, grid: Grid, **kwargs):
        super().__init__(grid, **kwargs)

class GridDirectedGraphTemplate(GridGraphTemplate,\
        LimitedDirectedGraphTemplate):
    
    def __init__(self, grid: Grid, store_in_degrees: bool=False,\
            **kwargs):
        super().__init__(grid, **kwargs)
        self.store_in_degrees = store_in_degrees
        self.store_in_adj = False
        if store_in_degrees:
            self._in_degrees_index = [None] * self.n

    def _getInAdjIndex(self, idx: int)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        grid_idx, state_idx = self._index2GridIndex(idx)
        res_grid_idx = self._getInAdjGridIndex(grid_idx, state_idx)
        return {self._gridIndex2Index(grid_idx2, state_idx2): w_lst\
                for (grid_idx2, state_idx2), w_lst in\
                res_grid_idx.items()}

class GridUnweightedUndirectedGraph(GridUnweightedGraphTemplate,\
        GridUndirectedGraphTemplate,\
        LimitedUnweightedUndirectedGraphTemplate):
    def __init__(self, grid: Grid,\
            move_kwargs: Optional[Dict[str, Any]]=None,\
            n_state_func: Optional[Callable[[Grid, int], int]]=None,\
            connected_func: Optional[\
                Callable[["GridUnweightedUndirectedGraph", int, int, int,\
                int, int, int], bool]]=None):
        super().__init__(grid, move_kwargs=move_kwargs,\
                n_state_func=n_state_func,\
                connected_func=connected_func)

class GridUnweightedDirectedGraph(GridUnweightedGraphTemplate,\
        GridDirectedGraphTemplate,\
        LimitedUnweightedDirectedGraphTemplate):
    def __init__(self, grid: Grid,\
            move_kwargs: Optional[Dict[str, Any]]=None,\
            n_state_func: Optional[Callable[[Grid, int], int]]=None,\
            connected_func: Optional[\
                Callable[["GridUnweightedDirectedGraph", int, int, int,\
                int, int, int], bool]]=None,\
            store_in_degrees: bool=False):
        super().__init__(grid, move_kwargs=move_kwargs,\
                n_state_func=n_state_func,\
                connected_func=connected_func,\
                store_in_degrees=store_in_degrees)
    
    def _getInAdjGridIndex(self, grid_idx: int, state_idx: int=0,\
            record_in_adj: bool=False)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        res = {}
        for (grid_idx0, mv, n_step) in\
                self.grid.movesIndexGenerator(grid_idx,\
                **self.move_kwargs):
            for state_idx0 in range(self.gridIndexNState(grid_idx0)):
                if self.connected_func(self.grid, grid_idx0,\
                        state_idx0, grid_idx, state_idx, -mv, n_step):
                    res[(grid_idx0, state_idx0)] = 1
        return res

class GridWeightedUndirectedGraph(GridWeightedGraphTemplate,\
        GridUndirectedGraphTemplate,\
        LimitedWeightedUndirectedGraphTemplate):
    def __init__(self, grid: Grid,\
            move_kwargs: Optional[Dict[str, Any]]=None,\
            n_state_func: Optional[Callable[[Grid, int], int]]=None,\
            weight_func: Optional[\
                Callable[["GridWeightedUndirectedGraph", int, int, int,\
                int, int, int], Optional[Union[int, float]]]]=None,\
            neg_weight_edge: Optional[bool]=None):
        super().__init__(grid, move_kwargs=move_kwargs,\
                n_state_func=n_state_func, weight_func=weight_func,\
                neg_weight_edge=neg_weight_edge)

class GridWeightedDirectedGraph(GridWeightedGraphTemplate,\
        GridDirectedGraphTemplate,\
        LimitedWeightedDirectedGraphTemplate):
    """
    Class whose instances represent a grid of an arbitrary number
    of dimensions as a weighted graph or directed graph where the
    elements of the grid are vertices, and the edges and their
    weights are defined by rules for moving between two given
    elements of the grid based on their values and relative
    positions in the grid.
    
    TODO
    """
    
    def __init__(self, grid: Grid,\
            move_kwargs: Optional[Dict[str, Any]]=None,\
            n_state_func: Optional[Callable[[Grid, int], int]]=None,\
            weight_func: Optional[\
                Callable[["GridWeightedDirectedGraph", int, int, int,\
                int, int, int], Optional[Union[int, float]]]]=None,\
            neg_weight_edge: Optional[bool]=None,\
            store_in_degrees: bool=False):
        super().__init__(grid, move_kwargs=move_kwargs,\
                n_state_func=n_state_func, weight_func=weight_func,\
                neg_weight_edge=neg_weight_edge,\
                store_in_degrees=store_in_degrees)
    
    def _getInAdjGridIndex(self, grid_idx: int, state_idx: int=0,\
            record_in_adj: bool=False)\
            -> Dict[int, Union[int, List[Union[int, float]]]]:
        res = {}
        for (grid_idx0, mv, n_step) in\
                self.grid.movesIndexGenerator(grid_idx,\
                **self.move_kwargs):
            for state_idx0 in range(self.gridIndexNState(grid_idx0)):
                w = self.weight_func(self.grid, grid_idx0, state_idx0,\
                        grid_idx, state_idx, -mv, n_step)
                if w is not None: res[(grid_idx0, state_idx0)] = [w]
        return res
