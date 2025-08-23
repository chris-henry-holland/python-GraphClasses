#!/usr/bin/env python

from typing import Dict, List, Set, Tuple, Optional, Union

import heapq
import math
import random
from sortedcontainers import SortedList

from algorithms.number_theory_algorithms import gcd
from data_structures.fractions import CustomFraction

def determinant(
    mat: List[List[Union[int, float]]]
) -> Union[int, float]:
    """
    Calculates the determinant of a square matrix mat.

    Args:
        Required positional:
        mat (list of lists of real numeric values): The matrix
                whose determinant is to be calculated, expressed
                as a list of the rows of the matrix. This is
                required be a square matrix (i.e. all of the
                lists within mat are the same length as each
                other and mat itself).
    
    Returns:
    Real numeric value (int or float) giving the determinant of
    the matrix mat.
    """
    n = len(mat)
    cols = SortedList(range(n))
    def recur(start_row_idx: int) -> Union[int, float]:
        if len(cols) == 1:
            return mat[start_row_idx][cols[0]]
        mult = 1
        res = 0
        for i in range(len(cols)):
            col_idx = cols.pop(i)
            mult2 = ((not col_idx & 1) << 1) - 1
            res += mult * mat[start_row_idx][col_idx] *\
                    recur(start_row_idx + 1)
            cols.add(col_idx)
            mult *= -1
        return res
    return recur(0)

def circumcircle(
    points: List[Tuple[Union[int, float]]],
) -> Tuple[Union[Tuple[Union[int, float]], Union[int, float]]]:
    """
    For a set of between 1 and 3 points (inclusive) in the 2d plane,
    finds the centre and radius squared of the smallest circle passing
    through every one of those points (the so-called circumcircle),
    specifying the circle by the Cartesian coordinates of its centre
    and its radius squared.
    
    If three points are given, these should not be colinear (i.e. there
    should not exist a single straight line that passes exactly through
    all three points).

    Args:
        Required positional:
        points (list of 2-tuples of real numeric values): A list of no
                more than 3 points, expressed in Cartesian coordinates
                for which the circumcircle is to be calculated. For
                lists of three points, these should not be colinear.
    
    Returns:
    2-tuple whose index 0 contains a 2-tuple of real numeric values
    specifying the location of the centre of the circumcircle
    identified in Cartesian coordinates and whose index 1 contains
    a non-negative numeric value specifying the radius squared of
    the circumcircle.
    """
    if not (1 <= len(points) <= 3):
        raise ValueError("Function circumcircle() may only be applied to "
                "between 1 and 3 points inclusive.")
    if len(points) == 1:
        return (points[0], 0)
    elif len(points) == 2:
        diam_sq = sum((x - y) ** 2 for x, y in zip(*points))
        rad_sq = (diam_sq >> 2) if isinstance(diam_sq, int) and not diam_sq & 3 else (diam_sq / 4)
        return (tuple((x + y) / 2 for x, y in zip(*points)), rad_sq)
    # Based on https://en.wikipedia.org/wiki/Circumcircle
    abs_sq_points = [sum(x ** 2 for x in pt) for pt in points]
    a = determinant([[*y, 1] for y in points])
    if not a:
        raise ValueError("The three points given to the function circumcircle() "
                "were colinear, in which case there is no finite circumcircle")
    S = (determinant([[x, y[1], 1] for x, y in zip(abs_sq_points, points)]),\
            -determinant([[x, y[0], 1] for x, y in zip(abs_sq_points, points)]))
    
    centre = tuple(x // (2 * a) if isinstance(x, int) and isinstance(a, int) and\
            not x % (2 * a) else x / (2 * a) for x in S)
    rad_sq = sum((x - y) ** 2 for x, y in zip(points[0], centre))
    return (centre, rad_sq)

def welzl(
    points: List[Tuple[Union[int, float]]],
) -> Tuple[Union[Tuple[Union[int, float]], Union[int, float]]]:
    """
    Uses the Welzl algorithm to find the centre and radius squared of
    the smallest circle that encloses every one of a set of points
    in the 2D plane (where points on the edge of circle itself are
    considered to be enclosed by the circle).

    Args:
        Required positional:
        points (list of 2-tuples with numeric values): The 2D Cartesian
                coordinates of the points to be enclosed.
    
    Returns:
    2-tuple whose index 0 contains a 2-tuple of real numeric values
    specifying the location of the centre of the enclosing circle
    identified in Cartesian coordinates and whose index 1 contains
    a non-negative numeric value specifying the radius squared of
    that enclosing circle.
    """

    # Based on https://en.wikipedia.org/wiki/Smallest-circle_problem
    points = list(set(tuple(x) for x in points))
    n = len(points)
    random.shuffle(points)
    boundary_points = []

    def recur(idx: int) -> Tuple[Union[Tuple[Union[int, float]],\
            Union[int, float]]]:
        if idx == n or len(boundary_points) == 3:
            #print(boundary_points)
            if not boundary_points:
                return ((0, 0), -1)
            return circumcircle(boundary_points)
        pt = points[idx]
        centre, rad_sq = recur(idx + 1)
        if sum((x - y) ** 2 for x, y in zip(centre, pt)) <= rad_sq:
            return centre, rad_sq
        boundary_points.append(pt)
        centre, rad_sq = recur(idx + 1)
        boundary_points.pop()
        return centre, rad_sq
    return recur(0)

def smallestCircularEnclosure(trees: List[List[int]]) -> List[float]:
    """
    Finds the circular enclosure with the smallest circumference that
    encloses every 2D point in the list trees. Points that fall on
    the edge of the enclosure are considered to be inside the enclosure.

    Args:
        Required positional:
        trees (list of list of ints): The 2D points to be enclosed,
                expressed in Cartesian coordinates (such that each
                element of the list is a length 2 list containing
                integers).
    
    Returns:
    A list containing exactly three real numeric values, with the
    first two giving the x and y coordinates of the centre of the
    circular enclosure respectively and the final value giving its
    radius.

    Solution to Leetcode #1924 (Premium): Erect the Fence II
    
    Original problem description of Leetcode #1924:
    
    You are given a 2D integer array trees where trees[i] = [xi, yi]
    represents the location of the ith tree in the garden.

    You are asked to fence the entire garden using the minimum length
    of rope possible. The garden is well-fenced only if all the trees
    are enclosed and the rope used forms a perfect circle. A tree is
    considered enclosed if it is inside or on the border of the circle.

    More formally, you must form a circle using the rope with a center
    (x, y) and radius r where all trees lie inside or on the circle and
    r is minimum.

    Return the center and radius of the circle as a length 3 array
    [x, y, r]. Answers within 10-5 of the actual answer will be
    accepted.
    """
    eps = 10 ** -5
    centre, rad_sq = welzl(trees, eps=eps)
    return [*centre, math.sqrt(rad_sq)]

def grahamScan(
    points: List[Tuple[Union[int, float]]],
    include_border_points: bool=False,
) -> List[Tuple[Union[int, float]]]:
    """
    Implementation of the Graham scan to find the convex hull of a set
    of points in 2 dimensional space expressed in Cartesian coordinates.

    Args:
        Required positional:
        points (list of 2-tuples of real numeric values): The points
                in two dimensional space for which the convex hull
                is to be found.
        
        Optional named:
        include_border_points (bool): Whether to include elements of
                points that fall on an edge of the convex hull but are
                not vertices of the convex hull (i.e. points that are
                on the line directly between two consecutive vertices
                of the convex hull).
            Default: False

    Returns:
    The points in the convex hull expressed in Cartesian coordinates,
    ordered such that they cycle the convex hull in an anticlockwise
    direction, with the point with the smallest x-coordinate (and if
    there are several of these, the one of those with the smallest
    y-coordinate) as the first element.

    Examples:
        >>> grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=False)
        [(1, 1), (2, 0), (4, 2), (2, 4)]

        >>> grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=True)
        [(1, 1), (2, 0), (4, 2), (3, 3), (2, 4)]

        Note that this includes the point (3, 3) which was omitted by
        the previous example (which consists of the same points but
        gives the parameter include_border_points as False) as it
        falls directly on the line between the points (4, 2) and (2, 4),
        which are consecutive vertices of the convex hull.
    """
    points = [tuple(x) for x in points]
    if len(points) < 3: return sorted(set(points))

    comp = (lambda x, y: x <= y) if include_border_points else (lambda x, y: x < y)
    
    ref_pt = min(points)
    sorted_pts = []
    min_x_pts = []
    for pos in points:
        if pos[0] == ref_pt[0]:
            min_x_pts.append(pos)
            continue
        diff = tuple(x - y for x, y in zip(pos, ref_pt))
        slope = diff[1] / diff[0]
        sorted_pts.append((slope, diff, pos))
    sorted_pts.sort()
    
    if len(min_x_pts) > 1:
        if include_border_points:
            tail = sorted(min_x_pts)
            pos = tail.pop()
            sorted_pts.append((None, tuple(x - y for x, y in zip(pos, ref_pt)), pos))
            tail = tail[::-1]
        else:
            pos = max(min_x_pts)
            sorted_pts.append((None, tuple(x - y for x, y in zip(pos, ref_pt)), pos))
            tail = [ref_pt]
    else:
        tail = []
        tup0 = sorted_pts.pop()
        diff0 = tup0[1]
        while sorted_pts and diff0[0] * sorted_pts[-1][1][1] == diff0[1] * sorted_pts[-1][1][0]:
            tail.append(sorted_pts.pop()[2])
        tail = tail[::-1] if include_border_points else []
        tail.append(ref_pt)
        sorted_pts.append(tup0)
    stk = [(sorted_pts[0][2], tuple(x - y for x, y in zip(sorted_pts[0][2], ref_pt)))]
    order = [x[0] for x in stk]
    for i in range(1, len(sorted_pts)):
        pos = sorted_pts[i][2]
        while stk:
            diff = tuple(x - y for x, y in zip(pos, stk[-1][0]))
            cross_prod = stk[-1][1][0] * diff[1] -\
                    stk[-1][1][1] * diff[0]
            if comp(0, cross_prod): break
            stk.pop()
        
        stk.append((pos, tuple(x - y for x, y in zip(pos, (stk[-1][0] if stk else ref_pt)))))
    res = [x[0] for x in stk] + tail

    return [res[-1], *res[:-1]]

def outerTrees(trees: List[List[int]]) -> List[List[int]]:
    """
    Given the two dimensional Cartesian coordinates of multiple trees,
    consider the fence of minimal length that encloses all of the
    trees (with trees in contact with the fence considered to be
    enclosed). This function finds the coordinates of the trees that
    are in contact with the fence.

    Args:
        Required positional:
        trees (list of lists of ints): List of 2-dimensional integer
                Cartesian coordinates (each given as lists of length 2
                with integer values) representing the positions of the
                trees to be enclosed.

    Returns:
    List of lists, where each element of the outer list has length 2
    and contains integers (int), giving the Cartesian coordinates of
    all the trees that are in contact with the minimal length fence
    that encloses all of the trees given.
    
    Examples:
        >>> outerTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]])
        [[1, 1], [2, 0], [4, 2], [3, 3], [2, 4]]

        >>> outerTrees([[1, 2], [2, 2], [4, 2]])
        [[1, 2], [4, 2], [2, 2]]
    
    Solution to Leetcode #587: Erect the Fence

    Original problem description for Leetcode #587:

    You are given an array trees where trees[i] = [xi, yi] represents the
    location of a tree in the garden.

    Fence the entire garden using the minimum length of rope, as it is
    expensive. The garden is well-fenced only if all the trees are
    enclosed.

    Return the coordinates of trees that are exactly located on the fence
    perimeter. You may return the answer in any order.
    """
    return [list(y) for y in grahamScan([tuple(x) for x in trees], include_border_points=True)]

def twoDimensionalLineEquationGivenTwoIntegerPoints(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
) -> Tuple[int, int, int]:
    """
    Calculates the equation for the line in two dimensional space
    which contains the two distinct points p1 and p2 that both have
    integer coordinates in the Cartesian plane. The equation is
    expressed as the ordered integer triple (a, b, c) representing
    the equation of a line of the form:
        a * x + b * y = c
    where (x, y) represents the Cartesian coordinates of a point on
    the line a, b and c are integers, a and b are coprime, a is
    non-negative and if a is zero then b is strictly positive.
    These conditions mean that there is exactly one such triple
    (a, b, c) whose corresponding equation describes this line
    (which is the ordered triple returned).

    Args:
        Required postional:
        p1 (2-tuple of ints): Cartesian coordinates of the first of
                the two points through which the line whose equation
                is to be found must pass through.
        p2 (2-tuple of ints): Cartesian coordinates of the first of
                the two points through which the line whose equation
                is to be found must pass through. Must be different
                from p1.
    
    Returns:
    3-tuple of ints giving the ordered triple (a, b, c) (in that
    order) uniquely representing the equation of the line passing
    through p1 and p2 subject to the stated conditions.
    """
    if (p1 == p2): raise ValueError("p1 and p2 must be different")
    a, b, c = ((p1[1] - p2[1]), (p2[0] - p1[0]), (p2[0] * p1[1] - p1[0] * p2[1]))
    if a < 0 or a == 0 and b < 0:
        a, b, c = -a, -b, -c
    g = gcd(a, b)
    #if g == 1: return (a, b, c)
    return (a // g, b // g, c // g)

def twoDimensionalLineSegmentPairCrossing(
        seg1: Tuple[Tuple[int, int], Tuple[int, int]],
        seg2: Tuple[Tuple[int, int], Tuple[int, int]],
        internal_only: bool=True,
        allow_endpoint_to_endpoint: bool=False,
) -> Optional[Tuple[CustomFraction, CustomFraction]]:
    """
    Identifies whether two line segments in two dimensions whose end points
    are at integer Cartesian coordinates cross each other, and if so gives the
    Cartesian coordinates at which they cross with the coordinates as fractions.

    When internal_only is True, only internal crossings of line segments are
    considered to be crossings, while when internal_only is False, having one
    line segment's endpoint coincide with a point on the other line segment
    that is not the end point is also considered an intersection, and if
    additionally allow_endpoint_to_endpoint is True then one of each line
    segment's end points coinciding is considered an intersection.

    A line segment is a straight line extending between two points (these
    points referred to as the end points of the line segment).

    An internal crossing of two line segments is a point that is on both line
    segments but not on either endpoint of either line segment (note that by
    Euclidean geometry at most one such line crossing can occur). Two line
    segments may be referred to as crossing internally if and only if there
    exists an internal crossing for those two line segments.

    Args:
        Required positional:
        seg1 (2-tuple of 2-tuple of ints): 2-tuple containing the two end
                points of one of the line segments whose status as crossing the
                other internally is being assessed, each as a 2-tuple containing
                the Cartesian coordinates of that end point.
        seg2 (2-tuple of 2-tuple of ints): 2-tuple containing the two end
                points of the other of the line segments whose status as crossing
                the other internally is being assessed, each as a 2-tuple
                containing the Cartesian coordinates of that end point.
        
        Optional named:
        internal_only (bool): Specifies whether only internal crossings are to
                be considered as intersections (if given as True), or if crossings
                involving end points of the line segments are also to be considered
                as intersections (if given as False)
            Default: True
        allow_endpoint_to_endpoint (bool): Specifies whether, when internal_only
                if False, coincidences between one of the end points of each
                line segment is to be considered as an intersection (if True
                and internal_only is False then these are considered to be
                intersections, otherwise not).
            Default: False
    
    Returns:
    A 2-tuple of CustomFraction objects representing the Cartesian coordinates
    of the intersection point as fractions if the two line segments intersect (where
    the events considered to be intersections are defined based on internal_only
    and allow_endpoint_to_endpoint as outlined above), otherwise None.
    """
    #e_ref = ((356, 290), (356, 396))
    #if e_ref in {tuple(seg1), tuple(seg2)}:
    #    print(seg1, seg2)
    #print("Using twoDimensionalLineSegmentPairCrossing()")
    failed_screen = False
    #print(seg1, seg2)
    for i in range(2):
        if min(seg1[0][i], seg1[1][i]) > max(seg2[0][i], seg2[1][i]) or min(seg2[0][i], seg2[1][i]) > max(seg1[0][i], seg1[1][i]):
            #failed_screen = True
            #print("bounding boxes do not overlap")
            return None
    """
    #ans1 = True
    pt_lst = [seg1[0], seg2[0], seg1[1], seg2[1]]
    vec_lst = [[y - x for x, y in zip(pt1, pt_lst[(i + 1) % len(pt_lst)])] for i, pt1 in enumerate(pt_lst)]
    angle_direction_set = set()
    for i, v1 in enumerate(vec_lst):
        v2 = vec_lst[(i + 1) % len(vec_lst)]
        num1, num2 = v1[0] * v2[1], v1[1] * v2[0]
        if num1 == num2: return None
        angle_direction_set.add(num1 > num2)
        if len(angle_direction_set) > 1: return None
    #if failed_screen: print(seg1, seg2, vec_lst, angle_direction_set)
    #return True
    a1, b1, c1 = twoDimensionalLineEquation(*seg1)
    a2, b2, c2 = twoDimensionalLineEquation(*seg2)
    denom = a1 * b2 - a2 * b1
    x, y = (b2 * c1 - b1 * c2, denom), (a2 * c1 - a1 * c2, denom)
    if denom > 0: y = y = tuple(-a for a in y)
    else: x = tuple(-a for a in x)
    g1, g2 = gcd(abs(x[0]), x[1]), gcd(abs(y[0]), y[1])
    #ans1 = True
    return (tuple(a // g1 for a in x), tuple(a // g2 for a in y))
    """
    
    ans2 = True
    eqn1 = twoDimensionalLineEquationGivenTwoIntegerPoints(*seg1)
    eqn2 = twoDimensionalLineEquationGivenTwoIntegerPoints(*seg2)
    a1, b1, c1 = eqn1
    a2, b2, c2 = eqn2

    
    #print(a1, b1, c1)
    #print(a2, b2, c2)

    if a1 == a2 and b1 == b2:
        if c1 == c2: print(f"collinear segments: {seg1}, {seg2}")

        ans2 = False#return False # Parallel
        return None

    denom = a1 * b2 - a2 * b1
    x, y = CustomFraction(b2 * c1 - b1 * c2, denom), CustomFraction(a1 * c2 - a2 * c1, denom)
    #if denom > 0: y = tuple(-a for a in y)
    #else: x = tuple(-a for a in x)
    #if denom < 0:
    #    x = tuple(-z for z in x)
    #    y = tuple(-z for z in y)
    #print(x, y, x[0] / x[1], y[0] / y[1])
    #k_rng1 = sorted([b1 * seg1[0][0] - a1 * seg1[0][1], b1 * seg1[1][0] - a1 * seg1[1][1]])
    #k_rng2 = sorted([b2 * seg2[0][0] - a2 * seg2[0][1], b1 * seg2[1][0] - a1 * seg2[1][1]])
    end_seen = False
    for seg, eqn in ((seg1, eqn1), (seg2, eqn2)):
        a, b, c = eqn
        k_rng = sorted([b * seg[0][0] - a * seg[0][1], b * seg[1][0] - a * seg[1][1]])
        k = b * x - a * y
        if k < k_rng[0] or k > k_rng[1]: return None
        if k == k_rng[0] or k == k_rng[1]:
            if internal_only: return None
            if allow_endpoint_to_endpoint and end_seen: return None
            end_seen = True
        #x1, x2 = sorted([seg[0][0], seg[1][0]])
        #y1, y2 = sorted([seg[0][1], seg[1][1]])
        #print(x1, x2)
        #print(y1, y2)
        #if (x[0] <= x1 * x[1] or x[0] >= x2 * x[1]) and (y[0] <= y1 * y[1] or y[0] >= y2 * y[1]):
        #    #print("x out of range")
        #    return None
    #g1, g2 = gcd(*x), gcd(*y)
    #return (tuple(a // g1 for a in x), tuple(a // g2 for a in y))
    #print("crossing found")
    #if e_ref in {tuple(seg1), tuple(seg2)}: print("intersection found")
    return (x, y)

def BentleyOttmannAlgorithmIntegerEndpoints(
    line_segments: List[Tuple[Tuple[int, int], Tuple[int, int]]],
) -> Dict[Tuple[CustomFraction, CustomFraction], List[Set[Tuple[Tuple[int, int], Tuple[int, int]]]]]:
    
    def gradient(
        p1: Tuple[int, int],
        p2: Tuple[int, int]
    ) -> CustomFraction:
        if p1 == p2: return 0
        diffs = [p2[i] - p1[i] for i in range(len(p1))]
        return CustomFraction(diffs[1], diffs[0])
        #if not diffs[0]:
        #    return (1, 0) if diffs[0] > 0 else -(-1, 0)
        #g = gcd(*diffs)
        #return (diffs[1] // g, diffs[0] // g)
    xings = {}
    
    # Review- add mechanism for counting all crossings for overlapping line segments
    seg_eqns = {}
    for i, seg in enumerate(line_segments):
        eqn = twoDimensionalLineEquationGivenTwoIntegerPoints(*seg)
        seg_eqns.setdefault(eqn, [])
        seg_eqns[eqn].append(i)

    points0 = set()
    for seg in line_segments:
        for p in seg:
            points0.add(p)
    points = sorted(points0)
    points_dict = {p: i for i, p in enumerate(points)}
    n = len(points)

    line_segments2 = []
    collinear_dict = {}
    for eqn, inds in seg_eqns.items():
        if len(inds) == 1:
            line_segments2.append(line_segments[inds[0]])
            continue
        print("found collinear line segments:")
        print([line_segments[i] for i in inds])
        a, b, c = eqn
        line_changes = []
        k_vals = {}
        for i in inds:
            p1, p2 = line_segments[i]
            k1 = b * p1[0] - a * p1[1]
            k2 = b * p2[0] - a * p2[1]
            if k1 > k2:
                k1, k2 = k2, k1
                p1, p2 = p2, p1
            #j1, j2 = points_dict[p1], points_dict[p2]
            k_vals[p1] = k1
            k_vals[p2] = k2
            line_changes.append((k1, 1, p1, i))
            line_changes.append((k2, -1, p2, i))
        line_changes.sort()
        #print(line_changes)
        curr = []
        cnt = 0
        incl_segs = []
        for k, d, p, i in line_changes:
            cnt += d
            if d > 0: incl_segs.append(i)
            if not curr:
                curr.append(p)
            if cnt: continue
            curr.append(p)
            curr_tup = tuple(curr)
            e = tuple(points_dict[p] for p in curr)
            line_segments2.append(curr_tup)
            #print(f"adding {curr_tup}")
            if len(incl_segs) > 1:
                collinear_dict[e] = (eqn, [])
                for i in incl_segs:
                    pairs = sorted((k_vals[p2], points_dict[p2]) for p2 in line_segments[i])
                    collinear_dict[e][1].append((tuple(pair[1] for pair in pairs), tuple(pair[0] for pair in pairs)))
            curr = []
            incl_segs = []
    #print(collinear_dict)
    out_adj = [set() for _ in range(n)]
    in_adj = [set() for _ in range(n)]
    for seg in line_segments2:
        i1, i2 = sorted([points_dict[x] for x in seg])
        out_adj[i1].add(i2)
        in_adj[i2].add(i1)

    #for e in [(4730, 5424), (2960, 5832), (638, 5600), (5240, 5734), (1689, 5768)]:
    #    print(f"edge {e} from {points[e[0]]} to {points[e[1]]}")

    events_heap = [(p, True, i) for i, p in enumerate(points)]
    heapq.heapify(events_heap)
    seen_crossings = {}
    
    def findGreaterXCrossing(
        lower_edge: Tuple[int, int],
        lower_grad: CustomFraction,
        upper_edge: Tuple[int, int],
        upper_grad: CustomFraction
    ) -> Optional[Tuple[CustomFraction, CustomFraction]]:
        if lower_grad < upper_grad: return None
        return twoDimensionalLineSegmentPairCrossing(
            (points[lower_edge[0]], points[lower_edge[1]]),
            (points[upper_edge[0]], points[upper_edge[1]]),
            internal_only=True,
            allow_endpoint_to_endpoint=False,
        )
    
    def addCrossing(
        edge1: Tuple[int, int],
        edge2: Tuple[int, int],
        crossing: Optional[Tuple[CustomFraction, CustomFraction]]
    ) -> None:
        if crossing is None: return
        if crossing not in xings.keys():
            xings[crossing] = set()
            heapq.heappush(events_heap, (crossing, False))
        #if (1410, 1412) in {tuple(edge1), tuple(edge2)}:
        #    print(f"adding crossing between {edge1} and {edge2}")
        xings[crossing].add(edge1)
        xings[crossing].add(edge2)
        return
        """
        #if edge2 < edge1:
        #    edge1, edge2 = edge2, edge1
        seen_crossings.setdefault(edge1[0], {})
        seen_crossings[edge1[0]].setdefault(edge1[1], {})
        seen_crossings[edge1[0]][edge1[1]].setdefault(edge2[0], set())
        if edge2[1] in seen_crossings[edge1[0]][edge1[1]][edge2[0]]: return
        #print(f"new crossing seen: {edge1} to {edge2} at ({crossing[0].numerator / crossing[0].denominator}, {crossing[1].numerator / crossing[1].denominator})")
        seen_crossings[edge1[0]][edge1[1]][edge2[0]].add(edge2[1])
        #if edge1[0] == 374:
        #    print(f"edge1 = {edge1}")
        #elif edge2[0] == 374:
        #    print(f"edge2 = {edge2}")
        heapq.heappush(events_heap, (crossing, False, edge1, edge2))
        return
        """
    
    seg_line = SortedList()
    in_seg_dict = {}

    def updateSurroundingSegments(
        p: Tuple[Union[CustomFraction, int], Union[CustomFraction, int]],
    ) -> None:
        #print("using updateSurroundingSegments()")
        #print(f"p = {p}")
        # From Below to Above
        #print(seg_line)
        #print(in_seg_dict)
        #p_ref = (124, 472)

        #if p == p_ref:
        #    print(f"updating surrounding for p = {p}")
        #    print(f"seg line is length {len(seg_line)}")
        #    print(seg_line)

        rm_segs = []
        add_segs = []
        seg_ref = (p[1], CustomFraction(1, 0))
        
        j0 = seg_line.bisect_left(seg_ref)
        #if p == p_ref:
        #    print(seg_ref[1].numerator, seg_ref[1].denominator)
        #    print(f"j0 = {j0}")
        #print("checking below")
        #print(f"j0 = {j0}")
        #p_curr = p
        for j in reversed(range(j0)):
            #if p == p_ref:
            #    print(j, seg_line[j], seg_ref, seg_line[j] > seg_ref)
            #    print(seg_ref[1].numerator, seg_ref[1].denominator, seg_ref[1].numerator * seg_line[j][1].denominator, seg_ref[1].denominator * seg_line[j][1].numerator)
            seg = seg_line[j]
            grad = seg[1]
            if grad <= 0: break
            elif not grad.denominator:
                p2 = p
                rm_segs.append(seg)
                add_segs.append((p2[1], grad, p2[0], seg[3]))
                continue
            p0 = (seg[2], seg[0])
            if p0[0] == p[0]: break
            #print(f"grad = {grad.numerator / grad.denominator}")
            diff = grad * (p[0] - p0[0])
            #print(f"diff = {diff.numerator / diff.denominator}")
            #print(p0[1], type(p0[1]), diff, type(diff))
            p2 = (p[0], p0[1] + diff)
            #print(f"below segment, edge = ({seg[3]}) p2 y = ({p2[1].numerator / p2[1].denominator})")
            if p2[1] < p[1]: break
            #print("replacing")
            #p_curr = p2
            rm_segs.append(seg)
            add_segs.append((p2[1], grad, p2[0], seg[3]))
        if add_segs:
            largest_add_seg = add_segs[0]
            for j in range(j0, len(seg_line)):
                seg = seg_line[j]
                if seg > largest_add_seg: break
                grad = seg[1]
                p0 = (seg[2], seg[0])
                diff = grad * (p[0] - p0[0])
                p2 = (p[0], p0[1] + diff)
                rm_segs.append(seg)
                add_segs.append((p2[1], grad, p2[0], seg[3]))
                largest_add_seg = add_segs[-1]
            #print(f"add_segs 1 = {add_segs}")
            #print(f"rm_segs 1 = {rm_segs}")
            for seg in rm_segs: seg_line.remove(seg)
            for seg in add_segs:
                seg_line.add(seg)
                grad = seg[1]
                p2 = (seg[2], seg[0])
                edge = seg[3]
                #print(edge)
                in_seg_dict[edge[1]][edge[0]] = (p2, grad)
            return
        # From Above to Below
        j2 = seg_line.bisect_right((p[1], CustomFraction(-1, 0)))
        #print("checking above")
        #print(f"j2 = {j2}")
        #p_curr = p
        for j in range(j2, len(seg_line)):
            #if p == p_ref: print(j)
            seg = seg_line[j]
            grad = seg[1]
            if grad >= 0: break
            elif not grad.denominator:
                p2 = p
                rm_segs.append(seg)
                add_segs.append((p2[1], grad, p2[0], seg[3]))
                continue
            p0 = (seg[2], seg[0])
            if p0[0] == p[0]: break
            diff = grad * (p[0] - p0[0])
            p2 = (p[0], p0[1] + diff)
            #print(f"p2 = ({p2[0].numerator / p2[0].denominator}, {p2[1].numerator / p2[1].denominator})")
            if p2[1] > p[1]: break
            #p_curr = p2
            #print("found")
            rm_segs.append(seg)
            add_segs.append((p2[1], grad, p2[0], seg[3]))
        if add_segs:
            smallest_add_seg = add_segs[0]
            for j in reversed(range(j2)):
                seg = seg_line[j]
                if seg < smallest_add_seg: break
                grad = seg[1]
                p0 = (seg[2], seg[0])
                diff = grad * (p[0] - p0[0])
                p2 = (p[0], p0[1] + diff)
                rm_segs.append(seg)
                add_segs.append((p2[1], grad, p2[0], seg[3]))
                smallest_add_seg = add_segs[-1]
            #print(f"add_segs 2 = {add_segs}")
            #print(f"rm_segs 2 = {rm_segs}")
            for seg in rm_segs: seg_line.remove(seg)
            for seg in add_segs:
                seg_line.add(seg)
                grad = seg[1]
                p2 = (seg[2], seg[0])
                edge = seg[3]
                in_seg_dict[edge[1]][edge[0]] = (p2, grad)
        return
    
    print_rng = (1, -1)# (CustomFraction(140597, 310), CustomFraction(140603, 310))

    n_vertices = 0
    
    while events_heap:
        event = heapq.heappop(events_heap)
        #print(seg_line)
        #print(event)
        p = event[0]
        if not event[1]:
            # Crossing event
            if event[0][0] >= print_rng[0] and event[0][0] <= print_rng[1]:
                print("Crossing")
                print(", ".join([f"{pt[0]} to {pt[1]}" for pt in xings[event[0]]]))
                print((event[0][0].numerator / event[0][0].denominator, event[0][1].numerator / event[0][1].denominator))
                print(seg_line)
                print(event)
            edges = xings.get(event[0], set())
            edge_mx = (CustomFraction(-1, 0),)
            edge_mn = (CustomFraction(1, 0),)
            edges2 = []
            for edge in edges:
                p2, grad2 = in_seg_dict[edge[1]][edge[0]]
                edges2.append((edge, grad2))
                #if (1410, 1412) in edges:
                #    print(edge_mx, (grad2, edge), (grad2, edge) > edge_mx)
                edge_mx = max(edge_mx, (grad2, edge))
                #if (1410, 1412) in edges:
                #    print(edge_mx)
                edge_mn = min(edge_mn, (grad2, edge))
                seg2 = (p2[1], grad2, p2[0], edge)
                seg_line.remove(seg2)
            updateSurroundingSegments(p)
            add_seg_mn = (p[1], edge_mn[0], p[0], edge_mn[1])
            j = seg_line.bisect_left(add_seg_mn) - 1
            if j >= 0:
                seg0 = seg_line[j]
                edge0 = seg0[3]
                grad0 = seg0[1]
                crossing = findGreaterXCrossing(edge0, grad0, edge_mn[1], edge_mn[0])
                addCrossing(edge0, edge_mn[1], crossing)
            add_seg_mx = (p[1], edge_mx[0], p[0], edge_mx[1])
            #if (1410, 1412) in edges:
            #    print(f"edge (1410, 1412):")
            #    print(edges2)
            #    print(edges2[0][1] > edges2[1][1])
            #    print(add_seg_mx)
            j = seg_line.bisect_right(add_seg_mx)
            #print(f"checking above: j = {j}, len(seg_line) = {len(seg_line)}")
            if j < len(seg_line):
                seg3 = seg_line[j]
                edge3 = seg3[3]
                grad3 = seg3[1]
                crossing = findGreaterXCrossing(edge_mx[1], edge_mx[0], edge3, grad3)
                addCrossing(edge_mx[1], edge3, crossing)
            for edge, grad in edges2:
                #print(grad)
                add_seg = (p[1], grad, p[0], edge)
                seg_line.add(add_seg)
                in_seg_dict[edge[1]][edge[0]] = (p, grad)
            continue
            """
            if event[0][0] >= print_rng[0] and event[0][0] <= print_rng[1]:
                print("Crossing")
                print(f"{points[event[2][0]]} to {points[event[2][1]]} and {points[event[3][0]]} to {points[event[3][1]]}")
                print((event[0][0].numerator / event[0][0].denominator, event[0][1].numerator / event[0][1].denominator))
                print(seg_line)
                print(event)
            p = event[0]
            edge1, edge2 = event[2:4]
            #xings.setdefault(event[2], set())
            #xings[event[2]].add(event[3])
            #xings.setdefault(event[3], set())
            #xings[event[3]].add(event[2])
            #print(event)
            #print((points[edge1[0]], points[edge1[1]]), (points[edge2[0]], points[edge2[1]]))
            #print(event[0][0], event[0][0])
            res.setdefault(p, set())
            res[p].add(edge1)
            res[p].add(edge2)
            #res.add(p)
            #res2 += 1
            #if edge1[0] == 374:
            #    print(edge1)
            #    #print(seg_line)
            #    print(in_seg_dict.get(edge1[1], {}))
            #    print(in_seg_dict.get(edge1[0], {}))

            p1, grad1 = in_seg_dict[edge1[1]][edge1[0]]
            seg1 = (p1[1], grad1, p1[0], edge1)
            seg_line.remove(seg1)
            p2, grad2 = in_seg_dict[edge2[1]][edge2[0]]
            seg2 = (p2[1], grad2, p2[0], edge2)
            seg_line.remove(seg2)
            updateSurroundingSegments(p)#, seg1)
            add_seg1 = (p[1], grad2, p[0], edge2)
            add_seg2 = (p[1], grad1, p[0], edge1)
            j = seg_line.bisect_left(add_seg1) - 1
            #print(f"checking below: j = {j}")
            if j >= 0:
                seg0 = seg_line[j]
                edge0 = seg0[3]
                grad0 = seg0[1]
                crossing = findGreaterXCrossing(edge0, grad0, edge2, grad2)
                addCrossing(edge0, edge2, crossing)
            j = seg_line.bisect_right(add_seg2)
            #print(f"checking above: j = {j}, len(seg_line) = {len(seg_line)}")
            if j < len(seg_line):
                seg3 = seg_line[j]
                edge3 = seg3[3]
                grad3 = seg3[1]
                crossing = findGreaterXCrossing(edge1, grad1, edge3, grad3)
                addCrossing(edge1, edge3, crossing)
            seg_line.add(add_seg1)
            in_seg_dict[edge2[1]][edge2[0]] = (p, grad2)
            seg_line.add(add_seg2)
            in_seg_dict[edge1[1]][edge1[0]] = (p, grad1)
            continue
            """
        # Vertex event
        if event[0][0] >= print_rng[0] and event[0][0] <= print_rng[1]:
            print("Vertex")
            print((event[0][0].numerator / event[0][0].denominator, event[0][1].numerator / event[0][1].denominator))
            print(event)
            print(seg_line)
        n_vertices += 1
        if not n_vertices % 50:
            print(f"{n_vertices} vertices processed, x = {p[0]}, number of unique crossing points found = {len(xings)}")
        i = event[2]
        rm_seg = None
        if i in in_seg_dict.keys():
            in_segs = in_seg_dict.pop(i)
            for i0, (p0, grad) in in_segs.items():
                rm_seg = (p0[1], grad, p0[0], (i0, i))
                seg_line.remove(rm_seg)
        if out_adj[i]:
            updateSurroundingSegments(p)#, (p[1], CustomFraction(0, 1), p[0]))
        if not out_adj[i]:
            if rm_seg is None: continue
            j2 = seg_line.bisect_right(rm_seg)
            j1 = j2 - 1
            if j1 < 0 or j2 >= len(seg_line): continue
            seg1 = seg_line[j1]
            seg2 = seg_line[j2]
            edge1 = seg1[3]
            grad1 = seg1[1]
            edge2 = seg2[3]
            grad2 = seg2[1]
            crossing = findGreaterXCrossing(edge1, grad1, edge2, grad2)
            addCrossing(edge1, edge2, crossing)
            #i1, i2 = seg1[3:5]
            #i3, i4 = seg2[3:5]
            #if i3 < i1:
            #    (i1, i2), (i3, i4) = (i3, i4), (i1, i2)
            #seen_crossings.setdefault(i1, {})
            #seen_crossings[i1].setdefault(i2, {})
            #seen_crossings[i1][i2].setdefault(i3, set())
            #if i4 in seen_crossings[i1][i2][i3]: continue
            #seen_crossings[i1][i2][i3].add(i4)
            #heapq.heappush(events_heap, (crossing, False, i1, i2, i3, i4))
            continue
        add_segs = []
        for i2 in out_adj[i]:
            p2 = points[i2]
            grad = gradient(p, p2)
            add_segs.append((grad, p2, i2))
        #if i == 374:
        #    print(add_segs)
        add_segs.sort()
        j = seg_line.bisect_left((p[1], add_segs[0][0])) - 1
        if p == (453, 51):
            print(f"{p} add segments: {add_segs}")
        if j >= 0:
            edge1 = (i, add_segs[0][2])
            grad1 = add_segs[0][0]
            seg0 = seg_line[j]
            edge0 = seg0[3]
            grad0 = seg0[1]
            if p == (453, 51):
                print(f"below: {j}, {edge0}, {grad0}")
            crossing = findGreaterXCrossing(edge0, grad0, edge1, grad1)
            addCrossing(edge0, edge1, crossing)
            #if crossing is not None:
            #    seen_crossings.setdefault(i, {})
            #    seen_crossings[i].setdefault(i2, {})
            #    seen_crossings[i][i2].setdefault(i3, set())
            #    seen_crossings[i][i2][i3].add(i4)
            #    heapq.heappush(events_heap, (crossing, False, i, i2, i3, i4))
        j = seg_line.bisect_right((p[1], add_segs[-1][0]))
        if j < len(seg_line):
            edge1 = (i, add_segs[-1][2])
            grad1 = add_segs[-1][0]
            seg2 = seg_line[j]
            edge2 = seg2[3]
            grad2 = seg2[1]
            if p == (453, 51):
                print(f"above: {j}, {edge2}, {grad2}")
            crossing = findGreaterXCrossing(edge1, grad1, edge2, grad2)
            addCrossing(edge1, edge2, crossing)
            #if crossing is not None:
            #    heapq.heappush(events_heap, (crossing, False, i, i2, i3, i4))
        for grad, p2, i2 in add_segs:
            #if i2 == 4802 and i == 374:
            #    print(i, i2)
            seg_line.add((p[1], grad, p[0], (i, i2)))
            in_seg_dict.setdefault(i2, {})
            in_seg_dict[i2][i] = (p, grad)
        #for i2 in out_adj:
        #    out_segs = 
    
    xings_final = {}
    for intersect, e_set in xings.items():
        xings_final[intersect] = []
        for e in e_set:
            e_tup = tuple(e)
            if e_tup not in collinear_dict.keys():
                seg = tuple(points[i] for i in e)
                xings_final[intersect].append({seg})
                continue
            xings_final[intersect].append(set())
            eqn, e_lst = collinear_dict[e_tup]
            a, b, c = eqn
            k = b * intersect[0] - a * intersect[1]
            for e2, k_vals in e_lst:
                if k <= k_vals[0] or k >= k_vals[1]: continue
                seg = tuple(points[i] for i in e2)
                xings_final[intersect][-1].add(seg)
            #print(intersect, k, e, e_lst, xings_final[intersect][-1])
            if not xings_final[intersect][-1]: xings_final[intersect].pop() # should not happen

    return xings_final

if __name__ == "__main__":

    res = grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=False)
    print(f"\ngrahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=False) = {res}")

    res = grahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=True)
    print(f"\ngrahamScan([(1, 1), (2, 2), (2, 0), (2, 4), (3, 3), (4, 2)], include_border_points=True) = {res}")

    res = outerTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]])
    print(f"\nouterTrees([[1, 1], [2, 2], [2, 0], [2, 4], [3, 3], [4, 2]]) = {res}")

    res = outerTrees([[1, 2], [2, 2], [4, 2]])
    print(f"\nouterTrees([[1, 2], [2, 2], [4, 2]]) = {res}")