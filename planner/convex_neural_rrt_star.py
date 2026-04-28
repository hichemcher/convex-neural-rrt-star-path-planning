"""
convex_neural_rrt_star.py
================================
Implementation of the Hybrid RRT* with Convex-Hull-Guided Sampling algorithm,
as proposed in the paper:

    "Neural-Guided RRT* with Adaptive Convex Hull Sampling for Robot Path Planning"

Algorithm: convex_neural_rrt_star
Author: Hichem Cheriet
University: USTO-MB
"""

import math
import random
from collections import namedtuple

import numpy as np
import numba
from scipy.spatial import ConvexHull
from matplotlib.path import Path


# ─── Node Definition ───────────────────────────────────────────────────────────

Node = namedtuple("Node", ["row", "col", "parent", "cost"])


# ─── Collision Checking ────────────────────────────────────────────────────────

@numba.njit
def _bresenham_line(x0, y0, x1, y1):
    """Numba-accelerated Bresenham line generation."""
    points = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return points


@numba.njit
def is_collision_free_nb(grid, x0, y0, x1, y1):
    """
    Check collision-free path between (x0,y0) and (x1,y1) on grid.
    Cells with value 0, 2 (start), or 3 (goal) are treated as free.

    Parameters
    ----------
    grid : 2D numpy array  — obstacle map (1 = obstacle)
    x0, y0 : int           — start column and row
    x1, y1 : int           — goal column and row

    Returns
    -------
    bool : True if path is collision-free
    """
    line = _bresenham_line(int(round(x0)), int(round(y0)),
                           int(round(x1)), int(round(y1)))
    h, w = grid.shape
    for x, y in line:
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        if grid[y, x] not in (0, 2, 3):
            return False
    return True


# ─── Sampling Utilities ────────────────────────────────────────────────────────

def sample_free(grid, tries=200):
    """Uniformly sample a collision-free cell from the grid."""
    h, w = grid.shape
    for _ in range(tries):
        col = random.uniform(0, w - 1)
        row = random.uniform(0, h - 1)
        if grid[int(round(row)), int(round(col))] == 0:
            return int(round(row)), int(round(col))
    return np.random.randint(0, h), np.random.randint(0, w)


# ─── Tree Utilities ────────────────────────────────────────────────────────────

def nearest_node(nodes, row, col):
    """Find the node in the tree closest to (row, col)."""
    best, best_d = None, float("inf")
    for n in nodes:
        d = math.hypot(n.row - row, n.col - col)
        if d < best_d:
            best, best_d = n, d
    return best


def steer_toward_limited(nearest, target, step_size):
    """
    Steer from nearest toward target, capped at step_size distance.

    Parameters
    ----------
    nearest   : Node   — current tree node
    target    : tuple  — (row, col) target point
    step_size : float  — maximum step length

    Returns
    -------
    Node : new candidate node
    """
    tr, tc = target
    dr, dc = tr - nearest.row, tc - nearest.col
    dist = math.hypot(dr, dc)

    if dist > step_size:
        scale = step_size / dist
        tr = nearest.row + dr * scale
        tc = nearest.col + dc * scale

    tr, tc = int(round(tr)), int(round(tc))
    cost = nearest.cost + min(dist, step_size)
    return Node(tr, tc, nearest, cost)


def build_path_from_goal(goal_node):
    """Reconstruct path from goal node back to root."""
    path = []
    cur = goal_node
    while cur is not None:
        path.append((cur.row, cur.col))
        cur = cur.parent
    path.reverse()
    return path


# ─── Main Algorithm ────────────────────────────────────────────────────────────

def convex_neural_rrt_star(
    grid,
    start,
    goal,
    conv_pts_all,
    pred_conv,
    conv_pts,
    alpha_in=0.5,
    alpha_out=0.2,
    max_iter=500,
    step_size=10,
    N=400,
    epsilon=1e-3,
):
    """
    Hybrid RRT* with Convex-Hull-Guided Adaptive Sampling.

    Combines neural network predicted path points with obstacle convex corner
    points, separated by a convex hull boundary around the predicted path.
    Points inside the hull are sampled with probability alpha_in, while
    outside points (providing exploration) are sampled with alpha_out.
    Early stopping is applied when path cost converges.

    Parameters
    ----------
    grid         : np.ndarray (H, W)   — binary obstacle map (1=obstacle, 0=free)
    start        : tuple (row, col)    — start position
    goal         : tuple (row, col)    — goal position
    conv_pts_all : list of (row, col)  — all detected convex corner points
    pred_conv    : list of (row, col)  — neural network predicted path points
    conv_pts     : list of (row, col)  — convex corner points (subset of conv_pts_all)
    alpha_in     : float               — sampling probability for inside-hull corners
    alpha_out    : float               — sampling probability for outside-hull corners
    max_iter     : int                 — maximum RRT* iterations
    step_size    : float               — maximum steer step size (pixels)
    N            : int                 — window size for early stopping check
    epsilon      : float               — cost convergence threshold for early stopping

    Returns
    -------
    path            : list of (row, col) — found path, empty if none found
    best_cost_history : list of float  — goal cost per iteration
    nodes           : list of Node     — all tree nodes
    pred_points     : list of (row, col) — sampled predicted path points (for analysis)
    sample_points   : list of (row, col) — sampled inside-hull corner points (for analysis)
    sampled_out_pts : list of (row, col) — sampled outside-hull corner points (for analysis)
    """

    # ── Shortcut: direct line of sight ──────────────────────────────────────
    if is_collision_free_nb(grid, start[1], start[0], goal[1], goal[0]):
        return [list(start), list(goal)], [], [0], [], [], []

    # ── Initialise tree ──────────────────────────────────────────────────────
    nodes = [Node(start[0], start[1], None, 0.0)]
    goal_node = Node(goal[0], goal[1], None, float("inf"))

    conv_pts = np.array(conv_pts)
    pred_conv = np.array(pred_conv)
    pred_points, sample_points, sampled_out_pts = [], [], []

    # ── Exclude predicted points from corner set ─────────────────────────────
    pred_set = set(map(tuple, pred_conv))
    conv_pts = [pt for pt in conv_pts if tuple(pt) not in pred_set]

    # ── Build convex hull around predicted path + start/goal ─────────────────
    all_points = np.vstack([pred_conv, np.array([start, goal])])
    if len(all_points) < 3:
        hull_vertices = all_points
    else:
        hull = ConvexHull(all_points)
        hull_vertices = all_points[hull.vertices]

    hull_path = Path(hull_vertices)

    # ── Partition corner points by hull membership ───────────────────────────
    inside_pts = [pt for pt in conv_pts if hull_path.contains_point(pt)]
    outside_pts = [pt for pt in conv_pts_all if not hull_path.contains_point(pt)]

    # ── Main RRT* loop ───────────────────────────────────────────────────────
    best_cost_history = []
    frames = []

    for i in range(max_iter):

        # 1. Sample target point
        r = random.random()

        if r < 0.1:
            # Goal bias
            target = goal

        elif r < 0.1 + alpha_out:
            # Explore outside hull
            if outside_pts:
                target = random.choice(outside_pts)
                sampled_out_pts.append(target)
            else:
                target = sample_free(grid)

        else:
            # Sample inside hull
            r2 = random.random()
            if r2 < alpha_in and len(pred_conv) > 0:
                target = tuple(random.choice(pred_conv))
                pred_points.append(target)
            elif inside_pts:
                target = tuple(random.choice(inside_pts))
                sample_points.append(target)
            else:
                target = sample_free(grid)

        # 2. Find nearest node in tree
        nearest = nearest_node(nodes, target[0], target[1])

        # 3. Steer toward target
        new_node = steer_toward_limited(nearest, target, step_size)

        # 4. Collision check
        if not is_collision_free_nb(
            grid, nearest.col, nearest.row, new_node.col, new_node.row
        ):
            continue

        # 5. Choose best parent (RRT* rewiring)
        best_parent = nearest
        best_cost = nearest.cost + math.hypot(
            nearest.row - new_node.row, nearest.col - new_node.col
        )
        for n in nodes:
            if is_collision_free_nb(grid, n.col, n.row, new_node.col, new_node.row):
                c = n.cost + math.hypot(n.row - new_node.row, n.col - new_node.col)
                if c < best_cost:
                    best_parent, best_cost = n, c

        new_node = Node(new_node.row, new_node.col, best_parent, best_cost)
        nodes.append(new_node)

        # 6. Rewire existing nodes through new_node if cheaper
        for i, n in enumerate(nodes):
            if is_collision_free_nb(grid, new_node.col, new_node.row, n.col, n.row):
                new_cost = new_node.cost + math.hypot(
                    new_node.row - n.row, new_node.col - n.col
                )
                if new_cost < n.cost:
                    nodes[i] = Node(n.row, n.col, new_node, new_cost)

        # 7. Try to connect new_node to goal
        if is_collision_free_nb(
            grid, new_node.col, new_node.row, goal_node.col, goal_node.row
        ):
            goal_cost = new_node.cost + math.hypot(
                new_node.row - goal_node.row, new_node.col - goal_node.col
            )
            if goal_cost < goal_node.cost:
                goal_node = Node(goal_node.row, goal_node.col, new_node, goal_cost)

        # 8. Record cost and check early stopping
        best_cost_history.append(goal_node.cost if goal_node.parent else np.nan)

        if len(best_cost_history) > N:
            window = best_cost_history[-N:]
            valid = [v for v in window if not np.isnan(v)]
            if valid and (max(valid) - min(valid)) < epsilon:
                break

    # ── Build and return path ────────────────────────────────────────────────
    path = build_path_from_goal(goal_node) if goal_node.parent else []
    return path, best_cost_history, nodes, pred_points, sample_points, sampled_out_pts
