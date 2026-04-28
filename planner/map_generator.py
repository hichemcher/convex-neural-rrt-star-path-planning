"""
map_generator.py
================
Random occupancy map generation utilities used for training and testing
the convex_neural_rrt_star planner.

Generates 224×224 binary grids populated with random convex (and optionally
concave) polygon obstacles, along with valid start/goal pairs that are
guaranteed to be far apart and connected.
"""

import numpy as np
import cv2
from scipy.ndimage import binary_dilation
from skimage.morphology import disk


# ─── Low-level geometry helpers ───────────────────────────────────────────────

def supercover_line(x0, y0, x1, y1):
    """
    Supercover (thick) Bresenham line — returns every cell the segment passes through.

    Parameters
    ----------
    x0, y0 : int — start column, row
    x1, y1 : int — end column, row

    Returns
    -------
    list of (row, col) tuples
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2
    cells = []
    for _ in range(n):
        cells.append((y, x))
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return cells


def detect_convex_corners(B):
    """
    Detect convex corner cells on a binary obstacle grid.
    A free cell adjacent to exactly one obstacle neighbour (8-connected) is a corner.

    Parameters
    ----------
    B : np.ndarray (H, W) — binary obstacle map (1 = obstacle)

    Returns
    -------
    np.ndarray (H, W) uint8 — corner mask (1 = convex corner)
    """
    n = (
        np.roll(B, 1, 0) + np.roll(B, -1, 0) +
        np.roll(B, 1, 1) + np.roll(B, -1, 1) +
        np.roll(np.roll(B, 1, 0), 1, 1) + np.roll(np.roll(B, 1, 0), -1, 1) +
        np.roll(np.roll(B, -1, 0), 1, 1) + np.roll(np.roll(B, -1, 0), -1, 1)
    )
    mask = (B == 0) & (n == 1)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    return mask.astype(np.uint8)


# ─── Shape generation ─────────────────────────────────────────────────────────

def generate_random_polygon_shape(size, shape_type='convex', max_size=40):
    """
    Generate a small binary mask containing a random polygon obstacle.

    Parameters
    ----------
    size       : tuple (h, w)          — bounding box of the mask
    shape_type : 'convex' or 'concave' — polygon type
    max_size   : int                   — maximum polygon radius

    Returns
    -------
    np.ndarray (h, w) uint8 — binary polygon mask
    """
    h, w = size
    mask = np.zeros((h, w), dtype=np.uint8)
    num_vertices = np.random.randint(5, 9)
    angles = np.linspace(0, 2 * np.pi, num_vertices, endpoint=False)
    angles += np.random.uniform(-0.2, 0.2, num_vertices)
    radii = np.random.uniform(0.6, 1.0, num_vertices) * min(h, w) / 2
    cx, cy = max_size // 2, max_size // 2
    vertices = np.array([
        [cx + r * np.cos(a), cy + r * np.sin(a)]
        for a, r in zip(angles, radii)
    ], dtype=np.int32)
    if shape_type == 'concave':
        idx = np.random.randint(0, num_vertices)
        vertices[idx] = (vertices[idx] * 0.3).astype(np.int32)
    cv2.fillPoly(mask, [vertices], color=1)
    return mask


def place_shape(grid, shape, max_attempts=100):
    """
    Place a polygon shape onto the grid at a random non-overlapping position.

    Parameters
    ----------
    grid         : np.ndarray (H, W) — obstacle grid (modified in-place)
    shape        : np.ndarray        — binary polygon mask
    max_attempts : int               — number of random placement tries

    Returns
    -------
    success : bool
    center  : tuple (row, col) or None
    """
    h, w = grid.shape
    sh, sw = shape.shape
    for _ in range(max_attempts):
        y = np.random.randint(0, h - sh)
        x = np.random.randint(0, w - sw)
        patch = grid[y:y + sh, x:x + sw]
        if patch.shape == shape.shape and not np.any(patch & shape):
            grid[y:y + sh, x:x + sw] |= shape
            return True, (y + sh // 2, x + sw // 2)
    return False, None


# ─── Path utilities ───────────────────────────────────────────────────────────

def densify_path(path):
    """Fill in all grid cells between consecutive waypoints using supercover lines."""
    if not path or len(path) < 2:
        return path
    dense = []
    for i in range(len(path) - 1):
        (r0, c0), (r1, c1) = path[i], path[i + 1]
        dense.extend(supercover_line(c0, r0, c1, r1))
    return list(set(dense))


def dilate_path_from_list(path, grid_shape, dilation_radius=4):
    """Convert a list of path cells into a dilated binary mask."""
    mask = np.zeros(grid_shape, dtype=bool)
    for r, c in path:
        mask[r, c] = True
    dilated = binary_dilation(
        mask,
        structure=np.ones((2 * dilation_radius + 1, 2 * dilation_radius + 1))
    )
    return dilated.astype(np.uint8)


# ─── Line-of-sight (used by LTA*) ─────────────────────────────────────────────

def _has_line_of_sight(grid, p0, p1):
    height, width = grid.shape
    for y, x in supercover_line(p0[1], p0[0], p1[1], p1[0]):
        if y < 0 or x < 0 or y >= height or x >= width:
            return False
        if grid[y, x] == 1:
            return False
    return True


def _calculate_local_tangents(grid, corner_mask, s):
    """Return a mask of corners visible from point s."""
    from planner.convex_neural_rrt_star import is_collision_free_nb
    local_mask = np.zeros_like(grid, dtype=np.uint8)
    for y, x in np.argwhere(corner_mask > 0):
        if is_collision_free_nb(grid, s[1], s[0], x, y):
            local_mask[y, x] = 1
    return local_mask


# ─── LTA* (Local Tangent A*) path planner ────────────────────────────────────

def lta_star_path(grid, convex_corners_mask, start, goal):
    """
    Local Tangent A* — finds a near-shortest path through convex corners.
    Used to generate ground-truth paths for training the UNet.

    Parameters
    ----------
    grid                : np.ndarray (H, W) — obstacle map
    convex_corners_mask : np.ndarray (H, W) — corner mask from detect_convex_corners
    start               : tuple (row, col)
    goal                : tuple (row, col)

    Returns
    -------
    list of (row, col) waypoints, or None if no path found
    """
    ClosedList = set()
    LocalList = []
    Parent = {}
    G = {}
    F = {}

    X_current = start
    G[start] = 0
    F[start] = np.linalg.norm(np.array(start) - np.array(goal))

    tangent_mask = _calculate_local_tangents(grid, convex_corners_mask, X_current)
    for tp in [tuple(c[:2]) for c in np.argwhere(tangent_mask > 0.5)]:
        LocalList.append(tp)
        Parent[tp] = X_current
        G[tp] = G[X_current] + np.linalg.norm(np.array(tp) - np.array(X_current))
        F[tp] = G[tp] + np.linalg.norm(np.array(tp) - np.array(goal))

    while LocalList:
        candidates = [x for x in LocalList if x not in ClosedList]
        if not candidates:
            return None

        X_best = goal if goal in candidates else min(candidates, key=lambda x: F.get(x, np.inf))
        LocalList.remove(X_best)
        ClosedList.add(X_best)

        if X_best == goal:
            path, node = [], goal
            while node != start:
                path.append(node)
                node = Parent[node]
            path.append(start)
            path.reverse()
            return path

        X_current = X_best
        tangent_mask = _calculate_local_tangents(grid, convex_corners_mask, X_current)
        for tp in [tuple(c[:2]) for c in np.argwhere(tangent_mask > 0.5)]:
            if tp not in ClosedList and tp not in LocalList:
                LocalList.append(tp)
                Parent[tp] = X_current
                G[tp] = G[X_current] + np.linalg.norm(np.array(tp) - np.array(X_current))
                F[tp] = G[tp] + np.linalg.norm(np.array(tp) - np.array(goal))

    return None


# ─── Main map generator ───────────────────────────────────────────────────────

def generate_map(
    size=224,
    num_shapes=None,
    concave_ratio=0.05,
    max_shape_size=None,
):
    """
    Generate a random 224×224 occupancy grid with polygon obstacles.

    Parameters
    ----------
    size          : int   — grid size (square)
    num_shapes    : int   — number of obstacles (default: random 75–80)
    concave_ratio : float — fraction of obstacles that are concave (default: 5%)
    max_shape_size: int   — max polygon bounding box size (default: random 55–56)

    Returns
    -------
    grid   : np.ndarray (H, W) uint8 — obstacle map (1 = obstacle, 0 = free)
    corners: np.ndarray (H, W) uint8 — convex corner mask
    """
    if num_shapes is None:
        num_shapes = np.random.randint(75, 80)
    if max_shape_size is None:
        max_shape_size = np.random.randint(55, 56)

    grid = np.zeros((size, size), dtype=np.uint8)
    for _ in range(num_shapes):
        shape_type = 'concave' if np.random.rand() < concave_ratio else 'convex'
        shape = generate_random_polygon_shape(
            (max_shape_size, max_shape_size), shape_type, max_shape_size
        )
        place_shape(grid, shape)

    corners = detect_convex_corners(grid)
    return grid, corners


def generate_start_goal(grid, min_dist_ratio=2/3):
    """
    Sample a valid (start, goal) pair that are far apart on the free map.

    Parameters
    ----------
    grid           : np.ndarray (H, W) — obstacle map
    min_dist_ratio : float             — minimum distance as fraction of map diagonal

    Returns
    -------
    start : tuple (row, col)
    goal  : tuple (row, col)
    """
    size = grid.shape[0]
    min_dist = min_dist_ratio * np.hypot(size, size)
    free_idx = np.column_stack(np.where(grid == 0))

    while True:
        s, g = free_idx[np.random.randint(len(free_idx), size=2)]
        if np.linalg.norm(s - g) >= min_dist:
            return tuple(s), tuple(g)
