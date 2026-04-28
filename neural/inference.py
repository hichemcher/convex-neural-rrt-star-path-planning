"""
inference.py
============
Inference utilities: encode grids for the UNet, run predictions,
extract convex corner points from the predicted path mask.

Typical pipeline
----------------
    grid[start] = 2  ;  grid[goal] = 3
    tensor      = grid_to_tensor(grid)               # (1,3,224,224)
    conv_mask   = detect_convex_corners(grid == 1)
    pred_conv   = get_predicted_convex_points(model, tensor, conv_mask)
    conv_pts    = [tuple(p) for p in np.argwhere(conv_mask)]
    path, *_    = hybrid_rrt_convex_alpha_out(grid, start, goal,
                      conv_pts_all=conv_pts, pred_conv=pred_conv, conv_pts=conv_pts)
"""

import numpy as np
import torch
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Convex corner detection constants (tune alpha to tighten/loosen polygon approx)
ALPHA   = 0.001
MIN_EPS = 2.0
MAX_EPS = 20.0


# ─── Grid encoding ────────────────────────────────────────────────────────────

def grid_to_tensor(grid, device: str = DEVICE) -> torch.Tensor:
    """
    Encode a labelled occupancy grid as an RGB tensor for the UNet.

    Colour convention
    -----------------
    - Free space (0) → white  [1, 1, 1]
    - Obstacle   (1) → black  [0, 0, 0]
    - Start      (2) → red    [1, 0, 0]
    - Goal       (3) → green  [0, 1, 0]

    Parameters
    ----------
    grid   : np.ndarray (H, W) int  — labelled grid
    device : str                    — torch device

    Returns
    -------
    torch.Tensor (1, 3, H, W) float32
    """
    H, W = grid.shape
    rgb = np.ones((H, W, 3), dtype=np.float32)
    rgb[grid == 1] = [0.0, 0.0, 0.0]
    rgb[grid == 2] = [1.0, 0.0, 0.0]
    rgb[grid == 3] = [0.0, 1.0, 0.0]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)  # (1,3,H,W)
    return tensor.to(device)


def tensor_to_binary_grid(rgb: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 float RGB array back to a binary obstacle map."""
    grid = np.zeros(rgb.shape[:2], dtype=np.uint8)
    grid[(rgb[..., 0] == 0) & (rgb[..., 1] == 0) & (rgb[..., 2] == 0)] = 1
    return grid


# ─── Corner detection ─────────────────────────────────────────────────────────

def detect_convex_corners(B: np.ndarray) -> np.ndarray:
    """
    Pixel-level convex corner detector.
    A free cell with exactly one obstacle neighbour (8-connected) is a corner.

    Parameters
    ----------
    B : np.ndarray (H, W)  — binary obstacle map (1 = obstacle)

    Returns
    -------
    np.ndarray (H, W) uint8  — corner mask
    """
    n = (
        np.roll(B, 1, 0) + np.roll(B, -1, 0) +
        np.roll(B, 1, 1) + np.roll(B, -1, 1) +
        np.roll(np.roll(B, 1, 0),  1, 1) + np.roll(np.roll(B, 1, 0), -1, 1) +
        np.roll(np.roll(B, -1, 0), 1, 1) + np.roll(np.roll(B, -1, 0), -1, 1)
    )
    mask = (B == 0) & (n == 1)
    mask[[0, -1], :] = False
    mask[:, [0, -1]] = False
    return mask.astype(np.uint8)


def find_convex_corners(binary_img: np.ndarray,
                        alpha: float = ALPHA,
                        min_eps: float = MIN_EPS,
                        max_eps: float = MAX_EPS):
    """
    Polygon-approximation convex corner detector (contour-based).
    Dilates obstacles slightly, finds contours, approximates each as a
    polygon, and returns vertices that are convex.

    Parameters
    ----------
    binary_img : np.ndarray (H, W) float/uint8  — obstacle map
    alpha      : float  — polygon approximation tightness (smaller = tighter)
    min_eps    : float  — minimum epsilon for cv2.approxPolyDP
    max_eps    : float  — maximum epsilon

    Returns
    -------
    corners : list of (col, row) int tuples
    overlay : np.ndarray (H, W, 3) uint8  — visualisation image
    """
    binary_u8 = (binary_img * 255).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(binary_u8, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = cv2.cvtColor(binary_u8, cv2.COLOR_GRAY2BGR)
    corners = []

    for cnt in contours:
        arc_len = cv2.arcLength(cnt, True)
        epsilon = float(np.clip(alpha * arc_len, min_eps, max_eps))
        approx  = cv2.approxPolyDP(cnt, epsilon, True).reshape(-1, 2)
        is_ccw  = _is_convex_polygon(approx)
        cv2.polylines(overlay, [approx], True, (255, 0, 255), 2)
        for idx, (x, y) in enumerate(approx):
            if _is_convex_corner(approx, idx, is_ccw):
                cv2.circle(overlay, (x, y), 4, (0, 255, 0), -1)
                corners.append((int(x), int(y)))

    return corners, overlay


def _is_convex_polygon(pts: np.ndarray) -> bool:
    pts = pts.reshape(-1, 2)
    total = sum(
        (pts[(i + 1) % len(pts)][0] - pts[i][0]) * (pts[(i + 1) % len(pts)][1] + pts[i][1])
        for i in range(len(pts))
    )
    return total < 0


def _is_convex_corner(polygon: np.ndarray, idx: int, is_ccw: bool) -> bool:
    n    = len(polygon)
    prev = polygon[(idx - 1) % n]
    curr = polygon[idx]
    nxt  = polygon[(idx + 1) % n]
    v1   = curr - prev
    v2   = nxt  - curr
    cross = v1[0] * v2[1] - v1[1] * v2[0]
    return cross > 0 if is_ccw else cross < 0


# ─── Prediction ───────────────────────────────────────────────────────────────

def get_predicted_convex_points(model,
                                input_tensor: torch.Tensor,
                                convex_mask: np.ndarray,
                                threshold: float = 0.5,
                                radius: int = 4) -> np.ndarray:
    """
    Run the UNet and return convex corner points that lie on the predicted path.

    Parameters
    ----------
    model        : UNet       — trained model (eval mode)
    input_tensor : Tensor     — (1, 3, H, W) from grid_to_tensor()
    convex_mask  : np.ndarray — (H, W) corner mask from detect_convex_corners()
    threshold    : float      — path probability threshold
    radius       : int        — neighbourhood expansion radius around each corner

    Returns
    -------
    np.ndarray (N, 2) — predicted convex path points as (row, col)
    """
    model.eval()
    with torch.no_grad():
        logits    = model(input_tensor)
        probs     = torch.softmax(logits, dim=1)
        path_prob = probs[0, 1]
        path_mask = (path_prob > threshold).cpu().numpy().astype(np.uint8)

    combined = path_mask * convex_mask.astype(np.uint8)
    conv_pts = np.column_stack(np.where(combined > 0)) if combined.any() else np.empty((0, 2), dtype=int)

    if radius > 0 and len(conv_pts) > 0:
        expanded = np.zeros_like(convex_mask, dtype=np.uint8)
        for r, c in conv_pts:
            cv2.circle(expanded, (c, r), radius, 1, -1)
        expanded = expanded * convex_mask.astype(np.uint8)
        conv_pts = np.column_stack(np.where(expanded > 0))

    return conv_pts


def get_predicted_path_points(model,
                               input_tensor: torch.Tensor,
                               threshold: float = 0.5) -> np.ndarray:
    """
    Return ALL cells predicted as path (not restricted to convex corners).

    Parameters
    ----------
    model        : UNet
    input_tensor : Tensor (1, 3, H, W)
    threshold    : float

    Returns
    -------
    np.ndarray (N, 2)  — path points as (row, col)
    """
    model.eval()
    with torch.no_grad():
        logits    = model(input_tensor)
        probs     = torch.softmax(logits, dim=1)
        path_prob = probs[0, 1]
        path_mask = (path_prob > threshold).cpu().numpy()
    return np.column_stack(np.where(path_mask)) if path_mask.any() else np.empty((0, 2), dtype=int)
