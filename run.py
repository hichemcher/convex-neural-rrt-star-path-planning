"""
run.py
======
End-to-end demo:
  1. Generate a random 224×224 map
  2. Load the trained UNet
  3. Run inference to get predicted path points
  4. Run convex_neural_rrt_star
  5. Visualise and save the result

Usage
-----
    python run.py --weights best_path_mode_multipoint_224.pth
    python run.py --weights best_path_mode_multipoint_224.pth --seed 42 --max_iter 1500
"""

import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from planner.map_generator import generate_map, generate_start_goal
from neural.model import load_model, DEVICE
from neural.inference import grid_to_tensor, detect_convex_corners, get_predicted_convex_points
from planner.convex_neural_rrt_star import convex_neural_rrt_star


# ─── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Hybrid RRT* with neural guidance demo")
    p.add_argument("--weights",   default="best_path_mode_multipoint_224.pth",
                   help="Path to trained UNet weights (.pth)")
    p.add_argument("--seed",      type=int, default=0,   help="Random seed")
    p.add_argument("--num_shapes",type=int, default=75,  help="Number of obstacles on the map")
    p.add_argument("--max_iter",  type=int, default=1500,help="Max RRT* iterations")
    p.add_argument("--alpha_in",  type=float, default=0.5)
    p.add_argument("--alpha_out", type=float, default=0.2)
    p.add_argument("--step_size", type=float, default=10.0)
    p.add_argument("--output",    default="result.png",  help="Output image path")
    return p.parse_args()


# ─── Visualisation ────────────────────────────────────────────────────────────

def visualise(grid, start, goal, path, nodes,
              pred_conv, conv_pts, cost_history, output_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0f1117")
    for ax in axes:
        ax.set_facecolor("#0f1117")
        for s in ax.spines.values():
            s.set_edgecolor("#333")

    # ── Panel 1: map + NN prediction ───────────────────────────────────────
    ax = axes[0]
    display = np.ones((*grid.shape, 3))
    display[grid == 1] = [0.15, 0.15, 0.15]
    ax.imshow(display, origin="upper")

    if len(pred_conv) > 0:
        pc = np.array(pred_conv)
        ax.scatter(pc[:, 1], pc[:, 0], c="#f39c12", s=6, alpha=0.6,
                   zorder=4, label="NN predicted pts")
    if len(conv_pts) > 0:
        cp = np.array(conv_pts)
        ax.scatter(cp[:, 1], cp[:, 0], c="cyan", s=4, alpha=0.3,
                   zorder=3, label="Conv. corners")

    ax.scatter(start[1], start[0], c="lime", s=150, zorder=7, marker="*", label="Start")
    ax.scatter(goal[1],  goal[0],  c="red",  s=150, zorder=7, marker="*", label="Goal")
    ax.set_title("Map + NN Prediction", color="white", fontsize=12)
    ax.legend(fontsize=7, facecolor="#111", labelcolor="white")
    ax.set_xlim(0, grid.shape[1]); ax.set_ylim(grid.shape[0], 0)

    # ── Panel 2: RRT* tree + path ──────────────────────────────────────────
    ax = axes[1]
    ax.imshow(display, origin="upper")
    for n in nodes:
        if n.parent:
            ax.plot([n.col, n.parent.col], [n.row, n.parent.row],
                    color="#1a4a8a", lw=0.3, alpha=0.4)
    if path:
        px = [p[1] for p in path]
        py = [p[0] for p in path]
        ax.plot(px, py, color="#00ff88", lw=2.5, zorder=6, label="Path")
    ax.scatter(start[1], start[0], c="lime", s=150, zorder=7, marker="*", label="Start")
    ax.scatter(goal[1],  goal[0],  c="red",  s=150, zorder=7, marker="*", label="Goal")
    ax.set_title("RRT* Tree + Path", color="white", fontsize=12)
    ax.legend(fontsize=7, facecolor="#111", labelcolor="white")
    ax.set_xlim(0, grid.shape[1]); ax.set_ylim(grid.shape[0], 0)

    # ── Panel 3: cost convergence ──────────────────────────────────────────
    ax = axes[2]
    valid = [c for c in cost_history if not np.isnan(c)]
    if valid:
        ax.plot(valid, color="#00aaff", lw=1.5)
    ax.set_title("Cost Convergence", color="white", fontsize=12)
    ax.set_xlabel("Iteration", color="#aaa")
    ax.set_ylabel("Path Cost",  color="#aaa")
    ax.tick_params(colors="#aaa")
    ax.grid(color="#222", ls="--", lw=0.5)
    for s in ax.spines.values():
        s.set_edgecolor("#333")
    cost_str = f"{valid[-1]:.2f}" if valid else "N/A"
    stats = (f"Waypoints : {len(path)}\n"
             f"Final cost: {cost_str}\n"
             f"Iterations: {len(cost_history)}\n"
             f"Tree nodes: {len(nodes)}")
    ax.text(0.97, 0.97, stats, transform=ax.transAxes, color="white",
            fontsize=9, va="top", ha="right",
            bbox=dict(facecolor="#1a1a2e", edgecolor="#444",
                      alpha=0.85, boxstyle="round,pad=0.5"))

    plt.tight_layout(pad=2)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"Visualisation saved → {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1. Generate map
    print("Generating map …")
    grid, corner_mask = generate_map(size=224, num_shapes=args.num_shapes)
    start, goal = generate_start_goal(grid)
    print(f"  Start: {start}   Goal: {goal}")

    # 2. Load model
    model = load_model(args.weights, device=DEVICE)

    # 3. Encode grid and run inference
    print("Running UNet inference …")
    labelled = grid.copy().astype(np.int32)
    labelled[start] = 2
    labelled[goal]  = 3
    tensor = grid_to_tensor(labelled, device=DEVICE)

    binary_grid  = (grid == 1).astype(np.uint8)
    convex_mask  = detect_convex_corners(binary_grid)
    conv_pts_all = [tuple(p) for p in np.argwhere(convex_mask > 0)]

    pred_conv_arr = get_predicted_convex_points(model, tensor, convex_mask,
                                                threshold=0.5, radius=4)
    pred_conv = [tuple(p) for p in pred_conv_arr]
    print(f"  Predicted conv. pts: {len(pred_conv)}   All corners: {len(conv_pts_all)}")

    # 4. Run planner
    print("Running convex_neural_rrt_star …")
    path, cost_history, nodes, pred_pts, inside_pts, outside_pts = convex_neural_rrt_star(
        grid         = labelled,
        start        = start,
        goal         = goal,
        conv_pts_all = conv_pts_all,
        pred_conv    = pred_conv,
        conv_pts     = conv_pts_all,
        alpha_in     = args.alpha_in,
        alpha_out    = args.alpha_out,
        max_iter     = args.max_iter,
        step_size    = args.step_size,
        N            = 400,
        epsilon      = 1e-3,
    )

    valid = [c for c in cost_history if not np.isnan(c)]
    print(f"  Path found : {len(path) > 0}")
    print(f"  Waypoints  : {len(path)}")
    print(f"  Final cost : {valid[-1]:.2f}" if valid else "  Final cost : N/A")
    print(f"  Iterations : {len(cost_history)}")

    # 5. Visualise
    visualise(grid, start, goal, path, nodes,
              pred_conv, conv_pts_all, cost_history, args.output)


if __name__ == "__main__":
    main()
