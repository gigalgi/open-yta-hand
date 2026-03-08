"""
workspace_vis.py — UMoBIC-Finger reachable workspace visualizer

Run from ANYWHERE — the script resolves its own path:
    python rl/workspace_vis.py
    python rl/workspace_vis.py --steps 10000
    python rl/workspace_vis.py --color-by z
    python rl/workspace_vis.py --color-by time

Why this lives in rl/ and not kinematics/:
    The workspace shown here is the ACTUAL fingertip arc produced by the
    MuJoCo simulation — it is the workspace the RL policy trains on.
    It is computed by running random actions through FingerEnv (rl/env.py),
    not from the kinematic equations. It belongs next to what it describes.

Color modes:
    density  (default) — highlights the most visited regions
    z                  — shows height distribution across the arc
    time               — shows how the finger sweeps over time
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Always resolve relative to THIS file so the script runs from any folder.
_RL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_RL_DIR))

from env import FingerEnv
from config import Config


def visualize(n_steps: int = 5000, color_by: str = "density") -> None:

    print(f"Sampling {n_steps} random actions through MuJoCo...")
    print("(Same reachable-workspace logic used in env.reset())\n")

    cfg = Config()
    env = FingerEnv(cfg)
    env.reset()

    points = []

    for i in range(n_steps):
        action = np.random.uniform(-1.0, 1.0, size=2)
        obs, _, terminated, truncated, _ = env.step(action)

        tip = obs[9:12].copy()   # fingertip xyz from observation (see env.py)
        points.append(tip)

        if terminated or truncated:
            env.reset()

        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n_steps} steps...")

    env.close()

    points = np.array(points)   # (N, 3)

    print(f"\nWorkspace bounds (from this sample):")
    print(f"  X: [{points[:,0].min()*1000:.1f}, {points[:,0].max()*1000:.1f}] mm")
    print(f"  Y: [{points[:,1].min()*1000:.1f}, {points[:,1].max()*1000:.1f}] mm")
    print(f"  Z: [{points[:,2].min()*1000:.1f}, {points[:,2].max()*1000:.1f}] mm")

    # Color mapping
    if color_by == "time":
        colors = np.arange(len(points))
        cmap   = "plasma"
        clabel = "Step index (time)"
    elif color_by == "z":
        colors = points[:, 2]
        cmap   = "viridis"
        clabel = "Z height (m)"
    else:   # density
        centroid = points.mean(axis=0)
        colors   = np.linalg.norm(points - centroid, axis=1)
        cmap     = "cool"
        clabel   = "Distance from centroid (m)"

    fig = plt.figure(figsize=(11, 8))
    ax  = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        points[:, 0], points[:, 1], points[:, 2],
        c=colors, cmap=cmap, s=1, alpha=0.5
    )
    plt.colorbar(sc, ax=ax, label=clabel, shrink=0.6)


    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(
        f"UMoBIC-Finger — Reachable Workspace\n"
        f"{n_steps} random actions · {len(points)} fingertip positions · "
        f"color={color_by}"
    )
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize the UMoBIC-Finger reachable fingertip workspace")
    parser.add_argument("--steps", type=int, default=5000,
        help="Number of random actions (default 5000, use 10000+ for denser plot)")
    parser.add_argument("--color-by", type=str, default="density",
        choices=["density", "time", "z"],
        help="Color scheme: density (default) | time | z")
    args = parser.parse_args()
    visualize(n_steps=args.steps, color_by=args.color_by)
