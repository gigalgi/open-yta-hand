"""
grasp_point_generator.py
========================
Stage 1 of 3 in the grasping pipeline.

    [1] GraspPointGenerator  →  (x, y) contact point  ← THIS FILE
    [2] predict.py           →  θ motor angle
    [3] impedance_control    →  Arduino motor control

Finds the grasp contact point as the minimum-distance collision between
the finger sweep trajectory and the object boundary.

The finger trajectory (finger_tray.npy) is pre-computed from the kinematic
model — it is always the same for a given hand configuration.

The object shape (circle_tray.npy) is placed in the palm frame using an
anchor rule: leftmost boundary touches x0, centroid_y aligns to y0.

The output — closest boundary point (p_contact) — is the (x, y) coordinate
sent to the LSTM in predict.py.

Usage (standalone):
    python grasp_point_generator.py

Usage (as module):
    from grasp_point_generator import GraspPointGenerator
    gen = GraspPointGenerator()
    result = gen.compute()
    print(result.contact_point)   # (x, y) → send to LSTM
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Config — paths and placement parameters
# ---------------------------------------------------------------------------

# Directory where this file lives (so paths work regardless of cwd)
_HERE = os.path.dirname(os.path.abspath(__file__))

# Finger trajectory: pre-computed from kinematic model, always the same
TRAJECTORY_NPY = os.path.join(_HERE, "trajectories", "finger_trajectory_simplified.npy")

# Object boundary: loaded at runtime, describes object cross-section in local frame
SHAPE_NPY = os.path.join(_HERE, "trajectories", "circle_tray.npy")

# Placement anchor in palm frame (mm):
#   leftmost boundary touches ANCHOR_X0
#   shape centroid_y aligns to ANCHOR_Y0
ANCHOR_X0 = 20.0
ANCHOR_Y0 = -40.0

# Optional rotation before placement (radians). 0 = no rotation.
THETA = 0.0


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class GraspResult:
    """
    Output of GraspPointGenerator.compute().

    Attributes:
        contact_point:    (x, y) on the object boundary — send this to LSTM.
        trajectory_point: Closest point on the finger trajectory to the object.
        min_distance:     Gap between finger and object at contact (mm).
        shape_world:      Object boundary in palm frame (N, 2).
        center_world:     Object centroid in palm frame (2,).
    """
    contact_point:    np.ndarray       # → LSTM input
    trajectory_point: np.ndarray
    min_distance:     float
    shape_world:      np.ndarray
    center_world:     np.ndarray


# ---------------------------------------------------------------------------
# Geometry — kept as plain functions so they are easy to test and reuse
# ---------------------------------------------------------------------------

def rotate_points(points: np.ndarray, theta: float) -> np.ndarray:
    """Rotate 2D points by angle theta (radians) around origin."""
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]], dtype=float)
    return np.asarray(points, dtype=float) @ R.T


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Ray-casting point-in-polygon test.
    polygon must be ordered (CW or CCW) and non-self-intersecting.
    """
    x, y = point
    poly = np.asarray(polygon, dtype=float)
    n = len(poly)
    inside = False
    x0, y0 = poly[-1]
    for i in range(n):
        x1, y1 = poly[i]
        if (y1 > y) != (y0 > y):
            x_int = (x0 - x1) * (y - y1) / (y0 - y1 + 1e-18) + x1
            if x_int > x:
                inside = not inside
        x0, y0 = x1, y1
    return inside


def dist_point_to_segment(p: np.ndarray,
                           a: np.ndarray,
                           b: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Minimum distance from point p to line segment ab.
    Returns (distance, closest_point_on_segment).
    """
    p, a, b = map(lambda v: np.asarray(v, float), [p, a, b])
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-18:
        return float(np.linalg.norm(p - a)), a.copy()
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q)), q


def dist_point_to_boundary(p: np.ndarray,
                            boundary: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Minimum distance from point p to a closed polygon boundary.
    Returns (distance, closest_point_on_boundary).
    """
    boundary = np.asarray(boundary, float)
    best_d, best_q = float("inf"), None
    n = len(boundary)
    for i in range(n):
        d, q = dist_point_to_segment(p, boundary[i], boundary[(i + 1) % n])
        if d < best_d:
            best_d, best_q = d, q
    return best_d, best_q


def place_shape(shape_local: np.ndarray,
                x0: float,
                y0: float,
                theta: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Place a shape in the palm frame using the anchor rule:
      1. Rotate shape by theta.
      2. Translate so centroid_y == y0.
      3. Translate so leftmost boundary x == x0.

    Returns:
        (shape_world, centroid_world, leftmost_x_world)
    """
    pts = rotate_points(shape_local[:, :2], theta)
    centroid = pts.mean(axis=0)
    tx = x0 - float(np.min(pts[:, 0]))
    ty = y0 - centroid[1]
    shape_world = pts + np.array([tx, ty])
    centroid_world = centroid + np.array([tx, ty])
    return shape_world, centroid_world, float(np.min(shape_world[:, 0]))


def find_collision_point(trajectory: np.ndarray,
                         boundary: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the minimum-distance collision between finger trajectory and object boundary.
    Only considers trajectory points that are OUTSIDE the object (pre-contact).

    Returns:
        (min_distance, closest_trajectory_point, contact_boundary_point)
        Returns (inf, None, None) if all trajectory points are inside the object.
    """
    best_d = float("inf")
    best_p = best_q = None

    for p in trajectory:
        if point_in_polygon(p, boundary):
            continue   # already inside — skip
        d, q = dist_point_to_boundary(p, boundary)
        if d < best_d:
            best_d, best_p, best_q = d, p.copy(), q

    return best_d, best_p, best_q


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class GraspPointGenerator:
    """
    Computes the 2D grasp contact point given a finger trajectory and an object shape.

    The contact point (result.contact_point) is the (x, y) coordinate in the
    palm reference frame that should be passed to the LSTM predictor in predict.py.

    Args:
        trajectory_path: Path to finger trajectory .npy file.
        shape_path:      Path to object boundary .npy file.
        anchor_x0:       Leftmost boundary x in palm frame (mm).
        anchor_y0:       Shape centroid y in palm frame (mm).
        theta:           Shape rotation before placement (radians).
    """

    def __init__(self,
                 trajectory_path: str = TRAJECTORY_NPY,
                 shape_path: str = SHAPE_NPY,
                 anchor_x0: float = ANCHOR_X0,
                 anchor_y0: float = ANCHOR_Y0,
                 theta: float = THETA):
        self.trajectory_path = trajectory_path
        self.shape_path = shape_path
        self.anchor_x0 = anchor_x0
        self.anchor_y0 = anchor_y0
        self.theta = theta

    def compute(self, visualize: bool = False) -> GraspResult:
        """
        Run the full grasp point computation.

        Args:
            visualize: If True, show the 2D grasp scene plot.

        Returns:
            GraspResult with contact_point (x, y) ready to send to LSTM.

        Raises:
            ValueError: If no trajectory point is found outside the object.
        """
        # Load data
        traj = np.load(self.trajectory_path)[:, :2].astype(float)
        shape_local = np.load(self.shape_path).reshape(-1, 2).astype(float)

        # Place object in palm frame
        shape_world, center_world, _ = place_shape(
            shape_local, self.anchor_x0, self.anchor_y0, self.theta
        )

        # Find collision / grasp point
        dmin, p_traj, p_contact = find_collision_point(traj, shape_world)

        if p_traj is None:
            raise ValueError(
                "All trajectory points are inside the object — "
                "check anchor position or object size."
            )

        result = GraspResult(
            contact_point    = p_contact,
            trajectory_point = p_traj,
            min_distance     = dmin,
            shape_world      = shape_world,
            center_world     = center_world,
        )

        if visualize:
            self._plot(traj, result)

        return result

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def _plot(self, traj: np.ndarray, result: GraspResult) -> None:
        """
        Two-panel plot:
          Left  — full scene (palm, finger sweep, object, contact)
          Right — zoomed on the contact region
        """
        bp = result.contact_point
        tp = result.trajectory_point
        sw = result.shape_world
        cc = result.center_world
        d  = result.min_distance

        # Estimate object radius for zoom
        r = float(np.max(np.linalg.norm(sw - cc, axis=1)))

        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        fig.suptitle(
            "UMoBIC-Finger  —  2D Grasp Point Detection  (Palm = Reference Origin)",
            fontsize=13, fontweight="bold"
        )

        for i, ax in enumerate(axes):
            ax.set_facecolor("#f4f6f8")
            ax.grid(True, color="white", linewidth=1.2, alpha=0.8)

            # Palm origin
            ax.scatter(0, 0, s=220, color="#2c3e50", zorder=7, marker="s",
                       label="Palm (origin)")
            ax.annotate("Palm\n(origin)", (0, 0), xytext=(2, 4), fontsize=8,
                        color="#2c3e50", fontweight="bold")

            # Finger trajectory
            ax.plot(traj[:, 0], traj[:, 1], color="#2980b9", linewidth=2,
                    alpha=0.75, label="Finger sweep trajectory", zorder=3)
            ax.scatter(*traj[0],  s=90, color="#2980b9", zorder=5, marker="o",
                       label=f"MCP  ({traj[0,0]:.0f}, {traj[0,1]:.0f})")
            ax.scatter(*traj[-1], s=90, color="#1a5276", zorder=5, marker="^",
                       label=f"Fingertip  ({traj[-1,0]:.0f}, {traj[-1,1]:.0f})")

            # Object boundary + fill
            closed = np.vstack([sw, sw[0]])
            ax.fill(sw[:, 0], sw[:, 1], color="#e67e22", alpha=0.15, zorder=2)
            ax.plot(closed[:, 0], closed[:, 1], color="#e67e22",
                    linewidth=2.5, zorder=3, label="Object boundary")
            ax.scatter(*cc, s=80, color="#e67e22", zorder=5,
                       marker="+", linewidths=2.5,
                       label=f"Object centroid  ({cc[0]:.1f}, {cc[1]:.1f})")

            # Min-distance line
            ax.plot([tp[0], bp[0]], [tp[1], bp[1]], "--", color="#8e44ad",
                    linewidth=1.8, alpha=0.9, zorder=4,
                    label=f"Min distance:  {d:.2f} mm")

            # Closest trajectory point
            ax.scatter(*tp, s=130, color="#8e44ad", zorder=6, marker="D",
                       label=f"Closest finger pt  ({tp[0]:.1f}, {tp[1]:.1f})")

            # Grasp contact point (the answer → goes to LSTM)
            ax.scatter(*bp, s=320, color="#e74c3c", zorder=8, marker="*",
                       label=f"GRASP POINT  ({bp[0]:.1f}, {bp[1]:.1f})")
            ax.annotate(
                f"  GRASP POINT\n  ({bp[0]:.1f}, {bp[1]:.1f}) mm\n  → LSTM input",
                bp, xytext=(bp[0] + 2, bp[1] + 4),
                fontsize=9, color="#c0392b", fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.3", fc="white",
                          ec="#e74c3c", alpha=0.92)
            )
            mid = (tp + bp) / 2
            ax.annotate(f"{d:.2f} mm", mid, xytext=(mid[0]+1, mid[1]+2),
                        fontsize=8, color="#8e44ad")

            ax.set_xlabel("X — horizontal (mm)", fontsize=10)
            ax.set_ylabel("Y — vertical (mm)", fontsize=10)
            ax.set_aspect("equal")

            if i == 0:
                all_x = np.concatenate([traj[:, 0], sw[:, 0], [0]])
                all_y = np.concatenate([traj[:, 1], sw[:, 1], [0]])
                ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
                ax.set_ylim(all_y.min() - 10, all_y.max() + 10)
                ax.set_title("Full Scene", fontsize=11)
            else:
                margin = max(r * 3, 15)
                ax.set_xlim(cc[0] - margin, cc[0] + margin)
                ax.set_ylim(cc[1] - margin, cc[1] + margin)
                ax.set_title("Zoomed — Contact Region", fontsize=11)

            ax.legend(loc="upper right", fontsize=7.5, framealpha=0.92)

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    gen    = GraspPointGenerator()
    result = gen.compute(visualize=True)

    print("=" * 50)
    print(f"GRASP POINT (→ LSTM):  x={result.contact_point[0]:.2f},  y={result.contact_point[1]:.2f}  mm")
    print(f"Closest tray point:    x={result.trajectory_point[0]:.2f},  y={result.trajectory_point[1]:.2f}  mm")
    print(f"Min distance:          {result.min_distance:.4f} mm")
    print("=" * 50)
