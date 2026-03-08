import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


# =========================
# Visualization
# =========================
def plot_trajectory(points: np.ndarray, title: str = "Simplified Trajectory") -> None:
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(6, 6))
    plt.plot(x, y, marker="o", linestyle="-")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")
    plt.show()


# =========================
# Geometry helpers
# =========================
def distance_point_to_segment(point: np.ndarray, start_point: np.ndarray, end_point: np.ndarray) -> float:
    """
    Euclidean distance from a point to a line segment (start_point -> end_point).
    """
    point = np.asarray(point, dtype=float)
    start_point = np.asarray(start_point, dtype=float)
    end_point = np.asarray(end_point, dtype=float)

    if np.allclose(start_point, end_point):
        return float(np.linalg.norm(point - start_point))

    segment_vector = end_point - start_point
    point_vector = point - start_point

    t = float(np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector))
    t = float(np.clip(t, 0.0, 1.0))

    projection = start_point + t * segment_vector
    return float(np.linalg.norm(point - projection))


# =========================
# Ramer–Douglas–Peucker (RDP)
# =========================
def rdp_simplify(points: np.ndarray, tolerance: float) -> np.ndarray:
    """
    Simplify a polyline using the Ramer–Douglas–Peucker algorithm.

    points: (N,2) array
    tolerance: max allowed perpendicular distance
    returns: (M,2) array with M <= N
    """
    points = np.asarray(points, dtype=float)

    if len(points) < 3:
        return points

    start_point = points[0]
    end_point = points[-1]

    max_distance = -1.0
    index_of_max = -1

    for i in range(1, len(points) - 1):
        d = distance_point_to_segment(points[i], start_point, end_point)
        if d > max_distance:
            max_distance = d
            index_of_max = i

    if max_distance > tolerance:
        left = rdp_simplify(points[: index_of_max + 1], tolerance)
        right = rdp_simplify(points[index_of_max:], tolerance)
        return np.vstack([left[:-1], right])
    else:
        return np.vstack([start_point, end_point])


# =========================
# Main
# =========================
if __name__ == "__main__":

    # Input trajectory CSV (assumed consistent format)
    trajectory_csv_path = "trajectories\finger_trajectory.csv"

    # Output .npy path (same folder, auto-name)
    simplified_npy_path = os.path.splitext(trajectory_csv_path)[0] + "_simplified.npy"

    # Simplification tolerance 
    tolerance = 1.0

    # Load CSV -> numpy
    trajectory = np.array(pd.read_csv(trajectory_csv_path))

    # Keep only X,Y in case there are extra columns (time, etc.)
    trajectory = trajectory[:, :2].astype(float)

    # Simplify
    simplified_trajectory = rdp_simplify(trajectory, tolerance)

    # Save simplified trajectory
    np.save(simplified_npy_path, simplified_trajectory)

    print("Original points:", len(trajectory))
    print("Simplified points:", len(simplified_trajectory))
    print("Saved simplified trajectory to:", simplified_npy_path)

    # Plot
    plot_trajectory(simplified_trajectory, title="Simplified Trajectory (RDP)")