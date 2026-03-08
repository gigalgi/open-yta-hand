"""
config.py — UMoBIC-Finger RL (sim-only)

Single dataclass. Every number has a comment explaining its unit and origin.
Change values here; nothing else needs to be touched.
"""

from dataclasses import dataclass, field
from pathlib import Path

# Project root = two levels above this file  (rl/ → open-yta-hand/)
_RL_DIR   = Path(__file__).parent
_ROOT_DIR = _RL_DIR.parent


@dataclass
class Config:

    # ── Paths ────────────────────────────────────────────────────────────────
    xml_path     : Path = _ROOT_DIR / "simulation" / "mujoco" / "mjmodel.xml"
    weights_dir  : Path = _RL_DIR   / "weights"
    log_dir      : Path = _RL_DIR   / "logs"

    # ── Simulation ───────────────────────────────────────────────────────────
    # Physics runs at 500 Hz (timestep=0.002 s set in mjmodel.xml).
    # We apply one action every FRAME_SKIP steps → 100 Hz control rate,
    # matching the real impedance controller loop.
    frame_skip   : int   = 5          # steps per action  →  0.01 s / action

    # ── Episode ──────────────────────────────────────────────────────────────
    max_steps    : int   = 300        # 300 × 0.01 s = 3 s per episode

    # ── Action space ─────────────────────────────────────────────────────────
    # Policy outputs [-1, 1]; mapped to real ctrlrange in env.step().
    # Values from mjmodel.xml <actuator> ctrlrange — [HW] measured limits.
    servo_range  : tuple = (-0.3,  0.3)   # rad  — MG90S horn travel
    motor_range  : tuple = (-0.3,  3.14)  # rad  — JA12-N20 full rotation

    # ── Pen object ───────────────────────────────────────────────────────────
    # Matches circle_tray.npy and LSTM training examples.
    # [HW] radius=6mm = real pen cross-section used in LSTM data collection.
    # [EQ] center position from grasp_point_generator.py ANCHOR_X0/Y0.
    pen_x      : float = 0.023   # m — pen center x in palm frame (23mm)
    pen_y      : float = -0.035  # m — pen center y in palm frame (-35mm)
    pen_z      : float = 0.100   # m — pen center z (mid fingertip workspace)
    pen_radius : float = 0.006   # m — 6mm radius, matches circle_tray.npy

    # ── Grasp target ──────────────────────────────────────────────────────────
    # Contact point = left surface of pen (closest side to approaching finger).
    # Computed as: pen_center_x - pen_radius = 0.023 - 0.006 = 0.017m
    # This is the exact point grasp_point_generator.py would return.
    # Randomization: small ±3mm offset at each reset to improve robustness.
    target_x_nominal : float = 0.017   # m — pen left surface contact point
    target_y_nominal : float = -0.035  # m — same as pen center y
    target_z_nominal : float = 0.100   # m — same as pen center z
    target_rand      : float = 0.003   # m — ±3mm randomization at reset

    # ── Reward ───────────────────────────────────────────────────────────────
    # Dense: negative distance (meters).  Simple and interpretable.
    # A perfect episode would score ~0; random policy scores ~ -0.3.
    reward_contact_bonus     : float = 1.0    # extra reward within threshold
    reward_contact_threshold : float = 0.005  # m  (5 mm = close enough)

    # ── SAC hyperparameters ──────────────────────────────────────────────────
    # Standard values well-validated for continuous manipulation tasks.
    total_timesteps : int   = 300_000   # ~30 min on a modern CPU
    learning_rate   : float = 3e-4
    buffer_size     : int   = 200_000
    batch_size      : int   = 256
    learning_starts : int   = 2_000    # random exploration before training
    gamma           : float = 0.99
    tau             : float = 0.005
    net_arch        : list  = field(default_factory=lambda: [256, 256])

    # ── Logging ──────────────────────────────────────────────────────────────
    checkpoint_every : int = 25_000   # save weights every N steps
    log_every        : int = 1_000    # print stats every N steps
