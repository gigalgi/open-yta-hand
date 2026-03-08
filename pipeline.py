"""
pipeline.py
===========
Orchestrator for the full UMoBIC-Finger grasping pipeline.

    Stage 1: GraspPointGenerator  ->  (x, y) contact point on object
    Stage 2: LSTMPredictor        ->  theta motor angle
    Stage 3: SerialBridge         ->  angle sent to Arduino impedance controller

This file owns the flow. Each stage is a separate importable module so they
can be tested, replaced, or extended independently.

Usage:
    # Full pipeline with hardware
    python pipeline.py --port COM5

    # Full pipeline, show grasp visualization
    python pipeline.py --port COM5 --visualize

    # Test without hardware (skips serial)
    python pipeline.py --dry-run --visualize

    # Override grasp point manually (skip stage 1)
    python pipeline.py --port COM5 --x 22.0 --y -37.0

    # Continuous loop mode (sends angles every --interval seconds)
    python pipeline.py --port COM5 --loop --interval 1.0
"""

import os
import time
import argparse
import numpy as np

from kinematics.grasping_generator.grasp_point_generator import GraspPointGenerator
from lstm.predict import LSTMPredictor, SerialBridge




# ---------------------------------------------------------------------------
# Paths — adjust to match your directory layout
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH     = os.path.join(_HERE, "lstm", "weights", "lstm_inverse_kinematics.pth")
TRAJECTORY_PATH  = os.path.join(_HERE, "kinematics","grasping_generator","trajectories", "finger_trajectory_simplified.npy")
SHAPE_PATH       = os.path.join(_HERE, "kinematics","grasping_generator","trajectories", "circle_tray.npy")


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_once(predictor: LSTMPredictor,
             bridge:    SerialBridge,
             grasp_x:   float,
             grasp_y:   float,
             verbose:   bool = True) -> float:
    """
    Execute one grasp cycle: predict angle -> send to Arduino -> read status.

    Args:
        predictor: Loaded LSTMPredictor.
        bridge:    Connected SerialBridge (or None in dry-run mode).
        grasp_x:   Fingertip target x in mm.
        grasp_y:   Fingertip target y in mm.
        verbose:   Print each stage to console.

    Returns:
        theta: Motor angle that was sent (degrees).
    """
    # Stage 2: LSTM inference
    theta = predictor.predict(grasp_x, grasp_y)

    if verbose:
        print(f"  Grasp point  x={grasp_x:.2f}  y={grasp_y:.2f}  ->  theta={theta:.2f} deg")

    # Stage 3: send to Arduino
    if bridge is not None:
        bridge.send_angle(theta)
        time.sleep(0.05)                    # one Arduino control cycle (10 ms)
        status = bridge.read_status()
        if verbose and status:
            print(f"  Arduino  ->  {status}")

    return theta


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="UMoBIC-Finger full grasping pipeline"
    )
    parser.add_argument("--port",       type=str,   default="COM5",
                        help="Serial port (COM5 or /dev/ttyUSB0)")
    parser.add_argument("--weights",    type=str,   default=WEIGHTS_PATH,
                        help="Path to LSTM .pth weights file")
    parser.add_argument("--traj",       type=str,   default=TRAJECTORY_PATH,
                        help="Path to finger trajectory .npy file")
    parser.add_argument("--shape",      type=str,   default=SHAPE_PATH,
                        help="Path to object shape boundary .npy file")
    parser.add_argument("--anchor-x",   type=float, default=20.0,
                        help="Object anchor x in palm frame (mm)")
    parser.add_argument("--anchor-y",   type=float, default=-35.0,
                        help="Object anchor y in palm frame (mm)")
    parser.add_argument("--x",          type=float, default=None,
                        help="Manual grasp x override — skips stage 1")
    parser.add_argument("--y",          type=float, default=None,
                        help="Manual grasp y override — skips stage 1")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run without serial (no Arduino needed)")
    parser.add_argument("--visualize",  action="store_true",
                        help="Show 2D grasp point visualization")
    parser.add_argument("--loop",       action="store_true",
                        help="Run pipeline continuously until Ctrl+C")
    parser.add_argument("--interval",   type=float, default=1.0,
                        help="Seconds between cycles in loop mode")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  UMoBIC-Finger Grasping Pipeline")
    print("=" * 55)

    # ------------------------------------------------------------------
    # Stage 1: Grasp point generation
    # ------------------------------------------------------------------
    if args.x is not None and args.y is not None:
        # Manual override — skip the generator
        grasp_x, grasp_y = args.x, args.y
        print(f"[Stage 1] Manual override  ->  ({grasp_x:.2f}, {grasp_y:.2f}) mm")
    else:
        # Use the built-in RDP / collision detection generator.
        # To swap in a different generator (GraspNet, AnyGrasp, etc.):
        #   1. Implement a class with a .compute() method
        #   2. Make it return an object with a .contact_point attribute
        #   3. Replace GraspPointGenerator() with your class here
        print("[Stage 1] Computing grasp point...")
        generator = GraspPointGenerator(
            trajectory_path = args.traj,
            shape_path      = args.shape,
            anchor_x0       = args.anchor_x,
            anchor_y0       = args.anchor_y,
        )
        result   = generator.compute(visualize=args.visualize)
        grasp_x  = float(result.contact_point[0])
        grasp_y  = float(result.contact_point[1])
        print(f"[Stage 1] Grasp point  ->  ({grasp_x:.2f}, {grasp_y:.2f}) mm  "
              f"[dist={result.min_distance:.2f} mm]")

    # ------------------------------------------------------------------
    # Stage 2: Load LSTM
    # ------------------------------------------------------------------
    print("[Stage 2] Loading LSTM model...")
    predictor = LSTMPredictor(weights_path=args.weights)

    # ------------------------------------------------------------------
    # Stage 3: Serial bridge
    # ------------------------------------------------------------------
    bridge = None
    if not args.dry_run:
        bridge = SerialBridge(port=args.port)
        bridge.connect()
    else:
        print("[Stage 3] Dry-run mode — serial bridge disabled")

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    try:
        if args.loop:
            print(f"\nRunning in loop mode  (interval={args.interval}s)  —  Ctrl+C to stop\n")
            cycle = 0
            while True:
                cycle += 1
                print(f"--- Cycle {cycle} ---")
                run_once(predictor, bridge, grasp_x, grasp_y)
                time.sleep(args.interval)
        else:
            print()
            run_once(predictor, bridge, grasp_x, grasp_y)
            print("\nDone.")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        if bridge is not None:
            bridge.disconnect()


if __name__ == "__main__":
    main()
