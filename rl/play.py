"""
play.py -- Interactive policy viewer for UMoBIC-Finger

Load a trained policy. Type a target coordinate. The finger moves
there and HOLDS that position until you type a new one.

Usage:
    python play.py --model weights/best.zip

Controls (terminal):
    x y z   set target in meters   e.g.  0.0 -0.040 0.130
    r       random target (sampled from real workspace via rollout)
    q       quit

Behavior:
    - On startup:  finger sits at home position
    - On new target: policy runs until dist < 5mm OR 300 steps, then HOLDS
    - Viewer stays open and syncs continuously -- rotate/zoom freely
    - No auto-reset loop -- finger only moves when you send a new target
"""

import sys
import time
import argparse
import threading
import numpy as np
from pathlib import Path

import mujoco
import mujoco.viewer

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import Config
from env import FingerEnv


# ---------------------------------------------------------------------------
# Thread-safe target container
# ---------------------------------------------------------------------------

class TargetInput:
    """Background thread: reads target coordinates from stdin."""

    def __init__(self, cfg: Config):
        self.cfg       = cfg
        self._target   = None          # None = no new target yet
        self._new      = False         # flag: new target arrived
        self.quit      = False
        self._lock     = threading.Lock()

    def has_new(self) -> bool:
        with self._lock:
            return self._new

    def consume(self) -> np.ndarray:
        """Return the new target and clear the flag."""
        with self._lock:
            self._new = False
            return self._target.copy()

    def run(self) -> None:
        print("\nTarget controls:")
        print("  x y z  ->  set target in meters   e.g.  0.0 -0.040 0.130")
        print("  r      ->  random target from workspace")
        print("  q      ->  quit\n")
        print("Waiting for target (finger is at home position)...")

        while not self.quit:
            try:
                line = input("> ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                self.quit = True
                break

            if line == "q":
                self.quit = True

            elif line == "r":
                # Sample a reachable target via a short MuJoCo rollout.
                # This mirrors the same logic in env.reset() so the target
                # is guaranteed to be inside the actual workspace.
                model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
                data  = mujoco.MjData(model)
                tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "fingertip")
                rng   = np.random.default_rng()
                tips  = []
                for _ in range(40):
                    a = rng.uniform(-1.0, 1.0, size=2)
                    lo_s, hi_s = self.cfg.servo_range
                    lo_m, hi_m = self.cfg.motor_range
                    data.ctrl[0] = lo_s + (a[0] + 1.0) * 0.5 * (hi_s - lo_s)
                    data.ctrl[1] = lo_m + (a[1] + 1.0) * 0.5 * (hi_m - lo_m)
                    for _ in range(self.cfg.frame_skip):
                        mujoco.mj_step(model, data)
                    tips.append(data.site_xpos[tip_id].copy())
                new = tips[rng.integers(len(tips))].astype(np.float32)
                with self._lock:
                    self._target = new
                    self._new    = True
                print(f"  target -> ({new[0]*1000:.1f}, {new[1]*1000:.1f}, {new[2]*1000:.1f}) mm")

            else:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        new = np.array(parts, dtype=np.float32)
                        with self._lock:
                            self._target = new
                            self._new    = True
                        print(f"  target -> ({new[0]*1000:.1f}, {new[1]*1000:.1f}, {new[2]*1000:.1f}) mm")
                    except ValueError:
                        print("  bad input -- use:  x y z  (floats in meters)")
                else:
                    print("  use:  x y z  |  r  |  q")


# ---------------------------------------------------------------------------
# Main play loop
# ---------------------------------------------------------------------------

def play(model_path: str) -> None:
    cfg = Config()

    # Load model + VecNormalize
    norm_path = model_path.replace(".zip", "") + "_vecnorm.pkl"
    env       = FingerEnv(cfg, render_mode="human")
    vec_env   = DummyVecEnv([lambda: env])

    if Path(norm_path).exists():
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"VecNormalize loaded from {norm_path}")
    else:
        print(f"Warning: no VecNormalize at {norm_path} -- obs may be unscaled")

    model = SAC.load(model_path, env=vec_env)
    print(f"Model loaded: {model_path}\n")

    # Start input thread
    ti     = TargetInput(cfg)
    thread = threading.Thread(target=ti.run, daemon=True)
    thread.start()

    # Reset env to home -- finger starts at neutral position
    obs = vec_env.reset()
    env.render()

    # State machine
    # "idle"   -- waiting for a target, viewer syncing, no policy steps
    # "moving" -- policy running toward target
    # "holding"-- target reached (or timed out), holding last command
    state      = "idle"
    hold_steps = 0

    print("Viewer open. Type a target in the terminal to move the finger.\n")

    while not ti.quit:

        # ── New target received ───────────────────────────────────────────────
        if ti.has_new():
            target = ti.consume()
            env.set_target(target)

            # Patch observation so policy immediately sees new target
            obs[0, 12:15] = target

            state      = "moving"
            hold_steps = 0
            print(f"  Moving to target...")

        # ── Policy step (moving state) ────────────────────────────────────────
        if state == "moving":
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _ = vec_env.step(action)

            tip  = env._fingertip()
            dist = float(np.linalg.norm(tip - env._target)) * 1000

            # Print progress every 20 steps
            if env._step_count % 20 == 0:
                print(f"  dist={dist:.1f} mm  "
                      f"tip=({tip[0]*1000:.1f}, {tip[1]*1000:.1f}, {tip[2]*1000:.1f}) mm")

            # Converged or timed out -> hold
            if dist < cfg.reward_contact_threshold * 1000 or done[0]:
                if dist < cfg.reward_contact_threshold * 1000:
                    print(f"  Reached! dist={dist:.1f} mm -- holding.")
                else:
                    print(f"  Timeout. dist={dist:.1f} mm -- holding best position.")
                state = "holding"

        # ── Hold state: keep applying last action, sync viewer ────────────────
        elif state == "holding":
            # Step physics with the last control values unchanged.
            # This keeps the finger in place without policy overhead.
            for _ in range(cfg.frame_skip):
                mujoco.mj_step(env.model, env.data)
            hold_steps += 1

        # ── Idle: just sync viewer ────────────────────────────────────────────
        elif state == "idle":
            pass   # nothing to do, just render below

        env.render()
        time.sleep(0.01)   # ~100 Hz render loop

    vec_env.close()
    print("Bye.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Interactive policy viewer for UMoBIC-Finger")
    parser.add_argument(
        "--model", type=str,
        default=str(Config().weights_dir / "best.zip"),
        help="Path to .zip model file")
    args = parser.parse_args()

    if not Path(args.model).exists():
        print(f"Model not found: {args.model}")
        print("Train first:  python train.py")
        sys.exit(1)

    play(args.model)
