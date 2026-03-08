"""
train.py -- SAC training for UMoBIC-Finger (sim-only)

Usage:
    python train.py                          # headless, fresh run
    python train.py --render                 # open MuJoCo viewer while training
    python train.py --resume weights/best    # continue from checkpoint
    python train.py --steps 500000           # custom budget
    python train.py --render --steps 100000  # watch + custom budget
"""

import argparse
import os
import numpy as np
from pathlib import Path

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config import Config
from env import FingerEnv


# ---------------------------------------------------------------------------
# Callback: logging, checkpoints, best-model tracking
# ---------------------------------------------------------------------------

class LogAndSaveCallback(BaseCallback):
    """Logs episode stats, saves checkpoints, tracks best model."""

    def __init__(self, cfg: Config):
        super().__init__(verbose=0)
        self.cfg        = cfg
        self._rewards   = []
        self._dists     = []
        self._best_mean = -np.inf

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self._rewards.append(info["episode"]["r"])
            if "dist_mm" in info:
                self._dists.append(info["dist_mm"])

        t = self.num_timesteps

        # Console + TensorBoard log
        if t % self.cfg.log_every == 0 and self._rewards:
            n        = min(50, len(self._rewards))
            mean_rew = float(np.mean(self._rewards[-n:]))
            mean_d   = float(np.mean(self._dists[-200:])) if self._dists else 0.0
            print(f"  [{t:>8} steps]  reward={mean_rew:+.2f}  dist={mean_d:.1f} mm")
            self.logger.record("eval/mean_reward",  mean_rew)
            self.logger.record("eval/mean_dist_mm", mean_d)
            self.logger.dump(t)

        # Periodic checkpoint
        if t % self.cfg.checkpoint_every == 0:
            path = self.cfg.weights_dir / f"ckpt_{t}"
            self.model.save(path)
            self.training_env.save(str(path) + "_vecnorm.pkl")
            print(f"  [saved] {path}.zip")

        # Best model by mean reward
        if len(self._rewards) >= 20:
            mean_rew = float(np.mean(self._rewards[-50:]))
            if mean_rew > self._best_mean:
                self._best_mean = mean_rew
                best = self.cfg.weights_dir / "best"
                self.model.save(best)
                self.training_env.save(str(best) + "_vecnorm.pkl")

        return True


# ---------------------------------------------------------------------------
# Render callback: syncs the MuJoCo viewer every N steps during training
# ---------------------------------------------------------------------------

class RenderCallback(BaseCallback):
    """
    Syncs the MuJoCo passive viewer at each training step.
    Only instantiated when --render flag is passed.

    Note: rendering slows training by ~30% -- use for debugging behavior,
    not for long training runs.
    """

    def __init__(self, env: FingerEnv):
        super().__init__(verbose=0)
        self._env = env

    def _on_step(self) -> bool:
        self._env.render()
        return True


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(cfg: Config, resume: str = None,
          total_steps: int = None, render: bool = False) -> None:

    cfg.weights_dir.mkdir(parents=True, exist_ok=True)
    cfg.log_dir.mkdir(parents=True, exist_ok=True)

    # Auto-select GPU if available.
    # MuJoCo physics always runs on CPU -- only the SAC network uses GPU.
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device.upper()}")

    # Build env -- with or without viewer
    render_mode = "human" if render else None
    inner_env   = FingerEnv(cfg, render_mode=render_mode)
    vec_env     = DummyVecEnv([lambda: Monitor(inner_env)])

    # VecNormalize: standardizes observations online.
    # qpos (rad), site_xpos (m), and target (m) live on different scales.
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                           clip_obs=10.0, gamma=cfg.gamma)

    if resume:
        print(f"Resuming from: {resume}")
        model = SAC.load(resume, env=vec_env, device=device,
                         tensorboard_log=str(cfg.log_dir))
        norm_path = resume.replace(".zip", "") + "_vecnorm.pkl"
        if os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env)
    else:
        model = SAC(
            policy          = "MlpPolicy",
            env             = vec_env,
            learning_rate   = cfg.learning_rate,
            buffer_size     = cfg.buffer_size,
            batch_size      = cfg.batch_size,
            learning_starts = cfg.learning_starts,
            gamma           = cfg.gamma,
            tau             = cfg.tau,
            policy_kwargs   = {"net_arch": cfg.net_arch},
            ent_coef        = "auto",
            device          = device,
            tensorboard_log = str(cfg.log_dir),
            verbose         = 0,
        )

    # Build callback list
    callbacks = [LogAndSaveCallback(cfg)]
    if render:
        callbacks.append(RenderCallback(inner_env))
        print("Render mode ON -- viewer will open. Training ~30% slower.")

    steps = total_steps or cfg.total_timesteps
    print(f"\nSAC training -- {steps:,} steps")
    print(f"  tensorboard --logdir {cfg.log_dir}\n")

    try:
        model.learn(
            total_timesteps     = steps,
            callback            = callbacks,
            reset_num_timesteps = resume is None,
            progress_bar        = True,
        )
    except KeyboardInterrupt:
        print("\nInterrupted -- saving...")

    final = cfg.weights_dir / "final"
    model.save(final)
    vec_env.save(str(final) + "_vecnorm.pkl")
    print(f"\nDone.  weights -> {final}.zip")
    print(f"Play:  python play.py --model {final}.zip")
    vec_env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="SAC training for UMoBIC-Finger")
    parser.add_argument("--resume", type=str, default=None,
        help="Resume from checkpoint (.zip path)")
    parser.add_argument("--steps",  type=int, default=None,
        help="Total training steps (overrides config)")
    parser.add_argument("--render", action="store_true",
        help="Open MuJoCo viewer during training (slower, for debugging)")
    args = parser.parse_args()

    train(Config(), resume=args.resume,
          total_steps=args.steps, render=args.render)
