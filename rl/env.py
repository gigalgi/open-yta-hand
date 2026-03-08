"""
env.py — UMoBIC-Finger Gymnasium environment (sim-only)

What this env teaches the policy:
    "Given where my joints are and where I want my fingertip to go,
     what should I command the servo and motor to do?"

Observation (15,):
    [0:9]   joint positions  — all 9 joints (rad)
    [9:12]  fingertip xyz    — world-frame tip of distal phalanx (m)
    [12:15] target xyz       — goal position the fingertip should reach (m)

Action (2,):
    [0]  servo command  — normalized [-1, 1]  ->  [-0.3,  0.3] rad
    [1]  motor command  — normalized [-1, 1]  ->  [-0.3, 3.14] rad

Reward:
    -distance(fingertip, target)   always on (meters)
    +contact_bonus                 when fingertip within 5 mm of target

WHY REACHABLE-WORKSPACE SAMPLING (not a fixed bounding box)
============================================================
This finger is underactuated — 2 actuators control 9 coupled joints
via tendon equality constraints. Its reachable workspace is a thin
arc in 3D space, NOT a rectangular bounding box.

If targets are sampled from a box, most targets are OUTSIDE the
reachable arc and the policy can never reach them. The reward never
improves — training diverges.

Solution: at each reset, run a short random rollout to discover
where the fingertip actually goes, then pick one of those visited
positions as the target. This guarantees every target is reachable
by construction.

Fingertip position is tracked via a MuJoCo <site> named "fingertip"
placed at the tip of the distal phalanx in mjmodel.xml.
"""

import numpy as np
import mujoco
import mujoco.viewer  
import gymnasium as gym
from gymnasium import spaces
from config import Config


class FingerEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: Config = None, render_mode: str = None):
        super().__init__()
        self.cfg         = config or Config()
        self.render_mode = render_mode

        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(self.cfg.xml_path))
        self.data  = mujoco.MjData(self.model)

        # Look up IDs once at init (faster than name lookup every step)
        self._servo_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_s_pos")
        self._motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "joint_m_pos")

        # Fingertip tracked via <site> in mjmodel.xml (distal phalanx tip).
        # site_xpos is more accurate than body xpos: site is at the physical
        # fingertip, body origin is at the joint center.
        self._fingertip_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "fingertip")

        if self._fingertip_id < 0:
            raise ValueError(
                "Site 'fingertip' not found in mjmodel.xml.\n"
                "Add inside <body name='phalanx_link1_dis'>:\n"
                "  <site name='fingertip' pos='0 0 -0.012' size='0.003'/>"
            )

        # Spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # Episode state
        self._target     = np.zeros(3, dtype=np.float32)
        self._step_count = 0
        self._viewer     = None

    # Gymnasium API

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Step 1: rollout to discover reachable fingertip positions.
        # Reset to home, apply random actions, record where fingertip goes.
        # Every recorded position is reachable by construction.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        tips      = []
        n_explore = 40

        for _ in range(n_explore):
            action = self.np_random.uniform(-1.0, 1.0, size=2)
            self._apply(action)
            for _ in range(self.cfg.frame_skip):
                mujoco.mj_step(self.model, self.data)
            tips.append(self._fingertip().copy())

        # Pick one visited position as target.
        # Small Gaussian noise (1 mm std) adds variety without leaving workspace.
        chosen = tips[self.np_random.integers(len(tips))]
        self._target = (chosen + self.np_random.normal(0, 0.001, size=3)).astype(np.float32)

        # Step 2: reset sim to home so the episode always starts from neutral.
        # Critical — without this the episode starts from the rollout end state.
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._step_count = 0
        return self._obs(), {"target": self._target.tolist()}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self._apply(action)

        for _ in range(self.cfg.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        tip  = self._fingertip()
        dist = float(np.linalg.norm(tip - self._target))

        reward    = -dist
        if dist < self.cfg.reward_contact_threshold:
            reward += self.cfg.reward_contact_bonus

        truncated  = self._step_count >= self.cfg.max_steps
        terminated = False

        return self._obs(), reward, terminated, truncated, {
            "dist_mm": dist * 1000,
            "target":  self._target.tolist(),
        }

    def render(self):
        if self.render_mode == "human":
            if self._viewer is None:
                self._viewer = mujoco.viewer.launch_passive(
                    self.model, self.data)
            self._viewer.sync()

    def close(self):
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # Internal helpers

    def _obs(self) -> np.ndarray:
        qpos      = self.data.qpos[:9].astype(np.float32)
        fingertip = self._fingertip().astype(np.float32)
        target    = self._target.astype(np.float32)
        return np.concatenate([qpos, fingertip, target])

    def _fingertip(self) -> np.ndarray:
        """World-frame position of the fingertip site (m)."""
        return self.data.site_xpos[self._fingertip_id].copy()

    def _apply(self, action: np.ndarray) -> None:
        """Map normalized [-1, 1] -> real actuator ctrlrange and apply."""
        def scale(v, lo, hi):
            return lo + (v + 1.0) * 0.5 * (hi - lo)
        self.data.ctrl[self._servo_id] = scale(action[0], *self.cfg.servo_range)
        self.data.ctrl[self._motor_id] = scale(action[1], *self.cfg.motor_range)

    def set_target(self, xyz: np.ndarray) -> None:
        """Override the target position — used by play.py."""
        self._target = np.asarray(xyz, dtype=np.float32)
