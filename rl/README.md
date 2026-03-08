# UMoBIC-Finger — RL Training (`rl/`)

Reinforcement learning pipeline for training a fingertip positioning policy
on the UMoBIC-Finger **in simulation only**. No hardware connection needed.

---

## Position in the full system

```
rl/train.py          trains the policy in MuJoCo simulation
      ↓
rl/play.py           interactive: type a target, finger moves and holds
      ↓
(future) pipeline.py → predict.py → SerialBridge → Arduino
```

The RL policy learns the relationship between **servo + motor commands** and
**fingertip position in 3D space**. Once trained, give it any reachable target
coordinate and it drives the finger there and holds that position.

---

## File structure

```
rl/
├── config.py            All hyperparameters and paths — edit here only
├── env.py               Gymnasium environment wrapping mjmodel.xml
├── train.py             SAC training + checkpointing + optional live viewer
├── play.py              Interactive viewer: type target → finger moves → holds
├── workspace_vis.py     3D plot of the real reachable fingertip workspace
├── requirements.txt     Python dependencies
├── weights/             Saved checkpoints (created on first run)
└── logs/                TensorBoard logs (created on first run)
```

---

## Quick start

### 1. Install dependencies

```bash
pip install -r rl/requirements.txt
```

### 2. Train

```bash
# Headless — full speed (recommended for full runs)
python rl/train.py

# With live MuJoCo viewer — ~30% slower, use for debugging behavior
python rl/train.py --render

# Short debug run — watch ~15 seconds of sim, then stop
python rl/train.py --render --steps 5000

# Resume from checkpoint
python rl/train.py --resume rl/weights/ckpt_100000.zip

# Custom step budget
python rl/train.py --steps 500000
```

Monitor training in a second terminal:

```bash
tensorboard --logdir rl/logs/SAC_1
```

The best model by mean reward is always saved at `rl/weights/best.zip`.

#### What `--steps` means

Each step = one action applied to the simulation (5 physics substeps at 0.002s each = 0.01s of simulated time).

```
5,000 steps  ≈  50 s simulated  ≈  ~2 min wall-clock  (with --render)
50,000 steps ≈   8 min simulated ≈  ~15 min wall-clock  (with --render)
300,000 steps ≈  50 min simulated ≈  ~30 min wall-clock  (headless)
```

Each episode is 300 steps (3 seconds of sim). So 300k steps ≈ 1,000 episodes.

#### Recommended debug workflow

```bash
# 1. Confirm viewer opens and finger moves
python rl/train.py --render --steps 5000

# 2. Watch early learning — is reward going up?
python rl/train.py --render --steps 50000

# 3. Full training, no viewer, full speed
python rl/train.py --steps 300000
```

### 3. Visualize the workspace

Before or after training, see where the fingertip can actually reach:

```bash
# Run from anywhere — script resolves its own path
python rl/workspace_vis.py
python rl/workspace_vis.py --steps 10000     # denser plot
python rl/workspace_vis.py --color-by z      # color by height
python rl/workspace_vis.py --color-by time   # color by sweep order
```

The plot shows the real reachable arc plus the pen center and grasp contact
point from `grasp_point_generator.py` — so you can verify the pen sits
inside the fingertip workspace.

#### Why workspace_vis.py lives in `rl/` not `kinematics/`

The workspace shown is the **actual fingertip arc produced by MuJoCo** —
computed by running random actions through `FingerEnv`, not from the
kinematic equations. It is the workspace the RL policy trains on.
It belongs next to the thing it describes. The script uses
`Path(__file__).resolve().parent` to always find `env.py` and `config.py`
regardless of which folder you run it from.

### 4. Play interactively

```bash
python rl/play.py --model rl/weights/best.zip
```

The MuJoCo 3D viewer opens. Type a fingertip target in meters — the finger
moves to that position and **holds it** until you type a new target.
No auto-reset loop.

```
Waiting for target (finger is at home position)...

> 0.022 -0.035 0.100       ← set target (x y z in meters)
  Moving to target...
  dist=28.4 mm  tip=(18.1, -31.2, 96.4) mm
  dist=9.7 mm   tip=(20.8, -33.8, 99.1) mm
  dist=2.1 mm   tip=(21.9, -35.0, 100.1) mm
  Reached! dist=2.1 mm -- holding.

> r                         ← random target from real workspace
> q                         ← quit
```

**Hold behavior:** once the finger reaches the target (dist < 5 mm) or
times out (300 steps), the last motor commands are frozen — the finger
stays in position. Only a new target restarts the policy.

---

## Algorithm — SAC

SAC (Soft Actor-Critic) was chosen over PPO for this system because:

| Criterion | PPO | **SAC** |
|---|---|---|
| Sample efficiency | Low (on-policy) | **High (replay buffer)** |
| Continuous 2-DOF actions | OK | **Excellent** |
| Nonlinear tendon dynamics | Struggles | **Handles well** |
| Training without GPU | Slow | **Faster** |

All hyperparameters with rationale are in `config.py`.

---

## Observation and action spaces

### Observation (15 values)

| Index | Meaning | Unit |
|---|---|---|
| 0–8   | Joint positions: servo, motor, abd, prx×2, int×3, dis | rad |
| 9–11  | Fingertip world position (x, y, z) via `fingertip` site | m |
| 12–14 | Target position (x, y, z) | m |

The fingertip position is tracked via a MuJoCo `<site>` named `fingertip`
placed at the tip of the distal phalanx in `mjmodel.xml`. This is more
accurate than body origin (`xpos`) which points to the joint center.

### Action (2 values)

| Index | Actuator | Normalized | Real range |
|---|---|---|---|
| 0 | `joint_s_pos` — MG90S servo | [-1, 1] | [-0.3, 0.3] rad |
| 1 | `joint_m_pos` — JA12-N20 motor | [-1, 1] | [-0.3, 3.14] rad |

The 7 passive phalanx joints (prx×2, int×3, dis, abd) are **not commanded**.
They follow `joint_motor` automatically via the tendon equality constraints
in `mjmodel.xml`.

---

## Reward

Dense, minimal, interpretable:

```
reward = -distance(fingertip, target)     always on (meters)
       + 1.0                              bonus when dist < 5 mm
```

A random policy scores around **−19**. A well-trained policy scores near **−9**
at 100k steps and approaches **0** by 300k steps.

---

## Why targets are sampled from the real workspace

This is the most important design decision in `env.py`.

The finger is **underactuated** — 2 actuators drive 9 joints through tendons.
Its reachable workspace is a **thin arc in 3D**, not a rectangular bounding box.
If targets were sampled from a box, most would fall outside the arc and the
policy could never reach them. Reward would never improve and training would diverge.

**Solution (in `env.reset()`):**

```
1. Reset sim to home position
2. Run 40 random actions — record where the fingertip actually goes
3. Pick one of those visited positions as the target
   (+ 1 mm Gaussian noise for variety)
4. Reset sim to home again — episode starts from neutral
```

Step 4 is critical. Without the second reset, the episode would start from
wherever the random rollout left the finger — not from home.

Every target is reachable by construction. Use `workspace_vis.py` to see
the shape of this workspace before training.

---

## Key design decisions

**VecNormalize** — Observations are normalized online during training because
`qpos` (rad), `site_xpos` (m), and target (m) live on different scales.
Normalizer stats are saved alongside the model as `*_vecnorm.pkl`.
**Always load both files together** — the model without its normalizer
performs poorly.

**`mujoco.viewer` must be imported explicitly** — `import mujoco` does not
auto-load the viewer submodule. `env.py` imports `import mujoco.viewer`
explicitly. If you see `AttributeError: module 'mujoco' has no attribute 'viewer'`
this import is missing.

**Single callback** — `LogAndSaveCallback` handles logging, checkpointing,
and best-model tracking in one place.

**`set_target()`** — `env.py` exposes this method so `play.py` can inject
new targets at runtime without resetting the episode.

---

## TensorBoard metrics

| Metric | Meaning |
|---|---|
| `eval/mean_reward`  | Mean episode reward (last 50 episodes) |
| `eval/mean_dist_mm` | Mean fingertip–target distance in mm |
| `rollout/ep_rew_mean` | SB3 native episode reward |
| `rollout/ep_len_mean` | Mean episode length (steps) |

Reading the plots:
- `mean_reward` rising from −19 toward 0 = **training is working**
- `mean_dist_mm` falling from ~55 mm toward 5 mm = **finger getting closer**
- S-curve shape on reward = **normal SAC learning curve**
- `ep_len_mean` flat at 300 = finger hasn't converged yet (still exploring)

---

## Changing the workspace

To see where the fingertip actually goes, run:

```bash
python rl/workspace_vis.py --steps 10000
```

Targets are always sampled from the **real reachable arc** (not a config box)
so you do not need to tune `target_x_range` etc. Those config values are
kept as documentation of the approximate workspace bounds but are not used
for sampling.
