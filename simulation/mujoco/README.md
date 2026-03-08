# simulation/mujoco/

MuJoCo simulation of the UMoBIC-Finger.

```
mujoco/
├── mjmodel.xml        main model file
├── objects/pen.xml    pen object included by mjmodel.xml
└── meshes/            STL files for all finger links
```

---

## Run standalone

```bash
conda activate mujoco-env
python -m mujoco.viewer simulation/mujoco/mjmodel.xml
```

The viewer opens with the finger at home position.
Click and drag to rotate. Scroll to zoom.
Double-click a body to select it and see its properties.

---

## What mjmodel.xml defines

**Kinematics** — the finger has 2 actuated inputs and 9 joints.
The 7 phalanx joints are coupled to servo and motor via tendon
equality constraints. These are exact rolling-contact kinematics,
not the linear approximation used in the URDF files.

**Actuators** — two position actuators with gain kp = 500:
- `joint_s_pos` — drives `joint_servo` (abduction, MG90S)
- `joint_m_pos` — drives `joint_motor` (flexion, JA12-N20)

**Fingertip site** — a site named `fingertip` is attached to the
distal phalanx at `pos="0 0 -0.012"`. MuJoCo updates its world
position every physics step. Used by the RL environment and the
ROS2 bridge to get exact fingertip position without any FK computation.

**Physics** — timestep 0.002 s (500 Hz). Damping and stiffness tuned
for stable position control at the actuator gains used.

---

## Actuator ranges

| Actuator | Joint | Range | Physical meaning |
|---|---|---|---|
| `joint_s_pos` | `joint_servo` | -0.3 to 0.3 rad | Abduction L/R |
| `joint_m_pos` | `joint_motor` | -0.3 to 3.14 rad | Open to fully closed |

---

## Tendon gear ratios

MuJoCo enforces these via equality constraints every physics step:

```
joint0_abd  =  0.98   × servo
joint1_prx  =  0.2222 × motor
joint2_prx  =  0.2222 × motor
joint3_int  =  0.2222 × motor
joint4_int  =  0.2222 × motor
joint5_int  = -0.2222 × motor   (cam-follower reversal)
joint6_dis  =  0.5333 × motor
```

---

## Used by

```
rl/env.py              loads mjmodel.xml directly for RL training
rl/train.py            trains SAC policy against this model
rl/play.py             runs trained policy in the viewer
ros2_ws/mujoco_bridge  loads mjmodel.xml for ROS2 control
```
