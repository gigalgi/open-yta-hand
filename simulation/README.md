# simulation/

Physics simulation files for the UMoBIC-Finger.
Two simulators, each with its own folder and model file.

```
simulation/
├── mujoco/
│   ├── mjmodel.xml        MuJoCo model — exact physics, ground truth
│   └── objects/pen.xml    pen object included by mjmodel.xml
│   └── meshes/            STL files shared by both simulators
└── isaac_sim/
    ├── finger_isaac.urdf  Isaac Sim model
    ├── isaac_play.py      standalone viewer — sliders, no ROS2
    └── isaac_bridge.py    ROS2 controlled — same topics as mujoco_bridge
```

---

## mujoco/

The ground truth model. Used by the RL training pipeline and the
ROS2 MuJoCo bridge.

`mjmodel.xml` defines:
- Exact rolling-contact joint kinematics via tendon equality constraints
- Position actuators for servo and motor (`joint_s_pos`, `joint_m_pos`)
- Contact capsules on each phalanx for pen interaction
- A `fingertip` site at the tip of the distal phalanx — used by
  `rl/env.py` and `ros2_ws/mujoco_bridge.py` to get exact fingertip
  world position without any FK computation

Run standalone (no ROS2):
```bash
python -m mujoco.viewer simulation/mujoco/mjmodel.xml
```

Run via ROS2:
```bash
ros2 launch umobic_finger sim.launch.py
```

Run via RL training:
```bash
python rl/train.py
python rl/play.py --model rl/weights/best.zip
```

---

## isaac_sim/

Isaac Sim simulation files. Two independent modes — use whichever
fits what you are doing.

### Standalone (no ROS2)

Quick visual testing. No build step. Just run.

```bash
python simulation/isaac_sim/isaac_play.py
```

A GUI panel opens with two sliders:

| Slider | Range | Effect |
|---|---|---|
| Servo | -0.3 to 0.3 rad | Abduction — side to side |
| Motor | -0.3 to 3.14 rad | Flexion — open to closed |

### ROS2 controlled

Isaac Sim runs the physics. All ROS2 nodes connect unchanged via
Isaac Sim's built-in ROS2 bridge extension.

One-time setup:
```
Isaac Sim → Window → Extensions → search "ROS2 Bridge" → enable + autoload
```

Run:
```bash
# Terminal 1
python simulation/isaac_sim/isaac_bridge.py

# Terminal 2 — identical to MuJoCo workflow
ros2 run umobic_finger finger_teleop.py
ros2 param set /finger_teleop motor 1.5
```

`finger_teleop.py` and `policy_node.py` work with both bridges.
The ROS2 nodes cannot tell which simulator is running.

---

## Why two simulators

| | MuJoCo | Isaac Sim |
|---|---|---|
| Model file | `mjmodel.xml` | `finger_isaac.urdf` |
| Physics | MuJoCo (exact tendons) | PhysX |
| Fingertip position | `<site>` tag | `fingertip_site` fixed link + USD FK |
| Mimic joints | tendon equality constraints | applied manually in Python |
| ROS2 bridge | `mujoco_bridge.py` (in ros2_ws) | `isaac_bridge.py` (here) |
| Use for | RL training, full pipeline, hardware prep | visual testing, demos |
| Requires | any machine | Isaac Sim + RTX GPU |

---

## Model files are not interchangeable

Each simulator has its own model file optimised for its own format
and capabilities. They share the same STL geometry from `meshes/`
but are otherwise independent.

```
mjmodel.xml           → loaded only by MuJoCo
finger_isaac.urdf     → loaded only by Isaac Sim
ros2_ws/urdf/finger.urdf  → loaded only by RViz / robot_state_publisher
```

Ground truth kinematics always live in `mjmodel.xml`.
The URDF files are approximations suitable for visualization.
