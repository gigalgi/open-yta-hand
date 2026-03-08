import mujoco
import mujoco.viewer

model = mujoco.MjModel.from_xml_path("mjmodel.xml")  # include_MPL or your converted MJCF file
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()