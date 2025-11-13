import mujoco
import mujoco.viewer

SCENE_PATH = "SO-ARM100/Simulation/SO101/scene.xml"
SO101_PATH = "SO-ARM100/Simulation/SO101/so101_new_calib.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
