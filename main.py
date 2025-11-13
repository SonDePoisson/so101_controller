import os
import math
import time
import threading
import mujoco
import mujoco.viewer
from so101_driver.so101_driver import SO101Driver

SCENE_PATH = "SO-ARM100/Simulation/SO101/scene.xml"
SO101_PORT = os.getenv("SO101_PORT")


def mujoco_to_servo(angle_rad):
    return int((angle_rad / math.pi) * 2048 + 2048)


def control_loop(driver: SO101Driver, data, stop_flag):
    while not stop_flag.is_set():
        for id in driver.servo_ids:
            target = mujoco_to_servo(data.qpos[id - 1])
            driver.move_servo(id, target, wait=False)
        time.sleep(1 / 60)  # Hz


if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)
    driver = SO101Driver(SO101_PORT)
    print("Detected servo IDs:", driver.servo_ids)

    stop_flag = threading.Event()
    thread = threading.Thread(target=control_loop, args=(driver, data, stop_flag))
    thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()

    stop_flag.set()
    thread.join()
    driver.free_robot()
