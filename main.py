import os
import math
import time
import threading
import argparse
import mujoco
import mujoco.viewer
import numpy as np

from so101_driver.so101_driver import SO101Driver
from ik import InverseKinematics

SCENE_PATH = "SO-ARM100/Simulation/SO101/scene.xml"
SO101_PORT = os.getenv("SO101_PORT")


def mujoco_to_servo(angle_rad):
    return int((angle_rad / math.pi) * 2048 + 2048)


def control_loop(driver: SO101Driver, data: mujoco.MjData, stop_flag):
    while not stop_flag.is_set():
        for id in driver.servo_ids:
            target = mujoco_to_servo(data.qpos[id - 1])
            driver.move_servo(id, target, wait=False)
        time.sleep(1 / 60)  # 60 Hz


if __name__ == "__main__":
    # Argparse #
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Launch SO101 Robot in Mujoco with real robot or just simulation",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run only the Mujoco simulation without controlling the real robot",
    )
    args = parser.parse_args()

    # Init MuJoCo #
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Init IK #
    target_pos = np.array([0.0, 0.0, 0.5])
    target_quat = None
    joint_ids = [0, 1, 2, 3, 4, 5]
    ik_solver = InverseKinematics(model, body_name="gripper", joint_ids=joint_ids, step_size=0.1)

    # Choose Simulation or Real Robot #
    driver = None
    if not args.simulation:
        driver = SO101Driver(SO101_PORT)
        print("Detected servo IDs:", driver.servo_ids)
        stop_flag = threading.Event()
        thread = threading.Thread(target=control_loop, args=(driver, data, stop_flag))
        thread.start()
    else:
        print("So101 Simulation")
        stop_flag = None
        thread = None

    # Main Loop #
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            ik_solver.compute(data, target_pos, target_quat)

            mujoco.mj_step(model, data)
            viewer.sync()

    # Close #
    if driver:
        stop_flag.set()
        thread.join()
        driver.free_robot()
