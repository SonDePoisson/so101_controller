import os
import math
import time
import threading
import argparse
import mujoco
import mujoco.viewer
import numpy as np

from loop_rate_limiters import RateLimiter

from so101_driver.so101_driver import SO101Driver
from ik import InverseKinematics
from teleop import Teleop

SCENE_PATH = "SO-ARM100/Simulation/SO101/scene.xml"
SO101_PORT = os.getenv("SO101_PORT")
RATE = 200  # Hz


def mujoco_to_servo(angle_rad):
    return int((angle_rad / math.pi) * 2048 + 2048)


def get_body_pose(model: mujoco.MjModel, data: mujoco.MjData, body_name: str = "target"):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    pos = np.array(data.xpos[body_id], dtype=float)
    quat = np.array(data.xquat[body_id], dtype=float)
    return pos, quat


def control_loop(driver: SO101Driver, data: mujoco.MjData, stop_flag):
    while not stop_flag.is_set():
        for id in driver.servo_ids:
            target = mujoco_to_servo(data.qpos[id - 1])
            driver.move_servo(id, target, wait=False)
        time.sleep(1 / RATE)  # 60 Hz


def init_arg_parse():
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Launch SO101 Robot in Mujoco with real robot or just simulation",
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run only the Mujoco simulation without controlling the real robot",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        choices=["keyboard", "gamepad"],
        help="Controller to use (keyboard or gamepad)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = init_arg_parse()

    # Init MuJoCo #
    model = mujoco.MjModel.from_xml_path(SCENE_PATH)
    data = mujoco.MjData(model)

    # Init IK #
    ik = InverseKinematics(model, data)

    # Initialize Controller
    controller = None
    key_callback = None
    if args.controller == "keyboard":
        key_callback = Teleop(data)
    elif args.controller == "gamepad":
        pass

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

    rate = RateLimiter(frequency=RATE, warn=False)

    # Main Loop #
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            ik.compute(model, data, rate)

            if key_callback is not None:
                key_callback.auto_key_move()

            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()

    # Close #
    if driver:
        stop_flag.set()
        thread.join()
        driver.free_robot()
