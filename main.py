import os
import math
import time
import threading
import argparse
import mujoco
import mujoco.viewer
import numpy as np

from so101_driver.so101_driver import SO101Driver
# from ik import InverseKinematics

import mink
from mink.contrib import TeleopMocap
from loop_rate_limiters import RateLimiter

SCENE_PATH = "SO-ARM100/Simulation/SO101/scene.xml"
SO101_PORT = os.getenv("SO101_PORT")


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
    configuration = mink.Configuration(model)
    tasks = [
        end_effector_task := mink.FrameTask(
            frame_name="gripperframe",
            frame_type="site",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1e-6,
        ),
        posture_task := mink.PostureTask(model, cost=1e-3),
    ]

    limits = [
        mink.ConfigurationLimit(model=configuration.model),
    ]

    mid = model.body("target").mocapid[0]

    # IK settings.
    solver = "daqp"
    pos_threshold = 1e-4
    ori_threshold = 1e-4
    max_iters = 20

    # Initialize key_callback function.
    key_callback = TeleopMocap(data)

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
    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        configuration.update(data.qpos)
        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(configuration)

        # Initialize the mocap target at the end-effector site.
        mink.move_mocap_to_frame(model, data, "target", "attachment_site", "site")

        rate = RateLimiter(frequency=200.0, warn=False)
        while viewer.is_running():
            T_wt = mink.SE3.from_mocap_name(model, data, "target")
            end_effector_task.set_target(T_wt)

            # Continuously check for autonomous key movement.
            key_callback.auto_key_move()

            # Compute velocity and integrate into the next configuration.
            for i in range(max_iters):
                vel = mink.solve_ik(configuration, tasks, rate.dt, solver, limits=limits)
                configuration.integrate_inplace(vel, rate.dt)
                err = end_effector_task.compute_error(configuration)
                pos_achieved = np.linalg.norm(err[:3]) <= pos_threshold
                ori_achieved = np.linalg.norm(err[3:]) <= ori_threshold
                if pos_achieved and ori_achieved:
                    break

            data.ctrl = configuration.q
            mujoco.mj_step(model, data)
            viewer.sync()
            rate.sleep()

    # Close #
    if driver:
        stop_flag.set()
        thread.join()
        driver.free_robot()
