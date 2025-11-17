import numpy as np
import mink
import mujoco
from mink.contrib import TeleopMocap
from loop_rate_limiters import RateLimiter


class InverseKinematics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.configuration = mink.Configuration(model)

        self.tasks = [
            end_effector_task := mink.FrameTask(
                frame_name="gripperframe",
                frame_type="site",
                position_cost=1.0,
                orientation_cost=1.0,
                lm_damping=1e-6,
            ),
            posture_task := mink.PostureTask(model, cost=1e-3),
        ]
        self.ee_task = end_effector_task

        self.limits = [
            mink.ConfigurationLimit(model=self.configuration.model),
        ]

        model.body("target").mocapid[0]

        # IK settings.
        self.solver = "daqp"
        self.pos_threshold = 1e-4
        self.ori_threshold = 1e-4
        self.max_iters = 20

        self.configuration.update(data.qpos)

        mujoco.mj_forward(model, data)

        posture_task.set_target_from_configuration(self.configuration)

        mink.move_mocap_to_frame(model, data, "target", "gripperframe", "site")

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData, key_callback: TeleopMocap, rate: RateLimiter):
        T_wt = mink.SE3.from_mocap_name(model, data, "target")
        self.ee_task.set_target(T_wt)

        key_callback.auto_key_move()

        for i in range(self.max_iters):
            vel = mink.solve_ik(self.configuration, self.tasks, rate.dt, self.solver, limits=self.limits)
            self.configuration.integrate_inplace(vel, rate.dt)
            err = self.ee_task.compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= self.pos_threshold
            ori_achieved = np.linalg.norm(err[3:]) <= self.ori_threshold
            if pos_achieved and ori_achieved:
                break

        data.ctrl = self.configuration.q
