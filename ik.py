import numpy as np
import mujoco


class InverseKinematics:
    def __init__(self, model: mujoco.MjModel, body_name: str, joint_ids: list, step_size: float = 0.5):
        self.model = model
        self.body_id = model.body(body_name).id
        self.joint_ids = joint_ids  # indices des articulations à contrôler
        self.step_size = step_size

        # buffers
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.err = np.zeros(6)

    def compute(self, data: mujoco.MjData, target_pos: np.ndarray, target_quat: np.ndarray = None):
        # 1. Calcul du Jacobien pour le body
        mujoco.mj_jacBody(self.model, data, self.jacp, self.jacr, self.body_id)

        # 2. Erreur de position
        self.err[:3] = target_pos - data.body(self.body_id).xpos

        # 3. Si orientation souhaitée donnée
        if target_quat is not None:
            # orientation actuelle
            cur_quat = data.body(self.body_id).xquat
            # calcule l’erreur quaternion
            conj = np.zeros(4)
            err_quat = np.zeros(4)
            mujoco.mju_negQuat(conj, cur_quat)
            mujoco.mju_mulQuat(err_quat, target_quat, conj)
            mujoco.mju_quat2Vel(self.err[3:], err_quat, 1.0)
        else:
            self.err[3:] = 0

        # 4. Construire le Jacobien complet (6 × nv)
        J = np.vstack((self.jacp, self.jacr))

        # 5. Calcul d’un changement de q : Δq = pseudo-inverse(J) * err
        dq = np.linalg.pinv(J) @ self.err

        # 6. Limiter ou filtrer Δq sur tes articulations contrôlées
        dq_ctrl = dq[self.joint_ids]

        # 7. Appliquer Δq à qpos
        for idx, joint_idx in enumerate(self.joint_ids):
            data.qpos[joint_idx] += self.step_size * dq_ctrl[idx]

        # 8. Appliquer la propagation
        mujoco.mj_forward(self.model, data)

        return dq_ctrl
