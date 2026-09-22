import mujoco
import numpy as np

from src.planning.ik import cartesian_ik_step


class ArmController:
    def __init__(self, model, data, side: str, actuator_start: int):
        self.model, self.data, self.side = model, data, side
        self.actuator_ids = np.arange(actuator_start, actuator_start + 7)
        self.joint_ids = np.array([
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{i}")
            for i in range(1, 8)
        ])
        self.qpos_ids = model.jnt_qposadr[self.joint_ids]
        self.dof_ids = model.jnt_dofadr[self.joint_ids]
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{side}_hand")
        self.grip_ids = np.array([14, 15]) if side == "left" else np.array([16, 17])

    @property
    def position(self):
        return self.data.xpos[self.body_id].copy()

    def move_toward(self, target):
        dq, error = cartesian_ik_step(
            self.model, self.data, self.body_id, self.dof_ids, target
        )
        command = self.data.qpos[self.qpos_ids] + dq
        ranges = self.model.jnt_range[self.joint_ids]
        self.data.ctrl[self.actuator_ids] = np.clip(command, ranges[:, 0], ranges[:, 1])
        return error

    def solve_position(self, target, max_iterations=500, seed=None):
        """Solve a downward-facing tool pose without changing simulation state."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        # Default waypoints use a repeatable branch. Close-spaced retreat
        # waypoints explicitly pass the previous solution to stay continuous.
        scratch.qpos[self.qpos_ids] = (
            [0.0, -0.55, 0.0, -1.75, 0.0, 1.20, 0.0] if seed is None else seed
        )
        mujoco.mj_forward(self.model, scratch)
        ranges = self.model.jnt_range[self.joint_ids]
        # Mirrored bases use opposite tool yaw; both finger axes remain
        # perpendicular to the rod and both tool Z axes point down.
        desired = np.array([0., 1., 0., 0.]) if self.side == "left" else np.array([0., 0., 1., 0.])
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        rotation_error = np.zeros(3)
        for _ in range(max_iterations):
            mujoco.mj_jacBody(self.model, scratch, jacp, jacr, self.body_id)
            error = np.asarray(target) - scratch.xpos[self.body_id]
            mujoco.mju_subQuat(rotation_error, desired, scratch.xquat[self.body_id])
            # mju_subQuat returns a local rotation vector; jacr is world-frame.
            rotation_error = scratch.xmat[self.body_id].reshape(3, 3) @ rotation_error
            jac = np.vstack((jacp[:, self.dof_ids], 0.3 * jacr[:, self.dof_ids]))
            residual = np.concatenate((error, 0.3 * rotation_error))
            dq = jac.T @ np.linalg.solve(jac @ jac.T + 1e-4 * np.eye(6), residual)
            dq = np.clip(dq, -0.045, 0.045)
            scratch.qpos[self.qpos_ids] = np.clip(
                scratch.qpos[self.qpos_ids] + dq, ranges[:, 0], ranges[:, 1]
            )
            mujoco.mj_forward(self.model, scratch)
            if np.linalg.norm(error) < 2e-4 and np.linalg.norm(rotation_error) < .01:
                break
        final_error = np.linalg.norm(np.asarray(target) - scratch.xpos[self.body_id])
        if final_error > 0.03:
            raise RuntimeError(f"{self.side} IK target unreachable (error={final_error:.3f} m)")
        mujoco.mju_subQuat(rotation_error, desired, scratch.xquat[self.body_id])
        if np.linalg.norm(rotation_error) > 0.05:
            raise RuntimeError(f"{self.side} IK cannot maintain downward gripper orientation")
        return scratch.qpos[self.qpos_ids].copy()

    def command_joints(self, command):
        self.data.ctrl[self.actuator_ids] = command

    def set_gripper(self, opened: bool):
        # Inner pad faces are at +/- joint displacement. Close around the
        # rod diameter with a small preload, never to an empty zero-width gap.
        radius = float(self.model.geom("rod_geom").size[0])
        self.data.ctrl[self.grip_ids] = 0.04 if opened else max(0.0, radius - 0.002)

    def hold(self):
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_ids]
