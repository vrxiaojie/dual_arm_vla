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

    def solve_position(self, target, max_iterations=500):
        """Solve a Cartesian waypoint off-line without changing simulation state."""
        scratch = mujoco.MjData(self.model)
        scratch.qpos[:] = self.data.qpos
        # Use a repeatable IK branch, including when picking up after retreat.
        # Seeding from the last retreat can fold the wrist into the tabletop.
        scratch.qpos[self.qpos_ids] = [0.0, -0.55, 0.0, -1.75, 0.0, 1.20, 0.0]
        mujoco.mj_forward(self.model, scratch)
        ranges = self.model.jnt_range[self.joint_ids]
        for _ in range(max_iterations):
            dq, error = cartesian_ik_step(
                self.model, scratch, self.body_id, self.dof_ids, target, damping=1e-3
            )
            scratch.qpos[self.qpos_ids] = np.clip(
                scratch.qpos[self.qpos_ids] + dq, ranges[:, 0], ranges[:, 1]
            )
            mujoco.mj_forward(self.model, scratch)
            if np.linalg.norm(error) < 2e-4:
                break
        final_error = np.linalg.norm(np.asarray(target) - scratch.xpos[self.body_id])
        if final_error > 0.03:
            raise RuntimeError(f"{self.side} IK target unreachable (error={final_error:.3f} m)")
        return scratch.qpos[self.qpos_ids].copy()

    def command_joints(self, command):
        self.data.ctrl[self.actuator_ids] = command

    def set_gripper(self, opened: bool):
        self.data.ctrl[self.grip_ids] = 0.04 if opened else 0.0

    def hold(self):
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_ids]
