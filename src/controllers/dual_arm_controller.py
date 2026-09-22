import mujoco
import numpy as np

from .arm_controller import ArmController


class DualArmController:
    def __init__(self, model, data):
        self.model, self.data = model, data
        self._collision_masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
        self.left = ArmController(model, data, "left", 0)
        self.right = ArmController(model, data, "right", 7)
        self.weld_ids = [mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{side}_grasp_weld"
        ) for side in ("left", "right")]

    def move_both(self, left_target, right_target):
        left_error = self.left.move_toward(left_target)
        right_error = self.right.move_toward(right_target)
        sync_error = abs(np.linalg.norm(left_error) - np.linalg.norm(right_error))
        return float(sync_error)

    def set_grippers(self, opened: bool):
        self.left.set_gripper(opened)
        self.right.set_gripper(opened)

    def stop_both_arms(self):
        self.left.hold()
        self.right.hold()
        self.data.qvel[:] = 0.0

    def attach_rod(self, offset=.42, height=0.0):
        """Enable two endpoint constraints after bilateral grasp validation."""
        # Connect layout stores the body1-local (hand) anchor followed by the
        # body2-local (rod) anchor. Preserve the verified grasp pose instead
        # of snapping the rod to the hand origin when constraints activate.
        rod_id = self.model.body("rod").id
        rod_rotation = self.data.xmat[rod_id].reshape(3, 3)
        for weld_id, endpoint in zip(self.weld_ids, (-offset, offset)):
            hand_id = self.model.eq_obj1id[weld_id]
            point = self.data.xpos[rod_id] + rod_rotation @ np.array([endpoint, 0.0, height])
            self.model.eq_data[weld_id, :3] = (
                self.data.xmat[hand_id].reshape(3, 3).T @ (point - self.data.xpos[hand_id])
            )
            self.model.eq_data[weld_id, 3:6] = (endpoint, 0.0, height)
        if height:
            weld = self.model.equality("box_orientation_grasp").id
            hand_id = self.left.body_id
            self.model.eq_data[weld, 3:6] = self.data.xmat[hand_id].reshape(3, 3).T @ (
                self.data.xpos[rod_id] - self.data.xpos[hand_id])
            inverse = self.data.xquat[hand_id].copy()
            inverse[1:] *= -1
            quat = np.zeros(4)
            mujoco.mju_mulQuat(quat, inverse, self.data.xquat[rod_id])
            self.model.eq_data[weld, 6:10] = quat
            self.data.eq_active[weld] = 1
        self.data.eq_active[self.weld_ids] = 1
        mujoco.mj_forward(self.model, self.data)

    def detach_rod(self):
        self.data.eq_active[self.weld_ids] = 0
        self.data.eq_active[self.model.equality("box_orientation_grasp").id] = 0

    def disable_rod_contacts(self):
        """Isolate the released rod while preserving arm/table and arm/arm contact."""
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(("left_", "right_")):
                self.model.geom_conaffinity[geom_id] &= ~4
            elif name == "rod_geom" or name.startswith("box_"):
                self.model.geom_conaffinity[geom_id] &= ~3

    def restore_arm_collisions(self):
        self.model.geom_contype[:] = self._collision_masks[0]
        self.model.geom_conaffinity[:] = self._collision_masks[1]
