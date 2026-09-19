import mujoco
import numpy as np

from .arm_controller import ArmController


class DualArmController:
    def __init__(self, model, data):
        self.model, self.data = model, data
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

    def attach_rod(self):
        """Enable two endpoint constraints after bilateral grasp validation."""
        # Connect layout stores the body1-local (hand) anchor followed by the
        # body2-local (rod) anchor.  Pin each hand origin to its rod endpoint.
        for weld_id, endpoint in zip(self.weld_ids, (-0.42, 0.42)):
            self.model.eq_data[weld_id, :3] = 0.0
            self.model.eq_data[weld_id, 3:6] = (endpoint, 0.0, 0.0)
        self.data.eq_active[self.weld_ids] = 1
        mujoco.mj_forward(self.model, self.data)

    def detach_rod(self):
        self.data.eq_active[self.weld_ids] = 0

    def disable_arm_collisions(self):
        """Prevent the open grippers from pushing the placed rod during retreat."""
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith(("left_", "right_")):
                self.model.geom_contype[geom_id] = 0
                self.model.geom_conaffinity[geom_id] = 0
