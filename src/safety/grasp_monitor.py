import numpy as np
import mujoco


class GraspMonitor:
    def __init__(self, controller, scene):
        self.controller, self.scene = controller, scene

    def check(self):
        rod = self.scene.body_position("rod")
        expected_left = rod + np.array([-0.42, 0.0, 0.02])
        expected_right = rod + np.array([0.42, 0.0, 0.02])
        near_left = np.linalg.norm(self.controller.left.position - expected_left) < 0.11
        near_right = np.linalg.norm(self.controller.right.position - expected_right) < 0.11
        contacts = {"left": False, "right": False}
        for index in range(self.scene.data.ncon):
            contact = self.scene.data.contact[index]
            a = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            b = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            if "rod_geom" not in {a, b}:
                continue
            other = b if a == "rod_geom" else a
            for side in contacts:
                if other.startswith(side + "_") and (
                    "finger" in other or "hand" in other or "link7" in other
                ):
                    contacts[side] = True
        upright = abs(rod[2] - 0.468) < 0.10
        return (
            bool(near_left and contacts["left"] and upright),
            bool(near_right and contacts["right"] and upright),
        )
