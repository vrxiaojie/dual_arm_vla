import numpy as np
import mujoco


class GraspMonitor:
    def __init__(self, controller, scene):
        self.controller, self.scene = controller, scene

    def check(self):
        rod = self.scene.body_position("rod")
        box = getattr(self.scene, "task", "task1") == "task2"
        offset, height = (.42, .26) if box else (.42, 0)
        expected_left = rod + np.array([-offset, 0.0, height])
        expected_right = rod + np.array([offset, 0.0, height])
        near_left = np.linalg.norm(self.controller.left.position - expected_left) < 0.02
        near_right = np.linalg.norm(self.controller.right.position - expected_right) < 0.02
        contacts = {"left": set(), "right": set()}
        for index in range(self.scene.data.ncon):
            contact = self.scene.data.contact[index]
            a = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
            ) or ""
            b = mujoco.mj_id2name(
                self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
            ) or ""
            objects = {"box_handle_left", "box_handle_right"} if box else {"rod_geom"}
            if not objects.intersection({a, b}):
                continue
            other = b if a in objects else a
            for side in contacts:
                for finger in (1, 2):
                    if other == f"{side}_finger{finger}_geom":
                        contacts[side].add(finger)
        upright = abs(rod[2] - (.30 if box else .468)) < 0.10
        return (
            bool(near_left and len(contacts["left"]) == 2 and upright),
            bool(near_right and len(contacts["right"]) == 2 and upright),
        )
