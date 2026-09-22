import mujoco


class CollisionChecker:
    """Classifies MuJoCo contacts; expected grasp/support contacts are allowed."""

    def __init__(self, model, data):
        self.model, self.data = model, data
        self.last_reason = ""

    def _name(self, geom_id):
        return mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""

    @staticmethod
    def _side(name):
        return "left" if name.startswith("left_") else "right" if name.startswith("right_") else ""

    def dangerous(self, state: str) -> bool:
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            a, b = self._name(contact.geom1), self._name(contact.geom2)
            names = {a, b}
            sa, sb = self._side(a), self._side(b)
            if sa and sb and sa != sb:
                self.last_reason = f"inter-arm collision: {a} / {b}"
                return True
            if "rod_geom" in names:
                other = b if a == "rod_geom" else a
                if other == "table_top" and state in {"INIT", "PRE_GRASP", "APPROACH", "GRASP", "LIFT", "PLACE", "RELEASE", "RETREAT", "SUCCESS"}:
                    continue
                if "finger" in other or "hand_geom" in other:
                    continue
                # The terminal Panda link is part of the permitted gripper
                # envelope in this compact model.
                if other.startswith(("left_link6_", "right_link6_", "left_link7_", "right_link7_")):
                    continue
                if other and other != "floor":
                    self.last_reason = f"unexpected rod contact: {other}"
                    return True
            if "table_top" in names:
                other = b if a == "table_top" else a
                terminal_link = other.startswith(
                    ("left_link6_", "right_link6_", "left_link7_", "right_link7_")
                )
                gripper = "finger" in other or "hand_geom" in other
                if (
                    other.startswith(("left_arm_base_", "right_arm_base_"))
                    or other == "rod_geom"
                    # Pickup from a placed pose uses the same terminal-link
                    # support contacts as placement; proximal links stay guarded.
                    or ((terminal_link or gripper) and state in {
                        "APPROACH", "GRASP", "CHECK_GRASP", "LIFT",
                        "PLACE",
                    })
                ):
                    continue
                if self._side(other):
                    self.last_reason = f"arm-table collision: {other}"
                    return True
        return False
