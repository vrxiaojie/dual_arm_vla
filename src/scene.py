from pathlib import Path

import mujoco

from src.robot_model import build_scene_xml


class TaskScene:
    """Owns the MuJoCo model/data and stable initial state."""

    def __init__(self, model_path: Path, robot_urdf_path: Path | None = None):
        if robot_urdf_path is None:
            robot_urdf_path = model_path.parents[1] / "robots" / "panda_arm.urdf"
        scene_xml = build_scene_xml(model_path, robot_urdf_path)
        self.model = mujoco.MjModel.from_xml_string(scene_xml.decode("utf-8"))
        # Panda links are gravity compensated, as on the real robot.  The rod
        # remains fully dynamic under gravity.
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            if name.startswith("left_") or name.startswith("right_"):
                self.model.body_gravcomp[body_id] = 1.0
        # Collision groups keep each serial chain's adjacent links from
        # physically fighting one another, while preserving inter-arm,
        # arm/rod and arm/table contacts for the safety monitor.
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("left_"):
                self.model.geom_contype[geom_id], self.model.geom_conaffinity[geom_id] = 1, 14
            elif name.startswith("right_"):
                self.model.geom_contype[geom_id], self.model.geom_conaffinity[geom_id] = 2, 13
            elif name == "rod_geom":
                self.model.geom_contype[geom_id], self.model.geom_conaffinity[geom_id] = 4, 11
            elif name == "table_top":
                self.model.geom_contype[geom_id], self.model.geom_conaffinity[geom_id] = 8, 7
        # Stiff joint servos are needed to track the coordinated Cartesian
        # waypoints under the mass of seven links.
        self.model.actuator_gainprm[:14, 0] = 650.0
        self.model.actuator_biasprm[:14, 1] = -650.0
        self.data = mujoco.MjData(self.model)
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        # Safe folded Panda-like pose; both grippers start open and clear.
        pose = [0.0, -0.55, 0.0, -1.75, 0.0, 1.20, 0.0]
        for side in ("left", "right"):
            for index, value in enumerate(pose, 1):
                joint = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_joint{index}"
                )
                self.data.qpos[self.model.jnt_qposadr[joint]] = value
            for index in (1, 2):
                joint = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{side}_finger_joint{index}"
                )
                self.data.qpos[self.model.jnt_qposadr[joint]] = 0.04
        self.data.ctrl[:7] = pose
        self.data.ctrl[7:14] = pose
        self.data.ctrl[14:] = 0.04
        mujoco.mj_forward(self.model, self.data)

    def body_position(self, name: str):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body].copy()
