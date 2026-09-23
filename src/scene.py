from pathlib import Path

import mujoco
import numpy as np

from src.config import HANDOVER_SOURCE, HANDOVER_SIZE
from src.robot_model import build_scene_xml


class TaskScene:
    """Owns the MuJoCo model/data and stable initial state."""

    def __init__(self, model_path: Path, robot_urdf_path: Path | None = None):
        if robot_urdf_path is None:
            robot_urdf_path = model_path.parents[1] / "robots" / "panda_arm.urdf"
        scene_xml = build_scene_xml(model_path, robot_urdf_path)
        self.model = mujoco.MjModel.from_xml_string(scene_xml.decode("utf-8"))
        # Collision groups keep each serial chain's adjacent links from
        # physically fighting one another, while preserving inter-arm,
        # arm/rod and arm/table contacts for the safety monitor.
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.endswith("_visual"):
                continue
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
        self._object_inertia = self.model.body("rod").inertia.copy()
        self._object_mass = float(self.model.body("rod").mass[0])
        self.task = "task1"
        self._rgba = self.model.geom_rgba.copy()
        self._masks = (self.model.geom_contype.copy(), self.model.geom_conaffinity.copy())
        self.switch_task("task1")

    def switch_task(self, task):
        if task not in ("task1", "task2", "task3"):
            raise ValueError("Unknown task")
        self.task = task
        self.model.body("rod").inertia[:] = [.06, .06, .06] if task == "task2" else self._object_inertia
        self.model.body("rod").mass = self._object_mass
        if task == "task3":
            mass = 1.0
            x, y, z = HANDOVER_SIZE
            self.model.body("rod").mass = mass
            self.model.body("rod").inertia[:] = mass / 12 * np.array([y*y+z*z, x*x+z*z, x*x+y*y])
        self.model.geom_rgba[:] = self._rgba
        self.model.geom_contype[:] = self._masks[0]
        self.model.geom_conaffinity[:] = self._masks[1]
        for i in range(self.model.ngeom):
            name = self.model.geom(i).name
            hidden = (name.startswith("handover_") if task != "task3" else
                      name.startswith("box_") or name in {"rod_geom", "start_region", "target_region"})
            if task == "task1":
                hidden = hidden or name.startswith("box_")
            elif task == "task2":
                hidden = hidden or name in {"rod_geom", "table_top", "start_region", "target_region"}
            if hidden:
                self.model.geom_rgba[i, 3] = 0
                self.model.geom_contype[i] = self.model.geom_conaffinity[i] = 0
        floor = self.model.geom("floor").id
        self.model.geom_contype[floor] = 8 if task == "task2" else 0
        self.model.geom_conaffinity[floor] = 7 if task == "task2" else 0
        # MuJoCo caches collision masks per body at compilation.
        self.model.body_contype[:] = 0
        self.model.body_conaffinity[:] = 0
        for i in range(self.model.ngeom):
            body = self.model.geom_bodyid[i]
            self.model.body_contype[body] |= self.model.geom_contype[i]
            self.model.body_conaffinity[body] |= self.model.geom_conaffinity[i]
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
        if self.task == "task2":
            adr = self.model.joint("rod_free").qposadr[0]
            self.data.qpos[adr:adr+3] = [0, 0, .30]
        if self.task == "task3":
            adr = self.model.joint("rod_free").qposadr[0]
            self.data.qpos[adr:adr+7] = [*HANDOVER_SOURCE, 1, 0, 0, 0]
        mujoco.mj_forward(self.model, self.data)

    def body_position(self, name: str):
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        return self.data.xpos[body].copy()
