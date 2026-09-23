from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class Metrics:
    sync_errors: list[float] = field(default_factory=list)
    max_rod_tilt_deg: float = 0.0

    def sample(self, scene, sync_error=0.0):
        self.sync_errors.append(float(sync_error))
        rod_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "rod")
        rod_axis = scene.data.xmat[rod_id].reshape(3, 3)[:, 0]
        tilt = np.degrees(np.arccos(np.clip(abs(rod_axis[0]), 0.0, 1.0)))
        if scene.task in ("task2", "task3"):
            quat = scene.data.xquat[rod_id]
            tilt = np.degrees(2 * np.arccos(np.clip(abs(quat[0]), 0, 1)))
        self.max_rod_tilt_deg = max(self.max_rod_tilt_deg, float(tilt))

    def summary(self, scene, target):
        rod_id = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, "rod")
        position = scene.data.xpos[rod_id]
        axis = scene.data.xmat[rod_id].reshape(3, 3)[:, 0]
        orientation = np.degrees(np.arccos(np.clip(abs(axis[0]), 0.0, 1.0)))
        if scene.task in ("task2", "task3"):
            orientation = np.degrees(2 * np.arccos(np.clip(abs(scene.data.xquat[rod_id, 0]), 0, 1)))
        values = np.asarray(self.sync_errors or [0.0])
        return {
            "max_synchronization_error_m": float(values.max()),
            "mean_synchronization_error_m": float(values.mean()),
            "final_rod_position_error_m": float(np.linalg.norm(position - target)),
            "final_rod_orientation_error_deg": float(orientation),
            "max_rod_tilt_deg": self.max_rod_tilt_deg,
        }

