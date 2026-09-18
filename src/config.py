from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class TaskConfig:
    model_path: Path = ROOT / "models" / "task1_scene.xml"
    output_dir: Path = ROOT / "outputs" / "logs"
    target_rod_position: np.ndarray = field(
        default_factory=lambda: np.array([0.0, -0.42, 0.468])
    )
    rod_half_length: float = 0.48
    grasp_offset: float = 0.42
    pregrasp_height: float = 0.18
    carry_height: float = 0.35
    position_tolerance: float = 0.10
    orientation_tolerance_deg: float = 10.0
    sync_tolerance: float = 0.05
    realtime: bool = False


@dataclass(frozen=True)
class TaskCommand:
    object: str = "rod"
    left_role: str = "grasp_left_end"
    right_role: str = "grasp_right_end"
    target: str = "target_region"
    sequence: tuple[str, ...] = (
        "approach", "grasp", "lift", "transport", "place", "release"
    )


ROLES = {
    "left_arm": "rod_left_grasper",
    "right_arm": "rod_right_grasper",
}
