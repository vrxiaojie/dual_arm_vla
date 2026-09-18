import unittest
import tempfile
from pathlib import Path

import mujoco

from src.config import TaskConfig
from src.scene import TaskScene
from src.state_machine import ALLOWED_TRANSITIONS, TaskState
from src.task import Task1Runner


class Task1StructureTest(unittest.TestCase):
    def setUp(self):
        self.scene = TaskScene(TaskConfig().model_path)

    def test_scene_has_required_bodies_and_regions(self):
        for body in ("left_arm_base", "right_arm_base", "rod", "table"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_BODY, body), 0
            )
        for geom in ("start_region", "target_region"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.scene.model, mujoco.mjtObj.mjOBJ_GEOM, geom), 0
            )

    def test_each_arm_has_seven_joints(self):
        names = [
            mujoco.mj_id2name(self.scene.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.scene.model.njnt)
        ]
        self.assertEqual(sum(n.startswith("left_joint") for n in names), 7)
        self.assertEqual(sum(n.startswith("right_joint") for n in names), 7)

    def test_success_path_is_explicit(self):
        state, seen = TaskState.INIT, [TaskState.INIT]
        while state is not TaskState.SUCCESS:
            state = ALLOWED_TRANSITIONS[state]
            seen.append(state)
        self.assertIn(TaskState.CHECK_GRASP, seen)
        self.assertIn(TaskState.RELEASE, seen)

    def test_headless_task_reaches_success(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TaskConfig(output_dir=Path(directory))
            report = Task1Runner(TaskScene(config.model_path), config).run()
        self.assertEqual(report["result"], "SUCCESS")
        self.assertTrue(report["left_grasp_success"])
        self.assertTrue(report["right_grasp_success"])
        self.assertFalse(report["collision_detected"])
        self.assertLess(report["final_rod_position_error_m"], 0.10)

    def test_single_side_grasp_failure_interlocks_both_arms(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TaskConfig(output_dir=Path(directory))
            runner = Task1Runner(TaskScene(config.model_path), config)
            runner.grasp.check = lambda: (False, True)
            report = runner.run()
        self.assertEqual(report["result"], "FAILURE")
        self.assertIn("left grasp", report["failure_reason"])
        self.assertEqual(report["state_history"][-1], "FAILURE_STOP")
        self.assertNotIn("LIFT", report["state_history"])


if __name__ == "__main__":
    unittest.main()
