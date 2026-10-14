import json
import time
from contextlib import nullcontext
from datetime import datetime

import mujoco
import numpy as np

from src.config import (HANDOVER_SIZE, HANDOVER_HANDLE_OFFSET, HANDOVER_HANDLE_SIZE,
                        HANDOVER_REGION_HALF_SIZE, ROLES, TaskCommand)
from src.controllers import DualArmController
from src.metrics import Metrics
from src.planning import smoothstep
from src.safety import CollisionChecker, GraspMonitor
from src.state_machine import StateMachine, TaskState


class TaskRunner:
    def __init__(self, scene, config, viewer=None, command=None):
        self.scene, self.config, self.viewer = scene, config, viewer
        self.controller = DualArmController(scene.model, scene.data)
        self.collision = CollisionChecker(scene.model, scene.data)
        self.grasp = GraspMonitor(self.controller, scene)
        self.metrics = Metrics()
        self.machine = StateMachine()
        self.command = command or TaskCommand()
        self.failure_reason = ""
        self.left_grasp = self.right_grasp = False
        self.collision_detected = False

    def _viewer_lock(self):
        """Return the passive viewer lock, or a no-op for headless runs."""
        return self.viewer.lock() if self.viewer is not None else nullcontext()

    def _step(self, sync_error=0.0, check_collision=True):
        if self.viewer is not None and not self.viewer.is_running():
            self._fail("viewer closed by user")
            return False

        with self._viewer_lock():
            mujoco.mj_step(self.scene.model, self.scene.data)
            self.metrics.sample(self.scene, sync_error)
            if check_collision and self.collision.dangerous(self.machine.state.name):
                self.collision_detected = True
                self._fail(self.collision.last_reason)
                return False

        if self.viewer is not None:
            self.viewer.sync()
        if self.config.realtime:
            time.sleep(self.scene.model.opt.timestep)
        return True

    def _move_segment(self, left_end, right_end, steps=300):
        left_start_pos = self.controller.left.position
        right_start_pos = self.controller.right.position
        left_start_q = self.scene.data.qpos[self.controller.left.qpos_ids].copy()
        right_start_q = self.scene.data.qpos[self.controller.right.qpos_ids].copy()
        try:
            left_end_q = self.controller.left.solve_position(left_end)
            right_end_q = self.controller.right.solve_position(right_end)
        except RuntimeError as exc:
            self._fail(str(exc))
            return False
        for i in range(steps):
            fraction = (i + 1) / steps
            left_reference = smoothstep(left_start_pos, left_end, fraction)
            right_reference = smoothstep(right_start_pos, right_end, fraction)
            with self._viewer_lock():
                self.controller.left.command_joints(
                    smoothstep(left_start_q, left_end_q, fraction)
                )
                self.controller.right.command_joints(
                    smoothstep(right_start_q, right_end_q, fraction)
                )
                left_error = np.linalg.norm(
                    self.controller.left.position - left_reference
                )
                right_error = np.linalg.norm(
                    self.controller.right.position - right_reference
                )
            sync = abs(left_error - right_error)
            if not self._step(sync):
                return False
        # Permit a small settling/convergence window at the exact target.
        for _ in range(100):
            with self._viewer_lock():
                left_error = np.linalg.norm(self.controller.left.position - left_end)
                right_error = np.linalg.norm(self.controller.right.position - right_end)
            sync = abs(left_error - right_error)
            if not self._step(sync):
                return False
        return True

    def _fail(self, reason):
        self.failure_reason = reason
        self.controller.stop_both_arms()
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.fail()

    def _retreat_segment(self, left_end, right_end, steps=300):
        """Follow a Cartesian path on the current IK branch after release."""
        arms = (self.controller.left, self.controller.right)
        starts = [arm.position for arm in arms]
        seeds = [self.scene.data.qpos[arm.qpos_ids].copy() for arm in arms]
        ends = (left_end, right_end)
        # Short Cartesian waypoints avoid the large joint-space arc produced
        # by switching back to the folded IK seed at table height.
        for index in range(0, steps, 10):
            batch_steps = min(10, steps - index)
            fraction = (index + batch_steps) / steps
            references = [smoothstep(start, end, fraction) for start, end in zip(starts, ends)]
            try:
                targets = [arm.solve_position(ref, seed=seed)
                           for arm, ref, seed in zip(arms, references, seeds)]
            except RuntimeError as exc:
                self._fail(str(exc))
                return False
            for j in range(batch_steps):
                with self._viewer_lock():
                    for arm, seed, target in zip(arms, seeds, targets):
                        arm.command_joints(smoothstep(seed, target, (j + 1) / batch_steps))
                if not self._step():
                    return False
            seeds = targets
        for _ in range(100):
            if not self._step():
                return False
        return True

    def replan(self, failure_reason):
        """Extension point for a future VLA/regrasp planner."""
        raise NotImplementedError(f"Replanning not implemented: {failure_reason}")

    def run(self):
        if self.scene.task == "task3":
            from src.handover import run_handover
            return run_handover(self)
        # INIT: let the rod settle and validate that both hands are initially clear.
        for _ in range(100):
            if not self._step(check_collision=False):
                break
        rod = self.scene.body_position("rod")
        if min(
            np.linalg.norm(self.controller.left.position - rod),
            np.linalg.norm(self.controller.right.position - rod),
        ) < 0.15:
            self._fail("an arm initially contacts the rod")

        # The tool frame sits between the fingertips. Hidden contact proxies
        # provide the grasp forces while the Panda meshes provide the visuals.
        box = self.scene.task == "task2"
        grasp_z = rod[2] + (.26 if box else 0)
        left_grasp_target = np.array([rod[0] - self.config.grasp_offset, rod[1], grasp_z])
        right_grasp_target = np.array([rod[0] + self.config.grasp_offset, rod[1], grasp_z])
        place_shift = self.command.target_position - np.array([rod[0], rod[1], .30 if box else .468])
        # Descend vertically so contact cannot shove the free rod toward the
        # destination before the grasp has been verified.
        pre_left = left_grasp_target + np.array([0.0, 0.0, self.config.pregrasp_height])
        pre_right = right_grasp_target + np.array([0.0, 0.0, self.config.pregrasp_height])

        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # PRE_GRASP
            self._move_segment(pre_left, pre_right, 380)
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # APPROACH
            self._move_segment(left_grasp_target, right_grasp_target, 260)
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # GRASP
            with self._viewer_lock():
                self.controller.set_grippers(False)
            for _ in range(250):
                if not self._step():
                    break
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # CHECK_GRASP
            self.left_grasp, self.right_grasp = self.grasp.check()
            if not self.left_grasp or not self.right_grasp:
                failed = "left" if not self.left_grasp else "right"
                if not self.left_grasp and not self.right_grasp:
                    failed = "left and right"
                self._fail(f"{failed} grasp verification failed")
            else:
                with self._viewer_lock():
                    self.controller.attach_rod(self.config.grasp_offset, .26 if box else 0)

        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # LIFT
            lift = np.array([0.0, 0.0, self.config.carry_height])
            self._move_segment(left_grasp_target + lift, right_grasp_target + lift, 300)
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # TRANSPORT
            transport = place_shift + np.array([0.0, 0.0, self.config.carry_height])
            self._move_segment(left_grasp_target + transport, right_grasp_target + transport, 420)
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # PLACE
            self._move_segment(left_grasp_target + place_shift, right_grasp_target + place_shift, 320)
            for _ in range(80):
                if not self._step():
                    break
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # RELEASE
            with self._viewer_lock():
                self.controller.left.hold()
                self.controller.right.hold()
                self.controller.disable_rod_contacts()
                self.controller.detach_rod()
                self.controller.set_grippers(True)
            for _ in range(180):
                if not self._step():
                    break
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # RETREAT
            left_start = self.controller.left.position
            right_start = self.controller.right.position
            up = np.array([0.0, 0.0, 0.22])
            self._retreat_segment(left_start + up, right_start + up)
            # Retreat outward from the rod.  With the mirrored base
            # orientations, moving both wrists in the same world-X direction
            # would make one gripper sweep back across the placed rod.
            left_retreat = up + np.array([-0.08, 0.08, 0.0])
            right_retreat = up + np.array([0.08, 0.08, 0.0])
            if self.machine.state is not TaskState.FAILURE_STOP:
                self._retreat_segment(
                    left_start + left_retreat,
                    right_start + right_retreat,
                    260,
                )
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()

        with self._viewer_lock():
            self.controller.restore_arm_collisions()
        return self._finish()

    def _finish(self):
        report = self.metrics.summary(self.scene, self.command.target_position)
        numerical_success = (
            report["final_rod_position_error_m"] < self.config.position_tolerance
            and report["final_rod_orientation_error_deg"] < self.config.orientation_tolerance_deg
        )
        if self.scene.task == "task2":
            rotation = self.scene.data.xmat[self.scene.model.body("rod").id].reshape(3, 3)
            extent = .30 * np.abs(rotation).sum(axis=1)
            delta = np.abs(self.scene.body_position("rod") - self.command.target_position)
            inside = bool(np.all(delta[:2] + extent[:2] < .334))
            report["inside_target_square"] = inside
            numerical_success = numerical_success and inside
        if self.scene.task == "task3":
            rotation = self.scene.data.xmat[self.scene.model.body("rod").id].reshape(3, 3)
            footprint = HANDOVER_SIZE / 2
            footprint[0] = max(footprint[0], HANDOVER_HANDLE_OFFSET + HANDOVER_HANDLE_SIZE[0] / 2)
            extent = np.abs(rotation[:2, :]) @ footprint
            delta = np.abs(self.scene.body_position("rod")[:2] - self.command.target_position[:2])
            inside = bool(np.all(delta + extent < HANDOVER_REGION_HALF_SIZE))
            report["inside_target_region"] = inside
            numerical_success = numerical_success and inside
        if self.machine.state is TaskState.SUCCESS and not numerical_success:
            self._fail("final object pose outside tolerance")
        if self.scene.task == "task2":
            report = {key.replace("rod", "box"): value for key, value in report.items()}
        if self.scene.task == "task3":
            report = {key.replace("rod", "object"): value for key, value in report.items()}
        report.update({
            "task": "task2_box_transport" if self.scene.task == "task2" else "task1_rod_transport",
            "command": self.command.direction,
            "target_region": "floor_square" if self.scene.task == "task2" else ("green" if self.command.direction == 1 else "blue"),
            "result": "SUCCESS" if self.machine.state is TaskState.SUCCESS else "FAILURE",
            "failure_reason": self.failure_reason,
            "collision_detected": self.collision_detected,
            "left_grasp_success": self.left_grasp,
            "right_grasp_success": self.right_grasp,
            "roles": {"left_arm": "box_left_handle_grasper", "right_arm": "box_right_handle_grasper"} if self.scene.task == "task2" else ROLES,
            "state_history": self.machine.history,
        })
        if self.scene.task == "task3":
            report.update(task="task3_handover", target_region="handover_target",
                          roles={"left_arm": "left_long_edge_grasper", "right_arm": "right_long_edge_grasper"})
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.config.output_dir / f"{self.scene.task}_{stamp}.json"
        if self.config.capture_images or self.config.vision_env_path is not None:
            from src.vision import inspect_final_state
            report["visual_evaluation"] = inspect_final_state(
                self.scene, self.command, path.with_suffix(''), self.config.vision_env_path,
                capture_lock=self._viewer_lock())
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["log_path"] = str(path)
        return report


# Backward compatibility for task1 integrations and regression tests.
Task1Runner = TaskRunner
