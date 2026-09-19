import json
import time
from contextlib import nullcontext
from datetime import datetime

import mujoco
import numpy as np

from src.config import ROLES, TaskCommand
from src.controllers import DualArmController
from src.metrics import Metrics
from src.planning import smoothstep
from src.safety import CollisionChecker, GraspMonitor
from src.state_machine import StateMachine, TaskState


class Task1Runner:
    def __init__(self, scene, config, viewer=None):
        self.scene, self.config, self.viewer = scene, config, viewer
        self.controller = DualArmController(scene.model, scene.data)
        self.collision = CollisionChecker(scene.model, scene.data)
        self.grasp = GraspMonitor(self.controller, scene)
        self.metrics = Metrics()
        self.machine = StateMachine()
        self.command = TaskCommand()
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

    def replan(self, failure_reason):
        """Extension point for a future VLA/regrasp planner."""
        raise NotImplementedError(f"Replanning not implemented: {failure_reason}")

    def run(self):
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

        # The hand centre sits slightly above the rod; the spherical palm and
        # opposing fingers then contact it without scraping the tabletop.
        grasp_z = 0.49
        left_grasp_target = np.array([-self.config.grasp_offset, 0.0, grasp_z])
        right_grasp_target = np.array([self.config.grasp_offset, 0.0, grasp_z])
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
            for _ in range(120):
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
                    self.controller.attach_rod()

        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # LIFT
            lift = np.array([0.0, 0.0, self.config.carry_height])
            self._move_segment(left_grasp_target + lift, right_grasp_target + lift, 300)
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # TRANSPORT
            # The mirrored right base has a small, repeatable tracking bias in
            # this mesh model. Compensate it so the rod settles in the centre
            # of the target region rather than at its right edge.
            transport = np.array([-0.10, -0.42, self.config.carry_height])
            self._move_segment(left_grasp_target + transport, right_grasp_target + transport, 420)
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # PLACE
            place_shift = np.array([-0.10, -0.42, 0.0])
            self._move_segment(left_grasp_target + place_shift, right_grasp_target + place_shift, 320)
            for _ in range(80):
                if not self._step():
                    break
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # RELEASE
            with self._viewer_lock():
                self.controller.disable_arm_collisions()
                self.controller.detach_rod()
                self.controller.set_grippers(True)
            for _ in range(180):
                if not self._step():
                    break
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()  # RETREAT
            # Retreat outward from the rod.  With the mirrored base
            # orientations, moving both wrists in the same world-X direction
            # would make one gripper sweep back across the placed rod.
            left_retreat = np.array([-0.20, 0.16, 0.20])
            right_retreat = np.array([0.20, 0.16, 0.20])
            self._move_segment(
                left_grasp_target + place_shift + left_retreat,
                right_grasp_target + place_shift + right_retreat,
                260,
            )
        if self.machine.state is not TaskState.FAILURE_STOP:
            self.machine.advance()

        return self._finish()

    def _finish(self):
        report = self.metrics.summary(self.scene, self.config.target_rod_position)
        numerical_success = (
            report["final_rod_position_error_m"] < self.config.position_tolerance
            and report["final_rod_orientation_error_deg"] < self.config.orientation_tolerance_deg
        )
        if self.machine.state is TaskState.SUCCESS and not numerical_success:
            self._fail("final rod pose outside tolerance")
        report.update({
            "task": "task1_rod_transport",
            "result": "SUCCESS" if self.machine.state is TaskState.SUCCESS else "FAILURE",
            "failure_reason": self.failure_reason,
            "collision_detected": self.collision_detected,
            "left_grasp_success": self.left_grasp,
            "right_grasp_success": self.right_grasp,
            "roles": ROLES,
            "state_history": self.machine.history,
        })
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = self.config.output_dir / f"task1_{stamp}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        report["log_path"] = str(path)
        return report
