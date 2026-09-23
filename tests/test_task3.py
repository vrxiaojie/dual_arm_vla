import contextlib
import io
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from src.command_server import CommandServer
from src.config import TaskConfig
from src.config import (HANDOVER_SIZE, HANDOVER_TARGET, HANDOVER_HANDLE_OFFSET,
                        HANDOVER_HANDLE_SIZE, HandoverCommand)
from src.main import serve
from src.scene import TaskScene
from src.task import TaskRunner
from test_commands import FakeViewer


class HandoverTests(unittest.TestCase):
    def test_large_block_handover_and_target_footprint(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TaskConfig(output_dir=Path(directory))
            scene = TaskScene(config.model_path)
            scene.switch_task("task3")
            np.testing.assert_allclose(HANDOVER_SIZE, [.27, .18, .09])
            np.testing.assert_allclose(scene.model.geom("handover_object").size[:3] * 2, HANDOVER_SIZE)
            for side, sign in (("left", -1), ("right", 1)):
                handle = scene.model.geom(f"handover_handle_{side}")
                np.testing.assert_allclose(handle.size[:3] * 2, HANDOVER_HANDLE_SIZE)
                self.assertAlmostEqual(handle.pos[0], sign * HANDOVER_HANDLE_OFFSET)
            np.testing.assert_allclose(scene.model.geom("handover_target").pos[:2], HANDOVER_TARGET[:2])
            self.assertEqual(scene.model.geom("left_hand_geom").conaffinity[0], 14)
            self.assertEqual(scene.model.geom("right_hand_geom").conaffinity[0], 13)
            report = TaskRunner(scene, config, command=HandoverCommand()).run()
            self.assertEqual(report["result"], "SUCCESS", report)
            self.assertTrue(report["inside_target_region"])
            self.assertFalse(report["collision_detected"])
            self.assertLess(report["state_history"].index("RIGHT_GRASP"),
                            report["state_history"].index("LEFT_RELEASE"))
            self.assertTrue(report["left_grasp_success"] and report["right_grasp_success"])
            scene.switch_task("task1")
            self.assertAlmostEqual(float(scene.model.body("rod").mass[0]), 1.0)

    def test_switch_reset_transport_and_recover_over_tcp(self):
        with tempfile.TemporaryDirectory() as directory, CommandServer(0) as server:
            config = TaskConfig(output_dir=Path(directory))
            scene = TaskScene(config.model_path)
            model, data = scene.model, scene.data
            bases = [scene.body_position(side + "_arm_base") for side in ("left", "right")]
            viewer = FakeViewer()
            def send(command):
                with socket.create_connection(server.server_address, timeout=30) as sock:
                    sock.sendall((json.dumps({"command": command}) + "\n").encode())
                    stream = sock.makefile("rb")
                    first = json.loads(stream.readline())
                    return json.loads(stream.readline()) if first["status"] == "accepted" else first
            with contextlib.redirect_stdout(io.StringIO()):
                worker = threading.Thread(target=serve, args=(scene, config, server, viewer))
                worker.start()
                try:
                    for _ in range(2):
                        self.assertEqual(send("task3")["status"], "scene_ready")
                        self.assertEqual(send(1)["status"], "invalid")
                        for _ in range(2):
                            result = send("start")
                            self.assertEqual(result["report"]["result"], "SUCCESS", result)
                            history = result["report"]["state_history"]
                            self.assertLess(history.index("RIGHT_GRASP"), history.index("LEFT_RELEASE"))
                            self.assertLess(result["report"]["final_object_position_error_m"], .025)
                            self.assertFalse(result["report"]["collision_detected"])
                            self.assertFalse(scene.data.eq_active.any())
                            self.assertTrue(result["report"]["left_grasp_success"])
                            self.assertTrue(result["report"]["right_grasp_success"])
                            self.assertEqual(send("start")["status"], "already_at_target")
                            self.assertEqual(send("reset")["status"], "scene_ready")
                            self.assertFalse(scene.data.eq_active.any())
                        self.assertEqual(send("task1")["status"], "scene_ready")
                        self.assertEqual(send("start")["status"], "invalid")
                        self.assertEqual(send(1)["report"]["result"], "SUCCESS")
                        self.assertEqual(send(2)["report"]["result"], "SUCCESS")
                    with server.guard:
                        server.state = "failed"
                    self.assertEqual(send(1)["status"], "failed")
                    self.assertEqual(send("reset")["status"], "scene_ready")
                    self.assertEqual(send(1)["report"]["result"], "SUCCESS")
                    self.assertIs(scene.model, model)
                    self.assertIs(scene.data, data)
                    for index, side in enumerate(("left", "right")):
                        np.testing.assert_array_equal(scene.body_position(side + "_arm_base"), bases[index])
                finally:
                    viewer.running = False
                    worker.join(10)
            self.assertFalse(worker.is_alive())


if __name__ == "__main__":
    unittest.main()
