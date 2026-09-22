import contextlib
import io
import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np

from src.command_server import CommandServer
from src.config import TaskConfig
from src.main import serve
from src.scene import TaskScene


class FakeViewer:
    def __init__(self):
        self.running = True

    def is_running(self):
        return self.running

    def sync(self):
        pass

    def lock(self):
        return contextlib.nullcontext()


class CommandTests(unittest.TestCase):
    def test_invalid_busy_and_failure_latch(self):
        with CommandServer(0) as server:
            for value in (0, 3, True, '1', None):
                with socket.create_connection(server.server_address) as sock:
                    sock.sendall((json.dumps({'command': value}) + '\n').encode())
                    self.assertEqual(json.loads(sock.makefile('rb').readline())['status'], 'invalid')
            job, _ = server.submit(1)
            self.assertEqual(server.submit(2)[1]['status'], 'busy')
            server.complete(job, {'status': 'failed'}, failed=True)
            self.assertEqual(server.submit(1)[1]['status'], 'failed')

    def test_idle_and_three_round_trips_over_tcp(self):
        with tempfile.TemporaryDirectory() as directory, CommandServer(0) as server:
            config = TaskConfig(output_dir=Path(directory))
            scene = TaskScene(config.model_path)
            initial = scene.data.qpos.copy()
            viewer = FakeViewer()
            worker = threading.Thread(target=serve, args=(scene, config, server, viewer))
            with contextlib.redirect_stdout(io.StringIO()):
                worker.start()
                try:
                    time.sleep(0.1)
                    np.testing.assert_array_equal(scene.data.qpos, initial)
                    self.assertEqual(scene.data.time, 0)
                    for direction in (2, 1, 1, 2, 1, 2, 1, 2):
                        with socket.create_connection(server.server_address, timeout=20) as sock:
                            sock.sendall((json.dumps({'command': direction}) + '\n').encode())
                            stream = sock.makefile('rb')
                            self.assertEqual(json.loads(stream.readline())['status'], 'accepted')
                            response = json.loads(stream.readline())
                        if response['status'] == 'already_at_target':
                            continue
                        self.assertEqual(response['report']['result'], 'SUCCESS', response)
                        self.assertFalse(response['report']['collision_detected'])
                    self.assertEqual(len(list(Path(directory).glob('*.json'))), 6)
                finally:
                    viewer.running = False
                    worker.join(10)
            self.assertFalse(worker.is_alive())


if __name__ == '__main__':
    unittest.main()
