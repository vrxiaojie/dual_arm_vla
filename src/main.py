import argparse
import json
import queue
import time
import numpy as np
from src.command_server import CommandServer
from src.config import TaskConfig
from src.scene import TaskScene
from src.task import Task1Runner


def serve(scene, config, server, viewer=None):
    print(f'Waiting for commands on 127.0.0.1:{server.server_address[1]} (1=green, 2=blue)', flush=True)
    while viewer is None or viewer.is_running():
        try:
            job = server.pending.get_nowait()
        except queue.Empty:
            # Freeze simulation while idle, but keep camera/UI responsive.
            if viewer is not None:
                viewer.sync()
            time.sleep(0.02)
            continue
        command = job[0]
        try:
            rod = scene.body_position('rod')
            if np.linalg.norm(rod - command.target_position) < config.position_tolerance:
                server.complete(job, {'status': 'already_at_target', 'command': command.direction})
                continue
            source = command.target_position.copy()
            source[1] = 0.0 if command.direction == 1 else -0.42
            if np.linalg.norm(rod - source) >= config.position_tolerance:
                server.complete(job, {'status': 'invalid_source', 'message': 'Rod is outside the source region'})
                continue
            report = Task1Runner(scene, config, viewer, command).run()
            server.complete(job, {'status': 'completed', 'report': report}, report['result'] != 'SUCCESS')
            print(json.dumps(report, indent=2), flush=True)
        except BaseException as exc:
            server.complete(job, {'status': 'failed', 'message': str(exc)}, failed=True)
            raise


def run(headless=False, realtime=False, port=8765):
    config = TaskConfig(realtime=realtime or not headless)
    scene = TaskScene(config.model_path, config.robot_urdf_path)
    with CommandServer(port) as server:
        if headless:
            serve(scene, config, server)
        else:
            import mujoco.viewer
            with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
                serve(scene, config, server, viewer)


def main():
    parser = argparse.ArgumentParser(description='Command-driven dual Panda rod transport')
    parser.add_argument('--headless', action='store_true', help='wait for commands without a viewer')
    parser.add_argument('--realtime', action='store_true', help='pace simulation steps to real time')
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    try:
        run(args.headless, args.realtime, args.port)
    except KeyboardInterrupt:
        return 0
    except OSError as exc:
        parser.exit(1, f'Command server error: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
