import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
import queue
import time
import numpy as np
from src.command_server import CommandServer
from src.config import TaskConfig, BoxCommand
from src.scene import TaskScene
from src.task import TaskRunner


def serve(scene, config, server, viewer=None):
    print(f'Waiting for commands on 127.0.0.1:{server.server_address[1]} (task1/task2=switch, 1=green, 2=blue, start=box, reset)', flush=True)
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
            if isinstance(command, str) and command in ("task1", "task2", "reset"):
                with viewer.lock() if viewer is not None else nullcontext():
                    scene.switch_task(scene.task if command == "reset" else command)
                if viewer is not None:
                    viewer.sync()
                server.complete(job, {'status': 'scene_ready', 'task': scene.task})
                continue
            if scene.task == "task2":
                if command != "start":
                    server.complete(job, {'status': 'invalid', 'message': 'Task2 accepts start or reset'})
                    continue
                command = BoxCommand()
                task_config = replace(config, grasp_offset=.42, carry_height=.25, position_tolerance=.025)
                if np.linalg.norm(scene.body_position('rod') - command.target_position) < .025:
                    server.complete(job, {'status': 'already_at_target', 'task': scene.task})
                    continue
                report = TaskRunner(scene, task_config, viewer, command).run()
                server.complete(job, {'status': 'completed', 'report': report}, report['result'] != 'SUCCESS')
                print(json.dumps(report, indent=2), flush=True)
                continue
            if isinstance(command, str):
                server.complete(job, {'status': 'invalid', 'message': 'Task1 accepts 1 or 2'})
                continue
            rod = scene.body_position('rod')
            if np.linalg.norm(rod - command.target_position) < config.position_tolerance:
                server.complete(job, {'status': 'already_at_target', 'command': command.direction})
                continue
            source = command.target_position.copy()
            source[1] = 0.0 if command.direction == 1 else -0.42
            if np.linalg.norm(rod - source) >= config.position_tolerance:
                server.complete(job, {'status': 'invalid_source', 'message': 'Rod is outside the source region'})
                continue
            report = TaskRunner(scene, config, viewer, command).run()
            server.complete(job, {'status': 'completed', 'report': report}, report['result'] != 'SUCCESS')
            print(json.dumps(report, indent=2), flush=True)
        except Exception as exc:
            server.complete(job, {'status': 'failed', 'message': str(exc)}, failed=True)
            print(f"Task failed: {exc}", flush=True)


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
    parser = argparse.ArgumentParser(description='Command-driven dual Panda rod and box transport')
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
