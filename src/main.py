import argparse
import json
import time

from src.config import TaskConfig
from src.scene import TaskScene
from src.task import Task1Runner


def run(headless=False, realtime=False):
    # A viewer is useful only if the task progresses at a human-observable
    # speed.  Keep headless runs fast, while making the normal interactive
    # command behave like a simulation rather than a batch calculation.
    config = TaskConfig(realtime=realtime or not headless)
    scene = TaskScene(config.model_path, config.robot_urdf_path)
    if headless:
        return Task1Runner(scene, config).run()

    import mujoco.viewer
    with mujoco.viewer.launch_passive(scene.model, scene.data) as viewer:
        report = Task1Runner(scene, config, viewer).run()

        # launch_passive does not own the Python process.  Without this loop,
        # the context manager closes the window as soon as the task finishes.
        # Keep the final scene available for inspection and camera/UI input
        # until the user explicitly closes the viewer.
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)
        return report


def main():
    parser = argparse.ArgumentParser(description="Dual Panda cooperative rod transport")
    parser.add_argument("--headless", action="store_true", help="run without a viewer")
    parser.add_argument("--realtime", action="store_true", help="pace viewer to simulation time")
    args = parser.parse_args()
    report = run(args.headless, args.realtime)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["result"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
