"""Render the shared viewer layout at task3 transport checkpoints."""
from pathlib import Path
import sys
import mujoco
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.inspect_transport import write_png
from src.config import TaskConfig, HandoverCommand, ROOT
from src.scene import TaskScene
from src.task import TaskRunner


def main():
    config = TaskConfig(carry_height=.25, position_tolerance=.025)
    scene = TaskScene(config.model_path)
    scene.switch_task("task3")
    output = ROOT / "outputs/screenshots/task3"
    output.mkdir(parents=True, exist_ok=True)
    scene.model.vis.global_.offwidth = 960
    scene.model.vis.global_.offheight = 720
    camera = mujoco.MjvCamera()
    camera.lookat[:] = [0, -.1, .5]
    camera.distance = 2.6
    camera.azimuth = 110
    camera.elevation = -25
    with mujoco.Renderer(scene.model, height=720, width=960) as renderer:
        def capture(name):
            renderer.update_scene(scene.data, camera=camera)
            write_png(output / (name + ".png"), renderer.render())
        capture("initial")
        runner = TaskRunner(scene, config, command=HandoverCommand())
        original_step = runner._step
        counts = {}
        def step(*args, **kwargs):
            ok = original_step(*args, **kwargs)
            state = runner.machine.state.name
            counts[state] = counts.get(state, 0) + 1
            if (state, counts[state]) in {("LEFT_LIFT",400), ("RIGHT_GRASP",180), ("RIGHT_TRANSPORT",400)}:
                capture(state.lower())
            return ok
        runner._step = step
        report = runner.run()
        capture("final")
        scene.switch_task("task1")
        capture("back_to_task1")
    print(report)
    return 0 if report["result"] == "SUCCESS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
