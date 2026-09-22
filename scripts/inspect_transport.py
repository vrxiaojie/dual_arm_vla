"""Render transport checkpoints and measure release/retreat table clearance."""
import json
from pathlib import Path
import struct
import sys
import zlib

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT, TaskConfig, TaskCommand
from src.scene import TaskScene
from src.task import Task1Runner


def write_png(path, pixels):
    height, width, _ = pixels.shape
    def chunk(tag, data):
        return struct.pack('!I', len(data)) + tag + data + struct.pack('!I', zlib.crc32(tag + data))
    raw = b''.join(b'\0' + row.tobytes() for row in pixels)
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', width, height, 8, 2, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw)) + chunk(b'IEND', b''))


def main():
    output = ROOT / 'outputs/screenshots/gripper_retreat'
    output.mkdir(parents=True, exist_ok=True)
    config = TaskConfig()
    scene = TaskScene(config.model_path)
    model, data = scene.model, scene.data
    meshes = []
    for gid in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ''
        if name.startswith(('left_', 'right_')) and model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[gid]
            start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
            meshes.append((gid, name, model.mesh_vert[start:start + count].copy()))
    summaries = []
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    with mujoco.Renderer(model, height=720, width=960) as renderer:
        for direction in (1, 2):
            runner = Task1Runner(scene, config, command=TaskCommand(direction))
            original_step = runner._step
            counts = {}
            lowest = {}
            hand_heights = {}
            def step(*args, **kwargs):
                ok = original_step(*args, **kwargs)
                state = runner.machine.state.name
                counts[state] = counts.get(state, 0) + 1
                tick = counts[state]
                if state in ('RELEASE', 'RETREAT'):
                    heights = [runner.controller.left.position[2], runner.controller.right.position[2]]
                    hand_heights.setdefault(state, []).append(heights)
                    for gid, name, vertices in meshes:
                        if '_link0_' in name:  # Fixed base is mounted to the tabletop.
                            continue
                        z = float((vertices @ data.geom_xmat[gid].reshape(3, 3)[2]).min() + data.geom_xpos[gid, 2])
                        if state not in lowest or z < lowest[state]['z']:
                            lowest[state] = {'z': z, 'geom': name, 'tick': tick}
                if ((state == 'GRASP' and tick == 250)
                        or (state == 'LIFT' and tick in (100, 400))
                        or (state == 'TRANSPORT' and tick in (260, 520))
                        or (state == 'RELEASE' and tick in (1, 90, 180))
                        or (state == 'RETREAT' and tick in (100, 400, 760))):
                    for arm in (runner.controller.left, runner.controller.right):
                        camera = mujoco.MjvCamera()
                        camera.lookat[:] = arm.position
                        camera.distance = .65
                        camera.azimuth = 135 if arm.side == 'left' else 45
                        camera.elevation = -20
                        renderer.update_scene(data, camera=camera)
                        write_png(output / f'{direction}_{arm.side}_{state.lower()}_{tick:03d}.png', renderer.render())
                return ok
            runner._step = step
            report = runner.run()
            heights = {state: {'start': values[0], 'minimum': np.min(values, axis=0).tolist(), 'end': values[-1]} for state, values in hand_heights.items()}
            summary = {'direction': direction, 'result': report['result'], 'failure_reason': report['failure_reason'], 'mesh_minimum': lowest, 'hand_height': heights}
            summaries.append(summary)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = [0, -.1, .6]
            camera.distance = 2.3
            camera.azimuth = 120
            camera.elevation = -25
            renderer.update_scene(data, camera=camera)
            write_png(output / f'{direction}_overview.png', renderer.render())
    (output / 'measurements.json').write_text(json.dumps(summaries, indent=2))
    print(json.dumps(summaries, indent=2))
    return 0 if all(item['result'] == 'SUCCESS' for item in summaries) else 1


if __name__ == '__main__':
    raise SystemExit(main())
