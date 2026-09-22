import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config import TaskCommand, TaskConfig
from src.safety.grasp_monitor import GraspMonitor
from src.scene import TaskScene
from src.task import Task1Runner


class GraspGeometryTest(unittest.TestCase):
    def test_both_directions_keep_rod_between_visible_finger_pads(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TaskConfig(output_dir=Path(directory))
            scene = TaskScene(config.model_path)
            model, data = scene.model, scene.data
            diameter = 2 * model.geom('rod_geom').size[0]
            for direction in (1, 2):
                runner = Task1Runner(scene, config, command=TaskCommand(direction))
                original_check = runner.grasp.check
                original_step = runner._step
                counts = {'grasp': 0, 'carrying': 0}

                def check():
                    counts['grasp'] += 1
                    self.assertEqual(original_check(), (True, True))
                    # Losing either opposing pad must invalidate that grasp,
                    # even if the palm or the other finger touches the rod.
                    for side in ('left', 'right'):
                        missing = model.geom(f'{side}_finger2_geom').id
                        contacts = [c for c in data.contact if missing not in (c.geom1, c.geom2)]
                        filtered_scene = SimpleNamespace(
                            model=model, body_position=scene.body_position,
                            data=SimpleNamespace(ncon=len(contacts), contact=contacts),
                        )
                        result = GraspMonitor(runner.controller, filtered_scene).check()
                        self.assertFalse(result[0 if side == 'left' else 1])
                    return original_check()

                def step(*args, **kwargs):
                    ok = original_step(*args, **kwargs)
                    if runner.machine.state.name in ('LIFT', 'TRANSPORT', 'PLACE'):
                        counts['carrying'] += 1
                        rod_id = model.body('rod').id
                        for arm, endpoint in ((runner.controller.left, -.42), (runner.controller.right, .42)):
                            point = data.xpos[rod_id] + data.xmat[rod_id].reshape(3, 3) @ np.array([endpoint, 0, 0])
                            local = data.xmat[arm.body_id].reshape(3, 3).T @ (point - arm.position)
                            self.assertLess(abs(local[2]), .002, 'Rod must be at pad height, not below the fingertips')
                            self.assertLess(abs(local[1]), .002, 'Rod must stay centred between the fingers')
                            # Inspect the actual visible mesh's inner surfaces
                            # in the hand frame, not just the actuator command.
                            inner = []
                            for finger in (1, 2):
                                geom = model.geom(f'{arm.side}_finger{finger}_visual').id
                                mesh = model.geom_dataid[geom]
                                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                                vertices = model.mesh_vert[start:start + count]
                                world = vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
                                hand = (world - arm.position) @ data.xmat[arm.body_id].reshape(3, 3)
                                inner.append(hand[:, 1].min() if finger == 1 else hand[:, 1].max())
                            gap = inner[0] - inner[1]
                            self.assertGreater(gap, diameter - .001)
                            self.assertLess(gap, diameter + .007)
                    return ok

                runner.grasp.check = check
                runner._step = step
                report = runner.run()
                self.assertEqual(report['result'], 'SUCCESS', report)
                self.assertEqual(counts['grasp'], 1)
                self.assertGreater(counts['carrying'], 1000)


if __name__ == '__main__':
    unittest.main()
