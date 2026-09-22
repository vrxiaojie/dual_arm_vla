import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

from src.config import TaskCommand, TaskConfig
from src.scene import TaskScene
from src.task import Task1Runner


class RetreatRegressionTest(unittest.TestCase):
    def test_compiled_gravity_compensation_holds_an_unloaded_arm(self):
        scene = TaskScene(TaskConfig().model_path)
        self.assertGreater(scene.model.ngravcomp, 0)
        initial = scene.data.qpos.copy()
        for _ in range(250):
            mujoco.mj_step(scene.model, scene.data)
        runner = Task1Runner(scene, TaskConfig())
        for arm in (runner.controller.left, runner.controller.right):
            np.testing.assert_allclose(scene.data.qpos[arm.qpos_ids], initial[arm.qpos_ids], atol=1e-5)
            self.assertGreater(np.linalg.norm(scene.data.qfrc_gravcomp[arm.dof_ids]), 1)
        rod = mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY, 'rod')
        self.assertEqual(scene.model.body_gravcomp[rod], 0)

    def test_both_destinations_release_without_sag_or_mesh_penetration(self):
        with tempfile.TemporaryDirectory() as directory:
            config = TaskConfig(output_dir=Path(directory))
            scene = TaskScene(config.model_path)
            model, data = scene.model, scene.data
            original_masks = (model.geom_contype.copy(), model.geom_conaffinity.copy())
            tabletop = model.geom('table_top').pos[2] + model.body('table').pos[2] + model.geom('table_top').size[2]
            meshes = []
            for gid in range(model.ngeom):
                name = model.geom(gid).name
                if not name.startswith(('left_', 'right_')) or '_link0_' in name:
                    continue
                if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                    mid = model.geom_dataid[gid]
                    start, count = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                    meshes.append((gid, model.mesh_vert[start:start + count].copy()))
            for direction in (1, 2):
                runner = Task1Runner(scene, config, command=TaskCommand(direction))
                original_step = runner._step
                samples = {'RELEASE': [], 'RETREAT': []}

                def checked_step(*args, **kwargs):
                    ok = original_step(*args, **kwargs)
                    state = runner.machine.state.name
                    if state in samples:
                        samples[state].append([runner.controller.left.position[2], runner.controller.right.position[2]])
                        for gid, vertices in meshes:
                            # Actual rendered vertices, not just wrist centres.
                            bottom = (vertices @ data.geom_xmat[gid].reshape(3, 3)[2]).min() + data.geom_xpos[gid, 2]
                            self.assertGreater(bottom, tabletop, (direction, state, model.geom(gid).name))
                        for side in ('left', 'right'):
                            geom = model.geom(f'{side}_hand_geom')
                            self.assertTrue(int(geom.conaffinity[0]) & 8, 'Table contacts must remain enabled')
                    return ok

                runner._step = checked_step
                report = runner.run()
                self.assertEqual(report['result'], 'SUCCESS', report)
                for state, positions in samples.items():
                    self.assertTrue(positions, state)
                    positions = np.asarray(positions)
                    self.assertTrue(np.all(positions.min(axis=0) >= positions[0] - 0.002),
                                    (direction, state, positions[0], positions.min(axis=0)))
                self.assertTrue(np.all(np.asarray(samples['RETREAT'][-1]) > np.asarray(samples['RELEASE'][-1]) + 0.20))
                np.testing.assert_array_equal(model.geom_contype, original_masks[0])
                np.testing.assert_array_equal(model.geom_conaffinity, original_masks[1])


if __name__ == '__main__':
    unittest.main()
