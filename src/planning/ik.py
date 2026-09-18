import mujoco
import numpy as np


def cartesian_ik_step(model, data, body_id: int, dof_ids, target, damping=2e-3):
    """Return one bounded damped-least-squares joint update."""
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
    jac = jacp[:, dof_ids]
    error = np.asarray(target) - data.xpos[body_id]
    dq = jac.T @ np.linalg.solve(jac @ jac.T + damping * np.eye(3), error)
    return np.clip(dq, -0.045, 0.045), error

