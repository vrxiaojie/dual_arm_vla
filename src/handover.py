"""Sequential contact-verified left-to-right handover of an upright block."""
import mujoco
import numpy as np
from src.state_machine import TaskState


def run_handover(r):
    scene, c = r.scene, r.controller
    model, data = scene.model, scene.data

    def phase(name):
        r.machine.state = TaskState[name]
        r.machine.history.append(name)

    def settle(steps=180):
        for _ in range(steps):
            if not r._step():
                return False
        return True

    def move(left, right, steps=300):
        return r._retreat_segment(np.array(left), np.array(right), steps)

    def weld(side, active):
        idx = model.equality(f"handover_{side}_weld").id
        if active:
            hand = getattr(c, side).body_id
            obj = model.body("rod").id
            model.eq_data[idx, 3:6] = data.xmat[hand].reshape(3, 3).T @ (data.xpos[obj] - data.xpos[hand])
            inverse = data.xquat[hand].copy()
            inverse[1:] *= -1
            mujoco.mju_mulQuat(model.eq_data[idx, 6:10], inverse, data.xquat[obj])
        data.eq_active[idx] = active
        mujoco.mj_forward(model, data)

    def grasp(side, offset):
        arm = getattr(c, side)
        with r._viewer_lock():
            arm.set_gripper(False)
        if not settle():
            return False
        contacts = set()
        for con in data.contact[:data.ncon]:
            names = {model.geom(con.geom1).name, model.geom(con.geom2).name}
            if "handover_object" in names:
                for finger in (1, 2):
                    if f"{side}_finger{finger}_geom" in names:
                        contacts.add(finger)
        expected = scene.body_position("rod") + np.array(offset)
        if len(contacts) != 2 or np.linalg.norm(arm.position - expected) > .025:
            r._fail(f"{side} handover grasp verification failed (contacts={contacts})")
            return False
        setattr(r, f"{side}_grasp", True)
        with r._viewer_lock():
            weld(side, True)
        return True

    def execute():
        if not settle(100):
            return
        right_wait = c.right.position.copy()
        left_offset = np.array([-.13, 0, .21])
        right_offset = np.array([.13, 0, .21])
        pickup = scene.body_position("rod") + left_offset
        phase("LEFT_PRE_GRASP")
        if not move(pickup + [0, 0, .12], right_wait): return
        phase("LEFT_APPROACH")
        if not move(pickup, right_wait): return
        phase("LEFT_GRASP")
        if not grasp("left", left_offset): return
        phase("LEFT_LIFT")
        if not move(pickup + [0, 0, .14], right_wait): return
        transfer = np.array([0, 0, .81])
        phase("MOVE_TO_HANDOVER")
        if not move(transfer + left_offset, right_wait): return
        phase("RIGHT_APPROACH")
        if not move(c.left.position, transfer + right_offset + [.08, 0, .04]): return
        if not move(c.left.position, scene.body_position("rod") + right_offset): return
        phase("RIGHT_GRASP")
        if not grasp("right", right_offset): return
        phase("LEFT_RELEASE")
        with r._viewer_lock():
            weld("left", False)
            c.left.set_gripper(True)
        if not settle(100): return
        phase("LEFT_RETREAT")
        if not move(c.left.position + [-.20, 0, .08], c.right.position): return
        left_wait = c.left.position.copy()
        phase("RIGHT_TRANSPORT")
        target = r.command.target_position
        if not move(left_wait, target + right_offset + [0, 0, .14]): return
        phase("RIGHT_PLACE")
        if not move(left_wait, target + right_offset): return
        phase("RIGHT_RELEASE")
        with r._viewer_lock():
            weld("right", False)
            c.right.set_gripper(True)
        if not settle(): return
        phase("RIGHT_RETREAT")
        if not move(left_wait, c.right.position + [0, 0, .15]): return
        if not settle(250): return
        r.machine.state = TaskState.SUCCESS
        phase("SUCCESS")

    try:
        execute()
    finally:
        with r._viewer_lock():
            c.restore_arm_collisions()
    return r._finish()

