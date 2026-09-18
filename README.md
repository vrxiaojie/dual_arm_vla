# Dual Panda Task 1

MuJoCo demo of two Franka Panda arms cooperatively transporting a rod with an
explicit, safety-interlocked state machine.

```bash
source ./.venv/bin/activate
python scripts/run_task1.py --headless
python scripts/run_task1.py                 # interactive viewer
```

In interactive mode the task is paced at simulation time and the final scene
remains open for inspection. Close the MuJoCo window to end the process.

The run summary is printed and written to `outputs/logs/`.  The controller
uses actuator position targets and damped least-squares IK; it never teleports
an arm or the rod.  Two MuJoCo endpoint constraints are enabled only after
both grasp contacts have been verified.  They represent the closed-gripper
grasp and are released during `RELEASE`.
