# Dual Panda Task 1

MuJoCo demo of two Franka Panda arms cooperatively transporting a rod with an
explicit, safety-interlocked state machine.

```powershell
# Terminal 1: start the viewer and wait for commands
.\.venv\Scripts\python.exe scripts/run_task1.py

# Terminal 2: enter 1 or 2 repeatedly; q exits the client
.\.venv\Scripts\python.exe scripts/send_command.py
```

`1` moves the rod from the blue region to the green region. `2` moves it from
green to blue. The simulation stays frozen until a command arrives, and waits
again after each completed transport. The same scene supports repeated round
trips without resetting or teleporting the rod. A command whose destination
already contains the rod returns `already_at_target` without moving anything.

You can also send one command with `python scripts/send_command.py 1` or
`python scripts/send_command.py 2`. The client prints acceptance immediately,
then waits for the result. It requires only the Python standard library.

The main program also supports `--headless` (still waits for commands) and
`--realtime`. Viewer mode always uses real-time pacing. Close the viewer or
press Ctrl+C in the main terminal to exit. Quitting the client does not close
the main program or cancel an accepted task.

Both programs accept `--port` (default `8765`). The server listens only on
`127.0.0.1`. Commands received while busy are rejected rather than queued.
After a motion failure, the server rejects further commands until restarted.
A rod outside the expected source region is rejected without motion.

For future VLA integration, connect via TCP and send one newline-terminated
JSON object: `{"command": 1}` or `{"command": 2}`. The server sends an
`accepted` JSON line followed by a final JSON line (`completed` with a report,
`already_at_target`, or `invalid_source`). Rejected requests receive one line
with status `invalid`, `busy`, or `failed`. Disconnecting after acceptance
does not cancel the task; do not automatically retry a timed-out command.

Run regression tests with `python -m unittest discover -s tests -v`. These
cover startup without motion, TCP commands, three consecutive round trips,
duplicate targets, invalid input, busy rejection, and failure interlocking.

The run summary is printed and written to `outputs/logs/`.  The controller
uses actuator position targets and damped least-squares IK; it never teleports
an arm or the rod.  Two MuJoCo endpoint constraints are enabled only after
both grasp contacts have been verified.  They represent the closed-gripper
grasp and are released during `RELEASE`.

The grippers use the checked-in Panda `hand.stl` / `finger.stl` meshes, with
opposing fingers driven by the existing slide joints. Simplified contact
proxies and tool markers are hidden in viewer group 3; the visible meshes
remain in group 2. The tool frame is between the fingertips and the planner
keeps both grippers facing downward.

Gravity compensation is enabled in the MJCF before model compilation, so
the unloaded arms hold their pose when the rod is released. Release holds
the current joint positions; retreat first raises both hands by 22 cm, then
moves them outward and back on continuous Cartesian waypoints. Only arm/rod
contacts are temporarily filtered during release: table and inter-arm
collisions remain enabled throughout.

To reproduce the visual checks, run `python scripts/inspect_transport.py`.
It renders both grippers at release/retreat checkpoints in both directions
and writes PNGs plus `measurements.json` to
`outputs/screenshots/gripper_retreat/`. The regression suite also checks
both destinations for wrist sag and mesh vertices crossing the tabletop.
