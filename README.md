# Dual Panda Tasks

MuJoCo demo of two Franka Panda arms cooperatively transporting a rod with an
explicit, safety-interlocked state machine.

```powershell
# Terminal 1: start the viewer and wait for commands
.\.venv\Scripts\python.exe scripts/run_task.py

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
After a motion failure, reset or switch the scene to clear the failure latch.
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

Grasp height follows the settled rod centre. The hidden palm collision box
sits above the gap, and the fingertip contact boxes align with the visible
inner pad faces. Closing targets half the rod diameter minus a 2 mm preload
per finger, so contact holds an approximately 57 mm gap around the 56 mm rod
instead of closing the fingers together. A 0.5 mm contact margin stabilizes
pad contact detection. Both pads on each gripper must contact the rod before
the grasp constraints can activate; their anchors preserve that verified
pose without pulling the rod up to a different point.

Gravity compensation is enabled in the MJCF before model compilation, so
the unloaded arms hold their pose when the rod is released. Release holds
the current joint positions; retreat first raises both hands by 22 cm, then
moves them outward and back on continuous Cartesian waypoints. Only arm/rod
contacts are temporarily filtered during release: table and inter-arm
collisions remain enabled throughout.

To reproduce the visual checks, run `python scripts/inspect_transport.py`.
It renders both grippers at grasp, lift, transport, release and retreat
checkpoints in both directions
and writes PNGs plus `measurements.json` to
`outputs/screenshots/gripper_retreat/`. The regression suite also checks
both destinations for wrist sag and mesh vertices crossing the tabletop.
It also measures the visible finger gap and rod height during carrying, and
verifies that missing either pad contact prevents grasp validation.


## Task2: box transport

Start `scripts/run_task.py`, then use the command client:

```powershell
python scripts/send_command.py task2
python scripts/send_command.py start
python scripts/send_command.py reset
python scripts/send_command.py task1
```

Run the client without arguments for its interactive menu. `task1` and `task2`
select and initialize a scene; `reset` restores the current scene; `start`
transports the box in task2. Task1 retains commands `1` and `2`.
Switching keeps the same MuJoCo window, model, data and camera. Commands
received during motion return `busy`. Reset or switch after a failure to recover.

Task2 retains both task1 base positions and orientations, with separate
pedestals replacing the table. A 0.60 m cube with handles on its left and right
sides is carried from the floor into the green square. Both finger pads on
both hands must contact the handles before grasp constraints activate.
An additional orientation constraint models a firm handle grasp and prevents
rotation about the line joining the two grasp points. All constraints are
released before retreat. The box uses uniform 1 kg cube inertia. Motion uses
MuJoCo dynamics without directly setting object positions during transport.

TCP also accepts `{"command":"task2"}`, `{"command":"task1"}`,
`{"command":"start"}` and `{"command":"reset"}`. Switching and resetting
return `accepted`, then `scene_ready` with the active task name.
Task2 logs include bilateral grasp verification, box position/orientation
errors and `inside_target_square`, which checks the full box footprint.
