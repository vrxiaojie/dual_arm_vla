# Dual Panda Tasks

MuJoCo demo of two Franka Panda arms cooperatively transporting a rod with an
explicit, safety-interlocked state machine.

## Natural-language commands via LLM

Copy `.env.example` to `.env` and fill in `LLM_BASE_URL` (the OpenAI-compatible
API root, such as `https://api.openai.com/v1`), `LLM_API_KEY`, and `LLM_MODEL`.
Set `LLM_REASONING_EFFORT` to `minimal`, `low`, `medium`, `high`, or `xhigh` only
if the provider supports that request field; otherwise leave it blank. The
local `.env` is ignored by Git. This client uses Python's standard library.

Start the simulation in one terminal, then send an instruction in another:

```powershell
python scripts/run_task.py
python scripts/natural_command.py "在搬运横杆的场景下，将杆子搬到绿色区域"
```

For an HTML interface, start the simulator and run `python scripts/llm_gui.py`
in a second terminal. It opens `http://127.0.0.1:8080/` in your browser. The page
includes editable instruction input, quick-test presets, a parsed command list,
right-side execution logs, and live LLM content/reasoning deltas (when returned
by the provider). Only validated, complete plans are executed. It reads the same
`.env`; `--port` selects the simulator port, `--env` selects its configuration,
`--web-port` changes the HTTP port, and `--no-browser` skips opening the browser.
Only one request runs at a time; closing the page does not cancel robot motion.
No additional dependencies are needed.

Start the simulator with `--vision` to show its final visual verdict, confidence,
reason, and top/side camera images on the page. The adjustable confidence
threshold defaults to 0.8: SUCCESS at or above the threshold passes visual
acceptance; low confidence or UNCERTAIN requires confirmation. This display
never overrides the robot controller's execution result. `--capture-images`
shows images without a visual verdict. Resets and scene switches have no final
motion images. The planning LLM streams to the page; the simulator's separate
visual LLM evaluation is displayed when its final report arrives.

The example plans `python scripts/send_command.py task1` followed by
`python scripts/send_command.py 1`. The LLM returns only the final argument
for each command; the client prints the full commands and executes them in
order. `--dry-run` shows the plan without sending it to the simulator. Use
`--port` if the command server is on a different local port.

The system prompt in `src/llm_command.py` describes all three scenes:
`task1` moves the rod to green (`1`) or blue (`2`); `task2` moves the box
to the green square (`start`); `task3` hands a block from the left arm to the
right, then places it in green (`start`). `reset` restores the current scene.
For an ambiguous or unsupported request, the model should return no commands
and a short explanation. The client validates the returned JSON against the
allowed command sequences before sending anything, and stops if a command
fails. To also send final-state camera images to the model, start the simulator
with `--vision` as described below.

## Final-state RGB cameras and visual LLM evaluation

The MJCF includes fixed `top_view` and `side_view` cameras for all three scenes.
`scene.capture_rgb()` uses `mujoco.Renderer` to return a dictionary of independent
960 × 720 × 3 `uint8` RGB arrays. Call it on the simulation thread while stepping
is paused; it does not advance physics or change the interactive viewer camera.
In the MuJoCo viewer, select either named camera from the camera selector.

```sh
# Capture and send both views to the model after each executed motion
python scripts/run_task.py --vision
# Or only save RGB PNGs without a model request
python scripts/run_task.py --headless --capture-images
```

`--vision` reads the existing `.env` (`--env PATH` overrides it). The provider
must support Chat Completions image input (`image_url` with base64 PNG data).
The visual request uses `response_format: {"type": "json_object"}` to request
structured output; the provider must support this option as well.
Both views are sent in one request with the task and destination; simulator
success flags and numerical measurements are not included. The prompt in
`src/vision.py` checks full target occupancy and visible placement/release,
and explicitly excludes claims about motion history or collision avoidance.
The occupancy criteria match the existing task evaluators: task2 checks the
cube body footprint (its handles extend beyond the square); task3 includes
both grips. The distant top camera reduces perspective enlargement of raised
objects while retaining compatibility with MuJoCo 3.1.

After every executed motion, including a failed motion, images are saved next
to its JSON log under `outputs/logs/<run>/`. The returned TCP report and saved
log contain `visual_evaluation`, with image paths, status, and a validated
`verdict` (`SUCCESS`, `FAILURE`, `UNCERTAIN`), numeric `confidence` in [0, 1], and
`reason`. An API/rendering/JSON error has `status: error`; captured files are
retained. `--capture-images` alone produces `status: captured` without a verdict.
Visual inspection is recorded independently of the controller's `result` and
safety interlocks; check both when deciding whether a task passed visual
acceptance. A static image cannot establish successful handover history or
stability over time. Scene switches, resets and `already_at_target` shortcuts
do not execute motion and do not request visual evaluation.

The simulator waits for the visual request before completing the command
(HTTP timeout: 60 seconds), so new motion commands remain busy during evaluation.
Offscreen rendering requires a working OpenGL backend even with `--headless`.
On macOS, launch the passive viewer using `mjpython scripts/run_task.py --vision`
if required by MuJoCo; the headless test runs with ordinary Python.

Run the dedicated test separately:

```sh
python -m unittest discover -s tests -p test_vision.py -v
# Real rendering and robot motion, with no network request
python scripts/test_vision.py
# Real model calls: source-state negatives and completed-state positives
python scripts/test_vision.py --live
# Limit to one scene (task1 includes both green and blue destinations)
python scripts/test_vision.py --live --task task1
```

The script saves screenshots and `summary.json` under
`outputs/screenshots/vision_test/` (`--output PATH` overrides it). Live tests
require exact expected verdicts: `UNCERTAIN`, API errors and motion failures
fail the test. Render-only mode checks capture/motion, without claiming model
accuracy. Live mode sends eight pairs of views for all scenes, uses the configured
API account, and exits nonzero if any case fails.

On macOS Python installations missing the default CA file, the API client uses
`/etc/ssl/cert.pem` with HTTPS verification enabled. An explicit `SSL_CERT_FILE`
setting takes precedence.

Local live check on 2026-10-14 with the configured model: 7 of 8 cases matched
the expected verdicts. Both rod destinations and box transport passed their
positive/negative checks. The handover initial state was correctly rejected,
but its completed state was judged `FAILURE`: the model interpreted an overhead
gripper/object overlap as possible continued contact despite the clearance in
the side view. This is a model judgment limitation, so the dedicated test exits
nonzero and retains the failed case rather than accepting it. Results and images
are in `outputs/screenshots/vision_json_mode/summary.json`. These eight examples
are a local smoke test, not an estimate of general visual accuracy.

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


### Task3: block handover

Task3 keeps the task1 arm base positions. The main block is 270 × 180 × 90 mm, with its 270 mm edge along the world X axis on a support at (-0.32, 0.00) m. The Panda grippers cannot span its 180 mm width, so each end has an attached 100 × 60 × 40 mm grip. The left arm grasps the left grip, lifts the block and offers the right grip to the right arm. Both right finger pads must contact the right grip before the left arm releases and retreats. The right arm places the block on a separate support in the green target region centered at (0.38, -0.30) m. Arm-to-arm collision detection remains active throughout the handover. The dimensions and target are defined in `src/config.py`. Grasp constraints activate only after contact verification. The result includes `inside_target_region`, which checks the full block and both grips.

Restart the main program to load the changes, then run in another terminal:

```powershell
python scripts/send_command.py task3
python scripts/send_command.py start
python scripts/send_command.py reset
```

`task3` switches and initializes the scene; `start` executes the handover; `reset` restores the current task's object, arms, grippers and inactive grasp constraints. Repeating `start` after success returns `already_at_target`; reset before another demonstration. Run the client without arguments for the interactive menu. Commands during motion return `busy`; reset after motion ends.
