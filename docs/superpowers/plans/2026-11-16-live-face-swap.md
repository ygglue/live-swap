# CPU-first Live Face Swap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap the largest face in an Iriun feed to a reference-photo identity and expose the result as a virtual camera for OBS.

**Architecture:** One command-line application, a latest-frame capture worker, CPU-only inference, and a Unity Capture sink. Model assets and the driver are explicit prerequisites; tests replace hardware and model objects.

**Tech Stack:** Windows 11 x64, Python 3.11, OpenCV, NumPy, ONNX Runtime CPU, InsightFace 2.1, pyvirtualcam, pytest.

**Spec:** `docs/superpowers/specs/2026-11-16-live-face-swap-design.md`

**Execution update:** User requested uv. Use `uv venv --python 3.11 .venv`, `uv pip install --python .venv\Scripts\python.exe ...`, `uv pip check --python .venv\Scripts\python.exe`, and `uv run --no-project --python .venv\Scripts\python.exe python ...` in place of the original equivalent commands below. Unity Capture's documented width-divisible-by-four and 3840×2160 maximum constraints are validated at the CLI.

## Execution status

The ledger at `.superpowers/sdd/2026-11-16-live-face-swap/progress.md` records RED/GREEN evidence and decisions. The original step checklists below describe the plan; this status is authoritative:

- [x] Task 1: CLI, uv environment, dependency installation and checks.
- [x] Task 2: Reference loading and CPU model interface, tested without weights.
- [x] Task 3: Latest-frame capture and aspect-preserving fitting, tested without hardware.
- [x] Task 4, steps 1–5: Output orchestration, error handling, README, automated verification.
- [ ] Task 4, step 6: Real model/reference/Iriun/OBS acceptance and measured FPS; blocked on user assets and Unity Capture setup.
- [x] Final fresh-context code review; two Important stale-feed findings fixed with RED→GREEN regression tests (46 tests passing).

## Global Constraints

- Use Python 3.11 in a project-local virtual environment; it is installed already.
- Default output is 320×240; actual inference FPS must be measured, not promised.
- Use CPUExecutionProvider only. No CUDA, DirectML, training, restoration, audio, recording, cloud processing, or GUI.
- Require `--reference PATH` with a clear photo containing exactly one face.
- Swap the largest detected live face. Frames with no detected face pass through unchanged; this is not an anonymization tool.
- Keep only the newest captured frame. Never accumulate an inference queue.
- Send BGR frames using pyvirtualcam's Unity Capture backend so OBS's own virtual-camera output remains available.
- All inference is local. Model downloads and the virtual-camera driver are separate setup steps, not surprise runtime actions.
- Use only reference images the user has permission to use. Supplied InsightFace pretrained models are restricted to non-commercial research; commercial workflows require appropriately licensed compatible models.
- No git repository exists; do not initialize one or create commits without a separate request.

## Review Focus

1. Unicode/spaced reference paths must load correctly on Windows; unreadable or multi-face references must fail clearly (Task 2).
2. Missing or incomplete model packs must fail before InsightFace can attempt downloads (Task 2).
3. Camera disconnects must terminate cleanly rather than keep publishing a stale feed indefinitely (Tasks 3–4).
4. Unusual source aspect ratios must not distort faces or produce incorrectly shaped virtual-camera frames (Task 3).
5. Absent Unity Capture or inference exceptions must give actionable errors and release already-open resources (Task 4).

## File structure

- `live_swap.py`: CLI, preflight, face processor, latest-frame capture, virtual-output orchestration.
- `requirements.txt`: tested runtime dependency constraints.
- `requirements-dev.txt`: runtime requirements plus pytest.
- `.gitignore`: exclude `.venv/`, model assets, local reference images, caches.
- `tests/test_cli.py`, `tests/test_swapper.py`, `tests/test_capture.py`, `tests/test_pipeline.py`: unit tests with injected fake dependencies.
- `README.md`: setup, licensing, reference selection, Iriun/OBS workflow, measurement and troubleshooting.

## Task 1: CLI and isolated dependencies

**Files:** Create `live_swap.py`, `requirements.txt`, `requirements-dev.txt`, `.gitignore`, `tests/test_cli.py`.

**Interfaces:** Produce `parse_args(argv: list[str] | None = None) -> argparse.Namespace` and `main(argv: list[str] | None = None) -> int`. Flags: required `--reference`; `--camera` nonnegative int default 0; `--width` positive int default 320; `--height` positive int default 240; `--fps` finite positive float default 5; `--model` Path default `models/inswapper_128.onnx`; `--model-root` Path default `.`. Heavy imports and resource opening occur only after CLI parsing.

- [ ] **Step 1:** Create `.venv` using `py -3.11 -m venv .venv`; install pytest into it. Write `test_defaults`, asserting reference `face.jpg`, camera 0, dimensions (320, 240), FPS 5, and the two default model paths; write parameterized `test_invalid_arguments` for missing reference, negative camera, zero dimensions, negative FPS, NaN, and infinity. Write `test_help_requires_no_heavy_dependencies`, running `--help` in a subprocess with heavy imports blocked.
- [ ] **Step 2:** Run `.venv\Scripts\python.exe -m pytest tests/test_cli.py -q`; confirm failure is the missing application/interface.
- [ ] **Step 3:** Implement the parser and guarded entry point. Add dependencies: InsightFace 2.1, ONNX Runtime CPU, OpenCV, NumPy compatible with Python 3.11, and pyvirtualcam; never install CUDA or GUI extras. Add dev requirements and exclusions. Install requirements in `.venv`, resolve any wheel compatibility issues, and record the successful runtime versions as pins.
- [ ] **Step 4:** Run the CLI tests, `.venv\Scripts\python.exe live_swap.py --help`, and `.venv\Scripts\python.exe -m pip check`; require passing tests, exit 0/help text, and no broken requirements.

## Task 2: Explicit model preflight and reference-photo swapping

**Files:** Modify `live_swap.py`; create `tests/test_swapper.py`.

**Interfaces:** Consume parsed `reference`, `model`, `model_root`. Produce `validate_model_paths(model: Path, model_root: Path) -> None`, `FaceSwapper(reference: Path, model: Path, model_root: Path)` and `FaceSwapper.process(frame: np.ndarray) -> np.ndarray`. Constructor imports/loads libraries lazily. Tests patch factories instead of loading real models.

- [ ] **Step 1:** Write `test_models_preflight_blocks_downloads`: missing swapper, detector `models/buffalo_l/det_10g.onnx`, or recognizer `models/buffalo_l/w600k_r50.onnx` under model root raises an actionable error and never calls the model factory. Write `test_reference_validation`: unreadable, zero-face, and two-face references fail; a Unicode/spaced path containing one face succeeds. Write `test_reference_encoded_once_and_cpu_only`: only detection/recognition modules load, all model factories receive `providers=["CPUExecutionProvider"]`, source embedding runs once despite multiple live frames. Write `test_largest_face_selected` with two unequal bounding boxes, checking the chosen target landmarks and cached source passed to `swapper.get(..., paste_back=True)`. Write `test_no_face_passthrough`, asserting the returned frame is unchanged.
- [ ] **Step 2:** Run `.venv\Scripts\python.exe -m pytest tests/test_swapper.py -q`; require failures for the missing interfaces.
- [ ] **Step 3:** Implement preflight before FaceAnalysis construction, disable optional modules, and use `ctx_id=-1` with explicit CPU providers. Load the reference with NumPy file bytes plus OpenCV decoding for Unicode support. Prepare detection at (320, 320). Compute the reference embedding once with compatible buffalo_l recognition. For live frames call the detector directly, construct an InsightFace Face from the selected bbox/keypoints, and swap without live recognition. Let inference exceptions propagate to orchestration.
- [ ] **Step 4:** Run the CLI and swapper tests; require all passing. Smoke-import installed library APIs without opening devices or downloading models.

## Task 3: Bounded live capture and output frame sizing

**Files:** Modify `live_swap.py`; create `tests/test_capture.py`.

**Interfaces:** Produce `LatestFrameCapture(camera_index: int, width: int, height: int)` with `start() -> None`, `read_after(sequence: int, timeout: float = 1.0) -> tuple[int, np.ndarray] | None`, and `close() -> None`; frames have monotonically increasing sequence IDs. `read_after` raises on open/read failure, returns None only on timeout, and returns a copy/latest independent frame. Produce `fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray`, preserving aspect ratio with centered black letterboxing.

- [ ] **Step 1:** Write `test_latest_frame_overwrites_older_frames`, using a controlled fake camera to publish sequences 1–3 before consumption and asserting sequence 3 is returned with no queue. Write `test_no_duplicate_sequence`, asserting a timed-out unchanged read returns None. Write `test_open_failure`, `test_disconnect_wakes_reader`, and `test_close_releases_and_stops_worker`, asserting actionable errors, capture release, and bounded worker shutdown. Write `test_fit_frame` for wide, tall, matching-aspect frames and tiny positive dimensions, asserting exact (height, width, 3) shape, uint8 contiguous BGR, undistorted content, and expected black bars.
- [ ] **Step 2:** Run `.venv\Scripts\python.exe -m pytest tests/test_capture.py -q`; require failures for the missing interfaces.
- [ ] **Step 3:** Implement a single-slot capture worker with a condition variable, stop event, shared failure state, idempotent cleanup, and Windows DirectShow input. Camera dimension properties are requests only; always fit actual returned frames. Ensure read_after observes failure even if the slot still contains an older frame. Resize using area interpolation for downscaling and linear for upscaling.
- [ ] **Step 4:** Run capture tests repeatedly and then the complete suite; require no surviving worker threads after fake-camera tests and all tests passing.

## Task 4: Virtual output, cleanup, documentation, and acceptance

**Files:** Modify `live_swap.py`; create `tests/test_pipeline.py`, `README.md`.

**Interfaces:** Consume all previous interfaces. Produce `run(args: argparse.Namespace) -> None`; `main` maps runtime/setup exceptions to concise stderr messages and exit 1, normal completion/Ctrl+C to exit 0. Instantiate `pyvirtualcam.Camera(width=..., height=..., fps=..., fmt=pyvirtualcam.PixelFormat.BGR, backend="unitycapture")`.

- [ ] **Step 1:** Write `test_pipeline_output_contract` with fake processor/capture/sink, asserting requested dimensions/rate, BGR format, Unity backend, and output frame shape. Write `test_unique_frames_only_inferred_once`, ensuring repeated sink sends never re-run inference for the same sequence. Write `test_capture_disconnect_exits`, `test_inference_failure_exits`, `test_sink_failure_releases_capture`, and `test_ctrl_c_cleanup`, asserting all acquired resources close once and errors contain actionable context. Write `test_main_exit_codes` for failure 1 and interruption 0. Use fake clock/pacing to avoid sleeps and test finite capture failure propagation.
- [ ] **Step 2:** Run `.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q`; require failures for the missing orchestration.
- [ ] **Step 3:** Implement startup order: model/reference preflight and loading, capture, sink. Consume latest frames, fit before inference, send only valid contiguous uint8 output, pace at configured FPS, and never queue. On timeout repeat last processed output, but keep checking capture failures. Log measured unique-frame inference FPS and inference duration every five seconds using a monotonic clock, explicitly distinguishing nominal output FPS. Use finally/context managers for cleanup and stop on inference errors.
- [ ] **Step 4:** Write README with exact Python 3.11 `.venv` commands; official InsightFace model links and `models/buffalo_l/` layout; model licensing/permission restrictions; manual Unity Capture installation/admin requirement; example `--reference "C:\Photos\face.jpg"` command; camera-index selection; OBS Video Capture Device → Unity Video Capture; how to change references by restarting; and low-FPS/no-face-passthrough caveats. Explain setup downloads versus offline runtime, large model/dependency size, and MSYS Python avoidance. Never auto-register drivers or fabricate benchmark results.
- [ ] **Step 5:** Run `.venv\Scripts\python.exe -m pytest -q`, `.venv\Scripts\python.exe -m compileall -q live_swap.py tests`, `.venv\Scripts\python.exe -m pip check`, and `--help`; record actual output.
- [ ] **Step 6:** Request the user's reference-photo path and completion of Unity Capture/model setup if still missing. With prerequisites and permission, run Iriun → script → OBS, confirm recognizable identity, aspect ratio, selectable output, no accumulating delay, and graceful Ctrl+C; record measured FPS. If blocked, explicitly report hardware/model integration as unverified, not complete.

## Self-review

All approved requirements map to tasks: reference selection/local compatible weights (2), CPU/CLI configuration (1–2), bounded delay/frame dimensions (3–4), OBS output/failure cleanup/measurement (4), and setup/license limits (4). All five review-focus conditions have explicit tests. No driver registration, git initialization, or real-camera/model test is hidden in automated verification.
