# Handoff: move face-swap processing to a Mac mini

## Read this first

The Windows CPU-only baseline is implemented. **Mac capture, CoreML acceleration, and network output are NOT implemented.** Continue from this code rather than starting another face-swap project.

The user's latest direction is:

> Run the face-swap processing on my Mac mini and stream the finished video to OBS on my existing Windows PC.

They requested this handoff and a local Git commit before continuing on the Mac. The folder was not previously a Git repository; that request authorizes initializing one. No remote or push was requested. Copy/clone the repository onto the Mac, but recreate its Python environment there.

## User requirements and preferences

- Small Python command-line app, managed with **uv**.
- Camera input such as Iriun; select a reference identity with `--reference PATH`.
- Real-time behavior means bounded delay, not a guaranteed FPS. Low resolution and low FPS are acceptable if the replacement face is recognizable.
- Process the largest visible face; compute reference identity once; no training.
- Keep processing local. For the new topology, send the processed video over the local network to Windows OBS.
- User chose native implementation in the current session rather than per-task implementation subagents.
- Use reference images with permission. InsightFace's supplied pretrained weights are **non-commercial research only**; commercial use requires appropriately licensed models.

Proposed topology, not yet validated:

```text
Camera / Iriun directly on Mac mini
    → face detection + reference-face swap on Mac
    → LAN video stream
    → OBS on Windows
```

MJPEG over HTTP with an OBS Browser Source was suggested as a simple low-resolution option. NDI was mentioned as an alternative. **The user has not selected a transport or approved a detailed Mac implementation design.** Do not describe those suggestions as existing features.

## Current code

| File | Responsibility |
|---|---|
| `live_swap.py` | Entire application: CLI, model preflight, reference loading, inference, latest-frame capture, virtual output, pacing, cleanup. |
| `requirements.txt` | Pinned direct runtime dependencies; tested on Windows Python 3.11. |
| `requirements-dev.txt` | Runtime dependencies plus pytest. |
| `tests/test_cli.py` | Defaults, invalid arguments, dependency-free help. |
| `tests/test_swapper.py` | Local model prerequisites, Unicode reference paths, one-face validation, cached reference embedding, CPU selection, largest-face/no-face behavior. |
| `tests/test_capture.py` | Single-slot capture, disconnect/stall handling, shutdown, aspect-ratio fitting. |
| `tests/test_pipeline.py` | Output format, no duplicate inference, failures, final blank frame, low-FPS health checks, exit codes. |
| `README.md` | Windows setup, official model links, uv commands, OBS setup, limitations. |
| `docs/superpowers/specs/2026-11-16-live-face-swap-design.md` | Approved original Windows design, not a Mac design. |
| `docs/superpowers/plans/2026-11-16-live-face-swap.md` | Original implementation plan and status; historical constraints describe Windows. |

### Important implementation details to preserve

- Defaults: camera index 0, 320×240 output, nominal output limit 5 FPS, detector input 320×320.
- `--fps` is configurable, not a hardware speed promise. Raising it does not accelerate inference.
- `FaceSwapper` uses compatible buffalo_l detection/recognition and `inswapper_128.onnx`. Live frames run detection/landmarks without recomputing recognition embeddings.
- Reference must contain exactly one detectable face. Windows Unicode paths use file bytes plus OpenCV decoding.
- Model files are checked locally before library initialization; there are no automatic runtime downloads.
- Capture uses one replaceable frame slot. Never introduce an unbounded capture, inference, encoder, or client-output queue.
- `fit_frame` preserves aspect ratio through black letterboxing.
- **No-face frames pass through unchanged. This is not an anonymization tool.**
- Errors stop the pipeline instead of silently disabling swapping.
- Camera failures take precedence over a cached frame. Five seconds without capture progress triggers a watchdog; pacing checks health every 100 ms even at very low FPS.
- Native inference calls cannot be interrupted by that watchdog.
- Driver reads/releases belong to the capture worker. Shutdown waits two seconds and reports a stuck read rather than releasing concurrently.
- Unity Capture retains its last frame after close. `close_output` best-effort sends black before closing, even on Ctrl+C/failure. Keep equivalent explicit stopped-stream behavior in any network output.

## Verification actually performed on Windows

Original PC: Windows 11 Pro x64, AMD Ryzen 5 5500 (6 cores/12 threads), 16 GB RAM, Radeon RX 550 (approximately 4 GB reported VRAM). CPUExecutionProvider only.

- Fresh pre-handoff test run: **46 passed**.
- `uv pip check`: **29 compatible installed packages**.
- `compileall` for application/tests: exit 0.
- Actual model smoke tests, with synthetic images/landmarks/embedding: detector loads/runs, recognition produces 512 values, swapper changes a uint8 320×240 BGR frame. All sessions report CPUExecutionProvider.
- One cold synthetic swap took approximately **0.804 seconds**. This is **not** sustained throughput, live FPS, or evidence of identity fidelity.
- Unity Capture was subsequently installed with explicit user authorization and UAC approval. Registry entry verified; actual pyvirtualcam Unity backend opened and accepted a black frame.
- No full Iriun → real-reference swap → OBS reception test has been performed by the agent. Do not claim live quality, latency, or FPS is verified.

One fresh read-only review found two Important issues, both fixed and regression-tested: frozen last frame retained after sender shutdown, and low-FPS pacing delaying camera health checks. No outstanding reviewer Critical/Important findings.

Upstream non-fatal smoke-test warnings: scikit-image deprecation of `SimilarityTransform.estimate` used by InsightFace, and an ONNX unused initializer. No third-party code was patched.

## Assets and transfer

These files are ignored and **are not part of the Git commit**:

- `.venv/`: Windows environment; never copy it as the Mac environment.
- `models/`: downloaded model weights; transfer separately or download using official links in README.
- `tools/UnityCapture/`: registered Windows driver files; leave them in place on Windows. They are not needed for Mac network output.
- `reference.png`: now present locally, but not inspected, modified, or validated by the agent. Transfer privately if this is the user's chosen reference. All JPG/PNG reference photos are ignored.
- `.superpowers/`: local execution scratch/ledger, not needed to continue. Relevant decisions are summarized here.

Required model layout on either machine:

```text
models/
  inswapper_128.onnx
  buffalo_l/
    det_10g.onnx
    w600k_r50.onnx
```

Local SHA256 records (integrity comparisons, not publisher signature verification):

| File | SHA256 |
|---|---|
| `inswapper_128.onnx` | `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` |
| `det_10g.onnx` | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |
| `w600k_r50.onnx` | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |

Unity Capture source commit: `schellingb/UnityCapture@3ed54c325e0ad71afcf4f246c07e5e17b3d7f2d2`. Its DLLs are unsigned; this was disclosed before installation. Windows installation path: `tools/UnityCapture/Install`. Unregister with its official uninstaller before moving/deleting that directory.

## Where Mac work must change the code

1. `LatestFrameCapture.start()` hardcodes `cv2.CAP_DSHOW`. macOS needs an appropriate capture backend (typically AVFoundation), permissions, and real Iriun/device discovery checks.
2. `FaceSwapper.__init__()` hardcodes CPU providers in both factories and `ctx_id=-1`. Establish a native Mac CPU baseline first. On Apple Silicon, investigate compatible CoreML execution for **both** analysis and swapping; explicitly inspect actual providers/fallback and benchmark rather than assume acceleration.
3. `run()` hardcodes `pyvirtualcam`'s `unitycapture` sink. A direct LAN output should replace or complement it; Unity Capture is Windows-only.
4. CLI dimension restrictions currently come from Unity Capture (width divisible by four, max 3840×2160). Make sink-specific constraints explicit rather than impose Windows-driver rules on every future transport.
5. Extend tests for Mac backend/provider selection, network output, slow/disconnected receivers, bounded buffers, shutdown indication, and preserved Windows behavior where retained.

Before settling the Mac design, establish the Mac chip/architecture, RAM, macOS version, actual camera connection, and whether the two machines share a suitable private LAN. None of these Mac details has been supplied. Avoid making raw camera video/reference data publicly accessible; decide LAN binding/firewall/authentication deliberately.

## First commands on the Mac

From the transferred repository, using native architecture Python rather than copying Windows binaries:

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements-dev.txt
uv pip check --python .venv/bin/python
uv run --no-project --python .venv/bin/python python -m pytest -q
uv run --no-project --python .venv/bin/python live_swap.py --help
```

Dependency pins were tested on Windows only. Check compatible macOS wheels/version requirements if installation fails; record justified changes rather than claiming the current pins are already Mac-tested.

`--help` and hardware-free tests are useful first checks. **Running the existing camera pipeline on macOS is not supported yet** because of the hardcoded capture/output backends. Continue with a short approved Mac/network design, implementation plan, tests, native model benchmarks, and finally an end-to-end OBS test.
