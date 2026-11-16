# CPU-first live face swap

## Intent and approval

Build a small local Python command-line tool that replaces the largest face in an Iriun camera feed with the identity from a user-selected reference photo, then publishes a camera feed OBS can capture. Recognizable output matters more than resolution or FPS. The user approved the CPU-first design and explicitly required a reference-face option.

## Goals and constraints

- Target this PC: Windows 11 x64, Ryzen 5 5500, 16 GB RAM, AMD RX 550.
- Use Python 3.11 in a project-local virtual environment managed with uv; both are installed already.
- Default output is 320×240; actual inference FPS must be measured, not promised.
- Use CPUExecutionProvider only. No CUDA, DirectML, training, restoration, audio, recording, cloud processing, or GUI.
- Require `--reference PATH` with a clear photo containing exactly one face.
- Allow camera index, output dimensions, output FPS, and local model paths to be configured.
- Swap the largest detected live face. Frames with no detected face pass through unchanged; this is not an anonymization tool.
- Keep only the newest captured frame. Never accumulate an inference queue.
- Send BGR frames using pyvirtualcam's Unity Capture backend so OBS's own virtual-camera output remains available.
- All inference is local. Model downloads and the virtual-camera driver are separate setup steps, not surprise runtime actions.
- Use only reference images the user has permission to use. Supplied InsightFace pretrained models are restricted to non-commercial research; commercial workflows require appropriately licensed compatible models.

## Approaches

1. **CPU-first (selected):** OpenCV, ONNX Runtime CPU, InsightFace, pyvirtualcam. Simplest setup and fits the accepted low-FPS/low-resolution tradeoff.
2. **AMD/DirectML:** potentially faster, but adds provider/model compatibility work. Deferred until a CPU baseline is measured.

## Design and implementation calls

Keep application code in `live_swap.py`, with tests and a README rather than a package or service. Parse configuration without opening devices or importing heavy libraries.

Example interface:

```powershell
python live_swap.py --reference "face.jpg" --camera 0 --model "models\inswapper_128.onnx"
```

Use compatible `buffalo_l` detection/recognition weights and `inswapper_128.onnx`. Load only detection and recognition modules. Compute the source identity once; live frames need detection and landmarks, not a new identity embedding. Reject missing, unreadable, faceless, or ambiguous multi-face reference images.

Default `--model-root .` follows InsightFace's `<root>/models/buffalo_l/` convention, i.e. `models/buffalo_l/` in this project. Preflight the expected detector and recognition files before constructing FaceAnalysis so missing assets do not trigger automatic downloads.

Capture runs on one background thread with a replaceable single-frame slot. Main-thread processing takes the newest sequence number, preserves aspect ratio with letterboxing to the selected output dimensions, swaps the largest face, and publishes the result. Start at a nominal output rate of 5 FPS. This is a pacing limit, not a promised inference rate. Repeat the latest processed output when appropriate without running duplicate inference; report measured inference FPS periodically.

On invalid arguments, absent models, failed reference loading, camera failure, inference failure, or missing Unity Capture, exit with an actionable message. On Ctrl+C, close capture, stop the capture worker, and close virtual output. No-face passthrough is deliberate; other failures must not silently disable swapping.

## Setup and validation

Install dependencies only into `.venv`. Document official model sources and their restrictions; do not bundle model weights or register camera drivers automatically. Unity Capture installation requires the user's administrator action. OBS should use a Video Capture Device source selecting Unity Video Capture.

Automated tests use fake cameras, models, and sinks: no real camera, model download, GPU, or driver required. A separate user-assisted acceptance test uses their reference photo, Iriun, and OBS to check identity, delay, resolution, and actual speed.

## Non-goals and remaining decisions

No guaranteed FPS, photorealism, multiple-face controls, tracking system, live reference switching, installer, or packaged executable. The implementer may settle dependency pins based on successful Python 3.11 installation and smoke tests, without changing the architecture. No git repository exists; do not initialize one or create commits without a separate request.
