# Live face swap → OBS

**Continuing on a Mac mini? Read [HANDOFF.md](HANDOFF.md).** The current app is the Windows baseline; Mac capture/acceleration and LAN streaming have not been implemented.

A small CPU-first Python script for Windows:

```text
Iriun Webcam → largest live face replaced with reference identity → Unity Capture → OBS
```

Defaults: **320×240**, up to **5 output FPS**, all processing local. Actual inference speed depends on the PC and has **not yet been benchmarked**. Lower FPS/resolution are intentional; there is no face restoration or training.

**Use a face you own or have permission to use.** InsightFace's supplied pretrained models are licensed for **non-commercial research only**, separately from its MIT-licensed code. For commercial use, obtain appropriately licensed compatible models. This is **not anonymization**: if no face is detected, the original frame is passed through. Other failures stop the script rather than silently bypass swapping.

## 1. Python setup with uv

Run PowerShell in this folder. Python 3.11 and uv were found on this PC; the runtime dependencies have already been installed into `.venv`.

To recreate the environment:

```powershell
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

The commands below explicitly select this environment, avoiding the MSYS Python on this PC's PATH. No activation or system-wide Python packages are needed.

## 2. Supply model weights

**On this PC, the three files below have now been downloaded and CPU-smoke-tested.** These instructions are for recreating that setup.

There are **no automatic model downloads**. Obtain the weights from the official InsightFace sources, subject to their model-use restrictions:

- [buffalo_l.zip](https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip) — about 289 MB to download. Extract the detector and recognizer below.
- [inswapper_128.onnx](https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx) — about 554 MB.
- [Model licensing](https://github.com/deepinsight/insightface/tree/master/python-package#license) and [swap model documentation](https://github.com/deepinsight/insightface/tree/master/examples/in_swapper).

Arrange them as follows, without an extra `buffalo_l/buffalo_l` nesting:

```text
deep-live/
  live_swap.py
  models/
    inswapper_128.onnx
    buffalo_l/
      det_10g.onnx
      w600k_r50.onnx
```

The other models from the buffalo_l archive are not needed; leave them out to reduce initialization work. The script is small, but its Python dependencies and pretrained weights are not.

`--model` overrides the swapper file. `--model-root` overrides the root **containing** `models/buffalo_l`, not the buffalo_l directory itself. Paths are relative to the current PowerShell directory.

## 3. Install Unity Capture once

**On this PC, Unity Capture is now registered and a Python output-frame smoke test passed.** Its installed files are in `tools/UnityCapture/Install`; keep that folder in place. Restart OBS to refresh its camera list. The steps below are for reinstalling:

1. Download [Unity Capture](https://github.com/schellingb/UnityCapture#installation) from its official repository.
2. Extract it to a permanent location.
3. Run `Install\Install.bat` as administrator to register its virtual camera.
4. Restart OBS after installation.

No Unity editor or Unity project is needed. The Python script sends frames through pyvirtualcam. Do not move/delete the driver's files after installation; use its uninstaller first.

This is a separate camera from **OBS Virtual Camera**, leaving OBS's own virtual-camera output free for other apps. The script never installs or registers drivers for you.

## 4. Choose the reference and start

Use a clear, front-facing photo containing **exactly one face**, preferably well lit and not cropped across the face. JPG and PNG work; paths with spaces/non-ASCII characters are supported.

Start Iriun on the PC and phone, verify its feed, then run:

```powershell
uv run --no-project --python .venv\Scripts\python.exe live_swap.py --reference "C:\Photos\face.jpg" --camera 0
```

To reduce processing/pacing further:

```powershell
uv run --no-project --python .venv\Scripts\python.exe live_swap.py --reference "C:\Photos\face.jpg" --camera 0 --width 160 --height 120 --fps 2
```

`--fps` is an output pacing limit, **not guaranteed new-face FPS**. The swap model processes 128×128 face crops, so reducing the overall frame dimensions does not reduce all of its inference work.

The reference identity is computed once. To select a different face, stop with **Ctrl+C** and restart with another `--reference` path. No training is required.

## 5. Select the output in OBS

1. Add **Video Capture Device** to your scene.
2. Select **Unity Video Capture**.
3. If automatic settings do not work, set **Resolution/FPS Type → Custom**, and match the script's resolution (default **320×240**) and FPS (default **5**).
4. Keep Iriun as input to the script, not as this OBS source.

If you also need OBS's mixed scene as a webcam in another app, use OBS's **Start Virtual Camera** as usual.

## Latency, speed, and failures

- The capture worker keeps only the latest frame. Slow inference drops older frames rather than building an unbounded queue; camera/phone/OBS buffering can still add delay.
- Aspect ratio is preserved with black letterboxing.
- Every five seconds, the terminal reports measured unique-frame processing throughput and average inference duration. Repeated output frames do not count as new inference.
- When capture temporarily has no new frame, the last processed output may repeat. A read failure or five seconds with no new captured frame stops the script. Pacing checks camera health at least every 100 ms, even at very low output FPS. A slow individual inference call cannot be interrupted by this watchdog.
- On Ctrl+C or a handled failure, the script attempts to send a black frame before closing the output. Unity Capture retains its last frame after the sender closes; the intended stopped view is therefore black, not a frozen face. If blanking fails (a warning is printed), or the process is forcibly killed/crashes, OBS may retain an old image; deactivate the OBS capture source manually.
- A native camera driver that hangs during a read may prevent its worker from releasing normally. Shutdown waits two seconds and reports this case; exit the process and restart Iriun if needed. The script avoids releasing the driver concurrently from another thread.
- Low light, profiles, occlusion, distant faces, or an unsuitable reference can reduce detection/recognizability. Recognition/identity fidelity needs a real-camera acceptance test.

| Problem | What to check |
|---|---|
| Missing/empty model | Match the folder layout and filenames above; do not use an HTML download-error page as an ONNX file. |
| Reference cannot load / wrong face count | Use a readable single-face photo; check the full quoted path and lighting. |
| Camera cannot open / read failed | Start Iriun, check the phone link and Windows camera permissions, and close competing apps. Try `--camera 1`, then `2` if necessary. |
| Unity Capture cannot open | Install/register the driver, restart OBS, and ensure another sender is not already using Unity Capture. |
| OBS shows a resolution mismatch/pattern | Select Unity Video Capture and match its custom resolution to the script. |
| High CPU / slow swap | Close other heavy apps; try `--fps 2`. No CUDA/DirectML acceleration is enabled on this AMD PC. |

Output width must be divisible by four, with dimensions no larger than **3840×2160**, to fit Unity Capture's documented limits.

## Tests and current verification

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements-dev.txt
uv run --no-project --python .venv\Scripts\python.exe python -m pytest -q
uv pip check --python .venv\Scripts\python.exe
```

Automated tests exercise argument validation, reference selection, model preflight, largest-face swapping, frame fitting, latest-frame capture, failures, and resource cleanup using fake model/driver boundaries. They do not activate a camera, download weights, or require an installed virtual-camera driver.

**All three real models have passed synthetic CPU inference checks, and the registered Unity Capture output accepted a black test frame.** These checks do not prove identity fidelity, OBS reception, or live throughput. Your chosen reference, full Iriun → face swap → OBS integration, and live FPS remain unverified until your reference photo is supplied and the complete pipeline is tested.
