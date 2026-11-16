"""Local CPU face swap: camera -> reference identity -> Unity Capture."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import math
import sys
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="Single-face reference photo")
    parser.add_argument("--camera", type=int, default=0, help="DirectShow camera index (default: 0)")
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--fps", type=float, default=5, help="Output pacing limit, not guaranteed inference FPS")
    parser.add_argument("--model", type=Path, default=Path("models/inswapper_128.onnx"))
    parser.add_argument("--model-root", type=Path, default=Path("."), help="Root containing models/buffalo_l")
    args = parser.parse_args(argv)
    if args.camera < 0:
        parser.error("--camera must be nonnegative")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if args.width % 4 or args.width > 3840 or args.height > 2160:
        parser.error("Unity Capture needs width divisible by 4, up to 3840x2160")
    if not math.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be finite and positive")
    return args


def validate_model_paths(model: Path, model_root: Path) -> None:
    pack = model_root / "models" / "buffalo_l"
    for path in (model, pack / "det_10g.onnx", pack / "w600k_r50.onnx"):
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(
                f"Missing or empty model: {path}. See README.md for model setup; "
                "models are never downloaded automatically."
            )


class FaceSwapper:
    def __init__(self, reference: Path, model: Path, model_root: Path):
        validate_model_paths(model, model_root)
        import cv2
        import numpy as np
        from insightface.app import FaceAnalysis
        from insightface.app.common import Face
        from insightface.model_zoo import get_model

        try:
            data = np.frombuffer(reference.read_bytes(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
        except OSError as exc:
            raise ValueError(f"Reference photo cannot be read: {reference}") from exc
        if image is None:
            raise ValueError(f"Reference photo cannot be decoded: {reference}")

        self._face_type = Face
        self.analysis = FaceAnalysis(
            name=str((model_root / "models" / "buffalo_l").resolve()),
            root=str(model_root.resolve()),
            allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        self.analysis.prepare(ctx_id=-1, det_size=(320, 320))
        faces = self.analysis.get(image)
        if len(faces) != 1:
            raise ValueError(
                f"Reference photo must contain exactly one face; found {len(faces)}. "
                "Use a clear, front-facing photo."
            )
        self.source = faces[0]
        self.swapper = get_model(str(model.resolve()), providers=["CPUExecutionProvider"])
        if self.swapper is None or not hasattr(self.swapper, "get"):
            raise ValueError(f"Unrecognized face-swap model: {model}")

    def process(self, frame: np.ndarray) -> np.ndarray:
        import numpy as np

        boxes, landmarks = self.analysis.det_model.detect(frame, max_num=0, metric="default")
        if len(boxes) == 0:
            return frame
        if landmarks is None:
            raise RuntimeError("Face detector returned no landmarks; use the buffalo_l detector.")
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        index = int(np.argmax(areas))
        face = self._face_type(
            bbox=boxes[index, :4], kps=landmarks[index], det_score=boxes[index, 4]
        )
        return self.swapper.get(frame, face, self.source, paste_back=True)


class LatestFrameCapture:
    """One replaceable frame; the worker owns driver reads and release."""

    def __init__(self, camera_index: int, width: int, height: int):
        self.camera_index, self.width, self.height = camera_index, width, height
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._sequence = 0
        self._frame = None
        self._failure = None
        self._thread = None
        self._last_frame_at = time.monotonic()

    def start(self) -> None:
        import cv2

        camera = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        try:
            if not camera.isOpened():
                raise RuntimeError(
                    f"Camera {self.camera_index} cannot open. Start Iriun, close other "
                    "apps using it, or choose another --camera index."
                )
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._last_frame_at = time.monotonic()
            self._thread = threading.Thread(
                target=self._capture, args=(camera,), name="latest-camera-frame", daemon=True
            )
            self._thread.start()
        except BaseException:
            camera.release()
            raise

    def _capture(self, camera) -> None:
        try:
            while not self._stop.is_set():
                ok, frame = camera.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera stopped providing frames. Check Iriun connection.")
                with self._condition:
                    self._frame = frame.copy()
                    self._sequence += 1
                    self._last_frame_at = time.monotonic()
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._failure = RuntimeError(f"Camera {self.camera_index} read failed: {exc}")
                self._condition.notify_all()
        finally:
            camera.release()

    def read_after(self, sequence: int, timeout: float = 1.0) -> tuple[int, np.ndarray] | None:
        with self._condition:
            remaining = max(0.0, 5.0 - (time.monotonic() - self._last_frame_at))
            self._condition.wait_for(
                lambda: self._failure is not None
                or self._sequence > sequence or self._stop.is_set(),
                timeout=min(timeout, remaining),
            )
            self.check_health()
            if self._sequence <= sequence:
                return None
            return self._sequence, self._frame.copy()

    def check_health(self) -> None:
        with self._condition:
            if self._failure is not None:
                raise self._failure
            if time.monotonic() - self._last_frame_at >= 5:
                raise RuntimeError("Camera stalled for 5 seconds. Check Iriun and restart.")
            if self._stop.is_set():
                raise RuntimeError("Camera capture is closed.")

    def close(self) -> None:
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2)
            if self._thread.is_alive():
                raise RuntimeError(
                    "Camera driver did not stop within 2 seconds. Close the process "
                    "and restart Iriun. Driver release is left to its owning worker."
                )


def fit_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    import cv2
    import numpy as np

    source_h, source_w = frame.shape[:2]
    scale = min(width / source_w, height / source_h)
    target_w = max(1, min(width, round(source_w * scale)))
    target_h = max(1, min(height, round(source_h * scale)))
    resized = cv2.resize(
        frame, (target_w, target_h),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    result = np.zeros((height, width, 3), dtype=np.uint8)
    x, y = (width - target_w) // 2, (height - target_h) // 2
    result[y:y + target_h, x:x + target_w] = resized
    return result


def pace_until(deadline: float, capture: LatestFrameCapture) -> None:
    """Unlike output.sleep_until_next_frame, never leave health unchecked for a long sleep."""
    while True:
        capture.check_health()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.1))


def close_output(output, width: int, height: int) -> None:
    """Unity Capture retains the final frame; replace it before closing."""
    import numpy as np

    try:
        output.send(np.zeros((height, width, 3), dtype=np.uint8))
    except Exception as exc:
        print(f"Warning: could not blank virtual output before closing: {exc}", file=sys.stderr)
    finally:
        output.close()


def run(args: argparse.Namespace) -> None:
    import numpy as np
    import pyvirtualcam

    print("Loading reference and models for CPU inference...", flush=True)
    processor = FaceSwapper(args.reference, args.model, args.model_root)
    with ExitStack() as resources:
        capture = LatestFrameCapture(args.camera, args.width, args.height)
        resources.callback(capture.close)
        capture.start()
        try:
            output = pyvirtualcam.Camera(
                width=args.width, height=args.height, fps=args.fps,
                fmt=pyvirtualcam.PixelFormat.BGR, backend="unitycapture",
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot open Unity Capture output. Install/register Unity Capture "
                "as described in README.md, then restart OBS. "
                f"Backend error: {exc}"
            ) from exc
        resources.callback(close_output, output, args.width, args.height)
        print(
            f"Output: {output.device} | {args.width}x{args.height} | "
            f"output limit {args.fps:g} FPS (actual inference may be slower).\n"
            "In OBS, select Unity Video Capture. Ctrl+C stops.",
            flush=True,
        )
        sequence, last_output = 0, None
        interval_start = time.monotonic()
        processed, inference_seconds = 0, 0.0
        while True:
            cycle_started = time.monotonic()
            item = capture.read_after(sequence, timeout=min(1.0, 1.0 / args.fps))
            if item is not None:
                sequence, frame = item
                frame = fit_frame(frame, args.width, args.height)
                started = time.monotonic()
                try:
                    result = processor.process(frame)
                except Exception as exc:
                    raise RuntimeError(f"Face inference failed: {exc}") from exc
                inference_seconds += time.monotonic() - started
                processed += 1
                if (
                    not isinstance(result, np.ndarray)
                    or result.shape != (args.height, args.width, 3)
                    or result.dtype != np.uint8
                ):
                    raise RuntimeError("Invalid inference output frame: expected uint8 BGR at output dimensions.")
                last_output = np.ascontiguousarray(result)
            if last_output is not None:
                capture.check_health()
                output.send(last_output)
                pace_until(cycle_started + 1.0 / args.fps, capture)
            now = time.monotonic()
            elapsed = now - interval_start
            if elapsed >= 5:
                average_ms = 1000 * inference_seconds / processed if processed else 0
                print(
                    f"CPU inference throughput: {processed / elapsed:.2f} FPS | "
                    f"mean inference {average_ms:.0f} ms | output limit {args.fps:g} FPS",
                    flush=True,
                )
                interval_start, processed, inference_seconds = now, 0, 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.", flush=True)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
