"""Capture tests control driver reads with events, not timing assumptions."""

import queue
import threading
import time
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

import live_swap


class FakeCamera:
    def __init__(self, opened=True):
        self.opened = opened
        self.frames = queue.Queue()
        self.waiting = threading.Event()
        self.release_count = 0

    def isOpened(self):
        return self.opened

    def set(self, *_):
        return True

    def read(self):
        self.waiting.set()
        item = self.frames.get(timeout=3)
        self.waiting.clear()
        if isinstance(item, Exception):
            raise item
        return item

    def release(self):
        self.release_count += 1


@pytest.fixture
def capture(monkeypatch):
    camera = FakeCamera()
    factory = Mock(return_value=camera)
    monkeypatch.setattr(cv2, "VideoCapture", factory)
    cap = live_swap.LatestFrameCapture(0, 320, 240)
    cap.start()
    yield cap, camera, factory
    # Unblock the test driver so the worker can observe the stop request.
    camera.frames.put((False, None))
    cap.close()


def test_latest_frame_overwrites_older_frames(capture):
    cap, camera, factory = capture
    for value in (1, 2, 3):
        camera.frames.put((True, np.full((2, 2, 3), value, np.uint8)))
    with cap._condition:
        assert cap._condition.wait_for(lambda: cap._sequence == 3, timeout=2)
    sequence, frame = cap.read_after(0)
    assert sequence == 3
    assert frame[0, 0].tolist() == [3, 3, 3]
    frame[:] = 0
    assert cap.read_after(0)[1][0, 0].tolist() == [3, 3, 3]
    factory.assert_called_once_with(0, cv2.CAP_DSHOW)


def test_no_duplicate_sequence(capture):
    cap, camera, _ = capture
    camera.frames.put((True, np.ones((2, 2, 3), np.uint8)))
    sequence, _ = cap.read_after(0)
    assert cap.read_after(sequence, timeout=0) is None


def test_open_failure(monkeypatch):
    camera = FakeCamera(opened=False)
    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: camera)
    cap = live_swap.LatestFrameCapture(2, 320, 240)
    with pytest.raises(RuntimeError, match="[Cc]amera.*2"):
        cap.start()
    cap.close()
    assert camera.release_count == 1


@pytest.mark.parametrize("failure", [(False, None), RuntimeError("driver error")])
def test_disconnect_wakes_reader(capture, failure):
    cap, camera, _ = capture
    camera.frames.put((True, np.ones((2, 2, 3), np.uint8)))
    sequence, _ = cap.read_after(0)
    camera.frames.put(failure)
    with pytest.raises(RuntimeError, match="[Cc]amera"):
        cap.read_after(sequence, timeout=2)
    # An old cached frame must not hide a disconnect.
    with pytest.raises(RuntimeError, match="[Cc]amera"):
        cap.read_after(0, timeout=0)


def test_close_releases_and_stops_worker(capture):
    cap, camera, _ = capture
    camera.frames.put((False, None))
    cap.close()
    cap.close()
    assert camera.release_count == 1
    assert not cap._thread.is_alive()


def test_stalled_camera_does_not_repeat_stale_frame_forever(capture):
    cap, camera, _ = capture
    camera.frames.put((True, np.ones((2, 2, 3), np.uint8)))
    sequence, _ = cap.read_after(0)
    cap._last_frame_at = time.monotonic() - 6
    with pytest.raises(RuntimeError, match="stalled"):
        cap.read_after(sequence, timeout=0)


def test_check_health_detects_stall_without_reading(capture):
    cap, _, _ = capture
    cap._last_frame_at = time.monotonic() - 6
    with pytest.raises(RuntimeError, match="stalled"):
        cap.check_health()


@pytest.mark.parametrize(
    "shape,width,height,content_slice",
    [
        ((2, 4, 3), 4, 4, (slice(1, 3), slice(0, 4))),
        ((4, 2, 3), 4, 4, (slice(0, 4), slice(1, 3))),
        ((4, 4, 3), 2, 2, (slice(0, 2), slice(0, 2))),
        ((4, 4, 3), 1, 1, (slice(0, 1), slice(0, 1))),
    ],
)
def test_fit_frame(shape, width, height, content_slice):
    result = live_swap.fit_frame(np.full(shape, 100, np.uint8), width, height)
    assert result.shape == (height, width, 3)
    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    expected = np.zeros((height, width, 3), np.uint8)
    expected[content_slice] = 100
    np.testing.assert_array_equal(result, expected)
