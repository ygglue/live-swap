"""Exercise orchestration with fake hardware/model boundaries and real frames."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import pyvirtualcam

import live_swap


@pytest.fixture
def pipeline(monkeypatch):
    frame = np.full((6, 8, 3), 30, np.uint8)
    results = [(1, frame), None, (2, frame), KeyboardInterrupt()]
    cap = SimpleNamespace(
        start=Mock(), close=Mock(),
        check_health=Mock(),
        read_after=Mock(side_effect=results),
    )
    processor = SimpleNamespace(process=Mock(side_effect=lambda frame: frame + 1))
    output = []
    sink = SimpleNamespace(
        send=Mock(side_effect=lambda frame: output.append(frame.copy())),
        close=Mock(), sleep_until_next_frame=Mock(), device="Unity Video Capture",
    )
    capture_factory = Mock(return_value=cap)
    swap_factory = Mock(return_value=processor)
    sink_factory = Mock(return_value=sink)
    monkeypatch.setattr(live_swap, "LatestFrameCapture", capture_factory)
    monkeypatch.setattr(live_swap, "FaceSwapper", swap_factory)
    monkeypatch.setattr(pyvirtualcam, "Camera", sink_factory)
    ticks = iter(float(i) for i in range(1000))
    monkeypatch.setattr(live_swap.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(live_swap.time, "sleep", lambda _: None)
    args = live_swap.parse_args(["--reference", "face.jpg"])
    return SimpleNamespace(
        args=args, frame=frame, cap=cap, processor=processor, sink=sink, output=output,
        capture_factory=capture_factory, swap_factory=swap_factory, sink_factory=sink_factory,
    )


def test_pipeline_output_contract(pipeline, capsys):
    with pytest.raises(KeyboardInterrupt):
        live_swap.run(pipeline.args)
    kwargs = pipeline.sink_factory.call_args.kwargs
    assert (kwargs["width"], kwargs["height"], kwargs["fps"]) == (320, 240, 5)
    assert kwargs["fmt"] == pyvirtualcam.PixelFormat.BGR
    assert kwargs["backend"] == "unitycapture"
    assert len(pipeline.output) == 4
    assert pipeline.output[0].shape == (240, 320, 3)
    assert pipeline.output[0].dtype == np.uint8
    assert pipeline.output[0][120, 160].tolist() == [31, 31, 31]
    assert not pipeline.output[-1].any()
    log = capsys.readouterr().out
    assert "Unity Video Capture" in log
    assert "inference" in log.lower()


def test_unique_frames_only_inferred_once(pipeline):
    with pytest.raises(KeyboardInterrupt):
        live_swap.run(pipeline.args)
    assert pipeline.processor.process.call_count == 2
    np.testing.assert_array_equal(pipeline.output[0], pipeline.output[1])


def test_capture_disconnect_exits(pipeline):
    pipeline.cap.read_after.side_effect = [(1, pipeline.frame), RuntimeError("Camera disconnected")]
    with pytest.raises(RuntimeError, match="Camera disconnected"):
        live_swap.run(pipeline.args)
    assert len(pipeline.output) == 2
    assert not pipeline.output[-1].any()
    pipeline.cap.close.assert_called_once()
    pipeline.sink.close.assert_called_once()


def test_inference_failure_exits(pipeline):
    pipeline.processor.process.side_effect = ValueError("bad model")
    with pytest.raises(RuntimeError, match="[Ii]nference"):
        live_swap.run(pipeline.args)
    assert len(pipeline.output) == 1
    assert not pipeline.output[0].any()
    pipeline.cap.close.assert_called_once()
    pipeline.sink.close.assert_called_once()


def test_sink_failure_releases_capture(pipeline):
    pipeline.sink_factory.side_effect = RuntimeError("backend absent")
    with pytest.raises(RuntimeError, match="Unity Capture"):
        live_swap.run(pipeline.args)
    pipeline.cap.close.assert_called_once()
    pipeline.sink.close.assert_not_called()


def test_ctrl_c_cleanup(pipeline):
    with pytest.raises(KeyboardInterrupt):
        live_swap.run(pipeline.args)
    pipeline.cap.close.assert_called_once()
    pipeline.sink.close.assert_called_once()
    assert not pipeline.output[-1].any()


@pytest.mark.parametrize("bad_output", [
    np.zeros((240, 320, 3), np.float32), np.zeros((10, 10, 3), np.uint8),
])
def test_invalid_processor_output_not_sent(pipeline, bad_output):
    pipeline.processor.process.side_effect = None
    pipeline.processor.process.return_value = bad_output
    with pytest.raises(RuntimeError, match="frame|output"):
        live_swap.run(pipeline.args)
    assert len(pipeline.output) == 1
    assert not pipeline.output[0].any()
    pipeline.cap.close.assert_called_once()
    pipeline.sink.close.assert_called_once()


@pytest.mark.parametrize("failure,expected", [(RuntimeError("setup failed"), 1), (KeyboardInterrupt(), 0)])
def test_main_exit_codes(monkeypatch, capsys, failure, expected):
    runner = Mock(side_effect=failure)
    monkeypatch.setattr(live_swap, "run", runner, raising=False)
    assert live_swap.main(["--reference", "face.jpg"]) == expected
    runner.assert_called_once()
    if expected == 1:
        assert "setup failed" in capsys.readouterr().err


def test_low_fps_pacing_still_detects_disconnect(monkeypatch):
    clock = SimpleNamespace(now=0.0, waits=[])

    def sleep(seconds):
        clock.waits.append(seconds)
        clock.now += seconds

    def check_health():
        if clock.now >= .25:
            raise RuntimeError("Camera disconnected")

    capture = SimpleNamespace(check_health=check_health)
    monkeypatch.setattr(live_swap.time, "monotonic", lambda: clock.now)
    monkeypatch.setattr(live_swap.time, "sleep", sleep)
    with pytest.raises(RuntimeError, match="Camera disconnected"):
        live_swap.pace_until(10.0, capture)  # The 10-second interval of --fps 0.1.
    assert clock.now <= .4
    assert max(clock.waits) <= .1


def test_output_close_attempted_even_when_final_blank_send_fails(pipeline, capsys):
    pipeline.sink.send.side_effect = RuntimeError("send failed")
    with pytest.raises(RuntimeError, match="send failed"):
        live_swap.run(pipeline.args)
    assert pipeline.sink.send.call_count == 2
    pipeline.sink.close.assert_called_once()
    pipeline.cap.close.assert_called_once()
    assert "blank" in capsys.readouterr().err
