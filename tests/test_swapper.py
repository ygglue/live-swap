"""Model boundary tests use real images/Face objects, but no inference assets."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
from insightface.app.common import Face

import live_swap


@pytest.fixture
def assets(tmp_path, monkeypatch):
    pack = tmp_path / "models" / "buffalo_l"
    pack.mkdir(parents=True)
    for name in ("det_10g.onnx", "w600k_r50.onnx"):
        (pack / name).write_bytes(b"model")
    model = tmp_path / "models" / "inswapper_128.onnx"
    model.write_bytes(b"model")
    reference = tmp_path / "référence face.png"
    reference.write_bytes(cv2.imencode(".png", np.full((32, 32, 3), 80, np.uint8))[1].tobytes())
    source = Face(bbox=np.array([0, 0, 20, 20]), embedding=np.ones(512, np.float32))
    boxes = np.array([[0, 0, 8, 8, .9], [5, 5, 25, 25, .8]], np.float32)
    landmarks = np.arange(20, dtype=np.float32).reshape(2, 5, 2)
    analysis = SimpleNamespace(
        prepare=Mock(),
        get=Mock(return_value=[source]),
        det_model=SimpleNamespace(detect=Mock(return_value=(boxes, landmarks))),
    )

    def swap(frame, target, identity, paste_back):
        assert identity is source
        assert paste_back is True
        out = frame.copy()
        out[0, 0] = target.bbox[2]
        return out

    swap_model = SimpleNamespace(get=Mock(side_effect=swap))
    factory = Mock(return_value=analysis)
    swap_factory = Mock(return_value=swap_model)
    monkeypatch.setattr("insightface.app.FaceAnalysis", factory)
    monkeypatch.setattr("insightface.model_zoo.get_model", swap_factory)
    return SimpleNamespace(
        root=tmp_path, pack=pack, model=model, reference=reference, source=source,
        boxes=boxes, landmarks=landmarks, analysis=analysis,
        factory=factory, swap_factory=swap_factory, swap_model=swap_model,
    )


@pytest.mark.parametrize("missing", ["swap", "det_10g.onnx", "w600k_r50.onnx"])
def test_models_preflight_blocks_downloads(assets, missing):
    path = assets.model if missing == "swap" else assets.pack / missing
    path.unlink()
    with pytest.raises((FileNotFoundError, RuntimeError), match="model|Model"):
        live_swap.FaceSwapper(assets.reference, assets.model, assets.root)
    assets.factory.assert_not_called()
    assets.swap_factory.assert_not_called()


@pytest.mark.parametrize("contents", [None, b"not a picture"])
def test_reference_unreadable(assets, contents):
    if contents is None:
        assets.reference.unlink()
    else:
        assets.reference.write_bytes(contents)
    with pytest.raises((ValueError, RuntimeError), match="[Rr]eference"):
        live_swap.FaceSwapper(assets.reference, assets.model, assets.root)


@pytest.mark.parametrize("count", [0, 2])
def test_reference_requires_one_face(assets, count):
    assets.analysis.get.return_value = [assets.source] * count
    with pytest.raises(ValueError, match="exactly one face"):
        live_swap.FaceSwapper(assets.reference, assets.model, assets.root)


def test_reference_encoded_once_and_cpu_only(assets):
    engine = live_swap.FaceSwapper(assets.reference, assets.model, assets.root)
    frame = np.zeros((32, 32, 3), np.uint8)
    engine.process(frame)
    engine.process(frame)
    assert assets.analysis.get.call_count == 1
    source_pixels = assets.analysis.get.call_args.args[0]
    assert source_pixels.shape == (32, 32, 3)
    assert np.all(source_pixels == 80)
    assert assets.factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assert assets.factory.call_args.kwargs["allowed_modules"] == ["detection", "recognition"]
    assert assets.swap_factory.call_args.kwargs["providers"] == ["CPUExecutionProvider"]
    assets.analysis.prepare.assert_called_once_with(ctx_id=-1, det_size=(320, 320))


def test_largest_face_selected(assets):
    engine = live_swap.FaceSwapper(assets.reference, assets.model, assets.root)
    result = engine.process(np.zeros((32, 32, 3), np.uint8))
    assert result[0, 0].tolist() == [25, 25, 25]
    target = assets.swap_model.get.call_args.args[1]
    np.testing.assert_array_equal(target.kps, assets.landmarks[1])


def test_no_face_passthrough(assets):
    assets.analysis.det_model.detect.return_value = (np.empty((0, 5)), None)
    engine = live_swap.FaceSwapper(assets.reference, assets.model, assets.root)
    frame = np.full((32, 32, 3), 60, np.uint8)
    result = engine.process(frame)
    np.testing.assert_array_equal(result, frame)
    assets.swap_model.get.assert_not_called()
