"""CLI regressions: validation and help must work without ML imports."""

import subprocess
import sys
from pathlib import Path

import pytest


def test_defaults():
    from live_swap import parse_args

    args = parse_args(["--reference", "face.jpg"])
    assert args.reference == Path("face.jpg")
    assert (args.camera, args.width, args.height, args.fps) == (0, 320, 240, 5)
    assert args.model == Path("models/inswapper_128.onnx")
    assert args.model_root == Path(".")


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--reference", "face.jpg", "--camera", "-1"],
        ["--reference", "face.jpg", "--width", "0"],
        ["--reference", "face.jpg", "--height", "-1"],
        ["--reference", "face.jpg", "--width", "321"],
        ["--reference", "face.jpg", "--width", "3844"],
        ["--reference", "face.jpg", "--height", "2161"],
        ["--reference", "face.jpg", "--fps", "-1"],
        ["--reference", "face.jpg", "--fps", "nan"],
        ["--reference", "face.jpg", "--fps", "inf"],
    ],
)
def test_invalid_arguments(argv):
    from live_swap import parse_args

    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_help_requires_no_heavy_dependencies():
    script = Path(__file__).resolve().parents[1] / "live_swap.py"
    code = """
import builtins, runpy, sys
original = builtins.__import__
def checked(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy', 'cv2', 'insightface', 'onnxruntime', 'pyvirtualcam'}:
        raise AssertionError('Heavy dependency imported before --help: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = checked
sys.argv = [sys.argv[1], '--help']
runpy.run_path(sys.argv[0], run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(script)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert "--reference" in result.stdout
