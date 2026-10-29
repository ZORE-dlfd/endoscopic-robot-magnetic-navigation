import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "源码"))

from magnetic_app.depth.service import build_service_command, resolve_depth_resources


def test_depth_service_resources_are_resolved_from_project_root():
    resources = resolve_depth_resources(ROOT / "软件_含深度版")
    assert resources.server.name == "server.py"
    assert resources.repo.name == "EndoStreamDepth"
    assert resources.checkpoint.name == "sim_depth_best.pth"


def test_depth_service_command_uses_project_resources():
    resources = resolve_depth_resources(ROOT / "软件_含深度版")
    command = build_service_command(resources, python_executable="D:/anaconda/python.exe")
    assert command[0] == "D:/anaconda/python.exe"
    assert str(resources.server) in command
    assert str(resources.repo) in command
    assert str(resources.checkpoint) in command
    assert "8765" in command


def test_default_resources_use_existing_depth_delivery_bundle():
    resources = resolve_depth_resources()
    assert resources.root == (ROOT / "软件_含深度版").resolve()
    assert resources.server.is_file()
    assert resources.checkpoint.is_file()
