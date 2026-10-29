"""Start and reuse the optional GPU depth service on demand."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time


DEPTH_HOST = "127.0.0.1"
DEPTH_PORT = 8765
_START_LOCK = threading.Lock()
_SERVICE_PROCESS = None


@dataclass(frozen=True)
class DepthResources:
    root: Path
    server: Path
    repo: Path
    checkpoint: Path


def _project_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[3]


def resolve_depth_resources(root: str | Path | None = None) -> DepthResources:
    """Resolve resources beside the EXE, or beside the source project."""
    project_root = Path(root).resolve() if root is not None else _project_root()
    candidates = [project_root, project_root / "软件_含深度版"]
    for candidate_root in candidates:
        resources = DepthResources(
            root=candidate_root,
            server=candidate_root / "深度服务" / "server.py",
            repo=candidate_root / "深度模型代码" / "EndoStreamDepth",
            checkpoint=candidate_root / "数据" / "深度模型" / "sim_depth_best.pth",
        )
        if all(path.exists() for path in (resources.server, resources.repo, resources.checkpoint)):
            return resources
    expected = [
        project_root / "深度服务" / "server.py",
        project_root / "深度模型代码" / "EndoStreamDepth",
        project_root / "数据" / "深度模型" / "sim_depth_best.pth",
    ]
    details = ", ".join(str(path) for path in expected)
    raise FileNotFoundError(f"深度服务资源缺失：{details}")


def _python_candidates() -> list[str]:
    candidates = []
    configured = os.environ.get("MAGNETIC_DEPTH_PYTHON")
    if configured:
        candidates.append(configured)
    candidates.extend([
        r"D:\anaconda\python.exe",
        sys.executable if not getattr(sys, "frozen", False) else "",
        shutil.which("python") or "",
    ])
    unique = []
    for candidate in candidates:
        if candidate and candidate not in unique and Path(candidate).exists():
            unique.append(candidate)
    return unique


def resolve_python_executable() -> str:
    if not _python_candidates():
        raise FileNotFoundError("找不到可运行深度服务的 Python；可设置 MAGNETIC_DEPTH_PYTHON")
    return _python_candidates()[0]


def build_service_command(resources: DepthResources, python_executable: str | None = None) -> list[str]:
    executable = python_executable or resolve_python_executable()
    return [
        executable,
        str(resources.server),
        "--repo", str(resources.repo),
        "--checkpoint", str(resources.checkpoint),
        "--host", DEPTH_HOST,
        "--port", str(DEPTH_PORT),
    ]


def depth_service_ready(host: str = DEPTH_HOST, port: int = DEPTH_PORT) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.35):
            return True
    except OSError:
        return False


def _log_path() -> Path:
    path = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "MagneticCapsule8Coil" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path / "depth_service.log"


def ensure_depth_service(timeout: float = 90.0) -> None:
    """Start the GPU service once, then wait until its TCP port is ready."""
    global _SERVICE_PROCESS
    if depth_service_ready():
        return
    with _START_LOCK:
        if depth_service_ready():
            return
        resources = resolve_depth_resources()
        command = build_service_command(resources)
        log_file = _log_path()
        log_stream = log_file.open("a", encoding="utf-8")
        log_stream.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] starting depth service\n")
        log_stream.write("command: " + " ".join(command) + "\n")
        log_stream.flush()
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        try:
            _SERVICE_PROCESS = subprocess.Popen(
                command,
                cwd=str(resources.root),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                creationflags=creation_flags,
                env={
                    **os.environ,
                    "PYTHONUNBUFFERED": "1",
                    # The installed Anaconda base environment contains both
                    # MKL's and PyTorch's OpenMP runtime on this machine.
                    "KMP_DUPLICATE_LIB_OK": "TRUE",
                },
            )
        except Exception:
            log_stream.close()
            raise
        log_stream.close()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if depth_service_ready():
                return
            if _SERVICE_PROCESS.poll() is not None:
                tail = ""
                try:
                    tail = "\n".join(log_file.read_text(encoding="utf-8", errors="replace").splitlines()[-12:])
                except OSError:
                    pass
                raise RuntimeError(f"深度服务启动失败（退出码 {_SERVICE_PROCESS.returncode}）。\n{tail}")
            time.sleep(0.25)
        _SERVICE_PROCESS.terminate()
        try:
            _SERVICE_PROCESS.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _SERVICE_PROCESS.kill()
            _SERVICE_PROCESS.wait(timeout=3)
        _SERVICE_PROCESS = None
        raise TimeoutError(f"深度服务在 {timeout:.0f} 秒内没有监听 {DEPTH_HOST}:{DEPTH_PORT}；日志：{log_file}")
