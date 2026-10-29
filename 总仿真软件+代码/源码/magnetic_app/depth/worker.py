"""Qt workers for on-demand and route-wide depth estimation."""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .client import estimate_depth
from .service import ensure_depth_service


class DepthEstimateWorker(QObject):
    result = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, png_bytes: bytes, parent=None):
        super().__init__(parent)
        self.png_bytes = png_bytes

    def run(self):
        try:
            ensure_depth_service()
            self.result.emit(estimate_depth(self.png_bytes, timeout=180.0))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class DepthBatchWorker(QObject):
    ready = Signal()
    result = Signal(int, object)
    error = Signal(str)
    finished = Signal()

    def start(self):
        try:
            ensure_depth_service()
            self.ready.emit()
        except Exception as exc:
            self.error.emit(str(exc))

    @Slot(int, bytes)
    def estimate(self, index: int, png_bytes: bytes):
        try:
            self.result.emit(index, estimate_depth(png_bytes, timeout=180.0))
        except Exception as exc:
            self.error.emit(f"第 {index + 1} 帧推理失败：{exc}")

    def stop(self):
        self.finished.emit()
