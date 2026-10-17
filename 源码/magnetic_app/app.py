import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PySide6.QtCore import QEvent, QPointF, Qt, QTimer, qInstallMessageHandler
from PySide6.QtGui import QMouseEvent, QSurfaceFormat
from PySide6.QtWidgets import QApplication, QComboBox, QMessageBox, QPlainTextEdit, QToolTip

from .domain.replay import Replay
from .ui.window import MainWindow


def summary(values):
    if not len(values):
        return {'count': 0}
    values = np.asarray(values)
    return {'count': len(values), 'median': float(np.median(values)), 'p95': float(np.percentile(values, 95)), 'max': float(values.max())}


class SmokeRun:
    def __init__(self, window, report, seconds=4.):
        self.window = window
        self.path = Path(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seconds = seconds
        self.results = {}
        self.stage = 0
        window.centralWidget().setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.requests = 0
        self.timer = QTimer(window)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.scrub)
        QTimer.singleShot(1800, self.begin)

    def clear_measurements(self):
        for view in [self.window.main_view, self.window.camera_view]:
            view.draw_times.clear()
            view.swap_times.clear()
            view.pose_draw_times.clear()
        self.window.tick_times.clear()
        self.window.ui_times.clear()

    def capture(self, name):
        window = self.window
        if not window.grab().save(str(self.path.parent/f'{name}.png')):
            raise RuntimeError('Cannot save screenshot')

    def begin(self):
        window = self.window
        window.scroll.verticalScrollBar().setValue(200)
        window.clear_probe()
        if window.replay.payload.get('routeDirection'):
            window.seek(0.)
            window.main_view.grabFramebuffer().save(str(self.path.parent/'native_anal_entry_view.png'))
            self.capture('native_anal_entry')
        window.seek(30)
        window.clock.set_speed(1)
        window.clock.set_playing(True)
        window.play_button.setText('Ⅱ 暂停')
        self.clear_measurements()
        QTimer.singleShot(round(self.seconds*1000), self.finish_insertion)

    def measurements(self):
        result = {'tick_ms': summary(self.window.tick_times), 'readout_ms': summary(self.window.ui_times)}
        for key, view in [('main', self.window.main_view), ('camera', self.window.camera_view)]:
            intervals = np.diff(view.swap_times)*1000
            result[key] = {'draw_cpu_ms': summary(view.draw_times), 'swap_interval_ms': summary(intervals), 'swap_fps': float(1000/np.mean(intervals)) if len(intervals) else None, 'driver': getattr(view, 'driver', {}), 'ready': view.ready, 'error': view.error}
            pose_intervals = np.diff(view.pose_draw_times)*1000
            result[key]['unique_pose_interval_ms'] = summary(pose_intervals)
            result[key]['unique_pose_draw_fps'] = float(1000/np.mean(pose_intervals)) if len(pose_intervals) else None
        return result

    def finish_insertion(self):
        self.results['insertion_1x'] = self.measurements()
        self.capture('native_insertion')
        self.window.seek(410)
        self.clear_measurements()
        QTimer.singleShot(round(self.seconds*1000), self.finish_withdrawal)

    def finish_withdrawal(self):
        self.results['withdrawal_1x'] = self.measurements()
        self.window.clock.set_speed(5)
        self.window.seek(430)
        self.clear_measurements()
        QTimer.singleShot(round(self.seconds*1000), self.finish_fast)

    def finish_fast(self):
        self.results['withdrawal_5x'] = self.measurements()
        self.window.clock.set_playing(False)
        self.window.play_button.setText('▶ 播放')
        self.clear_measurements()
        self.started = time.perf_counter()
        self.timer.start()
        QTimer.singleShot(round(self.seconds*1000), self.finish_scrub)

    def scrub(self):
        elapsed = time.perf_counter()-self.started
        value = int((elapsed/self.seconds % 1)*1000000)
        self.window.slider.setValue(value)
        self.requests += 1

    def finish_scrub(self):
        self.timer.stop()
        self.results['slider_16ms_requests'] = self.measurements()
        self.results['slider_16ms_requests']['requests'] = self.requests
        self.window.seek(self.window.replay.duration)
        self.window.controls['coverage'].setChecked(True)
        QTimer.singleShot(500, self.exercise_controls)

    def exercise_controls(self):
        window = self.window
        self.capture('native_coverage')
        window.scroll.verticalScrollBar().setValue(0)
        self.capture('native_overview')
        window.scroll.verticalScrollBar().setValue(1000)
        self.capture('native_evaluation')
        window.scroll.verticalScrollBar().setValue(200)
        for mode in range(window.state_mode.count()):
            window.state_mode.setCurrentIndex(mode)
        self.exercise_hover()
        QTimer.singleShot(50, self.inspect_algorithms)
        window.show_docs()
        window.write_evaluation(self.path.parent/'export_check.json')
        window.write_csv(self.path.parent/'export_check.csv')
        window.controls['field_scope'].setCurrentIndex(1)
        window.controls['field_mode'].setCurrentIndex(2)
        window.controls['lines'].setChecked(True)
        window.controls['slice'].setChecked(True)
        window.controls['torque'].setChecked(True)
        window.controls['tangent'].setChecked(True)
        window.controls['trajectory'].setChecked(True)
        window.controls['view_mode'].setCurrentIndex(1)
        window.controls['scale'].setCurrentIndex(1)
        QTimer.singleShot(800, self.finish)

    def exercise_hover(self):
        window = self.window
        view = window.main_view
        screen = view.project(window.replay.centerline)
        candidates = np.flatnonzero((screen[:, 0] > 25) & (screen[:, 0] < view.width() - 25) &
                                   (screen[:, 1] > 25) & (screen[:, 1] < view.height() - 25))
        if not len(candidates):
            raise RuntimeError("No projected centerline available for hover check")
        point = QPointF(*screen[candidates[len(candidates) // 2]])
        event = QMouseEvent(QEvent.Type.MouseMove, point, QPointF(view.mapToGlobal(point.toPoint())),
                            Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(view, event)
        tooltip = QToolTip.text()
        self.results["centerline_hover"] = dict(index=view.probe_index, tooltip=tooltip,
            readout=window.probe_readout.text(), chart_linked=window.risk_chart.probe is not None,
            event_source="Qt MouseMove sent to the native view; not a physical mouse test")
        self.capture("native_risk_hover")
        window.controls["path"].setChecked(False)
        self.results["centerline_hover"]["cleared_when_hidden"] = view.probe_index == -1 and window.risk_chart.probe is None
        window.controls["path"].setChecked(True)
        QApplication.sendEvent(view, event)
        QApplication.sendEvent(view, QEvent(QEvent.Type.Leave))
        self.results["centerline_hover"]["cleared_on_leave"] = view.probe_index == -1 and window.risk_chart.probe is None

    def inspect_algorithms(self):
        dialog = QApplication.activeModalWidget()
        if dialog is None:
            raise RuntimeError("Algorithm dialog did not open")
        try:
            selector = dialog.findChild(QComboBox, "algorithm_selector")
            text = dialog.findChild(QPlainTextEdit, "algorithm_text")
            self.results["algorithm_sections"] = []
            for index in range(selector.count()):
                selector.setCurrentIndex(index)
                content = text.toPlainText()
                self.results["algorithm_sections"].append(dict(title=selector.currentText(),
                    characters=len(content), plain_formula="$" not in content and chr(92) not in content))
                if not dialog.grab().save(str(self.path.parent / f"native_algorithm_{index + 1}.png")):
                    raise RuntimeError("Cannot save algorithm screenshot")
        finally:
            dialog.accept()

    def finish(self):
        window = self.window
        window.main_view.grabFramebuffer().save(str(self.path.parent/'native_layers.png'))
        self.results['render_errors'] = window.render_errors
        self.results['python_exceptions'] = getattr(window, 'runtime_exceptions', [])
        self.results['format'] = 'Native Qt Widgets + OpenGL; no HTML / JavaScript / browser runtime.'
        self.results['scientific_summary'] = window.replay.summary
        self.results['mount_error_mm'] = float(np.linalg.norm(window.current_frame.camera_position-window.current_frame.position-window.current_frame.body_rotation[:, 0]*5.8))
        self.results['final_pose_time'] = window.current_frame.time
        self.results['patient_posture'] = window.replay.payload.get('patientPosture')
        direction = window.replay.payload.get('routeDirection')
        if direction:
            self.results['route_direction'] = direction
            self.results['route_endpoints'] = dict(
                actual_start_mm=window.replay.frame(0.).position.tolist(),
                actual_turn_mm=window.replay.frame(window.replay.join_time).position.tolist(),
                actual_finish_mm=window.replay.frame(window.replay.duration).position.tolist(),
                timeline_captions=[window.entry_caption.text(),window.turn_caption.text(),window.exit_caption.text()])
        if self.results['patient_posture']:
            grid = window.main_view.make_grid()
            self.results['posture_display'] = dict(window_title=window.windowTitle(),
                grid_z_range=[float(grid[:, 2].min()),float(grid[:, 2].max())],
                geometry_span_mm=np.ptp(window.replay.points,axis=0).tolist(),
                gravity_origin_mm=window.main_view.gravity_origin.tolist(),
                gravity_direction=[0.,0.,-1.],default_elevation_deg=window.main_view.default_view[1])
        self.path.write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding='utf-8')
        QApplication.instance().exit(1 if window.render_errors or self.results['python_exceptions'] else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-report')
    parser.add_argument('--benchmark-seconds', type=float, default=4.)
    parser.add_argument('--data', type=Path)
    arguments = parser.parse_args()
    log_dir = Path(os.environ.get('LOCALAPPDATA', Path.home()))/'MagneticCapsule8Coil'/'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=log_dir/'native.log', level=logging.INFO, encoding='utf-8', format='%(asctime)s %(levelname)s %(message)s')
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setSamples(4)
    surface.setSwapInterval(1)
    QSurfaceFormat.setDefaultFormat(surface)
    application = QApplication(sys.argv[:1])
    application.setStyle('Fusion')
    application.styleHints().setColorScheme(Qt.ColorScheme.Light)
    application.setApplicationName('MagneticCapsule')
    application.setOrganizationName('MagneticNavigation')
    qInstallMessageHandler(lambda kind, context, message: logging.info('Qt %s: %s', kind, message))
    failures = []

    def exception_hook(exception_type, exception, tb):
        message = ''.join(traceback.format_exception(exception_type, exception, tb))
        failures.append(message)
        logging.error(message)

    sys.excepthook = exception_hook
    try:
        replay = Replay(arguments.data)
        window = MainWindow(replay)
        window.runtime_exceptions = failures
        window.show()
        if arguments.smoke_report:
            window.smoke_run = SmokeRun(window, arguments.smoke_report, arguments.benchmark_seconds)
        logging.info('Started native application; data=%s', replay.path)
        return application.exec()
    except Exception as exception:
        logging.exception('Startup failure')
        QMessageBox.critical(None, '启动失败', f'{exception}\n\n请保留软件文件夹内的 _internal 目录。\n诊断日志：{log_dir / "native.log"}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
