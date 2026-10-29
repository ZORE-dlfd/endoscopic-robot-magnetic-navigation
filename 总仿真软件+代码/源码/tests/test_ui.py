from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'构建'/'依赖'))
sys.path.insert(0, str(ROOT/'源码'))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QComboBox, QLabel, QPlainTextEdit, QTableWidget, QToolTip

from magnetic_app.domain.replay import Replay
from magnetic_app.ui.window import MainWindow


@pytest.fixture(scope='module')
def application():
    app = QApplication.instance() or QApplication([])
    app.setStyle('Fusion')
    app.styleHints().setColorScheme(Qt.ColorScheme.Light)
    return app


@pytest.fixture
def window(application):
    instance = MainWindow(Replay())
    instance.timer.stop()
    yield instance
    instance.close()
    instance.deleteLater()
    application.processEvents()


def test_supine_reference_plane_and_default_view(window):
    import numpy as np
    posture = window.replay.payload['patientPosture']
    view = window.main_view
    assert posture['posture'] == 'supine'
    assert '仰卧' in window.windowTitle()
    grid = view.make_grid()
    np.testing.assert_allclose(grid[:, 2], posture['table_z_mm'])
    assert view.elevation >= 40.
    np.testing.assert_allclose(window.replay.robot['config']['gravity_m_s2'], [0., 0., -9.81])


def test_timeline_and_scene_explicitly_identify_anal_entry(window):
    texts = [widget.text() for widget in window.findChildren(QLabel)]
    assert any('肛门入口 · 开始进镜' in text for text in texts)
    assert any('肠道深部 · 开始退镜' in text for text in texts)
    assert any('回到肛门入口' in text for text in texts)
    assert '肛门入口 → 肠道深部 → 肛门入口' in window.field_hint.text()
    assert window.replay.payload['routeDirection']['verification']['passed']


def test_cached_network_depth_tracks_interpolated_camera_pose(window):
    import numpy as np

    class CachedDepth:
        is_valid = True

        def __init__(self):
            self.request = None

        def get_interpolated(self, index, amount):
            self.request = (index, amount)
            return np.full((4, 4), index + amount, dtype=np.float32)

    cache = CachedDepth()
    window.depth_cache = cache
    target_time = float(window.replay.camera_times[4] * .75 + window.replay.camera_times[5] * .25)
    window.current_frame = window.replay.frame(target_time)
    window._show_playback_depth()

    assert cache.request == (window.current_frame.camera_index, round(window.current_frame.camera_amount, 2))
    assert window.last_depth_kind == '网络'
    np.testing.assert_allclose(window.last_depth, cache.request[0] + cache.request[1])


def test_estimate_depth_uses_complete_cache_without_capturing_or_starting_service(window):
    import numpy as np

    class CachedDepth:
        is_valid = True

        def __init__(self):
            self.request = None

        def get_interpolated(self, index, amount):
            self.request = (index, amount)
            return np.full((4, 4), 12.5, dtype=np.float32)

    cache = CachedDepth()
    window.depth_cache = cache
    capture_calls = []
    window.camera_view.capture_rgb = lambda: capture_calls.append(True)
    window.estimate_depth()

    expected = (window.current_frame.camera_index, round(window.current_frame.camera_amount, 2))
    assert cache.request == expected
    assert capture_calls == []
    assert window.depth_thread is None
    assert window.last_depth_kind == '网络'
    np.testing.assert_allclose(window.last_depth, 12.5)


@pytest.mark.parametrize('playing', [False, True])
def test_real_slider_drag_preserves_play_state(window, playing):
    slider = window.slider
    slider.resize(600, 30)
    window.clock.set_playing(playing)
    QTest.mousePress(slider, Qt.MouseButton.LeftButton, pos=QPoint(80, 15))
    assert window.clock.seeking and not window.clock.playing
    QTest.mouseMove(slider, QPoint(490, 15))
    selected = window.clock.current()
    assert selected > window.replay.duration*.75
    QTest.mouseRelease(slider, Qt.MouseButton.LeftButton, pos=QPoint(490, 15))
    assert window.clock.playing == playing and not window.clock.seeking
    assert abs(window.clock.current()-selected) < .1


def test_parameter_visibility_toggles(window):
    assert '八线圈' in window.field_hint.text()
    assert '仿射' not in window.field_hint.text()
    for key, readout in [('propulsion_parameters', window.propulsion_readout), ('turning_parameters', window.turning_readout)]:
        window.controls[key].setChecked(False)
        assert readout.isHidden()
        window.controls[key].setChecked(True)
        assert not readout.isHidden()


def test_coil_visibility_is_independent_of_field(window):
    assert window.controls['coils'].isChecked()
    window.controls['field'].setChecked(False)
    assert window.main_view.options['coils']
    window.controls['coils'].setChecked(False)
    assert not window.main_view.options['coils']
    window.controls['field'].setChecked(True)
    assert not window.main_view.options['coils']


def test_compact_coil_live_table_and_full_parameters(application, window):
    assert window.coil_table.rowCount() == 8 and window.coil_table.columnCount() == 4
    window.seek(window.replay.duration*.6)
    window.update_readouts()
    for index in range(8):
        assert window.coil_table.item(index, 0).text() == f'C{index+1}'
        assert abs(float(window.coil_table.item(index, 1).text())) <= window.replay.coils.limit+1e-4
    captured = []

    def inspect_dialog():
        dialog = application.activeModalWidget()
        try:
            table = dialog.findChild(QTableWidget)
            captured.append((table.rowCount(), table.columnCount(), table.item(1, 0).text(),
                             table.item(2, 0).text(), table.item(5, 0).text(), table.item(6, 0).text()))
        finally:
            dialog.accept()

    QTimer.singleShot(50, inspect_dialog)
    window.show_coil_parameters()
    assert captured == [(22, 8, '52', '32', '56', '1200')]


def test_all_state_charts_and_final_values(window):
    for index in range(window.state_mode.count()):
        window.state_mode.setCurrentIndex(index)
        assert len(window.state_chart.series) == 2
    window.seek(window.replay.duration)
    window.tick()
    window.update_readouts()
    assert f'{window.replay.summary["final_covered_area_mm2"]:,.2f}' in window.coverage_values['final_covered_area_mm2'][0].text()
    assert f'{window.replay.summary["withdrawal_added_area_mm2"]:,.2f}' in window.coverage_values['withdrawal_added_area_mm2'][0].text()
    assert window.state_table.rowCount() == 146


def test_native_formula_dialog(application, window):
    contents = []

    def inspect_dialog():
        dialog = application.activeModalWidget()
        assert dialog is not None
        text = dialog.findChild(QPlainTextEdit)
        selector = dialog.findChild(QComboBox, 'algorithm_selector')
        assert selector.count() == 4
        for index in range(selector.count()):
            selector.setCurrentIndex(index)
            contents.append(text.toPlainText())
        dialog.accept()

    QTimer.singleShot(50, inspect_dialog)
    window.show_docs()
    assert len(contents) == 4 and all(len(content) > 400 for content in contents)
    assert all('$' not in content and chr(92) not in content for content in contents)
    assert '1.2' in contents[-1] and '95%' in contents[-1]


def test_real_centerline_hover_and_clear(application, window):
    import numpy as np
    window.show()
    window.raise_()
    window.activateWindow()
    window.scroll.ensureWidgetVisible(window.main_view)
    QTest.qWait(400)
    view = window.main_view
    assert view.ready and not view.error
    screen = view.project(window.replay.centerline)
    valid = np.flatnonzero((screen[:, 0] > 25) & (screen[:, 0] < view.width() - 25) &
                          (screen[:, 1] > 25) & (screen[:, 1] < view.height() - 25))
    assert len(valid)
    point = QPoint(round(screen[valid[len(valid) // 2], 0]), round(screen[valid[len(valid) // 2], 1]))
    event = QMouseEvent(QEvent.Type.MouseMove, QPointF(point), QPointF(view.mapToGlobal(point)),
                        Qt.MouseButton.NoButton, Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier)
    QApplication.sendEvent(view, event)
    application.processEvents()
    assert view.probe_index >= 0
    assert '归一化风险' in QToolTip.text()
    assert 'mm⁻¹' in QToolTip.text() and '限速' in QToolTip.text()
    assert '限速' in window.probe_readout.text()
    assert window.risk_chart.probe is not None
    window.controls['path'].setChecked(False)
    assert view.probe_index == -1
    assert window.risk_chart.probe is None
    assert view.probe_at(QPointF(point)) == -1
    window.controls['path'].setChecked(True)
    assert view.probe_at(QPointF(point)) >= 0
    QApplication.sendEvent(view, QEvent(QEvent.Type.Leave))
    assert view.probe_index == -1


def test_main_view_double_click_resets_without_changing_camera_clip(window):
    view = window.main_view
    expected_far = view.first_person_far
    view.azimuth, view.elevation = 12., 18.
    view.distance = 10.
    view.pan[:] = 1.

    view.mouseDoubleClickEvent(None)

    assert (view.azimuth, view.elevation) == view.default_view
    assert view.distance == view.radius * 2.6
    assert view.first_person_far == expected_far
    assert not view.pan.any()
