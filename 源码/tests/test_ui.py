from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'构建'/'依赖'))
sys.path.insert(0, str(ROOT/'源码'))

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import QApplication, QComboBox, QPlainTextEdit, QToolTip

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
    for key, readout in [('propulsion_parameters', window.propulsion_readout), ('turning_parameters', window.turning_readout)]:
        window.controls[key].setChecked(False)
        assert readout.isHidden()
        window.controls[key].setChecked(True)
        assert not readout.isHidden()


def test_all_state_charts_and_final_values(window):
    for index in range(window.state_mode.count()):
        window.state_mode.setCurrentIndex(index)
        assert len(window.state_chart.series) == 2
    window.seek(window.replay.duration)
    window.tick()
    window.update_readouts()
    assert '198,000.33' in window.coverage_values['final_covered_area_mm2'][0].text()
    assert '20,909.91' in window.coverage_values['withdrawal_added_area_mm2'][0].text()
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
