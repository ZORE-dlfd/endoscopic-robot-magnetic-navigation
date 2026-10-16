from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'构建'/'依赖'))
sys.path.insert(0, str(ROOT/'源码'))

from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit

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
    assert '194,069.35' in window.coverage_values['final_covered_area_mm2'][0].text()
    assert '16,978.93' in window.coverage_values['withdrawal_added_area_mm2'][0].text()
    assert window.state_table.rowCount() == 146


def test_native_formula_dialog(application, window):
    contents = []

    def inspect_dialog():
        dialog = application.activeModalWidget()
        assert dialog is not None
        text = dialog.findChild(QPlainTextEdit)
        contents.append(text.toPlainText())
        dialog.accept()

    QTimer.singleShot(50, inspect_dialog)
    window.show_docs()
    assert len(contents) == 1 and len(contents[0]) > 1000
