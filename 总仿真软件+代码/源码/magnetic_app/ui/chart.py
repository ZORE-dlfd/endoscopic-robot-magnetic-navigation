import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget, QSlider, QStyle, QStyleOptionSlider


class Chart(QWidget):
    selected = Signal(float)
    hovered = Signal(float)

    def __init__(self, xvalues, series, xlabel='时间 / s', parent=None):
        super().__init__(parent)
        self.setMinimumHeight(235)
        self.setMouseTracking(True)
        self.xlabel = xlabel
        self.cursor = 0.
        self.probe = None
        self.cache_size = None
        self.set_series(xvalues, series)

    def set_series(self, xvalues, series):
        self.xvalues = np.asarray(xvalues)
        self.series = [(name, np.asarray(values, dtype=float), color, axis) for name, values, color, axis in series]
        self.xmin, self.xmax = float(self.xvalues.min()), float(self.xvalues.max())
        self.limits = {}
        for axis in [0, 1]:
            values = [values[np.isfinite(values)] for name, values, color, selected_axis in self.series if selected_axis == axis]
            if values:
                joined = np.concatenate(values)
                lower, upper = min(0, joined.min()), joined.max()
                span = max(upper-lower, 1e-8)
                self.limits[axis] = (lower-span*.06, upper+span*.12)
        self.cache_size = None
        self.update()

    def set_cursor(self, time):
        self.cursor = time
        self.update()

    def plot_rect(self):
        return QRectF(58, 36, max(1, self.width()-116), max(1, self.height()-83))

    def rebuild(self):
        rect = self.plot_rect()
        self.paths = []
        stride = max(1, len(self.xvalues)//max(1, int(rect.width()*2)))
        indices = np.unique(np.r_[np.arange(0, len(self.xvalues), stride), len(self.xvalues)-1])
        for name, values, color, axis in self.series:
            lower, upper = self.limits[axis]
            coords = np.column_stack([rect.left()+(self.xvalues[indices]-self.xmin)/max(1e-8, self.xmax-self.xmin)*rect.width(), rect.bottom()-(values[indices]-lower)/(upper-lower)*rect.height()])
            path = QPainterPath()
            fresh = True
            for horizontal, vertical in coords:
                if not np.isfinite(vertical):
                    fresh = True
                elif fresh:
                    path.moveTo(float(horizontal), float(vertical))
                    fresh = False
                else:
                    path.lineTo(float(horizontal), float(vertical))
            self.paths.append((path, color))
        self.cache_size = self.size()

    def paintEvent(self, event):
        if self.cache_size != self.size():
            self.rebuild()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.plot_rect()
        painter.fillRect(self.rect(), QColor('white'))
        painter.setPen(QColor('#dfe6ef'))
        for tick in range(5):
            fraction = tick/4
            horizontal = rect.left()+rect.width()*fraction
            vertical = rect.bottom()-rect.height()*fraction
            painter.drawLine(QPointF(horizontal, rect.top()), QPointF(horizontal, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), vertical), QPointF(rect.right(), vertical))
            painter.setPen(QColor('#7b8899'))
            painter.drawText(QRectF(horizontal-28, rect.bottom()+8, 56, 20), Qt.AlignmentFlag.AlignCenter, f'{self.xmin+fraction*(self.xmax-self.xmin):.0f}')
            for axis, (lower, upper) in self.limits.items():
                painter.drawText(QRectF(0 if axis == 0 else rect.right()+4, vertical-9, 52, 18), Qt.AlignmentFlag.AlignRight if axis == 0 else Qt.AlignmentFlag.AlignLeft, f'{lower+fraction*(upper-lower):.2g}')
            painter.setPen(QColor('#dfe6ef'))
        painter.save()
        painter.setClipRect(rect)
        for path, color in self.paths:
            painter.setPen(QPen(QColor(color), 1.6))
            painter.drawPath(path)
        for value, color in [(self.cursor, '#284b72'), (self.probe, '#e44d68')]:
            if value is not None:
                horizontal = rect.left()+(value-self.xmin)/max(1e-8, self.xmax-self.xmin)*rect.width()
                painter.setPen(QPen(QColor(color), 1.2, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(horizontal, rect.top()), QPointF(horizontal, rect.bottom()))
        painter.restore()
        left = rect.left()
        for name, values, color, axis in self.series:
            painter.setPen(QColor(color))
            painter.drawText(QPointF(left, 21), '━ '+name)
            left += painter.fontMetrics().horizontalAdvance('━ '+name)+18
        painter.setPen(QColor('#7b8899'))
        painter.drawText(QRectF(rect.left(), self.height()-23, rect.width(), 20), Qt.AlignmentFlag.AlignCenter, self.xlabel)
        painter.end()

    def coordinate(self, point):
        rect = self.plot_rect()
        return self.xmin+np.clip((point.x()-rect.left())/rect.width(), 0, 1)*(self.xmax-self.xmin)

    def mousePressEvent(self, event):
        self.selected.emit(float(self.coordinate(event.position())))

    def mouseMoveEvent(self, event):
        value = float(self.coordinate(event.position()))
        self.hovered.emit(value)
        index = int(np.argmin(np.abs(self.xvalues-value)))
        text = f'{self.xlabel}: {value:.2f}\n'+'\n'.join(f'{name}: {values[index]:.4g}' for name, values, color, axis in self.series)
        self.setToolTip(text)


class TimelineSlider(QSlider):
    def __init__(self, fraction, parent=None):
        super().__init__(Qt.Orientation.Horizontal, parent)
        self.fraction = fraction
        self.setRange(0, 1000000)
        self.setMinimumHeight(28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(event.position().x())-7, self.width()-14))
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.isSliderDown():
            self.setValue(QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), int(event.position().x())-7, self.width()-14))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setSliderDown(False)
        event.accept()

    def paintEvent(self, event):
        super().paintEvent(event)
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        groove = self.style().subControlRect(QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self)
        horizontal = groove.left()+self.fraction*groove.width()
        painter = QPainter(self)
        painter.setPen(QPen(QColor('#e44756'), 2))
        painter.drawLine(int(horizontal), 1, int(horizontal), self.height()-1)
        painter.end()
