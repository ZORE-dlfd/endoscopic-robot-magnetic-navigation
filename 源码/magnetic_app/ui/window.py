import csv
import json
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QFrame,
    QPushButton, QComboBox, QCheckBox, QScrollArea, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QDialog, QPlainTextEdit, QAbstractItemView, QSizePolicy)

from ..domain.clock import PlaybackClock
from ..domain.algorithms import algorithm_sections
from ..domain.probe import risk_description
from ..domain.replay import INSERTION, WITHDRAWAL
from ..rendering.view import SceneView
from .chart import Chart, TimelineSlider
from .style import STYLE


def label(text='', role=None, wrap=False):
    widget = QLabel(text)
    if role:
        widget.setObjectName(role)
    widget.setWordWrap(wrap)
    return widget


def row(*widgets):
    layout = QHBoxLayout()
    layout.setSpacing(9)
    for widget in widgets:
        if isinstance(widget, str):
            widget = label(widget, 'muted')
        layout.addWidget(widget)
    return layout


def button(text, callback, primary=False):
    widget = QPushButton(text)
    widget.clicked.connect(callback)
    if primary:
        widget.setObjectName('primary')
    return widget


def card(eyebrow='', title=''):
    widget = QFrame()
    widget.setObjectName('card')
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(17, 15, 17, 15)
    layout.setSpacing(10)
    if eyebrow:
        layout.addWidget(label(eyebrow, 'eyebrow'))
    if title:
        layout.addWidget(label(title, 'title'))
    return widget, layout


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


class MainWindow(QMainWindow):
    def __init__(self, replay):
        super().__init__()
        self.replay = replay
        self.clock = PlaybackClock(replay.duration)
        self.current_frame = replay.frame(0)
        version = '八线圈仰卧版' if replay.payload.get('patientPosture') else '八线圈磁源版'
        if replay.payload.get('routeDirection'):
            version += ' · 肛门进镜'
        self.setWindowTitle('Magnetic · 胶囊视觉导航实验台 — clon001 '+version)
        self.setWindowIcon(QIcon(str(replay.path / 'app.ico')))
        self.setStyleSheet(STYLE)
        self.resize(1650, 985)
        self.setMinimumSize(1120, 740)
        self.controls = {}
        self.last_readout = 0.
        self.dirty = True
        self.readout_pending = False
        self.tick_times = []
        self.ui_times = []
        self.render_errors = []
        self.build_ui()
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.tick)
        self.timer.start()
        self.shortcuts = [QShortcut(QKeySequence('Space'), self, activated=self.toggle_play), QShortcut(QKeySequence('Home'), self, activated=self.reset)]
        self.update_readouts()

    def build_ui(self):
        container = QWidget()
        container.setObjectName('container')
        outer = QVBoxLayout(container)
        outer.setContentsMargins(18, 10, 18, 12)
        outer.setSpacing(10)
        self.setCentralWidget(container)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        page = QWidget()
        page.setObjectName('page')
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 7, 12)
        layout.setSpacing(14)
        self.scroll.setWidget(page)
        outer.addWidget(self.scroll, 1)
        logo = label('M')
        logo.setFixedSize(42, 42)
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo.setStyleSheet('background:#1d334e;color:white;font-size:23px;font-weight:800;border-radius:12px;')
        brand = QWidget()
        brand_layout = QVBoxLayout(brand)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.addWidget(label('<b style="font-size:22px">Magnetic</b>　胶囊视觉导航实验台'))
        brand_layout.addWidget(label('双视角 · 八线圈磁源驱动 · 原生可视化仿真', 'muted'))
        header = row(logo, brand)
        header.addStretch()
        header.addWidget(label('离线回放', 'badge'))
        header.addWidget(button('参数与公式 ↗', self.show_docs))
        layout.addLayout(header)
        profile, profile_layout = card('ROBOT PROFILE')
        profile.setMinimumHeight(140)
        selector = QComboBox()
        selector.addItem('胶囊 · 只减速 · 进镜→退镜')
        selector.setMinimumWidth(250)
        self.controls['robot'] = selector
        profile_row = row(selector)
        for title, text in [('尺寸', '12 × 6 mm'), ('质量', f'{self.replay.robot["mass_g"]:.4f} g'), ('磁偶极矩', f'{self.replay.robot["magnetic_moment_A_m2"]:.5f} A·m²'), ('转动惯量', 'Jx 2.184e−9 / Jy,Jz 5.746e−9 kg·m²'), ('自由度', '6 DOF · 单偶极不直接控轴向滚转')]:
            profile_row.addWidget(label(f'<span style="color:#8b97a8;font-size:11px">{title}</span><br><br><b style="font-size:17px">{text}</b>'))
        profile_layout.addLayout(profile_row)
        layout.addWidget(profile)
        probe = QFrame()
        probe.setObjectName('probe')
        probe_layout = QHBoxLayout(probe)
        self.probe_readout = label('中心线取点　鼠标悬停 / 点击中心线，联动风险与力 / 梯度图', wrap=True)
        probe_layout.addWidget(self.probe_readout, 1)
        probe_layout.addWidget(button('清除取点', self.clear_probe))
        layout.addWidget(probe)
        workbench = QHBoxLayout()
        workbench.setSpacing(14)
        left = QVBoxLayout()
        left.setSpacing(14)
        views = QHBoxLayout()
        views.setSpacing(14)
        views.addWidget(self.make_main_view(), 1)
        views.addWidget(self.make_camera_view(), 1)
        left.addLayout(views)
        left.addWidget(self.make_evaluation())
        left.addWidget(self.make_force_chart())
        left.addWidget(self.make_state_chart())
        left.addWidget(self.make_risk_chart())
        workbench.addLayout(left, 1)
        sidebar = self.make_sidebar()
        sidebar.setFixedWidth(310)
        workbench.addWidget(sidebar, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(workbench)
        layout.addWidget(label('clon001 / 原网格 Colon001 · 离线几何与磁动力学仿真；不是实拍视频，也不代表临床安全性验证。', 'muted', True))
        outer.addWidget(self.make_timeline())

    def combo(self, key, entries, selected=None):
        widget = QComboBox()
        for title, value in entries:
            widget.addItem(title, value)
        if selected is not None:
            widget.setCurrentIndex(max(0, widget.findData(selected)))
        widget.currentIndexChanged.connect(lambda index: self.main_view.set_option(key, widget.itemData(index)))
        self.controls[key] = widget
        return widget

    def check(self, key, text, checked):
        widget = QCheckBox(text)
        widget.setChecked(checked)
        widget.toggled.connect(lambda value: self.main_view.set_option(key, value))
        self.controls[key] = widget
        return widget

    def make_main_view(self):
        panel, layout = card('PATH · ROBOT · MAGNETIC FIELD', '第三人称 · 肠道与机器人')
        self.main_view = SceneView(self.replay)
        self.main_view.setFixedHeight(400)
        self.main_view.probed.connect(self.set_probe)
        self.main_view.renderFailed.connect(self.render_error)
        layout.addLayout(row(self.check('bowel', '显示肠道', True), self.combo('view_mode', [('全路径', 'path'), ('跟随机器人', 'follow')]), self.combo('scale', [('实体 1×', 1.), ('展示 3×', 3.)])))
        layout.addLayout(row(self.combo('field_mode', [('完整场 · mT', 'total'), ('去偏置场 · mT', 'residual'), ('转向分量 · μT', 'turning')]), self.combo('field_scope', [('覆盖中心线', 'centerline'), ('局部', 'local'), ('整体', 'whole')], 'centerline')))
        layers = [
            [('coverage', '已观测 / 退镜补拍', False), ('frustum', '相机视锥', True), ('target', '相机目标', True), ('path', '风险中心线', True)],
            [('trajectory', '实际轨迹', False), ('robot', '机器人', True), ('field', '磁场箭头', True), ('lines', '磁感线', False), ('slice', '切片', False)],
            [('force', '总磁力', True), ('tangent', '沿路径推进', False), ('steering', '转向分量', True), ('torque', '转矩', False), ('gradient', '梯度', True)],
        ]
        for group in layers:
            line = row(*[self.check(key, text, checked) for key, text, checked in group])
            line.addStretch()
            layout.addLayout(line)
        layout.addLayout(row('透明度', self.combo('opacity', [('0.12', .12), ('0.22', .22), ('0.45', .45)], .22), '视锥', self.combo('frustum_length', [('15 mm', 15.), ('25 mm', 25.), ('40 mm', 40.)], 25.), self.combo('slice_plane', [('XY', 'XY'), ('XZ', 'XZ'), ('YZ', 'YZ')], 'XY')))
        route_hint = '肛门入口 → 肠道深部 → 肛门入口。' if self.replay.payload.get('routeDirection') else ''
        source_hint = ('八线圈有限绕组＋有限铁芯＋近似互耦；磁场显示同源查表，非有限元或实测场。'
                       if self.replay.coils is not None and self.replay.coils.finite
                       else '八线圈磁源空间叠加；含电流限幅，仍是点偶极仿真，非有限元或实测场。')
        self.field_hint = label(route_hint+source_hint, 'muted', True)
        layout.addWidget(self.field_hint)
        layout.addWidget(label(f'<span style="color:{INSERTION}">● 进镜有效覆盖</span>　<span style="color:{WITHDRAWAL}">● 退镜新增</span>　<span style="color:#687688">● 尚未覆盖</span>'))
        layout.addWidget(self.main_view)
        layout.addWidget(label('场强颜色：紫→蓝→绿→黄→红（低→高）；箭头长度按当前显示范围缩放。', 'muted', True))
        return panel

    def make_camera_view(self):
        panel, layout = card('ONBOARD CAMERA', '第一人称 · 胶囊前端')
        self.phase_badge = label('进镜观察 · 0.00 s', 'badge')
        layout.addWidget(self.phase_badge, 0, Qt.AlignmentFlag.AlignRight)
        camera_method = QComboBox()
        camera_method.addItem('进镜前瞻 / 退镜短时预测')
        camera_method.setEnabled(False)
        camera_method.setToolTip('进镜保留原前瞻控制，退镜采用密集搜索＋短时预测＋原PD。播放读取离线轨迹。')
        self.controls['camera_method'] = camera_method
        layout.addLayout(row(camera_method, 'FOV 60° · 4:3 · 同一几何 / 光照'))
        self.camera_view = SceneView(self.replay, first_person=True)
        self.camera_view.setFixedHeight(400)
        self.camera_view.renderFailed.connect(self.render_error)
        layout.addWidget(self.camera_view)
        self.camera_readout = label('', wrap=True)
        layout.addWidget(self.camera_readout)
        self.joint_readout = label('', 'muted', True)
        layout.addWidget(self.joint_readout)
        layout.addWidget(label('相机固定在机身前端 5.8 mm；沿机身前向 45° 锥内转动。', 'muted', True))
        layout.addStretch()
        return panel

    def make_evaluation(self):
        panel, layout = card('ROUND TRIP · UNIQUE SURFACE COVERAGE', '进镜后退镜：补拍面积统计')
        toolbar = row(label('只减速进镜 → 连续退镜　|　分母：完整肠壁面积　|　有效观察阈值 0.4 s', 'muted', True))
        toolbar.addStretch()
        toolbar.addWidget(button('导出评价 JSON ↓', self.export_evaluation))
        layout.addLayout(toolbar)
        layout.addWidget(label('A_in = Σ aᵢ Mᵢ　　A_new = Σ aᵢ (1−Mᵢ) Wᵢ　　A_final = A_in + A_new<br>覆盖率提升 = 100 × A_new / A_total；每个面片只计第一次有效观察，插值不增加有效面积。', 'muted', True))
        grid = QGridLayout()
        self.coverage_values = {}
        definitions = [('insertion_covered_area_mm2', '进镜有效覆盖', 'mm²', INSERTION), ('withdrawal_added_area_mm2', '退镜首次补拍', 'mm²', WITHDRAWAL), ('final_covered_area_mm2', '当前累计覆盖', 'mm²', '#273c56'), ('coverage_pct', '完整肠壁覆盖率', '%', '#273c56'), ('coverage_gain_percentage_points', '退镜覆盖率提升', '百分点', WITHDRAWAL), ('uncovered_area_mm2', '当前未覆盖面积', 'mm²', '#6c7e94')]
        for index, (key, title, unit, color) in enumerate(definitions):
            tile = QFrame()
            tile.setStyleSheet(f'QFrame {{background:#f7f9fc;border:1px solid #e4ebf3;border-left:4px solid {color};border-radius:8px;}} QLabel {{border:0;}}')
            tile_layout = QVBoxLayout(tile)
            tile_layout.addWidget(label(title, 'muted'))
            value = label(f'0.00 <small>{unit}</small>')
            value.setStyleSheet(f'font-size:21px;font-weight:700;color:{color};')
            tile_layout.addWidget(value)
            self.coverage_values[key] = (value, unit)
            grid.addWidget(tile, index//3, index%3)
        layout.addLayout(grid)
        rows = self.replay.metric_rows
        self.coverage_chart = Chart(self.replay.metric_times, [('进镜有效覆盖 / mm²', [entry['insertion_covered_area_mm2'] for entry in rows], INSERTION, 0), ('退镜新增 / mm²', [entry['withdrawal_added_area_mm2'] for entry in rows], WITHDRAWAL, 0)])
        self.coverage_chart.selected.connect(self.seek)
        layout.addWidget(self.coverage_chart)
        table = QTableWidget(0, 3)
        table.setHorizontalHeaderLabels(['全表面积统计（褶皱后保留在分母）', '结果', '单位'])
        table.verticalHeader().hide()
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        definitions = [('进镜有效覆盖', 'insertion_covered_area_mm2', 'mm²'), ('退镜首次补拍', 'withdrawal_added_area_mm2', 'mm²'), ('退镜结束累计覆盖', 'final_covered_area_mm2', 'mm²'), ('完整肠壁总面积', 'total_lumen_area_mm2', 'mm²'), ('实际约束越限采样点', 'constraint_violating_steps', '点'), ('退镜后未覆盖面积', 'uncovered_area_mm2', 'mm²'), ('进镜覆盖率', 'insertion_coverage_pct', '%'), ('退镜新增覆盖率', 'coverage_gain_percentage_points', '百分点'), ('退镜结束覆盖率', 'final_coverage_pct', '%'), ('退镜时长', 'duration_s', 's'), ('等待次数', 'wait_count', '次')]
        table.setRowCount(len(definitions))
        for index, (title, key, unit) in enumerate(definitions):
            for column, text in enumerate([title, f'{self.replay.summary[key]:,.2f}', unit]):
                item = QTableWidgetItem(text)
                if '退镜首次' in title or '退镜新增' in title:
                    from PySide6.QtGui import QColor
                    item.setForeground(QColor(WITHDRAWAL))
                if title == '进镜有效覆盖' or title == '进镜覆盖率':
                    from PySide6.QtGui import QColor
                    item.setForeground(QColor(INSERTION))
                table.setItem(index, column, item)
        table.setFixedHeight(375)
        layout.addWidget(table)
        layout.addWidget(label('本版重算八线圈磁驱动进退镜，交界不重置位置或姿态。退镜保留滑窗密集搜索＋1.2 s预测＋原PD；蓝橙面积按新轨迹重算。未覆盖不等于永远不可见；离散零越限不代表临床安全。', 'muted', True))
        return panel

    def make_force_chart(self):
        panel, layout = card('PROPULSION · FORCE & GRADIENT', '沿路径推进：磁力与梯度')
        data = self.replay.data
        force = np.column_stack([data[f'applied_force_{axis}_mN'] for axis in 'xyz'])
        tangent = np.column_stack([data[f'ref_axis_{axis}_unit'] for axis in 'xyz'])
        self.force_chart = Chart(self.replay.times, [('沿路径磁力 / mN', np.sum(force*tangent, axis=1), '#db7820', 0), ('梯度范数 / T·m⁻¹', data['applied_gradient_norm_T_m'], '#287dc0', 1)])
        self.force_chart.selected.connect(self.seek)
        layout.addWidget(self.force_chart)
        return panel

    def make_state_chart(self):
        panel, layout = card('TIME SERIES', '状态曲线')
        self.state_mode = QComboBox()
        for name, mode in [('运动：速度与加速度', 'motion'), ('磁场：场强与梯度', 'magnetics'), ('转向：垂直场与转矩', 'steering'), ('跟踪：位置与轴向误差', 'tracking')]:
            self.state_mode.addItem(name, mode)
        layout.addWidget(self.state_mode)
        self.state_chart = Chart(self.replay.times, [])
        self.state_mode.currentIndexChanged.connect(self.change_state_chart)
        self.state_chart.selected.connect(self.seek)
        layout.addWidget(self.state_chart)
        self.change_state_chart()
        return panel

    def change_state_chart(self):
        definitions = {'motion': [('速度 / mm·s⁻¹', 'actual_speed_mm_s'), ('加速度 / mm·s⁻²', 'actual_acceleration_norm_mm_s2')], 'magnetics': [('磁场 / mT', 'applied_field_norm_mT'), ('梯度 / T·m⁻¹', 'applied_gradient_norm_T_m')], 'steering': [('垂直场 / μT', 'applied_field_perpendicular_mT'), ('转矩 / μN·m', 'applied_torque_norm_uN_m')], 'tracking': [('位置误差 / mm', 'position_error_mm'), ('轴向误差 / °', 'axis_error_deg')]}
        mode = self.state_mode.currentData()
        series = [(name, self.replay.data[key]*(1000 if mode == 'steering' and index == 0 else 1), ['#277ccb', '#da5d63'][index], index) for index, (name, key) in enumerate(definitions[mode])]
        self.state_chart.set_series(self.replay.times, series)

    def make_risk_chart(self):
        panel, layout = card('CENTERLINE PROFILE', '风险与限速 · 中心线位置联动')
        source = self.replay.source
        self.risk_chart = Chart(source['arc_s_mm'], [('归一化风险', source['risk_norm'], '#d85b64', 0), ('目标速度 / mm·s⁻¹', source['speed_target_mm_s'], '#277ccb', 1)], '弧长 / mm')
        self.risk_chart.selected.connect(self.seek_arc)
        self.risk_chart.hovered.connect(lambda arc: self.set_probe(int(np.argmin(abs(source['arc_s_mm']-arc)))))
        layout.addWidget(self.risk_chart)
        return panel

    def make_sidebar(self):
        panel, layout = card('LIVE STATE', '当前时刻的核心状态')
        grid = QGridLayout()
        self.live_values = {}
        definitions = [('risk_norm', '当前风险', ''), ('actual_speed_mm_s', '实际速度', 'mm/s'), ('actual_acceleration_norm_mm_s2', '实际加速度', 'mm/s²'), ('position_error_mm', '位置误差', 'mm'), ('applied_field_norm_mT', '施加总场 |B|', 'mT'), ('axis_error_deg', '长轴方向误差', '°')]
        for index, (key, title, unit) in enumerate(definitions):
            tile = QFrame()
            tile.setStyleSheet('QFrame{background:#f5f8fc;border:1px solid #e5ebf3;border-radius:9px;}QLabel{border:0;}')
            inner = QVBoxLayout(tile)
            inner.addWidget(label(title, 'muted'))
            value = label()
            value.setStyleSheet('font-size:19px;font-weight:700;color:#243b59;')
            inner.addWidget(value)
            grid.addWidget(tile, index//2, index%2)
            self.live_values[key] = (value, unit)
        layout.addLayout(grid)
        self.coil_table = None
        if self.replay.coils is not None and self.replay.coils.finite:
            layout.addWidget(label('八路线圈 · 铁芯 52×32 mm\n绕组长 44 mm / 外径 56 mm · 1200 匝\n每路 ±10 A · 电阻 12.310 Ω', 'muted', True))
            self.coil_table = QTableWidget(8, 4)
            self.coil_table.setHorizontalHeaderLabels(['线圈', 'I / A', 'P / W', '磁化 / %'])
            self.coil_table.verticalHeader().hide()
            self.coil_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.coil_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            self.coil_table.setFixedHeight(262)
            for coil_index in range(8):
                for column in range(4):
                    self.coil_table.setItem(coil_index, column, QTableWidgetItem(''))
            layout.addWidget(self.coil_table)
            self.coil_power = label('', 'muted', True)
            layout.addWidget(self.coil_power)
            layout.addWidget(button('各路线圈完整参数 ↗', self.show_coil_parameters))
        propulsion_toggle = QCheckBox('推进参数 · 需求 / 施加')
        propulsion_toggle.setChecked(True)
        self.controls['propulsion_parameters'] = propulsion_toggle
        layout.addWidget(propulsion_toggle)
        self.propulsion_readout = label('', wrap=True)
        propulsion_toggle.toggled.connect(self.propulsion_readout.setVisible)
        layout.addWidget(self.propulsion_readout)
        turning_toggle = QCheckBox('转向参数 · 需求 / 施加')
        turning_toggle.setChecked(True)
        self.controls['turning_parameters'] = turning_toggle
        layout.addWidget(turning_toggle)
        self.turning_readout = label('', wrap=True)
        turning_toggle.toggled.connect(self.turning_readout.setVisible)
        layout.addWidget(self.turning_readout)
        layout.addWidget(label('实际梯度 G（T/m）', 'muted'))
        self.gradient_readout = label()
        self.gradient_readout.setStyleSheet('font-family:Consolas;font-size:12px;background:#f4f7fb;padding:9px;border-radius:8px;')
        layout.addWidget(self.gradient_readout)
        layout.addWidget(label('全状态 · 数值可滚动查看', 'muted'))
        self.state_keys = list(self.replay.data)
        self.state_table = QTableWidget(len(self.state_keys), 2)
        self.state_table.setHorizontalHeaderLabels(['状态', '当前值'])
        self.state_table.verticalHeader().hide()
        self.state_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.state_table.setColumnWidth(1, 94)
        self.state_table.setFixedHeight(460)
        self.state_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.state_table.setAlternatingRowColors(True)
        labels = self.replay.payload['labels']
        for index, key in enumerate(self.state_keys):
            name = str(labels.get(key, key))
            self.state_table.setItem(index, 0, QTableWidgetItem(name))
            self.state_table.item(index, 0).setToolTip(key+'\n'+name)
            self.state_table.setItem(index, 1, QTableWidgetItem(''))
        layout.addWidget(self.state_table)
        layout.addWidget(button('导出回放 CSV ↓', self.export_csv))
        layout.addWidget(label('6 DOF 刚体仿真，胶囊单磁矩无法直接控制轴向滚转。全空间箭头由八个有限磁源同模型查表叠加，单路±10 A；未模拟热、电感或肠壁接触。', 'muted', True))
        return panel

    def make_timeline(self):
        panel, layout = card()
        layout.setContentsMargins(15, 9, 15, 9)
        toolbar = QHBoxLayout()
        self.play_button = button('▶ 播放', self.toggle_play, True)
        toolbar.addWidget(self.play_button)
        toolbar.addWidget(button('↺ 复位', self.reset))
        self.time_readout = label()
        self.time_readout.setMinimumWidth(145)
        toolbar.addWidget(self.time_readout)
        self.slider = TimelineSlider(self.replay.join_time/self.replay.duration)
        self.slider.valueChanged.connect(lambda value: self.seek(value/1000000*self.replay.duration))
        self.slider.sliderPressed.connect(self.clock.begin_seek)
        self.slider.sliderReleased.connect(self.clock.end_seek)
        toolbar.addWidget(self.slider, 1)
        self.speed_combo = QComboBox()
        for speed in [.25, .5, 1, 2, 5, 10]:
            self.speed_combo.addItem(f'{speed:g}×', speed)
        self.speed_combo.setCurrentIndex(4)
        self.speed_combo.currentIndexChanged.connect(lambda index: self.clock.set_speed(self.speed_combo.itemData(index)))
        toolbar.addWidget(self.speed_combo)
        self.status_badge = label('clon001 已加载', 'badge')
        toolbar.addWidget(self.status_badge)
        layout.addLayout(toolbar)
        directed = bool(self.replay.payload.get('routeDirection'))
        self.entry_caption = label('肛门入口 · 开始进镜' if directed else '起点 · 进镜', 'muted')
        captions = row(self.entry_caption)
        captions.addStretch(1)
        marker = label(('肠道深部 · 开始退镜' if directed else '交接 / 开始退镜')+f'　{self.replay.join_time:.2f} s')
        self.turn_caption = marker
        marker.setStyleSheet('color:#d74e60;font-size:11px;font-weight:600;')
        captions.addWidget(marker)
        captions.addStretch(1)
        self.exit_caption = label('回到肛门入口' if directed else '退镜终点', 'muted')
        captions.addWidget(self.exit_caption)
        layout.addLayout(captions)
        return panel

    def tick(self):
        started = time.perf_counter()
        if not self.clock.playing and not self.dirty:
            if self.readout_pending and started-self.last_readout >= .10:
                self.update_readouts()
                self.last_readout = started
                self.readout_pending = False
            return
        current = self.clock.current()
        self.current_frame = self.replay.frame(current)
        self.main_view.set_frame(self.current_frame)
        self.camera_view.set_frame(self.current_frame)
        self.dirty = False
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(round(current/self.replay.duration*1000000))
            self.slider.blockSignals(False)
        self.time_readout.setText(f'{current:.2f} / {self.replay.duration:.2f} s')
        self.readout_pending = True
        if started-self.last_readout >= .10:
            self.update_readouts()
            self.last_readout = started
            self.readout_pending = False
        if current >= self.replay.duration and self.clock.playing:
            self.clock.set_playing(False)
            self.play_button.setText('▶ 播放')
        if len(self.tick_times) < 30000:
            self.tick_times.append((time.perf_counter()-started)*1000)

    def update_readouts(self):
        started = time.perf_counter()
        frame, replay = self.current_frame, self.replay
        self.phase_badge.setText(f'{replay.phase(frame.time)} · {frame.time:.2f} s')
        for key, (widget, unit) in self.live_values.items():
            widget.setText(f'{replay.value(frame, key):.3f}<small> {unit}</small>')
        if self.coil_table is not None:
            coil_rows = replay.coils.parameter_rows(frame)
            for coil_index, coil in enumerate(coil_rows):
                values = [coil['coil'], f'{coil["current_a"]:+.3f}', f'{coil["power_w"]:.1f}',
                          f'{coil["saturation_fraction"]*100:.1f}']
                for column, text in enumerate(values):
                    self.coil_table.item(coil_index, column).setText(text)
            total_power = sum(coil['power_w'] for coil in coil_rows)
            self.coil_power.setText(f'当前总阻性功耗 {total_power:.1f} W\n磁化为均匀铁芯模型的饱和占比；假设主动冷却，未计算温升或电感。')
        angles = replay.camera_value(frame, 'joint_angle_deg')
        velocities = replay.camera_value(frame, 'joint_velocity_deg_s')
        accelerations = replay.camera_value(frame, 'joint_acceleration_deg_s2')
        fmt = lambda values: ' / '.join(f'{value:.1f}' for value in values)
        preview_label = '前瞻' if frame.time < replay.join_time else '方向射线（仅示意）'
        preview = replay.camera_value(frame, 'preview_mm') if frame.time < replay.join_time else 20.
        self.camera_readout.setText(f'相机位置　<b>{fmt(frame.camera_position)} mm</b><br>偏航 / 俯仰 / 滚转　<b>{fmt(angles)} °</b><br>{preview_label}　<b>{preview:.1f} mm</b>　目标偏差　<b>{replay.camera_value(frame, "aim_error_deg"):.1f} °</b><br>可达姿态跟踪滞后　<b>{replay.camera_value(frame, "tracking_error_deg"):.1f} °</b>　锥外剩余　<b>{replay.camera_value(frame, "cone_residual_deg"):.1f} °</b>')
        roll_values = replay.payload['cameraMotion']['roundtrip_new']['roll_rate_deg_s']
        roll = roll_values[frame.camera_index]*(1-frame.camera_amount)+roll_values[frame.camera_index+1]*frame.camera_amount
        self.joint_readout.setText(f'关节角速度 {fmt(velocities)} °/s · 角加速度 {fmt(accelerations)} °/s²\n光轴滚转速率 {roll:.2f} °/s')
        force = replay.vector(frame, 'applied_force', 'mN')
        tangent = replay.vector(frame, 'ref_axis', 'unit')
        parallel = np.dot(force, tangent)
        required_force = replay.vector(frame, 'required_force', 'mN')
        required_parallel = np.dot(required_force, tangent)
        self.propulsion_readout.setText(f'需求 / 施加（mN）<br>沿路径分量　<b>{required_parallel:.5f} / {parallel:.5f}</b><br>横向分量　{np.linalg.norm(required_force-required_parallel*tangent):.5f} / {np.linalg.norm(force-parallel*tangent):.5f}<br>总磁力　{np.linalg.norm(required_force):.5f} / {np.linalg.norm(force):.5f}')
        self.turning_readout.setText(f'需求 / 施加<br>垂直场（μT）　<b>{replay.value(frame, "required_field_perpendicular_mT")*1000:.4f} / {replay.value(frame, "applied_field_perpendicular_mT")*1000:.4f}</b><br>可控转矩（μN·m）　{replay.value(frame, "required_torque_norm_uN_m"):.6f} / {replay.value(frame, "applied_torque_norm_uN_m"):.6f}<br>轴向不可用（μN·m）　{replay.value(frame, "required_unavailable_torque_norm_uN_m"):.6f} / {replay.value(frame, "applied_unavailable_torque_norm_uN_m"):.6f}')
        gradient = replay.gradient(frame)
        self.gradient_readout.setText('\n'.join('  '.join(f'{value: .5f}' for value in line) for line in gradient))
        for index, key in enumerate(self.state_keys):
            value = replay.value(frame, key)
            self.state_table.item(index, 1).setText(value if isinstance(value, str) else f'{value:.6g}')
        metrics = dict(replay.metrics(frame.time))
        metrics['uncovered_area_mm2'] = replay.summary['total_lumen_area_mm2']-metrics['final_covered_area_mm2']
        for key, (widget, unit) in self.coverage_values.items():
            widget.setText(f'{metrics.get(key, 0):,.2f} <small>{unit}</small>')
        self.coverage_chart.set_cursor(frame.time)
        self.force_chart.set_cursor(frame.time)
        self.state_chart.set_cursor(frame.time)
        self.risk_chart.set_cursor(replay.value(frame, 'arc_s_mm'))
        if len(self.ui_times) < 30000:
            self.ui_times.append((time.perf_counter()-started)*1000)

    def render_error(self, text):
        self.render_errors.append(text)
        self.status_badge.setText('图形初始化失败 · 查看日志')
        self.status_badge.setToolTip(text)

    def seek(self, time_value):
        self.clock.seek(float(time_value))
        self.dirty = True

    def seek_arc(self, arc):
        phase_mask = self.replay.times >= self.replay.join_time if self.current_frame.time >= self.replay.join_time else self.replay.times < self.replay.join_time
        candidates = np.flatnonzero(phase_mask)
        index = candidates[np.argmin(abs(self.replay.data['arc_s_mm'][candidates]-arc))]
        self.seek(self.replay.times[index])

    def toggle_play(self):
        if self.clock.current() >= self.replay.duration:
            self.clock.seek(0)
        self.clock.set_playing(not self.clock.playing)
        self.play_button.setText('Ⅱ 暂停' if self.clock.playing else '▶ 播放')
        self.dirty = True

    def reset(self):
        self.clock.set_playing(False)
        self.seek(0)
        self.play_button.setText('▶ 播放')

    def set_probe(self, index):
        if index < 0 or index >= len(self.replay.centerline):
            self.clear_probe()
            return
        self.main_view.probe_index = index
        source = self.replay.source
        arc = source['arc_s_mm'][index]
        phase_mask = self.replay.times >= self.replay.join_time if self.current_frame.time >= self.replay.join_time else self.replay.times < self.replay.join_time
        candidates = np.flatnonzero(phase_mask)
        nearest = candidates[np.argmin(abs(self.replay.data['arc_s_mm'][candidates]-arc))]
        time_value = self.replay.times[nearest]
        self.probe_readout.setText(risk_description(source, index).replace('\n', '　') + f'　对应当前阶段 t={time_value:.2f} s')
        self.risk_chart.probe = arc
        self.force_chart.probe = time_value
        self.state_chart.probe = time_value
        for chart in [self.risk_chart, self.force_chart, self.state_chart]:
            chart.update()
        self.main_view.update()

    def clear_probe(self):
        self.main_view.probe_index = -1
        self.main_view.update()
        self.probe_readout.setText('中心线取点　鼠标悬停 / 点击中心线，联动风险与力 / 梯度图')
        for chart in [self.risk_chart, self.force_chart, self.state_chart]:
            chart.probe = None
            chart.update()

    def write_evaluation(self, filename):
        report = {'example': 'clon001', 'original_model': 'Colon001', 'summary': self.replay.summary, 'prediction': self.replay.prediction, 'magnetic_simulation': self.replay.payload.get('magneticSimulation'), 'settings': self.replay.settings, 'round_trip': self.replay.payload['roundTrip'], 'rows': self.replay.metric_rows, 'display': {'insertion': INSERTION, 'withdrawal_added': WITHDRAWAL, 'pose_interpolation_used_for_area': True, 'area_sampling_s': {'insertion': 0.1, 'withdrawal': 0.1}, 'render_frames_add_coverage': False}}
        Path(filename).write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

    def write_csv(self, filename):
        with Path(filename).open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(self.state_keys)
            writer.writerows(zip(*[self.replay.data[key] for key in self.state_keys]))

    def export_evaluation(self):
        filename, selected_filter = QFileDialog.getSaveFileName(self, '导出评价', str(Path.home()/'Desktop'/'clon001_进退镜评价.json'), 'JSON (*.json)')
        if filename:
            self.safe_export(self.write_evaluation, filename)

    def export_csv(self):
        filename, selected_filter = QFileDialog.getSaveFileName(self, '导出回放数据', str(Path.home()/'Desktop'/'clon001_完整回放.csv'), 'CSV (*.csv)')
        if filename:
            self.safe_export(self.write_csv, filename)

    def safe_export(self, writer, filename):
        try:
            writer(filename)
            QMessageBox.information(self, '导出完成', f'已保存到：\n{filename}')
        except Exception as exception:
            QMessageBox.warning(self, '导出失败', str(exception))

    def show_coil_parameters(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('八路线圈参数 · 仿真假设，不是实测硬件')
        dialog.resize(1100, 700)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label('铁芯尺寸：长 52 mm、直径 32 mm；绕组外径是 56 mm，不是 32 mm。各路独立调电流，阵列保持相对布局整体平移。', wrap=True))
        rows = self.replay.coils.parameter_rows(self.current_frame)
        definitions = [('线圈', 'coil'), ('铁芯长 / mm', 'core_length_mm'), ('铁芯直径 / mm', 'core_diameter_mm'),
            ('绕组长 / mm', 'winding_length_mm'), ('绕组内径 / mm', 'winding_inner_diameter_mm'),
            ('绕组外径 / mm', 'winding_outer_diameter_mm'), ('匝数', 'turns'), ('铜线直径 / mm', 'copper_diameter_mm'),
            ('含绝缘直径 / mm', 'insulated_diameter_mm'), ('相对磁导率', 'relative_permeability'),
            ('假设饱和极化 / T', 'saturation_polarization_t'), ('电流上限 / A', 'current_limit_a'),
            ('电阻 / Ω', 'resistance_ohm'), ('理想驱动电压上限 / V', 'ideal_driver_voltage_limit_v'),
            ('距阵列中心 / mm', 'anchor_distance_mm'), ('相对中心 XYZ / mm', 'offset_mm'), ('轴线单位向量', 'axis'),
            ('当前中心 XYZ / mm', 'position_mm'), ('当前电流 / A', 'current_a'), ('当前阻性电压 / V', 'voltage_v'),
            ('当前阻性功耗 / W', 'power_w'), ('平均磁化饱和占比', 'saturation_fraction')]
        table = QTableWidget(len(definitions), 8)
        table.setHorizontalHeaderLabels([coil['coil'] for coil in rows])
        table.setVerticalHeaderLabels([title for title, key in definitions])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for row_index, (title, key) in enumerate(definitions):
            for column, coil in enumerate(rows):
                value = coil[key]
                text = ' / '.join(f'{component:.4g}' for component in value) if isinstance(value, list) else (f'{value:.6g}' if isinstance(value, (float, int)) else value)
                table.setItem(row_index, column, QTableWidgetItem(text))
        table.resizeColumnsToContents()
        layout.addWidget(table)
        layout.addWidget(label('此表固定在打开时刻。仅铁芯尺寸由用户指定，其余为假设；近似考虑铁芯互耦和平均磁化饱和检查，不含非线性 B–H、温升、电感动态、人体/机械臂碰撞。未做实物或独立有限元验证。', 'muted', True))
        layout.addWidget(button('关闭', dialog.accept))
        dialog.exec()

    def show_docs(self):
        dialog = QDialog(self)
        dialog.setWindowTitle('算法、参数与适用边界')
        dialog.resize(980, 760)
        layout = QVBoxLayout(dialog)
        sections = algorithm_sections(self.replay.path)
        selector = QComboBox(dialog)
        selector.setObjectName('algorithm_selector')
        selector.addItems(list(sections))
        layout.addWidget(selector)
        text = QPlainTextEdit(dialog)
        text.setReadOnly(True)
        text.setObjectName('algorithm_text')
        text.setPlainText(sections[selector.currentText()])
        selector.currentTextChanged.connect(lambda title: text.setPlainText(sections[title]))
        layout.addWidget(text)
        if self.replay.prediction:
            metrics = self.replay.prediction['metrics']
            layout.addWidget(label(f'当前退镜：预测方案 · 新增 {metrics["withdrawal_added_area_mm2"]:,.2f} mm² · '
                                  f'实际离散越限 {metrics["constraint_violating_steps"]} 点 · '
                                  f'离线规划均值 {metrics["planning_mean_ms"]:.1f} ms（不是视频帧耗时）', 'muted', True))
        layout.addWidget(button('关闭', dialog.accept))
        dialog.exec()
