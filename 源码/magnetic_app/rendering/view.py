import ctypes
import math
import logging
import time

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QToolTip

from ..domain.probe import risk_description
from .picking import pick_centerline

from .geometry import COLORS, STOPS, Domains, arrows, capsule, field_lengths, risk_color, segments, vertices


VERTEX_SHADER = '''#version 330 core
layout(location=0) in vec3 position;
layout(location=1) in vec3 normal;
layout(location=2) in vec3 color;
layout(location=3) in vec2 seen;
uniform mat4 mvp;
uniform mat4 model;
out vec3 worldPosition;
out vec3 worldNormal;
out vec3 vertexColor;
out vec2 seenTime;
void main() {
    worldPosition = (model * vec4(position,1)).xyz;
    worldNormal = mat3(model) * normal;
    vertexColor = color;
    seenTime = seen;
    gl_Position = mvp * model * vec4(position,1);
}'''

FRAGMENT_SHADER = '''#version 330 core
in vec3 worldPosition;
in vec3 worldNormal;
in vec3 vertexColor;
in vec2 seenTime;
uniform vec3 eye;
uniform float opacity;
uniform float playbackTime;
uniform float joinTime;
uniform int lighting;
uniform bool coverage;
out vec4 fragment;
void main() {
    vec3 base = vertexColor;
    if (coverage) {
        base = seenTime.x < -1.5 ? vec3(.25,.28,.34) : vec3(.72,.77,.83);
        if (seenTime.x >= 0. && seenTime.x <= min(playbackTime,joinTime)) base = vec3(.031,.498,.820);
        if (seenTime.y >= 0. && seenTime.y + joinTime <= playbackTime) base = vec3(.953,.420,.129);
    }
    if (lighting > 0) {
        vec3 normal = normalize(worldNormal);
        vec3 lightDirection = normalize(eye-worldPosition);
        float diffuse = abs(dot(normal,lightDirection));
        if (lighting == 2) {
            float distanceToEye = length(eye-worldPosition);
            float cutoff = pow(clamp(1.-pow(distanceToEye/160.,4.),0.,1.),2.);
            float lamp = 35. / max(distanceToEye,1.) * cutoff;
            vec3 linearBase = pow(vec3(.780,.498,.475),vec3(2.2));
            vec3 ambient = vec3(1.,.76,.65) * .65;
            vec3 linearColor = linearBase * (ambient + vec3(1.,.86,.77)*lamp*diffuse) / 3.14159265;
            linearColor += vec3(.076,.024,.020)*lamp*pow(diffuse,18.);
            base = pow(max(linearColor,vec3(0)),vec3(1./2.2));
        } else base *= .70 + .30*diffuse;
    }
    fragment = vec4(base,opacity);
}'''


def look_at(eye, target, up):
    forward = target-eye
    forward /= np.linalg.norm(forward)
    side = np.cross(forward, up)
    side /= max(np.linalg.norm(side), 1e-12)
    vertical = np.cross(side, forward)
    matrix = np.eye(4)
    matrix[:3, :3] = [side, vertical, -forward]
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def perspective(fov, aspect, near, far):
    focal = 1 / math.tan(math.radians(fov)/2)
    return np.array([[focal/aspect, 0, 0, 0], [0, focal, 0, 0], [0, 0, (far+near)/(near-far), 2*far*near/(near-far)], [0, 0, -1, 0]])


class Buffer:
    def __init__(self, data, mode=GL.GL_LINES):
        self.array = GL.glGenVertexArrays(1)
        self.buffer = GL.glGenBuffers(1)
        self.mode = mode
        self.upload(data)

    def upload(self, data):
        data = np.ascontiguousarray(data, dtype=np.float32)
        self.count = len(data)
        GL.glBindVertexArray(self.array)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.buffer)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_DYNAMIC_DRAW)
        for location, count, offset in [(0, 3, 0), (1, 3, 12), (2, 3, 24), (3, 2, 36)]:
            GL.glEnableVertexAttribArray(location)
            GL.glVertexAttribPointer(location, count, GL.GL_FLOAT, False, 44, ctypes.c_void_p(offset))
        GL.glBindVertexArray(0)

    def draw(self):
        GL.glBindVertexArray(self.array)
        GL.glDrawArrays(self.mode, 0, self.count)

    def release(self):
        GL.glDeleteBuffers(1, [self.buffer])
        GL.glDeleteVertexArrays(1, [self.array])


class SceneView(QOpenGLWidget):
    probed = Signal(int)
    renderFailed = Signal(str)

    def __init__(self, replay, first_person=False, parent=None):
        super().__init__(parent)
        self.replay = replay
        self.first_person = first_person
        self.frame = replay.frame(0)
        self.options = {'bowel': True, 'coverage': False, 'frustum': True, 'target': True, 'path': True, 'trajectory': False, 'robot': True, 'field': True, 'lines': False, 'slice': False, 'force': True, 'tangent': False, 'steering': True, 'torque': False, 'gradient': True, 'opacity': .22, 'scale': 1., 'frustum_length': 25., 'field_mode': 'total', 'field_scope': 'centerline', 'slice_plane': 'XY', 'view_mode': 'path'}
        self.options['coils'] = True
        self.domains = Domains(replay)
        bounds = np.array(replay.payload['geometry']['bounds'])
        self.center = bounds.mean(axis=0)
        self.radius = np.linalg.norm(bounds[1]-bounds[0])*.5
        self.distance = self.radius*2.6
        self.first_person_far = max(2000., float(np.linalg.norm(bounds[1] - bounds[0]) * 2.0))
        self.posture = replay.payload.get('patientPosture')
        self.default_view = (42., 48.) if self.posture else (42., 24.)
        self.azimuth, self.elevation = self.default_view
        self.pan = np.zeros(3)
        self.ready = False
        self.error = None
        self.buffers = []
        self.draw_times = []
        self.swap_times = []
        self.pose_draw_times = []
        self.last_drawn_pose = None
        self.last_dynamic = 0
        self.last_stream = 0
        self.last_dynamic_time = -1
        self.last_mouse = None
        self.drag_distance = 0
        self.probe_index = -1
        self.projection = np.eye(4)
        self.setMouseTracking(not first_person)
        self.setMinimumSize(240, 220)
        self.frameSwapped.connect(self.record_swap)

    def record_swap(self):
        if len(self.swap_times) < 30000:
            self.swap_times.append(time.perf_counter())

    def set_frame(self, frame):
        self.frame = frame
        self.update()

    def set_option(self, key, value):
        self.options[key] = value
        if key == 'path' and not value:
            self.clear_probe()
        self.last_dynamic = 0
        self.last_stream = 0
        self.update()

    def initializeGL(self):
        try:
            self.program = compileProgram(compileShader(VERTEX_SHADER, GL.GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER))
            self.uniforms = {key: GL.glGetUniformLocation(self.program, key) for key in ['mvp', 'model', 'eye', 'opacity', 'playbackTime', 'joinTime', 'lighting', 'coverage']}
            self.driver = {'vendor': GL.glGetString(GL.GL_VENDOR).decode(), 'renderer': GL.glGetString(GL.GL_RENDERER).decode(), 'version': GL.glGetString(GL.GL_VERSION).decode()}
            points, faces = self.replay.points, self.replay.faces
            triangles = points[faces]
            face_normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
            smooth = np.zeros_like(points)
            for corner in range(3):
                np.add.at(smooth, faces[:, corner], face_normals)
            smooth /= np.maximum(np.linalg.norm(smooth, axis=1, keepdims=True), 1e-12)
            seen = np.column_stack([self.replay.insertion_seen, self.replay.withdrawal_seen]).copy()
            seen[np.asarray(self.replay.masks['insertion_covered'], bool), 1] = -1
            self.wall = self.create_buffer(vertices(triangles.reshape(-1, 3), (.72, .77, .83), smooth[faces].reshape(-1, 3), np.repeat(seen, 3, axis=0)), GL.GL_TRIANGLES)
            if not self.first_person:
                self.path = self.create_buffer(segments(self.replay.centerline, risk_color(self.replay.source['risk_norm'])))
                direction = self.replay.payload.get('routeDirection')
                if direction:
                    markers = []
                    for position, color in [(direction['entry_position_mm'], (.0, .60, .38)),
                                            (direction['deep_position_mm'], (.55, .22, .75))]:
                        for offset in np.eye(3)*5.:
                            markers.append(segments([np.asarray(position)-offset,np.asarray(position)+offset],color))
                    self.route_markers = self.create_buffer(np.concatenate(markers))
                self.trajectory = self.create_buffer(segments(self.replay.positions, (.29, .47, .82)))
                self.body = self.create_buffer(capsule(), GL.GL_TRIANGLES)
                self.dynamic = self.create_buffer(vertices([]))
                self.coil_buffer = self.create_buffer(vertices([]))
                self.onboard = self.create_buffer(vertices([]))
                self.stream = self.create_buffer(vertices([]))
                self.slice = self.create_buffer(vertices([]), GL.GL_TRIANGLES)
                self.grid = self.create_buffer(self.make_grid())
                if self.posture:
                    lower, upper = np.asarray(self.replay.payload['geometry']['bounds'])
                    origin = np.array([lower[0]-10., lower[1]-10., upper[2]+15.])
                    self.gravity_origin = origin
                    self.gravity = self.create_buffer(arrows(origin, [0., 0., -1.], (.75, .18, .25), 60.))
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
            self.context().aboutToBeDestroyed.connect(self.cleanup)
            self.ready = True
        except Exception as exception:
            self.error = str(exception)
            self.renderFailed.emit(self.error)

    def create_buffer(self, data, mode=GL.GL_LINES):
        buffer = Buffer(data, mode)
        self.buffers.append(buffer)
        return buffer

    def make_grid(self):
        bounds = np.array(self.replay.payload['geometry']['bounds'])
        lower, upper = bounds[0]-15, bounds[1]+15
        if self.posture:
            lower[2] = upper[2] = self.posture['table_z_mm']
        coordinates = []
        for axis in [0, 1]:
            for value in np.arange(math.floor(lower[axis]/50)*50, upper[axis]+1, 50):
                start, end = lower.copy(), lower.copy()
                start[axis] = end[axis] = value
                end[1-axis] = upper[1-axis]
                coordinates.extend([start, end])
        return vertices(coordinates, (.83, .86, .90))

    def cleanup(self):
        if not self.ready:
            return
        self.makeCurrent()
        for buffer in self.buffers:
            buffer.release()
        GL.glDeleteProgram(self.program)
        self.buffers.clear()
        self.ready = False
        self.doneCurrent()

    def draw(self, buffer, model=None, lighting=0, opacity=1., coverage=False):
        GL.glUniformMatrix4fv(self.uniforms['model'], 1, GL.GL_TRUE, np.ascontiguousarray(np.eye(4) if model is None else model, dtype=np.float32))
        GL.glUniform1i(self.uniforms['lighting'], lighting)
        GL.glUniform1i(self.uniforms['coverage'], int(coverage))
        GL.glUniform1f(self.uniforms['opacity'], opacity)
        buffer.draw()

    def paintGL(self):
        try:
            self.render_scene()
        except Exception as exception:
            self.error = str(exception)
            logging.exception('OpenGL render failure')
            self.renderFailed.emit(self.error)
            self.ready = False

    def render_scene(self):
        started = time.perf_counter()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendEquation(GL.GL_FUNC_ADD)
        GL.glBlendFuncSeparate(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA, GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDisable(GL.GL_SCISSOR_TEST)
        GL.glColorMask(True, True, True, True)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glClearColor(*((.035, .047, .065, 1) if self.first_person else (.982, .987, .995, 1)))
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        if not self.ready:
            painter = QPainter(self)
            painter.setPen(QColor('#b82145'))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, '图形初始化失败：' + (self.error or '等待 OpenGL'))
            painter.end()
            return
        frame = self.frame
        width, height = max(1, self.width()), max(1, self.height())
        if self.first_person:
            ratio = self.devicePixelRatioF()
            viewport_width = min(width, height*4/3)
            viewport_height = viewport_width*3/4
            GL.glViewport(int((width-viewport_width)/2*ratio), int((height-viewport_height)/2*ratio), int(viewport_width*ratio), int(viewport_height*ratio))
            eye = frame.camera_position
            self.projection = perspective(60, 4/3, .2, self.first_person_far) @ look_at(eye, eye+frame.camera_rotation[:, 0], frame.camera_rotation[:, 2])
        else:
            ratio = self.devicePixelRatioF()
            GL.glViewport(0, 0, int(width*ratio), int(height*ratio))
            target = (frame.position if self.options['view_mode'] == 'follow' else self.center) + self.pan
            azimuth, elevation = np.radians([self.azimuth, self.elevation])
            distance = min(self.distance, 115) if self.options['view_mode'] == 'follow' else self.distance
            eye = target + distance*np.array([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)])
            self.projection = perspective(42, width/height, .2, 5000) @ look_at(eye, target, np.array([0., 0., 1.]))
        GL.glUseProgram(self.program)
        GL.glUniformMatrix4fv(self.uniforms['mvp'], 1, GL.GL_TRUE, np.ascontiguousarray(self.projection, dtype=np.float32))
        GL.glUniform3fv(self.uniforms['eye'], 1, eye)
        GL.glUniform1f(self.uniforms['playbackTime'], frame.time)
        GL.glUniform1f(self.uniforms['joinTime'], self.replay.join_time)
        if self.first_person:
            self.draw(self.wall, lighting=2)
        else:
            self.draw(self.grid)
            if self.posture:
                self.draw(self.gravity)
            if self.options['path']:
                self.draw(self.path)
                if self.replay.payload.get('routeDirection'):
                    self.draw(self.route_markers)
            if self.options['trajectory']:
                self.draw(self.trajectory)
            if self.options['robot']:
                model = np.eye(4)
                model[:3, :3] = frame.body_rotation*self.options['scale']
                model[:3, 3] = frame.position
                self.draw(self.body, model, lighting=1)
            if started-self.last_dynamic > .08 or abs(frame.time-self.last_dynamic_time) > 1.5:
                self.update_dynamic()
                self.last_dynamic = started
                self.last_dynamic_time = frame.time
            self.draw(self.dynamic)
            if self.options['coils'] and self.replay.coils is not None:
                orientation, anchor = self.replay.coils.pose_at(frame)
                transform = np.eye(4)
                transform[:3, :3] = orientation
                transform[:3, 3] = anchor
                self.draw(self.coil_buffer, transform)
            self.update_onboard()
            self.draw(self.onboard)
            if self.options['lines']:
                if started-self.last_stream > .4:
                    mode, scope = self.options['field_mode'], self.options['field_scope']
                    maximum = self.replay.registry['field_scope_ranges'][scope][mode][1]
                    self.stream.upload(self.domains.streamline_vertices(frame, scope, mode, maximum))
                    self.last_stream = started
                self.draw(self.stream, opacity=.65)
            if self.options['slice']:
                self.draw(self.slice, opacity=.55)
            if self.options['bowel']:
                GL.glDepthMask(GL.GL_FALSE)
                self.draw(self.wall, lighting=1, opacity=.86 if self.options['coverage'] else self.options['opacity'], coverage=self.options['coverage'])
                GL.glDepthMask(GL.GL_TRUE)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)
        if not self.first_person:
            painter = QPainter(self)
            painter.setPen(QColor('#718198'))
            painter.drawText(12, height-12, 'X / Y / Z  [mm]    左键旋转 · 右键平移 · 滚轮缩放 · 双击复位')
            if self.posture:
                painter.drawText(12, height-32, '仰卧（模型假设） · 水平参考面 Z=0 · 重力 −Z（9.81 m/s²）')
                point = self.project(self.gravity_origin[None])[0]
                if 5 < point[0] < width-80 and 14 < point[1] < height-50:
                    painter.setPen(QColor('#bf2e40'))
                    painter.drawText(int(point[0])+9, int(point[1])+15, '重力 g')
                    painter.setPen(QColor('#718198'))
            if self.replay.coils is not None and self.options['coils']:
                positions, currents, moments = self.replay.coils.at(frame)
                description = '有限铁芯 52×32 mm · 绕组外径 56 mm' if self.replay.coils.finite else '圆环仅示意'
                painter.drawText(12, 22, '八线圈磁源 · '+description+' · 电流 [A]')
                for row_index in range(2):
                    values = '   '.join(f'C{index+1} {currents[index]:+.2f}' for index in range(row_index*4,row_index*4+4))
                    painter.drawText(12, 40+row_index*17, values)
                projected = self.project(positions)
                for index, point in enumerate(projected):
                    if 8 < point[0] < width-80 and 70 < point[1] < height-40:
                        painter.drawText(int(point[0])+8,int(point[1]),f'C{index+1}')
            if self.options['path']:
                self.paint_colorbar(painter, width-48, 35, 105, '风险', 0., 1.)
                direction = self.replay.payload.get('routeDirection')
                if direction:
                    points = self.project(np.asarray([direction['entry_position_mm'],direction['deep_position_mm']]))
                    for point, text, color in zip(points,['肛门入口','肠道深部终点'],['#009961','#8c38bf']):
                        if 5 < point[0] < width-90 and 15 < point[1] < height-40:
                            painter.setPen(QColor(color))
                            painter.drawText(int(point[0])+8,int(point[1])-8,text)
                    painter.setPen(QColor('#718198'))
            if self.options['field'] or self.options['slice']:
                field_range = self.replay.registry['field_scope_ranges'][self.options['field_scope']][self.options['field_mode']]
                unit = 'μT' if self.options['field_mode'] == 'turning' else 'mT'
                self.paint_colorbar(painter, width-48, height-165, 105, f'|B| {unit}', *field_range)
            bounds = np.array(self.replay.payload['geometry']['bounds'])
            for axis, name in enumerate(['X', 'Y', 'Z']):
                points = np.tile(bounds[0], (3, 1))
                points[:, axis] = np.linspace(bounds[0, axis], bounds[1, axis], 3)
                projected = self.project(points)
                for point, value in zip(projected, points[:, axis]):
                    if 5 < point[0] < width-75 and 12 < point[1] < height-30:
                        painter.drawText(int(point[0]), int(point[1]), f'{name} {value:.0f}')
            if self.probe_index >= 0:
                point = self.project(self.replay.centerline[self.probe_index:self.probe_index+1])[0]
                painter.setPen(QPen(QColor('#e64c65'), 2))
                painter.drawEllipse(int(point[0])-5, int(point[1])-5, 10, 10)
            painter.end()
        if len(self.draw_times) < 30000:
            self.draw_times.append((time.perf_counter()-started)*1000)
        if frame.time != self.last_drawn_pose:
            self.last_drawn_pose = frame.time
            if len(self.pose_draw_times) < 30000:
                self.pose_draw_times.append(time.perf_counter())

    def paint_colorbar(self, painter, horizontal, top, length, title, lower, upper):
        gradient = QLinearGradient(horizontal, top+length, horizontal, top)
        for stop, color in zip(STOPS, COLORS):
            gradient.setColorAt(float(stop), QColor.fromRgbF(*map(float, color)))
        painter.fillRect(horizontal, top, 9, length, gradient)
        painter.setPen(QColor('#718198'))
        painter.drawText(horizontal-10, top-9, title)
        for fraction in [0., .5, 1.]:
            painter.drawText(horizontal+13, int(top+length*(1-fraction))+4, f'{lower+(upper-lower)*fraction:.2g}')

    def update_dynamic(self):
        frame, options = self.frame, self.options
        data = []
        if options['field']:
            points = self.domains.points(frame, options['field_scope'])
            points = points[self.domains.valid(points, frame, options['field_scope'])]
            field = self.replay.field(frame, points, options['field_mode'])
            magnitudes = np.linalg.norm(field, axis=1)
            maximum = self.replay.registry['field_scope_ranges'][options['field_scope']][options['field_mode']][1]
            data.append(arrows(points, field, risk_color(magnitudes/maximum), field_lengths(magnitudes, options['field_mode'], options['field_scope'])))
        self.update_coils()
        for key, prefix, unit, color, gain in [('force', 'applied_force', 'mN', (.10, .51, .37), 3), ('torque', 'applied_torque', 'uN_m', (.72, .13, .40), 450)]:
            if options[key]:
                vector = self.replay.vector(frame, prefix, unit)
                data.append(arrows(frame.position, vector, color, min(60, np.linalg.norm(vector)*gain)))
        field = self.replay.vector(frame, 'applied_field', 'mT')
        steering = (field - np.dot(field, frame.body_rotation[:, 0])*frame.body_rotation[:, 0])*1000
        if options['steering']:
            data.append(arrows(frame.position, steering, (.50, .33, .74), min(60, np.linalg.norm(steering)*12)))
        if options['tangent']:
            tangent = self.replay.vector(frame, 'ref_axis', 'unit')
            force = self.replay.vector(frame, 'applied_force', 'mN')
            projected = tangent*np.dot(force, tangent)
            data.append(arrows(frame.position, projected, (.86, .47, .13), min(60, np.linalg.norm(projected)*10)))
        if options['gradient']:
            gradient = self.replay.gradient(frame)
            data.append(arrows(np.tile(frame.position, (3, 1)), gradient.T, [[.82, .33, .33], [.27, .65, .38], [.28, .48, .85]], np.clip(np.linalg.norm(gradient, axis=0)*60, 2, 32)))
        self.dynamic.upload(np.concatenate(data) if data else vertices([]))
        if options['slice']:
            self.update_slice()

    def update_coils(self):
        source = self.replay.coils
        data = []
        if source is not None:
            positions, currents, moments = source.at(self.frame)
            if source.quaternions is not None:
                positions = source.offsets*1000.
            ring_angles = np.linspace(0., 2*np.pi, 33)
            for position, axis, current in zip(positions, source.axes, currents):
                reference = np.array([0., 0., 1.]) if abs(axis[2]) < .9 else np.array([0., 1., 0.])
                horizontal = np.cross(axis, reference)
                horizontal /= np.linalg.norm(horizontal)
                vertical = np.cross(axis, horizontal)
                color = (.08, .57, .71) if current >= 0 else (.91, .38, .12)
                sizes = [(12., 0., color)]
                if source.finite:
                    sizes = [(source.design['winding_outer_radius_m']*1000., source.design['winding_length_m']*1000., color),
                             (source.design['core_radius_m']*1000., source.design['core_length_m']*1000., (.40, .43, .49))]
                for radius, length, outline in sizes:
                    circle = radius*(np.cos(ring_angles)[:, None]*horizontal+np.sin(ring_angles)[:, None]*vertical)
                    for side in [-1., 1.]:
                        data.append(segments(position+side*length/2*axis+circle, outline))
                    for direction in [horizontal, -horizontal, vertical, -vertical]:
                        data.append(segments([position-length/2*axis+radius*direction,
                                              position+length/2*axis+radius*direction], outline))
                data.append(arrows(position, axis*np.sign(current), color, 18.))
        self.coil_buffer.upload(np.concatenate(data) if data else vertices([]))

    def update_onboard(self):
        frame, options = self.frame, self.options
        data = []
        if options['frustum']:
            length = options['frustum_length']
            half_height = length*np.tan(np.pi/6)
            half_width = half_height*4/3
            corners = np.array([[length, -half_width, -half_height], [length, half_width, -half_height], [length, half_width, half_height], [length, -half_width, half_height]]) @ frame.camera_rotation.T + frame.camera_position
            lines = []
            for index in range(4):
                lines.extend([frame.camera_position, corners[index], corners[index], corners[(index+1)%4]])
            data.append(vertices(lines, (.84, .46, .12)))
        if options['target']:
            data.append(vertices([frame.camera_position, frame.target], (.84, .26, .54)))
        self.onboard.upload(np.concatenate(data) if data else vertices([]))

    def update_slice(self):
        frame, options = self.frame, self.options
        axes = {'XY': [0, 1], 'XZ': [0, 2], 'YZ': [1, 2]}[options['slice_plane']]
        bounds = self.domains.bounds_for(frame, options['field_scope'])
        first, second = np.meshgrid(np.linspace(*bounds[axes[0]], 25), np.linspace(*bounds[axes[1]], 25), indexing='ij')
        points = np.tile(frame.position, (25, 25, 1))
        points[:, :, axes[0]], points[:, :, axes[1]] = first, second
        flattened = points.reshape(-1, 3)
        valid = self.domains.valid(flattened, frame, options['field_scope'])
        indices = np.arange(625).reshape(25, 25)
        triangles = np.stack([indices[:-1, :-1], indices[1:, :-1], indices[:-1, 1:], indices[:-1, 1:], indices[1:, :-1], indices[1:, 1:]], axis=-1).reshape(-1, 3)
        triangles = triangles[np.all(valid[triangles], axis=1)]
        maximum = self.replay.registry['field_scope_ranges'][options['field_scope']][options['field_mode']][1]
        values = np.linalg.norm(self.replay.field(frame, flattened, options['field_mode']), axis=1)
        self.slice.upload(vertices(flattened[triangles].reshape(-1, 3), risk_color(values[triangles].reshape(-1)/maximum)))

    def project(self, points):
        clip = np.column_stack([points, np.ones(len(points))]) @ self.projection.T
        normalized = clip[:, :3] / np.maximum(clip[:, 3, None], 1e-9)
        screen = np.column_stack([(normalized[:, 0]+1)*self.width()/2, (1-normalized[:, 1])*self.height()/2])
        screen[clip[:, 3] <= 0] = 1e8
        return screen

    def mousePressEvent(self, event):
        self.last_mouse = event.position()
        self.drag_distance = 0

    def mouseMoveEvent(self, event):
        if self.first_person:
            return
        if event.buttons() and self.last_mouse is not None:
            delta = event.position()-self.last_mouse
            self.drag_distance += abs(delta.x())+abs(delta.y())
            self.clear_probe()
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.azimuth -= delta.x()*.4
                self.elevation = float(np.clip(self.elevation+delta.y()*.4, -85, 85))
            else:
                azimuth = math.radians(self.azimuth)
                self.pan += np.array([-math.sin(azimuth), math.cos(azimuth), 0])*delta.x()*self.distance/800
                self.pan[2] += delta.y()*self.distance/800
            self.last_mouse = event.position()
            self.update()
        else:
            self.probe_at(event.position())

    def clear_probe(self):
        QToolTip.hideText()
        if self.probe_index >= 0:
            self.probe_index = -1
            self.probed.emit(-1)
            self.update()

    def leaveEvent(self, event):
        if not self.first_person:
            self.clear_probe()
        super().leaveEvent(event)

    def probe_at(self, point):
        if self.first_person or not self.ready or not self.options.get('path', True):
            self.clear_probe()
            return -1
        clip = np.column_stack([self.replay.centerline, np.ones(len(self.replay.centerline))]) @ self.projection.T
        index = pick_centerline(clip, (point.x(), point.y()), (self.width(), self.height()))
        if index < 0:
            self.clear_probe()
            return -1
        self.probe_index = index
        self.probed.emit(index)
        QToolTip.showText(self.mapToGlobal(point.toPoint()) + QPoint(14, 16), risk_description(self.replay.source, index), self)
        self.update()
        return index

    def mouseReleaseEvent(self, event):
        if self.drag_distance < 4 and not self.first_person:
            self.probe_at(event.position())
        self.last_mouse = None

    def wheelEvent(self, event):
        if not self.first_person:
            self.distance = float(np.clip(self.distance*math.exp(-event.angleDelta().y()/900), 15, self.radius*12))
            self.update()
            event.accept()

    def mouseDoubleClickEvent(self, event):
        self.azimuth, self.elevation = self.default_view
        self.distance = self.radius*2.6
        self.pan[:] = 0
        self.update()
