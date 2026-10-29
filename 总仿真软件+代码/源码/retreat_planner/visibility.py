import ctypes
import time

import numpy as np
from OpenGL import GL
from OpenGL.GL.shaders import compileProgram, compileShader
from PySide6.QtGui import QGuiApplication, QOffscreenSurface, QOpenGLContext, QSurfaceFormat

from .geometry import forward_frame


APPLICATION = None
VERTEX = '''#version 330 core
layout(location=0) in vec3 position;
uniform mat4 mvp;
flat out uint faceId;
void main() {
    faceId = uint(gl_VertexID / 3) + 1u;
    gl_Position = mvp * vec4(position, 1.0);
}'''
FRAGMENT = '''#version 330 core
flat in uint faceId;
layout(location=0) out uint fragment;
void main() { fragment = faceId; }
'''


class Visibility:
    def __init__(self, scene, atlas_size=192):
        global APPLICATION
        APPLICATION = QGuiApplication.instance() or QGuiApplication(['retreat-validation'])
        self.scene = scene
        self.atlas_size = atlas_size
        self.draw_count = 0
        self.draw_seconds = 0.
        self.closed = False
        surface_format = QSurfaceFormat()
        surface_format.setVersion(3, 3)
        surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        surface_format.setDepthBufferSize(24)
        surface_format.setSamples(0)
        self.surface = QOffscreenSurface()
        self.surface.setFormat(surface_format)
        self.surface.create()
        self.context = QOpenGLContext()
        self.context.setFormat(surface_format)
        if not self.surface.isValid() or not self.context.create() or not self.context.makeCurrent(self.surface):
            raise RuntimeError('Cannot create hidden OpenGL 3.3 context; no results were generated')
        self.environment = {name: GL.glGetString(value).decode('utf-8', errors='replace') for name, value in
                            [('vendor', GL.GL_VENDOR), ('renderer', GL.GL_RENDERER), ('version', GL.GL_VERSION)]}
        self.program = compileProgram(compileShader(VERTEX, GL.GL_VERTEX_SHADER), compileShader(FRAGMENT, GL.GL_FRAGMENT_SHADER))
        self.uniform = GL.glGetUniformLocation(self.program, 'mvp')
        self.vao = GL.glGenVertexArrays(1)
        self.vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        vertices = np.ascontiguousarray(scene.points[scene.faces].reshape(-1, 3), dtype=np.float32)
        self.vertex_count = len(vertices)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, False, 12, ctypes.c_void_p(0))
        self.fbo = GL.glGenFramebuffers(1)
        self.texture = GL.glGenTextures(1)
        self.depth = GL.glGenRenderbuffers(1)
        self.size = None
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glDepthFunc(GL.GL_LESS)
        for flag in [GL.GL_BLEND, GL.GL_CULL_FACE, GL.GL_MULTISAMPLE, GL.GL_DITHER, GL.GL_FRAMEBUFFER_SRGB]:
            GL.glDisable(flag)

    def _resize(self, width, height):
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.fbo)
        if self.size == (width, height):
            return
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_R32UI, width, height, 0, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.texture, 0)
        GL.glBindRenderbuffer(GL.GL_RENDERBUFFER, self.depth)
        GL.glRenderbufferStorage(GL.GL_RENDERBUFFER, GL.GL_DEPTH_COMPONENT24, width, height)
        GL.glFramebufferRenderbuffer(GL.GL_FRAMEBUFFER, GL.GL_DEPTH_ATTACHMENT, GL.GL_RENDERBUFFER, self.depth)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])
        if GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
            raise RuntimeError('Incomplete face-ID framebuffer')
        self.size = (width, height)

    def pixels(self, origin, frame, width=None, height=None, fov=None):
        started = time.perf_counter()
        settings = self.scene.settings
        width = int(width or settings['evaluation_width'])
        height = int(height or settings['evaluation_height'])
        fov = float(fov or settings['fov_vertical_deg'])
        self.context.makeCurrent(self.surface)
        self._resize(width, height)
        GL.glViewport(0, 0, width, height)
        GL.glClearBufferuiv(GL.GL_COLOR, 0, np.zeros(4, dtype=np.uint32))
        GL.glClear(GL.GL_DEPTH_BUFFER_BIT)
        view = np.eye(4)
        view[:3, :3] = np.array([-frame[:, 1], frame[:, 2], -frame[:, 0]])
        view[:3, 3] = -view[:3, :3] @ origin
        near, far = settings['near_mm'], settings['far_mm']
        scale = 1 / np.tan(np.deg2rad(fov / 2))
        projection = np.zeros((4, 4))
        projection[0, 0], projection[1, 1] = scale / (width / height), scale
        projection[2, 2], projection[2, 3] = -(far + near) / (far - near), -2 * far * near / (far - near)
        projection[3, 2] = -1
        GL.glUseProgram(self.program)
        GL.glUniformMatrix4fv(self.uniform, 1, GL.GL_TRUE, np.ascontiguousarray(projection @ view, dtype=np.float32))
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        raw = GL.glReadPixels(0, 0, width, height, GL.GL_RED_INTEGER, GL.GL_UNSIGNED_INT)
        pixels = np.frombuffer(raw, dtype=np.uint32).reshape(height, width)
        if pixels.max(initial=0) > len(self.scene.faces):
            raise RuntimeError('Face-ID render returned invalid IDs')
        self.draw_count += 1
        self.draw_seconds += time.perf_counter() - started
        return pixels

    def quality_mask(self, origin):
        settings = self.scene.settings
        delta = self.scene.centers - origin
        distance = np.linalg.norm(delta, axis=1)
        incidence = np.abs(np.einsum('ij,ij->i', self.scene.normals, delta)) / np.maximum(distance, 1e-8)
        return ((distance >= settings['min_observation_distance_mm']) &
                (distance <= settings['max_observation_distance_mm']) &
                (incidence >= np.cos(np.deg2rad(settings['max_incidence_deg']))))

    def visible(self, origin, frame):
        pixels = self.pixels(origin, frame)
        counts = np.bincount(pixels.ravel(), minlength=len(self.scene.faces) + 1)[1:]
        return np.flatnonzero(self.quality_mask(origin) & (counts >= self.scene.settings['min_projected_pixels']))

    def atlas(self, origin):
        visible = np.zeros(len(self.scene.faces), dtype=bool)
        for direction in np.r_[np.eye(3), -np.eye(3)]:
            up = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([0., 1., 0.])
            pixels = self.pixels(origin, forward_frame(direction, up), self.atlas_size, self.atlas_size, 90.)
            ids = np.unique(pixels)
            visible[ids[ids > 0] - 1] = True
        return np.flatnonzero(visible & self.quality_mask(origin))

    def close(self):
        if self.closed:
            return
        self.context.makeCurrent(self.surface)
        GL.glDeleteBuffers(1, [self.vbo])
        GL.glDeleteVertexArrays(1, [self.vao])
        GL.glDeleteFramebuffers(1, [self.fbo])
        GL.glDeleteRenderbuffers(1, [self.depth])
        GL.glDeleteTextures([self.texture])
        GL.glDeleteProgram(self.program)
        self.context.doneCurrent()
        self.surface.destroy()
        self.closed = True
