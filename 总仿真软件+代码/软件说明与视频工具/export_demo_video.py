# -*- coding: utf-8 -*-
"""
三视图展示视频：左=第三人称(线圈+肠道+机器人+视锥+蓝橙覆盖)，
右上=第一人称内镜，右下=TURBO深度热力图。进镜段+退镜段共约60秒。
"""
import sys
import time as _t
from pathlib import Path

SOFTWARE_DIR = Path(__file__).resolve().parent
SRC_DIR = SOFTWARE_DIR.parent / '源码'
ASSETS_DIR = SOFTWARE_DIR / '_internal' / 'assets'
CACHE_DIR = SOFTWARE_DIR / '缓存数据' / '3d545ecb0a1a5d67b5c12e7ad84c83671bb944a7bef23ab380fdfb2dba176f53'
sys.path.insert(0, str(SRC_DIR))

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QSurfaceFormat
from PySide6.QtWidgets import QApplication
import imageio.v2 as imageio

from magnetic_app.domain.replay import Replay
from magnetic_app.rendering.view import SceneView
from magnetic_app.depth.cache import DepthMapCache
from magnetic_app.depth.client import colorize


FPS = 30
# 输出画布
CW, CH = 1920, 1080
# 左：第三人称；右上：第一人称；右下：深度
MAIN_W, MAIN_H = 1280, 1080
CAM_W, CAM_H = 640, 480      # 4:3
DEPTH_W, DEPTH_H = 640, 518  # 深度图原始正方形


def qimage_to_rgb(img: QImage) -> np.ndarray:
    img = img.convertToFormat(QImage.Format.Format_RGB888)
    w, h = img.width(), img.height()
    bits = img.constBits()
    data = bits.tobytes() if hasattr(bits, 'tobytes') else bytes(bits)
    return np.frombuffer(data, np.uint8).reshape(h, img.bytesPerLine())[:, :w*3].reshape(h, w, 3).copy()


def resize_arr(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    return np.asarray(Image.fromarray(arr).resize((w, h), Image.BILINEAR))


def main():
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    surface = QSurfaceFormat()
    surface.setVersion(3, 3)
    surface.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface.setDepthBufferSize(24)
    surface.setSamples(4)
    surface.setSwapInterval(0)
    QSurfaceFormat.setDefaultFormat(surface)

    app = QApplication(sys.argv[:1])
    replay = Replay(ASSETS_DIR)

    depth_cache = DepthMapCache(CACHE_DIR, '3d545ecb0a1a5d67b5c12e7ad84c83671bb944a7bef23ab380fdfb2dba176f53', len(replay.camera_times))
    print('[info] depth cache valid =', depth_cache.is_valid, 'shape=', depth_cache.shape, flush=True)

    # 第三人称
    main_view = SceneView(replay, first_person=False)
    main_view.resize(MAIN_W, MAIN_H)
    opts = main_view.options
    opts['bowel'] = True
    opts['coverage'] = True      # 蓝橙覆盖
    opts['frustum'] = True        # 视锥
    opts['target'] = True
    opts['path'] = True
    opts['robot'] = True
    opts['coils'] = True
    opts['field'] = True
    opts['opacity'] = 0.22
    opts['frustum_length'] = 25.
    opts['view_mode'] = 'path'
    main_view.setWindowTitle('第三人称')
    main_view.show()

    # 第一人称
    cam_view = SceneView(replay, first_person=True)
    cam_view.resize(CAM_W, CAM_H)
    cam_view.setWindowTitle('第一人称')
    cam_view.show()

    state = {'ready': [False, False]}
    def poll():
        state['ready'][0] = main_view.ready
        state['ready'][1] = cam_view.ready
        if not all(state['ready']):
            QTimer.singleShot(30, poll)
    QTimer.singleShot(30, poll)
    for _ in range(300):
        app.processEvents()
        if all(state['ready']):
            break
        _t.sleep(0.02)
    print('[info] views ready =', state['ready'], flush=True)

    # 第三人称相机：距离要同时框住完整肠道 + 完整八线圈阵列
    main_view.distance = main_view.radius * 2.05
    print('[info] main_view distance =', main_view.distance, 'radius =', main_view.radius, flush=True)

    # 30秒精华：3段×10s
    #  进镜前瞻 t=40~50s：胶囊在肠腔中段，第一人称看前方腔道
    #  进镜过弯 t=160~170s：经过 167s 高曲率弯（曲率0.070）
    #  退镜补拍 t=380~390s：经过 385s 退镜弯（曲率0.067）
    seg1 = np.linspace(40.0, 50.0, int(10*FPS), endpoint=False)
    seg2 = np.linspace(160.0, 170.0, int(10*FPS), endpoint=False)
    seg3 = np.linspace(380.0, 390.0, int(10*FPS), endpoint=False)
    times = np.concatenate([seg1, seg2, seg3])
    n = len(times)
    print(f'[info] 共 {n} 帧 @ {FPS}fps = {n/FPS:.1f}s', flush=True)

    out_path = SOFTWARE_DIR / '方案可行性展示_三视图.mp4'
    writer = imageio.get_writer(
        str(out_path), fps=FPS, codec='libx264', quality=8, macro_block_size=8,
        ffmpeg_log_level='warning',
        output_params=['-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p'],
    )

    t0 = _t.time()
    try:
        for i, t in enumerate(times):
            frame = replay.frame(float(t))
            main_view.set_frame(frame)
            cam_view.set_frame(frame)
            main_view.repaint()
            cam_view.repaint()
            for _ in range(6):
                app.processEvents()
            main_img = qimage_to_rgb(main_view.grabFramebuffer())
            cam_img = qimage_to_rgb(cam_view.capture_rgb())
            depth = depth_cache.get_interpolated(frame.camera_index, round(frame.camera_amount, 2))
            depth_rgb = colorize(depth)

            # 缩放到目标尺寸
            main_r = resize_arr(main_img, MAIN_W, MAIN_H)
            cam_r = resize_arr(cam_img, CAM_W, CAM_H)
            depth_r = resize_arr(depth_rgb, DEPTH_W, DEPTH_H)

            canvas = np.full((CH, CW, 3), 18, np.uint8)  # 深灰底
            canvas[0:MAIN_H, 0:MAIN_W] = main_r
            canvas[0:CAM_H, MAIN_W:CAM_W+MAIN_W] = cam_r
            canvas[CAM_H:CAM_H+DEPTH_H, MAIN_W:DEPTH_W+MAIN_W] = depth_r

            writer.append_data(canvas)
            if (i+1) % 300 == 0 or i == 0:
                eta = (n-i-1)*(_t.time()-t0)/(i+1)
                print(f'[progress] {i+1}/{n}  已用时{_t.time()-t0:.0f}s  剩余{eta:.0f}s', flush=True)
    finally:
        writer.close()
        main_view.close()
        cam_view.close()

    size_mb = out_path.stat().st_size/1024/1024
    print(f'[done] {out_path}  {size_mb:.1f}MB  总耗时{_t.time()-t0:.0f}s', flush=True)
    app.quit()


if __name__ == '__main__':
    main()
