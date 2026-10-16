# Endoscopic Robot Magnetic Navigation

[中文版](README.zh-CN.md)

Native **Qt Widgets + OpenGL** desktop replay of the clon001 colon simulation. No HTML, browser, WebView, or JavaScript runtime. Research prototype: offline, one case, no clinical validation, no physical robot control, no real endoscopic video.

## Run and share

Open `软件/Magnetic_Camera.exe`. Share the **entire 软件 folder**, including `_internal`; extract a ZIP completely before launching. No Python, Anaconda, browser, or network is required on the receiving machine.

Target: 64-bit Windows 10/11 with OpenGL 3.3 Core support. The executable is unsigned. Delivery checks use a separate Unicode path and a restricted environment on the same computer, not another computer or a clean Windows VM. Do not disable security protection to run it.

## Predictive-withdrawal revision

Only the withdrawal camera-angle policy changes. All 3,172 original motion samples and 146 exported fields, magnetic states, body position/orientation, speed, duration, insertion camera, and insertion coverage remain unchanged.

- Dense camera-FOV search: 10° coarse grid, 2° local refinement, 60° vertical FOV, 4:3 aspect ratio, within the original 45° forward cone.
- Every 0.4 s, predict 1.2 s with the original PD controller. Among feasible plans with at least 95% of the maximum predicted new area, minimize normalized velocity, acceleration, and jerk cost. Future predicted coverage is never credited as actual coverage.
- Original per-axis PD, 60°/s velocity limit, 150°/s² acceleration limit, 0.01 s integration. Discrete checks are not a continuous-time safety certificate.
- The top-right algorithm button provides four readable sections: motion, magnetic-field synthesis, insertion camera, withdrawal camera. No raw LaTeX.
- Hover near the centerline for risk, clearance, curvature, target speed, and speed caps, linked to the chart. Hiding the path or leaving the view clears the probe.
- Blue = insertion coverage; orange = new withdrawal coverage; gray = not yet covered. Blue and orange are mutually exclusive.

This is dense sliding-FOV search inspired by a convolution window, **not a trained CNN**. Planning is precomputed, not run once per displayed frame. Measured planning mean/P95/max: 287.6/389.7/464.4 ms. The maximum exceeds the 0.4 s decision interval, so this is not an online real-time guarantee.

## Current clon001 results

| Quantity | Result |
| --- | ---: |
| Whole evaluated surface | 246,417.47 mm² |
| Insertion coverage, unchanged | 177,090.42 mm² |
| New withdrawal coverage | 20,909.91 mm² |
| Final coverage | 198,000.33 mm² / 80.3516% |
| Coverage gain | 8.4856 percentage points |
| Actual sampled constraint violations / fallback | 0 / 0 |

Withdrawal visibility is evaluated every 0.1 s using face-ID depth rendering and 0.4 s qualified dwell. Insertion coverage is preserved. Playback and mouse events never add coverage. Independent dwell accumulation from the actual replay poses reproduces the per-face first-seen times and final area.

The planner uses the same Hermite body position as the player. Do not substitute the older linear-body validation result of 20,968.55 mm². The previous 37-direction software also used different evaluation sampling/masks, so its area is not a strictly controlled ablation comparison.

## Layout

- `软件/`: distributable application, native runtime, bundled data, third-party notices.
- `源码/magnetic_app/`: domain, rendering, and UI modules.
- `源码/retreat_planner/`: offline planner, separate from playback.
- `源码/tests/`: numerical, controller, UI, picking, and integrity tests.
- `数据/clon001/`: original arrays/metadata, separate prediction overlay and hash manifest, algorithm text.
- `工具/`: generation, independent audit, native packaging, isolated delivery validation.
- `文档/`: use/build instructions, changes, reports, evidence, screenshots.
- `构建/`: ignored local build dependencies, cache, and previous-binary backup.

## Reproduce

Reuse installed libraries before creating environments. This machine uses Anaconda Python with NumPy 2.3.5, SciPy 1.16.3, PySide6 6.9.2; packaging dependencies are in `构建/依赖`. See [build instructions](文档/构建说明.md).

```powershell
$root = 'C:\Users\老白给\Desktop\navigate-final software'
$env:PYTHONPATH = "$root\构建\依赖;$root\源码"
& D:\anaconda\python.exe -m pytest "$root\源码\tests" -q
& D:\anaconda\python.exe "$root\工具\audit_predictive.py"
& D:\anaconda\python.exe "$root\工具\build_native.py"
# After replacing 软件 with the complete newly built distribution:
& D:\anaconda\python.exe "$root\工具\verify_native.py"
```

The original arrays/metadata are included and unchanged; normal use does not require the historical HTML, source anatomy, Slicer, or sibling projects. Re-running the original upstream motion solver is outside this revision. See [Chinese user guide](文档/使用说明.md), [validation](文档/验证报告.md), and [change log](文档/变更记录.md).

## Limits and licensing

Known 3D geometry and future body trajectory, no online image feedback. Lower joint cost does not guarantee a lower world-camera angular-speed peak. Geometric visibility is not lesion recognition. Existing magnetic states are not a calibrated full-space coil solution. Existing non-positive clearance samples and the approximately 0.5 mm insertion/withdrawal position residual are not removed. No tissue or clinical safety claim is made.

Bundled third-party notices are in `软件/第三方许可`. This revision does not add a license for the project as a whole.
