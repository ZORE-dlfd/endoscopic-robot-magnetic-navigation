# Endoscopic Robot Magnetic Navigation

[中文说明](README.zh-CN.md)

Native **Qt Widgets + OpenGL** desktop visualization of the clon001 capsule-navigation simulation. No HTML, browser, WebView, or JavaScript runtime. The current version uses eight moving equivalent magnetic sources, a supine-placement assumption, and the corrected **anal entry → deep endpoint → anal entry** route.

## Download and run

Download the complete Windows ZIP from the [existing v2.1.0 release](https://github.com/ZORE-dlfd/endoscopic-robot-magnetic-navigation/releases/tag/v2.1.0). Extract it completely, then open `八线圈胶囊导航软件/Magnetic_Camera_8Coil.exe`. Keep `_internal`, the data, and third-party notices beside the executable. Do not launch from inside the ZIP or share the EXE alone.

The repository also retains the complete application under `软件/`, following its existing layout. Open `软件/Magnetic_Camera_8Coil.exe` after downloading the full repository. Recipients do not need Python, Anaconda, Qt, or a browser.

Target: 64-bit Windows 10/11, with an OpenGL 3.3 Core-capable graphics driver. The executable is unsigned; do not disable security protection. Delivery checks were performed on the same computer in a separate Unicode path with Python/Conda/Qt environment variables removed, not on another physical computer or a clean Windows VM.

## What changed

- **Eight-source dynamics:** bounded, field-priority current allocation, ±10 A per source, and an array translating at up to 0.2 m/s. Actual field and gradient produce force and torque, which are integrated into motion. The capsule is not snapped onto the centerline.
- **Supine-placement assumption:** anatomy and reference states share one rigid transform; world gravity remains `(0, 0, -9.81) m/s²`. Currents, motion, camera planning, and coverage were recomputed. The horizontal grid is not a bed/contact model; anatomical registration has not been verified.
- **Correct entry:** the generator now uses the original confirmed insertion source instead of the reverse-ordered legacy CSV. Generation and loading check actual start, turn, return, and initial camera direction; this is not a label swap or reversed playback.
- **Preserved camera policy:** dense sliding-FOV search, 1.2 s short-horizon prediction, and the original PD controller. Every 0.4 s, search a 10° coarse grid with 2° local refinement. Among feasible candidates achieving at least 95% of the best predicted coverage gain, prefer smooth motion. This is not a trained CNN or the old fixed 37-direction policy.
- **Preserved UI:** dual views, timeline dragging, risk hover, field/streamline/slice displays, charts, and status tables. Blue is insertion coverage, orange is additional withdrawal coverage; the two are mutually exclusive. Slowdown is allowed, but no waiting segment is commanded.

Planning and dynamics are precomputed; the application smoothly replays their results rather than solving them per rendered frame. No live endoscopic video or physical robot is controlled.

## Current clon001 validation

| Quantity | Result |
| --- | ---: |
| Route length | 1,722.3136 mm |
| Insertion / withdrawal / total duration | 294.84 / 303.70 / 598.54 s |
| Start / deep turn / return position error | 0 / 0.126 / 0.146 mm |
| Peak absolute source current | 9.19 A, below the 10 A limit |
| Insertion coverage | 165,529.21 mm² |
| New withdrawal coverage | 31,240.72 mm² |
| Final coverage | 196,769.93 / 246,417.47 mm² = 79.8523% |
| Withdrawal decisions / directions per decision | 760 / 243–381 |
| Sampled constraint violations / fallback | 0 / 0 |

Coverage is reevaluated from actual camera poses, including occlusion and qualified dwell. Playback and mouse events do not add coverage. Planning mean/P95/max was about 261/326/376 ms; these measurements are not a worst-case or hard real-time guarantee. Older release coverage numbers are not the results of this corrected route.

## Project layout and reproduction

- `软件/`: complete distributable runtime and notices; `发布/` is ignored local packaging output.
- `源码/magnetic_app/`: UI, rendering, and replay; `源码/coil_simulation/`: magnetic sources, dynamics, posture, and route checks.
- `源码/retreat_planner/`: preserved dense predictive withdrawal planner; `源码/tests/`: numerical, integrity, and UI tests.
- `数据/clon001/`: current results and hash manifests; `数据/参考原版/`: unchanged reference; `数据/仿真输入/`: model inputs and direction contract.
- `工具/`: generation, audit, build, and delivery checks; `文档/`: reports and current evidence; `构建/`: ignored dependencies, caches, and local backups.

Reuse an existing compatible environment first. The verified environment is Anaconda Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, PySide6 6.9.2, PyOpenGL 3.1.10, pytest 8.4.2, and PyInstaller 6.22.3. See [build instructions](文档/构建说明.md), [implementation](文档/实现说明.md), [validation](文档/验证报告.md), and [entry-direction repair](文档/肛门入口方向修复.md). Build dependencies and personal backups are not committed.

## Limits and licensing

This is a single-case research simulation with point-dipole sources and idealized friction. It is not FEM, a measured coil field, a hardware demonstration, or clinical validation. Thermal/inductive effects, current slew, iron saturation, tissue contact/deformation, and manipulator joint/collision constraints are not modeled. Rendered rings do not establish real coil dimensions. Geometric coverage is not lesion detection.

Third-party notices are in `软件/第三方许可` and `软件/第三方组件说明.md`. No project-wide license is added. Existing component-source records are not a complete commercial distribution audit.
