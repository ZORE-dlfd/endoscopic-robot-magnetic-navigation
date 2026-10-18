# Endoscopic Robot Magnetic Navigation

[中文说明](README.zh-CN.md)

Native **Qt Widgets + OpenGL** desktop visualization of the clon001 capsule-navigation simulation. No HTML, browser, WebView, or JavaScript runtime. The 2026-10-18 update replaces equivalent point dipoles with eight compact finite windings, uniformly magnetized finite cores and approximate mutual coupling. The supine-placement assumption and **anal entry → deep endpoint → anal entry** route are unchanged.

## Download and run

Download the complete Windows ZIP from the [existing v2.1.0 release](https://github.com/ZORE-dlfd/endoscopic-robot-magnetic-navigation/releases/tag/v2.1.0). Extract it completely, then open `八线圈胶囊导航软件/Magnetic_Camera_8Coil.exe`. Keep `_internal`, the data, and third-party notices beside the executable. Do not launch from inside the ZIP or share the EXE alone.

The repository also retains the complete application under `软件/`, following its existing layout. Open `软件/Magnetic_Camera_8Coil.exe` after downloading the full repository. Recipients do not need Python, Anaconda, Qt, or a browser.

Target: 64-bit Windows 10/11, with an OpenGL 3.3 Core-capable graphics driver. The executable is unsigned; do not disable security protection. Delivery checks were performed on the same computer in a separate Unicode path with Python/Conda/Qt environment variables removed, not on another physical computer or a clean Windows VM.

## What changed

- **Compact finite sources:** 52 mm long × 32 mm diameter cores; 44 mm long, 56 mm outer-diameter windings; 1200 turns per coil, ±10 A. Only core size was specified; the remaining hardware parameters are hypotheses. Finite winding/core fields include center-sampled mutual coupling. Field-priority allocation is preserved, with the validated 8 mT bias and array translation up to 0.2 m/s. The capsule is not snapped onto the centerline.
- **Per-coil parameters:** C1–C8 live currents, resistive power and mean magnetization fractions; a separate table lists geometry, wire, material assumptions, resistance, layout, position and voltage. Source outlines use assumed dimensions. Field arrows, slices and streamlines use an axisymmetric response lookup of the same finite model; recorded dynamics use direct source integration.
- **Supine-placement assumption:** anatomy and reference states share one rigid transform; world gravity remains `(0, 0, -9.81) m/s²`. Currents, motion, camera planning, and coverage were recomputed. The horizontal grid is not a bed/contact model; anatomical registration has not been verified.
- **Correct entry:** the generator now uses the original confirmed insertion source instead of the reverse-ordered legacy CSV. Generation and loading check actual start, turn, return, and initial camera direction; this is not a label swap or reversed playback.
- **Preserved camera policy:** dense sliding-FOV search, 1.2 s short-horizon prediction, and the original PD controller. Every 0.4 s, search a 10° coarse grid with 2° local refinement. Among feasible candidates achieving at least 95% of the best predicted coverage gain, prefer smooth motion. This is not a trained CNN or the old fixed 37-direction policy.
- **Preserved UI:** dual views, timeline dragging, risk hover, field/streamline/slice displays, charts, and status tables. Blue is insertion coverage, orange is additional withdrawal coverage; the two are mutually exclusive. Slowdown is allowed, but no waiting segment is commanded.

Planning and dynamics are precomputed; the application smoothly replays their results rather than solving them per rendered frame. No live endoscopic video or physical robot is controlled.

## Current clon001 validation

| Quantity | Result |
| --- | ---: |
| Route length | 1,722.3136 mm |
| Insertion / withdrawal / total duration | 295.78 / 305.42 / 601.20 s |
| Start / deep turn / return position error | 0 / 0.1215 / 0.1316 mm |
| Peak absolute source current | 10 A, within floating-point tolerance |
| Insertion coverage | 167,744.73 mm² = 68.0734% |
| New withdrawal coverage | 32,974.75 mm² = 13.3817 percentage points |
| Final coverage | 200,719.47 / 246,417.47 mm² = 81.4550% |
| Withdrawal decisions / directions per decision | 764 / 243–419 |
| Sampled constraint violations / fallback | 0 / 0 |
| Mean total resistive power, insertion / withdrawal | 1,004.01 / 889.72 W |
| Peak mean-core saturation fraction | 62.73% |

Coverage is reevaluated from actual camera poses, including occlusion and qualified dwell. Playback and mouse events do not add coverage. Planning mean/P95/max was about 345/475/519 ms; this is precomputed offline, not a per-frame cost or hard real-time guarantee. Older point-dipole coverage numbers do not describe this magnetic-source update.

## Project layout and reproduction

- `软件/`: complete distributable runtime and notices; `发布/` is ignored local packaging output.
- `源码/magnetic_app/`: UI, rendering, and replay; `源码/coil_simulation/`: magnetic sources, dynamics, posture, and route checks.
- `源码/retreat_planner/`: preserved dense predictive withdrawal planner; `源码/tests/`: numerical, integrity, and UI tests.
- `数据/clon001/`: current results and hash manifests; `数据/参考原版/`: unchanged reference; `数据/仿真输入/`: model inputs and direction contract.
- `工具/`: generation, audit, build, and delivery checks; `文档/`: reports and current evidence; `构建/`: ignored dependencies, caches, and local backups.

Reuse an existing compatible environment first. The verified environment is Anaconda Python 3.13.9, NumPy 2.3.5, SciPy 1.16.3, pandas 2.3.3, PySide6 6.9.2, PyOpenGL 3.1.10, pytest 8.4.2, and PyInstaller 6.22.3; offline finite-source integration additionally uses the existing Numba installation. See [current source model and import reproduction](文档/紧凑磁源接入说明.md), [build instructions](文档/构建说明.md), and [validation](文档/验证报告.md). The import reproduces exports from the included completed experiment, not a new full experiment optimization. Build dependencies and personal backups are not committed.

## Limits and licensing

This is a single-case, model-consistent research simulation with finite windings, uniform finite cores, approximate mutual coupling and idealized friction. Controller and plant share the same model: it is not independent FEM, measured coil fields, a hardware demonstration or clinical validation. Mean-core saturation is bounded, but nonlinear B–H, hysteresis, heating, inductive dynamics, current slew, tissue contact/deformation, and manipulator/body collisions are absent. Mean resistive power is roughly 0.89–1.00 kW, with ideal cooling assumed; physical continuous operation is not established. The 32 mm diameter is the core only; winding diameter is 56 mm. Geometric coverage is not lesion detection.

Third-party notices are in `软件/第三方许可` and `软件/第三方组件说明.md`. No project-wide license is added. Existing component-source records are not a complete commercial distribution audit.
