# Endoscopic Robot Magnetic Navigation Simulation

> 内窥镜机器人磁导航仿真 — an offline, self-contained simulator of a magnetically-navigated capsule camera travelling through a colon model, with a forward-looking onboard camera and surface-coverage evaluation.
>
> [中文版](README.zh-CN.md)

**Status:** research prototype · single-case offline simulation · **not** clinically validated.

---

## What this is (in plain words)

Imagine a tiny capsule robot — 12 mm long, 6 mm wide — pulled and steered by an external magnetic field so it can travel through a model of the large intestine (colon). On its front there is a small camera. This project simulates, entirely offline:

1. how the capsule **moves** under magnetic force and torque;
2. where its front camera **looks** while it travels (kept forward and stable);
3. how much of the inner colon wall has actually been **seen** by that camera (coverage);
4. an interactive **replay** where you can play, pause, drag a timeline and switch camera strategies.

It is **not** a real robot and **not** real video. Everything is a deterministic numerical replay of one colon model.

## How to use it

Double-click `Magnetic_Camera.exe`. The left third-person view and the right first-person view share one time bar; playing, pausing and dragging are all deterministic offline replays. The motion scheme on top switches among original speed, slowdown only, wait only, and slowdown + wait; capsule size, mass and magnetic moment are identical across schemes. The camera strategy switches between the look-ahead camera and the Endoscopy baseline. On the left you can independently toggle the bowel, coverage colouring, frustum, look-ahead target, magnetic field and various physical vectors; the overall magnetic field is shown by default on open. This round only implements insertion coordination — no withdrawal or extra observation pass.

The evaluation card is the cumulative result of the current motion/camera combination; the comparison table below independently selects "camera only / full system" and "common budget / full path". So switching motion schemes does not silently change the experiment definition. Coverage colours: red = unseen, cyan = seen, yellow = re-seen. In coverage mode the wall is made more transparent for readability.

## Quick start

- **Windows:** double-click `Magnetic_Camera.exe`. It unpacks the embedded HTML into a temp folder and opens it in an Edge app window.
- **Any browser:** open `Magnetic_Camera.html` directly.

Both are the same self-contained page. No installation, no server, no external files are needed at run time.

## How the modeling pipeline works (cross-file)

The project is a small "assembly line". Each script reads results from upstream files (some live in *other* folders) and writes an intermediate JSON/CSV for the next step:

```
colon_model.stl            (source anatomy, 187,206 triangles) ──┐
source_trajectory_state.csv (centerline xyz + curvature/clearance/risk) ├─┐
capsule_replay.csv + capsule_config.json (baseline motion + robot params) ├─┤
magnetic_model.py          (force/torque physics)                  ├─┤
build_simulation.py        (dynamics solver)                       ├─┤
3D Slicer Endoscopy.py     (baseline camera)                       ├─┤
                                                                    ▼ ▼
 1) prepare_geometry.py   → geometry.json (decimated mesh, 33,696 triangles)
                            endoscopy_baseline.json
 2) build_trajectories.py → camera_trajectories.json
                            capsule_{slowdown,wait,cooperative}_replay.csv
 3) build_withdrawal.py   → capsule_withdrawal_replay.csv, withdrawal_camera.json
 4) evaluate.py           → evaluation.json, withdrawal_visibility_masks.json
 5) build_camera_viewer.py→ Magnetic_Camera.html (self-contained, ~35 MB)
 6) build_exe.py          → Magnetic_Camera.exe (embeds the gzipped HTML)
```

The key design choice: the **displayed mesh and the evaluated mesh are the same surface**, so the colours you see are exactly the numbers that were scored.

## Data and baseline

The baseline is `Magnetic 1 - Field scopes/Magnetic_Viewer_v4.exe` and its corresponding HTML and CSV. The anatomy is `结果_colon_001_含曲率/colon_model.stl`; the path shares the same source, and coordinate ranges were checked for agreement (no LPS/RAS flip). The original 187,206-face mesh is decimated by VTK QuadricDecimation to 33,696 faces; display and evaluation use the same surface. It is a segmented inner-wall proxy — no real endoscopic texture, and it does not include tiny un-segmented folds.

The Endoscopy baseline calls the local 3D Slicer 5.10.0 `EndoscopyLogic` directly: same linear input path, 0.5 mm equal-arc-length resampling, no manual pose keyframes, keeping the default camera direction and quaternion interpolation. Exported poses are interpolated to the same camera positions along the path arc length. Fair evaluation uses the same 60° vertical FOV, 4:3 aspect, geometry and observation thresholds. This is not the user's custom keyframes, FOV or default 30° display setting in the current Slicer scene.

> **Formula notation:** `_` means subscript (e.g. `F_mag` = F with subscript "mag"), `^T` means transpose, `·` is the dot product, `×` is the cross product, `|…|` is magnitude, `√` is square root, `d…/dt` is the time derivative, `clip[a, b, c]` means "clamp into [b, c]".

## Camera look-ahead and stabilization

The camera sits at a fixed offset ahead of the capsule front:

```
p_cam = p_robot + R_robot · [5.8, 0, 0]^T   (mm)
```

The robot body is 12 × 6 mm; the front lens is simplified to a steerable optical axis inside the shell. The look-ahead distance is a heuristic that reacts to speed `v` (mm/s), curvature `κ` (1/mm), normalized risk `r` and clearance `c`:

```
L     = clip[ (12 + 1.8·v) / (1 + 25·κ + 0.8·r),  5,  L_max ]
L_max = clip[ 6 + 2·c,  5,  45 ]
```

The look-ahead is shortened step by step along the centerline and occluded targets are excluded by triangle-mesh ray casting. Direction selection uses hysteresis: the dot-product threshold for a new target is 0.50 (≈60°), for an already-tracked target 0.45 (≈63.3°); the look-ahead arc-length change is limited to 10 mm/s. If no new target exists, the previous look-ahead distance is kept along the centerline; when there is no candidate at all the nearest forward centerline point is taken, never a virtual point reset to the robot front. The target always comes from centerline interpolation; an occluded flag is still recorded separately, so it does not claim the camera can actually see that point. The target orientation uses an orthogonal frame with the previous up-vector projected onto the new view plane to reduce unnecessary roll.

Relative orientation is `R_rel = R_robot^T · R_goal`, described by ZYX Euler joint angles (yaw / pitch / roll) for the steerable lens. The joint control law is:

```
d²θ/dt² = clip[ 12·e − 6·(dθ/dt),  −150,  150 ]   (deg/s²)
|dθ/dt| ≤ 60                                       (deg/s)
```

Yaw and pitch are limited to ±89°, roll is continuous. Numerical integration uses sub-steps ≤ 0.01 s with braking speed `√(2 · a_max · d)` before the angle limit. These are joint-motion assumptions, not a finished mechanism design.

Vertical FOV is 60°, horizontal ≈75.2°, rectangular frustum diagonal half-angle ≈43.9°. To keep the frustum from crossing the robot's rear half-space, the optical axis may deviate at most 45.0° from the body forward axis (≈1° of margin over the 90°−43.9° geometric limit); the camera target direction is first projected into this safety cone, then tracked by joint rate limits. Camera orientation and body orientation are kept separate; no active axial self-rotation is added to the single-dipole robot.

## Coordination: slowdown, waits, and magnetic dynamics

Two errors are distinguished. The **cone error** is the geometric error left after the centerline direction exceeds the safety cone:

```
e_cone = arccos( u_target · u_feasible )
```

The **tracking error** is the quaternion angle between the actual camera and the reachable desired pose:

```
e_track = 2 · arccos( | q_goal · q_actual | )
```

Both arccos results are converted to degrees. Only the solvable (tracking) class slows the robot down; when the target is beyond the forward cone but the lens has caught up to the reachable pose, the robot does not keep waiting because of the cone residual. The quaternion is used for the error metric; the gimbal actuator is still the per-axis Euler PD above — do not describe it as a full quaternion gimbal controller.

Slowdown factor with a 15° deadband and a 50°/s rate soft-limit:

```
f = clip[ 1 + max(e_track − 15°, 0)/80°  +  max(ω_rel − 50 deg/s, 0)/(120 deg/s),  1,  1.25 ]
```

`f` is smoothed with a Gaussian (σ = 6 replay samples), then `dt_new = f · dt_old`. The 15° deadband avoids treating mild normal error as a full-trip slowdown reason; the 50°/s gives margin over the 60°/s joint-rate assumption; `f ≤ 1.25` caps the local speed reduction at ≈20%. These are engineering parameters, not literature-optimal values.

Waits are decided from the simulated joint angle and rate: when the per-axis maximum error exceeds 24° and at least 8 s have passed since the last wait ended, the current reachable target is frozen and the same PD dynamics is integrated with 0.01 s sub-steps. A wait is inserted only if the joint is predicted to settle to per-axis error ≤3° and per-axis rate ≤5°/s within at most 1.2 s; otherwise it keeps going rather than turning an unsolvable error into repeated stops. The total wait budget is 3% of the original duration (≈8.61 s in this case). That budget is an experiment-efficiency limit, not a safety guarantee; lag beyond the budget is kept in the results, never hidden.

Candidate waits are additionally replay-verified: after recomputing magnetic dynamics and the camera, the start must exceed 24° per-axis error and the end must reach ≤3° error and ≤5°/s rate. Unnecessary or unsettled candidates are removed, motion and look-ahead targets recomputed, and remaining candidates re-checked until all pass. Each round only deletes candidates, so it terminates in finite rounds.

Retiming keeps a monotonic clock `τ(t)` (allowing `τ` to stay constant during a wait) and rescales by the chain rule:

```
v_new = v_old · (dτ/dt)
a_new = a_old · (dτ/dt)² + v_old · (d²τ/dt²)
```

The dynamics are then re-solved with the original capsule model, rather than simply stretching the old field sequence.

**Magnetic dynamics.** Translation (Newton's law):

```
M · d²p/dt² = F_mag + F_res + M · g
```

Resistive force:

```
F_res = −μ · M · |g| · v / √( v · v + ε² )
```

Rotation:

```
I · dω/dt + ω × (I · ω) = R^T · τ_mag
```

Magnetic force and torque:

```
F_mag = G^T · m        τ_mag = m × B
```

**Field synthesis.** Given a desired torque `τ`, only the part perpendicular to the magnetic moment is controllable (a single dipole cannot spin about its own axis). Let `u = m / |m|` be the unit magnetic moment:

```
τ_⊥ = τ − (τ · u) · u
```

```
B = (τ_⊥ × m) / |m|² + B_bias · u
```

The minimum-norm symmetric traceless gradient that realizes a desired force, with `a = (F · u)/|m|` and `b = [F − (F · u)·u]/|m|`:

```
G = a · (1.5 · u·u^T − 0.5 · I) + b · u^T + u · b^T
```

Robot parameters are unchanged: mass 0.5276 g, magnetic moment 0.02587 A·m², friction coefficient 0.2, bias field 15 mT, gravity and initial conditions kept. Joint reaction, contact and real coil-field solutions are not solved. The large-range field is a local affine extrapolation, not a full-space coil solution. Arrow lengths are visually increased 15% only; values and units are unchanged.

## Four evaluation metrics

The target area `A_total` is fixed before the experiment as the full shared inner surface, independent of how far the path completes. Evaluation resolution is 160×120 with a 0.4 s time step; each pixel's first visible face (occlusion included) is obtained by anti-aliasing-free, lighting-free per-face off-screen rendering. Both groups use the same renderer; play-back frame rate does not affect scoring.

A face is **validly observed** when: distance 2–80 mm, face-normal-to-ray angle ≤75°, at least 2 visible pixels, and a continuous/small-gap qualified observation reaching 0.4 s. Distance and angle use the face center approximation; angle uses the absolute cosine because STL normals may not be consistently oriented. This is a geometric quality gate, not a lesion-identifiability validation.

Triangle area from the cross product:

```
S = ½ · | (p2 − p1) × (p3 − p1) |
```

Coverage:

```
C(T) = 100 · Σ_i A_i · O_i(T) / A_total
```

Effective field-of-view and its time average (trapezoidal integration over [0, T]):

```
E(t)      = 100 · N_valid(t) / N_image
E_mean(T) = (1/T) · ∫ E(t) dt
```

Repeat rate (a face re-observed after leaving the valid view for more than 0.8 s):

```
R(T) = 100 · Σ_i A_i · max[ n_i(T) − 1, 0 ] / Σ_i A_i · n_i(T)
```

The camera-only comparison uses the same robot positions and time as the current motion scheme, only swapping the camera strategy; the full system compares the original-speed Endoscopy with the chosen retimed scheme. Four motion schemes × two cameras give eight evaluation groups. The common time budget is the original path time 286.99953 s, and every group inserts evaluation samples at exactly this instant. All metrics are precomputed; hovering, dragging and replaying do not increase coverage.

## Reproducibility and external dependencies

The **pre-built results are included**, so the viewer runs standalone. Re-running the full pipeline, however, needs inputs that live outside this folder:

- anatomy: `安全地图_路径规划/结果_colon_001_含曲率/colon_model.stl` (and `colon_model.mha`)
- source trajectory / baseline replay / robot config / HTML template: `../Magnetic 1 - Field scopes/`
- physics and solver: `../Magnetic_shape_comparison/magnetic_model.py`, `../Magnetic 1/build_simulation.py`
- baseline camera and ray-casting: a local 3D Slicer 5.10.0 install (`Endoscopy.py`, `PythonSlicer.exe`)

`prepare_geometry.py` reads the STL, decimates it with VTK (187,206 → 33,696 triangles; quadric error metrics, volume-preserving) and writes `geometry.json`.

## References

- Local `Endoscopy.py`: `D:/3dslicer/3D Slicer 5.10.0/lib/Slicer-5.10/qt-scripted-modules/Endoscopy.py`; source hash in `endoscopy_baseline.json`.
- 3D Slicer Endoscopy docs: <https://slicer.readthedocs.io/en/5.10/user_guide/modules/endoscopy.html>
- Three.js r160 (fixed): <https://github.com/mrdoob/three.js/tree/r160>; license in `vendor/THREE_LICENSE.txt`.
- VTK `vtkWindowToImageFilter` docs: <https://vtk.org/doc/nightly/html/classvtkWindowToImageFilter.html>
- MagRobot: An Open Simulator for Magnetically Navigated Robots — referenced for its modular field/robot/observation decomposition; MagRobot/SOFA were not executed.
- Image-Guided Control of an Endoscopic Robot for OCT Path Scanning, IEEE RA-L, 2021, DOI:10.1109/LRA.2021.3087085.
- Constrained Visual Predictive Control of a Robotic Flexible Endoscope With Visibility and Joint Limits Constraints, IEEE RA-L, 2025, DOI:10.1109/LRA.2024.3521679.

This implementation borrows the idea of considering visibility constraints together with actuator limits, using a known 3D centerline and wall mesh instead of image features; it does not implement online image feedback, CBF or full MPC.

## Limitations

- Single colon case (colon_001); descriptive differences, no statistical significance.
- Look-ahead does not guarantee all mucosa is visible; the lens has no extra sweep task.
- The 15°, 24°, 8 s and 3%-budget values are engineering initial values, not literature-validated.
- Offline replay-state verification — **not** live/real-time closed-loop control.
- No online closed-loop waiting, image sharpness, robot-lens occlusion, or real lens distortion.
- Some source samples have non-positive clearance; not verified against real robot collision.
- Does not guarantee real mucosa visibility or clinical safety.

## License

The viewer bundles Three.js r160 (see `vendor/THREE_LICENSE.txt`, MIT). Choose a license for the rest of the code before publishing.
