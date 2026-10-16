# Continuous Withdrawal Coverage Implementation Plan

> **For agentic workers:** Use the executing-plans workflow. Every implementation task starts with a failing test and ends with focused validation.

**Goal:** Add continuous, forward-facing capsule withdrawal and measure unique mucosal area newly observed during withdrawal, excluding fold-hidden geometry.

**Architecture:** Work only in `Magnetic 1 - Capsule Camera - Withdrawal v1`, reading coordination-v2 and magnetic solver artifacts as immutable inputs. Separate recoverability, insertion misses, reverse trajectory and magnetic replay, coverage evaluation, and viewer packaging.

**Tech Stack:** Existing Anaconda Python, NumPy, SciPy, Pandas, VTK/PythonSlicer, Three.js/Plotly, and .NET Framework. No dependency installation or upgrades.

**Spec:** Approved in conversation on 2026-10-15.

## Global Constraints
- Do not modify `Magnetic 1 - Capsule Camera - Coordination v2`, `Magnetic 1 - Field scopes`, `Magnetic 1`, or magnetic solver source directories; treat them as read-only.
- Put all new source, data, reports, HTML, and EXE under `C:\Users\老白给\Desktop\navigate\Magnetic 1 - Capsule Camera - Withdrawal v1`.
- Withdrawal is continuous with no observation dwell or tracking wait; camera lag may cause smooth slowdown only.
- Reverse centerline progress but keep body-forward orientation continuous; no 180-degree turn.
- Recompute velocity, acceleration, resistance, force, torque, field, and gradient with the existing magnetic model; never blindly negate exported insertion fields.
- Keep current visibility thresholds: 2–80 mm, incidence at most 75 degrees, at least 2 rendered pixels, 0.4 seconds effective observation, and the current forward-camera cone.
- Exclude a face only if all sampled admissible forward-camera poses geometrically occlude it; keep recoverable insertion misses eligible.
- Count each face at most once as new withdrawal coverage; repeats add no area.
- Do not claim clinical safety or image-quality validation; this is rigid-geometry offline simulation.
- Preserve existing offline viewer and build a new EXE without overwriting the old one.

## Review Focus
- Fold mask distinguishes all-view occlusion from transient occlusion.
- Dwell boundary: 0.4 seconds qualifies; existing 0.8-second gap reset is preserved.
- Withdrawal progress heads toward the entrance with no stop interval or body/camera flip.
- Reverse resistance opposes actual velocity; magnetic force still satisfies `G.T @ m`.
- Only eligible faces missed during insertion and first qualified during withdrawal add area.
- Missing clearance or magnetic-limit data is reported as a limitation, not clinical safety.

## File Structure
- `withdrawal_visibility.py`: visibility, fold-hidden mask, insertion-miss mask, unique-area accounting.
- `withdrawal_trajectory.py`: reverse timing, forward orientation, camera target schedule, magnetic replay.
- `evaluate_withdrawal.py`: raster visibility and additive area evaluation.
- `build_withdrawal.py`: read existing inputs and generate new artifacts in this directory.
- `camera_view.js`, `build_camera_viewer.py`: phase control and coverage-state colors.
- `test_withdrawal_visibility.py`, `test_withdrawal_trajectory.py`, `test_withdrawal_evaluation.py`: unit/synthetic tests.
- `withdrawal_summary.json`, `withdrawal_evaluation.csv`, `WITHDRAWAL_METHOD.md`: outputs and limitations.
- Copy launcher/vendor assets only as needed for the new HTML/EXE.

## Tasks

### Task 1: Interfaces and failing tests
**Files:** Create three test modules and four Python modules listed above.
**Interfaces:** `recoverable_face_mask(geometry, candidate_camera_poses, settings) -> bool array`; `insertion_missed_mask(recoverable_mask, insertion_first_seen_s) -> bool array`; `reverse_progress(source_arc_mm) -> float array`; `accumulate_withdrawal_area(areas_mm2, eligible_mask, insertion_seen_mask, withdrawal_seen_mask) -> dict`.
**Steps:**
- [x] Add synthetic visible, occluded, and insertion-missed face fixtures.
- [x] Test dwell boundary, fold exclusion, unique area, reverse order, and no-stop behavior.
- [x] Run tests and confirm expected failures.

### Task 2: Recoverable and missed surface masks
**Files:** Modify `withdrawal_visibility.py`; test `test_withdrawal_visibility.py`.
**Steps:**
- [x] Reuse current VTK raster/occlusion evaluator and camera settings.
- [x] Mark fold-hidden only if no sampled admissible pose sees a face unobstructed under existing thresholds.
- [x] Mark insertion-missed only for recoverable faces failing effective-observation duration.
- [x] Test transient occlusion remains eligible and permanent fold occlusion is excluded.
- [x] Run `python -m unittest test_withdrawal_visibility -v`.

### Task 3: Continuous reverse path and magnetic replay
**Files:** Modify `withdrawal_trajectory.py`, `build_withdrawal.py`; test `test_withdrawal_trajectory.py`.
**Interfaces:** `build_reverse_reference(source, slowdown_profile, camera_goals, settings) -> DataFrame`; `replay_with_existing_magnetic_model(reference, config, physics, builder) -> DataFrame`.
**Steps:**
- [x] Reverse centerline progress and preserve continuous body-forward tangent quaternion.
- [x] Derive velocity/acceleration from a smooth monotone time map; use slowdown-only for feasible camera lag, never insert waits.
- [x] Recompute resistance and magnetic force/torque/field/gradient with existing model functions.
- [x] Test route order, no 180-degree flip, no stop, joint limits, reverse friction, and force identity.
- [x] Run focused tests and validate the generated replay.

### Task 4: Additive withdrawal coverage
**Files:** Modify `evaluate_withdrawal.py`; test `test_withdrawal_evaluation.py`.
**Steps:**
- [x] Evaluate withdrawal frames with the same rasterizer and visibility thresholds as insertion.
- [x] Count only first qualifying withdrawal observation of eligible insertion-missed faces.
- [x] Assert final area equals insertion area plus withdrawal-added area; repeats add none.
- [x] Save per-face/per-frame CSV and JSON summary; test on hand-computed meshes.

### Task 5: Viewer and EXE
**Files:** Modify `camera_view.js`, `build_camera_viewer.py`; create `WITHDRAWAL_METHOD.md`, HTML, and EXE.
**Steps:**
- [x] Add insertion/withdrawal phase control with shared timeline and existing camera controls.
- [x] Show insertion-covered, insertion-missed, withdrawal-new, and fold-hidden faces separately.
- [x] Display insertion area, withdrawal-added area, final area, and gain without counting repeats.
- [x] Preserve whole-field default, dual views, and existing controls.
- [x] Package embedded offline HTML and verify embedded payload hash matches delivered HTML.

### Task 6: End-to-end validation
**Files:** Create `withdrawal_validation.json`; finalize `WITHDRAWAL_METHOD.md`.
**Steps:**
- [x] Run existing and new tests with installed Anaconda/Slicer environments.
- [x] Validate reverse progress, forward view, no stops, magnetic identities, fold mask, and area conservation.
- [x] Open HTML in a headless browser; verify phase switching, timeline, default whole-field view, and no runtime errors; verify EXE payload hash.
- [x] Verify original v2 and Field scopes hashes remain unchanged.
- [x] Report actual added area; no thresholds were changed to force a gain.

## Literature Boundary
- Prior visibility-map, deficient-coverage, and area-based surface-coverage work motivates per-surface memory and new-area accounting.
- Do not claim literature validates this exact continuous forward-facing magnetic withdrawal controller.
- Fold-hidden exclusion is the user's project definition even if prior papers count hidden regions as deficient coverage.
