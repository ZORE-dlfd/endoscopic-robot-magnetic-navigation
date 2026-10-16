# Continuous Insertion and Withdrawal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Deliver a new standalone EXE showing one continuous slowdown-only insertion-to-withdrawal animation and reporting unique added coverage.

**Architecture:** Build one time-offset replay dataset from the existing slowdown insertion and withdrawal solver outputs. Use the existing camera, centerline risk, field, bowel, and plot components. Keep alternate coordination policies as data-only comparisons; use the full lumen surface as the coverage denominator while restricting withdrawal target selection to the existing sampled observable region.

**Tech Stack:** Python 3.11 environment at D:\anaconda with NumPy, Pandas, SciPy; existing HTML/JavaScript, Three.js, Plotly; .NET launcher.

**Spec:** User-approved continuous round-trip requirements in the conversation on 2026-10-15.

## Global Constraints
- Work only in `Magnetic 1 - Capsule Camera - Round Trip v2`; preserve v1 and all older source folders.
- The only animation is slowdown-only insertion followed immediately by withdrawal; other coordination strategies remain data-only.
- Keep the same coverage mesh denominator (all lumen triangles), including faces excluded from camera target selection.
- Withdrawal camera targeting ignores faces not observed by the existing sampled candidate-view visibility union.
- Mark the insertion/withdrawal boundary clearly on the shared slider and coverage chart.
- Preserve current risk hover, camera views, field controls, camera stability, and fixed plot axes.

## Review Focus
- Time monotonicity and exactly one phase transition at the join.
- Body and camera pose continuity at the reversal point; report any residual discrepancy rather than hiding it.
- Coverage is monotone, counts each face once, and uses total lumen area as the denominator.
- Faces excluded from withdrawal targeting still remain in unseen/full-area statistics.
- Alternate coordination results remain accessible without selectable animations.
- Packaged EXE contains the final HTML and launches without network access.

---

### Task 1: Build a continuous round-trip dataset

**Files:** `build_camera_viewer.py`, `camera_trajectories.json`, existing replay CSVs.

- [ ] Add a builder test asserting monotonic combined time, insertion/withdrawal phase labels, one join index, and all-lumen denominator metadata.
- [ ] Build the combined slowdown-insertion and withdrawal replay in memory with a global time offset; retain only one robot/profile in the animation registry.
- [ ] Combine the slowdown and withdrawal camera tracks on the same clock and record exact join pose deltas for diagnostics.
- [ ] Verify the builder test and save join diagnostics in the new output directory.

### Task 2: Render one animation and expose the handoff

**Files:** `viewer.js`, `camera_view.js`, `viewer.css`, HTML builder/template integration.

- [ ] Add tests for one animation profile and an explicit join marker label/state.
- [ ] Remove alternate animation choices while retaining their evaluation rows in the comparison table.
- [ ] Add a prominent slider boundary marker and insertion/withdrawal phase labels; preserve single play/pause and shared time scrubber.
- [ ] Draw full-area cumulative coverage over global time, with withdrawal-only unique additions visually distinguished.
- [ ] Smoke-test both sides of the join and preserve existing hover, field, first/third-person, and fixed-axis behaviors.

### Task 3: Correct full-area coverage reporting

**Files:** `withdrawal_visibility.py`, `evaluate_withdrawal.py`, `camera_view.js`, `WITHDRAWAL_METHOD.md`.

- [ ] Add tests asserting hidden faces remain in the denominator and never become withdrawal targets; duplicate observations add no area.
- [ ] Keep the existing candidate-visible mask for camera target planning, but compute insertion/final coverage percentages against total lumen mesh area.
- [ ] Report insertion area, unique withdrawal-added area, final area, missed area, full lumen area, and percentages with explicit denominator labels.
- [ ] Explain visible pixels/depth occlusion, distance/incidence/pixel gates, discrete dwell approximation, and sampled target-direction limitation.
- [ ] Recompute and verify all reported values from the masks and mesh areas.

### Task 4: Build and validate the standalone EXE

**Files:** `Magnetic_Camera.html`, `Magnetic_Camera.exe`, validation JSON and tests in the new directory.

- [ ] Run focused unit tests, Python compilation, and JavaScript syntax checks.
- [ ] Build the HTML and EXE, then verify extracted embedded HTML hash equals the delivered HTML hash.
- [ ] Run browser smoke checks for phase label, slider join, key coverage values, and absence of runtime errors.
- [ ] Compare the old-source manifest and confirm no source files outside the new directory changed.

## Execution Notes
- Do not fabricate a physically smooth reversal if source trajectory endpoints disagree. Surface the measured position/orientation/velocity deltas; only claim continuity after verification.
- If the source datasets do not include a dynamically solved reversal transition, label the join accurately and do not imply that a pause/turnaround maneuver was magnetically solved.
- Distinguish percentage-point gain (new area / total lumen area) from relative gain (new area / insertion-covered area).
