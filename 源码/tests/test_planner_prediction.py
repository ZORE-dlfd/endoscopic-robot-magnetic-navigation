from types import SimpleNamespace

import numpy as np

from retreat_planner.config import Config
from retreat_planner.controller import State
from retreat_planner.coverage import Coverage
from retreat_planner.planning import Plan, rollout, select_predictive


class ConstantVisibility:
    def visible(self, origin, frame):
        return np.array([0])


def test_rollout_rewards_new_area_once_and_does_not_mutate_actual():
    scene = SimpleNamespace(duration=10., pose=lambda time: (np.zeros(3), np.eye(3)))
    coverage = Coverage(np.array([10.]), np.array([False]), .4, .8)
    coverage.observe(np.array([0]), 0, 0)
    state = State(np.zeros(3), np.zeros(3), np.zeros(3))
    result = rollout(Plan(np.zeros(3)), state, coverage, 0, scene, ConstantVisibility(), Config())
    assert result.gain == 10
    assert result.feasible
    assert coverage.added_area == 0
    assert result.cost == 0


def test_rollout_does_not_predict_beyond_original_end_time():
    queries = []
    def pose(time):
        assert time <= .5 + 1e-9
        queries.append(time)
        return np.zeros(3), np.eye(3)
    scene = SimpleNamespace(duration=.5, pose=pose)
    coverage = Coverage(np.array([10.]), np.array([False]), .4, .8)
    coverage.observe(np.array([0]), .4, 0)
    result = rollout(Plan(np.zeros(3)), State(np.zeros(3), np.zeros(3), np.zeros(3)),
                     coverage, .4, scene, ConstantVisibility(), Config())
    assert result.gain == 0
    assert max(queries) <= .5 + 1e-9


def test_rollout_rejects_unreachable_cone_during_transition():
    scene = SimpleNamespace(duration=10., pose=lambda time: (np.zeros(3), np.eye(3)))
    coverage = Coverage(np.array([10.]), np.array([False]), .4, .8)
    state = State(np.array([44., 0., 0.]), np.array([50., 0., 0.]), np.array([44., 0., 0.]))
    result = rollout(Plan(np.zeros(3)), state, coverage, 0, scene, ConstantVisibility(), Config())
    assert not result.feasible


def test_predictive_selects_achievable_coverage_not_larger_instantaneous_view():
    class DirectionalVisibility:
        def visible(self, origin, frame):
            yaw = np.rad2deg(np.arctan2(frame[1, 0], frame[0, 0]))
            if abs(yaw) < 5:
                return np.array([0])
            if yaw > 25:
                return np.array([1])
            return np.array([], dtype=int)
    scene = SimpleNamespace(duration=.4, pose=lambda time: (np.zeros(3), np.eye(3)))
    coverage = Coverage(np.array([10., 100.]), np.array([False, False]), .4, .8)
    coverage.observe(np.array([0]), 0, 0)
    state = State(np.zeros(3), np.zeros(3), np.zeros(3))
    plan, details = select_predictive([np.zeros(3), np.array([30., 0., 0.])], state, coverage, 0,
                                     scene, DirectionalVisibility(), Config())
    np.testing.assert_allclose(plan.target, np.zeros(3))
    assert details['predicted_gain_mm2'] == 10


def test_delay_is_not_offered_without_first_step_coverage_gain(monkeypatch):
    offered = []
    def fake_rollout(plan, state, coverage, time, scene, visibility, config):
        from retreat_planner.planning import Prediction
        offered.append(plan.delay_s)
        return Prediction(0., 0., True, 0.)
    monkeypatch.setattr('retreat_planner.planning.rollout', fake_rollout)
    coverage = Coverage(np.array([10.]), np.array([False]), .4, .8)
    state = State(np.zeros(3), np.zeros(3), np.zeros(3))
    select_predictive([np.zeros(3), np.array([30., 0., 0.])], state, coverage, 0,
                      SimpleNamespace(duration=10.), None, Config())
    assert not any(delay > 0 for delay in offered)
