import time

import numpy as np
from scipy.spatial.transform import Rotation

from retreat_planner.config import Config
from retreat_planner.controller import State
from retreat_planner.coverage import Coverage
from retreat_planner.search import dense_goals
from retreat_planner.planning import select_predictive
from retreat_planner.visibility import Visibility


class CoupledCamera:
    def __init__(self, scene, insertion_covered):
        self.scene = scene
        self.config = Config().validate()
        self.duration = 700.
        self.coverage = Coverage(scene.areas,insertion_covered,scene.settings['minimum_dwell_s'],scene.settings['repeat_gap_s'])
        self.visibility = Visibility(scene,self.config.atlas_size)
        self.next_decision = 0.
        self.previous_observation = 0.
        self.target = None
        self.start_goal = None
        self.decisions = []
        self.planning_times = []
        self.fallbacks = 0
        self.violations = 0

    def __getattr__(self, name):
        return getattr(self.scene,name)

    def pose(self, time_s):
        elapsed = max(0.,time_s-self.prediction_time)
        arc = min(self.route.arc_m[-1],self.arc+self.prediction_speed*elapsed)
        reference = self.route.at(arc)
        body = (self.rotation_error*reference['rotation']).as_matrix()
        position = reference['position_m']+self.position_error
        return position*1000.+body[:,0]*self.settings['mount_offset_mm'],body

    def update(self, time_s, state, arc, route, angles, velocity):
        self.route = route
        self.arc = arc
        self.prediction_time = time_s
        reference = route.at(arc)
        body = Rotation.from_quat(state[6:10])
        self.rotation_error = body*reference['rotation'].inv()
        self.position_error = state[:3]-reference['position_m']
        self.prediction_speed = max(0.,float(np.dot(state[3:6],reference['tangent'])))
        if self.target is None:
            self.target = angles.copy()
            self.start_goal = angles.copy()
        if time_s >= self.previous_observation+.1-1e-8 or not self.decisions:
            origin = state[:3]*1000.+body.as_matrix()[:,0]*self.settings['mount_offset_mm']
            camera = (body*Rotation.from_euler('ZYX',angles,degrees=True)).as_matrix()
            self.coverage.observe(self.visibility.visible(origin,camera),time_s,
                                  max(0.,time_s-self.previous_observation))
            self.previous_observation = time_s
        if time_s >= self.next_decision-1e-8:
            camera_state = State(angles.copy(),velocity.copy(),self.target.copy())
            started = time.perf_counter()
            goals,scores,count = dense_goals(self,self.visibility,camera_state,time_s,self.coverage,self.config)
            plan,details = select_predictive(goals,camera_state,self.coverage,time_s,self,self.visibility,self.config)
            elapsed = time.perf_counter()-started
            self.start_goal = self.target.copy()
            self.target = plan.next_target(camera_state)
            self.decision_start = time_s
            self.next_decision = time_s+self.config.decision_s
            self.planning_times.append(elapsed)
            self.fallbacks += int(details['fallback'])
            self.decisions.append(dict(time_s=time_s,searched_direction_count=count,
                target_deg=self.target.tolist(),planning_seconds=elapsed,**details))
            if len(self.decisions)%50 == 0:
                print(f'dense+prediction: {time_s:.1f}s, decisions={len(self.decisions)}, '
                      f'searched={count}, new_area={self.coverage.added_area:.1f}mm2',flush=True)

    def actuator_goal(self, time_s):
        ratio = np.clip((time_s-self.decision_start)/self.config.decision_s,0.,1.)
        return self.start_goal+ratio*((self.target-self.start_goal+180.)%360.-180.)

    def check_motion(self, angles, velocity, acceleration):
        cone = np.rad2deg(np.arccos(np.clip(np.cos(np.deg2rad(angles[0]))*np.cos(np.deg2rad(angles[1])),-1.,1.)))
        self.violations += int(cone>45.+1e-7 or np.max(np.abs(velocity))>60.+1e-7
                               or np.max(np.abs(acceleration))>150.+1e-5)

    def report(self):
        milliseconds = np.asarray(self.planning_times)*1000.
        return dict(method='dense-short-horizon-pd-v1',config=self.config.dictionary(),
            metrics=dict(withdrawal_added_area_mm2=self.coverage.added_area,
                constraint_violating_steps=self.violations,prediction_fallback_count=self.fallbacks,
                planning_mean_ms=float(np.mean(milliseconds)),planning_p95_ms=float(np.percentile(milliseconds,95)),
                planning_max_ms=float(np.max(milliseconds)),decision_count=len(self.decisions)),
            prediction_pose='Route-local forecast anchored to actual magnetic state and actual velocity; forecast never forces robot motion',
            decisions=self.decisions,runtime='Replanned offline jointly with magnetic dynamics, not during desktop playback')

    def close(self):
        self.visibility.close()
