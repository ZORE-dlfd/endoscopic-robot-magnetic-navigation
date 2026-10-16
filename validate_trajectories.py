import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from camera_model import SETTINGS, RetimedReference

ROOT = Path(__file__).resolve().parent


def validate():
    tracks = json.loads((ROOT / 'camera_trajectories.json').read_text(encoding='utf-8'))
    reports = {}
    profiles = [('fixed_new', ROOT.parent/'Magnetic 1 - Field scopes'/'capsule_replay.csv')]
    profiles += [(mode + '_new', ROOT / f'capsule_{mode}_replay.csv') for mode in ['slowdown', 'wait', 'coordinated']]
    for key, filename in profiles:
        frame=pd.read_csv(filename)
        track=tracks[key]
        times=np.asarray(track['time_s'])
        angles=np.unwrap(np.deg2rad(track['joint_angle_deg']),axis=0)*180/np.pi
        velocity=np.asarray(track['joint_velocity_deg_s'])
        acceleration=np.asarray(track['joint_acceleration_deg_s2'])
        assert abs(angles[:,:2]).max()<=SETTINGS['max_joint_angle_deg']+1e-8
        assert abs(velocity).max()<=60+1e-8
        assert abs(acceleration).max()<=150+1e-8
        assert abs(np.diff(angles,axis=0)/np.diff(times)[:,None]).max()<=60+1e-7
        assert abs(np.diff(velocity,axis=0)/np.diff(times)[:,None]).max()<=150+1e-7
        matrices=Rotation.from_quat(track['quaternion']).as_matrix()
        body=Rotation.from_quat(frame[[f'actual_q{axis}' for axis in 'xyzw']].to_numpy()).as_matrix()
        mounted=frame[[f'actual_position_{axis}_mm' for axis in 'xyz']].to_numpy()+5.8*body[:,:,0]
        assert np.max(abs(mounted-np.asarray(track['position'])))<1e-8
        half_vertical=np.tan(np.deg2rad(30))
        minimum_forward=1.
        for side in [-1,1]:
            for height in [-1,1]:
                rays=matrices[:,:,0]+side*half_vertical*4/3*matrices[:,:,1]+height*half_vertical*matrices[:,:,2]
                forward=np.einsum('ij,ij->i',rays,body[:,:,0])/np.linalg.norm(rays,axis=1)
                minimum_forward=min(minimum_forward,float(forward.min()))
        assert minimum_forward>0,'FOV escaped body forward hemisphere'
        force=frame[[f'applied_force_{axis}_mN' for axis in 'xyz']].to_numpy()/1000
        moment=frame[[f'actual_axis_{axis}_unit' for axis in 'xyz']].to_numpy()*0.025868875739644973
        gradient=frame[[f'applied_G_{first}{second}_T_m' for first in 'xyz' for second in 'xyz']].to_numpy().reshape(-1,3,3)
        predicted_force=np.einsum('nji,nj->ni',gradient,moment)
        assert np.max(abs(predicted_force-force))<1e-8
        reports[key]={'max_joint_speed_deg_s':float(abs(velocity).max()),'max_joint_acceleration_deg_s2':float(abs(acceleration).max()),'mount_error_mm':float(np.max(abs(mounted-np.asarray(track['position'])))), 'minimum_FOV_body_forward_cosine':minimum_forward,'force_identity_max_error_N':float(np.max(abs(predicted_force-force))), 'duration_s':float(times[-1])}
        mode = key.removesuffix('_new')
        if mode in tracks['ablation_modes']:
            goals = np.asarray(track['goal_joint_angle_deg'])
            axis_error = np.max(np.abs((goals - np.asarray(track['joint_angle_deg']) + 180.0) % 360.0 - 180.0), axis=1)
            for wait in tracks['ablation_modes'][mode].get('waits', []):
                for boundary in [wait['start_s'], wait['start_s'] + wait['duration_s']]:
                    assert np.min(np.abs(times - boundary)) < 1e-9
                start_index = np.argmin(np.abs(times - wait['start_s']))
                end_index = np.argmin(np.abs(times - wait['start_s'] - wait['duration_s']))
                assert axis_error[start_index] >= 24.0
                assert axis_error[end_index] <= 3.0
                assert np.max(np.abs(velocity[end_index])) <= 5.0
    originals=json.loads((ROOT/'baseline_readonly_manifest.json').read_text(encoding='utf-8'))
    for filename,digest in originals.items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest()==digest,'Original file changed: '+filename
    evaluation=json.loads((ROOT/'evaluation.json').read_text(encoding='utf-8'))
    assert len(evaluation['methods']) == 8
    for name,result in evaluation['methods'].items():
        rows=result['rows']
        assert np.all(np.diff([row['coverage_pct'] for row in rows])>=-1e-9)
        assert max(abs(row['coverage_pct']+row['unseen_pct']-100) for row in rows)<1e-8
        assert rows[0]['repeat_pct'] is None
        assert abs(result['common_budget']['time_s'] - evaluation['budget_s']) < 1e-9
    for mode, metadata in tracks['ablation_modes'].items():
        assert metadata['slowdown_max'] <= 1.25 + 1e-9
        assert metadata['wait_duration_s'] <= .03 * tracks['original_duration_s'] + 1e-9
    references=pd.read_csv(ROOT/'coordinated_reference.csv')
    original_source=pd.read_csv(ROOT.parent/'Magnetic 1 - Field scopes'/'source_trajectory_state.csv')
    original_times=np.interp(references.arc_s_mm,original_source.arc_s_mm,original_source.time_s)
    class GeometryOnlyReference:
        def at(self,time):
            from scipy.interpolate import CubicSpline
            if not hasattr(self,'spline'):
                self.spline=CubicSpline(original_source.time_s,original_source[['x_mm','y_mm','z_mm']].to_numpy()/1000,axis=0,bc_type=((1,np.zeros(3)),(1,np.zeros(3))))
            return {'position':self.spline(time),'velocity':self.spline(time,1),'acceleration':self.spline(time,2),'omega_body':np.zeros(3),'alpha_body':np.zeros(3),'rotation':Rotation.identity()}
    original=GeometryOnlyReference()
    retimed=RetimedReference(original,references.time_s,original_times)
    errors=[]
    for time in np.linspace(0,references.time_s.iloc[-1],1000):
        errors.append(np.linalg.norm(retimed.at(time)['position']-original.at(float(retimed.clock(time)))['position']))
    assert max(errors)<1e-12
    for wait in tracks.get('planned_waits',[]):
        midpoint=wait['start_s']+wait['duration_s']/2
        assert np.linalg.norm(retimed.at(midpoint)['velocity'])<1e-10
    reports['reference_geometry_error_mm']=float(max(errors)*1000)
    reports['planned_wait_count']=len(tracks.get('planned_waits',[]))
    reports['original_files_unchanged']=len(originals)
    (ROOT/'trajectory_validation.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports))


if __name__=='__main__':
    validate()
