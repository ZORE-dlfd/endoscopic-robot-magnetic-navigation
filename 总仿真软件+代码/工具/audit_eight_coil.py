import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'源码'),str(ROOT/'构建'/'依赖')]

import numpy as np
from magnetic_app.domain.replay import Replay


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reference_implementations(root):
    baseline = json.loads((root/'数据'/'参考原版'/'planner_sha256.json').read_text('utf-8'))['sha256']
    implementations = {}
    for name in ['search.py','planning.py','controller.py','config.py','coverage.py','visibility.py']:
        current_sha256 = digest(root/'源码'/'retreat_planner'/name)
        reference_sha256 = baseline[name]
        implementations[name] = dict(current_sha256=current_sha256,reference_sha256=reference_sha256,unchanged=current_sha256==reference_sha256)
    return implementations


def main():
    replay = Replay()
    implementations = reference_implementations(ROOT)
    assert all(value['unchanged'] for value in implementations.values())
    assert replay.coils is not None and replay.prediction['method']=='dense-short-horizon-pd-v1'
    assert replay.prediction['metrics']['constraint_violating_steps']==0
    assert replay.prediction['metrics']['prediction_fallback_count']==0
    assert np.all(np.diff(replay.times)>0.)
    assert np.all((replay.data['time_dilation']>=1.) & (replay.data['time_dilation']<=1.25))
    config = replay.robot['config']
    posture = replay.payload['patientPosture']
    assert posture['posture']=='supine' and not posture['contact_model']
    np.testing.assert_array_equal(config['gravity_m_s2'],[0.,0.,-9.81])
    reference = Replay(ROOT/'数据'/'参考原版')
    rotation = np.asarray(posture['rotation_from_original'])
    expected_points = reference.points@rotation.T+posture['translation_mm']
    maximum_geometry_error = float(np.max(np.linalg.norm(replay.points-expected_points,axis=1)))
    assert maximum_geometry_error<1e-8
    np.testing.assert_array_equal(replay.faces,reference.faces)
    np.testing.assert_array_equal(replay.areas,reference.areas)
    direction = replay.payload['routeDirection']
    entry,deep = reference.centerline[[0,-1]]@rotation.T+posture['translation_mm']
    direction_errors = dict(
        insertion_start_mm=float(np.linalg.norm(replay.frame(0.).position-entry)),
        deep_turn_mm=float(np.linalg.norm(replay.frame(replay.join_time).position-deep)),
        withdrawal_finish_mm=float(np.linalg.norm(replay.frame(replay.duration).position-entry)))
    assert all(value<1. for value in direction_errors.values())
    assert direction['verification']['passed']
    assert direction['insertion_start_label']=='肛门入口' and direction['withdrawal_end_label']=='肛门入口'
    assert np.all(replay.coils.positions[:,:,2]-replay.positions[:,None,2]>40.)
    from coil_simulation.finite_sources import CoilDesign, FiniteArray
    array = FiniteArray(CoilDesign(turns=1200), layout_scale=.75)
    maximum_lookup_field_error_mt = 0.
    maximum_lookup_gradient_error_t_m = 0.
    maximum_force_error = 0.
    maximum_dynamics_error = 0.
    for index in range(0,len(replay.times),11):
        frame = replay.frame(replay.times[index])
        moment = config['magnetic_moment_A_m2']*frame.body_rotation[:,0]
        positions, currents, moments = replay.coils.at(frame)
        anchor = positions[0]/1000.-array.offsets[0]
        exact_field, exact_gradient = array.responses(frame.position/1000.-anchor)
        exact_field = exact_field@currents
        exact_gradient = np.einsum('i,ijk->jk', currents, exact_gradient)
        maximum_lookup_field_error_mt = max(maximum_lookup_field_error_mt, float(np.linalg.norm(replay.field(frame, frame.position)-exact_field*1000.)))
        maximum_lookup_gradient_error_t_m = max(maximum_lookup_gradient_error_t_m, float(np.linalg.norm(replay.gradient(frame)-exact_gradient)))
        field_force = exact_gradient.T@moment
        recorded_force = replay.vector(frame,'applied_force','mN')/1000.
        maximum_force_error = max(maximum_force_error,float(np.linalg.norm(field_force-recorded_force)))
        resistance = replay.vector(frame,'applied_resistance','mN')/1000.
        acceleration = replay.vector(frame,'actual_acceleration','mm_s2')/1000.
        maximum_dynamics_error = max(maximum_dynamics_error,float(np.linalg.norm(acceleration-(field_force+resistance)/config['mass_kg']-config['gravity_m_s2'])))
    assert maximum_force_error<1e-10 and maximum_dynamics_error<1e-7
    insertion,withdrawal = replay.coverage(replay.duration)
    assert not (insertion&withdrawal).any()
    assert np.isclose(replay.areas[insertion|withdrawal].sum(),replay.summary['final_covered_area_mm2'])
    report = dict(passed=True,reference_retreat_implementation=implementations,
        patient_posture=posture,maximum_rigid_geometry_error_mm=maximum_geometry_error,
        route_direction=direction,actual_direction_errors_mm=direction_errors,
        method=replay.prediction['method'],prediction_config=replay.prediction['config'],
        replanning_decisions=len(replay.prediction['decisions']),
        minimum_searched_directions=min(row['searched_direction_count'] for row in replay.prediction['decisions']),
        maximum_searched_directions=max(row['searched_direction_count'] for row in replay.prediction['decisions']),
        maximum_force_reconstruction_error_n=maximum_force_error,maximum_dynamics_reconstruction_error_m_s2=maximum_dynamics_error,
        maximum_lookup_field_error_mt=maximum_lookup_field_error_mt,
        maximum_lookup_gradient_error_t_m=maximum_lookup_gradient_error_t_m,
        simulation_reports=replay.payload['magneticSimulation']['reports'],coverage=replay.summary,
        replay_rows=len(replay.times),replay_columns=len(replay.data),duration_s=replay.duration,
        limitation='Internal consistency of the same finite winding, uniform-core and approximate mutual-coupling model; not independent FEM, hardware or clinical validation')
    destination = ROOT/'文档'/'验证'/'磁源与相机逻辑核验.json'
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(json.dumps({key:value for key,value in report.items() if key!='reference_retreat_implementation'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
