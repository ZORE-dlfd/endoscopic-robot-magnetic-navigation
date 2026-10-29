import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'源码'),str(ROOT/'构建'/'依赖')]

import numpy as np

from coil_simulation.simulation import Settings, simulate_phase
from coil_simulation.export import make_payload, save_payload
from magnetic_app.domain.replay import Replay
from retreat_planner.scene import Scene
from retreat_planner.visibility import Visibility
from retreat_planner.coverage import Coverage
from coil_simulation.coupled_camera import CoupledCamera
from coil_simulation.posture import SupineTransform
from coil_simulation.route_direction import insertion_source, verify_phases, direction_metadata


def evaluate_coverage(destination):
    scene = Scene(destination)
    replay = scene.replay
    settings = replay.settings
    visibility = Visibility(scene)
    insertion = Coverage(replay.areas,np.zeros(len(replay.areas),bool),settings['minimum_dwell_s'],settings['repeat_gap_s'])
    rows = []
    phases = [(0.,replay.join_time,False),(replay.join_time,replay.duration,True)]
    started = time.perf_counter()
    withdrawal = None
    try:
        for start, stop, reverse in phases:
            if reverse:
                covered = insertion.first_seen >= 0
                withdrawal = Coverage(replay.areas,covered,settings['minimum_dwell_s'],settings['repeat_gap_s'])
            tracker = withdrawal if reverse else insertion
            times = np.unique(np.r_[np.arange(start,stop,.1),stop])
            previous = start
            for index, absolute in enumerate(times):
                frame = replay.frame(absolute)
                visible = visibility.visible(frame.camera_position,frame.camera_rotation)
                tracker.observe(visible,absolute-start,absolute-previous)
                previous = absolute
                insertion_area = float(replay.areas[insertion.first_seen>=0].sum())
                added = withdrawal.added_area if reverse else 0.
                total = float(replay.areas.sum())
                rows.append(dict(time_s=float(absolute),arc_s_mm=replay.value(frame,'arc_s_mm'),
                    coverage_pct=100*(insertion_area+added)/total,unseen_pct=100*(1-(insertion_area+added)/total),
                    effective_pct=None,repeat_pct=None,effective_time_mean_pct=None,
                    completion_pct=100*(1-replay.value(frame,'arc_s_mm')/replay.source['arc_s_mm'][-1]) if reverse else 100*replay.value(frame,'arc_s_mm')/replay.source['arc_s_mm'][-1],
                    insertion_covered_area_mm2=insertion_area,withdrawal_added_area_mm2=added,
                    withdrawal_added_pct_of_total_lumen=100*added/total,coverage_gain_percentage_points=100*added/total,
                    final_covered_area_mm2=insertion_area+added))
                if index % 500 == 0:
                    print(f'coverage {"withdrawal" if reverse else "insertion"}: {absolute-start:.1f}/{stop-start:.1f}s, elapsed={time.perf_counter()-started:.1f}s',flush=True)
        first_seen = withdrawal.first_seen.copy()
        first_seen[insertion.first_seen>=0] = -1.
        total = float(replay.areas.sum())
        insertion_area = float(replay.areas[insertion.first_seen>=0].sum())
        added = withdrawal.added_area
        summary = dict(insertion_covered_area_mm2=insertion_area,withdrawal_added_area_mm2=added,
            final_covered_area_mm2=insertion_area+added,total_lumen_area_mm2=total,
            uncovered_area_mm2=total-insertion_area-added,insertion_coverage_pct=100*insertion_area/total,
            withdrawal_added_pct_of_total_lumen=100*added/total,
            withdrawal_added_pct_of_insertion_covered=100*added/max(insertion_area,1e-9),
            coverage_gain_percentage_points=100*added/total,final_coverage_pct=100*(insertion_area+added)/total,
            coverage_increased=added>0,duration_s=replay.duration-replay.join_time,wait_count=0,
            forward_cone_limit_deg=45.,method_notice='Coverage recomputed at 0.1 s with new magnetic body poses and gimbal PD; original search goals arc-remapped. Not measured or clinical coverage.')
        if 'cameraPrediction' in replay.payload:
            summary['constraint_violating_steps'] = replay.payload['cameraPrediction']['metrics']['constraint_violating_steps']
            summary['method_notice'] = 'Eight-coil dynamics coupled with original dense window search + 1.2 s prediction + 0.01 s PD; actual wall coverage recomputed. Not measured or clinical coverage.'
        payload = replay.payload
        payload['evaluation']['methods']['slowdown_new']['first_seen_s'] = insertion.first_seen
        payload['evaluation']['methods']['withdrawal_new']['rows'] = rows
        payload['withdrawalCoverage'] = dict(masks=dict(insertion_covered=insertion.first_seen>=0,withdrawal_first_seen_s=first_seen),summary=summary)
        payload['roundTrip']['insertion_covered_area_mm2'] = insertion_area
        payload['magneticSimulation']['coverage_renderer'] = visibility.environment
        if 'cameraPrediction' in payload:
            metrics = payload['cameraPrediction']['metrics']
            metrics['online_observed_added_area_mm2'] = metrics['withdrawal_added_area_mm2']
            metrics['withdrawal_added_area_mm2'] = added
        return payload
    finally:
        visibility.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--step',type=float,default=.02)
    parser.add_argument('--destination',type=Path,default=ROOT/'数据'/'clon001')
    arguments = parser.parse_args()
    destination = arguments.destination.resolve()
    if destination == (ROOT/'数据'/'参考原版').resolve():
        raise ValueError('Cannot overwrite original reference assets')
    if (destination/'magnetic_manifest.json').exists():
        raise RuntimeError('Destination still has an old magnetic_manifest.json; generate into a fresh staging directory')
    reference = Replay(ROOT/'数据'/'参考原版')
    transform = SupineTransform.from_points(reference.points)
    reference_path = ROOT/'构建'/'仰卧参考'
    save_payload(transform.payload(reference.payload),reference_path)
    original = Replay(reference_path,load_predictive=False)
    inputs = ROOT/'数据'/'仿真输入'
    config = json.loads((inputs/'config.json').read_text('utf-8'))
    config['bias_field_T'] = .01
    config['gravity_m_s2'] = [0.,0.,-9.81]
    config['gravity_frame_assumption'] = 'Assumed supine anatomy; gravity fixed in world -Z; source anatomical axes unverified'
    route_contract = json.loads((inputs/'route_direction.json').read_text('utf-8'))
    source = transform.source(insertion_source(reference,route_contract))
    for key in ['arc_s_mm','x_mm','y_mm','z_mm','qx','qy','qz','qw']:
        np.testing.assert_allclose(source[key],original.source[key],atol=1e-10)
    risk = json.loads((inputs/'simulation_summary.json').read_text('utf-8'))['assumptions']
    settings = Settings(step_s=arguments.step)
    insertion, initial, forward_report = simulate_phase(source,config,risk,original,settings=settings)
    scene = Scene(reference_path)
    insertion_tracker = Coverage(scene.areas,np.zeros(len(scene.areas),bool),scene.settings['minimum_dwell_s'],scene.settings['repeat_gap_s'])
    visibility = Visibility(scene)
    try:
        previous = 0.
        from scipy.spatial.transform import Rotation
        for row in insertion:
            body = row['body'].as_matrix()
            origin = row['state'][:3]*1000.+body[:,0]*scene.settings['mount_offset_mm']
            camera = body@Rotation.from_euler('ZYX',row['camera_angles'],degrees=True).as_matrix()
            insertion_tracker.observe(visibility.visible(origin,camera),row['time_s'],row['time_s']-previous)
            previous = row['time_s']
    finally:
        visibility.close()
    planner = CoupledCamera(scene,insertion_tracker.first_seen>=0)
    try:
        withdrawal, final, reverse_report = simulate_phase(source,config,risk,original,reverse=True,initial=initial,
            settings=settings,camera_planner=planner)
        camera_prediction = planner.report()
    finally:
        planner.close()
    direction_check = verify_phases(source,insertion,withdrawal)
    payload = make_payload(original,insertion,withdrawal,config,settings,[forward_report,reverse_report])
    payload['routeDirection'] = direction_metadata(source,route_contract,direction_check)
    payload['magneticSimulation']['route_direction'] = payload['routeDirection']
    payload['cameraPrediction'] = camera_prediction
    payload['magneticSimulation']['camera_method'] = 'Original dense window search + 1.2 s prediction + original 0.01 s PD, replanned jointly with new magnetic withdrawal dynamics'
    destination.mkdir(parents=True,exist_ok=True)
    for name in ['algorithms.json','app.ico']:
        target = destination/name
        if not target.exists():
            shutil.copy2(ROOT/'数据'/'参考原版'/name,target)
    save_payload(payload,destination)
    print('Magnetic dynamics exported; recomputing all wall coverage.',flush=True)
    payload = evaluate_coverage(destination)
    save_payload(payload,destination)
    report = dict(simulation=payload['magneticSimulation'],coverage=payload['withdrawalCoverage']['summary'])
    (ROOT/'文档'/'八线圈仿真结果.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),'utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
