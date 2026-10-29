import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'源码'), str(ROOT/'构建'/'依赖')]

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from coil_simulation import model, simulation
from coil_simulation.export import make_payload, save_payload
from coil_simulation.finite_sources import CoilDesign, FiniteArray
from coil_simulation.path_guidance import magnetic_path_demand
from coil_simulation.route_direction import verify_phases
from magnetic_app.domain.replay import Replay
from magnetic_app.rendering.geometry import Domains


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def restore_rows(run, reverse, original, assumptions, decisions):
    phase = 'withdrawal' if reverse else 'insertion'
    with np.load(run/phase/'trajectory.npz', allow_pickle=False) as archive:
        saved = {name: archive[name] for name in archive.files}
    route = simulation.make_route(pd.DataFrame(original.source), reverse)
    policy = simulation.CameraPolicy(original, reverse)
    config = assumptions['robot_config']
    risk = assumptions['risk_assumptions']
    inertia = model.ellipsoid_inertia(config['mass_kg'], config['robot_length_m'], config['robot_diameter_m'])
    decision_times = np.array([item['time_s'] for item in decisions])
    camera_acceleration = np.gradient(saved['camera_velocity'], saved['time_s'], axis=0)
    rows = []
    for index, time_s in enumerate(saved['time_s']):
        state = saved['state'][index]
        arc = saved['arc'][index]
        reference = route.at(arc)
        body = Rotation.from_quat(state[6:10])
        source_arc = route.arc_m[-1]-arc if reverse else arc
        selected = max(0, int(np.searchsorted(decision_times, time_s+1e-9, side='right')-1))
        goal = (np.asarray(decisions[selected]['target_deg']) if reverse
                else policy.goal(source_arc, reference['rotation'], body))
        demand = magnetic_path_demand(model, reference, state, config, saved['speed_cap'][index],
            risk['tangent_accel_max']/1000., risk['brake_decel_max']/1000., risk['normal_accel_max']/1000.)
        derivative = np.empty(13)
        derivative[:3] = state[3:6]
        derivative[3:6] = (saved['force'][index]+demand['resistance'])/config['mass_kg']+config['gravity_m_s2']
        derivative[6:10] = model.quaternion_derivative(state[6:10], state[10:13])
        derivative[10:13] = (body.inv().apply(saved['torque'][index])-np.cross(state[10:13], inertia*state[10:13]))/inertia
        rows.append(dict(time_s=float(time_s), state=state, body=body, reference=reference, source_arc=source_arc,
            demand=demand, derivative=derivative, goal=goal, reverse=reverse,
            camera_angles=saved['camera_angles'][index], camera_velocity=saved['camera_velocity'][index],
            camera_acceleration=camera_acceleration[index], **{key: saved[key][index] for key in
                ['anchor', 'positions', 'currents', 'field', 'gradient', 'force', 'torque', 'speed_cap', 'dilation']}))
    print(f'Imported {phase}: {len(rows)} recorded samples', flush=True)
    return rows, saved


def restore_coverage(payload, run):
    with np.load(run/'insertion_coverage.npz') as archive:
        insertion_seen, areas = archive['first_seen'], archive['areas']
    with np.load(run/'withdrawal_coverage.npz') as archive:
        withdrawal_seen = archive['first_seen']
        np.testing.assert_array_equal(areas, archive['areas'])
    np.testing.assert_allclose(areas, payload['geometry']['areas_mm2'], rtol=0, atol=0)
    withdrawal_seen = np.where(insertion_seen >= 0., -1., withdrawal_seen)
    total = float(areas.sum())
    insertion_area = float(areas[insertion_seen >= 0.].sum())
    added_area = float(areas[withdrawal_seen >= 0.].sum())
    old = payload['withdrawalCoverage']['summary']
    old.update(insertion_covered_area_mm2=insertion_area, withdrawal_added_area_mm2=added_area,
        final_covered_area_mm2=insertion_area+added_area, total_lumen_area_mm2=total,
        uncovered_area_mm2=total-insertion_area-added_area, insertion_coverage_pct=100*insertion_area/total,
        withdrawal_added_pct_of_total_lumen=100*added_area/total,
        withdrawal_added_pct_of_insertion_covered=100*added_area/insertion_area,
        coverage_gain_percentage_points=100*added_area/total, final_coverage_pct=100*(insertion_area+added_area)/total,
        coverage_increased=added_area>0, duration_s=payload['roundTrip']['withdrawal_duration_s'], wait_count=0,
        constraint_violating_steps=payload['cameraPrediction']['metrics']['constraint_violating_steps'],
        method_notice='Exact saved wall-observation times from the verified rotating 20 mm run; dense sliding-window + short-horizon prediction + PD. No forecast counted as actual coverage.')
    payload['evaluation']['methods']['slowdown_new']['first_seen_s'] = insertion_seen
    payload['withdrawalCoverage']['masks'] = dict(insertion_covered=insertion_seen>=0., withdrawal_first_seen_s=withdrawal_seen)
    times = payload['datasets']['capsule_roundtrip']['time_s']
    arcs = payload['datasets']['capsule_roundtrip']['arc_s_mm']
    join = payload['roundTrip']['join_time_s']
    insertion_events = np.sort(insertion_seen[insertion_seen>=0.])
    insertion_order = np.argsort(insertion_seen[insertion_seen>=0.])
    insertion_cumulative = np.r_[0., np.cumsum(areas[insertion_seen>=0.][insertion_order])]
    withdrawal_events = np.sort(withdrawal_seen[withdrawal_seen>=0.])
    withdrawal_order = np.argsort(withdrawal_seen[withdrawal_seen>=0.])
    withdrawal_cumulative = np.r_[0., np.cumsum(areas[withdrawal_seen>=0.][withdrawal_order])]
    rows = []
    for time_s, arc in zip(times, arcs):
        inserted = float(insertion_cumulative[np.searchsorted(insertion_events, min(time_s, join), side='right')])
        added = float(withdrawal_cumulative[np.searchsorted(withdrawal_events, time_s-join, side='right')]) if time_s>=join else 0.
        rows.append(dict(time_s=float(time_s), arc_s_mm=float(arc), coverage_pct=100*(inserted+added)/total,
            unseen_pct=100*(1-(inserted+added)/total), effective_pct=None, repeat_pct=None, effective_time_mean_pct=None,
            completion_pct=100*(1-arc/payload['source']['arc_s_mm'][-1]) if time_s>=join else 100*arc/payload['source']['arc_s_mm'][-1],
            insertion_covered_area_mm2=inserted, withdrawal_added_area_mm2=added,
            withdrawal_added_pct_of_total_lumen=100*added/total, coverage_gain_percentage_points=100*added/total,
            final_covered_area_mm2=inserted+added))
    payload['evaluation']['methods']['withdrawal_new']['rows'] = rows
    payload['roundTrip']['insertion_covered_area_mm2'] = insertion_area
    return old


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'数据'/'20mm转动验证')
    args = parser.parse_args()
    assets = ROOT/'数据'/'clon001'
    assumptions = read_json(args.run/'assumptions.json')
    certificate = read_json(args.run/'完整核算.json')
    if not certificate['full_route_success'] or not all(certificate['checks'].values()):
        raise ValueError('A complete verified run is required')
    original = Replay()
    camera = read_json(args.run/'camera_report.json')
    insertion, first = restore_rows(args.run, False, original, assumptions, camera['decisions'])
    withdrawal, second = restore_rows(args.run, True, original, assumptions, camera['decisions'])
    reports = [read_json(args.run/phase/'metrics.json') for phase in ['insertion', 'withdrawal']]
    payload = make_payload(original, insertion, withdrawal, assumptions['robot_config'],
                           simulation.Settings(**assumptions['integrator']), reports)
    payload['routeDirection']['verification'] = verify_phases(pd.DataFrame(original.source), insertion, withdrawal)
    payload['cameraPrediction'] = camera
    parameters = {field.name: assumptions['design'][field.name] for field in fields(CoilDesign)}
    array = FiniteArray(CoilDesign(**parameters), assumptions['layout_scale'])
    combined = lambda key: np.concatenate([first[key], second[key][1:]], axis=0)
    payload['coilSources'] = dict(model='finite-winding-core-coupled-v1', positions_mm=combined('positions')*1000.,
        currents_a=combined('currents'), axes=np.asarray(assumptions['axes']), offsets_m=np.asarray(assumptions['offsets_m']),
        current_limit_a=assumptions['design']['current_limit_a'], design=assumptions['design'],
        magnetization_matrix_a_m_per_a=array.magnetization_matrix, layout_scale=assumptions['layout_scale'],
        field_table='finite_field.npz', controller_bias_field_t=assumptions['robot_config']['bias_field_T'],
        array_quaternion=combined('array_quaternion'), anchor_mm=combined('anchor')*1000.,
        certified_sample_excess_clearance_mm=combined('coil_clearance_lower_bound_m')*1000.,
        target_intestine_gap_mm=20.)
    payload['magneticSimulation'].update(method='rotating-eight-coil-20mm-recorded-v1',
        field_model=assumptions['robot_config']['field_model'], source_design=assumptions['design'],
        constraints='30 A, 400 V ideal driver; 2.4 T mean saturation; 30 deg/s rotation, 60 deg/s2; ideal cooling',
        camera_method=camera['method'], independent_reference=False, clearance_certificate=certificate,
        limitations=assumptions['limitations'], no_position_forcing=True, no_camera_wait=True)
    payload['registry']['robots'][0]['config'] = assumptions['robot_config']
    payload['registry']['robots'][0].update(
        magnetic_moment_A_m2=assumptions['robot_config']['magnetic_moment_A_m2'],
        mass_g=assumptions['robot_config']['mass_kg']*1000.,
        sample_count=len(payload['datasets']['capsule_roundtrip']['time_s']))
    summary = restore_coverage(payload, args.run)
    np.testing.assert_allclose(summary['final_coverage_pct'], certificate['combined_coverage_percent'], atol=1e-9)
    manifest = assets/'magnetic_manifest.json'
    manifest.unlink(missing_ok=True)
    save_payload(payload, assets)
    replay = Replay()
    domains = Domains(replay)
    for scope in ['local', 'centerline', 'whole']:
        for mode in ['total', 'residual', 'turning']:
            magnitudes = []
            for time_s in np.linspace(0., replay.duration, 31):
                frame = replay.frame(time_s)
                points = domains.points(frame, scope)
                points = points[domains.valid(points, frame, scope)]
                magnitudes.extend(np.linalg.norm(replay.field(frame, points, mode), axis=-1))
            replay.registry['field_scope_ranges'][scope][mode] = [0., max(1e-8, float(np.percentile(magnitudes, 99)))]
    save_payload(replay.payload, assets)
    sections = read_json(assets/'algorithms.json')
    sections['磁场解算'] = '''八线圈有限绕组、有限铁芯、近似互耦，整体平移并转动。
铁芯52×32 mm，绕组长44 mm、外径56 mm，每路1200匝、电阻12.3102 Ω。
每路电流±30 A；理想驱动400 V，仅含阻性压降。饱和极化2.4 T。
胶囊磁矩4.0 A·m²，偏置场0.05 mT，属于明显理想化假设，不代表现有小胶囊可制造。
刚体姿态使用记录四元数进行球面插值，八路中心位置和轴向一起旋转，不能用位置弦线插值压扁阵列。
世界磁场由相同有限磁源响应查表叠加；显示查表不驱动动力学。力F=Gᵀm，力矩τ=m×B。
动力学、相机姿态、减速和覆盖来自已核算通过的20 mm连续进退镜记录，非播放时重新规划。
角速上限30°/s，角加速60°/s²，平移0.5 m/s。线圈包络到肠道网格全程保守净距≥20.0298 mm。
只避肠道网格，不含皮肤、床面、机械臂和导线碰撞。原记录的连续运动避碰证书不等同于回放插值重新认证。
仍为均匀铁芯、近似互耦、理想散热；无完整温升、电感、磁滞，不是独立FEM或硬件验证。
磁铁模型显示开关只改变绘制，不改变磁场、轨迹、摄像头或覆盖。'''
    write_json(assets/'algorithms.json', sections)
    manifest_data = dict(schema=2, method='rotating-eight-coil-20mm-recorded-v1', sha256={
        name: hashlib.sha256((assets/name).read_bytes()).hexdigest()
        for name in ['metadata.json', 'arrays.npz', 'algorithms.json', 'finite_field.npz']})
    write_json(manifest, manifest_data)
    write_json(assets/'provenance.json', dict(example='clon001', source_run=args.run.name,
        recorded_clearance_mm=20., magnetic_validation='Model-consistent simulation, no independent FEM or hardware',
        camera_method=camera['method'], manifest=manifest_data))
    write_json(ROOT/'文档'/'20mm软件数据接入.json', dict(source_run=args.run.name, coverage=summary,
        certificate=certificate, display_interpolation='Quaternion SLERP + rigid offsets + linear anchor interpolation'))
    print(f'20 mm assets ready: {replay.duration:.2f}s, coverage {summary["final_coverage_pct"]:.8f}%', flush=True)


if __name__ == '__main__':
    main()
