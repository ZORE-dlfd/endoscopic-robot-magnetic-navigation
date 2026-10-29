import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'源码'), str(ROOT/'构建'/'依赖'), str(ROOT/'工具')]

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from coil_simulation import simulation
from coil_simulation.export import make_payload, save_payload
from coil_simulation.finite_sources import CoilDesign, FiniteArray
from coil_simulation.finite_adapter import SourceAdapter
from coil_simulation.route_direction import verify_phases
from magnetic_app.domain.replay import Replay
from magnetic_app.rendering.geometry import Domains
from generate_eight_coil import evaluate_coverage


MODEL = 'finite-winding-core-coupled-v1'


def read_json(path):
    return json.loads(path.read_text('utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), 'utf-8')


def restore_rows(folder, reverse, original, config, risk, settings, array, decisions):
    with np.load(folder/'trajectory.npz') as archive:
        saved = {name: archive[name] for name in archive.files}
    route = simulation.make_route(pd.DataFrame(original.source), reverse)
    policy = simulation.CameraPolicy(original, reverse)
    limits = dict(accelerate=risk['tangent_accel_max']/1000., brake=risk['brake_decel_max']/1000.,
                  lateral=risk['normal_accel_max']/1000.)
    decision_times = np.array([decision['time_s'] for decision in decisions])
    camera_acceleration = np.gradient(saved['camera_velocity'], saved['time_s'], axis=0)
    rows = []
    with SourceAdapter(array).installed(simulation):
        for index, time_s in enumerate(saved['time_s']):
            override = None
            if reverse:
                selected = max(0, int(np.searchsorted(decision_times, time_s+1e-9, side='right')-1))
                override = np.asarray(decisions[selected]['target_deg'])
            current = simulation.evaluate(saved['state'][index], route, saved['arc'][index],
                saved['positions'][index], array.axes, config, settings, limits,
                saved['camera_angles'][index], saved['camera_velocity'][index], policy, reverse, override)
            for key in ['currents', 'field', 'gradient', 'force', 'torque', 'speed_cap']:
                np.testing.assert_allclose(current[key], saved[key][index], atol=2e-10, rtol=1e-7,
                                           err_msg=f'Recorded finite-source dynamics differ: {folder.name} {time_s} {key}')
            rows.append(dict(time_s=float(time_s), state=saved['state'][index],
                camera_angles=saved['camera_angles'][index], camera_velocity=saved['camera_velocity'][index],
                camera_acceleration=camera_acceleration[index], anchor=saved['anchor'][index],
                positions=saved['positions'][index], reverse=reverse, **current))
    print(f'Restored and verified {folder.name}: {len(rows)} rows', flush=True)
    return rows


def finalize(assets, run, array):
    replay = Replay(assets, load_predictive=False)
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
    sections['磁场解算'] = '''紧凑八线圈：有限绕组＋有限铁芯＋近似互耦

铁芯长 52 mm、直径 32 mm；绕组长 44 mm、内径 34 mm、外径 56 mm；1200 匝。铜线直径 0.55 mm，含绝缘 0.60 mm，填充率 70.10%。每路线圈电阻约 12.3102 Ω，电流限制 ±10 A；理想驱动电压假设上限 150 V，仅核对阻性压降，不代表已验证电感驱动可达。假设相对磁导率 2000，饱和极化 1.8 T；仅铁芯尺寸由用户指定，其余尚未实测。

八个线圈方位角依次间隔 45°，极角交替 35°、60°，距阵列中心交替 120、97.5 mm，轴线朝中心。相对布局固定，阵列整体平移，速度上限 0.20 m/s。52×56 mm 保守外包络最小间距约 5.00 mm；不包含人体表面或机械臂碰撞。

绕组：对有限厚度、有限长度的圆形电流匝做 Biot–Savart 数值积分，B_w = μ₀/(4π)∫I dl×r/|r|³。铁芯：假设均匀轴向磁化，以两端等效磁荷圆盘积分。退磁因子采用长旋转椭球近似。铁芯互耦采用各铁芯中心轴向场，解 [diag(Nd+1/(μr−1))−C] M = H_w I。实际总场 B = B_w I + B_core M，空间梯度同样由有限磁源求出。

胶囊磁力 F = Gᵀm，磁转矩 τ = m×B。磁场优先分配电流：先满足朝向磁场，再在零空间尽量补足推进磁力；偏置场由旧版 10 mT 调整为这组验证使用的 8 mT。动力学、摩擦、重力、路线、风险减速和摄像头规划公式未改。

全空间箭头、切片和流线使用同一有限磁源的轴对称响应查表和双线性插值，叠加电流及互耦磁化。查表只为显示加速；动力学记录来自有限磁源直接积分。铁芯及绕组保守包络内不显示场。查表数值误差另行检验，不是独立物理精度验证。画面按真实假设尺寸画出铁芯和绕组线框，非 CAD。

实时功耗 P = R I²，总阻性功耗为各路之和；阻性电压 U = RI。磁化百分比为 |M|/Ms，属于均匀铁芯近似，不是局部磁密测量。仍使用理想主动冷却，没有热、电感、磁滞、非线性 B–H、人体接触或机械臂关节模型。控制器与仿真对象使用同一个模型，非独立 FEM 或实物验证。'''
    sections['机器人运动'] = sections['机器人运动'].replace('八个点偶极', '八个有限绕组与有限铁芯磁源')
    write_json(assets/'algorithms.json', sections)
    report = dict(simulation=replay.payload['magneticSimulation'], coverage=replay.summary,
                  camera_planning=replay.prediction, original_validation=read_json(run/'status.json'))
    write_json(ROOT/'文档'/'八线圈仿真结果.json', report)
    manifest = dict(schema=2, method=MODEL, sha256={name: hashlib.sha256((assets/name).read_bytes()).hexdigest()
        for name in ['metadata.json', 'arrays.npz', 'algorithms.json', 'finite_field.npz']})
    write_json(assets/'magnetic_manifest.json', manifest)
    write_json(assets/'provenance.json', dict(example='clon001', magnetic_model=MODEL,
        data='Recorded compact-source dynamics; same original camera and slowdown logic; hypothetical hardware parameters',
        magnetic_validation='Model-consistent feasibility; no independent FEM or hardware measurements', manifest=manifest))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, default=ROOT/'数据'/'紧凑磁源验证')
    parser.add_argument('--assets', type=Path, default=ROOT/'数据'/'clon001')
    args = parser.parse_args()
    assumptions = read_json(args.run/'assumptions.json')
    parameters = {field.name: assumptions['design'][field.name] for field in fields(CoilDesign)}
    array = FiniteArray(CoilDesign(**parameters), assumptions['layout_scale'])
    original = Replay(args.run/'控制参考')
    camera = read_json(args.run/'camera_report.json')
    settings = simulation.Settings(**assumptions['integrator'])
    config = assumptions['robot_config']
    risk = assumptions['risk_assumptions']
    insertion = restore_rows(args.run/'insertion', False, original, config, risk, settings, array, camera['decisions'])
    withdrawal = restore_rows(args.run/'withdrawal', True, original, config, risk, settings, array, camera['decisions'])
    reports = [read_json(args.run/phase/'metrics.json') for phase in ['insertion', 'withdrawal']]
    payload = make_payload(original, insertion, withdrawal, config, settings, reports)
    payload['routeDirection']['verification'] = verify_phases(pd.DataFrame(original.source), insertion, withdrawal)
    payload['cameraPrediction'] = camera
    payload['coilSources'] = dict(model=MODEL, positions_mm=payload['coilSources']['positions_mm'],
        currents_a=payload['coilSources']['currents_a'], axes=array.axes, offsets_m=array.offsets,
        current_limit_a=array.design.current_limit_a, design=array.design.dictionary(),
        magnetization_matrix_a_m_per_a=array.magnetization_matrix, layout_scale=array.layout_scale,
        field_table='finite_field.npz', controller_bias_field_t=config['bias_field_T'])
    payload['magneticSimulation'].update(method=MODEL, field_model='Finite Biot-Savart winding + uniform finite core + center-sampled mutual coupling; not FEM or measured hardware',
        constraints='10 A current and mean-core saturation bounds; rigid translation speed bound; ideal cooling and instantaneous currents',
        camera_method='Original dense sliding-window search + 1.2 s prediction + original PD, jointly replanned offline',
        independent_reference=False, source_design=array.design.dictionary(), route_direction=payload['routeDirection'],
        camera_acceleration_display='Derivative of saved camera velocity at recorded output times')
    payload['registry']['robots'][0]['config']['field_model'] = payload['magneticSimulation']['field_model']
    manifest = args.assets/'magnetic_manifest.json'
    manifest.unlink(missing_ok=True)
    save_payload(payload, args.assets)
    payload = evaluate_coverage(args.assets)
    save_payload(payload, args.assets)
    finalize(args.assets, args.run, array)
    print('Compact finite-source assets ready; original camera and slowdown modules unchanged.', flush=True)


if __name__ == '__main__':
    main()
