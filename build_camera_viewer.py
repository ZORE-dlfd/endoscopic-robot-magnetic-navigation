import copy
import hashlib
import json
import re
from pathlib import Path

import markdown
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from plotly.offline import get_plotlyjs

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'Magnetic 1 - Field scopes'


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def json_text(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<' + chr(92) + '/')


def combine_roundtrip(slowdown, withdrawal, insertion_track, withdrawal_track):
    insertion_duration = float(slowdown.time_s.iloc[-1])
    insertion = slowdown.copy()
    insertion['phase'] = '进镜'
    retreat = withdrawal.iloc[1:].copy()
    retreat['time_s'] = retreat['time_s'] + insertion_duration
    retreat['phase'] = '退镜'
    combined = pd.concat([insertion, retreat], ignore_index=True, sort=False)
    for column in combined.columns:
        if combined[column].isna().any():
            combined[column] = combined[column].fillna(1.0 if column == 'time_dilation' else 0.0)

    camera = {}
    camera_duration = insertion_duration
    for key, values in insertion_track.items():
        other = withdrawal_track.get(key)
        if not isinstance(values, list) or not isinstance(other, list):
            continue
        if len(values) != len(insertion_track['time_s']) or len(other) != len(withdrawal_track['time_s']):
            continue
        camera[key] = values + other[1:]
    camera['time_s'] = insertion_track['time_s'] + [camera_duration + value for value in withdrawal_track['time_s'][1:]]
    return combined, camera, insertion_duration


def roundtrip_coverage_rows(times, arc, areas, insertion_first_seen, recoverable, withdrawal_first_seen, join_time):
    times = np.asarray(times, dtype=float)
    arc = np.asarray(arc, dtype=float)
    areas = np.asarray(areas, dtype=float)
    insertion_first_seen = np.asarray(insertion_first_seen, dtype=float)
    recoverable = np.asarray(recoverable, dtype=bool)
    withdrawal_first_seen = np.asarray(withdrawal_first_seen, dtype=float)
    insertion_covered = insertion_first_seen >= 0.0
    withdrawal_new = recoverable & ~insertion_covered & (withdrawal_first_seen >= 0.0)
    total_area = float(areas.sum())
    insertion_area = float(areas[insertion_covered].sum())
    withdrawal_times = withdrawal_first_seen + join_time
    rows = []
    for time, position in zip(times, arc):
        inserted_area = float(areas[insertion_covered & (insertion_first_seen <= time)].sum())
        added_area = float(areas[withdrawal_new & (withdrawal_times <= time)].sum()) if time >= join_time else 0.0
        total_covered = inserted_area + added_area
        rows.append({'time_s': float(time), 'arc_s_mm': float(position),
                     'coverage_pct': 100.0 * total_covered / max(total_area, 1e-12),
                     'unseen_pct': 100.0 - 100.0 * total_covered / max(total_area, 1e-12),
                     'effective_pct': None, 'repeat_pct': None,
                     'effective_time_mean_pct': None, 'completion_pct': min(100.0, 100.0 * time / max(times[-1], 1e-12)),
                     'insertion_covered_area_mm2': inserted_area,
                     'withdrawal_added_area_mm2': added_area,
                     'withdrawal_added_pct_of_total_lumen': 100.0 * added_area / max(total_area, 1e-12),
                     'coverage_gain_percentage_points': 100.0 * added_area / max(total_area, 1e-12),
                     'final_covered_area_mm2': total_covered})
    return rows, insertion_area


def build():
    base_html = (BASE / 'Magnetic_Interactive_v4.html').read_text(encoding='utf-8')
    original_payload = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', base_html, re.S)[1])
    payload = {key: original_payload[key] for key in ['source', 'labels']}
    registry = copy.deepcopy(original_payload['registry'])
    original = pd.read_csv(BASE / 'capsule_replay.csv')
    cooperative = pd.read_csv(ROOT / 'capsule_cooperative_replay.csv')
    slowdown = pd.read_csv(ROOT / 'capsule_slowdown_replay.csv')
    wait = pd.read_csv(ROOT / 'capsule_wait_replay.csv')
    withdrawal = pd.read_csv(ROOT / 'capsule_withdrawal_replay.csv')
    robot = next(robot for robot in registry['robots'] if robot['id'] == 'capsule')
    cameras = read_json('camera_trajectories.json')
    withdrawal_camera = read_json('withdrawal_camera.json')
    roundtrip, roundtrip_camera, join_time = combine_roundtrip(
        slowdown, withdrawal, cameras['slowdown_new'], withdrawal_camera)
    profiles = [('capsule_roundtrip', '只减速 · 进镜→退镜', roundtrip, 'capsule_roundtrip_replay.csv')]
    registry['robots'] = []
    datasets = {}
    for identifier, label, frame, filename in profiles:
        profile = copy.deepcopy(robot)
        profile.update({'id': identifier, 'name': '胶囊 · ' + label, 'sample_count': len(frame), 'csv': filename})
        registry['robots'].append(profile)
        datasets[identifier] = frame.to_dict(orient='list')
    source_frame = pd.DataFrame(payload['source'])
    fields = cooperative[[f'applied_field_{axis}_mT' for axis in 'xyz']].to_numpy()
    origins = cooperative[[f'actual_position_{axis}_mm' for axis in 'xyz']].to_numpy()
    directions = cooperative[[f'actual_axis_{axis}_unit' for axis in 'xyz']].to_numpy()
    gradients = cooperative[[f'applied_G_{first}{second}_T_m' for first in 'xyz' for second in 'xyz']].to_numpy().reshape(-1,3,3)
    source_positions = source_frame[['x_mm','y_mm','z_mm']].to_numpy()
    bounds = [source_positions.min(axis=0)-30,source_positions.max(axis=0)+30]
    for scope in ['local', 'centerline', 'whole']:
        ranges = registry['field_scope_ranges'][scope]
        for first in [-1,1]:
            for second in [-1,1]:
                for third in [-1,1]:
                    signs=np.array([first,second,third])
                    delta=np.tile(signs*30,(len(cooperative),1)) if scope=='local' else np.where(signs>0,bounds[1],bounds[0])-origins
                    variation=np.einsum('nij,nj->ni',gradients,delta)
                    ranges['total'][1]=max(ranges['total'][1],float(np.ceil(np.linalg.norm(fields+variation,axis=1).max()*10)/10))
                    residual=fields-robot['config']['bias_field_T']*1000*directions+variation
                    ranges['residual'][1]=max(ranges['residual'][1],float(np.ceil(np.linalg.norm(residual,axis=1).max()*10)/10))
        ranges['turning'][1] = max(ranges['turning'][1], float(np.ceil(cooperative.applied_field_perpendicular_mT.max() * 100000)/100))
    payload.update({'registry': registry, 'datasets': datasets, 'geometry': read_json('geometry.json'),
                    'cameras': read_json('camera_trajectories.json'), 'evaluation': read_json('evaluation.json'),
                    'coordinationSummary': read_json('coordination_summary.json')})
    payload['cameras']['withdrawal_new'] = withdrawal_camera
    payload['cameras']['roundtrip_new'] = roundtrip_camera
    payload['withdrawalCoverage'] = {
        'summary': read_json('withdrawal_summary.json'),
        'masks': read_json('withdrawal_visibility_masks.json')}
    masks = payload['withdrawalCoverage']['masks']
    insertion_first_seen = payload['evaluation']['methods']['slowdown_new']['first_seen_s']
    withdrawal_rows, insertion_area = roundtrip_coverage_rows(
        roundtrip.time_s, roundtrip.arc_s_mm, payload['geometry']['areas_mm2'], insertion_first_seen,
        masks['recoverable'], masks['withdrawal_first_seen_s'], join_time)
    payload['evaluation']['methods']['withdrawal_new'] = {
        'rows': withdrawal_rows,
        'first_seen_s': insertion_first_seen,
        'repeat_seen_s': payload['evaluation']['methods']['slowdown_new']['repeat_seen_s'],
        'final': withdrawal_rows[-1], 'common_budget': withdrawal_rows[-1]}
    payload['roundTrip'] = {'join_time_s': join_time, 'join_index': len(slowdown)-1,
                            'total_duration_s': float(roundtrip.time_s.iloc[-1]),
                            'insertion_duration_s': join_time,
                            'withdrawal_duration_s': float(withdrawal.time_s.iloc[-1]),
                            'insertion_covered_area_mm2': insertion_area,
                            'endpoint_position_gap_mm': float(np.linalg.norm(
                                slowdown[['actual_position_x_mm','actual_position_y_mm','actual_position_z_mm']].iloc[-1].to_numpy(dtype=float) -
                                withdrawal[['actual_position_x_mm','actual_position_y_mm','actual_position_z_mm']].iloc[0].to_numpy(dtype=float))),
                            'note': 'The reverse replay is independently solved. Endpoint residual is exposed; no artificial pose correction is applied.'}
    payload['cameraMotion']={}
    for method in ['fixed_new','fixed_endoscopy','slowdown_new','slowdown_endoscopy','wait_new','wait_endoscopy',
                   'coordinated_new','coordinated_endoscopy','withdrawal_new','roundtrip_new']:
        track=payload['cameras'][method]
        rotations=Rotation.from_quat(track['quaternion'])
        intervals=np.diff(track['time_s'])
        relative=(rotations[:-1].inv()*rotations[1:]).as_rotvec()/intervals[:,None]
        roll=np.r_[0,np.rad2deg(relative[:,0])]
        world=np.vstack([np.zeros(3),rotations[:-1].apply(relative)])
        alpha=np.gradient(world,track['time_s'],axis=0)
        payload['cameraMotion'][method]={'roll_rate_deg_s':roll.tolist(),'roll_rate_p95_deg_s':float(np.percentile(abs(roll),95)),
                                        'angular_acceleration_p95_deg_s2':float(np.percentile(np.rad2deg(np.linalg.norm(alpha,axis=1)),95))}
    (ROOT/'camera_motion_summary.json').write_text(json.dumps({key:{metric:value for metric,value in result.items() if metric!='roll_rate_deg_s'} for key,result in payload['cameraMotion'].items()},indent=2),encoding='utf-8')
    page = (BASE / 'viewer_template.html').read_text(encoding='utf-8')
    page = page.replace('<title>Magnetic v4 · 中心线与磁场联动</title>', '<title>Magnetic · 胶囊双视角与观察覆盖</title>')
    page = page.replace('<script>__PLOTLY__</script>', '<script>__PLOTLY__</script><script>__THREE__</script>')
    page = page.replace('推进与转向实验台', '胶囊视觉导航实验台').replace('风险路径 · 推进力与梯度 · 磁场分解', '双视角 · 前瞻稳定 · 同条件观察覆盖评价')
    page = page.replace('当前机器人</label>', '运动方案</label>').replace('aria-label="机器人形状"', 'aria-label="胶囊运动方案"')
    page = page.replace('各形状使用独立计算结果', '胶囊型 · 只减速 · 进镜后连续退镜')
    slider_markup = '<div class="roundtrip-slider"><input id="slider" type="range" min="0" max="1435" step="1" value="0" aria-label="进镜与退镜总时间进度"><span id="roundTripJoinMarker" class="roundtrip-join-marker"><b>交接</b><small>开始退镜</small></span></div>'
    page = re.sub(r'<input id="slider" type="range"[^>]*>', slider_markup, page, count=1)
    page = page.replace('<span>拖动进度条查看任意采样时刻</span>', '<span>进镜 → 交接点 → 退镜</span>')
    page = page.replace('<h2>中心线、机器人与磁场</h2>', '<h2>第三人称 · 肠道与机器人</h2>')
    page = page.replace('<option value="1">真实尺寸 1×</option><option value="3" selected>', '<option value="1" selected>真实尺寸 1×</option><option value="3">')
    page = page.replace('<h2>第三人称 · 肠道与机器人</h2></div><div class="scene-options">', '<h2>第三人称 · 肠道与机器人</h2></div><label class="toggle scene-bowel-toggle"><input type="checkbox" id="showBowel" checked><span>显示肠道</span></label><div class="scene-options">')
    layer_controls = '<label class="toggle"><input type="checkbox" id="showCoverage"><span>覆盖染色</span></label><label class="toggle"><input type="checkbox" id="showFrustum" checked><span>视锥</span></label><label class="toggle"><input type="checkbox" id="showCameraTarget" checked><span>前瞻目标</span></label>'
    page = page.replace('<span class="layer-title">显示图层</span>', '<span class="layer-title">显示图层</span>' + layer_controls)
    page = page.replace('<div class="field-status">', '<div class="camera-controls"><label class="small-select">肠壁透明度<select id="bowelOpacity"><option value=".12">12%</option><option value=".22" selected>22%</option><option value=".45">45%</option></select></label><label class="small-select">视锥长度<select id="frustumLength"><option value="15">15 mm</option><option value="25" selected>25 mm</option><option value="40">40 mm</option></select></label></div><div class="field-status">')
    camera_panel = '''<section class="panel camera-panel"><div class="panel-heading"><div><span class="eyebrow">ONBOARD CAMERA</span><h2>第一人称 · 胶囊前端</h2></div><span id="cameraStamp" class="chip">0 s</span></div><div class="camera-controls"><label class="small-select">相机策略<select id="cameraMethod"><option value="new">前瞻稳定相机</option><option value="endoscopy">Endoscopy 基线</option></select></label><span class="caption">垂直 FOV 60° · 4:3 · 同一内壁几何</span></div><div class="camera-canvas-wrap"><canvas id="endoscopyCanvas" aria-label="胶囊第一人称肠道内视图"></canvas></div><div id="cameraReadout" class="camera-readout"></div><div id="cameraJointReadout" class="camera-joints"></div><p class="camera-caption">摄像头固定于真实尺寸胶囊前端。第一人称不叠加磁场和坐标轴；几何光照渲染，不是真实黏膜纹理。左侧视锥表示同一相机。</p></section>'''
    marker = '        </div>\n        <section class="panel propulsion-chart">'
    if marker not in page:
        raise ValueError('Unexpected baseline views structure')
    page = page.replace(marker, camera_panel + '\n        </div>\n__EVALUATION__\n        <section class="panel propulsion-chart">')
    evaluation_panel = '''<section class="panel evaluation-panel"><div class="panel-heading"><div><span class="eyebrow">ROUND TRIP · UNIQUE SURFACE COVERAGE</span><h2>进镜后退镜：补拍面积统计</h2></div><button id="downloadEvaluation" class="subtle">导出评价 JSON ↓</button></div><div class="camera-spec"><span>动画：只减速进镜 → 连续退镜</span><span>覆盖统计分母：完整肠壁面片</span><span>退镜目标：前向可见的进镜漏拍区域</span><span>观察门槛：0.4 s（离散近似）</span></div><div id="coverageCards" class="coverage-cards"></div><p id="coverageStamp" class="chart-note"></p><div class="coverage-legend"><b style="color:#df6260">■ 尚未覆盖（含褶皱后区域）</b><b style="color:#45a798">■ 已覆盖</b><b style="color:#338bd1">■ 退镜新增</b>勾选左侧“覆盖染色”查看；重复观察不重复计面积。</div><div class="camera-controls evaluation-head"><label class="small-select">比较方式<select id="evaluationMode"><option value="system">完整系统 · 含减速</option><option value="camera">相机方向对照</option></select></label><label class="small-select">结果口径<select id="evaluationSummary"><option value="final">走完全路径</option><option value="common_budget">共同时间预算</option></select></label><label class="small-select">评价曲线<select id="evaluationMetric"><option value="coverage_pct">覆盖率</option><option value="unseen_pct">未观察比例</option><option value="effective_pct">有效视野占比</option><option value="repeat_pct">重复观察率</option></select></label></div><div id="evaluationPlot" class="plot2d"></div><div id="evaluationTable" class="evaluation-table"></div><p id="evaluationNotice" class="camera-warning"></p><p class="camera-warning">面积统计保留全部肠壁，包括当前离散采样视角下不可见的褶皱后面片；退镜选点不追逐这些不可见面片。结果来自单例离线几何仿真，不代表临床覆盖率或安全性验证。</p></section>'''
    evaluation_panel = evaluation_panel.replace(
        '<div id="coverageCards" class="coverage-cards"></div><p id="coverageStamp" class="chart-note"></p>',
        '<div id="coverageCards" class="coverage-cards"></div><p id="coverageStamp" class="chart-note"></p>'
        '<div class="coverage-formula"><b>怎么计算</b>'
        '<span>完整肠壁 A<sub>total</sub> = Σa<sub>i</sub>（所有面片，含褶皱后区域）</span>'
        '<span>退镜新增 A<sub>new</sub> = Σa<sub>i</sub>R<sub>i</sub>(1−M<sub>i</sub>)W<sub>i</sub></span>'
        '<span>退镜增加百分点 = 100 × A<sub>new</sub> / A<sub>total</sub></span>'
        '<small>Mᵢ/Wᵢ分别表示进镜/退镜达到有效观察门槛；Rᵢ表示退镜候选视角下可见。褶皱后面片Rᵢ=0，不作为退镜目标，但仍留在总面积和未覆盖面积中。</small></div>')
    table_rows = ''.join(
        '<tr><td>' + row['label'] + '</td>' + ''.join(
            f'<td>{row[key]:.2f}</td>' for key in ['duration_s', 'tracking_error_p95_deg',
                                                  'full_path_coverage_pct', 'budget_coverage_pct',
                                                  'budget_completion_pct', 'wait_duration_s'])
        + '<td>' + (f"{row['wait_release_pass_count']}/{row['wait_count']}" if row['wait_count'] else '无等待') + '</td></tr>'
        for row in payload['coordinationSummary']['rows'])
    evaluation_panel += '''<section class="panel evaluation-panel"><div class="panel-heading"><div><span class="eyebrow">COORDINATION · DATA ONLY</span><h2>速度与镜头协调：策略数据对照</h2></div></div><p class="camera-caption">这里只保留原速、只减速、只等待、减速＋等待的评价数据；动画只播放只减速进镜并连续退镜。</p><div id="coordinationTable" class="evaluation-table"><table><thead><tr><th>策略</th><th>全程时间 s</th><th>跟踪误差 P95 °</th><th>全程覆盖 %</th><th>预算内覆盖 %</th><th>预算内完成 %</th><th>等待总时长 s</th><th>释放核验</th></tr></thead><tbody>''' + table_rows + '''</tbody></table></div><p class="camera-warning">表格中的策略不是可切换动画；指标沿用原离线比较口径。</p></section>'''
    page = page.replace('__EVALUATION__', evaluation_panel)
    page = page.replace('<p id="evaluationNotice"', '<details class="camera-caption"><summary>展开相机运动辅助指标</summary><div id="motionComparison"></div></details><p id="evaluationNotice"')
    docs = markdown.markdown((ROOT / '说明与评价公式.md').read_text(encoding='utf-8'), extensions=['tables', 'fenced_code'])
    docs += markdown.markdown((ROOT / 'WITHDRAWAL_METHOD.md').read_text(encoding='utf-8'), extensions=['tables', 'fenced_code'])
    css = (ROOT / 'viewer.css').read_text(encoding='utf-8') + '\n' + (ROOT / 'camera.css').read_text(encoding='utf-8')
    scripts = '\n'.join((ROOT / name).read_text(encoding='utf-8') for name in ['viewer_core.js', 'field_scope.js', 'field_view.js', 'camera_view.js', 'viewer.js'])
    for marker, value in {'__CSS__': css, '__PLOTLY__': get_plotlyjs(), '__THREE__': (ROOT / 'vendor' / 'three.min.js').read_text(encoding='utf-8'), '__PAYLOAD__': json_text(payload), '__SCRIPT__': scripts, '__FORMULAS__': docs}.items():
        page = page.replace(marker, value)
    (ROOT / 'Magnetic_Camera.html').write_text(page, encoding='utf-8')
    print(json.dumps({'html_bytes': len(page.encode('utf-8')), 'datasets': list(datasets), 'triangles': payload['geometry']['evaluation_triangles']}))


if __name__ == '__main__':
    build()
