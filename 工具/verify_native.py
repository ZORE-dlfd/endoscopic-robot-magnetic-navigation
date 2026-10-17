import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil


ROOT = Path(__file__).resolve().parents[1]


def validate_smoke(directory):
    result = json.loads((directory/'smoke.json').read_text(encoding='utf-8'))
    assert result['render_errors'] == [], result['render_errors']
    assert result['python_exceptions'] == [], result['python_exceptions']
    assert result['mount_error_mm'] < 1e-9
    for stage in ['insertion_1x', 'withdrawal_1x', 'withdrawal_5x', 'slider_16ms_requests']:
        for view in ['main', 'camera']:
            measurement = result[stage][view]
            assert measurement['ready'] and not measurement['error']
            assert measurement['unique_pose_interval_ms']['count'] > 10, (stage, view, measurement)
    with (directory/'export_check.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.reader(stream))
    assert len(rows) > 5000 and len(rows[0]) == 146
    assert all(len(row) == 146 for row in rows)
    evaluation = json.loads((directory/'export_check.json').read_text(encoding='utf-8'))
    assert evaluation['summary'] == result['scientific_summary']
    assert evaluation['display']['pose_interpolation_used_for_area'] is True
    assert evaluation['display']['render_frames_add_coverage'] is False
    assert evaluation['prediction']['method'] == 'dense-short-horizon-pd-v1'
    assert evaluation['prediction']['config']['visibility_s'] == .1
    assert evaluation['prediction']['metrics']['constraint_violating_steps'] == 0
    sections = result['algorithm_sections']
    assert [section['title'] for section in sections] == ['机器人运动', '磁场解算', '进镜摄像头', '退镜摄像头']
    assert all(section['characters'] > 400 and section['plain_formula'] for section in sections)
    hover = result['centerline_hover']
    assert hover['index'] >= 0 and hover['chart_linked']
    assert hover['cleared_when_hidden'] and hover['cleared_on_leave']
    assert '归一化风险' in hover['tooltip'] and '限速' in hover['readout']
    reference = json.loads((ROOT/'文档'/'八线圈仿真结果.json').read_text('utf-8'))['coverage']
    for key in ['insertion_covered_area_mm2','withdrawal_added_area_mm2','final_coverage_pct']:
        assert abs(result['scientific_summary'][key]-reference[key]) < 1e-8
    assert evaluation['magnetic_simulation']['coil_count'] == 8
    posture = evaluation['magnetic_simulation']['patient_posture']
    assert posture['posture'] == 'supine' and posture['gravity_m_s2'] == [0., 0., -9.81]
    assert posture['gravity_frame'] == 'fixed world -Z, not rotated with anatomy'
    assert result['patient_posture'] == posture
    display = result['posture_display']
    assert '仰卧' in display['window_title']
    assert display['grid_z_range'] == [0.,0.]
    assert display['gravity_direction'] == [0.,0.,-1.]
    assert display['geometry_span_mm'][1] > display['geometry_span_mm'][2]
    assert display['default_elevation_deg'] >= 40.
    direction = evaluation['magnetic_simulation']['route_direction']
    assert result['route_direction'] == direction and direction['verification']['passed']
    assert direction['reference_entry_original_mm'] == [.970171,-145.567,-506.25]
    assert direction['insertion_start_label'] == direction['withdrawal_end_label'] == '肛门入口'
    endpoints = result['route_endpoints']
    assert math.dist(endpoints['actual_start_mm'],direction['entry_position_mm']) < 1.
    assert math.dist(endpoints['actual_turn_mm'],direction['deep_position_mm']) < 1.
    assert math.dist(endpoints['actual_finish_mm'],direction['entry_position_mm']) < 1.
    assert '肛门入口 · 开始进镜' in endpoints['timeline_captions'][0]
    assert '肠道深部 · 开始退镜' in endpoints['timeline_captions'][1]
    assert '回到肛门入口' in endpoints['timeline_captions'][2]
    return result


def main():
    software = ROOT/'软件'
    files = [path for path in software.rglob('*') if path.is_file()]
    forbidden = [str(path) for path in files if path.suffix.lower() in {'.html', '.htm', '.js', '.qml'} or any(name in path.name.lower() for name in ['webengine', 'qtquick', 'qtqml', 'msedge', 'chrome'])]
    assert not forbidden, forbidden
    isolated = ROOT/'构建'/'独立验收'/'给朋友的程序'
    if isolated.exists():
        archive = isolated.with_name('给朋友的程序_' + str(time.time_ns()))
        isolated.rename(archive)
    shutil.copytree(software, isolated, dirs_exist_ok=True)
    report_dir = ROOT/'文档'/'验证'/'独立成品'
    report_dir.mkdir(parents=True, exist_ok=True)
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith(('PYTHON', 'QT_', 'QML', 'CONDA'))}
    windows = Path(environment.get('SystemRoot', r'C:\Windows'))
    environment['PATH'] = os.pathsep.join([str(windows/'System32'), str(windows)])
    executable = isolated/'Magnetic_Camera_8Coil.exe'
    with (report_dir/'process.log').open('w', encoding='utf-8') as output:
        process = subprocess.Popen([str(executable), '--smoke-report', str(report_dir/'smoke.json')], cwd=isolated.parent, env=environment, stdout=output, stderr=output)
        inspected = psutil.Process(process.pid)
        modules = set()
        children = set()
        deadline = time.monotonic()+90
        while process.poll() is None and time.monotonic() < deadline:
            try:
                modules.update(item.path for item in inspected.memory_maps())
                children.update(item.name() for item in inspected.children(recursive=True))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            time.sleep(.5)
        if process.poll() is None:
            process.kill()
            raise RuntimeError('Isolated native test exceeded 90 seconds; see process.log')
        assert process.returncode == 0, (process.returncode, 'see process.log')
    assert modules, 'No loaded module inspection available'
    external_anaconda = [path for path in modules if str(Path(r'D:\anaconda')).lower() in path.lower()]
    assert not external_anaconda, external_anaconda
    assert not children, children
    smoke = validate_smoke(report_dir)
    report = {
        'passed': True,
        'isolated_executable': str(executable),
        'cwd': str(isolated.parent),
        'path': environment['PATH'],
        'exit_code': process.returncode,
        'external_anaconda_modules': external_anaconda,
        'child_process_names': sorted(children),
        'loaded_module_paths': sorted(modules),
        'runtime_file_count': len(files),
        'runtime_bytes': sum(path.stat().st_size for path in files),
        'forbidden_runtime_files': forbidden,
        'exe_sha256': hashlib.sha256((software/'Magnetic_Camera_8Coil.exe').read_bytes()).hexdigest(),
        'patient_posture': smoke['patient_posture'],
        'posture_display': smoke['posture_display'],
        'route_direction': smoke['route_direction'],
        'route_endpoints': smoke['route_endpoints'],
        'csv_rows': sum(1 for line in (report_dir/'export_check.csv').open(encoding='utf-8-sig'))-1,
        'csv_columns': 146,
        'unique_pose_draw_fps': {key: {view: smoke[key][view]['unique_pose_draw_fps'] for view in ['main', 'camera']} for key in ['insertion_1x', 'withdrawal_1x', 'withdrawal_5x', 'slider_16ms_requests']},
        'limitation': 'Same computer, separate Unicode path and restricted environment; not a clean Windows VM or another physical computer.'
    }
    (report_dir/'delivery_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'loaded_module_paths'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
