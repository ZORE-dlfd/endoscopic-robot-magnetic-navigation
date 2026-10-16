import csv
import hashlib
import json
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
    assert len(rows) == 3173 and len(rows[0]) == 146
    assert all(len(row) == 146 for row in rows)
    evaluation = json.loads((directory/'export_check.json').read_text(encoding='utf-8'))
    assert evaluation['summary'] == result['scientific_summary']
    assert evaluation['display']['pose_interpolation_used_for_area'] is False
    assert abs(result['scientific_summary']['insertion_covered_area_mm2']-177090.4177645445) < 1e-8
    assert abs(result['scientific_summary']['withdrawal_added_area_mm2']-16978.93053162098) < 1e-8
    assert abs(result['scientific_summary']['final_coverage_pct']-78.75632566461537) < 1e-10
    return result


def main():
    software = ROOT/'软件'
    files = [path for path in software.rglob('*') if path.is_file()]
    forbidden = [str(path) for path in files if path.suffix.lower() in {'.html', '.htm', '.js', '.qml'} or any(name in path.name.lower() for name in ['webengine', 'qtquick', 'qtqml', 'msedge', 'chrome'])]
    assert not forbidden, forbidden
    isolated = ROOT/'构建'/'独立验收'/'给朋友的程序'
    shutil.copytree(software, isolated, dirs_exist_ok=True)
    report_dir = ROOT/'文档'/'验证'/'独立成品'
    report_dir.mkdir(parents=True, exist_ok=True)
    environment = {key: value for key, value in os.environ.items() if not key.upper().startswith(('PYTHON', 'QT_', 'QML', 'CONDA'))}
    windows = Path(environment.get('SystemRoot', r'C:\Windows'))
    environment['PATH'] = os.pathsep.join([str(windows/'System32'), str(windows)])
    executable = isolated/'Magnetic_Camera.exe'
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
        'exe_sha256': hashlib.sha256((software/'Magnetic_Camera.exe').read_bytes()).hexdigest(),
        'csv_rows': 3172,
        'csv_columns': 146,
        'unique_pose_draw_fps': {key: {view: smoke[key][view]['unique_pose_draw_fps'] for view in ['main', 'camera']} for key in ['insertion_1x', 'withdrawal_1x', 'withdrawal_5x', 'slider_16ms_requests']},
        'limitation': 'Same computer, separate Unicode path and restricted environment; not a clean Windows VM or another physical computer.'
    }
    (report_dir/'delivery_check.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'loaded_module_paths'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
