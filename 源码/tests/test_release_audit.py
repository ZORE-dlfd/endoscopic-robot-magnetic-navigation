import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPECIFICATION = importlib.util.spec_from_file_location('release_audit', ROOT/'工具'/'audit_eight_coil.py')
AUDIT = importlib.util.module_from_spec(SPECIFICATION)
SPECIFICATION.loader.exec_module(AUDIT)


def test_reference_audit_uses_bundled_baseline_without_sibling_project():
    implementations = AUDIT.reference_implementations(ROOT)
    assert set(implementations) == {'search.py', 'planning.py', 'controller.py', 'config.py', 'coverage.py', 'visibility.py'}
    assert all(value['unchanged'] for value in implementations.values())


def test_reference_audit_detects_changed_source(tmp_path):
    import json

    reference_directory = tmp_path/'数据'/'参考原版'
    reference_directory.mkdir(parents=True)
    source_directory = tmp_path/'源码'/'retreat_planner'
    source_directory.mkdir(parents=True)
    baseline = {}
    for name in ['search.py', 'planning.py', 'controller.py', 'config.py', 'coverage.py', 'visibility.py']:
        source = source_directory/name
        source.write_text('original\n', encoding='utf-8')
        baseline[name] = AUDIT.digest(source)
    (reference_directory/'planner_sha256.json').write_text(json.dumps({'sha256': baseline}), encoding='utf-8')
    (source_directory/'search.py').write_text('changed\n', encoding='utf-8')
    implementations = AUDIT.reference_implementations(tmp_path)
    assert not implementations['search.py']['unchanged']
    assert all(value['unchanged'] for name, value in implementations.items() if name != 'search.py')
