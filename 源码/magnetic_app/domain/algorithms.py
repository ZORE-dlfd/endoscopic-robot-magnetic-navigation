import json

from .replay import assets_path


def algorithm_sections(path=None):
    directory = path if path is not None else assets_path()
    return json.loads((directory / 'algorithms.json').read_text(encoding='utf-8'))
