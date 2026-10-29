import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from html.parser import HTMLParser

import numpy as np


class FormulaText(HTMLParser):
    def __init__(self):
        super().__init__()
        self.inside = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag == 'dialog' and dict(attrs).get('id') == 'docsDialog':
            self.inside = True
        if self.inside and tag in ['h1', 'h2', 'h3', 'p', 'li', 'tr', 'br', 'pre']:
            self.parts.append('\n')

    def handle_endtag(self, tag):
        if tag == 'dialog':
            self.inside = False
        if self.inside and tag in ['p', 'li', 'tr', 'pre']:
            self.parts.append('\n')

    def handle_data(self, data):
        if self.inside:
            self.parts.append(data)


def migrate(source, destination):
    content = source.read_bytes()
    payload = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', content.decode('utf-8'), re.S)[1])
    arrays = {}

    def encode(value, path='payload'):
        if isinstance(value, dict):
            return {key: encode(item, path + '/' + key) for key, item in value.items()}
        if isinstance(value, list):
            try:
                array = np.asarray(value)
                if array.dtype.kind in 'fibu' and array.size > 8:
                    arrays[path] = array
                    return {'__array__': path}
            except (ValueError, TypeError):
                pass
            return [encode(item, path + '/' + str(index)) for index, item in enumerate(value)]
        return value

    destination.mkdir(parents=True, exist_ok=True)
    schema = encode(payload)
    (destination / 'metadata.json').write_text(json.dumps(schema, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
    np.savez_compressed(destination / 'arrays.npz', **arrays)
    shutil.copyfile(source.parent / 'viewer.ico', destination / 'app.ico')
    formula = FormulaText()
    formula.feed(content.decode('utf-8'))
    (destination / 'formulas.txt').write_text(''.join(formula.parts).strip(), encoding='utf-8')
    provenance = {
        'example_display_name': 'clon001', 'original_model_name': 'Colon001',
        'source_html_sha256': hashlib.sha256(content).hexdigest(),
        'source_file': str(source), 'migration': 'Lossless numeric array extraction; original simulation not recomputed.',
        'files': {name: hashlib.sha256((destination / name).read_bytes()).hexdigest() for name in ['metadata.json', 'arrays.npz', 'formulas.txt']},
    }
    (destination / 'provenance.json').write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(provenance, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    arguments = parser.parse_args()
    migrate(arguments.source, arguments.destination)
