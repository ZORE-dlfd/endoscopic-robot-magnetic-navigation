from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import shutil
import struct
import tarfile
from urllib.request import Request, urlopen
import zipfile
import zlib

import zstandard


ROOT = Path(__file__).resolve().parents[1]
ENVIRONMENT = Path(r'D:\anaconda')
DESTINATION = ROOT/'软件'/'第三方许可'
CACHE = ROOT/'构建'/'许可来源'


def read_range(url, selection):
    request = Request(url, headers={'Range': selection})
    with urlopen(request, timeout=30) as response:
        if response.status != 206:
            raise RuntimeError(f'Byte ranges unavailable: {url}')
        total = int(response.headers['Content-Range'].split('/')[-1])
        start = int(response.headers['Content-Range'].split()[1].split('-')[0])
        content = response.read(4*1024*1024+1)
        if len(content) > 4*1024*1024:
            raise RuntimeError(f'Unexpected metadata size: {url}')
        return content, start, total


def conda_license(record):
    identity = f"{record['name']}-{record['version']}-{record['build']}"
    cached = CACHE/(identity+'.info.tar.zst')
    if not cached.exists():
        tail, start, total = read_range(record['url'], 'bytes=-65536')
        with zipfile.ZipFile(io.BytesIO(tail)) as archive:
            member = next(item for item in archive.infolist() if item.filename.startswith('info-'))
            offset = start+member.header_offset
            section, _, _ = read_range(record['url'], f'bytes={offset}-{min(total-1, offset+member.compress_size+65535)}')
            header = struct.unpack('<4s5H3I2H', section[:30])
            data_start = 30+header[-2]+header[-1]
            compressed = section[data_start:data_start+member.compress_size]
            content = compressed if member.compress_type == zipfile.ZIP_STORED else zlib.decompress(compressed, -15)
            cached.write_bytes(content)
    directory = DESTINATION/record['name']
    extracted = []
    upstream = {}
    recipe = ''
    with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(cached.read_bytes())) as decoded:
        with tarfile.open(fileobj=decoded, mode='r|') as archive:
            for member in archive:
                normalized = member.name.replace('\\', '/')
                if member.isfile() and normalized == 'info/about.json':
                    about = json.load(archive.extractfile(member))
                    upstream = {key: about.get(key) for key in ['home', 'dev_url', 'license', 'license_file', 'license_url'] if about.get(key)}
                if member.isfile() and normalized == 'info/recipe/meta.yaml':
                    recipe = archive.extractfile(member).read().decode('utf-8')
                if not member.isfile() or not normalized.startswith('info/licenses/'):
                    continue
                relative = Path(normalized.removeprefix('info/licenses/'))
                if relative.is_absolute() or '..' in relative.parts:
                    raise RuntimeError(f'Unsafe license path: {normalized}')
                target = directory/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.extractfile(member).read())
                extracted.append(str(target.relative_to(DESTINATION)))
    source_urls = []
    for line in recipe.splitlines():
        stripped = line.strip().removeprefix('- ')
        if stripped.startswith('url: '):
            source_urls.append(stripped.removeprefix('url: '))
    return {
        'component': record['name'], 'version': record['version'],
        'license_metadata': record.get('license'), 'package_source': record['url'],
        'upstream': upstream, 'upstream_sources': source_urls,
        'license_files': extracted, 'notice': 'Original Conda metadata; an empty list means this package did not include license text in its info archive.',
    }


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    shipped = {path.name.lower() for path in (ROOT/'软件'/'_internal').rglob('*') if path.is_file()}
    records = []
    for path in (ENVIRONMENT/'conda-meta').glob('*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        if record['name'] in {'qt-main', 'qt-webengine'}:
            continue
        if any(Path(name).name.lower() in shipped and name.lower().endswith(('.dll', '.pyd')) for name in record.get('files', [])):
            records.append(record)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        pending = [(record, executor.submit(conda_license, record)) for record in records]
        for record, future in pending:
            try:
                result = future.result()
                results.append(result)
                print(result['component'], len(result['license_files']), flush=True)
            except Exception as error:
                failures.append({'component': record['name'], 'error': str(error)})
                print('FAILED', record['name'], str(error), flush=True)
    for name, source in {
        'Python-LICENSE.txt': ENVIRONMENT/'LICENSE_PYTHON.txt',
        'NumPy-LICENSE.txt': ENVIRONMENT/'Lib'/'site-packages'/'numpy-2.3.5.dist-info'/'LICENSE.txt',
        'PyInstaller-COPYING.txt': ROOT/'构建'/'依赖'/'pyinstaller-6.22.3.dist-info'/'licenses'/'COPYING.txt',
    }.items():
        if source.is_file():
            shutil.copy2(source, DESTINATION/name)
    with tarfile.open(CACHE/'pyopengl-3.1.10.tar.gz') as archive:
        (DESTINATION/'PyOpenGL-LICENSE.txt').write_bytes(archive.extractfile('pyopengl-3.1.10/license.txt').read())
    with zipfile.ZipFile(CACHE/'pyopengl-3.1.10-py3-none-any.whl') as archive:
        for name in archive.namelist():
            if 'COPYING' in name:
                (DESTINATION/('PyOpenGL-'+Path(name).name)).write_bytes(archive.read(name))
    (DESTINATION/'组件来源.json').write_text(json.dumps({'components': results, 'failures': failures}, ensure_ascii=False, indent=2), encoding='utf-8')
    if failures:
        raise RuntimeError(f'{len(failures)} license sources could not be collected')


if __name__ == '__main__':
    main()
