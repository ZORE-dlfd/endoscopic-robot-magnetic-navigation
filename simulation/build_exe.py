from __future__ import annotations

import gzip
import hashlib
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def build():
    html = ROOT / 'Magnetic_Camera.html'
    content = html.read_bytes()
    sha256 = hashlib.sha256(content).hexdigest()
    resource = ROOT / 'Magnetic.Html.gz'
    resource.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
    (ROOT / 'PayloadInfo.cs').write_text(
        'using System.Reflection;\n'
        '[assembly: AssemblyTitle("Magnetic Capsule Camera Withdrawal Viewer")]\n'
        '[assembly: AssemblyDescription("Offline capsule camera and withdrawal coverage simulator")]\n'
        '[assembly: AssemblyVersion("1.0.0.7")]\n'
        'internal static class PayloadInfo { internal const string Sha256 = "' + sha256 + '"; }\n', encoding='utf-8')
    image = Image.new('RGBA', (256, 256), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((4, 4, 252, 252), 52, fill='#182c42')
    font = ImageFont.truetype(r'C:\Windows\Fonts\georgiab.ttf', 190)
    bounds = draw.textbbox((0, 0), 'M', font=font)
    draw.text(((256 - bounds[2]) / 2, (256 - (bounds[3] - bounds[1])) / 2 - bounds[1]), 'M', fill='white', font=font)
    image.save(ROOT / 'viewer.ico', sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    executable = ROOT / 'Magnetic_Camera.exe'
    compiler = Path(r'C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe')
    command = [str(compiler), '/nologo', '/target:winexe', '/platform:anycpu', '/optimize+', '/utf8output',
               '/out:' + str(executable), '/win32icon:' + str(ROOT / 'viewer.ico'),
               '/reference:System.Windows.Forms.dll', '/resource:' + str(resource) + ',Magnetic.Html.gz',
               str(ROOT / 'Launcher.cs'), str(ROOT / 'PayloadInfo.cs')]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    extraction = ROOT / 'exe_payload_check'
    verification = subprocess.run([str(executable), '--extract-only', str(extraction)], timeout=30)
    if verification.returncode:
        raise RuntimeError('EXE extraction failed')
    extracted = extraction / html.name
    embedded_hash = hashlib.sha256(extracted.read_bytes()).hexdigest()
    if embedded_hash != sha256:
        raise RuntimeError('Embedded HTML differs from delivered HTML')
    report = {'embedded_html_verified': True, 'html_sha256': sha256, 'embedded_sha256': embedded_hash,
              'exe_sha256': hashlib.sha256(executable.read_bytes()).hexdigest(),
              'exe_bytes': executable.stat().st_size, 'dependency': 'Windows .NET Framework and browser',
              'launcher_type': 'standalone embedded offline HTML launcher', 'signed': False}
    (ROOT / 'exe_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    build()
