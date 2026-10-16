import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT/'构建'/'依赖'
environment = dict(os.environ)
environment['PYTHONPATH'] = str(DEPENDENCIES)
environment['PYTHONIOENCODING'] = 'utf-8'
interpreter = Path(sys.executable).parent
library_bin = interpreter/'Library'/'bin'
environment['PATH'] = os.pathsep.join([str(library_bin), str(interpreter/'DLLs'), str(interpreter), environment.get('PATH', '')])
command = [sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--windowed', '--onedir', '--name', 'Magnetic_Camera',
    '--distpath', str(ROOT/'构建'/'dist'), '--workpath', str(ROOT/'构建'/'work'), '--specpath', str(ROOT/'构建'),
    '--paths', str(ROOT/'源码'), '--paths', str(DEPENDENCIES), '--icon', str(ROOT/'数据'/'clon001'/'app.ico'),
    '--add-data', str(ROOT/'数据'/'clon001') + ';assets', '--hidden-import', 'OpenGL.platform.win32',
    '--hidden-import', 'OpenGL.arrays.numpymodule']
for filename in ['pyside6.cp313-win_amd64.dll', 'shiboken6.cp313-win_amd64.dll']:
    binary = library_bin/filename
    if binary.exists():
        command += ['--add-binary', str(binary)+';.']
for name in ['PyQt5', 'PyQt6', 'tkinter', 'matplotlib', 'scipy', 'pandas', 'IPython', 'pytest', 'torch', 'tensorflow', 'cv2', 'PySide6.QtWebEngineCore', 'PySide6.QtWebEngineWidgets', 'PySide6.QtQml', 'PySide6.QtQuick']:
    command += ['--exclude-module', name]
command.append(str(ROOT/'源码'/'main.py'))
subprocess.run(command, env=environment, cwd=ROOT, check=True)
print('Build ready:', ROOT/'构建'/'dist'/'Magnetic_Camera')
