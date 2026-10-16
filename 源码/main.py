from pathlib import Path
import sys


if not getattr(sys, 'frozen', False):
    dependencies = Path(__file__).resolve().parents[1]/'构建'/'依赖'
    sys.path.insert(0, str(dependencies))

from magnetic_app.app import main


if __name__ == '__main__':
    raise SystemExit(main())
