"""项目路径引导：从子目录运行脚本时，确保项目根目录在 sys.path 中。"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
for sub in ("", "experiments", "plotting", "paper", "tests"):
    path = PROJECT_ROOT / sub if sub else PROJECT_ROOT
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
