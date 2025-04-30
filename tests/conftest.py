# -*- coding: utf-8 -*-
# tests/conftest.py
import sys
from pathlib import Path

# ensure project root is on pytestÕs PYTHONPATH
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
