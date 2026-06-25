# -*- coding: utf-8 -*-
# tests/conftest.py
import os
import sys
from pathlib import Path

# ensure project root is on pytest's PYTHONPATH
root = Path(__file__).resolve().parent.parent
src_root = root / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

# Unit tests import and call run_pipeline.main() directly from arbitrary
# interpreters. Skip strict runtime env checks in test context.
os.environ.setdefault("IMPACT_SKIP_ENV_CHECK", "1")
