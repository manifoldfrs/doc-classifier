"""tests/conftest.py
###############################################################################
Pytest *global* configuration – adds the repository **root directory** to
``sys.path`` so that the internal ``src`` package can be imported reliably
regardless of where pytest is invoked from.
###############################################################################
Why is this necessary?
---------------------
Running ``pytest`` from sub-directories or certain IDEs may result in Python's
import system not being able to resolve the top-level *src* package, leading to
errors such as::

    ModuleNotFoundError: No module named 'src'

Unit-tests already passed due to incidental path layouts, but the new
integration suite (located deeper in the hierarchy) exposed the issue.  The
snippet below ensures **deterministic** behaviour across all environments
without polluting runtime code.
"""

from __future__ import annotations

# stdlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure repository root is first on sys.path
# ---------------------------------------------------------------------------
_repo_root: Path = Path(__file__).resolve().parent.parent  # tests/ -> repo root
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
