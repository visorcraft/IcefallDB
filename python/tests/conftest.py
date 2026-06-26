# Ensure tests import the `icefalldb` package from the current repo checkout first,
# even when the virtual environment has an editable install pointing elsewhere
# (e.g., inside a git worktree).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
