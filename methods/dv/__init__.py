import sys
from pathlib import Path

# The vendored upstream imports its own modules by top level name, so its source
# root has to be importable before anything in this package is loaded.
_DV_SRC = Path(__file__).resolve().parent / "external" / "diff-vectors" / "src"
if str(_DV_SRC) not in sys.path:
    sys.path.insert(0, str(_DV_SRC))
