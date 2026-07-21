"""Make the project root importable (pipeline/, gui/) when pytest runs
from anywhere. The project has no installable package — imports are rooted
at the project directory, same as `python autodub.py` / `python -m gui`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
