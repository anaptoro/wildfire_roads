# run.py
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from wildfire_pl.main import cli  # type: ignore

if __name__ == "__main__":
    cli()
