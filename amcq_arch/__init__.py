"""Separate architecture experiments; import the original toolkit read-only."""
from pathlib import Path
import sys

toolkit = Path(__file__).resolve().parent.parent / "amcq"
if (toolkit / "amcq" / "__init__.py").is_file():
    sys.path.insert(0, str(toolkit))
