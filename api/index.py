import sys
from pathlib import Path

# Add repository root to Python path so 'src' is discoverable on Vercel
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.api import app

__all__ = ["app"]
