"""
Shared pytest configuration.
Adds the project root to sys.path so `import app` always works
regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
