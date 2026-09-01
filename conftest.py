"""
Pytest configuration for AdLens tests.

Ensures that tests can import from the project root without installing
the package, and sets sensible defaults.
"""

import sys
from pathlib import Path

# Add project root to sys.path so tests can import adlens modules
sys.path.insert(0, str(Path(__file__).parent))
