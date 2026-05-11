# Testing Guide

## Test Location

Place test files in the `tests/` folder. Files must start with `test_`.

## Basic Test

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.classes.cable import Cable

def test_cable_length():
    cable = Cable(id=0, length=1, stiffness=100, damping=10)
    assert cable.length == 1  # Assert statement required for test to pass/fail

if __name__ == "__main__":
    import pytest
    pytest.main(["-v", __file__])
```

## Run Commands

**Run all tests:**
```bash
uv run pytest -v
```

**Run with coverage report:**
```bash
uv run pytest --cov=src --cov-report=term-missing
```

**Run single test file directly:**
```bash
uv run python tests/test_cable_tension.py
```

