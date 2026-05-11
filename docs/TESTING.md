# Testing Guide

This guide covers how to write, organize, and run tests using `uv` and `pytest`.

## Test Folder Structure

Tests are organized in the `tests/` directory at the project root:

```
dynamic-simulation/
├── src/
│   ├── __init__.py
│   ├── core.py
│   └── classes/
│       ├── cable.py
│       ├── drone.py
│       └── payload.py
├── tests/
│   ├── __init__.py
│   ├── test_cable_tension.py
│   ├── test_core.py
├── pyproject.toml
└── README.md
```


## Test Style & Conventions

### Basic Structure

```python
# Package imports
# (example)
from src.utils.default_params import DEFAULT_PARAMS
from src.classes.cable import Cable

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_function_name(test_name, param1, param2, expected):
    """Clear docstring explaining what is tested."""
    # Setup
    obj = Cable(...)
    
    # Action
    result = obj.force_vectors()
    
    # Assert (must use assert for it to be a test)
    assert np.allclose(result, expected), f"[{test_name}] Expected {expected}, got {result}"

# Add this to be able to run just the singular file instead of all tests
if __name__ == "__main__":
    pytest.main(["-v", __file__])
```

### Best Practices

- **One responsibility per test**: Each test should verify one behavior
- **Clear names**: Use descriptive test names like `test_cable_stretched`, not `test_1`
- **Docstrings**: Include docstrings explaining the test's purpose

## Running Tests

### Run a Single Test File

```bash
uv run pytest tests/test_cable_tension.py -v
```

Or, if you add this to the bottom of your test file, just run the file in vscode:

```python
if __name__ == "__main__":
    pytest.main(["-v", __file__])
```

### Run All Tests

```bash
uv run pytest -v
```

## Code Coverage

### Terminal Report (No Files Saved)

Shows coverage percentage for each module:

```bash
uv run pytest --cov=src --cov-report=term-missing
```

Example Output

```
Name                              Stmts   Miss  Cover   Missing
---------------------------------------------------------------
src/__init__.py                       0      0   100%
src/classes/cable.py                45      3    93%    34-36, 50
src/classes/drone.py                20      0   100%
---------------------------------------------------------------
TOTAL                               65      3    95%
```
