# Contributing to NOVA

Thank you for your interest in contributing to NOVA. This guide will help
you get started with the development process.

---

## Getting Started

### Prerequisites

- Python 3.10 or later
- Git

### Development Setup

```bash
# Clone the repository
git clone https://github.com/dot-gabriel-ferrer/Nova.git
cd Nova

# Install in development mode with all optional dependencies
pip install -e "nova-py[all,dev]"

# Run the test suite
cd nova-py
python -m pytest tests/ -v
```

### Optional Dependencies

| Extra | Description |
|-------|-------------|
| `cloud` | Remote access via fsspec (S3, GCS, Azure) |
| `plots` | Matplotlib for benchmark plots |
| `analysis` | SciPy for advanced analysis functions |
| `ml` | PyTorch for tensor export |
| `notebooks` | Jupyter for interactive notebooks |
| `dev` | pytest and coverage tools |
| `all` | All of the above (except dev) |

---

## Code Style

### General Rules

- All files (`.py`, `.md`, `.json`) MUST use pure ASCII only.
  No emojis, no Unicode symbols, no box-drawing characters.
- Use `[done]` / `[todo]` instead of checkmarks.
- Use `--` instead of em-dash.
- Use `>>` instead of arrows.
- Follow PEP 8 for Python code.
- All public functions MUST have docstrings.
- All public functions MUST have type hints.
- Use named constants from `nova/constants.py` instead of magic numbers.

### Module Conventions

- Module-level docstrings are required.
- Import shared constants from `nova.constants`.
- Raise specific exception types (never bare `except:`).
- Keep imports alphabetically sorted within groups.

---

## Testing

### Running Tests

```bash
cd nova-py

# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=nova --cov-report=term-missing

# Run a specific test file
python -m pytest tests/test_container.py -v

# Run tests matching a pattern
python -m pytest tests/ -k "test_pipeline" -v
```

### Writing Tests

- Test files go in `nova-py/tests/` and must start with `test_`.
- Test functions must start with `test_`.
- Use temporary directories for any file I/O (pytest `tmp_path` fixture).
- Group related tests in classes (e.g. `TestPipeline`).
- Test both success and failure cases.
- Test edge cases (empty arrays, NaN values, etc.).

---

## Pull Request Process

1. Create a branch from `main` with a descriptive name.
2. Make your changes, following the code style guide.
3. Add or update tests for your changes.
4. Run the full test suite and confirm all tests pass.
5. Update documentation if your changes affect the public API.
6. Submit a pull request with a clear description of the changes.

---

## Areas for Contribution

### Code

- New features and optimizations
- Bug fixes
- Performance improvements (Cython/Numba acceleration)
- GPU support (CuPy backend)
- DASK integration for parallel processing

### Tests

- Edge cases and real-world validation
- Property-based tests (hypothesis)
- Integration tests with external libraries

### Documentation

- Tutorials and examples
- API reference improvements
- Migration guides for specific instruments

### Specification

- Review and improve the NOVA format specification
- Propose extensions for new use cases

### Adoption

- Try NOVA in your pipeline and report your experience
- Create example pipelines for specific instruments

---

## Reporting Issues

When reporting bugs, please include:

1. Python version and OS
2. NOVA version (`python -c "import nova; print(nova.__version__)"`)
3. Steps to reproduce the issue
4. Expected behavior vs actual behavior
5. Any error messages or tracebacks

---

## License

By contributing to NOVA, you agree that your contributions will be
licensed under the MIT License.
