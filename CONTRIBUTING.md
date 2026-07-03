# Contributing to Omage

Thank you for your interest in contributing! 🎉

## How to Contribute

### Report a Bug
Open an issue with:
- Python version
- Omage version
- Minimal code to reproduce
- Expected vs actual behavior

### Suggest a Feature
Open an issue describing:
- The problem you want to solve
- Your proposed solution
- Why it would benefit others

### Submit a Pull Request

1. Fork the repository
2. Create your branch: `git checkout -b feature/amazing-feature`
3. Write tests for your changes
4. Make sure all tests pass: `pytest tests/ -v`
5. Commit: `git commit -m "Add amazing feature"`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

## Development Setup

```bash
git clone https://github.com/your-username/omage.git
cd omage
pip install -e ".[dev]"
pytest tests/ -v
```

## Code Style

- Use `black` for formatting: `black omage/`
- Use `ruff` for linting: `ruff check omage/`
- Add type hints to all functions
- Write docstrings for public APIs

## Questions?

Open an issue or start a discussion on GitHub.
