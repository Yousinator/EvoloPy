# Contributing to EvoloPy

Thank you for your interest in contributing to EvoloPy! This document provides guidelines and instructions for contributing.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for everyone.

## How to Contribute

### Reporting Bugs

- Use the bug report template when creating a new issue
- Include detailed steps to reproduce the bug
- Include your environment details
- If possible, include a minimal code example that reproduces the issue

### Suggesting Features

- Use the feature request template when creating a new issue
- Clearly describe the feature and its benefits
- If possible, provide implementation suggestions

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature/fix
3. Make your changes
4. Add tests if applicable
5. Update documentation if needed
6. Submit a pull request

### Development Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/EvoloPy.git
cd EvoloPy
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to all public functions and classes
- Keep functions focused and small
- Write unit tests for new features

### Testing

Run tests using:
```bash
pytest
```

### Documentation

- Update README.md if needed
- Add docstrings to new functions and classes
- Update examples if you change functionality

## Questions?

Feel free to open an issue for any questions about contributing.