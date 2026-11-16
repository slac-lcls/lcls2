"""Locate source-only campaign tools without depending on directory depth."""
from pathlib import Path


def find_repository(explicit=None):
    candidates = [Path(explicit).resolve()] if explicit else Path(__file__).resolve().parents
    for path in candidates:
        if (path / 'pyproject.toml').is_file() and (path / 'psana/psana/gpu').is_dir():
            return path
    raise ValueError('Source repository required; pass --repository explicitly')
