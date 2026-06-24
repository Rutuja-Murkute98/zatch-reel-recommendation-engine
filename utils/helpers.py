from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def existing_path(*parts: str) -> Path:
    path = project_root().joinpath(*parts)
    if not path.exists():
        raise FileNotFoundError(path)
    return path
