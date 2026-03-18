"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_reports_cwd(tmp_path, monkeypatch):
    """Run code with cwd = tmp_path so default reports/ lands in tmp, not repo root."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
