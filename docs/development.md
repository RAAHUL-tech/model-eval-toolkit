# Development & contributing

## Clone and editable install

```bash
git clone https://github.com/RAAHUL-tech/model-eval-toolkit.git
cd model-eval-toolkit
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[test,pdf]"
pytest -q
```

### Coverage (optional)

```bash
pytest --cov=evalreport --cov-report=term-missing
```

## Build and sanity-check the package

```bash
pip install build twine
python -m build
twine check dist/*
```

## CI and PyPI

Workflow: `.github/workflows/ci.yml`

- **Pull requests** — tests on Python 3.9, 3.10, 3.11.
- **Push to `main`** — tests, then a **`publish`** job targets the **`pypi`** [environment](https://docs.github.com/en/actions/deployment/targeting-different-environments/using-environments-for-deployment). If that environment has **required reviewers**, the workflow waits for approval before uploading to PyPI.

**Secrets:** add **`PYPI_API_TOKEN`** as an **environment secret** on `pypi` (recommended) or as a **repository** secret — both are visible to the job once it enters that environment.

**Releases:** bump `version` in `pyproject.toml` before each upload (PyPI rejects duplicate versions).

Trusted Publishing (OIDC) is optional; the workflow requests `id-token: write` for compatibility.

## Documentation

- User docs live in **`docs/`** (this folder).
- The **README** on GitHub/PyPI summarizes the project and links here for depth.

## Contributing

Issues and PRs are welcome: [GitHub Issues](https://github.com/RAAHUL-tech/model-eval-toolkit/issues).

- Add or update **tests** under `tests/` for behavior changes.
- Keep **task names** and `generate_report` routing in sync with docs.
- Follow existing style (type hints, small focused modules).

## License

See [LICENSE](../LICENSE) in the repository root.
