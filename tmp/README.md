# Local smoke tests for `model-eval-toolkit`

This folder holds a small script that runs `generate_report()` once per supported task and writes HTML (and plot PNGs) under `tmp/reports/`.

## Run

From the **project root**:

```bash
python tmp/run_all_tasks.py
```

If you prefer an editable install:

```bash
pip install -e .
python tmp/run_all_tasks.py
```

The script adds the project root to `sys.path`, so it works without installing as long as you run it from the repo.

## Output

- `tmp/reports/*.html` — one report per task
- `tmp/reports/evalreport_plots/*.png` — figures referenced by the HTML

`tmp/reports/` is gitignored so generated files are not committed.
