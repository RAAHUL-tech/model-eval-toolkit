# Output formats

Reports are saved via `generate_report` (automatically) or `report.save(path, format=...)`.

## HTML (`html`, `.html`)

- Styled layout: **metrics** (with short descriptions), **insights**, and **embedded PNG plots**.
- Best default for sharing with stakeholders.

## JSON (`json`, `.json`)

- Serializes `to_dict()`: `metrics`, `insights`, `plots` (paths to generated PNG files), `metric_descriptions`.
- Use for downstream tooling, dashboards, or CI artifacts.

## Markdown (`markdown`, `md`, `.md`)

- Textual metrics and insights.
- **No embedded images** — plot paths may still appear in structured data if you use JSON, or regenerate HTML for figures.

## PDF (`pdf`, `.pdf`)

- Text summary: metrics, descriptions, insights.
- Requires **ReportLab**: `pip install "model-eval-toolkit[pdf]"` or `pip install reportlab`.
- Does not embed plot images in the same way as HTML (implementation is text-focused).

## Choosing a format

```python
generate_report(..., output_path="out/report.json", format="json")
# or
generate_report(..., format="json")  # → reports/<task>_report.json
```

If `output_path` has a suffix, that suffix can imply the format when you call `save` on a report instance.

## Plot directory

All raster plots are written under **`<parent_of_report>/evalreport_plots/`**.  
HTML embeds them relative to the report file location, so keep that folder next to the HTML when moving files.
