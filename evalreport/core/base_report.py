from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

@dataclass
class BaseReport:
    """Base class for all task-specific reports.

    Handles the common API:
    - ``run_all()`` to compute metrics and generate plots
    - ``to_dict()`` to get a JSON-serializable summary
    - ``save()`` to persist HTML/Markdown/JSON/PDF outputs
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    plots: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)
    metric_descriptions: Dict[str, str] = field(default_factory=dict)
    # Base directory where report outputs (HTML/JSON/PDF) and plots live.
    # This is typically set by the entrypoint based on output_path and
    # defaults to "reports/" when not provided.
    output_dir: Optional[Path] = None

    def run_all(self) -> Dict[str, Any]:
        """Run metrics, visualization, and insights.

        Subclasses should override ``_compute_metrics``,
        ``_generate_plots``, and ``_generate_insights``.
        """
        self._compute_metrics()
        self._generate_plots()
        self._generate_insights()
        return self.to_dict()

    # Hooks for subclasses -------------------------------------------------
    def _compute_metrics(self) -> None:  # pragma: no cover - to be overridden
        ...

    def _generate_plots(self) -> None:  # pragma: no cover - to be overridden
        ...

    def _generate_insights(self) -> None:  # pragma: no cover - to be overridden
        ...

    # Public helpers -------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics": self.metrics,
            "insights": self.insights,
            "plots": self.plots,
        }

    def save(self, path: str, format: Optional[str] = None) -> None:
        """Save the report to disk.

        For v0.1 this implements a minimal HTML/JSON writer;
        richer templating can be added incrementally.
        """
        target = Path(path)
        fmt = (format or target.suffix.lstrip(".") or "html").lower()

        if fmt == "json":
            target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        elif fmt in {"html", "htm"}:
            html = self._to_simple_html()
            target.write_text(html, encoding="utf-8")
        elif fmt in {"md", "markdown"}:
            md = self._to_simple_markdown()
            target.write_text(md, encoding="utf-8")
        elif fmt == "pdf":
            self._to_simple_pdf(target)
        else:
            raise ValueError(f"Unsupported output format: {fmt!r}")

    # Simple renderers -----------------------------------------------------
    def _to_simple_html(self) -> str:
        # basic inline CSS for a clean, readable layout
        def metric_row(name: str, value: Any, desc: str) -> str:
            return (
                "<tr>"
                f"<td class='metric-name'><strong>{name}</strong><br/><span class='metric-desc'>{desc}</span></td>"
                f"<td class='metric-value'>{value}</td>"
                "</tr>"
            )

        rows = []
        for k, v in sorted(self.metrics.items()):
            desc = self.metric_descriptions.get(k, "")
            rows.append(metric_row(k, v, desc))
        rows_html = "\n".join(rows)

        insights_html = "".join(f"<li>{i}</li>" for i in self.insights)

        # plots: simple gallery
        plots_html_parts: List[str] = []
        for name, path in self.plots.items():
            plots_html_parts.append(
                f"<div class='plot-card'>"
                f"<h3>{name.replace('_', ' ').title()}</h3>"
                f"<img src='{path}' alt='{name}' />"
                f"</div>"
            )
        plots_html = "\n".join(plots_html_parts) or "<p>No plots available.</p>"

        return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Evaluation Report</title>
    <style>
      body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        max-width: 960px;
        margin: 0 auto;
        padding: 24px;
        background: #f7f8fa;
        color: #222;
      }}
      h1 {{
        margin-bottom: 0.5rem;
      }}
      .section {{
        background: #fff;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 24px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      }}
      table.metrics {{
        width: 100%;
        border-collapse: collapse;
      }}
      table.metrics th, table.metrics td {{
        border-bottom: 1px solid #e5e7eb;
        padding: 8px 6px;
        vertical-align: top;
      }}
      table.metrics th {{
        text-align: left;
        background: #f3f4f6;
      }}
      .metric-name {{
        width: 40%;
      }}
      .metric-desc {{
        display: block;
        font-size: 0.8rem;
        color: #6b7280;
        margin-top: 4px;
      }}
      .metric-value {{
        font-family: Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
      }}
      .plots-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
        gap: 16px;
      }}
      .plot-card img {{
        width: 100%;
        border-radius: 4px;
        border: 1px solid #e5e7eb;
      }}
    </style>
  </head>
  <body>
    <h1>Evaluation Report</h1>

    <div class="section">
      <h2>Metrics</h2>
      <table class="metrics">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>

    <div class="section">
      <h2>Insights</h2>
      <ul>
        {insights_html}
      </ul>
    </div>

    <div class="section">
      <h2>Plots</h2>
      <div class="plots-grid">
        {plots_html}
      </div>
    </div>
  </body>
</html>
""".strip()

    def _to_simple_markdown(self) -> str:
        lines = ["# Evaluation Report", "", "## Metrics", ""]
        for k, v in sorted(self.metrics.items()):
            lines.append(f"- **{k}**: {v}")
        lines.append("")
        lines.append("## Insights")
        lines.append("")
        for ins in self.insights:
            lines.append(f"- {ins}")
        return "\n".join(lines)

    def _to_simple_pdf(self, target: Path) -> None:
        """Write a simple PDF summary of metrics and insights.

        This intentionally focuses on textual content; plots are not
        embedded in v0.1.
        """
        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except Exception as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "PDF output requires the 'reportlab' package. "
                "Install with: pip install reportlab."
            ) from exc

        c = canvas.Canvas(str(target), pagesize=A4)
        width, height = A4

        text = c.beginText(40, height - 40)
        text.setFont("Helvetica-Bold", 16)
        text.textLine("Evaluation Report")
        text.moveCursor(0, 20)

        text.setFont("Helvetica-Bold", 12)
        text.textLine("Metrics")
        text.setFont("Helvetica", 10)

        for name, value in sorted(self.metrics.items()):
            desc = self.metric_descriptions.get(name, "")
            line = f"- {name}: {value}"
            text.textLine(line)
            if desc:
                text.textLine(f"    {desc}")

        text.moveCursor(0, 20)
        text.setFont("Helvetica-Bold", 12)
        text.textLine("Insights")
        text.setFont("Helvetica", 10)
        for ins in self.insights:
            text.textLine(f"- {ins}")

        c.drawText(text)
        c.showPage()
        c.save()

