from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BaseReport:
    """Base class for all task-specific reports.

    Handles the common API:
    - ``run_all()`` to compute metrics and generate plots
    - ``to_dict()`` to get a JSON-serializable summary
    - ``save()`` to persist HTML/Markdown/JSON outputs
    """

    metrics: Dict[str, Any] = field(default_factory=dict)
    plots: Dict[str, Any] = field(default_factory=dict)
    insights: List[str] = field(default_factory=list)

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
            "plots": list(self.plots.keys()),
        }

    def save(self, path: str, format: Optional[str] = None) -> None:
        """Save the report to disk.

        For v0.1 this implements a minimal HTML/JSON writer;
        richer templating can be added incrementally.
        """
        target = Path(path)
        fmt = (format or target.suffix.lstrip(".") or "html").lower()

        if fmt == "json":
            import json

            target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        elif fmt in {"html", "htm"}:
            html = self._to_simple_html()
            target.write_text(html, encoding="utf-8")
        elif fmt in {"md", "markdown"}:
            md = self._to_simple_markdown()
            target.write_text(md, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported output format: {fmt!r}")

    # Simple renderers -----------------------------------------------------
    def _to_simple_html(self) -> str:
        rows = "\n".join(
            f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in sorted(self.metrics.items())
        )
        insights_html = "".join(f"<li>{i}</li>" for i in self.insights)
        return f"""
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Evaluation Report</title>
  </head>
  <body>
    <h1>Evaluation Report</h1>
    <h2>Metrics</h2>
    <table border="1" cellspacing="0" cellpadding="4">
      <thead><tr><th>Metric</th><th>Value</th></tr></thead>
      <tbody>
        {rows}
      </tbody>
    </table>
    <h2>Insights</h2>
    <ul>
      {insights_html}
    </ul>
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

