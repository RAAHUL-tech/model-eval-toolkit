from __future__ import annotations

import html as html_module
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


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
            "metric_descriptions": self.metric_descriptions,
        }

    def save(self, path: str, format: Optional[str] = None) -> None:
        """Save the report to disk.

        For v0.1 this implements a minimal HTML/JSON/Markdown/PDF writer;
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

    @staticmethod
    def _format_metric_value(v: Any) -> str:
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, indent=2, default=str)
            except TypeError:
                return str(v)
        return str(v)

    def _to_simple_html(self) -> str:
        """Order: Metrics → Plots → Insights → Metric reference (descriptions)."""

        def esc(s: Any) -> str:
            return html_module.escape(self._format_metric_value(s), quote=True)

        # 1) Metrics: name + value only (clean table)
        metric_rows = []
        for k, v in sorted(self.metrics.items()):
            metric_rows.append(
                "<tr>"
                f"<td class='metric-key'><strong>{html_module.escape(str(k))}</strong></td>"
                f"<td class='metric-value'><pre class='metric-pre'>{esc(v)}</pre></td>"
                "</tr>"
            )
        rows_html = "\n".join(metric_rows) or "<tr><td colspan='2'>No metrics.</td></tr>"

        # 2) Plots
        plots_html_parts: List[str] = []
        for name, path in self.plots.items():
            title = html_module.escape(name.replace("_", " ").title())
            path_esc = html_module.escape(str(path), quote=True)
            plots_html_parts.append(
                f"<div class='plot-card'>"
                f"<h3 class='plot-title'>{title}</h3>"
                f"<img src='{path_esc}' alt='{title}' loading='lazy' />"
                f"<p class='plot-path'><code>{path_esc}</code></p>"
                f"</div>"
            )
        plots_html = "\n".join(plots_html_parts) or "<p class='muted'>No plots generated.</p>"

        # 3) Insights
        insights_items = "".join(f"<li class='insight-item'>{html_module.escape(ins)}</li>" for ins in self.insights)
        insights_block = (
            f"<ul class='insights-list'>{insights_items}</ul>"
            if insights_items
            else "<p class='muted'>No automated insights.</p>"
        )

        # 4) Metric reference (descriptions at end)
        desc_rows = []
        for name in sorted(self.metrics.keys()):
            desc = self.metric_descriptions.get(name, "")
            desc_text = desc if desc else "—"
            desc_rows.append(
                "<tr>"
                f"<td class='ref-metric'>{html_module.escape(str(name))}</td>"
                f"<td class='ref-desc'>{html_module.escape(desc_text)}</td>"
                "</tr>"
            )
        ref_html = "\n".join(desc_rows) or "<tr><td colspan='2'>No metrics to describe.</td></tr>"

        return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Evaluation Report</title>
    <style>
      :root {{
        --bg: #f0f4f8;
        --card: #ffffff;
        --border: #e2e8f0;
        --text: #1e293b;
        --muted: #64748b;
        --accent: #0f766e;
        --accent-soft: #ccfbf1;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        max-width: 1000px;
        margin: 0 auto;
        padding: 28px 20px 48px;
        background: var(--bg);
        color: var(--text);
        line-height: 1.5;
      }}
      h1 {{
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 8px;
        letter-spacing: -0.02em;
      }}
      .subtitle {{
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 28px;
      }}
      .section {{
        background: var(--card);
        border-radius: 12px;
        padding: 20px 22px;
        margin-bottom: 20px;
        border: 1px solid var(--border);
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
      }}
      .section h2 {{
        font-size: 1.1rem;
        margin: 0 0 14px;
        padding-bottom: 10px;
        border-bottom: 2px solid var(--accent-soft);
        color: var(--accent);
        font-weight: 600;
      }}
      .section-num {{
        display: inline-block;
        width: 1.5rem;
        height: 1.5rem;
        line-height: 1.5rem;
        text-align: center;
        border-radius: 6px;
        background: var(--accent);
        color: #fff;
        font-size: 0.75rem;
        margin-right: 8px;
        vertical-align: middle;
      }}
      table.data {{
        width: 100%;
        border-collapse: collapse;
        font-size: 0.9rem;
      }}
      table.data th, table.data td {{
        border-bottom: 1px solid var(--border);
        padding: 10px 12px;
        text-align: left;
        vertical-align: top;
      }}
      table.data thead th {{
        background: #f8fafc;
        font-weight: 600;
        color: var(--muted);
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
      }}
      .metric-key {{ width: 32%; font-weight: 500; }}
      .metric-value {{ font-family: ui-monospace, Menlo, Monaco, Consolas, monospace; font-size: 0.85rem; }}
      pre.metric-pre {{
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        font-family: inherit;
        font-size: inherit;
      }}
      .ref-metric {{ width: 28%; font-weight: 500; color: #334155; }}
      .ref-desc {{ color: #475569; font-size: 0.9rem; }}
      .plots-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
        gap: 18px;
      }}
      .plot-card {{
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px;
        background: #fafbfc;
      }}
      .plot-title {{ font-size: 0.95rem; margin: 0 0 10px; color: #334155; }}
      .plot-card img {{
        width: 100%;
        border-radius: 8px;
        border: 1px solid var(--border);
        display: block;
      }}
      .plot-path {{ margin: 8px 0 0; font-size: 0.75rem; color: var(--muted); }}
      .plot-path code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
      .insights-list {{
        margin: 0;
        padding-left: 1.25rem;
      }}
      .insight-item {{ margin-bottom: 8px; padding-left: 4px; }}
      .muted {{ color: var(--muted); font-style: italic; margin: 0; }}
    </style>
  </head>
  <body>
    <h1>Evaluation Report</h1>
    <p class="subtitle">Metrics, visualizations, insights, and a reference for what each metric means.</p>

    <div class="section">
      <h2><span class="section-num">1</span>Metrics</h2>
      <table class="data">
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>

    <div class="section">
      <h2><span class="section-num">2</span>Plots</h2>
      <div class="plots-grid">{plots_html}</div>
    </div>

    <div class="section">
      <h2><span class="section-num">3</span>Insights</h2>
      {insights_block}
    </div>

    <div class="section">
      <h2><span class="section-num">4</span>Metric reference</h2>
      <p class="muted" style="margin-top:0;margin-bottom:12px;">What each reported metric indicates (same order as the metrics table).</p>
      <table class="data">
        <thead><tr><th>Metric</th><th>Description</th></tr></thead>
        <tbody>{ref_html}</tbody>
      </table>
    </div>
  </body>
</html>"""

    def _to_simple_markdown(self) -> str:
        """Order: Metrics → Plots → Insights → Metric reference."""

        lines: List[str] = [
            "# Evaluation Report",
            "",
            "Structured as: **Metrics** → **Plots** → **Insights** → **Metric reference**.",
            "",
            "---",
            "",
            "## 1. Metrics",
            "",
        ]
        for k, v in sorted(self.metrics.items()):
            val = self._format_metric_value(v)
            if "\n" in val:
                lines.append(f"### `{k}`")
                lines.append("")
                lines.append("```")
                lines.append(val)
                lines.append("```")
                lines.append("")
            else:
                lines.append(f"- **`{k}`**: {val}")
        if not self.metrics:
            lines.append("_No metrics._")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 2. Plots")
        lines.append("")
        if self.plots:
            for name, p in sorted(self.plots.items()):
                title = name.replace("_", " ").title()
                lines.append(f"### {title}")
                lines.append("")
                lines.append(f"![{title}]({p})")
                lines.append("")
                lines.append(f"*Path:* `{p}`")
                lines.append("")
        else:
            lines.append("_No plots._")
        lines.append("---")
        lines.append("")
        lines.append("## 3. Insights")
        lines.append("")
        if self.insights:
            for ins in self.insights:
                lines.append(f"- {ins}")
        else:
            lines.append("_No automated insights._")
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("## 4. Metric reference")
        lines.append("")
        lines.append("What each metric means (aligned with section 1).")
        lines.append("")
        for name in sorted(self.metrics.keys()):
            desc = self.metric_descriptions.get(name, "").strip()
            lines.append(f"- **`{name}`**")
            lines.append(f"  - {desc if desc else '_No description._'}")
        if not self.metrics:
            lines.append("_No metrics._")
        lines.append("")
        return "\n".join(lines)

    def _to_simple_pdf(self, target: Path) -> None:
        """PDF: Metrics → Plot paths → Insights → Metric reference. Paginates when needed."""

        try:
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
        except Exception as exc:  # pragma: no cover - dependency issue
            raise RuntimeError(
                "PDF output requires the 'reportlab' package. "
                "Install with: pip install reportlab."
            ) from exc

        width, height = A4
        margin_x = 48
        margin_top = height - 52
        line_h = 12
        section_gap = 18
        bottom_y = 52

        c = canvas.Canvas(str(target), pagesize=A4)

        y = margin_top

        def ensure_space(need: int) -> None:
            nonlocal y, c
            if y - need < bottom_y:
                c.showPage()
                y = margin_top

        def draw_heading(title: str, size: int = 14) -> None:
            nonlocal y
            ensure_space(section_gap + line_h + 4)
            c.setFont("Helvetica-Bold", size)
            c.drawString(margin_x, y, title)
            y -= size + 6

        def draw_body_line(text: str, indent: int = 0) -> None:
            nonlocal y
            ensure_space(line_h)
            c.setFont("Helvetica", 9)
            x = margin_x + indent
            # simple wrap for long lines
            max_w = width - margin_x * 2 - indent
            words = text.split()
            line = ""
            for w in words:
                test = (line + " " + w).strip()
                if c.stringWidth(test, "Helvetica", 9) <= max_w:
                    line = test
                else:
                    if line:
                        c.drawString(x, y, line[:500])
                        y -= line_h
                        ensure_space(line_h)
                    line = w
            if line:
                c.drawString(x, y, line[:500])
                y -= line_h

        # Title
        c.setFont("Helvetica-Bold", 18)
        c.drawString(margin_x, y, "Evaluation Report")
        y -= 22
        c.setFont("Helvetica-Oblique", 9)
        c.setFillColorRGB(0.35, 0.35, 0.35)
        c.drawString(margin_x, y, "Order: Metrics | Plots (paths) | Insights | Metric reference")
        c.setFillColorRGB(0, 0, 0)
        y -= section_gap

        # 1 Metrics
        draw_heading("1. Metrics", 12)
        for name, value in sorted(self.metrics.items()):
            val_str = self._format_metric_value(value)
            draw_body_line(f"{name}: {val_str[:200]}{'...' if len(val_str) > 200 else ''}", 0)
            if len(val_str) > 200:
                # continuation chunks
                rest = val_str[200:]
                while rest:
                    chunk = rest[:120]
                    rest = rest[120:]
                    draw_body_line(chunk, 12)
        if not self.metrics:
            draw_body_line("(No metrics.)")
        y -= section_gap // 2

        # 2 Plots (paths only)
        draw_heading("2. Plots (file paths)", 12)
        if self.plots:
            for name, p in sorted(self.plots.items()):
                draw_body_line(f"{name}: {p}", 0)
        else:
            draw_body_line("(No plots.)")
        y -= section_gap // 2

        # 3 Insights
        draw_heading("3. Insights", 12)
        if self.insights:
            for ins in self.insights:
                draw_body_line(f"- {ins}", 0)
        else:
            draw_body_line("(No insights.)")
        y -= section_gap // 2

        # 4 Metric reference
        draw_heading("4. Metric reference", 12)
        for name in sorted(self.metrics.keys()):
            desc = self.metric_descriptions.get(name, "").strip() or "(No description.)"
            draw_body_line(f"{name}", 0)
            draw_body_line(desc, 10)
            y -= 4
        if not self.metrics:
            draw_body_line("(No metrics.)")

        c.save()
