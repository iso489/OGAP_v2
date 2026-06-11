"""Export evaluation results as CSV and LaTeX (booktabs) tables.

Pure formatting utilities with no pandas dependency: inputs are lists of row
dicts (e.g. the output of :func:`ogap.evaluation.stats.paired_significance_table`)
and outputs are strings / files. The best value per numeric column can be bolded.
"""
from __future__ import annotations

import csv
import io
import math
from typing import Dict, Optional, Sequence


def _fmt(v) -> str:
    if isinstance(v, float):
        if math.isnan(v):
            return "NaN"
        # Small magnitudes (e.g. p-values like 1e-8) collapse to "0.0000" at fixed
        # 4 dp, which is not reportable in a journal table; use scientific notation
        # below 1e-3 and keep 4 dp for normal-range values (Dice, HD95, effect sizes).
        if v != 0.0 and abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.4f}"
    return str(v)


def rows_to_csv(rows: Sequence[Dict[str, object]], path: Optional[str] = None) -> str:
    """Serialise row dicts to CSV; write to ``path`` if given, return the text."""
    if not rows:
        return ""
    cols = list(rows[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols)
    w.writeheader()
    for r in rows:
        w.writerow({k: _fmt(r.get(k, "")) for k in cols})
    text = buf.getvalue()
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


def _latex_escape(s: str) -> str:
    s = str(s)
    for a, b in (("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        s = s.replace(a, b)
    return s


def rows_to_latex(rows: Sequence[Dict[str, object]],
                  columns: Optional[Sequence[str]] = None,
                  bold_best: Optional[Dict[str, str]] = None,
                  caption: str = "", label: str = "",
                  path: Optional[str] = None) -> str:
    """Render row dicts as a booktabs LaTeX table.

    Args:
        rows: list of row dicts.
        columns: column order (defaults to keys of the first row).
        bold_best: optional ``{column: "max"|"min"}`` to bold the best value.
        caption, label: LaTeX caption and label.
        path: if given, write the table there.

    Returns:
        The LaTeX source string.
    """
    if not rows:
        return ""
    cols = list(columns or rows[0].keys())
    best: Dict[str, float] = {}
    if bold_best:
        for col, mode in bold_best.items():
            vals = [r[col] for r in rows if isinstance(r.get(col), (int, float))
                    and not (isinstance(r.get(col), float) and math.isnan(r[col]))]
            if vals:
                best[col] = max(vals) if mode == "max" else min(vals)
    lines = ["\\begin{table}[t]", "\\centering",
             f"\\caption{{{caption}}}", f"\\label{{{label}}}" if label else "",
             "\\begin{tabular}{" + "l" * len(cols) + "}", "\\toprule",
             " & ".join(_latex_escape(c) for c in cols) + " \\\\", "\\midrule"]
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c, "")
            s = _fmt(v)
            if not isinstance(v, (int, float)):
                s = _latex_escape(s)   # escape string data cells (numbers are safe)
            if c in best and isinstance(v, (int, float)) and v == best[c]:
                s = f"\\textbf{{{s}}}"
            cells.append(s)
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    text = "\n".join(line for line in lines if line != "")
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
