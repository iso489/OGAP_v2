"""Write evaluation results to CSV / LaTeX, reusing :mod:`ogap.evaluation.tables`.

Two layers:

* thin wrappers (:func:`export_results_csv`, :func:`export_results_latex`) over the
  row-dict writers in ``tables.py``;
* a DataFrame bridge (:func:`dataframe_to_rows`) that flattens MultiIndex columns
  - e.g. the ``(dsc_et_mean, teacher)`` columns of
  :func:`ogap.evaluation.modality_robustness.modality_robustness_table` - into
  flat ``dsc_et_mean_teacher`` keys, plus :func:`export_modality_robustness` which
  bundles the robustness table and (optionally) the significance sweep to files.

No new formatting logic lives here; the LaTeX/CSV rendering stays in ``tables.py``.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .tables import rows_to_csv, rows_to_latex


def _flatten_col(col: Any) -> str:
    """Join a (possibly MultiIndex) column key into a flat string."""
    if isinstance(col, tuple):
        return "_".join(str(c) for c in col if c != "" and c is not None)
    return str(col)


def _coerce(v: Any) -> Any:
    """Convert numpy scalars to native Python so downstream formatting is plain."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def dataframe_to_rows(df, index_label: Optional[str] = None) -> List[Dict[str, object]]:
    """Flatten a DataFrame to a list of row dicts.

    Args:
        df: a pandas DataFrame (MultiIndex columns are joined with ``_``).
        index_label: if given, each row includes the DataFrame index value under
            this key (first); if ``None`` the index is dropped.

    Returns:
        A list of flat dicts ready for :func:`export_results_csv` / ``_latex``.
    """
    flat_cols = [(_flatten_col(c), c) for c in df.columns]
    rows: List[Dict[str, object]] = []
    for idx, series in df.iterrows():
        row: Dict[str, object] = {}
        if index_label is not None:
            row[index_label] = _coerce(idx)
        for flat, orig in flat_cols:
            row[flat] = _coerce(series[orig])
        rows.append(row)
    return rows


def export_results_csv(rows: Sequence[Dict[str, object]], path: str) -> str:
    """Write ``rows`` to ``path`` as CSV and return the text."""
    return rows_to_csv(rows, path=path)


def export_results_latex(rows: Sequence[Dict[str, object]], path: str,
                         caption: str = "", label: str = "",
                         columns: Optional[Sequence[str]] = None,
                         bold_best: Optional[Dict[str, str]] = None) -> str:
    """Write ``rows`` to ``path`` as a booktabs LaTeX table and return the source."""
    return rows_to_latex(rows, columns=columns, bold_best=bold_best,
                         caption=caption, label=label, path=path)


def export_modality_robustness(
    robustness_table,
    significance=None,
    out_dir: str = ".",
    prefix: str = "modality_robustness",
    caption: str = "",
    label: str = "",
) -> Dict[str, str]:
    """Write the modality-robustness table (and optional significance sweep) to files.

    Args:
        robustness_table: DataFrame from
            :func:`ogap.evaluation.modality_robustness.modality_robustness_table`
            (combo label as index, MultiIndex metric columns + ``n_modalities``).
        significance: optional DataFrame from
            :func:`ogap.evaluation.modality_robustness.full_significance_sweep`.
        out_dir: directory to write into (created if missing).
        prefix: output filename stem.
        caption, label: optional LaTeX caption/label for the main table.

    Returns:
        ``{"csv", "latex"[, "significance_csv", "significance_latex"]}`` -> paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows = dataframe_to_rows(robustness_table, index_label="combo_label")
    csv_path = os.path.join(out_dir, f"{prefix}.csv")
    tex_path = os.path.join(out_dir, f"{prefix}.tex")
    export_results_csv(rows, csv_path)
    export_results_latex(rows, tex_path,
                         caption=caption or "Modality-robustness results.",
                         label=label or f"tab:{prefix}")
    paths = {"csv": csv_path, "latex": tex_path}

    if significance is not None:
        srows = dataframe_to_rows(significance)        # flat sweep: no index column
        s_csv = os.path.join(out_dir, f"{prefix}_significance.csv")
        s_tex = os.path.join(out_dir, f"{prefix}_significance.tex")
        export_results_csv(srows, s_csv)
        export_results_latex(srows, s_tex,
                             caption=caption or "Multi-model significance sweep.",
                             label=f"tab:{prefix}_significance")
        paths["significance_csv"] = s_csv
        paths["significance_latex"] = s_tex
    return paths
