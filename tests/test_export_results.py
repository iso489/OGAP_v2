"""Result export: CSV/LaTeX writers that reuse tables.py and a DataFrame bridge.

Covers the plain row writers, the MultiIndex-column flattening used by the
modality-robustness table, and the bundled modality-robustness export (table +
optional significance sweep). All file output goes to pytest tmp_path.
"""
import os

import numpy as np
import pandas as pd

from ogap.evaluation.export_results import (
    dataframe_to_rows,
    export_modality_robustness,
    export_results_csv,
    export_results_latex,
)
from ogap.evaluation.modality_robustness import (
    ALL_COMBINATIONS,
    combo_label,
    full_significance_sweep,
    modality_robustness_table,
)

METRICS = ("dsc_wt", "dsc_tc", "dsc_et", "hd95_wt", "hd95_tc", "hd95_et")


def _synth_results(models, n=3, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for m in models:
        out[m] = {}
        for combo in ALL_COMBINATIONS:
            lbl = combo_label(combo)
            out[m][lbl] = {
                f"case_{i:04d}": {k: float(rng.uniform(0.4, 0.9)) for k in METRICS}
                for i in range(n)
            }
    return out


def test_export_results_csv_writes_file(tmp_path):
    rows = [{"metric": "dsc_wt", "value": 0.9}, {"metric": "dsc_et", "value": 0.7}]
    path = str(tmp_path / "r.csv")
    text = export_results_csv(rows, path)
    assert os.path.exists(path)
    # newline="" so RFC-4180 \r\n endings are not translated on read-back.
    content = open(path, encoding="utf-8", newline="").read()
    assert "metric" in content and "dsc_wt" in content
    assert text == content


def test_export_results_latex_writes_booktabs(tmp_path):
    rows = [{"metric": "dsc_wt", "value": 0.9}]
    path = str(tmp_path / "r.tex")
    text = export_results_latex(rows, path, caption="Cap", label="tab:x")
    assert os.path.exists(path)
    assert "\\toprule" in text and "\\bottomrule" in text
    assert "\\caption{Cap}" in text and "tab:x" in text


def test_dataframe_to_rows_flattens_multiindex_with_index_label():
    cols = pd.MultiIndex.from_tuples([("dsc_et_mean", "teacher"),
                                      ("dsc_et_std", "teacher")])
    df = pd.DataFrame([[0.8, 0.05]], index=["T1+FLAIR"], columns=cols)
    rows = dataframe_to_rows(df, index_label="combo_label")
    assert len(rows) == 1
    r = rows[0]
    assert r["combo_label"] == "T1+FLAIR"
    assert r["dsc_et_mean_teacher"] == 0.8
    assert r["dsc_et_std_teacher"] == 0.05


def test_dataframe_to_rows_without_index_label_omits_index():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    rows = dataframe_to_rows(df)
    assert rows == [{"a": 1, "b": 3}, {"a": 2, "b": 4}]


def test_export_modality_robustness_writes_table(tmp_path):
    table = modality_robustness_table(_synth_results(["teacher", "student"]),
                                      metrics=("dsc_et",))
    paths = export_modality_robustness(table, out_dir=str(tmp_path), prefix="modrob")
    assert os.path.exists(paths["csv"]) and os.path.exists(paths["latex"])
    csv_text = open(paths["csv"], encoding="utf-8").read()
    assert "n_modalities" in csv_text
    assert "combo_label" in csv_text
    assert "T1+T1ce+T2+FLAIR" in csv_text                 # full-modality row present


def test_export_modality_robustness_includes_significance(tmp_path):
    results = _synth_results(["a", "b", "c"], n=4)
    table = modality_robustness_table(results, metrics=("dsc_et",))
    sweep = full_significance_sweep(results)
    paths = export_modality_robustness(table, significance=sweep,
                                       out_dir=str(tmp_path), prefix="modrob")
    assert os.path.exists(paths["significance_csv"])
    assert os.path.exists(paths["significance_latex"])
    sig_text = open(paths["significance_csv"], encoding="utf-8").read()
    assert "FRIEDMAN" in sig_text
