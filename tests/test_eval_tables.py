import math

from ogap.evaluation.tables import rows_to_csv, rows_to_latex
from ogap.evaluation.hardware_metrics import measure_hardware_metrics
from ogap.models import build_student


def _rows():
    return [{"metric": "dsc_wt", "value": 0.90, "stars": "***"},
            {"metric": "dsc_et", "value": 0.65, "stars": "*"}]


def test_csv_headers():
    txt = rows_to_csv(_rows())
    assert "metric,value,stars" in txt.splitlines()[0] and "dsc_wt" in txt


def test_latex_booktabs_and_bold():
    tex = rows_to_latex(_rows(), bold_best={"value": "max"}, caption="Test", label="tab:t")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "\\midrule" in tex
    assert "\\textbf{0.9000}" in tex and "dsc\\_wt" in tex


def test_empty_rows_safe():
    assert rows_to_csv([]) == "" and rows_to_latex([]) == ""


def test_hardware_metrics_fields():
    m = build_student(in_channels=4, num_classes=4, base=8)
    out = measure_hardware_metrics(m, input_shape=(1, 4, 16, 16, 16),
                                   device="cpu", n_warmup=1, n_runs=2)
    for k in ("params_M", "int8_size_mb", "latency_mean_ms", "latency_std_ms",
              "peak_mem_mb", "flops_G"):
        assert k in out
    assert out["params_M"] > 0 and out["int8_size_mb"] > 0
    assert math.isnan(out["peak_mem_mb"])
