"""Evaluation: BraTS metrics, robust paired statistics, hardware profiling, tables."""
from __future__ import annotations

from .brats_metrics import compute_brats_metrics
from .stats import (
    noninferiority_table,
    paired_significance_table,
    wilcoxon_significance_table,
)
from .hardware_metrics import measure_hardware_metrics
from .tables import rows_to_csv, rows_to_latex
from .modality_robustness import (
    ALL_COMBINATIONS,
    apply_modality_mask,
    combo_label,
    evaluate_modality_combinations,
    friedman_test,
    full_significance_sweep,
    modality_degradation_analysis,
    modality_robustness_table,
    multimodel_significance_matrix,
    quantize_student_int8,
)
from .ablation import (
    COMPONENTS,
    ablation_matrix,
    ablation_results_table,
    component_states,
    load_ablation_configs,
)
from .baselines import (
    BASELINES,
    baseline_description,
    list_baselines,
    run_baseline,
)
from .export_results import (
    dataframe_to_rows,
    export_modality_robustness,
    export_results_csv,
    export_results_latex,
)

__all__ = [
    "compute_brats_metrics",
    "paired_significance_table",
    "wilcoxon_significance_table",
    "noninferiority_table",
    "measure_hardware_metrics",
    "rows_to_csv",
    "rows_to_latex",
    # modality-robustness suite
    "ALL_COMBINATIONS",
    "apply_modality_mask",
    "combo_label",
    "evaluate_modality_combinations",
    "friedman_test",
    "full_significance_sweep",
    "modality_degradation_analysis",
    "modality_robustness_table",
    "multimodel_significance_matrix",
    "quantize_student_int8",
    # ablation study
    "COMPONENTS",
    "ablation_matrix",
    "ablation_results_table",
    "component_states",
    "load_ablation_configs",
    # comparison baselines
    "BASELINES",
    "baseline_description",
    "list_baselines",
    "run_baseline",
    # result export
    "dataframe_to_rows",
    "export_modality_robustness",
    "export_results_csv",
    "export_results_latex",
]
