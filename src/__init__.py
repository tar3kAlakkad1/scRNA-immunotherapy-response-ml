"""
PRECISE Pipeline Implementation
===============================

This package implements the core PRECISE framework from Pinhasi & Yizhak (2025)
for predicting immune checkpoint inhibitor (ICI) response using single-cell RNA-seq data.

Modules:
    - data_loading: Load raw GEO files and build AnnData objects
    - preprocessing: Filter genes/cells and normalize expression
    - labels: Patient ID mapping and response labels
    - model: XGBoost training with leave-one-patient-out CV
    - feature_selection: Boruta and importance-based feature selection
    - evaluation: Metrics, ROC curves, and visualizations
    - utils: Shared helpers (logging, seeds, etc.)
"""

__version__ = "0.1.0"

# Data loading exports
from .data_loading import (
    load_expression_matrix,
    load_patient_mapping,
    build_anndata,
    load_melanoma_data,
    DEFAULT_EXPR_PATH,
    DEFAULT_PATIENT_PATH,
)

# Preprocessing exports
from .preprocessing import (
    filter_genes,
    filter_cells,
    compute_qc_metrics,
    normalize_expression,
    add_response_labels as add_response_labels_preprocessing,  # Alias to avoid conflict
    run_preprocessing_pipeline,
    save_preprocessed_data,
    load_preprocessed_data,
    DEFAULT_OUTPUT_PATH,
)

# Labels exports (Task 2.1)
from .labels import (
    get_response_labels,
    add_response_labels,
    get_patient_metadata,
    get_response_distribution,
    validate_labels,
    RESPONSE_FULL_TO_SHORT,
    RESPONSE_SHORT_TO_FULL,
    RESPONSE_FULL_TO_BINARY,
    RESPONSE_BINARY_TO_FULL,
)

# Model exports (Task 3.1)
from .model import (
    train_xgboost,
    predict_cells,
    aggregate_to_patient,
    leave_one_patient_out_cv,
    get_patient_true_labels,
    compute_feature_importances,
    evaluate_signature_genes,
    DEFAULT_XGBOOST_PARAMS,
    PRECISE_11_GENE_SIGNATURE,
)

# Evaluation exports (Task 3.2)
from .evaluation import (
    compute_roc_auc,
    plot_roc_curve,
    plot_patient_predictions,
    generate_results_table,
    generate_patient_results_table,
    evaluate_model_results,
)

# Feature selection exports (Task 4.1)
from .feature_selection import (
    run_boruta_selection,
    importance_based_selection,
    get_feature_importance_df,
    compute_importances_from_loocv,
    run_loocv_with_selected_genes,
    run_nested_loocv_with_selection,  # Correct approach - no data leakage
    save_selected_genes,
    plot_feature_importance,
    run_feature_selection_pipeline,
)

# Signature analysis exports (Task 5)
from .signature_analysis import (
    check_signature_genes_presence,
    get_signature_gene_ranks,
    plot_signature_gene_expression,
    plot_signature_importance_comparison,
    plot_model_comparison,
    plot_combined_roc_curves,
    run_signature_analysis,
    generate_final_comparison_table,
)

# Stretch goals exports (Phase 7)
from .stretch_goals import (
    # Goal A: Cell Filtration
    compute_cell_predictivity_scores,
    run_cell_filtration_experiment,
    plot_cell_filtration_results,
    # Goal B: External Validation
    BCC_PATIENT_RESPONSES,
    load_bcc_data,
    compute_signature_score,
    run_external_validation,
    plot_external_validation,
    # Goal C: SHAP
    compute_shap_values,
    plot_shap_analysis,
    plot_shap_beeswarm,
    # Main runner
    run_all_stretch_goals,
)
