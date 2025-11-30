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
