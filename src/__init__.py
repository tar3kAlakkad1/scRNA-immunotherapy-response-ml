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

