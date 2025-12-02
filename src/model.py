"""
Model Module
============

Implements the core XGBoost training and Leave-One-Patient-Out Cross-Validation (LOO-CV)
logic for predicting immunotherapy response from single-cell RNA-seq data.

This module implements the PRECISE pipeline's core machine learning approach:
    - Train XGBoost classifiers on per-cell gene expression
    - Each cell is labeled with its patient's response (Responder/Non-responder)
    - Use LOO-CV where all cells from one patient are held out as test
    - Aggregate per-cell predictions to patient-level scores (mean probability)
    - Evaluate performance with ROC AUC at the patient level

Key Design Decisions:
    - Uses XGBClassifier (sklearn API) for easier hyperparameter management
    - LOO-CV is at the patient level, not cell level (prevents data leakage)
    - Patient-level score = mean of per-cell probabilities for "responder" class
    - Supports custom hyperparameters while providing sensible defaults

Expected Performance (from PRECISE paper):
    - Base XGBoost model: AUC ≈ 0.84 on GSE120575 melanoma cohort
    - With Boruta feature selection: AUC ≈ 0.89
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import xgboost as xgb


# Default XGBoost hyperparameters based on typical scRNA-seq settings
# and the PRECISE paper's description (relatively shallow trees, moderate learning rate)
DEFAULT_XGBOOST_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "max_depth": 4,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,  # Suppress XGBoost warnings during training
}


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: Optional[Dict[str, Any]] = None,
) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier on cell-level data.

    Parameters
    ----------
    X_train : np.ndarray
        Training expression matrix of shape (n_cells, n_genes).
        Each row is a cell, each column is a gene.
    y_train : np.ndarray
        Training labels of shape (n_cells,).
        Binary labels: 1 = Responder, 0 = Non-responder.
    params : dict, optional
        XGBoost hyperparameters. If None, uses DEFAULT_XGBOOST_PARAMS.
        Custom params are merged with defaults (custom values override).

    Returns
    -------
    xgb.XGBClassifier
        Trained XGBoost classifier model.

    Examples
    --------
    >>> X_train = adata[train_mask, :].X
    >>> y_train = adata[train_mask, :].obs['response_binary'].values
    >>> model = train_xgboost(X_train, y_train)
    >>> model.predict_proba(X_test)[:, 1]  # Get responder probabilities
    """
    # Merge custom params with defaults
    model_params = DEFAULT_XGBOOST_PARAMS.copy()
    if params is not None:
        model_params.update(params)

    # Create and train the classifier
    model = xgb.XGBClassifier(**model_params)
    model.fit(X_train, y_train)

    return model


def predict_cells(
    model: xgb.XGBClassifier,
    X_test: np.ndarray,
) -> np.ndarray:
    """
    Predict per-cell probabilities of being 'responder'.

    Parameters
    ----------
    model : xgb.XGBClassifier
        Trained XGBoost classifier.
    X_test : np.ndarray
        Test expression matrix of shape (n_cells, n_genes).

    Returns
    -------
    np.ndarray
        Array of shape (n_cells,) with probability of responder class (class 1).
        Values are in range [0, 1].

    Examples
    --------
    >>> probs = predict_cells(model, X_test)
    >>> probs.mean()  # Average probability across cells
    0.42
    """
    # Get probabilities for the positive class (responder = 1)
    probs = model.predict_proba(X_test)[:, 1]
    return probs


def aggregate_to_patient(
    cell_probs: np.ndarray,
    cell_patient_ids: np.ndarray,
) -> Dict[str, float]:
    """
    Aggregate cell-level predictions to patient-level scores.

    For each patient, computes the mean probability across all their cells.
    This is the PRECISE paper's approach: a patient's response score is
    the average predicted responder probability of their cells.

    Parameters
    ----------
    cell_probs : np.ndarray
        Array of shape (n_cells,) with per-cell responder probabilities.
    cell_patient_ids : np.ndarray
        Array of shape (n_cells,) with patient ID for each cell.

    Returns
    -------
    dict
        Mapping of {patient_id: mean_probability} for each unique patient.

    Examples
    --------
    >>> cell_probs = np.array([0.3, 0.5, 0.7, 0.4])
    >>> cell_patient_ids = np.array(['P1', 'P1', 'P2', 'P2'])
    >>> aggregate_to_patient(cell_probs, cell_patient_ids)
    {'P1': 0.4, 'P2': 0.55}
    """
    patient_scores = {}

    # Group cells by patient and compute mean probability
    for patient_id in np.unique(cell_patient_ids):
        mask = cell_patient_ids == patient_id
        patient_scores[patient_id] = float(cell_probs[mask].mean())

    return patient_scores


def get_patient_true_labels(
    adata: ad.AnnData,
) -> Dict[str, int]:
    """
    Extract the true response label for each patient.

    Parameters
    ----------
    adata : AnnData
        AnnData object with 'patient_id' and 'response_binary' in .obs

    Returns
    -------
    dict
        Mapping of {patient_id: true_label} where true_label is 0 or 1.
    """
    patient_labels = {}
    for patient_id in adata.obs["patient_id"].unique():
        mask = adata.obs["patient_id"] == patient_id
        # All cells from same patient should have same label
        label = adata.obs.loc[mask, "response_binary"].iloc[0]
        patient_labels[patient_id] = int(label)
    return patient_labels


def leave_one_patient_out_cv(
    adata: ad.AnnData,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    return_models: bool = False,
) -> Dict[str, Any]:
    """
    Run Leave-One-Patient-Out Cross-Validation (LOO-CV).

    For each of the 48 patients:
        1. Hold out all cells from that patient as test set
        2. Train XGBoost on cells from all other patients
        3. Predict responder probability for each held-out cell
        4. Aggregate to patient-level score (mean probability)
        5. Compare to true patient label

    Final evaluation: ROC AUC across all 48 patient-level predictions.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with:
            - X: Expression matrix (cells x genes)
            - obs['patient_id']: Patient identifier for each cell
            - obs['response_binary']: Binary response label (0/1)
    params : dict, optional
        XGBoost hyperparameters. If None, uses DEFAULT_XGBOOST_PARAMS.
    verbose : bool, default=True
        Whether to print progress information.
    return_models : bool, default=False
        Whether to return the trained models from each fold.

    Returns
    -------
    dict
        Results dictionary containing:
            - patient_scores: {patient_id: predicted_score}
            - patient_labels: {patient_id: true_label}
            - auc: ROC AUC score at patient level
            - fpr: False positive rates for ROC curve
            - tpr: True positive rates for ROC curve
            - thresholds: Thresholds for ROC curve
            - fold_models: List of trained models (if return_models=True)
            - runtime_seconds: Total runtime in seconds
            - n_patients: Number of patients in CV
            - n_cells: Total number of cells
            - n_genes: Number of genes used

    Examples
    --------
    >>> results = leave_one_patient_out_cv(adata)
    >>> print(f"LOO-CV AUC: {results['auc']:.3f}")
    LOO-CV AUC: 0.842
    """
    start_time = time.time()

    # Validate required columns
    required_cols = ["patient_id", "response_binary"]
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    # Get unique patients
    patients = adata.obs["patient_id"].unique()
    n_patients = len(patients)

    if verbose:
        print("=" * 60)
        print("LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION")
        print("=" * 60)
        print(f"\nDataset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"Patients: {n_patients}")
        print(f"XGBoost params: {params if params else 'defaults'}")
        print("\nRunning LOO-CV...")

    # Initialize result containers
    patient_scores = {}
    patient_labels = {}
    fold_models = [] if return_models else None

    # Get expression matrix (ensure dense for XGBoost)
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Get labels and patient IDs
    y = adata.obs["response_binary"].values
    patient_ids = adata.obs["patient_id"].values

    # Run LOO-CV
    for fold_idx, test_patient in enumerate(patients):
        # Create train/test masks
        test_mask = patient_ids == test_patient
        train_mask = ~test_mask

        # Split data
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Train model
        model = train_xgboost(X_train, y_train, params)

        # Predict on held-out cells
        cell_probs = predict_cells(model, X_test)

        # Aggregate to patient-level score
        test_patient_score = float(cell_probs.mean())
        patient_scores[test_patient] = test_patient_score

        # Get true label for this patient
        true_label = int(y[test_mask][0])  # All cells have same label
        patient_labels[test_patient] = true_label

        if return_models:
            fold_models.append(model)

        # Progress update
        if verbose and (fold_idx + 1) % 10 == 0:
            print(f"  Completed fold {fold_idx + 1}/{n_patients}")

    # Compute patient-level ROC AUC
    y_true = np.array([patient_labels[p] for p in patients])
    y_scores = np.array([patient_scores[p] for p in patients])

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Compute runtime
    runtime = time.time() - start_time

    if verbose:
        print(f"\n  Completed all {n_patients} folds in {runtime:.1f} seconds")
        print(f"\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"\nPatient-level ROC AUC: {auc:.3f}")
        print(f"Score range: [{min(y_scores):.3f}, {max(y_scores):.3f}]")
        print(
            f"Label distribution: {sum(y_true)} responders, "
            f"{len(y_true) - sum(y_true)} non-responders"
        )

    # Build results dictionary
    results = {
        "patient_scores": patient_scores,
        "patient_labels": patient_labels,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "runtime_seconds": runtime,
        "n_patients": n_patients,
        "n_cells": adata.n_obs,
        "n_genes": adata.n_vars,
    }

    if return_models:
        results["fold_models"] = fold_models

    return results


def compute_feature_importances(
    adata: ad.AnnData,
    params: Optional[Dict[str, Any]] = None,
    importance_type: str = "gain",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compute average feature importances across LOO-CV folds.

    Trains a model on each LOO fold and extracts feature importances,
    then averages across all folds to get robust importance estimates.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object.
    params : dict, optional
        XGBoost hyperparameters.
    importance_type : str, default="gain"
        Type of feature importance:
            - "gain": Average gain of splits using the feature
            - "weight": Number of times feature is used in splits
            - "cover": Average coverage of splits using the feature
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - gene: Gene name
            - importance: Average importance across folds
            - importance_std: Standard deviation across folds
        Sorted by importance (descending).

    Examples
    --------
    >>> importance_df = compute_feature_importances(adata)
    >>> importance_df.head(10)  # Top 10 most important genes
    """
    if verbose:
        print("Computing feature importances across LOO-CV folds...")

    # Get unique patients
    patients = adata.obs["patient_id"].unique()
    n_patients = len(patients)

    # Get expression matrix and metadata
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs["response_binary"].values
    patient_ids = adata.obs["patient_id"].values
    gene_names = adata.var_names.tolist()

    # Collect importances from each fold
    all_importances = []

    for fold_idx, test_patient in enumerate(patients):
        # Create train mask
        train_mask = patient_ids != test_patient

        # Split data
        X_train = X[train_mask]
        y_train = y[train_mask]

        # Train model
        model = train_xgboost(X_train, y_train, params)

        # Get feature importances
        importances = model.get_booster().get_score(importance_type=importance_type)

        # Convert to array (XGBoost uses f0, f1, ... as feature names)
        fold_importance = np.zeros(len(gene_names))
        for fname, score in importances.items():
            feature_idx = int(fname[1:])  # Remove 'f' prefix
            fold_importance[feature_idx] = score

        all_importances.append(fold_importance)

        if verbose and (fold_idx + 1) % 10 == 0:
            print(f"  Processed fold {fold_idx + 1}/{n_patients}")

    # Average importances across folds
    all_importances = np.array(all_importances)
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    # Create DataFrame sorted by importance
    importance_df = (
        pd.DataFrame(
            {
                "gene": gene_names,
                "importance": mean_importance,
                "importance_std": std_importance,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\nTop 10 most important genes:")
        print(importance_df.head(10).to_string(index=False))

    return importance_df


def evaluate_signature_genes(
    adata: ad.AnnData,
    signature_genes: List[str],
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Evaluate model performance using only a subset of signature genes.

    This is useful for testing the 11-gene signature from the PRECISE paper.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object.
    signature_genes : list of str
        List of gene names to use for the model.
    params : dict, optional
        XGBoost hyperparameters.
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    dict
        Same format as leave_one_patient_out_cv results, plus:
            - signature_genes: List of genes used
            - genes_found: List of signature genes present in data
            - genes_missing: List of signature genes not found in data

    Examples
    --------
    >>> SIGNATURE_11 = ['GAPDH', 'CD38', 'CCR7', 'HLA-DRB5', 'STAT1',
    ...                 'GZMH', 'LGALS1', 'IFI6', 'EPSTI1', 'HLA-G', 'GBP5']
    >>> results = evaluate_signature_genes(adata, SIGNATURE_11)
    >>> print(f"11-gene signature AUC: {results['auc']:.3f}")
    """
    # Check which signature genes are present
    genes_found = [g for g in signature_genes if g in adata.var_names]
    genes_missing = [g for g in signature_genes if g not in adata.var_names]

    if verbose:
        print(f"\nSignature genes found: {len(genes_found)}/{len(signature_genes)}")
        if genes_missing:
            print(f"Missing genes: {genes_missing}")

    if not genes_found:
        raise ValueError("No signature genes found in the dataset")

    # Subset to signature genes
    adata_subset = adata[:, genes_found].copy()

    if verbose:
        print(f"Running LOO-CV with {len(genes_found)} signature genes...")

    # Run LOO-CV on subset
    results = leave_one_patient_out_cv(
        adata_subset,
        params=params,
        verbose=verbose,
    )

    # Add signature gene info
    results["signature_genes"] = signature_genes
    results["genes_found"] = genes_found
    results["genes_missing"] = genes_missing

    return results


# The 11-gene signature from the PRECISE paper
PRECISE_11_GENE_SIGNATURE = [
    "GAPDH",
    "CD38",
    "CCR7",
    "HLA-DRB5",
    "STAT1",
    "GZMH",
    "LGALS1",
    "IFI6",
    "EPSTI1",
    "HLA-G",
    "GBP5",
]


# ============================================================================
# Module-level script for testing
# ============================================================================

if __name__ == "__main__":
    """
    Run LOO-CV when module is executed directly.

    Usage:
        python -m src.model

    Or from project root:
        python src/model.py
    """
    import sys

    # Import sibling modules
    from preprocessing import load_preprocessed_data, DEFAULT_OUTPUT_PATH

    project_root = Path(__file__).parent.parent

    print("Loading preprocessed data...")
    print("-" * 60)
    input_path = project_root / DEFAULT_OUTPUT_PATH
    adata = load_preprocessed_data(input_path)

    # Run LOO-CV with default parameters
    print("\n")
    results = leave_one_patient_out_cv(adata, verbose=True)

    # Acceptance criteria check
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA CHECK")
    print("=" * 60)

    # Check 1: LOO-CV completes for all 48 patients
    n_patients = results["n_patients"]
    patients_ok = n_patients == 48
    print(f"\n1. LOO-CV completed for all patients:")
    print(f"   Patients processed: {n_patients}")
    print(f"   Expected: 48")
    print(f"   Status: {'PASS' if patients_ok else 'FAIL'}")

    # Check 2: Runtime is reasonable (<15 min = 900 seconds)
    runtime = results["runtime_seconds"]
    runtime_ok = runtime < 900
    print(f"\n2. Runtime is reasonable:")
    print(f"   Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")
    print(f"   Expected: <15 minutes (900 seconds)")
    print(f"   Status: {'PASS' if runtime_ok else 'FAIL'}")

    # Check 3: Patient-level scores are between 0 and 1
    scores = list(results["patient_scores"].values())
    min_score, max_score = min(scores), max(scores)
    scores_ok = min_score >= 0 and max_score <= 1
    print(f"\n3. Patient-level scores in valid range:")
    print(f"   Score range: [{min_score:.3f}, {max_score:.3f}]")
    print(f"   Expected: [0.0, 1.0]")
    print(f"   Status: {'PASS' if scores_ok else 'FAIL'}")

    # Additional: Report AUC for comparison with paper
    auc = results["auc"]
    print(f"\n4. Model performance (informational):")
    print(f"   ROC AUC: {auc:.3f}")
    print(f"   Paper reports: ~0.84 for base model")
    print(f"   Difference: {abs(auc - 0.84):.3f}")

    # Overall status
    all_pass = patients_ok and runtime_ok and scores_ok
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)
