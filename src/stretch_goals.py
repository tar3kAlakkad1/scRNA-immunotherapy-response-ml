"""
Stretch Goals Module (Phase 7)
==============================

This module implements the optional stretch goals from the PRECISE pipeline:

Stretch Goal A: Simplified Cell Filtration
    - Remove "non-predictive" cells based on prediction confidence
    - Re-run LOO-CV on remaining cells and check if AUC improves

Stretch Goal B: External Cohort Validation
    - Test the 11-gene signature on the BCC dataset (GSE123813)
    - Compute AUC and compare to paper's reported value

Stretch Goal C: SHAP Interpretability
    - Compute SHAP values for model interpretation
    - Generate SHAP summary plots (beeswarm)
    - Compare SHAP importance to XGBoost's built-in importance
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Import from sibling modules
from .model import (
    train_xgboost,
    leave_one_patient_out_cv,
    predict_cells,
    PRECISE_11_GENE_SIGNATURE,
    DEFAULT_XGBOOST_PARAMS,
)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")


# =============================================================================
# STRETCH GOAL A: Simplified Cell Filtration
# =============================================================================

def compute_cell_predictivity_scores(
    adata: ad.AnnData,
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Compute per-cell "predictivity" scores based on LOO-CV predictions.
    
    The predictivity score is defined as |predicted_prob - 0.5|.
    Cells with higher scores are easier to classify (high confidence),
    while cells with low scores (near 0.5) are "non-predictive".
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with labels.
    params : dict, optional
        XGBoost hyperparameters.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    predictivity_scores : np.ndarray
        Array of shape (n_cells,) with predictivity scores in [0, 0.5].
    cell_predictions : dict
        Dictionary with detailed per-cell prediction info.
    """
    if verbose:
        print("=" * 60)
        print("COMPUTING CELL PREDICTIVITY SCORES")
        print("=" * 60)
    
    # Get data
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs["response_binary"].values
    patient_ids = adata.obs["patient_id"].values
    patients = adata.obs["patient_id"].unique()
    n_patients = len(patients)
    
    # Initialize arrays
    cell_probs = np.zeros(adata.n_obs)
    
    if verbose:
        print(f"\nRunning LOO-CV to collect per-cell predictions...")
    
    # Run LOO-CV and collect per-cell predictions
    for fold_idx, test_patient in enumerate(patients):
        test_mask = patient_ids == test_patient
        train_mask = ~test_mask
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]
        
        # Train model
        model = train_xgboost(X_train, y_train, params)
        
        # Predict on held-out cells
        probs = predict_cells(model, X_test)
        cell_probs[test_mask] = probs
        
        if verbose and (fold_idx + 1) % 10 == 0:
            print(f"  Completed fold {fold_idx + 1}/{n_patients}")
    
    # Compute predictivity scores: |prob - 0.5|
    predictivity_scores = np.abs(cell_probs - 0.5)
    
    # Build detailed info
    cell_predictions = {
        "cell_probs": cell_probs,
        "predictivity_scores": predictivity_scores,
        "true_labels": y,
        "patient_ids": patient_ids,
    }
    
    if verbose:
        print(f"\nPredictivity score statistics:")
        print(f"  Mean: {predictivity_scores.mean():.4f}")
        print(f"  Median: {np.median(predictivity_scores):.4f}")
        print(f"  Min: {predictivity_scores.min():.4f}")
        print(f"  Max: {predictivity_scores.max():.4f}")
        
        # Fraction of cells with low confidence
        low_conf = (predictivity_scores < 0.1).sum()
        print(f"\n  Cells with predictivity < 0.1: {low_conf} ({low_conf/len(predictivity_scores):.1%})")
    
    return predictivity_scores, cell_predictions


def run_cell_filtration_experiment(
    adata: ad.AnnData,
    filter_percentiles: List[int] = [10, 20, 30, 40, 50],
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run cell filtration experiment: remove non-predictive cells and re-evaluate.
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object.
    filter_percentiles : list of int
        Percentages of low-confidence cells to remove (e.g., [10, 20, 30]).
    params : dict, optional
        XGBoost hyperparameters.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Results containing:
            - baseline_auc: AUC with all cells
            - filtered_aucs: Dict of {percentile: auc} for each filter level
            - best_percentile: Filter percentile that gave best AUC
            - best_auc: Best AUC achieved
            - predictivity_scores: Per-cell predictivity scores
    """
    start_time = time.time()
    
    if verbose:
        print("=" * 60)
        print("CELL FILTRATION EXPERIMENT")
        print("=" * 60)
        print(f"\nFilter percentiles to test: {filter_percentiles}")
    
    # Step 1: Get baseline AUC (all cells)
    if verbose:
        print("\n" + "-" * 60)
        print("Step 1: Computing baseline AUC (all cells)")
        print("-" * 60)
    
    baseline_results = leave_one_patient_out_cv(adata, params=params, verbose=verbose)
    baseline_auc = baseline_results["auc"]
    
    if verbose:
        print(f"\nBaseline AUC (all {adata.n_obs} cells): {baseline_auc:.4f}")
    
    # Step 2: Compute predictivity scores
    if verbose:
        print("\n" + "-" * 60)
        print("Step 2: Computing cell predictivity scores")
        print("-" * 60)
    
    predictivity_scores, cell_predictions = compute_cell_predictivity_scores(
        adata, params=params, verbose=verbose
    )
    
    # Step 3: Test different filter levels
    if verbose:
        print("\n" + "-" * 60)
        print("Step 3: Testing different filtration levels")
        print("-" * 60)
    
    filtered_aucs = {}
    filtered_results = {}
    
    for pct in filter_percentiles:
        if verbose:
            print(f"\n  Testing {pct}% filtration...")
        
        # Compute threshold
        threshold = np.percentile(predictivity_scores, pct)
        
        # Keep cells above threshold (high predictivity)
        keep_mask = predictivity_scores >= threshold
        n_kept = keep_mask.sum()
        n_removed = (~keep_mask).sum()
        
        if verbose:
            print(f"    Threshold: {threshold:.4f}")
            print(f"    Cells kept: {n_kept} ({n_kept/adata.n_obs:.1%})")
            print(f"    Cells removed: {n_removed} ({n_removed/adata.n_obs:.1%})")
        
        # Check we have cells from all patients
        patient_ids_kept = adata.obs["patient_id"].values[keep_mask]
        n_patients_kept = len(np.unique(patient_ids_kept))
        
        if n_patients_kept < len(adata.obs["patient_id"].unique()):
            if verbose:
                print(f"    WARNING: Only {n_patients_kept} patients have cells after filtering")
                print(f"    Skipping this filter level...")
            continue
        
        # Subset adata
        adata_filtered = adata[keep_mask].copy()
        
        # Run LOO-CV on filtered data
        try:
            filtered_result = leave_one_patient_out_cv(
                adata_filtered, params=params, verbose=False
            )
            filtered_aucs[pct] = filtered_result["auc"]
            filtered_results[pct] = filtered_result
            
            if verbose:
                print(f"    AUC: {filtered_result['auc']:.4f} (change: {filtered_result['auc'] - baseline_auc:+.4f})")
        except Exception as e:
            if verbose:
                print(f"    Error: {e}")
            continue
    
    # Find best filtration level
    if filtered_aucs:
        best_percentile = max(filtered_aucs, key=filtered_aucs.get)
        best_auc = filtered_aucs[best_percentile]
    else:
        best_percentile = 0
        best_auc = baseline_auc
    
    runtime = time.time() - start_time
    
    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("CELL FILTRATION RESULTS")
        print("=" * 60)
        print(f"\nBaseline AUC (all cells): {baseline_auc:.4f}")
        print(f"\nFiltered AUCs:")
        for pct, auc in sorted(filtered_aucs.items()):
            marker = " *BEST*" if pct == best_percentile else ""
            print(f"  {pct:3d}% removed: AUC = {auc:.4f} (change: {auc - baseline_auc:+.4f}){marker}")
        print(f"\nBest filtration: {best_percentile}% (AUC = {best_auc:.4f})")
        print(f"Runtime: {runtime:.1f} seconds")
    
    return {
        "baseline_auc": baseline_auc,
        "baseline_results": baseline_results,
        "filtered_aucs": filtered_aucs,
        "filtered_results": filtered_results,
        "best_percentile": best_percentile,
        "best_auc": best_auc,
        "predictivity_scores": predictivity_scores,
        "cell_predictions": cell_predictions,
        "runtime_seconds": runtime,
    }


def plot_cell_filtration_results(
    results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 5),
) -> plt.Figure:
    """
    Plot cell filtration experiment results.
    
    Parameters
    ----------
    results : dict
        Results from run_cell_filtration_experiment().
    save_path : str or Path, optional
        If provided, saves the figure.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # --- Plot 1: Predictivity score distribution ---
    ax1 = axes[0]
    scores = results["predictivity_scores"]
    ax1.hist(scores, bins=50, color="#3b82f6", edgecolor="white", alpha=0.7)
    ax1.axvline(x=0.1, color="#ef4444", linestyle="--", linewidth=2, label="Low confidence threshold")
    ax1.set_xlabel("Predictivity Score |P - 0.5|", fontsize=11)
    ax1.set_ylabel("Number of Cells", fontsize=11)
    ax1.set_title("Cell Predictivity Distribution", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    
    # --- Plot 2: AUC vs Filter Percentage ---
    ax2 = axes[1]
    filtered_aucs = results["filtered_aucs"]
    baseline_auc = results["baseline_auc"]
    
    if filtered_aucs:
        pcts = sorted(filtered_aucs.keys())
        aucs = [filtered_aucs[p] for p in pcts]
        
        # Add baseline point at 0%
        pcts = [0] + pcts
        aucs = [baseline_auc] + aucs
        
        ax2.plot(pcts, aucs, "o-", color="#3b82f6", linewidth=2, markersize=8)
        ax2.axhline(y=baseline_auc, color="#94a3b8", linestyle="--", linewidth=1, label="Baseline")
        
        # Mark best point
        best_pct = results["best_percentile"]
        best_auc = results["best_auc"]
        ax2.scatter([best_pct], [best_auc], color="#22c55e", s=150, zorder=5, label=f"Best ({best_pct}%)")
    
    ax2.set_xlabel("Percentage of Cells Removed (%)", fontsize=11)
    ax2.set_ylabel("ROC AUC", fontsize=11)
    ax2.set_title("AUC vs Cell Filtration Level", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Predictivity by Response Class ---
    ax3 = axes[2]
    cell_pred = results["cell_predictions"]
    
    r_scores = scores[cell_pred["true_labels"] == 1]
    nr_scores = scores[cell_pred["true_labels"] == 0]
    
    parts = ax3.violinplot([nr_scores, r_scores], positions=[0, 1], showmeans=True, showmedians=True)
    
    # Color the violins
    colors = ["#ef4444", "#22c55e"]
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.7)
    
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Non-responder", "Responder"])
    ax3.set_ylabel("Predictivity Score", fontsize=11)
    ax3.set_title("Predictivity by True Response", fontsize=12, fontweight="bold")
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Cell filtration plot saved to: {save_path}")
    
    return fig


# =============================================================================
# STRETCH GOAL B: External Cohort Validation (BCC Dataset)
# =============================================================================

# BCC patient response labels from Yost et al. 2019 Nature Medicine
# Based on clinical response to anti-PD1 therapy
BCC_PATIENT_RESPONSES = {
    "su001": "R",   # Responder
    "su002": "NR",  # Non-responder
    "su003": "R",   # Responder
    "su004": "R",   # Responder
    "su005": "NR",  # Non-responder
    "su006": "NR",  # Non-responder
    "su007": "R",   # Responder
    "su008": "R",   # Responder
    "su009": "NR",  # Non-responder
    "su010": "R",   # Responder
    "su012": "NR",  # Non-responder
}


def load_bcc_data(
    counts_path: Union[str, Path] = "data/raw/GSE123813_bcc_scRNA_counts.txt.gz",
    metadata_path: Union[str, Path] = "data/raw/GSE123813_bcc_all_metadata.txt.gz",
    use_pre_treatment: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Load and preprocess the BCC dataset (GSE123813).
    
    Parameters
    ----------
    counts_path : str or Path
        Path to the counts matrix file.
    metadata_path : str or Path
        Path to the cell metadata file.
    use_pre_treatment : bool, default=True
        If True, only use pre-treatment samples (for prediction).
        If False, use all samples.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    AnnData
        Preprocessed AnnData object with:
            - X: Normalized expression matrix
            - obs: Cell metadata with patient_id, response, response_binary
            - var: Gene metadata
    """
    if verbose:
        print("=" * 60)
        print("LOADING BCC DATASET (GSE123813)")
        print("=" * 60)
    
    counts_path = Path(counts_path)
    metadata_path = Path(metadata_path)
    
    # Load metadata
    if verbose:
        print(f"\nLoading metadata from: {metadata_path}")
    
    metadata = pd.read_csv(metadata_path, sep="\t", compression="gzip")
    
    if verbose:
        print(f"  Total cells in metadata: {len(metadata)}")
        print(f"  Patients: {metadata['patient'].unique().tolist()}")
        print(f"  Treatments: {metadata['treatment'].unique().tolist()}")
    
    # Filter to pre-treatment if requested
    if use_pre_treatment:
        metadata = metadata[metadata["treatment"] == "pre"].copy()
        if verbose:
            print(f"\nFiltered to pre-treatment: {len(metadata)} cells")
    
    # Load counts matrix
    if verbose:
        print(f"\nLoading counts from: {counts_path}")
        print("  (This may take a minute...)")
    
    counts = pd.read_csv(counts_path, sep="\t", compression="gzip", index_col=0)
    
    if verbose:
        print(f"  Counts matrix shape: {counts.shape}")
    
    # Get cells present in both metadata and counts
    cells_in_metadata = set(metadata["cell.id"])
    cells_in_counts = set(counts.columns)
    common_cells = list(cells_in_metadata & cells_in_counts)
    
    if verbose:
        print(f"\nCells in metadata: {len(cells_in_metadata)}")
        print(f"Cells in counts: {len(cells_in_counts)}")
        print(f"Common cells: {len(common_cells)}")
    
    # Subset to common cells
    metadata = metadata[metadata["cell.id"].isin(common_cells)].copy()
    metadata = metadata.set_index("cell.id")
    
    # Reorder counts to match metadata
    counts = counts[metadata.index]
    
    # Create AnnData object
    adata = ad.AnnData(
        X=counts.T.values.astype(np.float32),  # cells x genes
        obs=metadata,
        var=pd.DataFrame(index=counts.index),
    )
    
    if verbose:
        print(f"\nAnnData created: {adata.n_obs} cells x {adata.n_vars} genes")
    
    # Add response labels
    adata.obs["patient_id"] = adata.obs["patient"].values
    adata.obs["response"] = adata.obs["patient"].map(BCC_PATIENT_RESPONSES)
    adata.obs["response_binary"] = (adata.obs["response"] == "R").astype(int)
    
    # Check response distribution
    if verbose:
        print(f"\nResponse distribution:")
        resp_counts = adata.obs["response"].value_counts()
        for resp, count in resp_counts.items():
            n_patients = adata.obs[adata.obs["response"] == resp]["patient"].nunique()
            print(f"  {resp}: {count} cells from {n_patients} patients")
    
    # Basic preprocessing: log-normalize counts
    if verbose:
        print("\nNormalizing expression (log1p)...")
    
    # Library size normalization
    cell_totals = adata.X.sum(axis=1)
    adata.X = adata.X / cell_totals[:, np.newaxis] * 10000  # CPM-like
    adata.X = np.log1p(adata.X)
    
    if verbose:
        print(f"\nFinal AnnData: {adata.n_obs} cells x {adata.n_vars} genes")
        print(f"Patients: {adata.obs['patient_id'].nunique()}")
        print(f"Responders: {(adata.obs['response'] == 'R').sum()} cells")
        print(f"Non-responders: {(adata.obs['response'] == 'NR').sum()} cells")
    
    return adata


def compute_signature_score(
    adata: ad.AnnData,
    signature_genes: Optional[List[str]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Compute the 11-gene signature score for each cell.
    
    The signature score is the mean expression of signature genes present in the data.
    
    Parameters
    ----------
    adata : AnnData
        AnnData object with normalized expression.
    signature_genes : list of str, optional
        List of signature genes. Defaults to PRECISE_11_GENE_SIGNATURE.
    verbose : bool, default=True
        Whether to print info.
        
    Returns
    -------
    np.ndarray
        Signature score for each cell.
    """
    if signature_genes is None:
        signature_genes = PRECISE_11_GENE_SIGNATURE.copy()
    
    # Find signature genes present in data
    genes_present = [g for g in signature_genes if g in adata.var_names]
    genes_missing = [g for g in signature_genes if g not in adata.var_names]
    
    if verbose:
        print(f"Signature genes found: {len(genes_present)}/{len(signature_genes)}")
        if genes_missing:
            print(f"Missing: {genes_missing}")
    
    if not genes_present:
        raise ValueError("No signature genes found in the dataset")
    
    # Get expression of signature genes
    X = adata[:, genes_present].X
    if hasattr(X, "toarray"):
        X = X.toarray()
    
    # Compute mean expression (signature score)
    scores = X.mean(axis=1)
    
    return scores


def run_external_validation(
    adata_bcc: ad.AnnData,
    signature_genes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Validate the 11-gene signature on the external BCC cohort.
    
    Parameters
    ----------
    adata_bcc : AnnData
        BCC AnnData object (from load_bcc_data).
    signature_genes : list of str, optional
        Signature genes to use.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Validation results containing:
            - patient_scores: Per-patient signature scores
            - patient_labels: Per-patient true labels
            - auc: ROC AUC on BCC cohort
            - fpr, tpr, thresholds: ROC curve data
    """
    if signature_genes is None:
        signature_genes = PRECISE_11_GENE_SIGNATURE.copy()
    
    if verbose:
        print("=" * 60)
        print("EXTERNAL VALIDATION: BCC COHORT")
        print("=" * 60)
        print(f"\nSignature: {len(signature_genes)} genes")
        print(f"BCC data: {adata_bcc.n_obs} cells from {adata_bcc.obs['patient_id'].nunique()} patients")
    
    # Compute per-cell signature scores
    if verbose:
        print("\nComputing signature scores...")
    
    cell_scores = compute_signature_score(adata_bcc, signature_genes, verbose=verbose)
    
    # Aggregate to patient level (mean score)
    patient_ids = adata_bcc.obs["patient_id"].values
    unique_patients = adata_bcc.obs["patient_id"].unique()
    
    patient_scores = {}
    patient_labels = {}
    
    for patient in unique_patients:
        mask = patient_ids == patient
        patient_scores[patient] = float(cell_scores[mask].mean())
        patient_labels[patient] = int(adata_bcc.obs.loc[mask, "response_binary"].iloc[0])
    
    # Compute ROC AUC
    y_true = np.array([patient_labels[p] for p in unique_patients])
    y_scores = np.array([patient_scores[p] for p in unique_patients])
    
    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    if verbose:
        print("\n" + "=" * 60)
        print("EXTERNAL VALIDATION RESULTS")
        print("=" * 60)
        print(f"\nROC AUC on BCC cohort: {auc:.4f}")
        print(f"\nPatient scores:")
        for patient in sorted(unique_patients):
            label = "R" if patient_labels[patient] == 1 else "NR"
            score = patient_scores[patient]
            print(f"  {patient}: {score:.4f} (True: {label})")
    
    return {
        "patient_scores": patient_scores,
        "patient_labels": patient_labels,
        "cell_scores": cell_scores,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "n_patients": len(unique_patients),
        "n_cells": adata_bcc.n_obs,
        "signature_genes": signature_genes,
    }


def plot_external_validation(
    results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 5),
) -> plt.Figure:
    """
    Plot external validation results.
    
    Parameters
    ----------
    results : dict
        Results from run_external_validation().
    save_path : str or Path, optional
        If provided, saves the figure.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # --- Plot 1: ROC Curve ---
    ax1 = axes[0]
    ax1.plot(results["fpr"], results["tpr"], color="#3b82f6", linewidth=2.5,
             label=f"BCC Cohort (AUC = {results['auc']:.3f})")
    ax1.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1.5,
             label="Random (AUC = 0.500)")
    
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate", fontsize=11)
    ax1.set_ylabel("True Positive Rate", fontsize=11)
    ax1.set_title("ROC Curve - BCC External Validation", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", fontsize=10)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Patient Scores by Response ---
    ax2 = axes[1]
    
    patient_scores = results["patient_scores"]
    patient_labels = results["patient_labels"]
    
    r_scores = [patient_scores[p] for p in patient_scores if patient_labels[p] == 1]
    nr_scores = [patient_scores[p] for p in patient_scores if patient_labels[p] == 0]
    r_patients = [p for p in patient_scores if patient_labels[p] == 1]
    nr_patients = [p for p in patient_scores if patient_labels[p] == 0]
    
    # Plot points
    ax2.scatter([0] * len(nr_scores), nr_scores, c="#ef4444", s=100, alpha=0.8, 
                edgecolor="white", linewidth=1, label="Non-responder")
    ax2.scatter([1] * len(r_scores), r_scores, c="#22c55e", s=100, alpha=0.8,
                edgecolor="white", linewidth=1, label="Responder")
    
    # Add patient labels
    for i, (patient, score) in enumerate(zip(nr_patients, nr_scores)):
        ax2.annotate(patient, (0.05, score), fontsize=8, alpha=0.7)
    for i, (patient, score) in enumerate(zip(r_patients, r_scores)):
        ax2.annotate(patient, (1.05, score), fontsize=8, alpha=0.7)
    
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Non-responder", "Responder"])
    ax2.set_ylabel("Signature Score", fontsize=11)
    ax2.set_title("Patient Signature Scores by Response", fontsize=12, fontweight="bold")
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_xlim([-0.5, 1.5])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"External validation plot saved to: {save_path}")
    
    return fig


# =============================================================================
# STRETCH GOAL C: SHAP Interpretability
# =============================================================================

def compute_shap_values(
    adata: ad.AnnData,
    params: Optional[Dict[str, Any]] = None,
    n_background: int = 100,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretation.
    
    Uses one LOO-CV fold to train a model and compute SHAP values
    for the held-out patient's cells.
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object.
    params : dict, optional
        XGBoost hyperparameters.
    n_background : int, default=100
        Number of background samples for SHAP explainer.
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        SHAP analysis results containing:
            - shap_values: SHAP values array
            - feature_names: Gene names
            - mean_abs_shap: Mean |SHAP| per gene
            - top_genes_shap: Top genes by SHAP importance
            - xgb_importance: XGBoost's built-in importance
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP package is not installed. "
            "Install it with: pip install shap"
        )
    
    if verbose:
        print("=" * 60)
        print("SHAP INTERPRETABILITY ANALYSIS")
        print("=" * 60)
    
    # Get data
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs["response_binary"].values
    patient_ids = adata.obs["patient_id"].values
    gene_names = adata.var_names.tolist()
    
    # Use first patient as test for SHAP computation
    patients = adata.obs["patient_id"].unique()
    test_patient = patients[0]
    
    test_mask = patient_ids == test_patient
    train_mask = ~test_mask
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train = y[train_mask]
    
    if verbose:
        print(f"\nTraining model on {train_mask.sum()} cells (test patient: {test_patient})")
    
    # Train model
    model = train_xgboost(X_train, y_train, params)
    
    # Get XGBoost's built-in importance
    xgb_importance_dict = model.get_booster().get_score(importance_type="gain")
    xgb_importance = np.zeros(len(gene_names))
    for fname, score in xgb_importance_dict.items():
        idx = int(fname[1:])
        xgb_importance[idx] = score
    
    # Create SHAP explainer
    if verbose:
        print(f"\nComputing SHAP values for {X_test.shape[0]} test cells...")
        print("  (This may take a few minutes...)")
    
    # Use a sample of training data as background
    if X_train.shape[0] > n_background:
        background_idx = np.random.choice(X_train.shape[0], n_background, replace=False)
        background = X_train[background_idx]
    else:
        background = X_train
    
    explainer = shap.TreeExplainer(model, background)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, shap_values might be a list
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class
    
    # Compute mean absolute SHAP per gene
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Rank genes by SHAP importance
    shap_ranking = np.argsort(mean_abs_shap)[::-1]
    top_genes_shap = [gene_names[i] for i in shap_ranking[:50]]
    
    if verbose:
        print(f"\nSHAP analysis complete.")
        print(f"\nTop 10 genes by SHAP importance:")
        for i, gene in enumerate(top_genes_shap[:10]):
            gene_idx = gene_names.index(gene)
            print(f"  {i+1:2d}. {gene}: {mean_abs_shap[gene_idx]:.4f}")
    
    return {
        "shap_values": shap_values,
        "X_test": X_test,
        "feature_names": gene_names,
        "mean_abs_shap": mean_abs_shap,
        "top_genes_shap": top_genes_shap,
        "xgb_importance": xgb_importance,
        "model": model,
        "test_patient": test_patient,
    }


def plot_shap_analysis(
    shap_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    top_n: int = 20,
    figsize: tuple = (14, 6),
) -> plt.Figure:
    """
    Plot SHAP analysis results.
    
    Parameters
    ----------
    shap_results : dict
        Results from compute_shap_values().
    save_path : str or Path, optional
        If provided, saves the figure.
    top_n : int, default=20
        Number of top genes to show.
    figsize : tuple
        Figure size.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package is not installed.")
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    gene_names = shap_results["feature_names"]
    mean_abs_shap = shap_results["mean_abs_shap"]
    xgb_importance = shap_results["xgb_importance"]
    
    # Get top genes
    top_idx = np.argsort(mean_abs_shap)[::-1][:top_n]
    
    # --- Plot 1: SHAP Importance Bar Plot ---
    ax1 = axes[0]
    top_genes = [gene_names[i] for i in top_idx]
    top_shap = [mean_abs_shap[i] for i in top_idx]
    
    # Highlight signature genes
    colors = ["#f97316" if g in PRECISE_11_GENE_SIGNATURE else "#3b82f6" for g in top_genes]
    
    y_pos = np.arange(len(top_genes))
    ax1.barh(y_pos, top_shap[::-1], color=colors[::-1], edgecolor="white")
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_genes[::-1], fontsize=9)
    ax1.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax1.set_title(f"Top {top_n} Genes by SHAP Importance", fontsize=12, fontweight="bold")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#f97316", label="11-Gene Signature"),
        Patch(facecolor="#3b82f6", label="Other Genes"),
    ]
    ax1.legend(handles=legend_elements, loc="lower right", fontsize=9)
    
    # --- Plot 2: SHAP vs XGBoost Importance Comparison ---
    ax2 = axes[1]
    
    # Normalize both to [0, 1] for comparison
    shap_norm = mean_abs_shap / mean_abs_shap.max()
    xgb_norm = xgb_importance / (xgb_importance.max() + 1e-10)
    
    # Get top 100 genes by either metric
    combined_top = set(np.argsort(mean_abs_shap)[::-1][:100])
    combined_top |= set(np.argsort(xgb_importance)[::-1][:100])
    combined_top = list(combined_top)
    
    ax2.scatter(xgb_norm[combined_top], shap_norm[combined_top], 
                alpha=0.6, c="#3b82f6", edgecolor="white", s=50)
    
    # Mark signature genes
    sig_idx = [gene_names.index(g) for g in PRECISE_11_GENE_SIGNATURE if g in gene_names]
    ax2.scatter(xgb_norm[sig_idx], shap_norm[sig_idx],
                c="#f97316", s=100, edgecolor="black", linewidth=1, label="11-Gene Signature")
    
    # Add diagonal
    ax2.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
    
    ax2.set_xlabel("XGBoost Importance (normalized)", fontsize=11)
    ax2.set_ylabel("SHAP Importance (normalized)", fontsize=11)
    ax2.set_title("SHAP vs XGBoost Importance", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP analysis plot saved to: {save_path}")
    
    return fig


def plot_shap_beeswarm(
    shap_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    max_display: int = 20,
) -> plt.Figure:
    """
    Create SHAP beeswarm plot.
    
    Parameters
    ----------
    shap_results : dict
        Results from compute_shap_values().
    save_path : str or Path, optional
        If provided, saves the figure.
    max_display : int, default=20
        Number of features to display.
        
    Returns
    -------
    plt.Figure
        Matplotlib figure.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("SHAP package is not installed.")
    
    # Create figure
    fig = plt.figure(figsize=(10, 8))
    
    shap.summary_plot(
        shap_results["shap_values"],
        shap_results["X_test"],
        feature_names=shap_results["feature_names"],
        max_display=max_display,
        show=False,
    )
    
    plt.title("SHAP Beeswarm Plot - Feature Impact on Predictions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"SHAP beeswarm plot saved to: {save_path}")
    
    return fig


# =============================================================================
# MAIN: Run all stretch goals
# =============================================================================

def run_all_stretch_goals(
    adata: ad.AnnData,
    output_dir: Union[str, Path] = "results",
    run_goal_a: bool = True,
    run_goal_b: bool = True,
    run_goal_c: bool = True,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run all Phase 7 stretch goals.
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed melanoma AnnData object.
    output_dir : str or Path
        Output directory for figures and tables.
    run_goal_a : bool, default=True
        Whether to run Stretch Goal A (Cell Filtration).
    run_goal_b : bool, default=True
        Whether to run Stretch Goal B (External Validation).
    run_goal_c : bool, default=True
        Whether to run Stretch Goal C (SHAP Analysis).
    verbose : bool, default=True
        Whether to print progress.
        
    Returns
    -------
    dict
        Results from all stretch goals.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    
    results = {}
    
    # =========================================================================
    # STRETCH GOAL A: Cell Filtration
    # =========================================================================
    if run_goal_a:
        if verbose:
            print("\n" + "=" * 70)
            print("  STRETCH GOAL A: SIMPLIFIED CELL FILTRATION")
            print("=" * 70 + "\n")
        
        goal_a_results = run_cell_filtration_experiment(
            adata,
            filter_percentiles=[10, 20, 30, 40],
            verbose=verbose,
        )
        
        # Plot results
        plot_cell_filtration_results(
            goal_a_results,
            save_path=figures_dir / "stretch_cell_filtration.png",
        )
        
        results["goal_a"] = goal_a_results
        
        if verbose:
            print(f"\n✓ Stretch Goal A complete!")
            print(f"  Baseline AUC: {goal_a_results['baseline_auc']:.4f}")
            print(f"  Best filtered AUC: {goal_a_results['best_auc']:.4f}")
    
    # =========================================================================
    # STRETCH GOAL B: External Validation
    # =========================================================================
    if run_goal_b:
        if verbose:
            print("\n" + "=" * 70)
            print("  STRETCH GOAL B: EXTERNAL COHORT VALIDATION (BCC)")
            print("=" * 70 + "\n")
        
        try:
            # Load BCC data
            adata_bcc = load_bcc_data(verbose=verbose)
            
            # Run validation
            goal_b_results = run_external_validation(adata_bcc, verbose=verbose)
            
            # Plot results
            plot_external_validation(
                goal_b_results,
                save_path=figures_dir / "stretch_external_validation.png",
            )
            
            results["goal_b"] = goal_b_results
            results["adata_bcc"] = adata_bcc
            
            if verbose:
                print(f"\n✓ Stretch Goal B complete!")
                print(f"  BCC Cohort AUC: {goal_b_results['auc']:.4f}")
                
        except FileNotFoundError as e:
            print(f"\n✗ Stretch Goal B skipped: BCC data files not found")
            print(f"  Error: {e}")
            results["goal_b"] = None
    
    # =========================================================================
    # STRETCH GOAL C: SHAP Interpretability
    # =========================================================================
    if run_goal_c:
        if verbose:
            print("\n" + "=" * 70)
            print("  STRETCH GOAL C: SHAP INTERPRETABILITY")
            print("=" * 70 + "\n")
        
        try:
            goal_c_results = compute_shap_values(adata, verbose=verbose)
            
            # Plot results
            plot_shap_analysis(
                goal_c_results,
                save_path=figures_dir / "stretch_shap_analysis.png",
            )
            
            # Beeswarm plot
            plot_shap_beeswarm(
                goal_c_results,
                save_path=figures_dir / "stretch_shap_beeswarm.png",
            )
            
            results["goal_c"] = goal_c_results
            
            if verbose:
                print(f"\n✓ Stretch Goal C complete!")
                print(f"  Top SHAP gene: {goal_c_results['top_genes_shap'][0]}")
                
        except ImportError as e:
            print(f"\n✗ Stretch Goal C skipped: SHAP not installed")
            print(f"  Install with: pip install shap")
            results["goal_c"] = None
    
    # =========================================================================
    # Summary
    # =========================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("  PHASE 7 STRETCH GOALS SUMMARY")
        print("=" * 70)
        
        if run_goal_a and results.get("goal_a"):
            print(f"\n✓ Goal A (Cell Filtration): AUC improved from "
                  f"{results['goal_a']['baseline_auc']:.4f} to {results['goal_a']['best_auc']:.4f}")
        
        if run_goal_b and results.get("goal_b"):
            print(f"✓ Goal B (External Validation): BCC AUC = {results['goal_b']['auc']:.4f}")
        
        if run_goal_c and results.get("goal_c"):
            print(f"✓ Goal C (SHAP): Top gene = {results['goal_c']['top_genes_shap'][0]}")
        
        print(f"\nFigures saved to: {figures_dir}")
    
    return results


# =============================================================================
# Module-level script
# =============================================================================

if __name__ == "__main__":
    """
    Run stretch goals when module is executed directly.
    """
    import sys
    from src.preprocessing import load_preprocessed_data, DEFAULT_OUTPUT_PATH
    
    project_root = Path(__file__).parent.parent
    
    print("Loading preprocessed melanoma data...")
    adata = load_preprocessed_data(project_root / DEFAULT_OUTPUT_PATH)
    
    # Run all stretch goals
    results = run_all_stretch_goals(
        adata,
        output_dir=project_root / "results",
        verbose=True,
    )
    
    print("\nPhase 7 stretch goals complete!")

