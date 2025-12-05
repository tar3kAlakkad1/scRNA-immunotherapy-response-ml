"""
Feature Selection Module
========================

Implements feature selection methods for identifying predictive genes in
immunotherapy response prediction from single-cell RNA-seq data.

This module provides:
    - Boruta feature selection (as used in the PRECISE paper)
    - Importance-based feature selection (faster alternative)
    - Aggregation of feature importances across LOO-CV folds
    - Visualization of top genes
    - Utility functions for running LOO-CV with selected features

The PRECISE paper uses Boruta to reduce from ~10k genes to ~500-2000 genes,
improving AUC from ~0.84 to ~0.89 on the melanoma cohort.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Try to import BorutaPy; provide fallback message if not available
try:
    from boruta import BorutaPy

    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False
    BorutaPy = None

# Import from sibling modules
from .model import (
    train_xgboost,
    leave_one_patient_out_cv,
    DEFAULT_XGBOOST_PARAMS,
)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")


def run_boruta_selection(
    X: np.ndarray,
    y: np.ndarray,
    gene_names: Optional[List[str]] = None,
    random_state: int = 42,
    max_iter: int = 100,
    n_estimators: int = 100,
    verbose: int = 1,
) -> Tuple[List[int], List[str], Dict[str, Any]]:
    """
    Run Boruta feature selection to identify important genes.

    Boruta is a wrapper method around Random Forest that identifies all features
    that are statistically significantly more important than random probes.
    The PRECISE paper uses Boruta to select predictive genes.

    Parameters
    ----------
    X : np.ndarray
        Expression matrix of shape (n_cells, n_genes).
    y : np.ndarray
        Binary labels of shape (n_cells,). 1 = Responder, 0 = Non-responder.
    gene_names : list of str, optional
        Gene names corresponding to columns of X. If None, uses indices.
    random_state : int, default=42
        Random seed for reproducibility.
    max_iter : int, default=100
        Maximum number of iterations for Boruta.
    n_estimators : int, default=100
        Number of estimators for the underlying Random Forest.
    verbose : int, default=1
        Verbosity level (0=silent, 1=progress, 2=detailed).

    Returns
    -------
    selected_indices : list of int
        Indices of selected genes (columns in X).
    selected_genes : list of str
        Names of selected genes (if gene_names provided, else str(index)).
    info : dict
        Additional information about the selection:
            - n_selected: Number of genes selected
            - n_total: Total number of genes
            - ranking: Boruta ranking for all genes
            - support: Boolean mask of selected features
            - runtime_seconds: Runtime in seconds

    Raises
    ------
    ImportError
        If boruta package is not installed.
    ValueError
        If X and y have incompatible shapes.

    Examples
    --------
    >>> X = adata.X
    >>> y = adata.obs['response_binary'].values
    >>> gene_names = adata.var_names.tolist()
    >>> indices, genes, info = run_boruta_selection(X, y, gene_names)
    >>> print(f"Selected {len(genes)} genes")
    Selected 523 genes
    """
    if not BORUTA_AVAILABLE:
        raise ImportError(
            "Boruta package is not installed. " "Install it with: pip install boruta"
        )

    if X.shape[0] != len(y):
        raise ValueError(f"X has {X.shape[0]} samples but y has {len(y)} labels")

    start_time = time.time()

    if verbose >= 1:
        print("=" * 60)
        print("BORUTA FEATURE SELECTION")
        print("=" * 60)
        print(f"\nInput: {X.shape[0]:,} cells x {X.shape[1]:,} genes")
        print(f"Random state: {random_state}")
        print(f"Max iterations: {max_iter}")
        print(f"\nRunning Boruta (this may take a while)...")

    # Ensure X is dense
    if hasattr(X, "toarray"):
        X = X.toarray()

    # Create Random Forest classifier for Boruta
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=-1,
        random_state=random_state,
        max_depth=5,  # Limit depth for speed on high-dimensional data
    )

    # Initialize and run Boruta
    boruta_selector = BorutaPy(
        rf,
        n_estimators="auto",
        max_iter=max_iter,
        random_state=random_state,
        verbose=verbose,
    )

    boruta_selector.fit(X, y)

    # Extract results
    support = boruta_selector.support_
    ranking = boruta_selector.ranking_

    # Get selected indices
    selected_indices = np.where(support)[0].tolist()

    # Get selected gene names
    if gene_names is not None:
        selected_genes = [gene_names[i] for i in selected_indices]
    else:
        selected_genes = [str(i) for i in selected_indices]

    runtime = time.time() - start_time

    if verbose >= 1:
        print(f"\n{'=' * 60}")
        print("BORUTA RESULTS")
        print("=" * 60)
        print(f"\nSelected genes: {len(selected_indices)} / {X.shape[1]}")
        print(f"Selection ratio: {len(selected_indices) / X.shape[1]:.1%}")
        print(f"Runtime: {runtime:.1f} seconds ({runtime/60:.1f} minutes)")

        if len(selected_genes) <= 20:
            print(f"\nSelected genes: {selected_genes}")
        else:
            print(f"\nFirst 20 selected genes: {selected_genes[:20]}")

    # Build info dictionary
    info = {
        "n_selected": len(selected_indices),
        "n_total": X.shape[1],
        "ranking": ranking.tolist(),
        "support": support.tolist(),
        "runtime_seconds": runtime,
        "max_iter": max_iter,
        "random_state": random_state,
    }

    return selected_indices, selected_genes, info


def importance_based_selection(
    model: Union[xgb.XGBClassifier, None],
    gene_names: List[str],
    top_n: int = 50,
    importance_type: str = "gain",
    importance_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[int], List[str]]:
    """
    Select top N genes by XGBoost feature importance.

    This is a faster alternative to Boruta that simply takes the top N genes
    ranked by their feature importance from a trained XGBoost model.

    Note: Empirically, top_n=40-50 gives optimal AUC (~0.91) on the melanoma
    cohort. Using more genes (500+) actually hurts performance due to noise.

    Parameters
    ----------
    model : xgb.XGBClassifier or None
        Trained XGBoost classifier. If None, importance_df must be provided.
    gene_names : list of str
        Gene names corresponding to the model's features.
    top_n : int, default=500
        Number of top genes to select.
    importance_type : str, default="gain"
        Type of feature importance (used if extracting from model):
            - "gain": Average gain of splits using the feature
            - "weight": Number of times feature is used in splits
            - "cover": Average coverage of splits using the feature
    importance_df : pd.DataFrame, optional
        Pre-computed importance DataFrame with 'gene' and 'importance' columns.
        If provided, model parameter is ignored.

    Returns
    -------
    selected_indices : list of int
        Indices of selected genes.
    selected_genes : list of str
        Names of selected genes.

    Examples
    --------
    >>> # From a trained model
    >>> indices, genes = importance_based_selection(model, gene_names, top_n=500)

    >>> # From pre-computed importances
    >>> indices, genes = importance_based_selection(
    ...     None, gene_names, importance_df=importance_df
    ... )
    """
    if importance_df is not None:
        # Use pre-computed importances
        df = importance_df.copy()
    elif model is not None:
        # Extract importances from model
        importances = model.get_booster().get_score(importance_type=importance_type)

        # Convert to array
        importance_array = np.zeros(len(gene_names))
        for fname, score in importances.items():
            feature_idx = int(fname[1:])  # Remove 'f' prefix
            importance_array[feature_idx] = score

        df = pd.DataFrame(
            {
                "gene": gene_names,
                "importance": importance_array,
            }
        )
    else:
        raise ValueError("Either model or importance_df must be provided")

    # Ensure importance column exists
    if "importance" not in df.columns:
        raise ValueError("importance_df must have an 'importance' column")
    if "gene" not in df.columns:
        raise ValueError("importance_df must have a 'gene' column")

    # Sort by importance and select top N
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    # Limit to top_n
    top_n = min(top_n, len(df))
    df_top = df.head(top_n)

    # Get selected genes and their indices
    selected_genes = df_top["gene"].tolist()

    # Map back to original indices
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    selected_indices = [gene_to_idx[g] for g in selected_genes]

    return selected_indices, selected_genes


def get_feature_importance_df(
    models: List[xgb.XGBClassifier],
    gene_names: List[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Aggregate feature importances across LOO-CV folds.

    Computes the mean and standard deviation of feature importances across
    multiple trained models (e.g., from LOO-CV folds).

    Parameters
    ----------
    models : list of xgb.XGBClassifier
        List of trained XGBoost classifiers from each LOO-CV fold.
    gene_names : list of str
        Gene names corresponding to the model's features.
    importance_type : str, default="gain"
        Type of feature importance:
            - "gain": Average gain of splits using the feature
            - "weight": Number of times feature is used in splits
            - "cover": Average coverage of splits using the feature

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - gene: Gene name
            - importance: Mean importance across folds (alias: mean_importance)
            - importance_std: Standard deviation across folds (alias: std_importance)
            - rank: Rank by mean importance (1 = highest)
        Sorted by importance (descending).

    Examples
    --------
    >>> # Run LOO-CV with return_models=True
    >>> results = leave_one_patient_out_cv(adata, return_models=True)
    >>> importance_df = get_feature_importance_df(
    ...     results['fold_models'],
    ...     adata.var_names.tolist()
    ... )
    >>> importance_df.head(10)  # Top 10 genes
    """
    n_models = len(models)
    n_genes = len(gene_names)

    # Collect importances from each model
    all_importances = np.zeros((n_models, n_genes))

    for fold_idx, model in enumerate(models):
        importances = model.get_booster().get_score(importance_type=importance_type)

        for fname, score in importances.items():
            feature_idx = int(fname[1:])  # Remove 'f' prefix
            all_importances[fold_idx, feature_idx] = score

    # Compute mean and std across folds
    mean_importance = all_importances.mean(axis=0)
    std_importance = all_importances.std(axis=0)

    # Create DataFrame
    df = pd.DataFrame(
        {
            "gene": gene_names,
            "importance": mean_importance,
            "mean_importance": mean_importance,  # Alias for clarity
            "importance_std": std_importance,
            "std_importance": std_importance,  # Alias for clarity
        }
    )

    # Sort by importance and add rank
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df


def compute_importances_from_loocv(
    adata: ad.AnnData,
    params: Optional[Dict[str, Any]] = None,
    importance_type: str = "gain",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run LOO-CV and compute aggregated feature importances.

    This is a convenience function that:
    1. Runs LOO-CV with return_models=True
    2. Aggregates importances across all fold models
    3. Returns a sorted DataFrame of gene importances

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with labels.
    params : dict, optional
        XGBoost hyperparameters.
    importance_type : str, default="gain"
        Type of feature importance.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    pd.DataFrame
        Feature importance DataFrame sorted by importance.
    """
    if verbose:
        print("Running LOO-CV to collect feature importances...")

    # Run LOO-CV with models
    results = leave_one_patient_out_cv(
        adata,
        params=params,
        verbose=verbose,
        return_models=True,
    )

    if verbose:
        print(f"\nAggregating importances from {len(results['fold_models'])} folds...")

    # Aggregate importances
    importance_df = get_feature_importance_df(
        results["fold_models"],
        adata.var_names.tolist(),
        importance_type=importance_type,
    )

    return importance_df


def run_loocv_with_selected_genes(
    adata: ad.AnnData,
    selected_genes: List[str],
    params: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run LOO-CV using only a subset of selected genes.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with labels.
    selected_genes : list of str
        List of gene names to use for training/prediction.
    params : dict, optional
        XGBoost hyperparameters.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    dict
        Results from leave_one_patient_out_cv(), plus:
            - selected_genes: List of genes used
            - n_selected_genes: Number of genes used
            - genes_found: Genes that were found in the data
            - genes_missing: Genes that were not found
    """
    # Check which genes are present
    genes_found = [g for g in selected_genes if g in adata.var_names]
    genes_missing = [g for g in selected_genes if g not in adata.var_names]

    if verbose:
        print(
            f"\nSelected genes found in data: {len(genes_found)}/{len(selected_genes)}"
        )
        if genes_missing:
            print(
                f"Missing genes: {genes_missing[:10]}{'...' if len(genes_missing) > 10 else ''}"
            )

    if not genes_found:
        raise ValueError("No selected genes found in the dataset")

    # Subset to selected genes
    adata_subset = adata[:, genes_found].copy()

    if verbose:
        print(f"Running LOO-CV with {len(genes_found)} selected genes...")

    # Run LOO-CV
    results = leave_one_patient_out_cv(
        adata_subset,
        params=params,
        verbose=verbose,
    )

    # Add gene selection info
    results["selected_genes"] = selected_genes
    results["n_selected_genes"] = len(genes_found)
    results["genes_found"] = genes_found
    results["genes_missing"] = genes_missing

    return results


def run_nested_loocv_with_selection(
    adata: ad.AnnData,
    top_n: int = 50,
    params: Optional[Dict[str, Any]] = None,
    importance_type: str = "gain",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run LOO-CV with NESTED feature selection (no data leakage).

    This is the methodologically correct approach for evaluating feature selection:
    For each LOO fold:
        1. Train model on training data (47 patients) to get feature importances
        2. Select top N genes based on TRAINING-ONLY importances
        3. Retrain with selected genes on training data
        4. Test on held-out patient

    This prevents information leakage where test patient data influences
    which genes are selected.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with labels.
    top_n : int, default=50
        Number of top genes to select in each fold.
    params : dict, optional
        XGBoost hyperparameters.
    importance_type : str, default="gain"
        Type of feature importance for gene selection.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    dict
        Results dictionary containing:
            - patient_scores: {patient_id: predicted_score}
            - patient_labels: {patient_id: true_label}
            - auc: ROC AUC score (without data leakage)
            - fpr, tpr, thresholds: ROC curve data
            - runtime_seconds: Total runtime
            - n_patients, n_cells, n_genes: Dataset info
            - top_n: Number of genes selected per fold
            - genes_selected_per_fold: List of gene sets selected in each fold
            - gene_selection_overlap: How often each gene was selected across folds

    Notes
    -----
    This approach gives unbiased AUC estimates but is slower than the non-nested
    approach since it requires training 2 models per fold (one for importance,
    one for final prediction).

    The non-nested approach (run_loocv_with_selected_genes with pre-computed
    importances) is still useful for:
    - Identifying which genes are generally important (biological interpretation)
    - Faster iteration during development
    - Comparing to the correct nested AUC to understand bias magnitude
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    start_time = time.time()

    # Validate required columns
    required_cols = ["patient_id", "response_binary"]
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    # Get data
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    y = adata.obs["response_binary"].values
    patient_ids = adata.obs["patient_id"].values
    patients = adata.obs["patient_id"].unique()
    gene_names = adata.var_names.tolist()
    n_patients = len(patients)

    if verbose:
        print("=" * 60)
        print("NESTED LOO-CV WITH FEATURE SELECTION")
        print("=" * 60)
        print(f"\nDataset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"Patients: {n_patients}")
        print(f"Top N genes per fold: {top_n}")
        print(f"\nRunning nested LOO-CV (correct, no data leakage)...")

    # Initialize containers
    patient_scores = {}
    patient_labels = {}
    genes_selected_per_fold = []
    gene_selection_counts = {}

    for fold_idx, test_patient in enumerate(patients):
        # Split data
        test_mask = patient_ids == test_patient
        train_mask = ~test_mask

        X_train, X_test = X[train_mask], X[test_mask]
        y_train = y[train_mask]

        # Step 1: Train model on ALL genes to get importances (training data only)
        model_full = train_xgboost(X_train, y_train, params)

        # Extract importances
        importances_dict = model_full.get_booster().get_score(
            importance_type=importance_type
        )
        importance_array = np.zeros(len(gene_names))
        for fname, score in importances_dict.items():
            feature_idx = int(fname[1:])
            importance_array[feature_idx] = score

        # Step 2: Select top N genes from TRAINING importances only
        top_indices = np.argsort(importance_array)[::-1][:top_n]
        selected_genes = [gene_names[i] for i in top_indices]
        genes_selected_per_fold.append(selected_genes)

        # Track gene selection frequency
        for gene in selected_genes:
            gene_selection_counts[gene] = gene_selection_counts.get(gene, 0) + 1

        # Step 3: Retrain on selected genes only
        X_train_selected = X_train[:, top_indices]
        X_test_selected = X_test[:, top_indices]

        model_selected = train_xgboost(X_train_selected, y_train, params)

        # Step 4: Predict on held-out patient
        cell_probs = model_selected.predict_proba(X_test_selected)[:, 1]
        patient_score = float(cell_probs.mean())

        patient_scores[test_patient] = patient_score
        patient_labels[test_patient] = int(y[test_mask][0])

        if verbose and (fold_idx + 1) % 10 == 0:
            print(f"  Completed fold {fold_idx + 1}/{n_patients}")

    # Compute patient-level ROC AUC
    y_true = np.array([patient_labels[p] for p in patients])
    y_scores = np.array([patient_scores[p] for p in patients])

    auc = roc_auc_score(y_true, y_scores)
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    runtime = time.time() - start_time

    # Sort gene selection counts
    gene_selection_overlap = (
        pd.DataFrame(
            [
                {
                    "gene": gene,
                    "times_selected": count,
                    "selection_rate": count / n_patients,
                }
                for gene, count in gene_selection_counts.items()
            ]
        )
        .sort_values("times_selected", ascending=False)
        .reset_index(drop=True)
    )

    if verbose:
        print(f"\n  Completed all {n_patients} folds in {runtime:.1f} seconds")
        print(f"\n" + "=" * 60)
        print("RESULTS (Nested LOO-CV - No Data Leakage)")
        print("=" * 60)
        print(f"\nPatient-level ROC AUC: {auc:.4f}")
        print(f"Score range: [{min(y_scores):.3f}, {max(y_scores):.3f}]")
        print(f"\nGene selection consistency:")
        print(
            f"  Genes selected in ALL folds: {len(gene_selection_overlap[gene_selection_overlap['times_selected'] == n_patients])}"
        )
        print(
            f"  Genes selected in >50% folds: {len(gene_selection_overlap[gene_selection_overlap['selection_rate'] > 0.5])}"
        )
        if len(gene_selection_overlap) > 0:
            print(f"\n  Top 10 most consistently selected genes:")
            for _, row in gene_selection_overlap.head(10).iterrows():
                print(
                    f"    {row['gene']}: {row['times_selected']}/{n_patients} folds ({row['selection_rate']:.0%})"
                )

    return {
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
        "top_n": top_n,
        "genes_selected_per_fold": genes_selected_per_fold,
        "gene_selection_overlap": gene_selection_overlap,
    }


def save_selected_genes(
    selected_genes: List[str],
    importance_df: Optional[pd.DataFrame] = None,
    save_path: Union[str, Path] = "results/tables/selected_genes.csv",
) -> pd.DataFrame:
    """
    Save selected genes list to CSV file.

    Parameters
    ----------
    selected_genes : list of str
        List of selected gene names.
    importance_df : pd.DataFrame, optional
        Importance DataFrame to merge with (for importance scores).
    save_path : str or Path
        Path to save the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame of selected genes with their importance scores (if available).
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if importance_df is not None:
        # Filter to selected genes and preserve order
        df = importance_df[importance_df["gene"].isin(selected_genes)].copy()
        # Re-rank within selected genes
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)
        df["rank_in_selection"] = range(1, len(df) + 1)
    else:
        df = pd.DataFrame(
            {
                "gene": selected_genes,
                "rank_in_selection": range(1, len(selected_genes) + 1),
            }
        )

    df.to_csv(save_path, index=False)
    print(f"Selected genes saved to: {save_path}")

    return df


def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 30,
    highlight_genes: Optional[List[str]] = None,
    title: str = "Top Feature Importances",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 10),
) -> plt.Figure:
    """
    Create a horizontal bar plot of top feature importances.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame with 'gene' and 'importance' columns.
    top_n : int, default=30
        Number of top genes to display.
    highlight_genes : list of str, optional
        List of gene names to highlight (e.g., signature genes).
    title : str, default="Top Feature Importances"
        Plot title.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(12, 10)
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    # Get top N genes
    df_top = importance_df.head(top_n).copy()
    df_top = df_top.sort_values("importance", ascending=True)  # For horizontal bar

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine colors
    if highlight_genes is not None:
        colors = [
            (
                "#f97316" if gene in highlight_genes else "#3b82f6"
            )  # Orange for highlight, blue otherwise
            for gene in df_top["gene"]
        ]
    else:
        colors = "#3b82f6"

    # Create horizontal bar plot
    bars = ax.barh(
        df_top["gene"],
        df_top["importance"],
        color=colors,
        edgecolor="white",
        linewidth=0.5,
    )

    # Add error bars if available
    if "importance_std" in df_top.columns:
        ax.errorbar(
            df_top["importance"],
            df_top["gene"],
            xerr=df_top["importance_std"],
            fmt="none",
            color="#1e3a5f",
            capsize=3,
            capthick=1,
            alpha=0.7,
        )

    # Formatting
    ax.set_xlabel("Feature Importance (Gain)", fontsize=12)
    ax.set_ylabel("Gene", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend if highlighting
    if highlight_genes is not None:
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="#f97316", label="Signature Genes"),
            Patch(facecolor="#3b82f6", label="Other Genes"),
        ]
        ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Feature importance plot saved to: {save_path}")

    return fig


def run_feature_selection_pipeline(
    adata: ad.AnnData,
    method: str = "importance",
    top_n: int = 50,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "results",
    verbose: bool = True,
    use_nested_cv: bool = True,
) -> Dict[str, Any]:
    """
    Run complete feature selection pipeline with proper nested cross-validation.

    This function:
    1. Runs baseline LOO-CV (all genes) and collects feature importances
    2. Identifies top genes for biological interpretation
    3. Runs NESTED LOO-CV with feature selection (correct, no data leakage)
    4. Saves results and plots

    IMPORTANT: The nested CV approach (use_nested_cv=True) is methodologically
    correct and prevents data leakage. The non-nested approach gives inflated
    AUC estimates because gene selection uses test patient information.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with labels.
    method : str, default="importance"
        Selection method: "importance" (fast) or "boruta" (slow but thorough).
    top_n : int, default=50
        Number of genes to select per fold.
    params : dict, optional
        XGBoost hyperparameters.
    output_dir : str or Path, default="results"
        Base output directory.
    verbose : bool, default=True
        Whether to print progress.
    use_nested_cv : bool, default=True
        If True (recommended), uses nested CV where feature selection happens
        inside each fold. This is the correct approach that prevents data leakage.
        If False, uses pre-computed importances (faster but gives inflated AUC).

    Returns
    -------
    dict
        Results dictionary containing:
            - importance_df: DataFrame of all gene importances (for interpretation)
            - selected_genes: List of top gene names (from global importances)
            - baseline_auc: AUC before feature selection
            - nested_auc: AUC from nested CV (correct, no leakage) - REPORT THIS
            - baseline_results: Full results from baseline LOO-CV
            - nested_results: Full results from nested LOO-CV
            - gene_selection_overlap: How consistently genes are selected
    """
    output_dir = Path(output_dir)

    if verbose:
        print("=" * 60)
        print("FEATURE SELECTION PIPELINE")
        print("=" * 60)
        print(f"\nMethod: {method}")
        print(f"Dataset: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        print(f"Nested CV (correct): {use_nested_cv}")

    # Step 1: Run baseline LOO-CV and collect importances
    if verbose:
        print("\n" + "-" * 60)
        print("Step 1: Baseline LOO-CV (all genes) + Feature Importances")
        print("-" * 60)

    baseline_results = leave_one_patient_out_cv(
        adata,
        params=params,
        verbose=verbose,
        return_models=True,
    )
    baseline_auc = baseline_results["auc"]

    if verbose:
        print(f"\nBaseline AUC: {baseline_auc:.4f}")

    # Get importances from models (for biological interpretation)
    importance_df = get_feature_importance_df(
        baseline_results["fold_models"],
        adata.var_names.tolist(),
    )

    # Step 2: Identify top genes (for interpretation, not evaluation)
    if verbose:
        print("\n" + "-" * 60)
        print(f"Step 2: Identifying top {top_n} genes (for biological interpretation)")
        print("-" * 60)

    if method == "importance":
        selected_indices, selected_genes = importance_based_selection(
            None,
            adata.var_names.tolist(),
            top_n=top_n,
            importance_df=importance_df,
        )
        selection_info = {"method": "importance", "top_n": top_n}
    elif method == "boruta":
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        y = adata.obs["response_binary"].values

        selected_indices, selected_genes, selection_info = run_boruta_selection(
            X,
            y,
            gene_names=adata.var_names.tolist(),
            verbose=verbose,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'importance' or 'boruta'")

    if verbose:
        print(f"\nTop {len(selected_genes)} genes identified")

    # Step 3: Run nested LOO-CV with feature selection (CORRECT approach)
    if verbose:
        print("\n" + "-" * 60)
        print("Step 3: Nested LOO-CV with Feature Selection")
        print("         (Correct evaluation - no data leakage)")
        print("-" * 60)

    if use_nested_cv:
        nested_results = run_nested_loocv_with_selection(
            adata,
            top_n=top_n,
            params=params,
            importance_type="gain",
            verbose=verbose,
        )
        nested_auc = nested_results["auc"]
        gene_selection_overlap = nested_results["gene_selection_overlap"]
    else:
        # Non-nested (faster but inflated AUC - for comparison only)
        if verbose:
            print("\n  WARNING: Using non-nested CV gives INFLATED AUC estimates!")
        nested_results = run_loocv_with_selected_genes(
            adata,
            selected_genes,
            params=params,
            verbose=verbose,
        )
        nested_auc = nested_results["auc"]
        gene_selection_overlap = None

    # Step 4: Save results
    if verbose:
        print("\n" + "-" * 60)
        print("Step 4: Saving results")
        print("-" * 60)

    # Save importance DataFrame
    importance_path = output_dir / "tables" / "feature_importances.csv"
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(importance_path, index=False)
    if verbose:
        print(f"Feature importances saved to: {importance_path}")

    # Save selected genes
    selected_genes_df = save_selected_genes(
        selected_genes,
        importance_df=importance_df,
        save_path=output_dir / "tables" / "selected_genes.csv",
    )

    # Plot feature importance
    from .model import PRECISE_11_GENE_SIGNATURE

    plot_feature_importance(
        importance_df,
        top_n=30,
        highlight_genes=PRECISE_11_GENE_SIGNATURE,
        title="Top 30 Feature Importances (LOO-CV Average)",
        save_path=output_dir / "figures" / "feature_importance_top30.png",
    )

    # Save gene selection overlap if available
    if gene_selection_overlap is not None:
        overlap_path = output_dir / "tables" / "gene_selection_overlap.csv"
        gene_selection_overlap.to_csv(overlap_path, index=False)
        if verbose:
            print(f"Gene selection overlap saved to: {overlap_path}")

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("FEATURE SELECTION RESULTS")
        print("=" * 60)
        print(f"\nTop genes identified: {len(selected_genes)}")
        print(f"\nBaseline AUC (all genes):        {baseline_auc:.4f}")
        print(f"Nested CV AUC (top {top_n} genes): {nested_auc:.4f}")
        print(f"AUC change:                      {nested_auc - baseline_auc:+.4f}")
        if use_nested_cv:
            print(f"\n(Nested CV is the correct, unbiased estimate)")

    return {
        "importance_df": importance_df,
        "selected_genes": selected_genes,
        "selected_genes_df": selected_genes_df,
        "selected_indices": selected_indices,
        "baseline_auc": baseline_auc,
        "nested_auc": nested_auc,  # This is the correct AUC to report
        "baseline_results": baseline_results,
        "nested_results": nested_results,
        "selection_info": selection_info,
        "gene_selection_overlap": gene_selection_overlap,
    }


# ============================================================================
# Module-level script for testing
# ============================================================================

if __name__ == "__main__":
    """
    Run feature selection when module is executed directly.

    Usage:
        python -m src.feature_selection

    Or from project root:
        python src/feature_selection.py
    """
    import sys

    # Import sibling modules
    from src.preprocessing import load_preprocessed_data, DEFAULT_OUTPUT_PATH

    project_root = Path(__file__).parent.parent

    print("Loading preprocessed data...")
    print("-" * 60)
    input_path = project_root / DEFAULT_OUTPUT_PATH
    adata = load_preprocessed_data(input_path)

    # Run feature selection pipeline with correct nested CV
    print("\n")
    results = run_feature_selection_pipeline(
        adata,
        method="importance",
        top_n=50,
        output_dir=project_root / "results",
        verbose=True,
        use_nested_cv=True,  # Correct approach - no data leakage
    )

    # Acceptance criteria check
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA CHECK (Task 4.1)")
    print("=" * 60)

    # Check 1: Feature selection reduces gene count significantly
    n_original = adata.n_vars
    n_selected = len(results["selected_genes"])
    reduction_ok = n_selected < n_original * 0.5  # At least 50% reduction
    print(f"\n1. Gene count reduction:")
    print(f"   Original: {n_original:,}")
    print(f"   Top genes identified: {n_selected:,}")
    print(f"   Reduction: {(n_original - n_selected) / n_original:.1%}")
    print(f"   Status: {'PASS' if reduction_ok else 'NEEDS REVIEW'}")

    # Check 2: Selected genes list saved
    selected_genes_path = project_root / "results" / "tables" / "selected_genes.csv"
    file_ok = selected_genes_path.exists()
    print(f"\n2. Selected genes file saved:")
    print(f"   Path: {selected_genes_path}")
    print(f"   Status: {'PASS' if file_ok else 'FAIL'}")

    # Check 3: Verify file contents
    if file_ok:
        df = pd.read_csv(selected_genes_path)
        content_ok = len(df) == n_selected and "gene" in df.columns
        print(f"\n3. File contents verified:")
        print(f"   Rows: {len(df)}")
        print(f"   Has 'gene' column: {'gene' in df.columns}")
        print(f"   Status: {'PASS' if content_ok else 'FAIL'}")
    else:
        content_ok = False

    # Check 4: Report AUC comparison (nested CV is correct)
    print(f"\n4. AUC comparison (nested CV - no data leakage):")
    print(f"   Baseline AUC (all genes):     {results['baseline_auc']:.4f}")
    print(f"   Nested CV AUC (top 50/fold):  {results['nested_auc']:.4f}")
    print(
        f"   AUC change:                   {results['nested_auc'] - results['baseline_auc']:+.4f}"
    )
    print(f"   Paper baseline:               0.84")
    print(f"   Paper with Boruta:            0.89")

    # Check 5: Gene selection consistency
    if results.get("gene_selection_overlap") is not None:
        overlap = results["gene_selection_overlap"]
        consistent_genes = len(overlap[overlap["selection_rate"] > 0.8])
        print(f"\n5. Gene selection consistency:")
        print(f"   Genes selected in >80% of folds: {consistent_genes}")
        print(f"   Status: {'PASS' if consistent_genes > 0 else 'REVIEW'}")

    # Overall status
    all_pass = reduction_ok and file_ok and content_ok
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS NEED REVIEW'}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)
