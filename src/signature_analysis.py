"""
Signature Gene Analysis Module
==============================

Comprehensive analysis of the 11-gene signature from the PRECISE paper.

This module provides functions to:
    - Verify presence of signature genes in the dataset
    - Rank signature genes by feature importance
    - Visualize expression patterns across responders vs non-responders
    - Train and evaluate a model using only the 11 signature genes
    - Compare model performances (baseline, feature-selected, signature-only)

The 11-Gene Signature from PRECISE paper:
    GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# Import from sibling modules
from .model import (
    PRECISE_11_GENE_SIGNATURE,
    leave_one_patient_out_cv,
    evaluate_signature_genes,
    DEFAULT_XGBOOST_PARAMS,
)

# Set matplotlib style
plt.style.use("seaborn-v0_8-whitegrid")


def check_signature_genes_presence(
    adata: ad.AnnData,
    signature_genes: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Verify which signature genes are present in the dataset.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object.
    signature_genes : list of str, optional
        List of signature genes to check. Defaults to PRECISE_11_GENE_SIGNATURE.
    verbose : bool, default=True
        Whether to print detailed information.

    Returns
    -------
    dict
        Dictionary containing:
            - genes_present: List of genes found in the data
            - genes_missing: List of genes not found
            - n_present: Number of genes present
            - n_total: Total number of signature genes
            - all_present: Boolean indicating if all genes are present
            - presence_df: DataFrame with detailed presence info
    """
    if signature_genes is None:
        signature_genes = PRECISE_11_GENE_SIGNATURE.copy()

    genes_present = []
    genes_missing = []

    for gene in signature_genes:
        if gene in adata.var_names:
            genes_present.append(gene)
        else:
            genes_missing.append(gene)

    # Create detailed DataFrame
    presence_data = []
    for gene in signature_genes:
        is_present = gene in adata.var_names
        presence_data.append(
            {
                "gene": gene,
                "present": is_present,
                "status": "Found" if is_present else "Missing",
            }
        )

    presence_df = pd.DataFrame(presence_data)

    all_present = len(genes_missing) == 0

    if verbose:
        print("=" * 60)
        print("SIGNATURE GENE PRESENCE CHECK")
        print("=" * 60)
        print(f"\nSignature genes found: {len(genes_present)}/{len(signature_genes)}")
        print(f"All genes present: {all_present}")

        if genes_present:
            print(f"\nGenes FOUND ({len(genes_present)}):")
            for gene in genes_present:
                print(f"  ✓ {gene}")

        if genes_missing:
            print(f"\nGenes MISSING ({len(genes_missing)}):")
            for gene in genes_missing:
                print(f"  ✗ {gene}")

    return {
        "genes_present": genes_present,
        "genes_missing": genes_missing,
        "n_present": len(genes_present),
        "n_total": len(signature_genes),
        "all_present": all_present,
        "presence_df": presence_df,
    }


def get_signature_gene_ranks(
    importance_df: pd.DataFrame,
    signature_genes: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Find the rank of each signature gene in the feature importance analysis.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame with 'gene' and 'importance' columns.
        Should be sorted by importance (descending).
    signature_genes : list of str, optional
        List of signature genes. Defaults to PRECISE_11_GENE_SIGNATURE.
    verbose : bool, default=True
        Whether to print results.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
            - gene: Gene name
            - importance: Importance score
            - rank: Overall rank in all genes
            - in_top_50: Boolean if in top 50 genes
            - in_top_100: Boolean if in top 100 genes
    """
    if signature_genes is None:
        signature_genes = PRECISE_11_GENE_SIGNATURE.copy()

    # Ensure rank column exists
    if "rank" not in importance_df.columns:
        importance_df = importance_df.sort_values("importance", ascending=False).copy()
        importance_df["rank"] = range(1, len(importance_df) + 1)

    # Filter to signature genes
    sig_df = importance_df[importance_df["gene"].isin(signature_genes)].copy()

    # Mark genes not found
    missing_genes = [
        g for g in signature_genes if g not in importance_df["gene"].values
    ]
    if missing_genes:
        missing_df = pd.DataFrame(
            {
                "gene": missing_genes,
                "importance": [np.nan] * len(missing_genes),
                "rank": [np.nan] * len(missing_genes),
            }
        )
        sig_df = pd.concat([sig_df, missing_df], ignore_index=True)

    # Add convenience columns
    sig_df["in_top_50"] = sig_df["rank"] <= 50
    sig_df["in_top_100"] = sig_df["rank"] <= 100
    sig_df["in_top_250"] = sig_df["rank"] <= 250

    # Sort by rank
    sig_df = sig_df.sort_values("rank").reset_index(drop=True)

    # Add position in signature for display
    gene_order = {g: i for i, g in enumerate(signature_genes)}
    sig_df["signature_position"] = sig_df["gene"].map(
        lambda x: gene_order.get(x, -1) + 1
    )

    if verbose:
        print("=" * 60)
        print("SIGNATURE GENE RANKINGS")
        print("=" * 60)
        print(
            f"\n{'Gene':<12} {'Rank':>8} {'Importance':>12} {'Top 50':>8} {'Top 100':>9}"
        )
        print("-" * 50)
        for _, row in sig_df.iterrows():
            rank_str = f"{int(row['rank'])}" if not pd.isna(row["rank"]) else "N/A"
            imp_str = (
                f"{row['importance']:.2f}" if not pd.isna(row["importance"]) else "N/A"
            )
            top50_str = "Yes" if row["in_top_50"] else "No"
            top100_str = "Yes" if row["in_top_100"] else "No"
            print(
                f"{row['gene']:<12} {rank_str:>8} {imp_str:>12} {top50_str:>8} {top100_str:>9}"
            )

        # Summary stats
        n_top_50 = sig_df["in_top_50"].sum()
        n_top_100 = sig_df["in_top_100"].sum()
        n_total = len(sig_df)
        avg_rank = sig_df["rank"].mean()
        median_rank = sig_df["rank"].median()

        print(f"\n{'='*60}")
        print("SUMMARY")
        print("=" * 60)
        print(f"Signature genes in top 50:  {n_top_50}/{n_total}")
        print(f"Signature genes in top 100: {n_top_100}/{n_total}")
        print(f"Average rank: {avg_rank:.1f}")
        print(f"Median rank:  {median_rank:.1f}")

    return sig_df


def plot_signature_gene_expression(
    adata: ad.AnnData,
    signature_genes: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (16, 12),
) -> plt.Figure:
    """
    Create violin plots showing expression of signature genes by response class.

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with 'response' column in .obs
    signature_genes : list of str, optional
        List of signature genes. Defaults to PRECISE_11_GENE_SIGNATURE.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(16, 12)
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    if signature_genes is None:
        signature_genes = PRECISE_11_GENE_SIGNATURE.copy()

    # Filter to genes present in data
    genes_present = [g for g in signature_genes if g in adata.var_names]

    if not genes_present:
        raise ValueError("No signature genes found in the dataset")

    # Determine grid size
    n_genes = len(genes_present)
    n_cols = 4
    n_rows = int(np.ceil(n_genes / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()

    # Color palette
    palette = {"Non-responder": "#ef4444", "Responder": "#22c55e"}

    # Get expression data
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()

    for idx, gene in enumerate(genes_present):
        ax = axes[idx]

        # Get gene expression
        gene_idx = adata.var_names.tolist().index(gene)
        expression = X[:, gene_idx]

        # Get response labels - handle both short (R/NR) and full (Responder/Non-responder) formats
        response_raw = adata.obs["response"].values
        # Check the format and standardize
        unique_responses = set(response_raw)
        if "R" in unique_responses or "NR" in unique_responses:
            # Short format - map to full
            response_map = {"NR": "Non-responder", "R": "Responder"}
            response_labels = [response_map.get(r, r) for r in response_raw]
        else:
            # Already in full format
            response_labels = list(response_raw)

        plot_df = pd.DataFrame(
            {
                "Expression": expression,
                "Response": response_labels,
            }
        )

        # Separate data by response for manual violin plotting
        nr_data = plot_df[plot_df["Response"] == "Non-responder"]["Expression"].values
        r_data = plot_df[plot_df["Response"] == "Responder"]["Expression"].values

        # Create violin plot using matplotlib's violinplot for better compatibility
        parts_nr = ax.violinplot(
            [nr_data], positions=[0], showmeans=False, showmedians=True
        )
        parts_r = ax.violinplot(
            [r_data], positions=[1], showmeans=False, showmedians=True
        )

        # Color the violins
        for pc in parts_nr["bodies"]:
            pc.set_facecolor("#ef4444")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)
        for pc in parts_r["bodies"]:
            pc.set_facecolor("#22c55e")
            pc.set_edgecolor("black")
            pc.set_alpha(0.7)

        # Style the other parts
        for partname in ["cbars", "cmins", "cmaxes", "cmedians"]:
            if partname in parts_nr:
                parts_nr[partname].set_edgecolor("black")
            if partname in parts_r:
                parts_r[partname].set_edgecolor("black")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Non-responder", "Responder"], rotation=45, ha="right")
        ax.set_title(gene, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Log Expression" if idx % n_cols == 0 else "")

    # Hide empty subplots
    for idx in range(len(genes_present), len(axes)):
        axes[idx].set_visible(False)

    plt.suptitle(
        "11-Gene Signature Expression: Responders vs Non-Responders",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Signature expression plot saved to: {save_path}")

    return fig


def plot_signature_importance_comparison(
    importance_df: pd.DataFrame,
    signature_genes: Optional[List[str]] = None,
    top_n: int = 50,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (14, 10),
) -> plt.Figure:
    """
    Create bar plot highlighting signature genes among top features.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Feature importance DataFrame with 'gene' and 'importance' columns.
    signature_genes : list of str, optional
        List of signature genes to highlight.
    top_n : int, default=50
        Number of top genes to display.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(14, 10)
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    if signature_genes is None:
        signature_genes = PRECISE_11_GENE_SIGNATURE.copy()

    # Get top N genes
    df_top = importance_df.head(top_n).copy()
    df_top = df_top.sort_values("importance", ascending=True)  # For horizontal bars

    # Determine colors
    colors = [
        (
            "#f97316" if gene in signature_genes else "#3b82f6"
        )  # Orange for signature, blue otherwise
        for gene in df_top["gene"]
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

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
    ax.set_title(
        f"Top {top_n} Feature Importances\n(11-Gene Signature Highlighted in Orange)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#f97316", label="11-Gene Signature"),
        Patch(facecolor="#3b82f6", label="Other Genes"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)

    # Annotate signature genes in top N
    sig_in_top = [g for g in df_top["gene"] if g in signature_genes]
    if sig_in_top:
        # Add count annotation
        ax.text(
            0.02,
            0.98,
            f"Signature genes in top {top_n}: {len(sig_in_top)}/11",
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Importance comparison plot saved to: {save_path}")

    return fig


def plot_model_comparison(
    model_aucs: Dict[str, float],
    paper_aucs: Optional[Dict[str, float]] = None,
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 6),
) -> plt.Figure:
    """
    Create bar plot comparing AUCs across different models.

    Parameters
    ----------
    model_aucs : dict
        Dictionary of {model_name: auc_score}.
    paper_aucs : dict, optional
        Dictionary of {model_name: paper_reported_auc} for comparison.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(10, 6)
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    models = list(model_aucs.keys())
    our_aucs = [model_aucs[m] for m in models]
    x = np.arange(len(models))
    width = 0.35

    # Plot our AUCs
    bars1 = ax.bar(
        x - width / 2 if paper_aucs else x,
        our_aucs,
        width if paper_aucs else width * 2,
        label="Our Results",
        color="#3b82f6",
        edgecolor="white",
        linewidth=1,
    )

    # Add value labels
    for bar, auc in zip(bars1, our_aucs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{auc:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # Plot paper AUCs if provided
    if paper_aucs:
        paper_values = [paper_aucs.get(m, 0) for m in models]
        bars2 = ax.bar(
            x + width / 2,
            paper_values,
            width,
            label="Paper (PRECISE)",
            color="#22c55e",
            edgecolor="white",
            linewidth=1,
        )

        for bar, auc in zip(bars2, paper_values):
            if auc > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{auc:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                )

    # Formatting
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim([0, 1.1])
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    # Add reference line at 0.5 (random)
    ax.axhline(y=0.5, color="#94a3b8", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(len(models) - 0.5, 0.52, "Random (0.5)", fontsize=9, color="#64748b")

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Model comparison plot saved to: {save_path}")

    return fig


def plot_combined_roc_curves(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (10, 10),
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same plot.

    Parameters
    ----------
    results_dict : dict
        Dictionary of {model_name: results} where results contains
        'fpr', 'tpr', and 'auc' keys.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(10, 10)
        Figure size in inches.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color palette for different models
    colors = {
        "Baseline (All Genes)": "#3b82f6",  # Blue
        "Feature Selected": "#22c55e",  # Green
        "11-Gene Signature": "#f97316",  # Orange
        "Nested CV": "#8b5cf6",  # Purple
    }

    for model_name, results in results_dict.items():
        color = colors.get(model_name, "#64748b")
        auc = results.get("auc", 0)

        ax.plot(
            results["fpr"],
            results["tpr"],
            color=color,
            linewidth=2.5,
            label=f"{model_name} (AUC = {auc:.3f})",
        )

    # Plot diagonal reference line (random classifier)
    ax.plot(
        [0, 1],
        [0, 1],
        color="#94a3b8",
        linestyle="--",
        linewidth=1.5,
        label="Random (AUC = 0.500)",
    )

    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title("ROC Curve Comparison: All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Combined ROC curve saved to: {save_path}")

    return fig


def run_signature_analysis(
    adata: ad.AnnData,
    importance_df: Optional[pd.DataFrame] = None,
    params: Optional[Dict[str, Any]] = None,
    output_dir: Union[str, Path] = "results",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete 11-gene signature analysis pipeline.

    This function:
    1. Checks presence of all 11 signature genes
    2. Ranks signature genes by feature importance
    3. Trains and evaluates an 11-gene-only model (LOO-CV)
    4. Generates visualizations and comparison tables
    5. Saves all outputs

    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object with response labels.
    importance_df : pd.DataFrame, optional
        Pre-computed feature importance DataFrame. If None, will compute it.
    params : dict, optional
        XGBoost hyperparameters.
    output_dir : str or Path, default="results"
        Base output directory.
    verbose : bool, default=True
        Whether to print progress.

    Returns
    -------
    dict
        Results dictionary containing:
            - presence_info: Gene presence check results
            - ranks_df: Signature gene rankings
            - signature_results: LOO-CV results with 11-gene model
            - signature_auc: AUC of 11-gene model
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    if verbose:
        print("=" * 60)
        print("11-GENE SIGNATURE ANALYSIS")
        print("=" * 60)

    # Step 1: Check gene presence
    if verbose:
        print("\n" + "-" * 60)
        print("Step 1: Checking Signature Gene Presence")
        print("-" * 60)

    presence_info = check_signature_genes_presence(adata, verbose=verbose)

    # Step 2: Get importance rankings (if importance_df provided)
    ranks_df = None
    if importance_df is not None:
        if verbose:
            print("\n" + "-" * 60)
            print("Step 2: Ranking Signature Genes by Importance")
            print("-" * 60)

        ranks_df = get_signature_gene_ranks(importance_df, verbose=verbose)

        # Save ranks table
        ranks_path = tables_dir / "signature_gene_ranks.csv"
        ranks_path.parent.mkdir(parents=True, exist_ok=True)
        ranks_df.to_csv(ranks_path, index=False)
        if verbose:
            print(f"\nSignature gene ranks saved to: {ranks_path}")

    # Step 3: Train 11-gene-only model
    if verbose:
        print("\n" + "-" * 60)
        print("Step 3: Training 11-Gene-Only Model (LOO-CV)")
        print("-" * 60)

    signature_results = evaluate_signature_genes(
        adata,
        PRECISE_11_GENE_SIGNATURE,
        params=params,
        verbose=verbose,
    )
    signature_auc = signature_results["auc"]

    if verbose:
        print(f"\n11-Gene Signature AUC: {signature_auc:.4f}")

    # Step 4: Generate visualizations
    if verbose:
        print("\n" + "-" * 60)
        print("Step 4: Generating Visualizations")
        print("-" * 60)

    # Expression violin plots
    if verbose:
        print("\n  Creating expression violin plots...")
    try:
        plot_signature_gene_expression(
            adata,
            save_path=figures_dir / "signature_genes_expression.png",
        )
    except Exception as e:
        if verbose:
            print(f"  Warning: Could not create expression plot: {e}")

    # Importance comparison (if importance_df provided)
    if importance_df is not None:
        if verbose:
            print("  Creating importance comparison plot...")
        plot_signature_importance_comparison(
            importance_df,
            top_n=50,
            save_path=figures_dir / "signature_importance_comparison.png",
        )

    # Summary
    if verbose:
        print("\n" + "=" * 60)
        print("SIGNATURE ANALYSIS COMPLETE")
        print("=" * 60)
        print(f"\n11-Gene Signature AUC: {signature_auc:.4f}")
        print(f"Genes found: {presence_info['n_present']}/{presence_info['n_total']}")
        if ranks_df is not None:
            n_top_50 = ranks_df["in_top_50"].sum()
            print(f"Signature genes in our top 50: {n_top_50}/11")

    return {
        "presence_info": presence_info,
        "ranks_df": ranks_df,
        "signature_results": signature_results,
        "signature_auc": signature_auc,
    }


def generate_final_comparison_table(
    baseline_auc: float,
    feature_selected_auc: float,
    signature_auc: float,
    baseline_n_genes: int,
    feature_selected_n_genes: int,
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Generate a final comparison table of all models vs paper results.

    Parameters
    ----------
    baseline_auc : float
        AUC from baseline model (all genes).
    feature_selected_auc : float
        AUC from feature-selected model.
    signature_auc : float
        AUC from 11-gene signature model.
    baseline_n_genes : int
        Number of genes in baseline model.
    feature_selected_n_genes : int
        Number of genes after feature selection.
    save_path : str or Path, optional
        If provided, saves the table to this path.

    Returns
    -------
    pd.DataFrame
        Comparison DataFrame.
    """
    comparison_data = {
        "Model": [
            "Baseline XGBoost (All Genes)",
            "Feature Selected",
            "11-Gene Signature",
        ],
        "Our AUC": [
            f"{baseline_auc:.3f}",
            f"{feature_selected_auc:.3f}",
            f"{signature_auc:.3f}",
        ],
        "Paper AUC": [
            "0.84",
            "0.89",
            "~0.85*",
        ],
        "N Genes": [
            baseline_n_genes,
            feature_selected_n_genes,
            11,
        ],
        "Notes": [
            "All genes after QC",
            "Top genes by importance (nested CV)",
            "GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5",
        ],
    }

    df = pd.DataFrame(comparison_data)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Final comparison table saved to: {save_path}")

    return df


# ============================================================================
# Module-level script for testing
# ============================================================================

if __name__ == "__main__":
    """
    Run signature analysis when module is executed directly.

    Usage:
        python -m src.signature_analysis

    Or from project root:
        python src/signature_analysis.py
    """
    import sys
    from pathlib import Path

    # Import sibling modules
    from src.preprocessing import load_preprocessed_data, DEFAULT_OUTPUT_PATH

    project_root = Path(__file__).parent.parent

    print("Loading preprocessed data...")
    print("-" * 60)
    input_path = project_root / DEFAULT_OUTPUT_PATH
    adata = load_preprocessed_data(input_path)

    # Load feature importances if available
    importance_path = project_root / "results" / "tables" / "feature_importances.csv"
    if importance_path.exists():
        print(f"\nLoading feature importances from: {importance_path}")
        importance_df = pd.read_csv(importance_path)
    else:
        print("\nFeature importances not found. Running without importance rankings.")
        importance_df = None

    # Run signature analysis
    print("\n")
    results = run_signature_analysis(
        adata,
        importance_df=importance_df,
        output_dir=project_root / "results",
        verbose=True,
    )

    # Check acceptance criteria
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA CHECK (Phase 5)")
    print("=" * 60)

    # Task 5.1: All 11 genes present (or document missing)
    presence = results["presence_info"]
    check_5_1 = presence["n_present"] >= 10  # Allow 1 missing
    print(f"\n1. Signature genes present:")
    print(f"   Found: {presence['n_present']}/{presence['n_total']}")
    print(f"   Status: {'PASS' if check_5_1 else 'REVIEW'}")

    # Task 5.2: Ranks computed
    check_5_2 = results["ranks_df"] is not None
    print(f"\n2. Gene rankings computed:")
    print(f"   Status: {'PASS' if check_5_2 else 'FAIL'}")

    # Task 5.3: 11-gene model AUC computed
    sig_auc = results["signature_auc"]
    check_5_3 = 0.5 < sig_auc < 1.0
    print(f"\n3. 11-Gene model evaluated:")
    print(f"   AUC: {sig_auc:.4f}")
    print(f"   Status: {'PASS' if check_5_3 else 'FAIL'}")

    all_pass = check_5_1 and check_5_2 and check_5_3
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS NEED REVIEW'}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)
