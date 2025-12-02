"""
Evaluation Module
=================

Computes metrics and generates evaluation plots for the XGBoost immunotherapy
response prediction model.

This module provides functions to:
    - Compute ROC AUC from patient-level predictions
    - Generate ROC curve plots
    - Create scatter plots of patient predictions by true label
    - Save results summaries as CSV tables

Expected Usage:
    After running leave_one_patient_out_cv() from src/model.py, use these
    functions to evaluate and visualize the results.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# Set matplotlib style for publication-quality figures
plt.style.use("seaborn-v0_8-whitegrid")


def compute_roc_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Compute ROC AUC from true labels and predicted scores.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels (0 = Non-responder, 1 = Responder).
    y_scores : np.ndarray
        Predicted scores/probabilities for the positive class (Responder).

    Returns
    -------
    float
        ROC AUC score in range [0, 1].

    Examples
    --------
    >>> y_true = np.array([0, 0, 1, 1])
    >>> y_scores = np.array([0.1, 0.4, 0.35, 0.8])
    >>> compute_roc_auc(y_true, y_scores)
    0.75
    """
    return roc_auc_score(y_true, y_scores)


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    title: str = "ROC Curve - Patient-Level Predictions",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (8, 8),
    show_optimal_threshold: bool = True,
) -> plt.Figure:
    """
    Plot and optionally save ROC curve.

    Creates a publication-quality ROC curve showing:
        - ROC curve with AUC in legend
        - Diagonal reference line (random classifier)
        - Optionally, the optimal threshold point (Youden's J)

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_scores : np.ndarray
        Predicted scores/probabilities.
    title : str, default="ROC Curve - Patient-Level Predictions"
        Plot title.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(8, 8)
        Figure size in inches.
    show_optimal_threshold : bool, default=True
        Whether to mark the optimal threshold point on the curve.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.

    Examples
    --------
    >>> fig = plot_roc_curve(y_true, y_scores, save_path="results/figures/roc.png")
    """
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot ROC curve
    ax.plot(
        fpr,
        tpr,
        color="#2563eb",  # Blue
        linewidth=2.5,
        label=f"XGBoost Model (AUC = {auc:.3f})",
    )

    # Plot diagonal reference line (random classifier)
    ax.plot(
        [0, 1],
        [0, 1],
        color="#94a3b8",  # Gray
        linestyle="--",
        linewidth=1.5,
        label="Random Classifier (AUC = 0.500)",
    )

    # Mark optimal threshold point (Youden's J statistic)
    if show_optimal_threshold:
        # Youden's J = TPR - FPR (maximize sensitivity + specificity)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]

        ax.scatter(
            optimal_fpr,
            optimal_tpr,
            color="#dc2626",  # Red
            s=100,
            zorder=5,
            label=f"Optimal Threshold = {optimal_threshold:.3f}",
        )

    # Formatting
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=12)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_aspect("equal")

    # Add grid
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to: {save_path}")

    return fig


def plot_patient_predictions(
    patient_scores: Dict[str, float],
    patient_labels: Dict[str, int],
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 6),
    title: str = "Patient-Level Prediction Scores",
) -> plt.Figure:
    """
    Create a scatter/strip plot of patient scores colored by true label.

    Shows the distribution of predicted scores for responders vs non-responders,
    helping visualize the model's discrimination ability.

    Parameters
    ----------
    patient_scores : dict
        Mapping of {patient_id: predicted_score}.
    patient_labels : dict
        Mapping of {patient_id: true_label} where 0=NR, 1=R.
    save_path : str or Path, optional
        If provided, saves the figure to this path.
    figsize : tuple, default=(12, 6)
        Figure size in inches.
    title : str, default="Patient-Level Prediction Scores"
        Plot title.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object.
    """
    # Build DataFrame for plotting
    patients = list(patient_scores.keys())
    scores = [patient_scores[p] for p in patients]
    labels = [patient_labels[p] for p in patients]
    label_names = ["Non-responder" if l == 0 else "Responder" for l in labels]

    df = pd.DataFrame(
        {
            "Patient": patients,
            "Score": scores,
            "Label": labels,
            "Label_Name": label_names,
        }
    )

    # Sort by score for better visualization
    df = df.sort_values("Score").reset_index(drop=True)

    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Color palette
    colors = {"Non-responder": "#ef4444", "Responder": "#22c55e"}  # Red  # Green

    # --- Subplot 1: Strip plot by class ---
    ax1 = axes[0]
    for label_name in ["Non-responder", "Responder"]:
        subset = df[df["Label_Name"] == label_name]
        jitter = np.random.normal(0, 0.08, size=len(subset))
        x_pos = 0 if label_name == "Non-responder" else 1
        ax1.scatter(
            x_pos + jitter,
            subset["Score"],
            c=colors[label_name],
            alpha=0.7,
            s=80,
            edgecolor="white",
            linewidth=0.5,
            label=label_name,
        )

    # Add box plot overlay
    nr_scores = df[df["Label_Name"] == "Non-responder"]["Score"]
    r_scores = df[df["Label_Name"] == "Responder"]["Score"]
    bp = ax1.boxplot(
        [nr_scores, r_scores],
        positions=[0, 1],
        widths=0.3,
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], [colors["Non-responder"], colors["Responder"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(["Non-responder", "Responder"])
    ax1.set_ylabel("Predicted Score (P(Responder))", fontsize=11)
    ax1.set_xlabel("True Label", fontsize=11)
    ax1.set_title("Score Distribution by True Label", fontsize=12, fontweight="bold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3, axis="y")

    # --- Subplot 2: Ordered patient plot ---
    ax2 = axes[1]
    for idx, row in df.iterrows():
        ax2.scatter(
            idx,
            row["Score"],
            c=colors[row["Label_Name"]],
            s=60,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

    # Add horizontal line at 0.5 threshold
    ax2.axhline(y=0.5, color="#64748b", linestyle="--", linewidth=1, alpha=0.7)
    ax2.text(
        len(df) - 1,
        0.52,
        "Threshold = 0.5",
        ha="right",
        fontsize=9,
        color="#64748b",
    )

    ax2.set_xlabel("Patients (sorted by score)", fontsize=11)
    ax2.set_ylabel("Predicted Score (P(Responder))", fontsize=11)
    ax2.set_title("All Patients Ranked by Prediction", fontsize=12, fontweight="bold")
    ax2.set_ylim([-0.05, 1.05])
    ax2.grid(True, alpha=0.3, axis="y")

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=colors["Non-responder"], label="Non-responder (True)"),
        Patch(facecolor=colors["Responder"], label="Responder (True)"),
    ]
    ax2.legend(handles=legend_elements, loc="upper left", fontsize=9)

    # Overall title
    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Patient predictions plot saved to: {save_path}")

    return fig


def generate_results_table(
    results_dict: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    model_name: str = "Baseline XGBoost",
) -> pd.DataFrame:
    """
    Generate and optionally save a results summary table as CSV.

    Parameters
    ----------
    results_dict : dict
        Results dictionary from leave_one_patient_out_cv(), containing:
            - auc: ROC AUC score
            - n_patients: Number of patients
            - n_cells: Total number of cells
            - n_genes: Number of genes
            - runtime_seconds: Total runtime
            - patient_scores: Per-patient predicted scores
            - patient_labels: Per-patient true labels
    save_path : str or Path, optional
        If provided, saves the table to this path as CSV.
    model_name : str, default="Baseline XGBoost"
        Name of the model for the results table.

    Returns
    -------
    pd.DataFrame
        Results summary DataFrame.
    """
    # Extract key metrics
    auc = results_dict["auc"]
    n_patients = results_dict["n_patients"]
    n_cells = results_dict["n_cells"]
    n_genes = results_dict["n_genes"]
    runtime = results_dict["runtime_seconds"]

    # Compute additional metrics
    patient_scores = results_dict["patient_scores"]
    patient_labels = results_dict["patient_labels"]

    scores = np.array(list(patient_scores.values()))
    labels = np.array(list(patient_labels.values()))

    # Compute score statistics
    score_min = scores.min()
    score_max = scores.max()
    score_mean = scores.mean()
    score_std = scores.std()

    # Compute score statistics by class
    r_scores = scores[labels == 1]
    nr_scores = scores[labels == 0]
    r_mean = r_scores.mean() if len(r_scores) > 0 else np.nan
    nr_mean = nr_scores.mean() if len(nr_scores) > 0 else np.nan

    # Label distribution
    n_responders = int(labels.sum())
    n_non_responders = len(labels) - n_responders

    # Build summary table
    summary_data = {
        "Metric": [
            "Model",
            "ROC AUC",
            "Number of Patients",
            "  - Responders",
            "  - Non-responders",
            "Number of Cells",
            "Number of Genes",
            "Runtime (seconds)",
            "Runtime (minutes)",
            "Score Range (min)",
            "Score Range (max)",
            "Score Mean (all)",
            "Score Std (all)",
            "Score Mean (Responders)",
            "Score Mean (Non-responders)",
        ],
        "Value": [
            model_name,
            f"{auc:.4f}",
            n_patients,
            n_responders,
            n_non_responders,
            f"{n_cells:,}",
            f"{n_genes:,}",
            f"{runtime:.1f}",
            f"{runtime/60:.1f}",
            f"{score_min:.4f}",
            f"{score_max:.4f}",
            f"{score_mean:.4f}",
            f"{score_std:.4f}",
            f"{r_mean:.4f}",
            f"{nr_mean:.4f}",
        ],
    }

    df = pd.DataFrame(summary_data)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Results table saved to: {save_path}")

    return df


def generate_patient_results_table(
    patient_scores: Dict[str, float],
    patient_labels: Dict[str, int],
    save_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Generate a detailed per-patient results table.

    Parameters
    ----------
    patient_scores : dict
        Mapping of {patient_id: predicted_score}.
    patient_labels : dict
        Mapping of {patient_id: true_label}.
    save_path : str or Path, optional
        If provided, saves the table to this path as CSV.

    Returns
    -------
    pd.DataFrame
        Per-patient results DataFrame sorted by predicted score.
    """
    patients = list(patient_scores.keys())
    
    df = pd.DataFrame(
        {
            "Patient_ID": patients,
            "True_Label": [patient_labels[p] for p in patients],
            "True_Label_Name": [
                "Responder" if patient_labels[p] == 1 else "Non-responder"
                for p in patients
            ],
            "Predicted_Score": [patient_scores[p] for p in patients],
            "Predicted_Label": [
                "Responder" if patient_scores[p] >= 0.5 else "Non-responder"
                for p in patients
            ],
            "Correct": [
                (patient_scores[p] >= 0.5) == (patient_labels[p] == 1)
                for p in patients
            ],
        }
    )

    # Sort by predicted score (descending)
    df = df.sort_values("Predicted_Score", ascending=False).reset_index(drop=True)

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Patient results table saved to: {save_path}")

    return df


def evaluate_model_results(
    results_dict: Dict[str, Any],
    output_dir: Union[str, Path] = "results",
    model_name: str = "baseline",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run complete evaluation pipeline: compute metrics, generate plots, save tables.

    This is a convenience function that calls all evaluation functions and
    saves outputs to the appropriate directories.

    Parameters
    ----------
    results_dict : dict
        Results from leave_one_patient_out_cv().
    output_dir : str or Path, default="results"
        Base output directory for figures and tables.
    model_name : str, default="baseline"
        Name prefix for output files (e.g., "baseline" -> "baseline_roc.png").
    verbose : bool, default=True
        Whether to print progress information.

    Returns
    -------
    dict
        Dictionary containing:
            - auc: ROC AUC score
            - summary_df: Summary DataFrame
            - patient_df: Per-patient results DataFrame
            - roc_fig: ROC curve Figure
            - pred_fig: Patient predictions Figure
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"

    # Extract data
    patient_scores = results_dict["patient_scores"]
    patient_labels = results_dict["patient_labels"]

    # Convert to arrays for plotting
    patients = list(patient_scores.keys())
    y_true = np.array([patient_labels[p] for p in patients])
    y_scores = np.array([patient_scores[p] for p in patients])

    if verbose:
        print("=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

    # 1. Compute AUC
    auc = compute_roc_auc(y_true, y_scores)
    if verbose:
        print(f"\nROC AUC: {auc:.4f}")

    # 2. Generate ROC curve
    if verbose:
        print("\nGenerating ROC curve...")
    roc_path = figures_dir / f"{model_name}_roc.png"
    roc_fig = plot_roc_curve(
        y_true,
        y_scores,
        title=f"ROC Curve - {model_name.replace('_', ' ').title()} Model",
        save_path=roc_path,
    )

    # 3. Generate patient predictions plot
    if verbose:
        print("\nGenerating patient predictions plot...")
    pred_path = figures_dir / f"{model_name}_patient_predictions.png"
    pred_fig = plot_patient_predictions(
        patient_scores,
        patient_labels,
        save_path=pred_path,
        title=f"Patient Predictions - {model_name.replace('_', ' ').title()} Model",
    )

    # 4. Generate summary table
    if verbose:
        print("\nGenerating results summary...")
    summary_path = tables_dir / f"{model_name}_results.csv"
    summary_df = generate_results_table(
        results_dict,
        save_path=summary_path,
        model_name=model_name.replace("_", " ").title(),
    )

    # 5. Generate per-patient table
    patient_path = tables_dir / f"{model_name}_patient_results.csv"
    patient_df = generate_patient_results_table(
        patient_scores,
        patient_labels,
        save_path=patient_path,
    )

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print("=" * 60)
        print(f"\nOutputs saved to:")
        print(f"  - ROC curve: {roc_path}")
        print(f"  - Patient predictions: {pred_path}")
        print(f"  - Results summary: {summary_path}")
        print(f"  - Patient details: {patient_path}")

    return {
        "auc": auc,
        "summary_df": summary_df,
        "patient_df": patient_df,
        "roc_fig": roc_fig,
        "pred_fig": pred_fig,
    }


# ============================================================================
# Module-level script for testing
# ============================================================================

if __name__ == "__main__":
    """
    Run evaluation when module is executed directly.

    This requires that the model has been run and results are available.

    Usage:
        python -m src.evaluation

    Or from project root:
        python src/evaluation.py
    """
    import sys
    from pathlib import Path

    # Import sibling modules
    from src.preprocessing import load_preprocessed_data, DEFAULT_OUTPUT_PATH
    from src.model import leave_one_patient_out_cv

    project_root = Path(__file__).parent.parent

    print("Loading preprocessed data...")
    print("-" * 60)
    input_path = project_root / DEFAULT_OUTPUT_PATH
    adata = load_preprocessed_data(input_path)

    # Run LOO-CV
    print("\nRunning LOO-CV...")
    print("-" * 60)
    results = leave_one_patient_out_cv(adata, verbose=True)

    # Run evaluation
    print("\n")
    output_dir = project_root / "results"
    eval_results = evaluate_model_results(
        results,
        output_dir=output_dir,
        model_name="baseline",
        verbose=True,
    )

    # Acceptance criteria check
    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA CHECK (Task 3.2)")
    print("=" * 60)

    # Check 1: ROC curve saved
    roc_path = output_dir / "figures" / "baseline_roc.png"
    roc_ok = roc_path.exists()
    print(f"\n1. ROC curve saved:")
    print(f"   Path: {roc_path}")
    print(f"   Status: {'PASS' if roc_ok else 'FAIL'}")

    # Check 2: Results CSV saved
    results_path = output_dir / "tables" / "baseline_results.csv"
    results_ok = results_path.exists()
    print(f"\n2. Results table saved:")
    print(f"   Path: {results_path}")
    print(f"   Status: {'PASS' if results_ok else 'FAIL'}")

    # Check 3: AUC in expected range
    auc = eval_results["auc"]
    auc_ok = 0.75 <= auc <= 0.90
    print(f"\n3. AUC in expected range:")
    print(f"   AUC: {auc:.4f}")
    print(f"   Expected: 0.75-0.90 (paper reports ~0.84)")
    print(f"   Status: {'PASS' if auc_ok else 'NEEDS REVIEW'}")

    # Overall status
    all_pass = roc_ok and results_ok and auc_ok
    print("\n" + "=" * 60)
    print(f"OVERALL: {'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS NEED REVIEW'}")
    print("=" * 60)

    sys.exit(0 if all_pass else 1)

