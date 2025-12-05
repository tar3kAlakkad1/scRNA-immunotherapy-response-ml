#!/usr/bin/env python3
"""
Run Full PRECISE Reproduction Pipeline
========================================

This script runs the complete PRECISE reproduction pipeline from raw data
to final results, generating all figures and tables for the report/poster.

The pipeline reproduces the core PRECISE framework from Pinhasi & Yizhak (2025)
for predicting immune checkpoint inhibitor (ICI) response using single-cell 
RNA-seq data from the GSE120575 melanoma cohort.

Pipeline Steps:
    1. Load raw data from GEO files
    2. Preprocess: filter genes/cells, add response labels
    3. Train baseline XGBoost model with LOO-CV
    4. Run feature selection and evaluate improved model
    5. Analyze 11-gene signature from the paper
    6. Generate publication-quality figures and comparison tables

Usage:
    python run_pipeline.py
    python run_pipeline.py --skip-preprocessing  # Use cached preprocessed data
    python run_pipeline.py --quick               # Skip feature selection (faster)

Expected Runtime:
    - Full pipeline: ~15-25 minutes on a standard laptop
    - With --skip-preprocessing: ~10-20 minutes
    - With --quick: ~5-10 minutes

Outputs:
    - results/figures/*.png: ROC curves, importance plots, patient predictions
    - results/tables/*.csv: AUC comparisons, feature importances, gene rankings

Author: CSC 427 Final Project
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

# Add src to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import pipeline modules
from src.data_loading import load_melanoma_data, DEFAULT_EXPR_PATH, DEFAULT_PATIENT_PATH
from src.preprocessing import (
    run_preprocessing_pipeline,
    save_preprocessed_data,
    load_preprocessed_data,
    DEFAULT_OUTPUT_PATH,
)
from src.labels import add_response_labels, validate_labels
from src.model import (
    leave_one_patient_out_cv,
    evaluate_signature_genes,
    compute_feature_importances,
    PRECISE_11_GENE_SIGNATURE,
    DEFAULT_XGBOOST_PARAMS,
)
from src.evaluation import (
    compute_roc_auc,
    plot_roc_curve,
    plot_patient_predictions,
    generate_results_table,
    evaluate_model_results,
)
from src.feature_selection import (
    importance_based_selection,
    run_nested_loocv_with_selection,
    save_selected_genes,
    plot_feature_importance,
)
from src.signature_analysis import (
    check_signature_genes_presence,
    get_signature_gene_ranks,
    plot_signature_gene_expression,
    plot_signature_importance_comparison,
    plot_model_comparison,
    plot_combined_roc_curves,
    generate_final_comparison_table,
)


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_step(step_num: int, total_steps: int, description: str) -> None:
    """Print a formatted step indicator."""
    print(f"\n[Step {step_num}/{total_steps}] {description}")
    print("-" * 60)


def step1_load_data(args) -> "AnnData":
    """Step 1: Load raw data and build AnnData object."""
    print_step(1, 8, "Loading Raw Data")
    
    expr_path = PROJECT_ROOT / DEFAULT_EXPR_PATH
    patient_path = PROJECT_ROOT / DEFAULT_PATIENT_PATH
    
    print(f"Expression file: {expr_path}")
    print(f"Patient mapping: {patient_path}")
    
    adata = load_melanoma_data(
        expr_path=expr_path,
        patient_path=patient_path,
        verbose=True,
    )
    
    print(f"\n✓ Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print(f"✓ Patients: {adata.obs['patient_id'].nunique()}")
    
    return adata


def step2_preprocess(adata, args) -> "AnnData":
    """Step 2: Preprocess data - filter genes/cells, add labels."""
    print_step(2, 8, "Preprocessing Data")
    
    output_path = PROJECT_ROOT / DEFAULT_OUTPUT_PATH
    
    # Check if we can use cached preprocessed data
    if args.skip_preprocessing and output_path.exists():
        print(f"Loading cached preprocessed data from: {output_path}")
        adata = load_preprocessed_data(output_path)
        print(f"✓ Loaded preprocessed: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        return adata
    
    # Run preprocessing pipeline
    adata = run_preprocessing_pipeline(
        adata,
        min_cells_fraction=0.03,  # Genes in at least 3% of cells
        min_genes=200,            # Cells with at least 200 genes
        remove_mt_genes=True,
        normalization="none",     # Data already log-transformed
        verbose=True,
    )
    
    # Add response labels from labels module
    adata = add_response_labels(adata, verbose=True)
    
    # Validate labels
    validation = validate_labels(adata, verbose=True)
    if not validation["all_checks_passed"]:
        print("⚠ Warning: Some label validation checks failed")
    
    # Save preprocessed data
    save_preprocessed_data(adata, output_path)
    print(f"\n✓ Preprocessed data saved to: {output_path}")
    print(f"✓ Final shape: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    
    return adata


def step3_baseline_model(adata, args) -> dict:
    """Step 3: Train baseline XGBoost model with LOO-CV."""
    print_step(3, 8, "Training Baseline XGBoost Model (LOO-CV)")
    
    results = leave_one_patient_out_cv(
        adata,
        params=DEFAULT_XGBOOST_PARAMS,
        verbose=True,
        return_models=False,
    )
    
    # Evaluate and save results
    output_dir = PROJECT_ROOT / "results"
    eval_results = evaluate_model_results(
        results,
        output_dir=output_dir,
        model_name="baseline",
        verbose=True,
    )
    
    print(f"\n✓ Baseline AUC: {results['auc']:.4f}")
    print(f"✓ Paper reports: ~0.84")
    print(f"✓ Difference: {abs(results['auc'] - 0.84):.4f}")
    
    return results


def step4_feature_importances(adata, args) -> pd.DataFrame:
    """Step 4: Compute feature importances across LOO-CV folds."""
    print_step(4, 8, "Computing Feature Importances")
    
    importance_df = compute_feature_importances(
        adata,
        params=DEFAULT_XGBOOST_PARAMS,
        importance_type="gain",
        verbose=True,
    )
    
    # Add rank column
    importance_df["rank"] = range(1, len(importance_df) + 1)
    
    # Save feature importances
    output_path = PROJECT_ROOT / "results" / "tables" / "feature_importances.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(output_path, index=False)
    print(f"\n✓ Feature importances saved to: {output_path}")
    
    # Generate importance plot
    figures_dir = PROJECT_ROOT / "results" / "figures"
    plot_feature_importance(
        importance_df,
        top_n=30,
        save_path=figures_dir / "feature_importance_top30.png",
    )
    
    return importance_df


def step5_feature_selection(adata, importance_df, args) -> dict:
    """Step 5: Run feature selection and evaluate improved model."""
    print_step(5, 8, "Feature Selection & Improved Model")
    
    if args.quick:
        print("⚡ Quick mode: Skipping nested CV feature selection")
        print("   Using importance-based selection instead...")
        
        # Simple importance-based selection
        top_n = 50
        selected_genes = importance_df.head(top_n)["gene"].tolist()
        
        # Subset data and run LOO-CV
        adata_subset = adata[:, selected_genes].copy()
        results = leave_one_patient_out_cv(
            adata_subset,
            params=DEFAULT_XGBOOST_PARAMS,
            verbose=True,
        )
        
        # Save selected genes
        tables_dir = PROJECT_ROOT / "results" / "tables"
        selected_df = pd.DataFrame({"gene": selected_genes})
        selected_df.to_csv(tables_dir / "selected_genes.csv", index=False)
        
    else:
        # Full nested LOO-CV with feature selection (prevents data leakage)
        print("Running nested LOO-CV with feature selection...")
        print("(This may take 10-15 minutes)\n")
        
        results = run_nested_loocv_with_selection(
            adata,
            top_n=50,
            params=DEFAULT_XGBOOST_PARAMS,
            verbose=True,
        )
        selected_genes = results.get("selected_genes", [])
    
    # Evaluate and save results
    output_dir = PROJECT_ROOT / "results"
    eval_results = evaluate_model_results(
        results,
        output_dir=output_dir,
        model_name="feature_selection",
        verbose=True,
    )
    
    print(f"\n✓ Feature Selection AUC: {results['auc']:.4f}")
    print(f"✓ Paper reports: ~0.89 (with Boruta)")
    print(f"✓ Genes selected: {len(selected_genes)}")
    
    return results


def step6_signature_analysis(adata, importance_df, args) -> dict:
    """Step 6: Analyze the 11-gene signature from the PRECISE paper."""
    print_step(6, 8, "11-Gene Signature Analysis")
    
    output_dir = PROJECT_ROOT / "results"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    
    # Check signature gene presence
    presence_info = check_signature_genes_presence(adata, verbose=True)
    
    # Get signature gene ranks
    ranks_df = get_signature_gene_ranks(importance_df, verbose=True)
    ranks_df.to_csv(tables_dir / "signature_gene_ranks.csv", index=False)
    
    # Train 11-gene-only model
    print("\nTraining 11-gene-only model...")
    signature_results = evaluate_signature_genes(
        adata,
        PRECISE_11_GENE_SIGNATURE,
        params=DEFAULT_XGBOOST_PARAMS,
        verbose=True,
    )
    
    # Generate signature expression plot
    print("\nGenerating signature expression plot...")
    try:
        plot_signature_gene_expression(
            adata,
            save_path=figures_dir / "signature_genes_expression.png",
        )
    except Exception as e:
        print(f"  Warning: Could not create expression plot: {e}")
    
    # Generate signature importance comparison plot
    print("Generating importance comparison plot...")
    plot_signature_importance_comparison(
        importance_df,
        top_n=50,
        save_path=figures_dir / "signature_importance_comparison.png",
    )
    
    # Generate signature ROC curve
    patients = list(signature_results["patient_scores"].keys())
    y_true = np.array([signature_results["patient_labels"][p] for p in patients])
    y_scores = np.array([signature_results["patient_scores"][p] for p in patients])
    
    plot_roc_curve(
        y_true,
        y_scores,
        title="ROC Curve - 11-Gene Signature Model",
        save_path=figures_dir / "signature_roc.png",
    )
    
    print(f"\n✓ 11-Gene Signature AUC: {signature_results['auc']:.4f}")
    print(f"✓ Paper reports: ~0.85 (estimated)")
    print(f"✓ Signature genes in our top 50: {ranks_df['in_top_50'].sum()}/11")
    
    return signature_results


def step7_generate_comparison(baseline_results, fs_results, sig_results, adata, args) -> None:
    """Step 7: Generate final comparison table and combined plots."""
    print_step(7, 8, "Generating Final Comparison Table & Plots")
    
    output_dir = PROJECT_ROOT / "results"
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    
    # Generate final comparison table
    comparison_df = generate_final_comparison_table(
        baseline_auc=baseline_results["auc"],
        feature_selected_auc=fs_results["auc"],
        signature_auc=sig_results["auc"],
        baseline_n_genes=adata.n_vars,
        feature_selected_n_genes=fs_results.get("n_genes", 50),
        save_path=tables_dir / "final_comparison.csv",
    )
    
    print("\nFinal Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Generate model comparison bar plot
    model_aucs = {
        "Baseline\n(All Genes)": baseline_results["auc"],
        "Feature\nSelected": fs_results["auc"],
        "11-Gene\nSignature": sig_results["auc"],
    }
    paper_aucs = {
        "Baseline\n(All Genes)": 0.84,
        "Feature\nSelected": 0.89,
        "11-Gene\nSignature": 0.85,
    }
    
    plot_model_comparison(
        model_aucs,
        paper_aucs=paper_aucs,
        save_path=figures_dir / "model_auc_comparison.png",
    )
    
    # Generate combined ROC curves
    # Compute ROC curves for each model
    def get_roc_data(results):
        patients = list(results["patient_scores"].keys())
        y_true = np.array([results["patient_labels"][p] for p in patients])
        y_scores = np.array([results["patient_scores"][p] for p in patients])
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        return {"fpr": fpr, "tpr": tpr, "auc": results["auc"]}
    
    roc_data = {
        "Baseline (All Genes)": get_roc_data(baseline_results),
        "Feature Selected": get_roc_data(fs_results),
        "11-Gene Signature": get_roc_data(sig_results),
    }
    
    plot_combined_roc_curves(
        roc_data,
        save_path=figures_dir / "combined_roc_comparison.png",
    )
    
    print(f"\n✓ Comparison table saved to: {tables_dir / 'final_comparison.csv'}")
    print(f"✓ Model comparison plot saved to: {figures_dir / 'model_auc_comparison.png'}")
    print(f"✓ Combined ROC curves saved to: {figures_dir / 'combined_roc_comparison.png'}")


def step8_summary(baseline_results, fs_results, sig_results, total_time, args) -> None:
    """Step 8: Print final summary and acceptance criteria check."""
    print_step(8, 8, "Final Summary")
    
    print_header("PIPELINE COMPLETE")
    
    print("\n📊 MODEL PERFORMANCE:")
    print(f"   {'Model':<30} {'Our AUC':>10} {'Paper AUC':>12}")
    print(f"   {'-'*54}")
    print(f"   {'Baseline XGBoost (All Genes)':<30} {baseline_results['auc']:>10.3f} {'~0.84':>12}")
    print(f"   {'Feature Selected':<30} {fs_results['auc']:>10.3f} {'~0.89':>12}")
    print(f"   {'11-Gene Signature':<30} {sig_results['auc']:>10.3f} {'~0.85':>12}")
    
    print(f"\n⏱  Total Runtime: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n📁 OUTPUT FILES:")
    print("   Figures:")
    print("     - results/figures/baseline_roc.png")
    print("     - results/figures/feature_selection_roc.png")
    print("     - results/figures/signature_roc.png")
    print("     - results/figures/combined_roc_comparison.png")
    print("     - results/figures/model_auc_comparison.png")
    print("     - results/figures/signature_genes_expression.png")
    print("     - results/figures/signature_importance_comparison.png")
    print("   Tables:")
    print("     - results/tables/final_comparison.csv")
    print("     - results/tables/feature_importances.csv")
    print("     - results/tables/signature_gene_ranks.csv")
    
    print("\n✅ ACCEPTANCE CRITERIA:")
    checks = []
    
    # Check 1: Baseline AUC in expected range
    auc_range = 0.65 <= baseline_results["auc"] <= 0.95
    checks.append(("Baseline AUC in reasonable range (0.65-0.95)", auc_range))
    
    # Check 2: Pipeline completes in reasonable time (<30 min)
    time_ok = total_time < 1800
    checks.append(("Runtime < 30 minutes", time_ok))
    
    # Check 3: All outputs generated
    output_files = [
        PROJECT_ROOT / "results" / "figures" / "combined_roc_comparison.png",
        PROJECT_ROOT / "results" / "tables" / "final_comparison.csv",
    ]
    outputs_ok = all(f.exists() for f in output_files)
    checks.append(("All required outputs generated", outputs_ok))
    
    for desc, passed in checks:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"   {status}: {desc}")
    
    all_passed = all(passed for _, passed in checks)
    print(f"\n{'='*70}")
    print(f" {'ALL CHECKS PASSED ✓' if all_passed else 'SOME CHECKS FAILED ✗'}")
    print(f"{'='*70}")
    
    return all_passed


def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the PRECISE reproduction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_pipeline.py                     # Full pipeline
    python run_pipeline.py --skip-preprocessing  # Skip data loading/preprocessing
    python run_pipeline.py --quick             # Fast mode (skip nested CV)
        """
    )
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip preprocessing and use cached data if available",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: skip slow nested CV feature selection",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print detailed progress information",
    )
    
    args = parser.parse_args()
    
    print_header("PRECISE REPRODUCTION PIPELINE")
    print("""
Reproducing: Pinhasi & Yizhak (2025)
Paper: "Uncovering gene and cellular signatures of immune checkpoint 
       response via machine learning and single-cell RNA-seq"
Dataset: GSE120575 Melanoma Cohort (~16k cells, 48 patients)
    """)
    
    start_time = time.time()
    
    try:
        # Step 1: Load data (or skip if using cached)
        if args.skip_preprocessing:
            print_step(1, 8, "Loading Preprocessed Data (skip_preprocessing=True)")
            adata = load_preprocessed_data(PROJECT_ROOT / DEFAULT_OUTPUT_PATH)
            print(f"✓ Loaded: {adata.n_obs:,} cells x {adata.n_vars:,} genes")
        else:
            adata = step1_load_data(args)
            adata = step2_preprocess(adata, args)
        
        # Step 3: Baseline model
        baseline_results = step3_baseline_model(adata, args)
        
        # Step 4: Feature importances
        importance_df = step4_feature_importances(adata, args)
        
        # Step 5: Feature selection
        fs_results = step5_feature_selection(adata, importance_df, args)
        
        # Step 6: Signature analysis
        sig_results = step6_signature_analysis(adata, importance_df, args)
        
        # Step 7: Generate comparison
        step7_generate_comparison(baseline_results, fs_results, sig_results, adata, args)
        
        # Step 8: Summary
        total_time = time.time() - start_time
        all_passed = step8_summary(baseline_results, fs_results, sig_results, total_time, args)
        
        return 0 if all_passed else 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())


