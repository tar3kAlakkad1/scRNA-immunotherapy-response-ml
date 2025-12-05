# PRECISE Reproduction Summary

## Brief overview of what was implemented
- Reproduced the core PRECISE pipeline on the GSE120575 melanoma cohort (16,290 cells; 12,785 genes after QC; 48 patients with 17 R / 31 NR).
- Preprocessing: filtered genes expressed in <3% of cells, dropped cells with <200 detected genes, removed mitochondrial genes, kept log-transformed values, and saved an `AnnData` object for reuse.
- Labels: attached patient-level responder/non-responder labels to every cell and validated class balance.
- Modeling: baseline XGBoost with leave-one-patient-out CV; patient scores are mean per-cell probabilities. Generated ROC curves and per-patient predictions.
- Feature selection: importance-based gene ranking per fold; nested LOO-CV with top 50 genes to avoid leakage (quick mode uses a non-nested shortcut).
- Signature analysis: verified the paper’s 11-gene panel, ranked those genes, and trained an 11-gene-only model; produced comparison plots and tables.
- Stretch work: simple cell-filtration experiment (confidence-based) and a small external test on a basal cell carcinoma (BCC) cohort.

## Results achieved in this project
- Baseline LOO-CV (all genes): AUC 0.7704; runtime ~25.1 min; score range 0.034–0.675; 48/48 folds completed.
- Feature selection (nested, top 50 genes): AUC ≈0.772; negligible gain over baseline but avoids leakage. A quick/non-nested run reached ~0.913 AUC (likely optimistic); final comparison uses the nested number.
- 11-gene signature model: AUC 0.909; 10/11 signature genes rank in our top 50 (HLA-G ranks 142).
- Cell filtration trial: filtering low-confidence cells did not help (10% filtered AUC 0.769; 20% filtered AUC 0.748).
- External BCC cohort (11 patients, 21,328 cells): 11-gene signature AUC 0.40 — signature did not transfer.
- Outputs: figures for ROC/model comparisons and importance plots; tables for per-patient scores, AUCs, selected genes, signature ranks, and the final comparison.

## What the paper reports
- Melanoma training cohort: base XGBoost AUC ~0.84; Boruta feature selection improved to ~0.89; derived an 11-gene signature.
- Signature generalization: strong AUCs across multiple cancers (e.g., TNBC ~0.95, NSCLC high-0.8s, glioblastoma/BCC ~0.68 pre-filtration) with most cohorts improving after their reinforcement-learning–guided cell filtration.
- Cell filtration via RL: removing ~40% “non-predictive” cells raised melanoma AUC to >0.94 and boosted most validation cohorts (BCC to ~0.70; TNBC to ~0.98; one NSCLC cohort dropped).
- Interpretability: SHAP analyses highlighted non-linear gene–gene interactions and confirmed the robustness of the 11-gene panel.

## Comparison to the paper
- Our baseline AUC (0.770) trails the paper’s 0.84, likely due to slightly different QC thresholds, hyperparameters, and no cell-type–specific training.
- Feature selection closed almost none of that gap when run with nested CV (0.772 vs paper’s 0.89); the higher 0.913 quick run is not trusted because it can leak information across folds.
- The 11-gene signature performed strongly here (0.909) and even exceeded the paper’s reported ~0.85 on melanoma, but it failed to generalize to BCC (paper saw ~0.68→0.70 after filtration).

| Model / Setting | Our AUC | Paper AUC | Notes |
| --- | --- | --- | --- |
| Baseline XGBoost (all genes) | 0.770 | 0.84 | Same LOO-CV design; differences likely from preprocessing/hyperparameters |
| Feature-selected (nested, 50 genes) | 0.772 | 0.89 | Uses per-fold top genes; quick/non-nested run hit ~0.913 but is optimistic |
| 11-gene signature (melanoma) | 0.909 | ~0.85 | 10/11 genes in our top 50; strong on training cohort |
| 11-gene signature (BCC external) | 0.40 | ~0.68 (→0.70 after RL filtration) | Indicates limited cross-cohort transfer without RL filtration |

## Evaluation and must-fixes
- Overall: Pipeline is complete end-to-end with clear outputs and strong signature performance, but baseline and feature-selected models lag paper targets and cross-cohort generalization is weak.
- Must-fix #1: Reconcile feature-selection reporting — `feature_selection_results.csv` shows 0.913 (quick run) while `final_comparison.csv` uses 0.772 (nested). Document the canonical number and regenerate tables/figures consistently.
- Must-fix #2: Investigate baseline underperformance (0.770 vs 0.84) by aligning preprocessing thresholds (e.g., gene filtering, normalization), tuning XGBoost (depth/learning rate/scale_pos_weight), and confirming label mapping.
- Must-fix #3: External validation and cell filtration need cleanup — re-run BCC with standardized normalization and consider a closer proxy to the paper’s RL filtration to test transferability.
- Nice-to-have: Record seeds/runtimes in the tables, and optionally add SHAP/interpretability outputs to the summary for transparency.

