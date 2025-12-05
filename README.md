# scRNA-immunotherapy-response-ml

## Overview
Reproduction of the PRECISE framework (Pinhasi & Yizhak, 2025) to predict immune checkpoint inhibitor (ICI) response from single-cell RNA-seq of melanoma-infiltrating immune cells. The pipeline runs end-to-end: data loading, preprocessing, label attachment, LOO-CV modeling, feature selection, 11-gene signature analysis, and figure/table generation.

## Paper at a Glance (Original Research Summary)
- PRECISE uses XGBoost on per-cell expression, aggregates to patient scores via leave-one-patient-out CV, and reports AUC ~0.84 on melanoma.
- Boruta feature selection boosts performance to ~0.89.
- Derives an 11-gene signature (GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5) that generalizes across multiple cancer cohorts; RL-based cell filtration further improves AUCs (>0.94 on melanoma).

## Data & Labels
- Primary dataset: GSE120575 (Sade-Feldman melanoma immune cells), 16,290 cells x 12,785 genes post-QC, 48 patients (17 responders, 31 non-responders).
- Inputs stored in `data/raw/` (TPM expression + cell→patient mapping); processed AnnData cached at `data/processed/melanoma_adata.h5ad`.
- Labels are patient-level response mapped onto all cells; validated distribution matches the paper.

## Implemented Pipeline
- Preprocessing: filter genes expressed in <3% of cells; drop cells with <200 genes; remove mitochondrial genes; keep log-transformed values; store AnnData.
- Modeling: XGBoost classifier with leave-one-patient-out CV; patient score = mean per-cell probability.
- Feature selection: importance-based gene ranking; nested LOO-CV using top 50 genes to avoid leakage (quick mode uses a non-nested shortcut).
- Signature analysis: verify 11-gene presence, rank them, and train an 11-gene-only model; generate comparison plots/tables.
- Stretch experiments: simple confidence-based cell filtration and a small external BCC cohort test.

## Results

| Model / Setting | Our AUC | Paper AUC | Notes |
| --- | --- | --- | --- |
| Baseline XGBoost (all genes) | 0.770 | 0.84 | LOO-CV, full gene set after QC |
| Feature-selected (nested, top 50 genes) | 0.772 | 0.89 | Nested LOO to prevent leakage (quick/non-nested run ~0.913 is optimistic) |
| 11-gene signature (melanoma) | 0.909 | ~0.85 | 10/11 signature genes rank in our top 50 |
| 11-gene signature (BCC external) | 0.40 | ~0.68→0.70 after RL filtration | Limited transfer without RL-style filtration |

Figures and tables are under `results/figures/` and `results/tables/` (e.g., `final_comparison.csv`, ROC curves, importance plots, signature ranks).

## How to run
Before getting started, make sure you follow the setup steps below. 

### Setup
```bash
conda create -n csc427-project python=3.11 -y
conda activate csc427-project
pip install -r requirements.txt
```

### Running the full pipeline (from repo root):
```bash
python run_pipeline.py
```

Flags:
- `--skip-preprocessing` use cached `data/processed/melanoma_adata.h5ad` if present.
- `--quick` skip nested feature selection (faster, but less conservative).

Notes:
- Full run can take on the order of tens of minutes depending on hardware; `--quick` is suitable for fast iteration.
- Outputs land in `results/figures/` and `results/tables/`; check the console summary for AUCs and file paths.
