# PLAN.md – CSC 427 Final Project Implementation Plan

## Project Summary

This project reproduces the **PRECISE framework** from Pinhasi & Yizhak (2025) for predicting immune checkpoint inhibitor (ICI) response using single-cell RNA-seq data. The core idea is to train an **XGBoost classifier** on per-cell gene expression from tumor-infiltrating immune cells, then aggregate cell-level predictions into patient-level response scores. We use **leave-one-patient-out cross-validation (LOO-CV)** to evaluate performance.

Our primary focus is the **GSE120575 melanoma cohort** (~16k cells from 48 patients). The main deliverables are: (1) a baseline XGBoost model achieving ~0.84 AUC, (2) an improved model with feature selection achieving ~0.89 AUC, and (3) comparison of our feature importance rankings to the paper's **11-gene signature** (`GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5`).

---

## Goals & Success Criteria

### Primary Goals (Required for Course)

| Goal | Target Metric | Acceptance Criteria |
|------|---------------|---------------------|
| Load and preprocess GSE120575 | N/A | Both files load; cell IDs match; ~16k cells, 48 patients |
| Baseline XGBoost LOO-CV | AUC ~0.84 | AUC between 0.75–0.90; ROC curve plotted |
| Feature selection model | AUC ~0.89 | AUC improves over baseline; selected genes logged |
| 11-gene signature analysis | Gene rankings | All 11 genes present in data; importance ranks computed |
| Reproducibility | End-to-end | Pipeline runs from raw data to final AUC in <30 min on laptop |

### Stretch Goals (Optional)

| Goal | Description |
|------|-------------|
| 11-gene-only model | Train XGBoost using only the 11 signature genes; compare AUC |
| Simplified cell filtration | Remove "non-predictive" cells based on prediction confidence; re-evaluate AUC |
| External cohort validation | Apply 11-gene signature to one additional dataset (e.g., TNBC) |

---

## Repo & File Organization Plan

### Target Directory Structure

```
scRNA-immunotherapy-response-ml/
├── data/
│   ├── raw/                              # Original GEO files (gitignored)
│   │   ├── GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz
│   │   └── GSE120575_patient_ID_single_cells.txt.gz
│   └── processed/                        # Processed outputs (gitignored)
│       └── melanoma_adata.h5ad           # Final AnnData object
│
├── src/                                  # Python modules
│   ├── __init__.py
│   ├── data_loading.py                   # Load raw files, build AnnData
│   ├── preprocessing.py                  # Filter genes/cells, normalize
│   ├── labels.py                         # Patient ID mapping & response labels
│   ├── model.py                          # XGBoost training, LOO-CV logic
│   ├── feature_selection.py              # Boruta or importance-based selection
│   ├── evaluation.py                     # Metrics, ROC curves, plots
│   └── utils.py                          # Shared helpers (logging, seeds, etc.)
│
├── notebooks/                            # Jupyter notebooks for exploration
│   ├── 01_explore_data.ipynb             # Initial data exploration
│   ├── 02_preprocessing.ipynb            # Preprocessing walkthrough
│   ├── 03_baseline_model.ipynb           # Baseline XGBoost + LOO-CV
│   ├── 04_feature_selection.ipynb        # Feature selection experiments
│   ├── 05_signature_analysis.ipynb       # 11-gene signature analysis
│   └── 06_stretch_experiments.ipynb      # Optional: cell filtration, external data
│
├── results/                              # Output artifacts
│   ├── figures/                          # PNG/PDF plots for report/poster
│   ├── tables/                           # CSV outputs (AUCs, gene rankings)
│   └── logs/                             # Training logs, hyperparameters
│
├── tests/                                # Unit tests (optional but encouraged)
│   └── test_data_loading.py
│
├── environment.yml                       # Conda environment spec
├── requirements.txt                      # Pip fallback
├── PLAN.md                               # This file
├── README.md                             # Project overview
├── selected-topic-overview.md            # Paper summary
├── piazza-requirments.md                 # Course requirements
└── report.md                             # Final report (written manually, no LLM)
```

### .gitignore Additions

Create/update `.gitignore` to exclude large data files:

```gitignore
# Data files (too large for git)
data/raw/*.gz
data/raw/*.txt
data/processed/*.h5ad
data/processed/*.pkl

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Environment
.env
*.egg-info/

# OS
.DS_Store
```

---

## Environment & Dependencies

### Required Packages

| Package | Purpose | Min Version |
|---------|---------|-------------|
| `scanpy` | scRNA-seq data handling, AnnData | 1.9+ |
| `anndata` | Data structure for expression + metadata | 0.9+ |
| `pandas` | Data wrangling | 2.0+ |
| `numpy` | Numerical operations | 1.24+ |
| `xgboost` | Main classifier | 2.0+ |
| `scikit-learn` | Metrics, CV utilities | 1.3+ |
| `matplotlib` | Plotting | 3.7+ |
| `seaborn` | Statistical visualizations | 0.12+ |
| `boruta` | Feature selection (via `boruta_py`) | 0.3+ |
| `shap` | Model interpretability (optional) | 0.42+ |
| `tqdm` | Progress bars | 4.65+ |
| `jupyter` | Notebooks | latest |

### Setup Commands

```bash
# Option 1: Conda (recommended)
conda create -n csc427-project python=3.11 -y
conda activate csc427-project
pip install scanpy anndata pandas numpy xgboost scikit-learn matplotlib seaborn boruta tqdm shap jupyter

# Option 2: Pip only
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Acceptance Criteria (Environment)

- [x] `python -c "import scanpy, xgboost, sklearn; print('OK')"` runs without error
- [x] Jupyter notebook kernel uses the correct environment

---

## Phase 1: Data Loading & Preprocessing

### Task 1.1: Organize Raw Data

**Goal:** Move downloaded files into `data/raw/` and verify integrity.

**Steps:**
1. Create `data/raw/` directory
2. Move the two `.txt.gz` files into it
3. Verify files can be decompressed

**Acceptance Criteria:**
- [x] `data/raw/GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz` exists
- [x] `data/raw/GSE120575_patient_ID_single_cells.txt.gz` exists
- [x] `zcat data/raw/*.gz | head` shows valid TSV content

---

### Task 1.2: Implement `src/data_loading.py`

**Goal:** Load the expression matrix and patient ID mapping into Python.

**Module Responsibilities:**
- `load_expression_matrix(path) -> pd.DataFrame`: Load the TPM matrix (genes × cells)
- `load_patient_mapping(path) -> pd.DataFrame`: Load cell-to-patient mapping
- `build_anndata(expr_df, patient_df) -> AnnData`: Combine into AnnData object

**Expected Data Format:**
- Expression file: First column is gene names; remaining columns are cell barcodes; values are TPM
- Patient file: Two columns – `cell_id` and `patient_id` (or `sample_id`)

**Output:** Raw AnnData with:
- `adata.X` = expression matrix (cells × genes, transposed from file)
- `adata.obs` = metadata per cell (at minimum: `cell_id`, `patient_id`)
- `adata.var` = gene-level metadata (gene names as index)

**Acceptance Criteria:**
- [x] `adata.shape` is approximately (16000, 17000) – ~16k cells, ~17k genes
- [x] `adata.obs['patient_id']` has ~48 unique values
- [x] No NaN values in expression matrix (or handled appropriately)

---

### Task 1.3: Implement `src/preprocessing.py`

**Goal:** Filter genes and cells, normalize expression values.

**Module Responsibilities:**
- `filter_genes(adata, min_cells=...) -> AnnData`: Remove genes expressed in <N cells
- `filter_cells(adata, min_genes=...) -> AnnData`: Remove cells with <N genes expressed
- `normalize_expression(adata) -> AnnData`: Apply log1p(TPM + 1) or similar
- `run_preprocessing_pipeline(adata) -> AnnData`: Orchestrate all steps

**Paper-Specific Details:**
- The paper mentions keeping genes expressed in at least a small percentage of cells (exact threshold unclear; start with 1% of cells, ~160 cells)
- Consider removing mitochondrial genes (prefix `MT-`) if they cause noise
- Standard scRNA-seq practice: `scanpy.pp.normalize_total()` followed by `scanpy.pp.log1p()`

**Output:** Save processed AnnData to `data/processed/melanoma_adata.h5ad`

**Acceptance Criteria:**
- [x] Gene count reduced from ~17k to ~8k–12k after filtering
- [x] Expression values are log-transformed (check `adata.X.max()` is reasonable, e.g., <20)
- [x] File saved and can be reloaded: `adata = sc.read_h5ad('data/processed/melanoma_adata.h5ad')`

---

### Task 1.4: Create `notebooks/01_explore_data.ipynb`

**Goal:** Explore the raw data interactively before building the full pipeline.

**Notebook Contents:**
1. Load both raw files
2. Print shapes, column names, sample values
3. Check for missing values
4. Visualize distribution of genes per cell, cells per gene
5. List unique patient IDs
6. Save exploratory plots to `results/figures/`

**Output:** Understanding of data structure; any data quality issues identified

---

### Task 1.5: Create `notebooks/02_preprocessing.ipynb`

**Goal:** Walk through preprocessing steps interactively.

**Notebook Contents:**
1. Load raw data
2. Apply filtering step-by-step, visualizing effect on dimensions
3. Apply normalization
4. Create UMAP/PCA visualization (optional but useful)
5. Save final processed AnnData

**Output:** `data/processed/melanoma_adata.h5ad` ready for modeling

---

## Phase 2: Label Construction

### Task 2.1: Implement `src/labels.py`

**Goal:** Map patient IDs to ICI response labels (Responder vs Non-Responder).

**Module Responsibilities:**
- `get_response_labels() -> dict`: Return mapping `{patient_id: 'R' or 'NR'}`
- `add_response_labels(adata) -> AnnData`: Add `response` column to `adata.obs`
- `get_patient_metadata(adata) -> pd.DataFrame`: Summary of patients, cell counts, labels

**Label Source:**
The GSE120575 patient ID file or the original Sade-Feldman paper supplementary tables contain response labels. If not directly in the file, labels can be reconstructed from:
- The PRECISE paper's supplementary data
- The GEO series metadata (GSE120575)
- Sade-Feldman et al. (2018) supplementary tables

**Hardcoded Fallback:**
If labels cannot be parsed automatically, create a manual mapping based on the paper. The paper mentions 17 responders and 31 non-responders (48 total patients).

**Output:** AnnData with:
- `adata.obs['response']` = 'R' or 'NR' for each cell
- `adata.obs['response_binary']` = 1 (R) or 0 (NR) for XGBoost

**Acceptance Criteria:**
- [x] All 48 patients have a response label
- [x] ~17 responders, ~31 non-responders (or close to paper's split)
- [x] Every cell has a non-null `response` value

---

### Task 2.2: Verify Label Distribution

**Goal:** Confirm label balance matches the paper.

**Checks:**
1. Count patients per response class
2. Count cells per response class
3. Visualize class imbalance

**Expected:**
- ~17 patients labeled 'R' (responder)
- ~31 patients labeled 'NR' (non-responder)
- Cell counts may be imbalanced (some patients have more cells)

**Acceptance Criteria:**
- [x] Patient-level distribution verified: 17 Responders (35.4%), 31 Non-responders (64.6%)
- [x] Cell-level distribution computed: 5,564 Responder cells (34.2%), 10,726 Non-responder cells (65.8%)
- [x] Class imbalance visualized in `results/figures/label_distribution_verification.png`
- [x] Therapy distribution analyzed in `results/figures/label_therapy_distribution.png`
- [x] All validation checks passed (via `labels.validate_labels()`)

**Notebook:** `notebooks/03_label_verification.ipynb`

---

## Phase 3: Baseline XGBoost Model (LOO-CV)

### Task 3.1: Implement `src/model.py`

**Goal:** Implement the core XGBoost training and LOO-CV logic.

**Module Responsibilities:**

```python
def train_xgboost(X_train, y_train, params=None) -> xgb.Booster:
    """Train XGBoost classifier on cell-level data."""
    pass

def predict_cells(model, X_test) -> np.ndarray:
    """Predict per-cell probabilities of being 'responder'."""
    pass

def aggregate_to_patient(cell_probs, cell_patient_ids) -> dict:
    """Aggregate cell predictions to patient-level score (mean probability)."""
    pass

def leave_one_patient_out_cv(adata, params=None) -> dict:
    """
    Run LOO-CV over all patients.
    Returns dict with:
      - patient_scores: {patient_id: predicted_score}
      - patient_labels: {patient_id: true_label}
      - fold_models: list of trained models (optional)
    """
    pass
```

**XGBoost Hyperparameters (starting point):**
Based on typical scRNA-seq settings and the paper's description:
```python
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 4,           # Paper uses relatively shallow trees
    'learning_rate': 0.1,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'n_jobs': -1,
}
```

**LOO-CV Logic:**
```
for each patient p in 48 patients:
    1. Split: test_cells = cells from patient p
             train_cells = cells from all other patients
    2. Train XGBoost on train_cells (X=expression, y=response_binary)
    3. Predict probabilities for test_cells
    4. Aggregate: patient_score[p] = mean(test_cell_probabilities)
    5. Store true label: patient_label[p] = response of patient p
    
Compute ROC AUC using patient_scores vs patient_labels (48 data points)
```

**Acceptance Criteria:**
- [ ] LOO-CV completes for all 48 patients
- [ ] Runtime is reasonable (<15 min on laptop for full CV)
- [ ] Patient-level scores are between 0 and 1

---

### Task 3.2: Implement `src/evaluation.py`

**Goal:** Compute metrics and generate evaluation plots.

**Module Responsibilities:**

```python
def compute_roc_auc(y_true, y_scores) -> float:
    """Compute ROC AUC from true labels and predicted scores."""
    pass

def plot_roc_curve(y_true, y_scores, title, save_path) -> None:
    """Plot and save ROC curve."""
    pass

def plot_patient_predictions(patient_scores, patient_labels, save_path) -> None:
    """Scatter plot of patient scores colored by true label."""
    pass

def generate_results_table(results_dict, save_path) -> pd.DataFrame:
    """Save results summary as CSV."""
    pass
```

**Acceptance Criteria:**
- [ ] ROC curve saved to `results/figures/baseline_roc.png`
- [ ] AUC printed and saved to `results/tables/baseline_results.csv`
- [ ] Target: AUC between 0.75–0.90 (paper reports ~0.84)

---

### Task 3.3: Create `notebooks/03_baseline_model.ipynb`

**Goal:** Run and visualize the baseline model interactively.

**Notebook Contents:**
1. Load processed AnnData with labels
2. Run LOO-CV using `src/model.py` functions
3. Compute and display AUC
4. Plot ROC curve
5. Visualize patient-level predictions
6. Save all outputs

**Key Output:**
- **Baseline AUC:** Document the achieved AUC
- **Comparison to paper:** Paper reports ~0.84; note any difference

---

## Phase 4: Feature Selection & Improved Model

### Task 4.1: Implement `src/feature_selection.py`

**Goal:** Select predictive genes using Boruta or importance-based methods.

**Module Responsibilities:**

```python
def run_boruta_selection(X, y, random_state=42) -> list:
    """
    Run Boruta feature selection.
    Returns list of selected gene indices or names.
    """
    pass

def importance_based_selection(model, gene_names, top_n=500) -> list:
    """
    Select top N genes by XGBoost feature importance.
    Alternative to Boruta if computational constraints.
    """
    pass

def get_feature_importance_df(models, gene_names) -> pd.DataFrame:
    """
    Aggregate feature importances across LOO-CV folds.
    Returns DataFrame with columns: gene, mean_importance, std_importance
    """
    pass
```

**Boruta Notes:**
- Boruta can be slow on high-dimensional data (~10k genes)
- Alternative: Use XGBoost's built-in feature importance from baseline model
- Paper uses Boruta, but importance-based selection is acceptable for reproduction

**Strategy:**
1. Run baseline LOO-CV and collect feature importances from each fold
2. Average importances across folds
3. Select top N genes (try N=500, 1000, or Boruta's selection)
4. Re-run LOO-CV using only selected genes

**Acceptance Criteria:**
- [ ] Feature selection reduces gene count significantly (e.g., 10k → 500–2000)
- [ ] Selected genes list is saved to `results/tables/selected_genes.csv`

---

### Task 4.2: Re-run LOO-CV with Selected Features

**Goal:** Evaluate improved model with feature selection.

**Steps:**
1. Subset AnnData to selected genes only
2. Re-run `leave_one_patient_out_cv()` with same hyperparameters
3. Compute new AUC

**Target:** AUC ~0.89 (paper's result with Boruta)

**Acceptance Criteria:**
- [ ] AUC improves over baseline (or close to)
- [ ] Results saved to `results/tables/feature_selection_results.csv`
- [ ] ROC curve saved to `results/figures/feature_selection_roc.png`

---

### Task 4.3: Create `notebooks/04_feature_selection.ipynb`

**Goal:** Interactively explore feature selection.

**Notebook Contents:**
1. Load baseline model results and feature importances
2. Run Boruta (or importance-based selection)
3. Visualize top genes (bar plot of importances)
4. Re-run LOO-CV with selected genes
5. Compare AUCs: baseline vs. feature-selected
6. Save outputs

---

## Phase 5: 11-Gene Signature Analysis

### Task 5.1: Check Presence of 11 Signature Genes

**Goal:** Verify all 11 genes from the paper's signature are in our dataset.

**The 11-Gene Signature:**
```python
SIGNATURE_GENES = [
    'GAPDH', 'CD38', 'CCR7', 'HLA-DRB5', 'STAT1',
    'GZMH', 'LGALS1', 'IFI6', 'EPSTI1', 'HLA-G', 'GBP5'
]
```

**Checks:**
1. For each gene, check if it's in `adata.var_names`
2. If missing, check for aliases or slightly different naming
3. Document which genes are present/missing

**Acceptance Criteria:**
- [ ] All 11 genes present (or document which are missing and why)
- [ ] Gene expression distributions visualized (violin plots)

---

### Task 5.2: Rank Signature Genes by Importance

**Goal:** See where the 11 signature genes rank in our feature importance analysis.

**Steps:**
1. Load feature importance DataFrame from Phase 4
2. Find rank of each signature gene
3. Create summary table: gene, our_rank, paper_mentioned
4. Visualize: highlight signature genes in importance bar plot

**Output:** `results/tables/signature_gene_ranks.csv`

**Acceptance Criteria:**
- [ ] Ranks computed for all 11 genes (or noted as missing)
- [ ] Discussion point: Do our top genes overlap with the signature?

---

### Task 5.3: Train 11-Gene-Only Model (Optional)

**Goal:** Train XGBoost using ONLY the 11 signature genes.

**Steps:**
1. Subset AnnData to only the 11 genes
2. Run LOO-CV
3. Compute AUC

**This tests:** Whether the 11-gene signature alone is sufficient for good predictions.

**Acceptance Criteria:**
- [ ] AUC computed for 11-gene model
- [ ] Compare to baseline (~0.84) and feature-selected (~0.89) models

---

### Task 5.4: Create `notebooks/05_signature_analysis.ipynb`

**Goal:** Comprehensive analysis of the 11-gene signature.

**Notebook Contents:**
1. Verify signature genes are present
2. Visualize expression of signature genes across responders vs. non-responders
3. Compute importance ranks
4. Train 11-gene-only model
5. Compare all model AUCs in a summary table/plot
6. Save outputs

---

## Phase 6: Evaluation, Plots & Reproducibility Checks

### Task 6.1: Generate Summary Comparison Table

**Goal:** Create a single table comparing our results to the paper.

**Table Format:**

| Model | Our AUC | Paper AUC | Notes |
|-------|---------|-----------|-------|
| Baseline XGBoost (all genes) | X.XX | 0.84 | |
| + Feature Selection | X.XX | 0.89 | Boruta/importance-based |
| 11-Gene Signature Only | X.XX | ~0.85* | *Estimated from paper |

Save to: `results/tables/final_comparison.csv`

---

### Task 6.2: Generate Publication-Quality Figures

**Goal:** Create figures for report and poster.

**Required Figures:**
1. **ROC Curves:** Overlay all three models on one plot
2. **Feature Importance:** Top 20 genes bar plot, with signature genes highlighted
3. **Patient Predictions:** Scatter plot (x=patient, y=predicted score, color=true label)
4. **Gene Expression Heatmap:** Top genes × patients, clustered (optional)

Save to: `results/figures/` as PNG and PDF

---

### Task 6.3: Reproducibility Script

**Goal:** Create a single script that runs the entire pipeline.

**File:** `run_pipeline.py` (or Makefile)

```python
"""
Run full PRECISE reproduction pipeline.
Usage: python run_pipeline.py
"""
from src.data_loading import load_and_build_anndata
from src.preprocessing import run_preprocessing_pipeline
from src.labels import add_response_labels
from src.model import leave_one_patient_out_cv
from src.feature_selection import run_feature_selection
from src.evaluation import compute_roc_auc, plot_roc_curve, generate_results_table

def main():
    # Step 1: Load data
    # Step 2: Preprocess
    # Step 3: Add labels
    # Step 4: Baseline model
    # Step 5: Feature selection
    # Step 6: Improved model
    # Step 7: Signature analysis
    # Step 8: Generate outputs
    pass

if __name__ == '__main__':
    main()
```

**Acceptance Criteria:**
- [ ] `python run_pipeline.py` completes without error
- [ ] All outputs generated in `results/`
- [ ] Total runtime <30 minutes on standard laptop

---

## Phase 7: Optional Stretch Goals

### Stretch Goal A: Simplified Cell Filtration

**Idea:** Remove "non-predictive" cells to improve predictions.

**Simplified Approach (not full RL):**
1. From LOO-CV, each test cell has a predicted probability
2. Define "non-predictive" as cells with prediction confidence near 0.5
3. Compute a "predictivity score" per cell: `|predicted_prob - 0.5|`
4. Remove bottom X% of cells (least confident predictions)
5. Re-run LOO-CV on remaining cells
6. Check if AUC improves

**Alternative:** Train a logistic regression on cell features to predict whether a cell is "easy to classify" (high confidence) or not.

**Acceptance Criteria:**
- [ ] Cell filtration implemented
- [ ] AUC comparison: with vs. without filtration

---

### Stretch Goal B: External Cohort Validation

**Idea:** Test the 11-gene signature on a different cancer cohort.

**Candidate Datasets (from paper):**
- TNBC (triple-negative breast cancer)
- NSCLC (non-small cell lung cancer)
- BCC (basal cell carcinoma)

**Steps:**
1. Download external dataset from GEO
2. Preprocess similarly
3. Compute 11-gene signature score per cell
4. Aggregate to patient level
5. Compute AUC if response labels available

**Acceptance Criteria:**
- [ ] External data loaded and processed
- [ ] AUC computed and compared to paper's reported value

---

### Stretch Goal C: SHAP Interpretability

**Idea:** Use SHAP values for model interpretation.

**Steps:**
1. For one LOO-CV fold, compute SHAP values
2. Plot SHAP summary (beeswarm plot)
3. Identify top genes by SHAP importance
4. Compare to XGBoost's built-in importance

**Acceptance Criteria:**
- [ ] SHAP summary plot generated
- [ ] Discussion of gene interactions (if visible)

---

## Timeline / Milestones

### Suggested Schedule (2 weeks)

| Day | Phase | Milestone |
|-----|-------|-----------|
| 1-2 | Setup | Environment configured; data organized; `01_explore_data.ipynb` complete |
| 3-4 | Phase 1-2 | Data loading and preprocessing complete; labels assigned |
| 5-6 | Phase 3 | Baseline XGBoost LOO-CV working; AUC computed |
| 7-8 | Phase 4 | Feature selection implemented; improved AUC achieved |
| 9-10 | Phase 5-6 | Signature analysis complete; all figures generated |
| 11-12 | Report | `report.md` drafted (manually, no LLM) |
| 13-14 | Polish | Final testing; poster designed in Canva; submission |

### Key Checkpoints

- [ ] **End of Day 2:** Can load data and see ~16k cells, 48 patients
- [ ] **End of Day 4:** Have labeled AnnData saved to disk
- [ ] **End of Day 6:** Have baseline AUC number (target ~0.84)
- [ ] **End of Day 8:** Have improved AUC number (target ~0.89)
- [ ] **End of Day 10:** All required figures and tables generated
- [ ] **End of Day 12:** `report.md` complete (written by you, not LLM)
- [ ] **End of Day 14:** Everything submitted

---

## Quick Reference: Paper Targets

| Metric | Paper Value | Our Target Range |
|--------|-------------|------------------|
| Baseline XGBoost AUC | 0.84 | 0.75 – 0.90 |
| Feature Selection AUC | 0.89 | 0.82 – 0.92 |
| Number of Patients | 48 | 48 |
| Number of Cells | ~16,000 | ~16,000 |
| Responders | 17 | ~17 |
| Non-Responders | 31 | ~31 |

**11-Gene Signature:**
```
GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5
```

---

## Checklist Summary

### Phase 1: Data Loading & Preprocessing
- [ ] Create directory structure (`data/`, `src/`, `notebooks/`, `results/`)
- [ ] Move raw data to `data/raw/`
- [ ] Implement `src/data_loading.py`
- [ ] Implement `src/preprocessing.py`
- [ ] Complete `notebooks/01_explore_data.ipynb`
- [ ] Complete `notebooks/02_preprocessing.ipynb`
- [ ] Save `data/processed/melanoma_adata.h5ad`

### Phase 2: Label Construction
- [ ] Implement `src/labels.py`
- [ ] Add response labels to AnnData
- [ ] Verify ~17 responders, ~31 non-responders

### Phase 3: Baseline Model
- [ ] Implement `src/model.py`
- [ ] Implement `src/evaluation.py`
- [ ] Complete `notebooks/03_baseline_model.ipynb`
- [ ] Achieve baseline AUC ~0.84
- [ ] Save ROC curve to `results/figures/`

### Phase 4: Feature Selection
- [ ] Implement `src/feature_selection.py`
- [ ] Complete `notebooks/04_feature_selection.ipynb`
- [ ] Achieve improved AUC ~0.89
- [ ] Save selected genes list

### Phase 5: Signature Analysis
- [ ] Verify 11 genes present
- [ ] Compute signature gene ranks
- [ ] (Optional) Train 11-gene-only model
- [ ] Complete `notebooks/05_signature_analysis.ipynb`

### Phase 6: Evaluation & Outputs
- [ ] Generate final comparison table
- [ ] Generate publication-quality figures
- [ ] Create `run_pipeline.py`
- [ ] Verify end-to-end reproducibility

### Deliverables
- [ ] `report.md` written manually (no LLM)
- [ ] Poster PDF designed in Canva
- [ ] GitHub repo complete and clean

---

## Notes for Future LLM Sessions

When returning to this project:

1. **Read `selected-topic-overview.md`** for paper context
2. **Read this `PLAN.md`** for implementation details
3. **Check the checklist** to see what's done and what's next
4. **Look at existing code in `src/`** to understand current state
5. **Run notebooks in order** if starting fresh

Key technical decisions documented in this plan:
- Using `AnnData` from scanpy for data structure
- XGBoost with LOO-CV as core method
- Feature selection via Boruta or importance-based
- Target AUCs: 0.84 baseline, 0.89 improved
- 11-gene signature must be analyzed explicitly

---
