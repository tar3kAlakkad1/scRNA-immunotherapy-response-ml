# Selected Topic Overview – CSC 427 Final Project

## 1. Course Context

This repository contains my final project for **CSC 427 (Bioinformatics)** at the University of Victoria.

Per the official project description - found in the repo under the file `piazza-requirements.md`, the goals are:

- Select an **interesting bioinformatics paper**.
- **Reimplement the main idea / code** from that paper.
- **Rerun the method on the original datasets** and compare my results to the published ones.
- Use **accessible and manageable datasets** (no GPU datacenters or extreme RAM requirements).
- Document the work in:
  - a **GitHub repo** with code and a Markdown **`report.md`**, and  
  - a **poster PDF** submitted via Brightspace (poster will be designed manually in Canva).

Negative or partial reproduction results are acceptable if they are well-explained (why the results differ, what might be non-reproducible, etc.). LLMs are **not allowed** to write the final report text; they may only be used for planning, coding, debugging, and high-level guidance.

---

## 2. Selected Paper

**Paper**

> Asaf Pinhasi & Keren Yizhak (2025).  
> *Uncovering gene and cellular signatures of immune checkpoint response via machine learning and single-cell RNA-seq.*  
> npj Precision Oncology, 9:95. DOI: 10.1038/s41698-025-00883-z.

**High-level idea**

The paper introduces **PRECISE**, a machine learning framework that uses **single-cell RNA-seq (scRNA-seq)** data from **tumor-infiltrating immune cells** to predict **immune checkpoint inhibitor (ICI) response** (responder vs non-responder).

Key points:

- **Input data (training cohort)**  
  - scRNA-seq from melanoma-infiltrating immune cells (CD45+), pre-treatment.  
  - Dataset: **Sade-Feldman et al.**; public in GEO as **GSE120575**.  
  - Around **16k cells** from **48 samples/patients**, each patient labeled as ICI responder or non-responder.

- **Modeling goal**  
  - Predict **patient-level ICI response** using gene expression from single cells.  
  - Maintain **single-cell resolution** while generating a **sample-level prediction**.

- **Core ML approach**  
  - Label each cell with its **patient’s response label** (R/NR).  
  - Train an **XGBoost** classifier on **per-cell gene expression**.  
  - Use **leave-one-patient-out cross-validation (LOO-CV)**:
    - For each fold, hold out all cells from one patient as test.
    - Train on cells from the remaining patients.
    - Aggregate per-cell predictions to a **sample-level score** for the held-out patient.
  - Evaluate performance with **ROC AUC** at the patient level.

- **Main headline results** (training melanoma cohort):
  - Base XGBoost model achieves AUC ≈ **0.84**.   
  - Applying **Boruta** feature selection to choose predictive genes improves AUC to ≈ **0.89**.   

- **11-gene signature**  
  - From feature importance across folds, they derive an **11-gene signature** that is predictive across multiple cancer types.  
  - Genes in the signature (from the paper’s survival analysis figure) are:  
    - **GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5**.   
  - They show that a score derived from these 11 genes:
    - Predicts ICI response in multiple validation cohorts (melanoma, TNBC, NSCLC, BCC, glioblastoma, breast cancer).
    - Is associated with better overall survival in bulk RNA-seq data for many cancer types.

- **Interpretability (SHAP)**  
  - They compute **SHAP values** for the XGBoost model to see:
    - Which genes are most important overall.
    - How gene–gene interactions influence predictions (non-linear and context dependent).

- **Reinforcement learning (RL) cell scoring and filtration**   
  - They train a **reinforcement-learning–style model** that assigns each cell a “predictivity score”:
    - Cells are categorized as:
      - Predictive for responders,
      - Predictive for non-responders,
      - Non-predictive.
  - They then train a **logistic regression** classifier to identify **non-predictive cells**, using the most informative genes.
  - A **cell-filtration score** is constructed from top genes in that logistic model.
  - By **removing ~40% of cells** with the highest “non-predictive” score, then re-applying the 11-gene signature, they further **improve AUC** on multiple datasets:
    - Melanoma AUC rises to >0.94 (from ~0.89).  
    - Several other cohorts also show improved AUC; one NSCLC cohort is an exception.

- **External validation cohorts** (not all accession IDs needed for this project):
  - TNBC (triple-negative breast cancer), NSCLC (non-small cell lung cancer), BCC (basal cell carcinoma), glioblastoma, and several breast cancer cohorts.
  - They apply the 11-gene signature (with/without cell filtration) and compute ROC curves within T cells or immune subsets.

---

## 3. Project Goal (My Implementation)

**High-level goal**

Reimplement the **core PRECISE pipeline** in Python, run it on the **GSE120575 melanoma dataset**, and compare my performance and gene-level findings to the published results.

### What exactly needs to be recreated

1. **Data-level setup (melanoma cohort only)**  
   - Input:
     - Expression matrix: `GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz`.
     - Cell→patient mapping: `GSE120575_patient_ID_single_cells.txt.gz`.  
   - Tasks:
     - Load both files and align cells.
     - Construct:
       - A gene × cell matrix (TPM).
       - Metadata per cell with: cell ID, patient/sample ID, and ICI response label (R/NR) for that patient.

2. **Preprocessing and filtering (melanoma)**  
   - Implement filtering and normalization as closely as possible to the paper:
     - Remove low-quality genes (e.g., very low expression, possibly mitochondrial/ribosomal genes if specified).
     - Keep genes expressed in at least a small fraction of cells (paper mentions a percentage threshold).
     - Apply suitable normalization (e.g., log1p(TPM + 1) or scanpy standard pipeline).
   - Represent data as an `AnnData` object (`scanpy`) or equivalent.

3. **Baseline XGBoost model with leave-one-patient-out CV**  
   - For each fold:
     - Choose one patient as test; train on all other patients.
     - Training data: all cells from training patients; each cell labeled with its patient’s response (R/NR).
     - Model: XGBoost classifier with hyperparameters approximating those in the paper (they use a relatively small tree depth and non-extreme learning rate).
   - For each test patient:
     - Predict per-cell probabilities.
     - Aggregate per-cell predictions to a **single score per patient** (e.g., mean predicted probability of “responder” over all their cells).
   - Compute **ROC AUC** over all patients, using sample-level scores and true response labels.
   - Target: reproduce an AUC close to the paper’s base model (~0.84).

4. **Feature selection and improved AUC**  
   - Implement **feature selection**:
     - Either Boruta, or another importance-based method that mirrors the main idea:
       - Start from all genes,
       - Identify a subset of predictive genes,
       - Retrain XGBoost using only that subset.
   - Re-run the LOO-CV pipeline using the selected genes.
   - Compute AUC again; target is to approach the improved AUC (~0.89).

5. **Feature importance and 11-gene signature comparison**  
   - From the full model (or Boruta runs), compute **gene importance scores**:
     - Per fold, then averaged across folds.
   - Rank genes and compare:
     - My top genes vs. the paper’s reported top genes and 11-gene signature.
   - Specifically check how the following 11 genes rank:
     - `GAPDH, CD38, CCR7, HLA-DRB5, STAT1, GZMH, LGALS1, IFI6, EPSTI1, HLA-G, GBP5`.
   - Optionally:
     - Train a **restricted model** that only uses these 11 genes and measure its AUC.
     - This provides a direct reproducibility check for the signature itself.

6. **(Optional) Simplified cell filtration**  
   - Full RL implementation may be out of scope. Instead, aim for a simplified version inspired by the paper:
     - Use the per-cell predictions and features to derive a “predictivity” measure (e.g., based on model confidence and/or a simple classifier of non-predictive cells).
     - Remove a fraction of cells considered “non-predictive” and re-evaluate the 11-gene signature AUC.
   - The goal is to see **whether filtration can improve AUC**, even if the exact gains differ from the paper.

7. **(Optional) One external cohort test**  
   - If time allows, pick one external dataset from the paper (e.g., a TNBC or NSCLC scRNA-seq dataset).
   - Apply:
     - The 11-gene signature score,
     - Possibly my trained melanoma model (if compatible),
   - Compute AUC for response vs non-response within that cohort’s T cells or immune cells.
   - This tests how well the signature generalizes in my hands.

---

## 4. Scope and Non-Goals

To keep the project feasible within course timelines and standard hardware:

### In Scope

- Single-dataset focus: **GSE120575** melanoma cohort as the **primary** target.
- Reimplementation of the **core classification pipeline**:
  - Preprocessing and filtering,
  - XGBoost model with leave-one-patient-out CV,
  - Sample-level ROC AUC,
  - Feature importance analysis.
- Exploration of the **11-gene signature**:
  - Importance ranking of those genes,
  - Optional “11-gene-only” model.
- Clear comparisons to the paper’s key metrics (AUCs and gene-level findings).

### Out of Scope (unless time remains)

- Full reproduction of **all external cohorts** and all figures.
- A full, faithful reimplementation of the **reinforcement learning module** for cell scoring.
- Complete SHAP analysis and visualization as in the paper.
- Heavy GPU-based or extremely large-scale experiments.

---

## 5. Planned Deliverables

As required by CSC 427:

1. **GitHub repository** (this repo)
   - All source code (Python) for:
     - data loading and preprocessing,
     - model training and evaluation,
     - result generation (plots, metrics, tables).
   - A Markdown report: **`report.md`** in a project subdirectory, written manually (no LLMs), containing:
     - introduction and background,
     - methods (my implementation of PRECISE),
     - results and comparisons to the original paper,
     - discussion of reproducibility / discrepancies,
     - conclusion and possible future work.

2. **Poster (PDF)**  
   - A visually clear summary of:
     - the problem and paper,
     - my implementation,
     - key results and comparisons,
     - main takeaways.
   - The poster will be designed manually in **Canva**, then exported as PDF and submitted to Brightspace.

---

## 6. Technical Stack & Tools

- **Language:** Python
- **Core libraries (planned):**
  - `scanpy` / `anndata` – scRNA-seq data structure and basic processing
  - `pandas`, `numpy` – data wrangling
  - `xgboost` – main classifier
  - `scikit-learn` – metrics, cross-validation helpers
  - `matplotlib` / `seaborn` – plots and visualizations
- **Version control:** Git + GitHub (this repository)
- **Environment management:** `conda` or `mamba` environment with a pinned `environment.yml` or `requirements.txt`

---

## 7. High-Level Task Breakdown

1. **Setup**
   - Create and configure this repository.
   - Set up Python environment and install dependencies.
   - Add this `selected-topic-overview.md` to document the plan.

2. **Data Acquisition & Preprocessing**
   - Download from GEO (GSE120575 supplemental files):
     - `GSE120575_Sade_Feldman_melanoma_single_cells_TPM_GEO.txt.gz`
     - `GSE120575_patient_ID_single_cells.txt.gz`
   - Load into Python and build an `AnnData` object.
   - Implement filtering and normalization approximating the paper.

3. **Label Construction**
   - Map cells to patients using the patient ID file.
   - Retrieve or reconstruct patient-level response labels (R/NR) based on Sade-Feldman metadata / the PRECISE paper’s mapping.
   - Store labels in `.obs` (e.g., `response`, `sample_id`).

4. **Model Implementation**
   - Implement XGBoost training with leave-one-patient-out CV.
   - Aggregate cell predictions to patient-level scores.
   - Compute ROC AUC and log all hyperparameters and seeds.

5. **Feature Selection & Importance**
   - Implement Boruta or a similar feature selection method.
   - Retrain XGBoost on selected genes and recompute AUC.
   - Compute and plot feature importances; compare to 11-gene signature.

6. **Result Comparison & Analysis**
   - Compare my AUCs and feature rankings to the paper’s reported values.
   - Explicitly discuss differences and possible reasons (preprocessing, randomness, thresholding, etc.).

7. **Optional Extensions**
   - Simple cell-filtration experiment inspired by the RL filtration in the paper.
   - Apply the 11-gene signature to one external ICI dataset to see if it generalizes.

8. **Report & Poster**
   - Manually write `report.md` (no LLM assistance for text).
   - Generate figures and tables from code for use in:
     - `report.md`
     - The Canva poster.

---

## 8. LLM Usage Policy (Self-Reminder)

- **Allowed:**
  - Planning tasks and project structure.
  - Writing and refactoring code.
  - Debugging and understanding error messages.
  - Brainstorming analysis ideas and figure concepts.
  - Generating helper documents like *this* overview file.

- **Not allowed:**
  - Writing or editing any prose that will go into the final `report.md`.
  - Writing textual content of the final poster.

All report and poster text must be drafted by me manually to comply with the course policy (“Do not use LLMs to write your report, or you will get 0 points.”).

---
