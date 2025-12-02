# scRNA-immunotherapy-response-ml
Reproducing the PRECISE framework from a research paper of interest to me. 

## Setup
```bash
conda create -n csc427-project python=3.11 -y
conda activate csc427-project
pip install pysam scanpy anndata pandas numpy xgboost scikit-learn matplotlib seaborn boruta tqdm shap jupyter
```

## Task 3.1 - Baseline XGModel trained output

$ cd /Users/tarekalakkadp/Desktop/uvic/fourth-year/fall/csc427/final-project/scRNA-immunotherapy-response-ml && pip install xgboost scikit-learn && python src/model.py
Loading preprocessed data...
------------------------------------------------------------
Loaded preprocessed data from: /Users/tarekalakkadp/Desktop/uvic/fourth-year/fall/csc427/final-project/scRNA-immunotherapy-response-ml/data/processed/melanoma_adata.h5ad
Shape: 16,290 cells x 12,785 genes


============================================================
LEAVE-ONE-PATIENT-OUT CROSS-VALIDATION
============================================================

Dataset: 16,290 cells x 12,785 genes
Patients: 48
XGBoost params: defaults

Running LOO-CV...
  Completed fold 10/48
  Completed fold 20/48
  Completed fold 30/48
  Completed fold 40/48

  Completed all 48 folds in 1485.3 seconds

============================================================
RESULTS
============================================================

Patient-level ROC AUC: 0.770
Score range: [0.034, 0.675]
Label distribution: 17 responders, 31 non-responders

============================================================
ACCEPTANCE CRITERIA CHECK
============================================================

1. LOO-CV completed for all patients:
   Patients processed: 48
   Expected: 48
   Status: PASS

2. Runtime is reasonable:
   Runtime: 1485.3 seconds (24.8 minutes)
   Expected: <15 minutes (900 seconds)
   Status: FAIL

3. Patient-level scores in valid range:
   Score range: [0.034, 0.675]
   Expected: [0.0, 1.0]
   Status: PASS

4. Model performance (informational):
   ROC AUC: 0.770
   Paper reports: ~0.84 for base model
   Difference: 0.070

============================================================
OVERALL: SOME CHECKS FAILED
============================================================