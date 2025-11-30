"""
Labels Module
=============

Maps patient IDs to ICI response labels (Responder vs Non-Responder) and provides
utilities for managing response labels in the single-cell data.

This module handles the label construction step of the PRECISE pipeline:
    - Extract patient-level response labels from cell metadata
    - Map "Responder"/"Non-responder" to shorthand "R"/"NR" codes
    - Add response labels to AnnData objects
    - Generate patient-level metadata summaries

Label Source:
    The GSE120575 patient ID file contains response labels in the metadata.
    Labels are extracted during data loading (see data_loading.py) and stored
    in adata.obs['response'] as "Responder" or "Non-responder".

Expected Distribution (from Sade-Feldman et al., 2018 / PRECISE paper):
    - 48 unique patient/timepoint combinations
    - ~17 responders, ~31 non-responders

Response Label Conventions:
    - Full string: "Responder" / "Non-responder" (as stored in GEO metadata)
    - Short code: "R" / "NR" (used in some outputs and the PLAN.md spec)
    - Binary: 1 (Responder) / 0 (Non-responder) (for ML classifiers)
"""

from pathlib import Path
from typing import Dict, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd


# Response label mappings
RESPONSE_FULL_TO_SHORT = {
    "Responder": "R",
    "Non-responder": "NR",
}

RESPONSE_SHORT_TO_FULL = {
    "R": "Responder",
    "NR": "Non-responder",
}

RESPONSE_FULL_TO_BINARY = {
    "Responder": 1,
    "Non-responder": 0,
}

RESPONSE_BINARY_TO_FULL = {
    1: "Responder",
    0: "Non-responder",
}


def get_response_labels(
    adata: ad.AnnData,
    use_short_codes: bool = True,
) -> Dict[str, str]:
    """
    Extract patient-level response labels from an AnnData object.

    Returns a dictionary mapping each unique patient_id to its response label.
    Each patient should have a consistent label across all their cells.

    Parameters
    ----------
    adata : AnnData
        AnnData object with 'patient_id' and 'response' columns in .obs
    use_short_codes : bool, default=True
        If True, return "R"/"NR" codes; if False, return full
        "Responder"/"Non-responder" strings.

    Returns
    -------
    dict
        Mapping of {patient_id: label} where label is "R"/"NR" or full string.

    Raises
    ------
    ValueError
        If required columns are missing or if a patient has inconsistent labels.

    Examples
    --------
    >>> labels = get_response_labels(adata)
    >>> labels["Pre_P1"]
    'R'
    >>> labels["Post_P6"]
    'NR'
    """
    # Validate required columns exist
    if "patient_id" not in adata.obs.columns:
        raise ValueError("'patient_id' column not found in adata.obs")
    if "response" not in adata.obs.columns:
        raise ValueError("'response' column not found in adata.obs")

    # Extract unique patient-response pairs
    patient_response_df = adata.obs.groupby("patient_id", observed=True)[
        "response"
    ].unique()

    # Build the mapping, checking for consistency
    response_labels = {}
    inconsistent_patients = []

    for patient_id, responses in patient_response_df.items():
        if len(responses) > 1:
            inconsistent_patients.append((patient_id, list(responses)))
        else:
            response = responses[0]
            if use_short_codes:
                response_labels[patient_id] = RESPONSE_FULL_TO_SHORT.get(
                    response, response
                )
            else:
                response_labels[patient_id] = response

    if inconsistent_patients:
        msg = "Inconsistent response labels found for patients:\n"
        for pid, resps in inconsistent_patients:
            msg += f"  {pid}: {resps}\n"
        raise ValueError(msg)

    return response_labels


def add_response_labels(
    adata: ad.AnnData,
    add_binary: bool = True,
    add_short_code: bool = True,
    verbose: bool = True,
) -> ad.AnnData:
    """
    Add or update response labels in an AnnData object.

    This function ensures that all necessary response label columns are present:
        - 'response': Full string ("Responder" / "Non-responder")
        - 'response_binary': Binary label (1 / 0) for ML classifiers
        - 'response_short': Short code ("R" / "NR")

    If 'response' already exists, derived columns are computed from it.
    If 'response' is missing but 'response_binary' exists, 'response' is reconstructed.

    Parameters
    ----------
    adata : AnnData
        Input AnnData object. Modified in place.
    add_binary : bool, default=True
        Whether to add/update 'response_binary' column.
    add_short_code : bool, default=True
        Whether to add 'response_short' column ("R"/"NR").
    verbose : bool, default=True
        Whether to print label distribution summary.

    Returns
    -------
    AnnData
        AnnData object with response label columns added/updated.

    Raises
    ------
    ValueError
        If neither 'response' nor 'response_binary' exists in adata.obs,
        or if unknown response values are encountered.

    Examples
    --------
    >>> adata = add_response_labels(adata)
    >>> adata.obs['response_binary'].value_counts()
    0    10726
    1     5564
    Name: response_binary, dtype: int64
    """
    # Check if we have a source for response labels
    has_response = "response" in adata.obs.columns
    has_binary = "response_binary" in adata.obs.columns

    if not has_response and not has_binary:
        raise ValueError(
            "No response labels found. adata.obs must contain either "
            "'response' or 'response_binary' column. "
            "Did you load the patient metadata correctly?"
        )

    # If we only have binary, reconstruct full response strings
    if not has_response and has_binary:
        adata.obs["response"] = adata.obs["response_binary"].map(
            RESPONSE_BINARY_TO_FULL
        )

    # Validate response values
    unique_responses = set(adata.obs["response"].dropna().unique())
    valid_responses = set(RESPONSE_FULL_TO_BINARY.keys())
    unknown = unique_responses - valid_responses

    if unknown:
        raise ValueError(
            f"Unknown response values: {unknown}. "
            f"Expected one of: {valid_responses}"
        )

    # Check for missing response values
    n_missing = adata.obs["response"].isna().sum()
    if n_missing > 0:
        raise ValueError(
            f"Found {n_missing} cells with missing response labels. "
            "All cells must have a response label assigned."
        )

    # Add binary labels
    if add_binary:
        adata.obs["response_binary"] = (
            adata.obs["response"].map(RESPONSE_FULL_TO_BINARY).astype(int)
        )

    # Add short code labels
    if add_short_code:
        adata.obs["response_short"] = adata.obs["response"].map(RESPONSE_FULL_TO_SHORT)

    if verbose:
        _print_label_summary(adata)

    return adata


def get_patient_metadata(
    adata: ad.AnnData,
    sort_by: str = "patient_id",
) -> pd.DataFrame:
    """
    Generate a summary DataFrame of patient-level metadata.

    This function aggregates cell-level data to produce one row per patient
    with their response label, therapy type, and cell count.

    Parameters
    ----------
    adata : AnnData
        AnnData object with patient metadata in .obs
    sort_by : str, default="patient_id"
        Column to sort the output by. Options: "patient_id", "response",
        "n_cells", "therapy".

    Returns
    -------
    pd.DataFrame
        Summary DataFrame with columns:
            - patient_id: Unique patient identifier
            - response: Full response string ("Responder" / "Non-responder")
            - response_short: Short code ("R" / "NR")
            - response_binary: Binary label (1 / 0)
            - therapy: Treatment type (anti-PD1, anti-CTLA4, etc.)
            - n_cells: Number of cells from this patient
            - cell_fraction: Fraction of total cells from this patient

    Examples
    --------
    >>> patient_df = get_patient_metadata(adata)
    >>> patient_df.head()
       patient_id       response response_short  response_binary     therapy  n_cells  cell_fraction
    0     Post_P1      Responder              R                1    anti-PD1      412          0.025
    1    Post_P10  Non-responder             NR                0    anti-PD1      189          0.012
    ...
    """
    # Validate required columns
    required_cols = ["patient_id", "response"]
    missing = [col for col in required_cols if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Missing required columns in adata.obs: {missing}")

    # Aggregate by patient
    patient_groups = adata.obs.groupby("patient_id", observed=True)

    # Build summary DataFrame
    patient_df = pd.DataFrame(
        {
            "patient_id": list(patient_groups.groups.keys()),
            "response": patient_groups["response"].first().values,
            "n_cells": patient_groups.size().values,
        }
    )

    # Add short code
    patient_df["response_short"] = patient_df["response"].map(RESPONSE_FULL_TO_SHORT)

    # Add binary
    patient_df["response_binary"] = patient_df["response"].map(RESPONSE_FULL_TO_BINARY)

    # Add therapy if available
    if "therapy" in adata.obs.columns:
        therapy_per_patient = patient_groups["therapy"].first().values
        patient_df["therapy"] = therapy_per_patient

    # Add cell fraction
    total_cells = adata.n_obs
    patient_df["cell_fraction"] = patient_df["n_cells"] / total_cells

    # Reorder columns for readability
    col_order = ["patient_id", "response", "response_short", "response_binary"]
    if "therapy" in patient_df.columns:
        col_order.append("therapy")
    col_order.extend(["n_cells", "cell_fraction"])
    patient_df = patient_df[col_order]

    # Sort by requested column
    if sort_by in patient_df.columns:
        patient_df = patient_df.sort_values(sort_by).reset_index(drop=True)
    elif sort_by == "patient_id":
        # Natural sort for patient IDs (Pre_P1, Pre_P2, ..., Post_P1, ...)
        patient_df = patient_df.sort_values("patient_id").reset_index(drop=True)

    return patient_df


def get_response_distribution(
    adata: ad.AnnData,
    level: str = "patient",
) -> pd.DataFrame:
    """
    Get the distribution of response labels at patient or cell level.

    Parameters
    ----------
    adata : AnnData
        AnnData object with response labels in .obs
    level : str, default="patient"
        Aggregation level: "patient" (one count per patient) or "cell"
        (count of cells).

    Returns
    -------
    pd.DataFrame
        Distribution summary with counts and percentages.

    Examples
    --------
    >>> get_response_distribution(adata, level="patient")
            response  count  percentage
    0      Responder     17       35.4%
    1  Non-responder     31       64.6%
    """
    if "response" not in adata.obs.columns:
        raise ValueError("'response' column not found in adata.obs")

    if level == "patient":
        # Patient-level: count unique patients per response
        patient_response = adata.obs.groupby("patient_id", observed=True)[
            "response"
        ].first()
        counts = patient_response.value_counts()
    elif level == "cell":
        # Cell-level: count cells per response
        counts = adata.obs["response"].value_counts()
    else:
        raise ValueError(f"Invalid level '{level}'. Use 'patient' or 'cell'.")

    total = counts.sum()
    dist_df = pd.DataFrame(
        {
            "response": counts.index,
            "count": counts.values,
            "percentage": (counts.values / total * 100).round(1),
        }
    )
    dist_df["percentage"] = dist_df["percentage"].astype(str) + "%"

    return dist_df


def validate_labels(
    adata: ad.AnnData,
    expected_n_patients: int = 48,
    expected_n_responders: Optional[int] = 17,
    expected_n_nonresponders: Optional[int] = 31,
    verbose: bool = True,
) -> bool:
    """
    Validate that response labels meet expected criteria.

    Checks:
        1. All cells have a non-null response value
        2. Total number of unique patients matches expectation
        3. Number of responders/non-responders matches expectation (if specified)

    Parameters
    ----------
    adata : AnnData
        AnnData object with response labels in .obs
    expected_n_patients : int, default=48
        Expected number of unique patient_ids.
    expected_n_responders : int, optional, default=17
        Expected number of responder patients. If None, not checked.
    expected_n_nonresponders : int, optional, default=31
        Expected number of non-responder patients. If None, not checked.
    verbose : bool, default=True
        Whether to print validation results.

    Returns
    -------
    bool
        True if all validations pass, False otherwise.
    """
    all_passed = True

    if verbose:
        print("=" * 60)
        print("LABEL VALIDATION")
        print("=" * 60)

    # Check 1: No null responses
    n_missing = adata.obs["response"].isna().sum()
    check1_pass = n_missing == 0
    if verbose:
        print(f"\n1. All cells have response labels:")
        print(f"   Missing values: {n_missing}")
        print(f"   Status: {'PASS' if check1_pass else 'FAIL'}")
    all_passed &= check1_pass

    # Check 2: Number of patients
    n_patients = adata.obs["patient_id"].nunique()
    check2_pass = n_patients == expected_n_patients
    if verbose:
        print(f"\n2. Patient count:")
        print(f"   Found: {n_patients}")
        print(f"   Expected: {expected_n_patients}")
        print(f"   Status: {'PASS' if check2_pass else 'FAIL'}")
    all_passed &= check2_pass

    # Check 3: Responder count
    if expected_n_responders is not None:
        patient_response = adata.obs.groupby("patient_id", observed=True)[
            "response"
        ].first()
        n_responders = (patient_response == "Responder").sum()
        check3_pass = n_responders == expected_n_responders
        if verbose:
            print(f"\n3. Responder count:")
            print(f"   Found: {n_responders}")
            print(f"   Expected: {expected_n_responders}")
            print(f"   Status: {'PASS' if check3_pass else 'FAIL'}")
        all_passed &= check3_pass

    # Check 4: Non-responder count
    if expected_n_nonresponders is not None:
        patient_response = adata.obs.groupby("patient_id", observed=True)[
            "response"
        ].first()
        n_nonresponders = (patient_response == "Non-responder").sum()
        check4_pass = n_nonresponders == expected_n_nonresponders
        if verbose:
            print(f"\n4. Non-responder count:")
            print(f"   Found: {n_nonresponders}")
            print(f"   Expected: {expected_n_nonresponders}")
            print(f"   Status: {'PASS' if check4_pass else 'FAIL'}")
        all_passed &= check4_pass

    if verbose:
        print("\n" + "=" * 60)
        print(f"OVERALL: {'ALL CHECKS PASSED' if all_passed else 'SOME CHECKS FAILED'}")
        print("=" * 60)

    return all_passed


def _print_label_summary(adata: ad.AnnData) -> None:
    """Print a summary of response labels in the AnnData object."""
    print("  Response label summary:")

    # Cell-level counts
    cell_counts = adata.obs["response"].value_counts()
    total_cells = adata.n_obs
    print(f"  Cell-level distribution ({total_cells:,} total cells):")
    for response, count in cell_counts.items():
        pct = count / total_cells * 100
        short = RESPONSE_FULL_TO_SHORT.get(response, response)
        print(f"    {response} ({short}): {count:,} cells ({pct:.1f}%)")

    # Patient-level counts
    patient_response = adata.obs.groupby("patient_id", observed=True)[
        "response"
    ].first()
    patient_counts = patient_response.value_counts()
    total_patients = len(patient_response)
    print(f"  Patient-level distribution ({total_patients} patients):")
    for response, count in patient_counts.items():
        pct = count / total_patients * 100
        short = RESPONSE_FULL_TO_SHORT.get(response, response)
        print(f"    {response} ({short}): {count} patients ({pct:.1f}%)")


# ============================================================================
# Module-level script for testing
# ============================================================================

if __name__ == "__main__":
    """
    Test the labels module when run directly.

    Usage:
        python -m src.labels

    Or from project root:
        python src/labels.py
    """
    import sys

    # Import sibling modules
    from preprocessing import load_preprocessed_data, DEFAULT_OUTPUT_PATH

    project_root = Path(__file__).parent.parent

    print("Loading preprocessed data...")
    print("-" * 60)
    input_path = project_root / DEFAULT_OUTPUT_PATH
    adata = load_preprocessed_data(input_path)

    # Test get_response_labels
    print("\n" + "=" * 60)
    print("TEST: get_response_labels()")
    print("=" * 60)
    labels = get_response_labels(adata, use_short_codes=True)
    print(f"Number of patients: {len(labels)}")
    print(f"Sample labels (first 5):")
    for i, (pid, label) in enumerate(sorted(labels.items())[:5]):
        print(f"  {pid}: {label}")

    # Test add_response_labels (should add response_short column)
    print("\n" + "=" * 60)
    print("TEST: add_response_labels()")
    print("=" * 60)
    adata = add_response_labels(adata, verbose=True)
    print(f"\nNew columns in adata.obs: {list(adata.obs.columns)}")

    # Test get_patient_metadata
    print("\n" + "=" * 60)
    print("TEST: get_patient_metadata()")
    print("=" * 60)
    patient_df = get_patient_metadata(adata, sort_by="response")
    print(f"\nPatient metadata DataFrame shape: {patient_df.shape}")
    print("\nFirst 10 rows (sorted by response):")
    print(patient_df.head(10).to_string())

    # Test get_response_distribution
    print("\n" + "=" * 60)
    print("TEST: get_response_distribution()")
    print("=" * 60)
    print("\nPatient-level distribution:")
    print(get_response_distribution(adata, level="patient").to_string(index=False))
    print("\nCell-level distribution:")
    print(get_response_distribution(adata, level="cell").to_string(index=False))

    # Run validation checks
    print("\n")
    passed = validate_labels(adata, verbose=True)

    sys.exit(0 if passed else 1)
