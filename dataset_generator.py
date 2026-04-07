"""
datasets.py
===========
Three real-world style datasets with known data quality issues baked in.
Each dataset has a ground truth issue manifest — the grader uses this
to verify the agent fixed every issue EDA should have found.

Task 1 — Customer Churn      (easy)   : missing values, wrong dtypes, high cardinality
Task 2 — Loan Default        (medium) : leaky column, class imbalance, duplicates, correlations
Task 3 — Medical Readmission (hard)   : multiple leaky cols, severe imbalance, heavy missing,
                                        skewed distributions, dtype errors, correlations
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, List

RNG = np.random.RandomState(42)


# ---------------------------------------------------------------------------
# Task 1 — Customer Churn (easy)
# ---------------------------------------------------------------------------

def make_churn_dataset() -> pd.DataFrame:
    n = 1000

    # Core features
    age             = RNG.randint(18, 70, n).astype(float)
    tenure_months   = RNG.randint(1, 72, n).astype(float)
    monthly_charges = RNG.uniform(20, 120, n)
    total_charges   = tenure_months * monthly_charges + RNG.randn(n) * 50

    contract        = RNG.choice(["Month-to-month", "One year", "Two year"], n,
                                  p=[0.55, 0.25, 0.20])
    internet        = RNG.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    payment         = RNG.choice(["Electronic check", "Mailed check",
                                   "Bank transfer", "Credit card"], n)

    # 150 unique cities — high cardinality issue
    cities = [f"City_{i}" for i in range(150)]
    city   = RNG.choice(cities, n)

    # Target: churn
    churn_prob = (
        0.3
        + 0.25 * (contract == "Month-to-month").astype(float)
        - 0.15 * (tenure_months / 72)
        + 0.1  * (internet == "Fiber optic").astype(float)
    )
    churn = (RNG.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id":     [f"CUST_{i:05d}" for i in range(n)],
        "age":             age,
        "tenure_months":   tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges":   total_charges,
        "contract":        contract,
        "internet_service":internet,
        "payment_method":  payment,
        "city":            city,
        "churn":           churn,
    })

    # ── Inject issues ──────────────────────────────────────────────────────

    # 1. Missing values in age (15%) and monthly_charges (12%)
    df.loc[RNG.choice(n, int(n * 0.15), replace=False), "age"] = np.nan
    df.loc[RNG.choice(n, int(n * 0.12), replace=False), "monthly_charges"] = np.nan

    # 2. Wrong dtype: tenure_months stored as string with "months" suffix
    df["tenure_months"] = df["tenure_months"].apply(
        lambda x: f"{int(x)} months" if pd.notna(x) else np.nan
    )

    # 3. total_charges has negative values (data entry error) → outliers
    df.loc[RNG.choice(n, 15, replace=False), "total_charges"] = RNG.uniform(-500, -10, 15)

    # 4. customer_id is useless (should be dropped)
    # 5. city has 150 unique values → high cardinality

    return df


CHURN_ISSUES = [
    {
        "issue_id": "missing_age",
        "issue_type": "missing",
        "column": "age",
        "severity": "medium",
        "description": "Column 'age' has ~15% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "missing_monthly_charges",
        "issue_type": "missing",
        "column": "monthly_charges",
        "severity": "medium",
        "description": "Column 'monthly_charges' has ~12% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "dtype_tenure",
        "issue_type": "dtype",
        "column": "tenure_months",
        "severity": "high",
        "description": "Column 'tenure_months' stored as string ('X months') instead of numeric.",
        "recommended_action": "fix_dtypes",
    },
    {
        "issue_id": "outlier_total_charges",
        "issue_type": "outlier",
        "column": "total_charges",
        "severity": "medium",
        "description": "Column 'total_charges' has negative values (impossible — data entry errors).",
        "recommended_action": "clip_outliers",
    },
    {
        "issue_id": "cardinality_city",
        "issue_type": "cardinality",
        "column": "city",
        "severity": "low",
        "description": "Column 'city' has 150 unique values — too high for one-hot encoding.",
        "recommended_action": "reduce_cardinality",
    },
]

CHURN_META = {
    "task_id": 1,
    "name": "customer-churn-prep",
    "difficulty": "easy",
    "max_steps": 18,
    "target_column": "churn",
    "description": (
        "Prepare a customer churn dataset for binary classification. "
        "Profile the data, fix dtype errors, handle missing values, "
        "clip outliers, and reduce high-cardinality features. "
        "The dataset must be fully model-ready at the end."
    ),
    "known_issues": CHURN_ISSUES,
    "eda_min_actions": 4,
    "cleaning_required_fixes": ["missing_age", "missing_monthly_charges",
                                  "dtype_tenure", "outlier_total_charges"],
}


# ---------------------------------------------------------------------------
# Task 2 — Loan Default (medium)
# ---------------------------------------------------------------------------

def make_loan_dataset() -> pd.DataFrame:
    n = 1200

    age         = RNG.randint(21, 65, n).astype(float)
    income      = RNG.exponential(scale=50000, size=n) + 20000
    loan_amount = RNG.uniform(5000, 100000, n)
    loan_term   = RNG.choice([12, 24, 36, 48, 60], n)
    credit_score= RNG.randint(300, 850, n).astype(float)
    employment  = RNG.choice(["Employed", "Self-employed", "Unemployed"], n,
                              p=[0.65, 0.25, 0.10])
    purpose     = RNG.choice(["Home", "Auto", "Education", "Business", "Personal"], n)
    dti_ratio   = loan_amount / income  # debt-to-income

    # Target: default (imbalanced — 10:1)
    default_prob = (
        0.05
        + 0.15 * (employment == "Unemployed").astype(float)
        + 0.10 * (dti_ratio > 0.4).astype(float)
        - 0.05 * (credit_score / 850)
    )
    default = (RNG.rand(n) < default_prob).astype(int)

    df = pd.DataFrame({
        "loan_id":      [f"LOAN_{i:05d}" for i in range(n)],
        "age":          age,
        "income":       income,
        "loan_amount":  loan_amount,
        "loan_term":    loan_term,
        "credit_score": credit_score,
        "dti_ratio":    dti_ratio,
        "employment":   employment,
        "purpose":      purpose,
        "default":      default,
    })

    # ── Inject issues ──────────────────────────────────────────────────────

    # 1. LEAKY COLUMN: loan_status — directly derived from target
    df["loan_status"] = df["default"].map({0: "Current", 1: "Charged-off"})

    # 2. Missing values
    df.loc[RNG.choice(n, int(n * 0.10), replace=False), "credit_score"] = np.nan
    df.loc[RNG.choice(n, int(n * 0.08), replace=False), "income"]       = np.nan

    # 3. Duplicate rows (50 exact duplicates)
    dup_idx = RNG.choice(n, 50, replace=False)
    df = pd.concat([df, df.iloc[dup_idx]], ignore_index=True)

    # 4. High correlation: dti_ratio is directly computed from loan_amount/income
    #    (agent should detect loan_amount & dti_ratio are ~redundant)

    # 5. income is right-skewed (exponential) — needs log transform

    # 6. Class imbalance: ~9:1
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


LOAN_ISSUES = [
    {
        "issue_id": "leaky_loan_status",
        "issue_type": "leakage",
        "column": "loan_status",
        "severity": "critical",
        "description": "'loan_status' directly encodes the target 'default'. Must be removed.",
        "recommended_action": "remove_leaky_col",
    },
    {
        "issue_id": "imbalance_default",
        "issue_type": "imbalance",
        "column": "default",
        "severity": "high",
        "description": "Target 'default' is ~9:1 imbalanced. Needs SMOTE or undersampling.",
        "recommended_action": "handle_imbalance_smote",
    },
    {
        "issue_id": "duplicate_rows",
        "issue_type": "duplicate",
        "column": None,
        "severity": "medium",
        "description": "50 duplicate rows detected in the dataset.",
        "recommended_action": "drop_duplicates",
    },
    {
        "issue_id": "missing_credit_score",
        "issue_type": "missing",
        "column": "credit_score",
        "severity": "medium",
        "description": "Column 'credit_score' has ~10% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "missing_income",
        "issue_type": "missing",
        "column": "income",
        "severity": "medium",
        "description": "Column 'income' has ~8% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "skewed_income",
        "issue_type": "outlier",
        "column": "income",
        "severity": "low",
        "description": "Column 'income' is heavily right-skewed. Log transform recommended.",
        "recommended_action": "log_transform",
    },
    {
        "issue_id": "corr_dti_loan",
        "issue_type": "correlation",
        "column": "dti_ratio",
        "severity": "low",
        "description": "'dti_ratio' is derived from 'loan_amount' / 'income' — high redundancy.",
        "recommended_action": "drop_correlated_col",
    },
]

LOAN_META = {
    "task_id": 2,
    "name": "loan-default-prep",
    "difficulty": "medium",
    "max_steps": 22,
    "target_column": "default",
    "description": (
        "Prepare a loan default dataset. Critical: detect and remove the leaky "
        "'loan_status' column before any encoding. Handle class imbalance (9:1), "
        "drop duplicates, fix missing values, and address the skewed income distribution."
    ),
    "known_issues": LOAN_ISSUES,
    "eda_min_actions": 5,
    "cleaning_required_fixes": ["leaky_loan_status", "duplicate_rows",
                                  "missing_credit_score", "missing_income"],
}


# ---------------------------------------------------------------------------
# Task 3 — Medical Readmission (hard)
# ---------------------------------------------------------------------------

def make_medical_dataset() -> pd.DataFrame:
    n = 1500

    age            = RNG.randint(18, 95, n).astype(float)
    bmi            = RNG.normal(28, 6, n)
    hba1c          = RNG.normal(7.5, 2.0, n)       # blood glucose proxy
    num_meds       = RNG.randint(1, 20, n).astype(float)
    num_procedures = RNG.randint(0, 10, n).astype(float)
    los_days       = RNG.exponential(scale=4, size=n) + 1  # length of stay, skewed
    num_diagnoses  = RNG.randint(1, 15, n).astype(float)
    gender         = RNG.choice(["Male", "Female"], n)
    race           = RNG.choice(["Caucasian", "AfricanAmerican", "Hispanic",
                                  "Asian", "Other"], n, p=[0.55, 0.20, 0.12, 0.08, 0.05])
    admission_type = RNG.choice(["Emergency", "Elective", "Newborn"], n,
                                  p=[0.60, 0.35, 0.05])
    discharge_to   = RNG.choice(["Home", "SNF", "Rehab", "AMA", "Expired"], n,
                                  p=[0.60, 0.15, 0.12, 0.08, 0.05])
    payer          = RNG.choice(["Medicare", "Medicaid", "Private", "Self"], n,
                                  p=[0.45, 0.25, 0.20, 0.10])

    # Target: readmitted within 30 days (severely imbalanced ~12:1)
    readmit_prob = (
        0.06
        + 0.08 * (num_meds / 20)
        + 0.06 * (hba1c > 9).astype(float)
        + 0.05 * (age > 70).astype(float)
        - 0.03 * (discharge_to == "Home").astype(float)
    )
    readmitted = (RNG.rand(n) < readmit_prob).astype(int)

    df = pd.DataFrame({
        "patient_id":     [f"PT_{i:06d}" for i in range(n)],
        "age":            age,
        "gender":         gender,
        "race":           race,
        "bmi":            bmi,
        "hba1c_result":   hba1c,
        "num_medications":num_meds,
        "num_procedures": num_procedures,
        "num_diagnoses":  num_diagnoses,
        "los_days":       los_days,
        "admission_type": admission_type,
        "discharge_to":   discharge_to,
        "payer_code":     payer,
        "readmitted":     readmitted,
    })

    # ── Inject issues ──────────────────────────────────────────────────────

    # 1. LEAKY COL 1: discharge_summary encodes outcome
    df["discharge_summary"] = df["readmitted"].map(
        {0: "Stable discharge", 1: "High risk — follow up required"}
    )

    # 2. LEAKY COL 2: days_to_readmission (0 for non-readmitted → direct leakage)
    df["days_to_readmission"] = df["readmitted"].apply(
        lambda x: RNG.randint(1, 30) if x == 1 else 0
    )

    # 3. Missing values — heavy
    df.loc[RNG.choice(n, int(n * 0.22), replace=False), "bmi"]          = np.nan
    df.loc[RNG.choice(n, int(n * 0.18), replace=False), "hba1c_result"] = np.nan
    df.loc[RNG.choice(n, int(n * 0.12), replace=False), "num_medications"] = np.nan
    df.loc[RNG.choice(n, int(n * 0.15), replace=False), "race"]         = np.nan

    # 4. Wrong dtype: age stored as string range
    bins = [0, 30, 40, 50, 60, 70, 80, 100]
    labels = ["[0-30)", "[30-40)", "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80+)"]
    df["age"] = pd.cut(df["age"], bins=bins, labels=labels, right=False).astype(str)

    # 5. los_days is heavily right-skewed
    # 6. num_medications and num_procedures are highly correlated with num_diagnoses
    # 7. Severe class imbalance: ~12:1

    return df


MEDICAL_ISSUES = [
    {
        "issue_id": "leaky_discharge_summary",
        "issue_type": "leakage",
        "column": "discharge_summary",
        "severity": "critical",
        "description": "'discharge_summary' encodes the readmission outcome directly.",
        "recommended_action": "remove_leaky_col",
    },
    {
        "issue_id": "leaky_days_to_readmission",
        "issue_type": "leakage",
        "column": "days_to_readmission",
        "severity": "critical",
        "description": "'days_to_readmission' is 0 for all non-readmitted patients — direct leakage.",
        "recommended_action": "remove_leaky_col",
    },
    {
        "issue_id": "missing_bmi",
        "issue_type": "missing",
        "column": "bmi",
        "severity": "high",
        "description": "Column 'bmi' has ~22% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "missing_hba1c",
        "issue_type": "missing",
        "column": "hba1c_result",
        "severity": "high",
        "description": "Column 'hba1c_result' has ~18% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "missing_num_medications",
        "issue_type": "missing",
        "column": "num_medications",
        "severity": "medium",
        "description": "Column 'num_medications' has ~12% missing values.",
        "recommended_action": "fill_missing_median",
    },
    {
        "issue_id": "missing_race",
        "issue_type": "missing",
        "column": "race",
        "severity": "medium",
        "description": "Column 'race' has ~15% missing values.",
        "recommended_action": "fill_missing_mode",
    },
    {
        "issue_id": "dtype_age",
        "issue_type": "dtype",
        "column": "age",
        "severity": "high",
        "description": "'age' is stored as string age-range bins instead of numeric.",
        "recommended_action": "fix_dtypes",
    },
    {
        "issue_id": "skewed_los",
        "issue_type": "outlier",
        "column": "los_days",
        "severity": "medium",
        "description": "'los_days' is heavily right-skewed (exponential dist). Log transform recommended.",
        "recommended_action": "log_transform",
    },
    {
        "issue_id": "imbalance_readmitted",
        "issue_type": "imbalance",
        "column": "readmitted",
        "severity": "high",
        "description": "Target 'readmitted' is ~12:1 imbalanced. SMOTE or undersampling required.",
        "recommended_action": "handle_imbalance_smote",
    },
    {
        "issue_id": "corr_procedures_diagnoses",
        "issue_type": "correlation",
        "column": "num_procedures",
        "severity": "low",
        "description": "'num_procedures' is highly correlated with 'num_diagnoses' (r>0.85).",
        "recommended_action": "drop_correlated_col",
    },
]

MEDICAL_META = {
    "task_id": 3,
    "name": "medical-readmission-prep",
    "difficulty": "hard",
    "max_steps": 28,
    "target_column": "readmitted",
    "description": (
        "Prepare a hospital readmission dataset. Two leaky columns must be detected "
        "and removed. Handle severe class imbalance (12:1), heavy missing values in "
        "clinical features, wrong dtype in 'age', skewed distributions, and correlated "
        "features. This is the hardest data preparation task — the agent must catch "
        "all critical issues or the grader will penalize heavily."
    ),
    "known_issues": MEDICAL_ISSUES,
    "eda_min_actions": 6,
    "cleaning_required_fixes": [
        "leaky_discharge_summary", "leaky_days_to_readmission",
        "missing_bmi", "missing_hba1c", "missing_num_medications",
        "missing_race", "dtype_age",
    ],
}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[int, Dict[str, Any]] = {
    1: {**CHURN_META,   "generator": make_churn_dataset},
    2: {**LOAN_META,    "generator": make_loan_dataset},
    3: {**MEDICAL_META, "generator": make_medical_dataset},
}

def get_task(task_id: int) -> Dict[str, Any]:
    if task_id not in TASK_REGISTRY:
        raise ValueError(f"task_id must be 1, 2, or 3. Got {task_id}.")
    meta = TASK_REGISTRY[task_id].copy()
    meta["dataframe"] = meta["generator"]()
    return meta