"""
models.py
=========
Type-safe contracts for the Data Preparation Pipeline Agent.
4-phase workflow: EDA → Cleaning → Feature Engineering → Validation
Phase gating enforced: agent must complete each phase before unlocking next.
"""

from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Phase definitions
# ---------------------------------------------------------------------------

class Phase(str, Enum):
    EDA         = "eda"
    CLEANING    = "cleaning"
    ENGINEERING = "engineering"
    VALIDATION  = "validation"
    DONE        = "done"


# ---------------------------------------------------------------------------
# Action types grouped by phase
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    # Phase 1: EDA
    PROFILE_DATASET     = "profile_dataset"
    DETECT_DTYPES       = "detect_dtypes"
    CHECK_MISSING       = "check_missing"
    CHECK_CLASS_BALANCE = "check_class_balance"
    DETECT_LEAKAGE      = "detect_leakage"
    CHECK_CORRELATIONS  = "check_correlations"
    DETECT_OUTLIERS     = "detect_outliers"
    DETECT_DUPLICATES   = "detect_duplicates"

    # Phase 2: Cleaning
    FIX_DTYPES               = "fix_dtypes"
    DROP_DUPLICATES          = "drop_duplicates"
    REMOVE_LEAKY_COL         = "remove_leaky_col"
    FILL_MISSING_MEDIAN      = "fill_missing_median"
    FILL_MISSING_MODE        = "fill_missing_mode"
    FILL_MISSING_KNN         = "fill_missing_knn"
    FILL_MISSING_CONSTANT    = "fill_missing_constant"
    CLIP_OUTLIERS            = "clip_outliers"
    DROP_HIGH_MISSING_COL    = "drop_high_missing_col"
    REDUCE_CARDINALITY       = "reduce_cardinality"
    DROP_CORRELATED_COL      = "drop_correlated_col"

    # Phase 3: Feature Engineering
    ENCODE_ONEHOT            = "encode_onehot"
    ENCODE_LABEL             = "encode_label"
    ENCODE_TARGET            = "encode_target"
    NORMALIZE_STANDARD       = "normalize_standard"
    NORMALIZE_ROBUST         = "normalize_robust"
    LOG_TRANSFORM            = "log_transform"
    FEATURE_IMPORTANCE_RANK  = "feature_importance_rank"
    SELECT_TOP_K             = "select_top_k"
    CREATE_INTERACTION       = "create_interaction"

    # Phase 4: Validation
    TRAIN_TEST_SPLIT             = "train_test_split"
    HANDLE_IMBALANCE_SMOTE       = "handle_imbalance_smote"
    HANDLE_IMBALANCE_UNDERSAMPLE = "handle_imbalance_undersample"
    VALIDATE_NO_LEAKAGE          = "validate_no_leakage"
    GENERATE_DATA_REPORT         = "generate_data_report"
    FINISH                       = "finish"


# Phase membership
PHASE_ACTIONS: Dict[str, List[str]] = {
    "eda": [
        "profile_dataset", "detect_dtypes", "check_missing",
        "check_class_balance", "detect_leakage", "check_correlations",
        "detect_outliers", "detect_duplicates",
    ],
    "cleaning": [
        "fix_dtypes", "drop_duplicates", "remove_leaky_col",
        "fill_missing_median", "fill_missing_mode", "fill_missing_knn",
        "fill_missing_constant", "clip_outliers", "drop_high_missing_col",
        "reduce_cardinality", "drop_correlated_col",
    ],
    "engineering": [
        "encode_onehot", "encode_label", "encode_target",
        "normalize_standard", "normalize_robust", "log_transform",
        "feature_importance_rank", "select_top_k", "create_interaction",
    ],
    "validation": [
        "train_test_split", "handle_imbalance_smote",
        "handle_imbalance_undersample", "validate_no_leakage",
        "generate_data_report", "finish",
    ],
}

def get_action_phase(action_type: str) -> str:
    for phase, actions in PHASE_ACTIONS.items():
        if action_type in actions:
            return phase
    return "done"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class Action(BaseModel):
    type: ActionType
    column: Optional[str]  = None
    column2: Optional[str] = None
    top_k: Optional[int]   = Field(None, ge=1)

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Data issue — what EDA flags
# ---------------------------------------------------------------------------

class DataIssue(BaseModel):
    issue_id: str
    issue_type: str        # missing|leakage|imbalance|dtype|duplicate|outlier|correlation|cardinality
    column: Optional[str]
    severity: str          # low|medium|high|critical
    description: str
    recommended_action: str
    resolved: bool = False


# ---------------------------------------------------------------------------
# Column stat
# ---------------------------------------------------------------------------

class ColumnStat(BaseModel):
    name: str
    dtype: str
    inferred_dtype: str
    missing_count: int
    missing_pct: float
    unique_count: int
    is_leaky: bool = False
    is_high_cardinality: bool = False
    is_skewed: bool = False
    has_outliers: bool = False
    correlated_with: Optional[str] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    skewness: Optional[float] = None
    top_categories: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    # Episode
    task_id: int
    task_description: str
    step: int
    max_steps: int
    steps_remaining: int

    # Phase tracking
    current_phase: str
    phase_progress: Dict[str, int]
    eda_complete: bool
    cleaning_complete: bool
    engineering_complete: bool

    # Dataset
    n_rows: int
    n_cols: int
    target_column: str
    columns: List[ColumnStat]

    # Issues
    issues_found: List[DataIssue]
    issues_resolved: int
    issues_total: int

    # Pipeline
    pipeline_steps: List[Dict[str, Any]]

    # Scores
    data_readiness_score: float
    phase_order_score: float
    issue_coverage_score: float
    efficiency_score: float

    # Feedback
    last_action: Optional[Dict[str, Any]] = None
    last_action_success: bool = True
    last_action_message: str = ""

    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    value: float
    phase_order_delta: float = 0.0
    issue_fix_delta: float   = 0.0
    readiness_delta: float   = 0.0
    step_cost: float         = 0.0
    phase_bonus: float       = 0.0
    invalid_penalty: float   = 0.0
    cumulative_reward: float = 0.0


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------

class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResponse(BaseModel):
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


class EnvironmentState(BaseModel):
    task_id: int
    step: int
    done: bool
    current_phase: str
    issues_found: int
    issues_resolved: int
    data_readiness_score: float
    phase_order_score: float
    issue_coverage_score: float
    pipeline_steps: List[Dict[str, Any]]
    cumulative_reward: float
    grader_score: Optional[float] = None