"""
environment.py
==============
Full 4-phase Data Preparation Pipeline environment.

Phase gating (HARD):
  - Cleaning actions blocked until ≥ N EDA actions done
  - Engineering actions blocked until critical cleaning issues resolved
  - Validation actions blocked until all categoricals encoded + numerics scaled

Reward = phase_order_delta + issue_fix_delta + readiness_delta
       + phase_bonus - step_cost - invalid_penalty
"""

from __future__ import annotations

import copy
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import (
    Action, ActionType, ColumnStat, DataIssue, EnvironmentState,
    Observation, Reward, ResetResponse, StepResponse,
    PHASE_ACTIONS, get_action_phase,
)
from dataset_generator import get_task, TASK_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_COST        = -0.005
INVALID_PENALTY  = -0.05
PHASE_BONUS      = 0.10
ISSUE_FIX_REWARD = 0.08
CRITICAL_FIX_REWARD = 0.15
PHASE_ORDER      = ["eda", "cleaning", "engineering", "validation"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series)

def _is_categorical(series: pd.Series) -> bool:
    dtype_str = str(series.dtype).lower()
    return (pd.api.types.is_object_dtype(series)
            or pd.api.types.is_categorical_dtype(series)
            or "string" in dtype_str)

def _infer_dtype(series: pd.Series) -> str:
    if _is_numeric(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    # Check if it looks like numbers stored as strings
    sample = series.dropna().astype(str).head(50)
    try:
        sample.str.extract(r"(\d+\.?\d*)")[0].dropna().astype(float)
        if len(sample) > 5:
            return "numeric"  # string-encoded numbers
    except Exception:
        pass
    return "categorical"

def _compute_column_stats(df: pd.DataFrame, target: str,
                           leaky_cols: List[str]) -> List[ColumnStat]:
    stats = []
    numeric_cols = [c for c in df.columns if _is_numeric(df[c]) and c != target]

    # Compute pairwise correlations for numeric cols
    corr_pairs: Dict[str, str] = {}
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr().abs()
        for i, c1 in enumerate(numeric_cols):
            for c2 in numeric_cols[i+1:]:
                if corr_matrix.loc[c1, c2] > 0.85:
                    corr_pairs[c1] = c2
                    corr_pairs[c2] = c1

    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        dtype = "numeric" if _is_numeric(series) else "categorical"
        inferred = _infer_dtype(series)
        n_miss  = int(series.isna().sum())
        n_uniq  = int(series.nunique(dropna=True))

        skewness = None
        mean = std = None
        is_skewed = False
        has_outliers = False

        if _is_numeric(series) and not pd.api.types.is_bool_dtype(series):
            clean = series.dropna().astype(float)
            if len(clean) > 10:
                mean = round(float(clean.mean()), 3)
                std  = round(float(clean.std()),  3)
                skewness = round(float(clean.skew()), 3)
                is_skewed = abs(skewness) > 1.0
                q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                iqr = q3 - q1
                has_outliers = bool(((clean < q1 - 1.5*iqr) | (clean > q3 + 1.5*iqr)).sum() > 5)

        top_cats = None
        if _is_categorical(series):
            top_cats = series.value_counts(dropna=True).head(5).index.tolist()
            top_cats = [str(v) for v in top_cats]

        stats.append(ColumnStat(
            name               = col,
            dtype              = dtype,
            inferred_dtype     = inferred,
            missing_count      = n_miss,
            missing_pct        = round(n_miss / max(len(series), 1), 3),
            unique_count       = n_uniq,
            is_leaky           = col in leaky_cols,
            is_high_cardinality= n_uniq > 50 and _is_categorical(series),
            is_skewed          = is_skewed,
            has_outliers       = has_outliers,
            correlated_with    = corr_pairs.get(col),
            mean               = mean,
            std                = std,
            skewness           = skewness,
            top_categories     = top_cats,
        ))
    return stats


def _data_readiness_score(df: pd.DataFrame, target: str,
                           issues: List[DataIssue]) -> float:
    """
    Heuristic 0–1 score: how model-ready is the data right now?
    Checks: no nulls, no raw categoricals, no leaky cols, no extreme skew.
    """
    score = 1.0
    feature_cols = [c for c in df.columns if c != target]
    if not feature_cols:
        return 0.0

    # Penalize missing values
    total_missing = df[feature_cols].isnull().sum().sum()
    total_cells   = len(df) * len(feature_cols)
    score -= 0.30 * (total_missing / max(total_cells, 1))

    # Penalize unencoded categorical columns
    n_cats = sum(1 for c in feature_cols if _is_categorical(df[c]))
    score -= 0.25 * min(n_cats / max(len(feature_cols), 1), 1.0)

    # Penalize unresolved critical issues
    unresolved_critical = sum(
        1 for iss in issues
        if not iss.resolved and iss.severity == "critical"
    )
    score -= 0.20 * unresolved_critical

    # Penalize unresolved high issues
    unresolved_high = sum(
        1 for iss in issues
        if not iss.resolved and iss.severity == "high"
    )
    score -= 0.08 * unresolved_high

    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Core Environment
# ---------------------------------------------------------------------------

class DataPrepEnv:

    def __init__(self):
        self._task_id: int = 1
        self._df: Optional[pd.DataFrame] = None
        self._target: str = ""
        self._max_steps: int = 18
        self._step: int = 0
        self._done: bool = True
        self._phase: str = "eda"          # current unlocked phase
        self._phase_progress: Dict[str, int] = {p: 0 for p in PHASE_ORDER}
        self._eda_min: int = 4
        self._issues: List[DataIssue] = []
        self._leaky_cols: List[str] = []
        self._pipeline: List[Dict[str, Any]] = []
        self._cumulative_reward: float = 0.0
        self._phase_order_score: float = 1.0   # starts perfect, penalized on violations
        self._eda_done_actions: set = set()
        self._cleaning_required: List[str] = []
        self._split_done: bool = False

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    
    def reset(self, task_id: int = 1) -> ResetResponse:
        meta = get_task(task_id)
        self._task_id     = task_id
        self._df          = meta["dataframe"].copy()
        self._target      = meta["target_column"]
        self._max_steps   = meta["max_steps"]
        self._eda_min     = meta["eda_min_actions"]
        self._cleaning_required = meta["cleaning_required_fixes"]

        self._step             = 0
        self._done             = False
        self._phase            = "eda"
        self._phase_progress   = {p: 0 for p in PHASE_ORDER}
        self._pipeline         = []
        self._cumulative_reward= 0.0
        self._phase_order_score= 1.0
        self._eda_done_actions = set()
        self._split_done       = False

        # Load known issues (unresolved)
        self._issues = [
            DataIssue(**iss) for iss in meta["known_issues"]
        ]
        self._leaky_cols = [
            iss["column"] for iss in meta["known_issues"]
            if iss["issue_type"] == "leakage" and iss["column"]
        ]

        obs = self._build_obs(
            success=True,
            message=f"Episode started. Dataset loaded: {self._df.shape[0]} rows × "
                    f"{self._df.shape[1]} cols. Start with EDA actions to profile the data.",
        )
        return ResetResponse(observation=obs, info={"task_id": task_id})

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, action: Action) -> StepResponse:
        if self._done:
            raise RuntimeError("Episode done. Call reset() first.")

        self._step += 1
        action_type = str(action.type)
        action_phase = get_action_phase(action_type)

        prev_readiness = _data_readiness_score(self._df, self._target, self._issues)

        # ── Phase gate check ──────────────────────────────────────────
        blocked, block_msg = self._check_phase_gate(action_type, action_phase)
        if blocked:
            reward = Reward(
                value           = INVALID_PENALTY,
                invalid_penalty = INVALID_PENALTY,
                step_cost       = 0.0,
                cumulative_reward = round(self._cumulative_reward + INVALID_PENALTY, 4),
            )
            self._cumulative_reward += INVALID_PENALTY
            # Penalize phase order score
            self._phase_order_score = max(0.0, self._phase_order_score - 0.05)
            obs = self._build_obs(success=False, message=block_msg)
            self._done = self._step >= self._max_steps
            obs.done = self._done
            return StepResponse(observation=obs, reward=reward, done=self._done)

        # ── Apply action ──────────────────────────────────────────────
        success, message, issues_fixed = self._apply(action, action_type, action_phase)

        # Track EDA actions
        if action_phase == "eda" and success:
            self._eda_done_actions.add(action_type)
            self._phase_progress["eda"] += 1

        elif success and action_phase in ("cleaning", "engineering", "validation"):
            self._phase_progress[action_phase] += 1

        # Always try to advance phase after every successful action
        if success:
            self._update_phase()

        if success:
            self._pipeline.append({**action.dict(exclude_none=True),
                                    "phase": action_phase})

        # ── Compute reward ────────────────────────────────────────────
        new_readiness = _data_readiness_score(self._df, self._target, self._issues)
        readiness_delta = round(new_readiness - prev_readiness, 4)

        issue_fix_delta = 0.0
        for iid in issues_fixed:
            iss = next((i for i in self._issues if i.issue_id == iid), None)
            if iss:
                issue_fix_delta += CRITICAL_FIX_REWARD if iss.severity == "critical" else ISSUE_FIX_REWARD

        phase_bonus = 0.0
        if self._phase_just_completed(action_phase):
            phase_bonus = PHASE_BONUS

        invalid_p = 0.0 if success else INVALID_PENALTY
        step_cost  = STEP_COST if success else 0.0
        total = round(readiness_delta + issue_fix_delta + phase_bonus
                      + step_cost + invalid_p, 4)

        reward = Reward(
            value             = total,
            readiness_delta   = readiness_delta,
            issue_fix_delta   = issue_fix_delta,
            step_cost         = step_cost,
            phase_bonus       = phase_bonus,
            invalid_penalty   = invalid_p,
            cumulative_reward = round(self._cumulative_reward + total, 4),
        )
        self._cumulative_reward += total

        # ── Done check ────────────────────────────────────────────────
        budget_done = self._step >= self._max_steps
        agent_done  = action_type == "finish"
        self._done  = budget_done or agent_done

        obs = self._build_obs(success=success, message=message)
        obs.done = self._done

        return StepResponse(
            observation = obs,
            reward      = reward,
            done        = self._done,
            info        = {"grader_score": self._grader_score() if self._done else None},
        )

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def state(self) -> EnvironmentState:
        readiness = _data_readiness_score(self._df, self._target, self._issues)
        n_found   = len(self._issues)
        n_resolved= sum(1 for i in self._issues if i.resolved)
        coverage  = round(n_resolved / max(n_found, 1), 4)

        return EnvironmentState(
            task_id              = self._task_id,
            step                 = self._step,
            done                 = self._done,
            current_phase        = self._phase,
            issues_found         = n_found,
            issues_resolved      = n_resolved,
            data_readiness_score = round(readiness, 4),
            phase_order_score    = round(self._phase_order_score, 4),
            issue_coverage_score = coverage,
            pipeline_steps       = copy.deepcopy(self._pipeline),
            cumulative_reward    = round(self._cumulative_reward, 4),
            grader_score         = self._grader_score() if self._done else None,
        )

    # ------------------------------------------------------------------
    # Internal: phase gating
    # ------------------------------------------------------------------

    def _check_phase_gate(self, action_type: str, action_phase: str) -> Tuple[bool, str]:
        phase_order_idx = {p: i for i, p in enumerate(PHASE_ORDER)}
        current_idx     = phase_order_idx.get(self._phase, 0)
        action_idx      = phase_order_idx.get(action_phase, 0)

        # Always allow EDA actions
        if action_phase == "eda":
            return False, ""

        # Cleaning: needs min EDA actions done
        if action_phase == "cleaning":
            if len(self._eda_done_actions) < self._eda_min:
                return True, (
                    f"PHASE GATE: You need to run at least {self._eda_min} different EDA "
                    f"actions before cleaning. Done so far: {len(self._eda_done_actions)}. "
                    f"Try: profile_dataset, check_missing, detect_dtypes, detect_leakage, "
                    f"check_class_balance, detect_outliers."
                )

        # Engineering: needs critical issues resolved
        if action_phase == "engineering":
            if len(self._eda_done_actions) < self._eda_min:
                return True, "PHASE GATE: Complete EDA before feature engineering."
            unresolved_critical = [
                i for i in self._issues
                if not i.resolved and i.severity == "critical"
            ]
            if unresolved_critical:
                cols = [i.column or i.issue_id for i in unresolved_critical]
                return True, (
                    f"PHASE GATE: Resolve all CRITICAL issues before feature engineering. "
                    f"Unresolved: {cols}. These are data quality blockers."
                )

        # Validation: needs all meaningful categoricals encoded
        # Skip pure ID columns (unique > 50% of rows) — agent should drop those separately
        if action_phase == "validation":
            unencoded = [
                c for c in self._df.columns
                if c != self._target
                and _is_categorical(self._df[c])
                and self._df[c].nunique(dropna=True) < len(self._df) * 0.5
            ]
            if unencoded:
                return True, (
                    f"PHASE GATE: Encode all categorical columns before validation. "
                    f"Still raw: {unencoded[:5]}. Use encode_onehot or encode_label."
                )

        return False, ""

    def _update_phase(self):
        """Advance phase based on progress."""
        if self._phase == "eda" and len(self._eda_done_actions) >= self._eda_min:
            self._phase = "cleaning"

        if self._phase == "cleaning":
            all_critical_resolved = all(
                i.resolved for i in self._issues if i.severity == "critical"
            )
            if all_critical_resolved and self._phase_progress["cleaning"] >= 2:
                self._phase = "engineering"

        if self._phase == "engineering":
            # Check no meaningful categoricals remain (skip ID-like cols)
            unencoded_cats = [
                c for c in self._df.columns
                if c != self._target
                and _is_categorical(self._df[c])
                and self._df[c].nunique(dropna=True) < len(self._df) * 0.5
            ]
            if not unencoded_cats and self._phase_progress["engineering"] >= 2:
                self._phase = "validation"

    def _phase_just_completed(self, action_phase: str) -> bool:
        """Returns True if this action pushed us into the next phase."""
        phase_idx = PHASE_ORDER.index(action_phase) if action_phase in PHASE_ORDER else -1
        if phase_idx < 0 or phase_idx >= len(PHASE_ORDER) - 1:
            return False
        next_phase = PHASE_ORDER[phase_idx + 1]
        return self._phase == next_phase and self._phase_progress[action_phase] == 1

    # ------------------------------------------------------------------
    # Internal: action dispatcher
    # ------------------------------------------------------------------

    def _apply(self, action: Action, atype: str,
               aphase: str) -> Tuple[bool, str, List[str]]:
        """Returns (success, message, list_of_issue_ids_fixed)."""
        col    = action.column
        issues_fixed: List[str] = []

        def _col_exists(c: Optional[str]) -> Tuple[bool, str]:
            if not c:
                return False, f"Action '{atype}' requires a 'column' field."
            if c not in self._df.columns:
                return False, f"Column '{c}' not found. Available: {list(self._df.columns)[:8]}"
            if c == self._target:
                return False, f"Cannot modify target column '{c}'."
            return True, ""

        # ── EDA actions (read-only, update issue list) ─────────────────

        if atype == "profile_dataset":
            shape = self._df.shape
            dtypes = self._df.dtypes.value_counts().to_dict()
            return True, (
                f"Dataset: {shape[0]} rows × {shape[1]} cols. "
                f"Memory: {self._df.memory_usage(deep=True).sum() / 1024:.1f} KB. "
                f"Dtypes: {dtypes}"
            ), []

        if atype == "detect_dtypes":
            wrong = []
            for c in self._df.columns:
                if c == self._target: continue
                if _is_categorical(self._df[c]) and _infer_dtype(self._df[c]) == "numeric":
                    wrong.append(c)
            msg = f"Dtype issues found: {wrong}" if wrong else "All dtypes look correct."
            return True, msg, []

        if atype == "check_missing":
            missing = self._df.isnull().sum()
            missing = missing[missing > 0].to_dict()
            pct = {k: f"{v/len(self._df)*100:.1f}%" for k, v in missing.items()}
            return True, f"Missing values: {pct}" if pct else "No missing values.", []

        if atype == "check_class_balance":
            counts = self._df[self._target].value_counts().to_dict()
            vals   = list(counts.values())
            ratio  = round(max(vals) / max(min(vals), 1), 1)
            return True, f"Class distribution: {counts}. Imbalance ratio: {ratio}:1", []

        if atype == "detect_leakage":
            found = [c for c in self._df.columns if c in self._leaky_cols]
            if found:
                return True, (
                    f"LEAKAGE DETECTED: {found} are highly correlated with the target "
                    f"and would cause data leakage. Remove before training."
                ), []
            return True, "No obvious leaky columns detected.", []

        if atype == "check_correlations":
            num_cols = [c for c in self._df.select_dtypes(include=[np.number]).columns
                        if c != self._target and not pd.api.types.is_bool_dtype(self._df[c])]
            if len(num_cols) < 2:
                return True, "Not enough numeric columns for correlation analysis.", []
            corr = self._df[num_cols].corr().abs()
            high = []
            for i in range(len(num_cols)):
                for j in range(i+1, len(num_cols)):
                    if corr.iloc[i, j] > 0.85:
                        high.append(f"{num_cols[i]} ↔ {num_cols[j]}: {corr.iloc[i,j]:.2f}")
            return True, (f"High correlations: {high}" if high else "No high correlations found."), []

        if atype == "detect_outliers":
            results = []
            for c in self._df.select_dtypes(include=[np.number]).columns:
                if c == self._target: continue
                if pd.api.types.is_bool_dtype(self._df[c]): continue
                col_clean = self._df[c].dropna().astype(float)
                if len(col_clean) < 10: continue
                q1, q3 = col_clean.quantile(0.25), col_clean.quantile(0.75)
                iqr = q3 - q1
                n_out = int(((col_clean < q1-1.5*iqr) | (col_clean > q3+1.5*iqr)).sum())
                if n_out > 5:
                    results.append(f"{c}: {n_out} outliers")
            return True, (f"Outliers: {results}" if results else "No significant outliers."), []

        if atype == "detect_duplicates":
            n_dup = int(self._df.duplicated().sum())
            return True, f"Duplicate rows: {n_dup}", []

        # ── Cleaning actions ───────────────────────────────────────────

        if atype == "fix_dtypes":
            fixed = []
            # ID-like columns to never convert (high cardinality strings with no numeric meaning)
            id_like = [c for c in self._df.columns
                       if c != self._target and _is_categorical(self._df[c])
                       and self._df[c].nunique(dropna=True) > len(self._df) * 0.5]

            for c in self._df.columns:
                if c == self._target: continue
                if c in id_like: continue
                if _is_categorical(self._df[c]):
                    # Only convert if the string contains a leading number pattern
                    # e.g. "24 months" → 24, but NOT "City_12" or "LOAN_00001"
                    sample = self._df[c].dropna().astype(str).head(20)
                    # Must start with digit (e.g. "24 months", "[50-60)")
                    starts_numeric = sample.str.match(r"^\[?\d").sum()
                    if starts_numeric < len(sample) * 0.7:
                        continue
                    extracted = self._df[c].astype(str).str.extract(r"(\d+\.?\d*)")[0]
                    n_valid = extracted.notna().sum()
                    if n_valid > len(self._df) * 0.5:
                        self._df[c] = pd.to_numeric(extracted, errors="coerce")
                        fixed.append(c)
            if fixed:
                issues_fixed += self._resolve_issues_by_type("dtype")
                return True, f"Fixed dtypes for: {fixed}", issues_fixed
            return False, "No dtype issues found to fix.", []

        if atype == "drop_duplicates":
            before = len(self._df)
            self._df = self._df.drop_duplicates().reset_index(drop=True)
            after  = len(self._df)
            if before > after:
                issues_fixed += self._resolve_issues_by_type("duplicate")
                return True, f"Dropped {before - after} duplicate rows.", issues_fixed
            return True, "No duplicates found.", []

        if atype == "remove_leaky_col":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            if col not in self._leaky_cols:
                return False, f"'{col}' is not flagged as leaky. Run detect_leakage first.", []
            self._df = self._df.drop(columns=[col])
            self._leaky_cols = [c for c in self._leaky_cols if c != col]
            issues_fixed += self._resolve_issues_by_column(col, "leakage")
            return True, f"Removed leaky column '{col}'.", issues_fixed

        if atype in ("fill_missing_median", "fill_missing_mode",
                     "fill_missing_knn", "fill_missing_constant"):
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            n_before = int(self._df[col].isna().sum())
            if n_before == 0:
                return False, f"'{col}' has no missing values.", []

            if atype == "fill_missing_median":
                if not _is_numeric(self._df[col]):
                    return False, f"fill_missing_median requires numeric column.", []
                self._df[col] = self._df[col].fillna(self._df[col].median())
            elif atype == "fill_missing_mode":
                mode_v = self._df[col].mode()
                if len(mode_v) == 0: return False, f"Cannot compute mode for '{col}'.", []
                self._df[col] = self._df[col].fillna(mode_v[0])
            elif atype == "fill_missing_knn":
                if not _is_numeric(self._df[col]):
                    return False, "fill_missing_knn requires numeric column.", []
                # Simplified: median fill with slight jitter for KNN approximation
                median_val = self._df[col].median()
                std_val = self._df[col].std() * 0.1
                fill_vals = np.random.normal(median_val, std_val,
                                              self._df[col].isna().sum())
                self._df.loc[self._df[col].isna(), col] = fill_vals
            elif atype == "fill_missing_constant":
                fill = 0 if _is_numeric(self._df[col]) else "missing"
                self._df[col] = self._df[col].fillna(fill)

            issues_fixed += self._resolve_issues_by_column(col, "missing")
            return True, f"Filled {n_before} missing in '{col}'.", issues_fixed

        if atype == "clip_outliers":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            if not _is_numeric(self._df[col]):
                return False, f"clip_outliers requires numeric column.", []
            q1, q3 = self._df[col].quantile(0.25), self._df[col].quantile(0.75)
            iqr = q3 - q1
            lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
            n_clipped = int(((self._df[col] < lo) | (self._df[col] > hi)).sum())
            self._df[col] = self._df[col].clip(lo, hi)
            issues_fixed += self._resolve_issues_by_column(col, "outlier")
            return True, f"Clipped {n_clipped} outliers in '{col}'.", issues_fixed

        if atype == "drop_high_missing_col":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            pct = self._df[col].isna().mean()
            if pct < 0.40:
                return False, f"'{col}' has only {pct:.0%} missing. Threshold is 40%.", []
            self._df = self._df.drop(columns=[col])
            issues_fixed += self._resolve_issues_by_column(col, "missing")
            return True, f"Dropped '{col}' ({pct:.0%} missing).", issues_fixed

        if atype == "reduce_cardinality":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            if not _is_categorical(self._df[col]):
                return False, f"'{col}' is not categorical.", []
            top = self._df[col].value_counts().head(10).index
            self._df[col] = self._df[col].apply(lambda x: x if x in top else "Other")
            issues_fixed += self._resolve_issues_by_column(col, "cardinality")
            return True, f"Reduced '{col}' to top 10 categories + 'Other'.", issues_fixed

        if atype == "drop_correlated_col":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            self._df = self._df.drop(columns=[col])
            issues_fixed += self._resolve_issues_by_column(col, "correlation")
            return True, f"Dropped correlated column '{col}'.", issues_fixed

        # ── Feature Engineering ────────────────────────────────────────

        if atype in ("encode_onehot", "encode_label", "encode_target"):
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            if not _is_categorical(self._df[col]):
                return False, f"'{col}' is already numeric.", []

            self._df[col] = self._df[col].fillna("missing").astype(str)

            if atype == "encode_onehot":
                n_unique = self._df[col].nunique()
                if n_unique > 20:
                    return False, (f"'{col}' has {n_unique} unique values. "
                                   f"Use encode_label for high-cardinality columns."), []
                dummies = pd.get_dummies(self._df[col], prefix=col)
                self._df = pd.concat([self._df.drop(columns=[col]), dummies], axis=1)
                return True, f"One-hot encoded '{col}' → {dummies.shape[1]} cols.", []

            elif atype == "encode_label":
                le = LabelEncoder()
                self._df[col] = le.fit_transform(self._df[col])
                return True, f"Label encoded '{col}'.", []

            elif atype == "encode_target":
                means = self._df.groupby(col)[self._target].mean()
                self._df[col] = self._df[col].map(means)
                return True, f"Target encoded '{col}'.", []

        if atype in ("normalize_standard", "normalize_robust"):
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            if not _is_numeric(self._df[col]):
                return False, f"'{col}' is not numeric.", []
            clean = self._df[col].fillna(self._df[col].median())
            if atype == "normalize_standard":
                mean, std = clean.mean(), clean.std()
                if std == 0: return False, f"'{col}' has zero variance.", []
                self._df[col] = (clean - mean) / std
            else:
                med = clean.median()
                iqr = clean.quantile(0.75) - clean.quantile(0.25)
                if iqr == 0: return False, f"'{col}' has zero IQR.", []
                self._df[col] = (clean - med) / iqr
            return True, f"Applied {atype} to '{col}'.", []

        if atype == "log_transform":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            if not _is_numeric(self._df[col]):
                return False, f"'{col}' is not numeric.", []
            if self._df[col].dropna().min() < 0:
                return False, f"'{col}' has negative values. log1p requires non-negative.", []
            self._df[col] = np.log1p(self._df[col].fillna(self._df[col].median()))
            issues_fixed += self._resolve_issues_by_column(col, "outlier")
            return True, f"Applied log1p to '{col}'.", issues_fixed

        if atype == "feature_importance_rank":
            num_df = self._df.copy()
            for c in num_df.columns:
                if c == self._target: continue
                if _is_categorical(num_df[c]):
                    num_df[c] = LabelEncoder().fit_transform(
                        num_df[c].astype(str).fillna("missing"))
                else:
                    num_df[c] = num_df[c].fillna(num_df[c].median())
            X = num_df.drop(columns=[self._target])
            y = num_df[self._target]
            try:
                rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                imp = pd.Series(rf.feature_importances_, index=X.columns)
                top5 = imp.nlargest(5).to_dict()
                top5 = {k: round(v, 3) for k, v in top5.items()}
                return True, f"Top 5 features by importance: {top5}", []
            except Exception as e:
                return False, f"Feature importance failed: {e}", []

        if atype == "select_top_k":
            top_k = action.top_k or 10
            feat_cols = [c for c in self._df.columns if c != self._target]
            if len(feat_cols) <= top_k:
                return False, f"Only {len(feat_cols)} features — nothing to trim.", []
            num_df = self._df[feat_cols].copy()
            for c in num_df.columns:
                if _is_categorical(num_df[c]):
                    num_df[c] = LabelEncoder().fit_transform(
                        num_df[c].astype(str).fillna("missing"))
                else:
                    num_df[c] = num_df[c].fillna(num_df[c].median())
            variances = num_df.var().sort_values(ascending=False)
            keep = variances.head(top_k).index.tolist() + [self._target]
            self._df = self._df[keep]
            return True, f"Selected top {top_k} features by variance.", []

        if atype == "create_interaction":
            ok, msg = _col_exists(col)
            if not ok: return False, msg, []
            ok2, msg2 = _col_exists(action.column2)
            if not ok2: return False, msg2, []
            if not (_is_numeric(self._df[col]) and _is_numeric(self._df[action.column2])):
                return False, "Both columns must be numeric for interaction.", []
            new_col = f"{col}_x_{action.column2}"
            self._df[new_col] = self._df[col] * self._df[action.column2]
            return True, f"Created interaction feature '{new_col}'.", []

        # ── Validation ─────────────────────────────────────────────────

        if atype == "train_test_split":
            self._split_done = True
            n_test = int(len(self._df) * 0.2)
            return True, (
                f"Train/test split configured: {len(self._df)-n_test} train / "
                f"{n_test} test (stratified, random_state=42)."
            ), []

        if atype in ("handle_imbalance_smote", "handle_imbalance_undersample"):
            counts = self._df[self._target].value_counts()
            ratio  = counts.max() / max(counts.min(), 1)
            if ratio < 1.5:
                return False, "Class balance is already acceptable (ratio < 1.5).", []
            issues_fixed += self._resolve_issues_by_type("imbalance")
            strategy = "SMOTE oversampling" if "smote" in atype else "Random undersampling"
            return True, (
                f"{strategy} applied. Original ratio {ratio:.1f}:1 → balanced."
            ), issues_fixed

        if atype == "validate_no_leakage":
            remaining = [c for c in self._df.columns if c in self._leaky_cols]
            if remaining:
                return False, f"Leaky columns still present: {remaining}. Remove them first.", []
            return True, "Validation passed: no leaky columns detected in final dataset.", []

        if atype == "generate_data_report":
            n_issues = len(self._issues)
            n_resolved = sum(1 for i in self._issues if i.resolved)
            readiness = _data_readiness_score(self._df, self._target, self._issues)
            nulls = int(self._df.isnull().sum().sum())
            cats  = sum(1 for c in self._df.columns
                        if c != self._target and _is_categorical(self._df[c]))
            return True, (
                f"DATA REPORT — Shape: {self._df.shape} | "
                f"Nulls remaining: {nulls} | "
                f"Unencoded categoricals: {cats} | "
                f"Issues resolved: {n_resolved}/{n_issues} | "
                f"Readiness score: {readiness:.2f}"
            ), []

        if atype == "finish":
            return True, "Agent signalled pipeline complete.", []

        return False, f"Unknown action: {atype}", []

    # ------------------------------------------------------------------
    # Internal: issue resolution helpers
    # ------------------------------------------------------------------

    def _resolve_issues_by_column(self, col: str, issue_type: str) -> List[str]:
        fixed = []
        for iss in self._issues:
            if iss.column == col and iss.issue_type == issue_type and not iss.resolved:
                iss.resolved = True
                fixed.append(iss.issue_id)
        return fixed

    def _resolve_issues_by_type(self, issue_type: str) -> List[str]:
        fixed = []
        for iss in self._issues:
            if iss.issue_type == issue_type and not iss.resolved:
                iss.resolved = True
                fixed.append(iss.issue_id)
        return fixed

    # ------------------------------------------------------------------
    # Internal: build observation
    # ------------------------------------------------------------------

    def _build_obs(self, success: bool, message: str) -> Observation:
        meta = TASK_REGISTRY[self._task_id]
        readiness = _data_readiness_score(self._df, self._target, self._issues)
        n_found   = len(self._issues)
        n_resolved= sum(1 for i in self._issues if i.resolved)
        coverage  = round(n_resolved / max(n_found, 1), 4)
        eff       = round(1.0 - self._step / max(self._max_steps, 1), 4)

        eda_done  = len(self._eda_done_actions) >= self._eda_min
        clean_done= self._phase in ("engineering", "validation", "done")
        eng_done  = self._phase in ("validation", "done")

        return Observation(
            task_id              = self._task_id,
            task_description     = meta["description"],
            step                 = self._step,
            max_steps            = self._max_steps,
            steps_remaining      = self._max_steps - self._step,
            current_phase        = self._phase,
            phase_progress       = dict(self._phase_progress),
            eda_complete         = eda_done,
            cleaning_complete    = clean_done,
            engineering_complete = eng_done,
            n_rows               = len(self._df),
            n_cols               = len(self._df.columns),
            target_column        = self._target,
            columns              = _compute_column_stats(
                                       self._df, self._target, self._leaky_cols),
            issues_found         = list(self._issues),
            issues_resolved      = n_resolved,
            issues_total         = n_found,
            pipeline_steps       = copy.deepcopy(self._pipeline),
            data_readiness_score = round(readiness, 4),
            phase_order_score    = round(self._phase_order_score, 4),
            issue_coverage_score = coverage,
            efficiency_score     = eff,
            last_action_success  = success,
            last_action_message  = message,
        )

    # ------------------------------------------------------------------
    # Internal: grader score
    # ------------------------------------------------------------------

    def _grader_score(self) -> float:
        """
        Final grader score 0.0–1.0:
          35% issue coverage (critical issues weighted 2×)
          25% data readiness
          25% phase order score
          15% efficiency
        """
        n_issues  = len(self._issues)
        if n_issues == 0:
            coverage = 1.0
        else:
            # Weight critical issues 2×
            weighted_total    = sum(2 if i.severity == "critical" else 1 for i in self._issues)
            weighted_resolved = sum(
                (2 if i.severity == "critical" else 1)
                for i in self._issues if i.resolved
            )
            coverage = weighted_resolved / max(weighted_total, 1)

        readiness     = _data_readiness_score(self._df, self._target, self._issues)
        phase_order   = self._phase_order_score
        efficiency    = round(1.0 - self._step / max(self._max_steps, 1), 4)

        score = (
            0.35 * coverage
            + 0.25 * readiness
            + 0.25 * phase_order
            + 0.15 * efficiency
        )
        return round(float(np.clip(score, 0.0, 1.0)), 4)
    def _resolve_issues_by_column(self, col: str, issue_type: str) -> List[str]:
        fixed = []
        for iss in self._issues:
            if (
            iss.issue_type == issue_type
            and not iss.resolved
            and iss.column is not None
            and iss.column.strip().lower() == col.strip().lower()
        ):
                iss.resolved = True
                fixed.append(iss.issue_id)
        return fixed

    def _resolve_issues_by_type(self, issue_type: str) -> List[str]:
        fixed = []
        for iss in self._issues:
            if iss.issue_type == issue_type and not iss.resolved:
                iss.resolved = True
                fixed.append(iss.issue_id)
        return fixed