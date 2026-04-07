"""
grader.py
=========
Deterministic reference agent grader. No LLM needed.
Runs an optimal hand-crafted pipeline for each task and verifies grader scores.
"""

from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import Action, ActionType
from server.environment import DataPrepEnv


def grade_task1(env: DataPrepEnv, verbose: bool) -> float:
    """Customer Churn — easy."""
    env.reset(task_id=1)
    actions = [
        # EDA
        Action(type=ActionType.PROFILE_DATASET),
        Action(type=ActionType.CHECK_MISSING),
        Action(type=ActionType.DETECT_DTYPES),
        Action(type=ActionType.DETECT_OUTLIERS),
        Action(type=ActionType.CHECK_CLASS_BALANCE),
        # Cleaning
        Action(type=ActionType.REDUCE_CARDINALITY, column="city"),
        Action(type=ActionType.FIX_DTYPES),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="age"),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="monthly_charges"),
        Action(type=ActionType.CLIP_OUTLIERS, column="total_charges"),
        Action(type=ActionType.REDUCE_CARDINALITY, column="city"),
        # Engineering
        Action(type=ActionType.ENCODE_ONEHOT, column="contract"),
        Action(type=ActionType.ENCODE_ONEHOT, column="internet_service"),
        Action(type=ActionType.ENCODE_ONEHOT, column="payment_method"),
        Action(type=ActionType.ENCODE_LABEL,  column="city"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="age"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="monthly_charges"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="total_charges"),
        # Validation
        Action(type=ActionType.TRAIN_TEST_SPLIT),
        Action(type=ActionType.VALIDATE_NO_LEAKAGE),
        Action(type=ActionType.FINISH),
    ]
    return _run(env, actions, verbose)


def grade_task2(env: DataPrepEnv, verbose: bool) -> float:
    """Loan Default — medium."""
    env.reset(task_id=2)
    actions = [
        # EDA
        Action(type=ActionType.PROFILE_DATASET),
        Action(type=ActionType.DETECT_LEAKAGE),
        Action(type=ActionType.CHECK_MISSING),
        Action(type=ActionType.DETECT_DUPLICATES),
        Action(type=ActionType.CHECK_CLASS_BALANCE),
        Action(type=ActionType.CHECK_CORRELATIONS),
        # Cleaning
        Action(type=ActionType.REMOVE_LEAKY_COL, column="loan_status"),
        Action(type=ActionType.DROP_DUPLICATES),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="credit_score"),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="income"),
        Action(type=ActionType.LOG_TRANSFORM, column="income"),
        Action(type=ActionType.DROP_CORRELATED_COL, column="dti_ratio"),
        # Engineering
        Action(type=ActionType.ENCODE_ONEHOT, column="employment"),
        Action(type=ActionType.ENCODE_ONEHOT, column="purpose"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="age"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="loan_amount"),
        Action(type=ActionType.NORMALIZE_ROBUST,   column="credit_score"),
        # Validation
        Action(type=ActionType.TRAIN_TEST_SPLIT),
        Action(type=ActionType.HANDLE_IMBALANCE_SMOTE),
        Action(type=ActionType.VALIDATE_NO_LEAKAGE),
        Action(type=ActionType.FINISH),
    ]
    return _run(env, actions, verbose)


def grade_task3(env: DataPrepEnv, verbose: bool) -> float:
    """Medical Readmission — hard."""
    env.reset(task_id=3)
    actions = [
        # EDA — must do 6 before cleaning unlocks
        Action(type=ActionType.PROFILE_DATASET),
        Action(type=ActionType.DETECT_LEAKAGE),
        Action(type=ActionType.CHECK_MISSING),
        Action(type=ActionType.DETECT_DTYPES),
        Action(type=ActionType.CHECK_CLASS_BALANCE),
        Action(type=ActionType.DETECT_OUTLIERS),
        Action(type=ActionType.CHECK_CORRELATIONS),
        # Cleaning — remove CRITICAL (leakage) first to unlock engineering
        Action(type=ActionType.REMOVE_LEAKY_COL, column="discharge_summary"),
        Action(type=ActionType.REMOVE_LEAKY_COL, column="days_to_readmission"),
        Action(type=ActionType.FIX_DTYPES),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="bmi"),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="hba1c_result"),
        Action(type=ActionType.FILL_MISSING_MEDIAN, column="num_medications"),
        Action(type=ActionType.FILL_MISSING_MODE,   column="race"),
        Action(type=ActionType.DROP_CORRELATED_COL, column="num_procedures"),
        # Engineering — unlocked once leakage (only critical) issues are resolved
        Action(type=ActionType.LOG_TRANSFORM,      column="los_days"),
        Action(type=ActionType.ENCODE_ONEHOT,      column="gender"),
        Action(type=ActionType.ENCODE_LABEL,       column="race"),
        Action(type=ActionType.ENCODE_ONEHOT,      column="admission_type"),
        Action(type=ActionType.ENCODE_ONEHOT,      column="discharge_to"),
        Action(type=ActionType.ENCODE_LABEL,       column="payer_code"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="age"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="bmi"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="hba1c_result"),
        Action(type=ActionType.NORMALIZE_ROBUST,   column="num_medications"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="los_days"),
        Action(type=ActionType.NORMALIZE_STANDARD, column="num_diagnoses"),
        # Validation
        Action(type=ActionType.TRAIN_TEST_SPLIT),
        Action(type=ActionType.HANDLE_IMBALANCE_SMOTE),
        Action(type=ActionType.VALIDATE_NO_LEAKAGE),
        Action(type=ActionType.GENERATE_DATA_REPORT),
        Action(type=ActionType.FINISH),
    ]
    return _run(env, actions, verbose)


def _run(env: DataPrepEnv, actions: list, verbose: bool) -> float:
    step = 0
    obs  = None
    for action in actions:
        if env._done:
            break
        resp = env.step(action)
        obs  = resp.observation
        step += 1
        if verbose:
            status = "✓" if obs.last_action_success else "✗"
            print(f"    {step:02d} [{obs.current_phase:11s}] {status} "
                  f"{action.type:<35} "
                  f"readiness={obs.data_readiness_score:.2f} "
                  f"reward={resp.reward.value:+.3f}  "
                  f"{obs.last_action_message[:55]}")

    state = env.state()
    gs = state.grader_score or 0.0
    if verbose:
        print(f"\n    Issues resolved : {state.issues_resolved}/{state.issues_found}")
        print(f"    Readiness score : {state.data_readiness_score:.4f}")
        print(f"    Phase order     : {state.phase_order_score:.4f}")
        print(f"    GRADER SCORE    : {gs:.4f}\n")

    assert 0.0 <= gs <= 1.0, f"Score {gs} out of range!"
    return gs


def verify_reproducibility():
    print("\nVerifying reproducibility (3 runs per task)...")
    env = DataPrepEnv()
    all_ok = True
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    for tid, fn in graders.items():
        scores = [fn(env, verbose=False) for _ in range(3)]
        variance = max(scores) - min(scores)
        ok = variance < 0.001
        print(f"  Task {tid}: scores={[f'{s:.4f}' for s in scores]} "
              f"variance={variance:.6f}  [{'PASS' if ok else 'FAIL'}]")
        if not ok:
            all_ok = False
    return all_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, choices=[1, 2, 3])
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  Data Preparation Pipeline Agent — Grader")
    print("=" * 60)

    if args.verify:
        ok = verify_reproducibility()
        sys.exit(0 if ok else 1)

    env = DataPrepEnv()
    graders = {1: grade_task1, 2: grade_task2, 3: grade_task3}
    tasks   = [args.task] if args.task else [1, 2, 3]
    scores  = {}

    for tid in tasks:
        print(f"\n  Task {tid} | {['','easy','medium','hard'][tid]}")
        scores[tid] = graders[tid](env, verbose=True)

    print("=" * 60)
    print("  GRADER SUMMARY")
    print("=" * 60)
    total = 0.0
    diff  = {1:"easy  ", 2:"medium", 3:"hard  "}
    for tid, score in scores.items():
        print(f"  Task {tid} [{diff[tid]}]  grader_score = {score:.4f}")
        total += score
    if len(scores) == 3:
        print(f"\n  Average : {total/3:.4f}")
    print("=" * 60)
    sys.exit(0 if all(s > 0 for s in scores.values()) else 1)


if __name__ == "__main__":
    main()