"""
inference.py
============
Baseline inference script for the Data Preparation Pipeline Agent.

STDOUT FORMAT (mandatory):
    [START] task=<n> env=<benchmark> model=<model>
    [STEP]  step=<n> action=<json> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...>

Environment variables:
    API_BASE_URL  default: https://router.huggingface.co/v1
    MODEL_NAME    default: Qwen/Qwen2.5-72B-Instruct
    HF_TOKEN      REQUIRED
    ENV_BASE_URL  default: http://localhost:7860
"""

import os
import json
import requests
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://rash1453-data.hf.space")

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError(
        "HF_TOKEN environment variable is required.\n"
        "Set it: $env:HF_TOKEN = 'hf_your_token_here'"
    )

TEMPERATURE = 0.1
MAX_TOKENS  = 128

TASK_NAMES = {1: "customer-churn-prep", 2: "loan-default-prep", 3: "medical-readmission-prep"}
BENCHMARK  = "data-preparation-pipeline-agent"

VALID_ACTIONS = {
    "profile_dataset", "detect_leakage", "check_missing", "detect_dtypes",
    "check_class_balance", "detect_outliers", "check_correlations", "detect_duplicates",
    "remove_leaky_col", "fill_missing_median", "fill_missing_mode", "fill_missing_knn",
    "fill_missing_constant", "fix_dtypes", "drop_duplicates", "clip_outliers",
    "reduce_cardinality", "drop_correlated_col", "encode_onehot", "encode_label",
    "normalize_standard", "normalize_robust", "log_transform", "train_test_split",
    "handle_imbalance_smote", "validate_no_leakage", "generate_data_report", "finish",
}

# ── Lazy LLM client (initialized on first use, NOT at import time) ────────────
_llm_client = None

def get_llm_client():
    """Lazily initialize the OpenAI client to avoid module-level crashes."""
    global _llm_client
    if _llm_client is None:
        try:
            from openai import OpenAI
            _llm_client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        except Exception as exc:
            print(f"[WARN] Could not initialize LLM client: {exc}", flush=True)
            _llm_client = None
    return _llm_client

def call_llm(prompt: str) -> str:
    """Call the LLM, returning empty string on any failure."""
    try:
        client = get_llm_client()
        if client is None:
            return ""
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return completion.choices[0].message.content or ""
    except Exception as exc:
        print(f"[WARN] LLM call failed: {exc}", flush=True)
        return ""

# ── Stdout logging ─────────────────────────────────────────────────────────────
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}", flush=True)

def log_end(success, steps, score, rewards):
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}", flush=True)

# ── Env HTTP calls ─────────────────────────────────────────────────────────────
def env_health():
    try:
        return requests.get(f"{ENV_BASE_URL}/health", timeout=10).status_code == 200
    except Exception:
        return False

def env_wake():
    """Ping health repeatedly until the Space wakes up (handles cold starts / sleep)."""
    import time
    for attempt in range(12):          # up to ~60 s
        try:
            if requests.get(f"{ENV_BASE_URL}/health", timeout=10).status_code == 200:
                return True
        except Exception:
            pass
        print(f"[WARN] Space not ready, retrying ({attempt+1}/12)…", flush=True)
        time.sleep(5)
    return False

def env_reset(task_id):
    env_wake()   # make sure Space is awake before reset
    r = requests.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=120)
    r.raise_for_status()
    return r.json()

def env_step(action, retries=3):
    import time
    last_exc = None
    for attempt in range(retries):
        try:
            r = requests.post(f"{ENV_BASE_URL}/step", json=action, timeout=90)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] env_step attempt {attempt+1} failed: {exc}", flush=True)
            time.sleep(3)
            env_wake()   # Space may have slept mid-run
    raise last_exc

def env_state():
    r = requests.get(f"{ENV_BASE_URL}/state", timeout=60)
    r.raise_for_status()
    return r.json()

# ── Smart action engine ────────────────────────────────────────────────────────
def get_next_action(obs: dict) -> dict:
    phase      = obs.get("current_phase", "eda")
    issues     = obs.get("issues_found", [])
    unresolved = [i for i in issues if not i["resolved"]]
    columns    = obs.get("columns", [])
    pipeline   = obs.get("pipeline_steps", [])
    n_rows     = obs.get("n_rows", 1)

    done_types = {s.get("type") for s in pipeline}
    done_pairs = {(s.get("type"), s.get("column")) for s in pipeline}

    # ───────────── EDA ─────────────
    if phase == "eda":
        steps = [
            "profile_dataset",
            "detect_leakage",
            "check_missing",
            "detect_dtypes",
            "check_class_balance",
            "detect_outliers",
            "check_correlations",
        ]
        for step in steps:
            if step not in done_types:
                return {"type": step}
        return {"type": "profile_dataset"}

    # ───────────── CLEANING ─────────────
    if phase == "cleaning":

        # 1. Remove leakage FIRST
        for col in columns:
            if col.get("is_leaky"):
                if ("remove_leaky_col", col["name"]) not in done_pairs:
                    return {"type": "remove_leaky_col", "column": col["name"]}

        # 2. Fix dtypes
        if any(i["issue_type"] == "dtype" for i in unresolved):
            if "fix_dtypes" not in done_types:
                return {"type": "fix_dtypes"}

        # 3. Drop duplicates
        if any(i["issue_type"] == "duplicate" for i in unresolved):
            if "drop_duplicates" not in done_types:
                return {"type": "drop_duplicates"}

        # 4. Fill ALL missing values
        for col in columns:
            if col.get("missing_pct", 0) > 0 and not col.get("is_leaky"):
                already_filled = (
                    ("fill_missing_median", col["name"]) in done_pairs or
                    ("fill_missing_mode",   col["name"]) in done_pairs or
                    ("fill_missing_knn",    col["name"]) in done_pairs
                )
                if not already_filled:
                    if col.get("dtype") == "numeric":
                        return {"type": "fill_missing_median", "column": col["name"]}
                    else:
                        return {"type": "fill_missing_mode", "column": col["name"]}

        # 5. Reduce cardinality
        for col in columns:
            if col.get("is_high_cardinality") and col.get("dtype") == "categorical":
                if ("reduce_cardinality", col["name"]) not in done_pairs:
                    return {"type": "reduce_cardinality", "column": col["name"]}

        # 6. Clip outliers
        for col in columns:
            if col.get("has_outliers") and col.get("dtype") == "numeric":
                if ("clip_outliers", col["name"]) not in done_pairs:
                    return {"type": "clip_outliers", "column": col["name"]}

        # 7. Drop correlated columns
        for i in unresolved:
            if i["issue_type"] == "correlation" and i.get("column"):
                if ("drop_correlated_col", i["column"]) not in done_pairs:
                    return {"type": "drop_correlated_col", "column": i["column"]}

        if unresolved:
            return {"type": "fix_dtypes"}

    # ───────────── ENGINEERING ─────────────
    if phase == "engineering":

        # 1. Log transform skewed
        for col in columns:
            if col.get("is_skewed") and col.get("dtype") == "numeric":
                if ("log_transform", col["name"]) not in done_pairs:
                    return {"type": "log_transform", "column": col["name"]}

        # 2. Encode ALL categoricals
        for col in columns:
            if col.get("dtype") != "categorical" or col.get("is_leaky"):
                continue
            if col.get("unique_count", 0) >= n_rows * 0.5:
                continue
            encoded = (
                ("encode_onehot", col["name"]) in done_pairs or
                ("encode_label",  col["name"]) in done_pairs
            )
            if not encoded:
                if col.get("unique_count", 0) <= 15:
                    return {"type": "encode_onehot", "column": col["name"]}
                else:
                    return {"type": "encode_label", "column": col["name"]}

        # 3. Normalize ALL numerics
        for col in columns:
            if col.get("dtype") == "numeric":
                normalized = (
                    ("normalize_standard", col["name"]) in done_pairs or
                    ("normalize_robust",   col["name"]) in done_pairs
                )
                if not normalized:
                    return {"type": "normalize_standard", "column": col["name"]}

        return {"type": "train_test_split"}

    # ───────────── VALIDATION ─────────────
    if phase == "validation":

        if "train_test_split" not in done_types:
            return {"type": "train_test_split"}

        if any(i["issue_type"] == "imbalance" for i in unresolved):
            if "handle_imbalance_smote" not in done_types:
                return {"type": "handle_imbalance_smote"}

        if "validate_no_leakage" not in done_types:
            return {"type": "validate_no_leakage"}

        if "generate_data_report" not in done_types:
            return {"type": "generate_data_report"}

        return {"type": "finish"}

    return {"type": "profile_dataset"}


# ── LLM helpers ───────────────────────────────────────────────────────────────
def parse_llm_action(text: str) -> Optional[dict]:
    text = text.strip()
    if "```" in text:
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    start, end = text.find("{"), text.rfind("}") + 1
    if start == -1 or end == 0:
        return None
    try:
        action = json.loads(text[start:end])
        if action.get("type") in VALID_ACTIONS:
            return action
    except Exception:
        pass
    return None


def build_prompt(obs: dict, smart_action: dict) -> str:
    return f"Output this JSON exactly, nothing else:\n{json.dumps(smart_action)}"


# ── Agent loop ────────────────────────────────────────────────────────────────
def run_task(task_id: int) -> dict:
    task_name   = TASK_NAMES[task_id]
    rewards     = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs       = env_reset(task_id)["observation"]
        max_steps = obs["max_steps"]

        for step_num in range(1, max_steps + 1):
            if obs.get("done", False):
                break

            smart_action = get_next_action(obs)
            final_action = smart_action

            try:
                prompt     = build_prompt(obs, smart_action)
                llm_output = call_llm(prompt)
                llm_action = parse_llm_action(llm_output)

                if llm_action:
                    same_type = llm_action.get("type") == smart_action.get("type")
                    same_col  = llm_action.get("column") == smart_action.get("column")
                    if same_type and same_col:
                        final_action = llm_action
            except Exception:
                pass  # smart_action is the fallback

            action_str = json.dumps(final_action)

            try:
                step_resp = env_step(final_action)
            except Exception as exc:
                log_step(step_num, action_str, 0.0, True, str(exc)[:80])
                break

            obs       = step_resp["observation"]
            reward    = step_resp["reward"]["value"]
            done      = step_resp["done"]
            error_msg = None if obs["last_action_success"] else obs["last_action_message"][:80]

            rewards.append(reward)
            steps_taken = step_num

            log_step(step_num, action_str, reward, done, error_msg)

            if done:
                break

        state   = env_state()
        score   = float(state.get("grader_score") or 0.0)
        success = score > 0.0

    except Exception as exc:
        print(f"[ERROR] Task {task_id} failed: {exc}", flush=True)
        score   = 0.0
        success = False

    finally:
        log_end(success, steps_taken, score, rewards)

    return {"task_id": task_id, "task_name": task_name,
            "score": score, "steps": steps_taken, "success": success}


# ── run_inference ──────────────────────────────────────────────────────────────
def run_inference(prompt: str) -> str:
    return call_llm(prompt)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    if not env_health():
        raise ConnectionError(
            f"Env not reachable at {ENV_BASE_URL}\n"
            "Start: docker run -p 7860:7860 datapipe-env"
        )

    results = []
    for task_id in [1, 2, 3]:
        results.append(run_task(task_id))

    labels = {1: "easy  ", 2: "medium", 3: "hard  "}
    total  = 0.0
    print(f"\n{'='*50}", flush=True)
    for r in results:
        print(f"Task {r['task_id']} [{labels[r['task_id']]}] score={r['score']:.4f} steps={r['steps']}", flush=True)
        total += r["score"]
    print(f"Average: {total/len(results):.4f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()
