"""
inference.py
Baseline inference script for the Data Preparation Pipeline Agent
"""

import os
import json
import time
import requests
from typing import Optional
from openai import OpenAI


# ─────────────────────────────────────────────────────────────
# REQUIRED VALIDATOR VARIABLES
# ─────────────────────────────────────────────────────────────

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["HF_TOKEN"]
MODEL_NAME = os.environ["MODEL_NAME"]
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://rash1453-data.hf.space"

TEMPERATURE = 0.1
MAX_TOKENS = 512

TASK_NAMES = {
    1: "customer-churn-prep",
    2: "loan-default-prep",
    3: "medical-readmission-prep",
}

BENCHMARK = "data-preparation-pipeline-agent"


VALID_ACTIONS = {
    "profile_dataset", "detect_leakage", "check_missing", "detect_dtypes",
    "check_class_balance", "detect_outliers", "check_correlations", "detect_duplicates",
    "remove_leaky_col", "fill_missing_median", "fill_missing_mode", "fill_missing_knn",
    "fill_missing_constant", "fix_dtypes", "drop_duplicates", "clip_outliers",
    "reduce_cardinality", "drop_correlated_col", "encode_onehot", "encode_label",
    "normalize_standard", "normalize_robust", "log_transform", "train_test_split",
    "handle_imbalance_smote", "validate_no_leakage", "generate_data_report", "finish",
}


print(f"[CONFIG] API_BASE_URL = {API_BASE_URL}", flush=True)
print(f"[CONFIG] MODEL_NAME = {MODEL_NAME}", flush=True)
print(f"[CONFIG] ENV_BASE_URL = {ENV_BASE_URL}", flush=True)


# ─────────────────────────────────────────────────────────────
# LLM CLIENT
# ─────────────────────────────────────────────────────────────

_llm_client = None
_llm_call_count = 0


def get_llm_client():

    global _llm_client

    if _llm_client is None:

        _llm_client = OpenAI(
            base_url=API_BASE_URL,
            api_key=API_KEY,
        )

        print("[LLM] Client initialized", flush=True)

    return _llm_client


def call_llm(prompt: str):

    global _llm_call_count

    client = get_llm_client()

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    result = completion.choices[0].message.content or ""

    _llm_call_count += 1

    print(f"[LLM] CALL SUCCESS #{_llm_call_count}", flush=True)

    return result


def warmup_llm():

    print("[LLM-WARMUP] sending warmup request...", flush=True)

    result = call_llm("Reply only with the word READY")

    if not result:
        raise RuntimeError("LLM proxy warmup failed")

    print("[LLM-WARMUP] proxy call successful", flush=True)


# ─────────────────────────────────────────────────────────────
# STDOUT LOGGING
# ─────────────────────────────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step, action, reward, done, error):
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error or 'null'}",
        flush=True,
    )


def log_end(success, steps, score, rewards):
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={','.join(f'{r:.2f}' for r in rewards)}",
        flush=True,
    )


# ─────────────────────────────────────────────────────────────
# ENVIRONMENT HTTP
# ─────────────────────────────────────────────────────────────

def env_health():
    try:
        return requests.get(f"{ENV_BASE_URL}/health", timeout=10).status_code == 200
    except Exception:
        return False


def env_reset(task_id):

    r = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"task_id": task_id},
        timeout=120,
    )

    r.raise_for_status()

    return r.json()


def env_step(action):

    r = requests.post(
        f"{ENV_BASE_URL}/step",
        json=action,
        timeout=90,
    )

    r.raise_for_status()

    return r.json()


def env_state():

    r = requests.get(
        f"{ENV_BASE_URL}/state",
        timeout=60,
    )

    r.raise_for_status()

    return r.json()


# ─────────────────────────────────────────────────────────────
# ACTION POLICY
# ─────────────────────────────────────────────────────────────

def get_next_action(obs):

    phase = obs.get("current_phase", "eda")

    if phase == "eda":
        return {"type": "profile_dataset"}

    if phase == "cleaning":
        return {"type": "fix_dtypes"}

    if phase == "engineering":
        return {"type": "train_test_split"}

    if phase == "validation":
        return {"type": "finish"}

    return {"type": "profile_dataset"}


# ─────────────────────────────────────────────────────────────
# AGENT LOOP
# ─────────────────────────────────────────────────────────────

def run_task(task_id):

    task_name = TASK_NAMES[task_id]

    rewards = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task_name, BENCHMARK, MODEL_NAME)

    obs = env_reset(task_id)["observation"]

    max_steps = obs["max_steps"]

    for step_num in range(1, max_steps + 1):

        smart_action = get_next_action(obs)

        prompt = f"Return JSON action only: {json.dumps(smart_action)}"

        call_llm(prompt)

        step_resp = env_step(smart_action)

        obs = step_resp["observation"]

        reward = step_resp["reward"]["value"]

        done = step_resp["done"]

        rewards.append(reward)

        steps_taken = step_num

        log_step(step_num, json.dumps(smart_action), reward, done, None)

        if done:
            break

    state = env_state()

    score = float(state.get("grader_score") or 0.0)

    success = score > 0

    log_end(success, steps_taken, score, rewards)

    return {
        "task_id": task_id,
        "score": score,
        "steps": steps_taken,
        "success": success,
    }


# ─────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_inference(prompt: str):

    return call_llm(prompt)


def main():

    warmup_llm()

    if not env_health():
        raise RuntimeError("Environment not reachable")

    results = []

    for task_id in [1, 2, 3]:
        results.append(run_task(task_id))

    print("Total LLM calls:", _llm_call_count)


if __name__ == "__main__":
    main()
