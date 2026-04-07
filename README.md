---
title: Data Preparation Pipeline Agent
emoji: 🧹
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - data-science
  - reinforcement-learning
  - tabular
---

# Data Preparation Pipeline Agent — OpenEnv

An OpenEnv environment where an AI agent learns to build optimal data
preprocessing pipelines through sequential decision-making.

## What it does

The agent takes a raw messy dataset and must clean and prepare it for ML —
handling missing values, removing data leakage, encoding categoricals,
normalising features, and balancing classes — in the correct order across
4 enforced phases.

## Phases

| Phase | Actions |
|---|---|
| 1 EDA | profile_dataset, detect_leakage, check_missing, detect_dtypes, check_class_balance, detect_outliers |
| 2 Cleaning | remove_leaky_col, fill_missing_median/mode, fix_dtypes, clip_outliers, drop_duplicates |
| 3 Engineering | encode_onehot, encode_label, normalize_standard, normalize_robust, log_transform |
| 4 Validation | train_test_split, handle_imbalance_smote, validate_no_leakage, finish |

## Tasks

| Task | Difficulty | Key Issues |
|---|---|---|
| 1 — Customer Churn | Easy | Missing values, dtype error, outliers, high cardinality |
| 2 — Loan Default | Medium | Data leakage, 9:1 imbalance, duplicates, correlated features |
| 3 — Medical Readmission | Hard | 2 leaky cols, 12:1 imbalance, heavy missing, dtype errors |

## API

```
POST /reset   {"task_id": 1}          → observation
POST /step    {"type": "check_missing"} → observation + reward + done
GET  /state                           → state snapshot + grader_score
GET  /health                          → {"status": "ok"}
```

## Reward (dense progressive)

Reward fires at **every step** — not just at completion:
- `readiness_delta` — positive whenever data quality improves
- `issue_fix_delta` — +0.08 per issue resolved, +0.15 for critical
- `phase_bonus` — +0.10 per phase completed
- `step_cost` — -0.005 per step (efficiency incentive)
- `invalid_penalty` — -0.05 for phase-gate violations

## Running locally

```bash
# Start environment server
docker build -t datapipe-env .
docker run -p 7860:7860 datapipe-env

# Run grader (no LLM needed)
python grader.py

# Run inference agent
export HF_TOKEN=hf_your_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
python inference.py
```

## Grader scores (reference agent)

| Task | Grader score |
|---|---|
| 1 easy | ~0.85 |
| 2 medium | ~0.85 |
| 3 hard | ~0.82 |
| Average | ~0.84 |
