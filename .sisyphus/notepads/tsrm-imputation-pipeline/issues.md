
## 2026-02-19 F4 Scope Fidelity Issues
-  does not implement required  return contract from plan.
-  and  APIs differ from planned signatures/semantics.
-  does not execute training/evaluation loop; grid generation only.
-  masks test windows during training run path (violates task guardrail).
-  writes  instead of planned  default.
- Smoke tests and E2E coverage are incomplete/misaligned with planned test scope.

## 2026-02-19 F4 Scope Fidelity Issues (Correction)
- pipeline/data/loader.py does not implement required load() return contract from plan.
- pipeline/evaluation/baselines.py and pipeline/evaluation/metrics.py APIs differ from planned signatures/semantics.
- scripts/search_osse.py does not execute training/evaluation loop; grid generation only.
- scripts/train_osse.py masks test windows during training run path (violates task guardrail).
- scripts/evaluate_osse.py writes evaluation_fold<N>.json instead of planned evaluation_results.json default.
- Smoke tests and E2E coverage are incomplete/misaligned with planned test scope.

## 2026-02-19 Task 12 Verification Issue
- `python scripts/search_osse.py --config configs/osse_default.yaml --folds 0 --epochs 1 --quick` cannot run in this workspace because `torch` is not installed; script now fails fast with explicit dependency error message.

## 2026-02-19 Verification Issue After conv_dims Fix
- The Conv1d float-kernel TypeError is resolved, but full `search_osse` execution now stops later with an unrelated runtime error: in-place operation modifies a tensor needed for gradient computation during training.
