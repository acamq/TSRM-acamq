
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
