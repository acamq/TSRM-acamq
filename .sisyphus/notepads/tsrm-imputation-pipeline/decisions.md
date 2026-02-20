
## 2026-02-19 F4 Audit Decision
- Scope fidelity verdict is REJECT due to multiple task-level mismatches and explicit scope violations despite successful scaffolding and config hardcoding safeguards.

## 2026-02-19 Task 12 Implementation Decision
- Reuse `train_fold` from `scripts/train_osse.py` as the single source of truth for fold training to keep search behavior and standalone training behavior consistent.

## 2026-02-19 Conv1d conv_dims Fix Decision
- Standardize `conv_dims` to integer kernel definitions in OSSE search grid and config defaults; avoid implicit float-to-kernel conversions to keep Conv1d parameter semantics explicit and PyTorch-safe.
