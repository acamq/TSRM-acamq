# TSRM Imputation Pipeline for GDC OSSE Space Weather Data

## TL;DR

> **Quick Summary**: Build a complete data imputation pipeline in TSRM-acamq that adapts the spacew SAITS/PyPOTS pipeline to use the TSRM model, enabling direct TSRM vs SAITS comparison on GDC OSSE space weather data.
>
> **Deliverables**:
> - `pipeline/` package with data loading, preprocessing, masking, evaluation, tracking, visualization (adapted from spacew)
> - `architecture/tsrm_external.py` — TSRMImputationExternal subclass accepting pre-masked data
> - `pipeline/data/dataset.py` — OSSEDataset (PyTorch Dataset bridge with time embeddings)
> - `configs/osse_default.yaml` — Default TSRM config for OSSE imputation
> - `scripts/train_osse.py` — 4-fold leave-one-block-out CV training
> - `scripts/evaluate_osse.py` — Evaluation with baselines + skill scores
> - `scripts/search_osse.py` — ParameterGrid hyperparameter search
> - `tests/` — Smoke tests for critical components
>
> **Estimated Effort**: Large
> **Parallel Execution**: YES — 5 waves
> **Critical Path**: Task 1 → Task 9 → Task 13 → Task 14 → Task 17

---

## Path Aliases (for cross-repo references)

```
SPACEW  = C:\Users\aic\liua\work\projects\spacew_root\github\spacew
TSRM    = C:\Users\aic\liua\work\projects\spacew_root\github\TSRM-acamq  (this repo)
```

All spacew references use absolute paths. All TSRM references use relative paths from repo root.

---

## Context

### Original Request
Build a data imputation pipeline using the TSRM model in TSRM-acamq, adapting the existing spacew SAITS/PyPOTS pipeline. Reuse data loading, preprocessing, masking, and evaluation modules. Enable direct TSRM vs SAITS comparison.

### Interview Summary

**Key Decisions (9 total)**:
1. **Build location**: TSRM-acamq repo — copy/adapt spacew modules here, self-contained
2. **Training strategy**: Downstream only — train TSRMImputation from scratch, no pretrain+finetune
3. **Masking strategy**: External masking from spacew (3 patterns: point MCAR, subsequence, block) — bypass TSRM's internal `calc_mask`
4. **Data & evaluation**: Same OSSE data + same evaluation framework (4-fold leave-one-block-out CV, MSE/MAE/MSESS/MAESS, LOCF+linear baselines)
5. **Normalization**: External only — disable RevIN, use spacew's NaNSafeScaler + log1p for density vars
6. **Time embeddings**: Enabled — generate time marks from OSSE timestamps for TSRM's DataEmbedding
7. **Test strategy**: Minimal smoke tests with pytest + agent-executed QA
8. **Output directories**: Same .env pattern as spacew (DATA_DIR, SCRATCH_DIR)
9. **Hyperparameters**: Include ParameterGrid search — test attention functions, layer counts, heads, encoding sizes, conv_dims, dropout, learning rate

**Research Findings**:
- TSRM's `calc_mask()` detects NaN and adds random MCAR — must be bypassed via subclass
- TSRM's `ImputationLoss` has modes: "all", "imputation", "weighted_imputation", "imputation_only" — use "imputation" mode (loss on masked positions only)
- spacew remediation plan caught scaler data leakage, mask rate enforcement, shared eval mask issues — avoid repeating
- spacew best SAITS config: n_layers=1, d_model=512, n_heads=4, batch_size=64, epochs=120, patience=20, lr=0.00019

### Metis Review

**Identified Gaps (all resolved)**:
- **Mask integration conflict**: TSRMImputationExternal subclass with `missing_ratio=0`, override `calc_mask` to only detect NaN → mark as 1 for loss
- **Loss mode ambiguity**: Use "imputation" mode (loss on masked positions only), include as hyperparameter search option
- **Time embeddings**: Use real OSSE timestamps (physically meaningful), not synthetic sequential
- **Conv_dims for seq_len=30**: Include valid kernel/dilation combinations in search grid (receptive field ≤ 30)
- **Hyperparameter selection**: Aggregate validation performance across all 4 folds
- **Batch composition**: Shuffle training batches, sequential evaluation
- **Per-variable metrics**: Include (same as spacew)

---

## Work Objectives

### Core Objective
Replicate spacew's imputation pipeline architecture using the TSRM model, producing comparable metrics for direct SAITS vs TSRM evaluation on GDC OSSE space weather data.

### Target Directory Structure
```
TSRM-acamq/
├── architecture/                # EXISTING — add tsrm_external.py only
│   ├── model.py                 # DO NOT MODIFY
│   ├── loss_functions.py        # DO NOT MODIFY
│   └── tsrm_external.py        # NEW: TSRMImputationExternal subclass
├── pipeline/                    # NEW — imputation pipeline package
│   ├── __init__.py
│   ├── config.py                # Config loading + TSRM config bridge
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            # OSSEDataLoader (adapted from spacew)
│   │   ├── preprocessor.py      # NaNSafeScaler + Preprocessor (adapted from spacew)
│   │   ├── masking.py           # Missing patterns (from spacew, minimal changes)
│   │   └── dataset.py           # OSSEDataset (PyTorch Dataset bridge) — NEW
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # MSE/MAE/MSESS/MAESS (adapted from spacew)
│   │   └── baselines.py         # LOCF + linear (from spacew, minimal changes)
│   ├── tracking/
│   │   ├── __init__.py
│   │   └── experiment.py        # Experiment tracking (adapted from spacew)
│   └── visualization/
│       ├── __init__.py
│       └── plots.py             # Comparison plots (adapted from spacew)
├── configs/
│   └── osse_default.yaml        # NEW: Default TSRM config for OSSE imputation
├── scripts/
│   ├── train_osse.py            # NEW: 4-fold CV training script
│   ├── evaluate_osse.py         # NEW: Evaluation + baselines script
│   └── search_osse.py           # NEW: Hyperparameter search script
├── tests/
│   ├── test_tsrm_external.py    # NEW: TSRMImputationExternal smoke tests
│   └── test_osse_dataset.py     # NEW: OSSEDataset smoke tests
└── .env                         # NEW: DATA_DIR, SCRATCH_DIR paths
```

### Concrete Deliverables
- 15 new Python files in `pipeline/`, `architecture/`, `scripts/`, `tests/`
- 1 new YAML config in `configs/`
- 1 `.env` template
- Updated `requirements.txt`
- Working end-to-end: data → preprocess → mask → train TSRM → evaluate → compare with SAITS

### Definition of Done
- [ ] `python scripts/train_osse.py --config configs/osse_default.yaml --fold 0` completes without error
- [ ] `python scripts/evaluate_osse.py --config configs/osse_default.yaml` produces metrics JSON
- [ ] `python scripts/search_osse.py --config configs/osse_default.yaml` runs ParameterGrid search
- [ ] All 4 folds produce MSE/MAE/MSESS/MAESS metrics
- [ ] Metrics are directly comparable to spacew SAITS results
- [ ] `pytest tests/` passes

### Must Have
- TSRMImputationExternal that accepts pre-masked data and computes loss on masked positions only
- OSSEDataset producing (masked_data, original_data, time_marks_x, time_marks_y) tuples
- 4-fold leave-one-block-out CV with per-fold scaler fitting (train blocks only)
- 3 masking patterns (point, subsequence, block) with configurable rates
- MSE/MAE/MSESS/MAESS metrics with shared evaluation mask
- LOCF + linear interpolation baselines
- ParameterGrid hyperparameter search over attention_func, N, h, encoding_size, conv_dims, dropout, lr
- Experiment tracking with config snapshots and git metadata

### Must NOT Have (Guardrails)
- DO NOT modify any existing TSRM files (architecture/model.py, loss_functions.py, etc.) — only ADD new files
- DO NOT modify any spacew repository files — read-only reference
- DO NOT add masking patterns beyond point/subseq/block
- DO NOT add evaluation beyond aggregate metrics + per-variable (no significance tests, no per-satellite breakdown, no attention visualization)
- DO NOT use Bayesian optimization or nested CV — single ParameterGrid pass only
- DO NOT refactor or "improve" copied spacew modules (type hints, docstrings OK, logic changes NO)
- DO NOT enable RevIN — external normalization only (revin: false in all configs)
- DO NOT add dependencies not already in requirements.txt or spacew's environment.yml without explicit justification

---

## Verification Strategy

> **ZERO HUMAN INTERVENTION** — ALL verification is agent-executed. No exceptions.

### Test Decision
- **Infrastructure exists**: NO (no existing test framework in TSRM-acamq)
- **Automated tests**: YES (tests-after) — minimal smoke tests with pytest
- **Framework**: pytest (add to requirements.txt)
- **Scope**: Smoke tests for TSRMImputationExternal and OSSEDataset only

### QA Policy
Every task MUST include agent-executed QA scenarios.
Evidence saved to `.sisyphus/evidence/task-{N}-{scenario-slug}.{ext}`.

- **Data pipeline**: Use Bash (python -c) — import, call functions, verify shapes/values
- **Model**: Use Bash (python -c) — instantiate, forward pass, verify output shapes
- **Scripts**: Use Bash — run scripts with test config, verify output files exist
- **Evaluation**: Use Bash (python -c) — verify metric values are reasonable

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately — copy/adapt spacew modules, 7 parallel):
├── Task 1:  Project scaffolding (.env, dirs, requirements) [quick]
├── Task 2:  Config module (config.py + YAML template) [quick]
├── Task 3:  OSSEDataLoader (data/loader.py) [quick]
├── Task 4:  Preprocessor + NaNSafeScaler (data/preprocessor.py) [quick]
├── Task 5:  Masking module (data/masking.py) [quick]
├── Task 6:  Evaluation metrics (evaluation/metrics.py) [quick]
└── Task 7:  Baselines (evaluation/baselines.py) [quick]

Wave 2 (After Wave 1 — TSRM-specific adaptation, 5 parallel):
├── Task 8:  TSRMImputationExternal subclass (depends: 5) [deep]
├── Task 9:  OSSEDataset + time embeddings (depends: 3, 4) [deep]
├── Task 10: TSRM config YAML + defaults (depends: 2) [quick]
├── Task 11: Experiment tracking module (depends: 1) [quick]
└── Task 12: Hyperparameter search grid (depends: 10) [unspecified-high]

Wave 3 (After Wave 2 — pipeline scripts, 3 tasks):
├── Task 13: Training script with 4-fold CV (depends: 8, 9, 10, 11) [deep]
├── Task 14: Evaluation script (depends: 6, 7, 13) [unspecified-high]
└── Task 15: Visualization module (depends: 14) [quick]

Wave 4 (After Wave 3 — testing, 2 parallel):
├── Task 16: Smoke tests (depends: 8, 9) [quick]
└── Task 17: E2E integration test (depends: 13, 14, 15) [deep]

Wave FINAL (After ALL tasks — independent review, 4 parallel):
├── Task F1: Plan compliance audit (oracle)
├── Task F2: Code quality review (unspecified-high)
├── Task F3: Real manual QA (unspecified-high)
└── Task F4: Scope fidelity check (deep)

Critical Path: Task 1 → Task 9 → Task 13 → Task 14 → Task 17 → F1-F4
Parallel Speedup: ~65% faster than sequential
Max Concurrent: 7 (Wave 1)
```

### Dependency Matrix

| Task | Depends On | Blocks | Wave |
|------|-----------|--------|------|
| 1 | — | 8-12 | 1 |
| 2 | — | 10 | 1 |
| 3 | — | 9 | 1 |
| 4 | — | 9 | 1 |
| 5 | — | 8 | 1 |
| 6 | — | 14 | 1 |
| 7 | — | 14 | 1 |
| 8 | 5 | 13, 16 | 2 |
| 9 | 3, 4 | 13, 16 | 2 |
| 10 | 2 | 12, 13 | 2 |
| 11 | 1 | 13 | 2 |
| 12 | 10 | 13 | 2 |
| 13 | 8, 9, 10, 11, 12 | 14, 17 | 3 |
| 14 | 6, 7, 13 | 15, 17 | 3 |
| 15 | 14 | 17 | 3 |
| 16 | 8, 9 | — | 4 |
| 17 | 13, 14, 15 | F1-F4 | 4 |
| F1-F4 | 17 | — | FINAL |

### Agent Dispatch Summary

| Wave | Tasks | Categories |
|------|-------|------------|
| 1 | 7 | T1-T7 → `quick` |
| 2 | 5 | T8 → `deep`, T9 → `deep`, T10 → `quick`, T11 → `quick`, T12 → `unspecified-high` |
| 3 | 3 | T13 → `deep`, T14 → `unspecified-high`, T15 → `quick` |
| 4 | 2 | T16 → `quick`, T17 → `deep` |
| FINAL | 4 | F1 → `oracle`, F2 → `unspecified-high`, F3 → `unspecified-high`, F4 → `deep` |

---

## TODOs

> Implementation + Test = ONE Task. Never separate.
> EVERY task MUST have: Recommended Agent Profile + Parallelization info + QA Scenarios.

- [x] 1. Project Scaffolding

  **What to do**:
  - Create `pipeline/` package directory with all subdirectories and `__init__.py` files matching the Target Directory Structure above
  - Create `.env` template file at repo root with `DATA_DIR` and `SCRATCH_DIR` placeholders (see spacew's `.env` pattern)
  - Update `requirements.txt` — add `pytest`, `python-dotenv` if not already present (check existing deps first)
  - Create `scripts/` directory if not present, add empty `__init__.py`
  - Create `tests/` directory with `__init__.py`
  - Create `.sisyphus/evidence/` directory for QA evidence

  **Must NOT do**:
  - Do NOT create any Python logic files yet — only empty `__init__.py` and `.env`
  - Do NOT modify existing TSRM files

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple directory creation and file scaffolding — no logic, just structure
  - **Skills**: []
    - No specialized skills needed for directory/file creation

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 3, 4, 5, 6, 7)
  - **Blocks**: Tasks 8, 9, 10, 11, 12 (all Wave 2 tasks depend on directory structure existing)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `{SPACEW}/pyproject.toml` — spacew project structure for reference on package organization
  - `{SPACEW}/.env` or `{SPACEW}/configs/default.yaml:1-10` — see `data_dir` and `scratch_dir` env var names used by spacew

  **API/Type References**:
  - Target Directory Structure defined in this plan's Work Objectives section — follow EXACTLY

  **External References**:
  - None

  **WHY Each Reference Matters**:
  - The spacew `.env` pattern shows what env vars the data loader and experiment tracker expect
  - The Target Directory Structure is the canonical layout all subsequent tasks depend on

  **Acceptance Criteria**:

  ```
  QA Scenario: Directory structure matches plan
    Tool: Bash (python -c)
    Preconditions: Repo cloned, no prior pipeline/ directory
    Steps:
      1. Run: python -c "from pathlib import Path; dirs = ['pipeline', 'pipeline/data', 'pipeline/evaluation', 'pipeline/tracking', 'pipeline/visualization', 'scripts', 'tests']; missing = [d for d in dirs if not Path(d).is_dir()]; assert not missing, f'Missing dirs: {missing}'; print('All directories exist')"
      2. Run: python -c "from pathlib import Path; inits = ['pipeline/__init__.py', 'pipeline/data/__init__.py', 'pipeline/evaluation/__init__.py', 'pipeline/tracking/__init__.py', 'pipeline/visualization/__init__.py', 'tests/__init__.py']; missing = [f for f in inits if not Path(f).is_file()]; assert not missing, f'Missing __init__.py: {missing}'; print('All __init__.py files exist')"
      3. Verify .env template exists: python -c "p = open('.env').read(); assert 'DATA_DIR' in p and 'SCRATCH_DIR' in p; print('.env template OK')"
    Expected Result: All directories, __init__.py files, and .env template exist
    Failure Indicators: Any missing directory or file
    Evidence: .sisyphus/evidence/task-1-scaffolding.txt

  QA Scenario: requirements.txt updated
    Tool: Bash
    Preconditions: requirements.txt exists
    Steps:
      1. Run: python -c "reqs = open('requirements.txt').read().lower(); assert 'pytest' in reqs; assert 'python-dotenv' in reqs or 'dotenv' in reqs; print('requirements.txt updated')"
    Expected Result: pytest and python-dotenv are in requirements.txt
    Failure Indicators: Missing entries
    Evidence: .sisyphus/evidence/task-1-requirements.txt
  ```

  **Commit**: YES (groups with Tasks 2-7 in Commit A)
  - Message: `feat(pipeline): add foundation modules adapted from spacew`
  - Files: `pipeline/**/__init__.py`, `.env`, `requirements.txt`, `scripts/`, `tests/`

- [x] 2. Config Module

  **What to do**:
  - Copy `{SPACEW}/src/spacew/config.py` → `pipeline/config.py`
  - Adapt imports: remove spacew-specific imports, use local paths
  - Keep `load_config(yaml_path)` function that merges YAML + `.env` vars
  - Add `build_tsrm_config(yaml_cfg) -> dict` function that translates our YAML config into the TSRM config dict format expected by `TSRMImputation.__init__`
  - The TSRM config dict must include ALL required keys:
    ```python
    {
        "feature_dimension": int,   # 8 for OSSE
        "seq_len": int,             # window_size
        "pred_len": 0,              # imputation, not forecasting
        "encoding_size": int,       # embedding dimension
        "h": int,                   # attention heads
        "N": int,                   # encoder layers
        "conv_dims": list,          # [[kernel, dilation, groups], ...]
        "attention_func": str,      # "entmax15", "classic", "propsparse"
        "batch_size": int,
        "dropout": float,
        "revin": False,             # ALWAYS False — external norm only
        "loss_function_imputation": str,  # "mse+mae"
        "loss_imputation_mode": str,      # "imputation"
        "loss_weight_alpha": float,
        "missing_ratio": 0.0,       # ALWAYS 0 — external masking
        "embed": "timeF",           # time feature embedding
        "freq": "t",                # minute-level frequency
        "mask_size": int,
        "mask_count": int,
        "task": "imputation",       # REQUIRED by Transformations.__init__ (model.py:224)
        "phase": "downstream",      # REQUIRED by Transformations.__init__ (model.py:224)
    }
    ```
  - **CRITICAL**: `task` and `phase` are REQUIRED keys read by `Transformations.__init__` at `{TSRM}/architecture/model.py:224`. Without these, model instantiation will crash with KeyError. Hardcode these in `build_tsrm_config()`: `task="imputation"`, `phase="downstream"`.
  - Ensure `revin` is ALWAYS False and `missing_ratio` is ALWAYS 0 regardless of YAML input (hardcode these as overrides in `build_tsrm_config`)

  **Must NOT do**:
  - Do NOT add complex validation logic beyond basic type checks
  - Do NOT change the YAML loading mechanism (keep it simple like spacew)
  - Do NOT set revin=True or missing_ratio>0 anywhere

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward copy + adaptation of existing module, plus a config translation function
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 3, 4, 5, 6, 7)
  - **Blocks**: Task 10 (TSRM config YAML defaults depends on config module)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References** (existing code to follow):
  - `{SPACEW}/src/spacew/config.py` — Source config module to copy and adapt. Key functions: `load_config()`, path resolution logic, `.env` integration. Copy the structure but adapt key names.
  - `{TSRM}/experiments/configs/imputation_etth1.yml` — Example TSRM YAML config showing required keys and their nesting structure (NOTE: extension is `.yml` not `.yaml`)
  - `{TSRM}/experiments/configs/imputation_ettm1.yml` — Another TSRM config example for imputation task

  **API/Type References** (contracts to implement against):
  - `{TSRM}/architecture/model.py:102-140` — TSRM.__init__ which reads config dict. Shows ALL keys that MUST exist in the config dict and their expected types.
  - `{TSRM}/architecture/model.py:400-430` — TSRMImputation.__init__ for imputation-specific config keys (missing_ratio, loss_imputation_mode, etc.)

  **External References**:
  - `{SPACEW}/configs/default.yaml` — spacew's YAML structure for data/normalization/masking sections (our YAML will follow similar structure for those sections)

  **WHY Each Reference Matters**:
  - spacew config.py provides the copy source — our module should work the same way (YAML + .env merge)
  - TSRM YAML configs show the exact key names and nesting TSRM expects
  - model.py:102-140 is the CONTRACT — if our config dict is missing any key, TSRM will crash at init

  **Acceptance Criteria**:

  ```
  QA Scenario: Config loads and builds TSRM config dict
    Tool: Bash (python -c)
    Preconditions: pipeline/config.py exists, configs/osse_default.yaml exists (from Task 10, but test with a minimal inline YAML)
    Steps:
      1. Create a minimal test YAML inline: python -c "
         import yaml, tempfile, os
         cfg = {'data': {'variables': ['v1','v2'], 'window_size': 30}, 'tsrm': {'encoding_size': 64, 'h': 4, 'N': 2, 'conv_dims': [[3,1,1]], 'attention_func': 'entmax15', 'batch_size': 32, 'dropout': 0.1, 'learning_rate': 0.001, 'loss_function_imputation': 'mse+mae', 'loss_imputation_mode': 'imputation', 'loss_weight_alpha': 0.5, 'mask_size': 3, 'mask_count': 1, 'epochs': 10, 'patience': 5}}
         p = tempfile.mktemp(suffix='.yaml')
         yaml.dump(cfg, open(p, 'w'))
         from pipeline.config import load_config, build_tsrm_config
         loaded = load_config(p)
         tsrm_cfg = build_tsrm_config(loaded)
         assert tsrm_cfg['revin'] == False, 'revin must be False'
         assert tsrm_cfg['missing_ratio'] == 0.0, 'missing_ratio must be 0'
         assert tsrm_cfg['task'] == 'imputation', 'task must be imputation'
         assert tsrm_cfg['phase'] == 'downstream', 'phase must be downstream'
         assert tsrm_cfg['feature_dimension'] == 2
         assert tsrm_cfg['seq_len'] == 30
         assert tsrm_cfg['pred_len'] == 0
         assert 'encoding_size' in tsrm_cfg
         assert 'h' in tsrm_cfg
         assert 'N' in tsrm_cfg
         os.unlink(p)
         print('Config module OK')
         "
    Expected Result: Config loads, builds TSRM dict, enforces revin=False and missing_ratio=0
    Failure Indicators: KeyError on any required field, revin=True, missing_ratio>0
    Evidence: .sisyphus/evidence/task-2-config.txt

  QA Scenario: Config rejects revin override attempt
    Tool: Bash (python -c)
    Preconditions: pipeline/config.py exists
    Steps:
      1. Create YAML with revin: true and verify build_tsrm_config forces it to False
    Expected Result: revin is always False regardless of YAML input
    Failure Indicators: revin=True in output config
    Evidence: .sisyphus/evidence/task-2-config-revin.txt
  ```

  **Commit**: YES (groups with Commit A)

- [x] 3. OSSEDataLoader

  **What to do**:
  - Copy `{SPACEW}/src/spacew/data/loader.py` → `pipeline/data/loader.py`
  - Adapt imports: remove `from spacew.config import ...` → `from pipeline.config import ...`
  - Keep all core functionality:
    - `OSSEDataLoader` class that loads GDC OSSE pickle data from `DATA_DIR`
    - Variable selection (8 IT variables)
    - Block boundary definitions `[0, 60, 120, 180, 240]` (4 blocks × 60 timesteps)
    - 6-satellite data structure `[N_sat, T, F]`
    - Data validation (NaN checks, shape assertions)
  - Update docstrings to reference TSRM pipeline instead of spacew
  - Ensure `load()` returns the same data structure as spacew: dict with 'data' (numpy array [6, 240, 8]), 'timestamps', 'variable_names', 'satellite_ids'

  **Must NOT do**:
  - Do NOT change the data loading logic or block boundaries
  - Do NOT add new features to the loader
  - Do NOT change variable names or ordering

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct copy with minimal import path changes
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 4, 5, 6, 7)
  - **Blocks**: Task 9 (OSSEDataset depends on loader output format)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/data/loader.py` — COMPLETE source file to copy. Read every line. Key class: `OSSEDataLoader`. Key methods: `__init__`, `load`, `_load_pickle`, `_validate_data`.

  **API/Type References**:
  - `{SPACEW}/configs/default.yaml:1-15` — `data:` section showing variable names, block_boundaries, cv_folds, cv_strategy — these are the config keys the loader reads
  - `{SPACEW}/src/spacew/data/__init__.py` — Package exports pattern

  **WHY Each Reference Matters**:
  - loader.py IS the source — copy its logic faithfully, only changing import paths
  - default.yaml shows the config keys the loader expects, which our osse_default.yaml must match

  **Acceptance Criteria**:

  ```
  QA Scenario: OSSEDataLoader imports and has expected API
    Tool: Bash (python -c)
    Preconditions: pipeline/data/loader.py exists
    Steps:
      1. Run: python -c "from pipeline.data.loader import OSSEDataLoader; print(dir(OSSEDataLoader)); print('Import OK')"
      2. Verify class has load() method: python -c "from pipeline.data.loader import OSSEDataLoader; assert hasattr(OSSEDataLoader, 'load') or callable(getattr(OSSEDataLoader, 'load', None)); print('API OK')"
    Expected Result: Class imports without error, has load() method
    Failure Indicators: ImportError, missing methods
    Evidence: .sisyphus/evidence/task-3-loader-import.txt

  QA Scenario: No spacew imports remain
    Tool: Bash (grep)
    Preconditions: pipeline/data/loader.py exists
    Steps:
      1. Search for spacew references: grep -n "spacew" pipeline/data/loader.py
    Expected Result: Zero matches — no references to spacew package
    Failure Indicators: Any line containing "spacew" (imports or otherwise)
    Evidence: .sisyphus/evidence/task-3-no-spacew-refs.txt
  ```

  **Commit**: YES (groups with Commit A)

- [x] 4. Preprocessor + NaNSafeScaler

  **What to do**:
  - Copy `{SPACEW}/src/spacew/data/preprocessor.py` → `pipeline/data/preprocessor.py`
  - Adapt imports: `from spacew.config import ...` → `from pipeline.config import ...`
  - Keep ALL core functionality:
    - `NaNSafeScaler` class: per-feature mean/std computed via `np.nanmean/nanstd` over axes (0, 1) producing shape `[1, 1, F]`
    - `log1p` transform for density variables (`GDC_DENSITY_ION_OP`, `GDC_DENSITY_NEUTRAL_O`, `GDC_DENSITY_NEUTRAL_O2`) with epsilon=1e-30
    - `Preprocessor` class orchestrating: log transform → fit scaler → transform/inverse_transform
    - `prepare_splits_block_level()` — 4-fold leave-one-block-out splitting (train 2, val 1, test 1)
    - `create_windows_3d()` — windowing with stride, producing `[n_samples, window_size, n_features]`
    - No cross-block windows validation
    - Fold definitions: Fold 0=[train 2,3 / val 1 / test 0], Fold 1=[train 0,3 / val 2 / test 1], Fold 2=[train 0,1 / val 3 / test 2], Fold 3=[train 1,2 / val 0 / test 3]
  - CRITICAL: Scaler fits on TRAINING blocks only (per fold). Never fit on val/test data.
  - CRITICAL: Per-feature statistics are `[1, 1, F]` shaped (over satellites AND time), NOT per-satellite or per-timestep

  **Must NOT do**:
  - Do NOT change the normalization formula or log epsilon value
  - Do NOT change fold definitions or CV strategy
  - Do NOT add per-satellite or time-varying normalization
  - Do NOT change windowing stride logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct copy with import path changes — the logic is complex but we're not modifying it
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 5, 6, 7)
  - **Blocks**: Task 9 (OSSEDataset depends on preprocessor output format)
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/data/preprocessor.py` — COMPLETE source file to copy. Read every line. Key classes: `NaNSafeScaler`, `Preprocessor`. Key functions: `prepare_splits_block_level`, `create_windows_3d`.

  **API/Type References**:
  - `{SPACEW}/configs/default.yaml:20-30` — `normalization:` section showing density_vars, log_epsilon, scaler_type, fit_on_train_only
  - `{SPACEW}/scripts/train.py:50-100` — Shows how Preprocessor is used in the training loop: fit on train blocks, transform all splits

  **Test References**:
  - `{SPACEW}/tests/test_data.py` — Shows round-trip normalization tests and NaN preservation checks that our smoke tests should mirror

  **WHY Each Reference Matters**:
  - preprocessor.py IS the source — copy faithfully
  - train.py shows the CORRECT usage pattern (fit on train only) that our training script must follow
  - test_data.py shows what properties to verify in our smoke tests
  - spacew remediation plan found a scaler data leakage bug — verify our copy doesn't have it

  **Acceptance Criteria**:

  ```
  QA Scenario: Preprocessor and NaNSafeScaler import correctly
    Tool: Bash (python -c)
    Preconditions: pipeline/data/preprocessor.py exists
    Steps:
      1. Run: python -c "from pipeline.data.preprocessor import NaNSafeScaler, Preprocessor, prepare_splits_block_level, create_windows_3d; print('All preprocessor imports OK')"
    Expected Result: All classes and functions import without error
    Failure Indicators: ImportError
    Evidence: .sisyphus/evidence/task-4-preprocessor-import.txt

  QA Scenario: NaNSafeScaler produces correct statistics shape
    Tool: Bash (python -c)
    Preconditions: pipeline/data/preprocessor.py exists, numpy installed
    Steps:
      1. Run: python -c "
         import numpy as np
         from pipeline.data.preprocessor import NaNSafeScaler
         data = np.random.randn(6, 240, 8)
         data[0, 10, 3] = np.nan  # inject NaN
         scaler = NaNSafeScaler()
         scaler.fit(data)
         assert scaler.mean_.shape == (1, 1, 8), f'Mean shape wrong: {scaler.mean_.shape}'
         assert scaler.std_.shape == (1, 1, 8), f'Std shape wrong: {scaler.std_.shape}'
         transformed = scaler.transform(data)
         assert np.isnan(transformed[0, 10, 3]), 'NaN should be preserved'
         print('NaNSafeScaler OK')
         "
    Expected Result: Scaler produces [1,1,8] stats, preserves NaN
    Failure Indicators: Wrong shape, NaN not preserved
    Evidence: .sisyphus/evidence/task-4-scaler-shape.txt

  QA Scenario: No spacew imports remain
    Tool: Bash (grep)
    Steps:
      1. grep -n "spacew" pipeline/data/preprocessor.py
    Expected Result: Zero matches
    Evidence: .sisyphus/evidence/task-4-no-spacew-refs.txt
  ```

  **Commit**: YES (groups with Commit A)

- [x] 5. Masking Module

  **What to do**:
  - Copy `{SPACEW}/src/spacew/data/masking.py` → `pipeline/data/masking.py`
  - This module should need MINIMAL changes — masking is pure numpy, no spacew-specific imports
  - Verify the following functions exist and work:
    - `apply_missing_pattern(data, pattern, rate, seed)` — main entry point
    - Point MCAR: random scattered missing values
    - Subsequence: contiguous segments per feature
    - Block: time intervals affecting all features
    - `enforce_exact_rate` parameter support
  - Fix any import paths if needed (likely none — this is self-contained numpy code)

  **Must NOT do**:
  - Do NOT add new masking patterns
  - Do NOT change masking logic
  - Do NOT change rate enforcement behavior

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Near-direct copy, masking module is self-contained numpy code
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 6, 7)
  - **Blocks**: Task 8 (TSRMImputationExternal needs to understand mask format)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/data/masking.py` — COMPLETE source file. Read every line. Key function: `apply_missing_pattern()`. Returns tuple: `(masked_data, mask, actual_rate)`. Mask convention: 1 = observed, 0 = missing.

  **Test References**:
  - `{SPACEW}/tests/test_masking.py` — Shows rate enforcement tests, pattern validation, edge cases

  **WHY Each Reference Matters**:
  - masking.py is the definitive source — copy as-is if no spacew imports exist
  - test_masking.py shows what properties the masking module guarantees

  **Acceptance Criteria**:

  ```
  QA Scenario: All masking patterns work
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import numpy as np
         from pipeline.data.masking import apply_missing_pattern
         data = np.random.randn(10, 30, 8)
         for pattern in ['point', 'subseq', 'block']:
             masked, mask, rate = apply_missing_pattern(data.copy(), pattern, 0.2, seed=42)
             assert masked.shape == data.shape, f'{pattern}: shape mismatch'
             assert mask.shape == data.shape, f'{pattern}: mask shape mismatch'
             assert np.isnan(masked[mask == 0]).all(), f'{pattern}: masked positions should be NaN'
             assert 0.1 < rate < 0.35, f'{pattern}: rate {rate} out of range'
             print(f'{pattern}: OK (rate={rate:.3f})')
         print('All masking patterns OK')
         "
    Expected Result: All 3 patterns produce correct shapes, NaN at masked positions, reasonable rates
    Failure Indicators: Shape mismatch, non-NaN at masked positions, rate outside range
    Evidence: .sisyphus/evidence/task-5-masking.txt

  QA Scenario: No spacew imports remain
    Tool: Bash (grep)
    Steps:
      1. grep -n "spacew" pipeline/data/masking.py
    Expected Result: Zero matches
    Evidence: .sisyphus/evidence/task-5-no-spacew-refs.txt
  ```

  **Commit**: YES (groups with Commit A)

- [x] 6. Evaluation Metrics

  **What to do**:
  - Copy `{SPACEW}/src/spacew/evaluation/metrics.py` → `pipeline/evaluation/metrics.py`
  - Adapt imports: remove spacew-specific imports
  - Keep ALL core functionality:
    - `compute_metrics(imputed, truth, mask)` — MSE and MAE on masked positions only
    - `compute_skill_scores_vs_baselines(model_metrics, baseline_metrics, eps=1e-10)` — MSESS and MAESS
    - `compute_shared_eval_mask(art_mask, truth, *method_outputs)` — shared evaluation mask excluding NaN from ANY method
    - Per-variable metric computation
    - Division-safe handling: eps=1e-10, both-perfect=0.0, baseline-perfect-model-not=-inf
  - Ensure mask convention matches masking module: 1 = observed, 0 = missing (metrics compute on mask == 0 positions)

  **Must NOT do**:
  - Do NOT add significance tests or statistical analysis
  - Do NOT add per-satellite breakdown
  - Do NOT change metric formulas or eps values

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct copy with import path changes
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 5, 7)
  - **Blocks**: Task 14 (evaluation script depends on metrics module)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/evaluation/metrics.py` — COMPLETE source file. Read every line. Key functions: `compute_metrics`, `compute_skill_scores_vs_baselines`, `compute_shared_eval_mask`, per-variable aggregation.

  **Test References**:
  - `{SPACEW}/tests/test_eval.py` — Evaluation test patterns

  **WHY Each Reference Matters**:
  - metrics.py IS the source — copy faithfully
  - Must preserve the shared eval mask logic (spacew remediation plan identified this as critical for fair comparison)

  **Acceptance Criteria**:

  ```
  QA Scenario: Metrics compute correctly on synthetic data
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import numpy as np
         from pipeline.evaluation.metrics import compute_metrics
         truth = np.ones((10, 30, 8))
         imputed = truth + 0.1  # small error
         mask = np.ones_like(truth)
         mask[:, :5, :] = 0  # mask first 5 timesteps
         metrics = compute_metrics(imputed, truth, mask)
         assert 'mse' in metrics, 'Missing MSE'
         assert 'mae' in metrics, 'Missing MAE'
         assert metrics['mse'] > 0, 'MSE should be positive'
         assert metrics['mae'] > 0, 'MAE should be positive'
         print(f'MSE={metrics[\"mse\"]:.6f}, MAE={metrics[\"mae\"]:.6f}')
         print('Metrics OK')
         "
    Expected Result: MSE and MAE computed, both positive for imperfect imputation
    Failure Indicators: Missing keys, negative values, zero for non-perfect imputation
    Evidence: .sisyphus/evidence/task-6-metrics.txt

  QA Scenario: Shared eval mask excludes NaN from any method
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import numpy as np
         from pipeline.evaluation.metrics import compute_shared_eval_mask
         art_mask = np.zeros((10, 30, 8))  # all masked
         truth = np.ones((10, 30, 8))
         method1 = np.ones((10, 30, 8))
         method1[0, 0, 0] = np.nan  # one NaN in method1
         shared = compute_shared_eval_mask(art_mask, truth, method1)
         assert shared[0, 0, 0] == 1, 'NaN position should be excluded (marked as observed/not-evaluated)'
         print('Shared eval mask OK')
         "
    Expected Result: Positions with NaN in any method are excluded from evaluation
    Evidence: .sisyphus/evidence/task-6-shared-mask.txt
  ```

  **Commit**: YES (groups with Commit A)

- [x] 7. Baselines

  **What to do**:
  - Copy `{SPACEW}/src/spacew/evaluation/baselines.py` → `pipeline/evaluation/baselines.py`
  - This module should need MINIMAL changes — baselines are pure numpy/pandas
  - Keep functions:
    - `locf_impute(data, mask)` — Last Observation Carried Forward, with backward-fill for leading NaN
    - `linear_impute(data, mask)` — Linear interpolation via pandas `interpolate(method='linear', limit_direction='both')`
  - Fix any import paths if needed

  **Must NOT do**:
  - Do NOT add new baseline methods
  - Do NOT change interpolation parameters

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Near-direct copy, self-contained module
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2, 3, 4, 5, 6)
  - **Blocks**: Task 14 (evaluation script depends on baselines)
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/evaluation/baselines.py` — COMPLETE source file. Key functions: `locf_impute`, `linear_impute`. Both take `(data, mask)` and return imputed array.

  **WHY Each Reference Matters**:
  - baselines.py IS the source — copy faithfully, these are simple operations

  **Acceptance Criteria**:

  ```
  QA Scenario: Both baselines produce valid imputations
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import numpy as np
         from pipeline.evaluation.baselines import locf_impute, linear_impute
         data = np.array([1, 2, np.nan, np.nan, 5, 6, np.nan, 8]).reshape(1, 8, 1)
         mask = np.where(np.isnan(data), 0, 1)
         locf_result = locf_impute(data.copy(), mask)
         linear_result = linear_impute(data.copy(), mask)
         assert not np.isnan(locf_result).any(), 'LOCF should fill all NaN'
         assert not np.isnan(linear_result).any(), 'Linear should fill all NaN'
         print(f'LOCF: {locf_result.flatten()}')
         print(f'Linear: {linear_result.flatten()}')
         print('Baselines OK')
         "
    Expected Result: Both methods fill all NaN positions
    Failure Indicators: Remaining NaN in output
    Evidence: .sisyphus/evidence/task-7-baselines.txt
  ```

  **Commit**: YES (groups with Commit A)

- [x] 8. TSRMImputationExternal Subclass

  **What to do**:
  - Create `architecture/tsrm_external.py` containing class `TSRMImputationExternal(TSRMImputation)`
  - This is the CRITICAL integration point between spacew's external masking and TSRM's model

  **Design specification**:
  ```python
  class TSRMImputationExternal(TSRMImputation):
      """TSRM imputation variant that accepts pre-masked data from external masking.

      Instead of TSRM's internal calc_mask (which detects NaN + adds random MCAR),
      this subclass:
      1. Accepts data where NaN positions = artificially masked by spacew's masking module
      2. Creates mask from NaN positions: mask=1 where NaN (masked), mask=0 where observed
      3. Fills NaN with 0 for forward pass
      4. Computes loss on mask==1 positions only (all artificially masked positions)
      5. Does NOT add any additional random masking (missing_ratio is forced to 0)

      Input format (training_step batch):
          batch = (masked_data, original_data, time_marks_x, time_marks_y)
          - masked_data: [B, seq_len, F] with NaN at masked positions
          - original_data: [B, seq_len, F] ground truth (no NaN at masked positions)
          - time_marks_x: [B, seq_len, time_features] time embeddings
          - time_marks_y: [B, seq_len, time_features] (same as x for imputation)

      Output: reconstruction [B, seq_len, F], attention maps
      Loss: ImputationLoss on mask==1 positions comparing reconstruction vs original_data
      """
  ```

  - **Override `_run()` method** (NOT `calc_mask`) to replace the ENTIRE imputation forward pass:
    1. Unpack batch: `masked_data, original_data, time_marks_x, time_marks_y = iteration_batch`
    2. Create mask: `mask = torch.isnan(masked_data)` — True where masked (NaN), False where observed (boolean for `torch.masked_fill`)
    3. Fill NaN: `input_data = torch.nan_to_num(masked_data, nan=0.0)`
    4. Forward: `output, attn_map = self.forward(input_data, time_marks_x, mask=mask)`
       - NOTE: `TSRM.forward(encoding, x_mark, mask=None)` does embedding internally via `self.encoding_ff(encoding, x_mark)`
       - DO NOT call `self.embedding()` directly — pass `time_marks_x` as `x_mark` parameter
    5. Compute loss: `loss = self.loss(output, original_data, mask)` using inherited `ImputationLoss`
    6. Log metrics: reuse parent's logging pattern
    7. Return loss

  - **Override `training_step()`** to call our `_run()` instead of parent's
  - **Override `validation_step()`** similarly
  - **Add `impute()` method** for inference:
    - Takes masked_data + time_marks
    - Returns reconstructed data with only masked positions filled (observed positions keep original values)

  - Ensure config enforces: `revin=False`, `missing_ratio=0`
  - Import ImputationLoss from `architecture.loss_functions`

  **Must NOT do**:
  - Do NOT modify `architecture/model.py` — only ADD `architecture/tsrm_external.py`
  - Do NOT change the forward() method — reuse parent's transformer forward pass
  - Do NOT implement any masking logic — all masking is external
  - Do NOT enable RevIN

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: This is the most technically complex task — requires deep understanding of TSRM's internal flow, PyTorch Lightning lifecycle, and careful overriding of methods
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 9, 10, 11, 12)
  - **Blocks**: Tasks 13, 16 (training script and smoke tests depend on this)
  - **Blocked By**: Task 5 (need to understand mask format from masking module)

  **References**:

  **Pattern References** (CRITICAL — read ALL of these):
  - `{TSRM}/architecture/model.py:360-430` — `TSRMImputation` class. Read the ENTIRE class. Shows `_run()` method, `calc_mask()`, `training_step()`, `validation_step()`, how loss is computed. This is the parent class you're subclassing.
  - `{TSRM}/architecture/model.py:85-200` — `TSRM` base class. Shows `forward()`, `__init__`, `configure_optimizers`. Your subclass inherits this.
  - `{TSRM}/architecture/model.py:200-260` — `Transformations` class and `EncodingLayer`. Shows the transformer architecture that `forward()` uses.

  **API/Type References**:
  - `{TSRM}/architecture/loss_functions.py:1-40` — `ImputationLoss` class. Shows constructor params (`mode`, `alpha`, `loss_func`) and `forward(output, target, mask)` signature. Mode "imputation" computes loss on mask==1 positions only.
  - `{TSRM}/architecture/utils.py:1-41` — `build_mask`, `build_mask_from_data`, `add_noise` — these are what `calc_mask` uses internally. You DON'T call these, but understand what they do to understand what you're replacing.
  - `{TSRM}/architecture/model.py:305-330` — `TSRM.forward(encoding, x_mark, mask=None)` signature. CRITICAL: This method does embedding INTERNALLY via `self.encoding_ff(encoding, x_mark)` at line 310. DO NOT precompute embeddings — pass `time_marks_x` directly as `x_mark`.
  - `{TSRM}/embedding/data_embedding.py:1-111` — `DataEmbedding` class. Shows how time features are embedded internally. You don't call this directly — `TSRM.forward()` calls it.

  **External References**:
  - `{SPACEW}/src/spacew/data/masking.py` — Mask convention: returned mask has 1=observed, 0=missing. NaN is placed at mask==0 positions. Your subclass must handle this convention.

  **WHY Each Reference Matters**:
  - model.py:360-430 is the PARENT you're overriding — must understand every line to override correctly
  - loss_functions.py defines the loss API — your `_run()` must call it with correct arguments
  - data_embedding.py shows how to get embeddings — your `_run()` needs embeddings for the forward pass
  - masking.py defines the mask convention (1=observed, 0=missing) — but TSRM internally uses (1=masked, 0=observed) for loss. You must CONVERT: `loss_mask = 1 - external_mask` or detect NaN directly as `torch.isnan(masked_data).float()`

  **Acceptance Criteria**:

  ```
  QA Scenario: TSRMImputationExternal instantiates with config
    Tool: Bash (python -c)
    Preconditions: architecture/tsrm_external.py exists
    Steps:
      1. Run: python -c "
         from architecture.tsrm_external import TSRMImputationExternal
         config = {
              'feature_dimension': 8, 'seq_len': 30, 'pred_len': 0,
              'encoding_size': 64, 'h': 4, 'N': 2,
              'conv_dims': [[3, 1, 1], [5, 1, 1]],
              'attention_func': 'entmax15',
              'batch_size': 32, 'dropout': 0.1,
              'revin': False, 'missing_ratio': 0.0,
              'loss_function_imputation': 'mse+mae',
              'loss_imputation_mode': 'imputation',
              'loss_weight_alpha': 0.5,
              'embed': 'timeF', 'freq': 't',
              'mask_size': 3, 'mask_count': 1,
              'task': 'imputation', 'phase': 'downstream',  # REQUIRED by Transformations.__init__
          }
         model = TSRMImputationExternal(config)
         print(f'Model created: {type(model).__name__}')
         print(f'Parameters: {sum(p.numel() for p in model.parameters())}')
         print('Instantiation OK')
         "
    Expected Result: Model instantiates without error, has parameters
    Failure Indicators: ImportError, KeyError on config, any crash
    Evidence: .sisyphus/evidence/task-8-instantiation.txt

  QA Scenario: Forward pass produces correct output shape
    Tool: Bash (python -c)
    Preconditions: TSRMImputationExternal exists
    Steps:
      1. Run: python -c "
         import torch
         from architecture.tsrm_external import TSRMImputationExternal
         config = {
              'feature_dimension': 8, 'seq_len': 30, 'pred_len': 0,
              'encoding_size': 64, 'h': 4, 'N': 2,
              'conv_dims': [[3, 1, 1], [5, 1, 1]],
              'attention_func': 'entmax15',
              'batch_size': 4, 'dropout': 0.1,
              'revin': False, 'missing_ratio': 0.0,
              'loss_function_imputation': 'mse+mae',
              'loss_imputation_mode': 'imputation',
              'loss_weight_alpha': 0.5,
              'embed': 'timeF', 'freq': 't',
              'mask_size': 3, 'mask_count': 1,
              'task': 'imputation', 'phase': 'downstream',  # REQUIRED by Transformations.__init__
          }
         model = TSRMImputationExternal(config)
         model.eval()
         B, T, F = 4, 30, 8
         masked_data = torch.randn(B, T, F)
         masked_data[:, 5:10, :] = float('nan')  # mask timesteps 5-9
         original_data = torch.randn(B, T, F)
         time_marks = torch.randn(B, T, 5)  # 5 time features
         with torch.no_grad():
             result = model.impute(masked_data, original_data, time_marks, time_marks)
         assert result.shape == (B, T, F), f'Output shape wrong: {result.shape}'
         assert not torch.isnan(result).any(), 'Output should have no NaN'
         print(f'Output shape: {result.shape}')
         print('Forward pass OK')
         "
    Expected Result: Output shape [4, 30, 8], no NaN in output
    Failure Indicators: Wrong shape, NaN in output, crash
    Evidence: .sisyphus/evidence/task-8-forward.txt

  QA Scenario: No existing TSRM files modified
    Tool: Bash (git)
    Steps:
      1. Run: git diff --name-only architecture/model.py architecture/loss_functions.py architecture/utils.py
    Expected Result: No changes to existing files — only architecture/tsrm_external.py is new
    Failure Indicators: Any diff in existing architecture/ files
    Evidence: .sisyphus/evidence/task-8-no-modifications.txt
  ```

  **Commit**: YES (groups with Tasks 9-12 in Commit B)
  - Message: `feat(pipeline): add TSRM adaptation layer (external masking, OSSE dataset, HP grid)`

- [x] 9. OSSEDataset + Time Embeddings

  **What to do**:
  - Create `pipeline/data/dataset.py` containing class `OSSEDataset(torch.utils.data.Dataset)`
  - This bridges spacew's windowed numpy arrays into TSRM's expected tuple format

  **Design specification**:
  ```python
  class OSSEDataset(torch.utils.data.Dataset):
      """PyTorch Dataset wrapping spacew's preprocessed windows for TSRM.

      Takes:
          masked_windows: np.ndarray [n_samples, window_size, n_features] — NaN at masked positions
          original_windows: np.ndarray [n_samples, window_size, n_features] — ground truth
          timestamps: np.ndarray [n_samples, window_size] — datetime-like timestamps per window
          freq: str — time frequency for feature generation (default 't' for minute)

      Returns per __getitem__:
          (masked_data, original_data, time_marks_x, time_marks_y) — all torch.Tensor
          - masked_data: [window_size, n_features] float32
          - original_data: [window_size, n_features] float32
          - time_marks_x: [window_size, n_time_features] float32
          - time_marks_y: [window_size, n_time_features] float32 (same as x for imputation)
      """
  ```

  - **Time embedding generation**:
    - Generate time features from real OSSE timestamps (not synthetic sequential)
    - Use TSRM's time feature convention: extract [month, day, weekday, hour, minute] normalized to [-0.5, 0.5]
    - Reference TSRM's existing `TimeFeatureEmbedding` input format in `embedding/data_embedding.py`
    - If timestamps are unavailable (e.g., test mode), generate synthetic sequential timestamps at 1-minute intervals within each block

  - **Add `create_dataloaders()` factory function**:
    ```python
    def create_dataloaders(train_masked, train_original, train_timestamps,
                           val_masked, val_original, val_timestamps,
                           batch_size=32, num_workers=0):
        """Create train and val DataLoaders from preprocessed windows."""
        # Shuffle train, don't shuffle val
    ```

  - Handle edge cases:
    - Windows where ALL positions are masked (skip or handle gracefully)
    - Windows where NO positions are masked (valid — loss will be 0)
    - NaN propagation: ensure NaN in masked_data is preserved until model receives it

  **Must NOT do**:
  - Do NOT perform any normalization — that's the preprocessor's job
  - Do NOT apply any masking — that's the masking module's job
  - Do NOT modify timestamps or generate fake timestamps when real ones are available

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding TSRM's expected input format, time feature generation, and correct tensor conversion from numpy with NaN handling
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 10, 11, 12)
  - **Blocks**: Tasks 13, 16 (training script and smoke tests depend on this)
  - **Blocked By**: Tasks 3, 4 (need to understand loader output format and preprocessor window format)

  **References**:

  **Pattern References**:
  - `{TSRM}/data_provider/data_loader.py:1-100` — `Dataset_ETT_hour` class. Shows TSRM's expected Dataset pattern: `__getitem__` returning `(seq_x, seq_y, seq_x_mark, seq_y_mark)`. Our OSSEDataset must return the same tuple structure but with (masked_data, original_data, time_marks_x, time_marks_y).
  - `{TSRM}/data_provider/data_loader.py:100-180` — Time stamp feature extraction pattern. Shows how existing datasets extract time features from datetime columns.
  - `{SPACEW}/src/spacew/data/preprocessor.py` — `create_windows_3d()` function. Shows the output format of windowed data: `[n_samples, window_size, n_features]`. This is what OSSEDataset receives as input.
  - `{SPACEW}/src/spacew/model/saits_wrapper.py:60-120` — `SAITSWrapper._prepare_pypots_data()`. Shows how spacew converts windowed numpy to model input. Our OSSEDataset does the equivalent for TSRM.

  **API/Type References**:
  - `{TSRM}/embedding/data_embedding.py:50-80` — `TimeFeatureEmbedding`. Shows the expected dimensionality of time features (depends on `freq` parameter). For freq='t' (minutely): 5 features [month, day, weekday, hour, minute].
  - `{TSRM}/data_provider/data_factory.py:1-53` — `data_provider()` factory. Shows how DataLoaders are created with batch_size, shuffle, num_workers. Our `create_dataloaders()` should follow this pattern.

  **WHY Each Reference Matters**:
  - Dataset_ETT_hour defines the CONTRACT — our dataset must return same tuple shape
  - TimeFeatureEmbedding shows expected time feature count — our time marks must have correct dimensions
  - SAITSWrapper shows the equivalent spacew pattern — understand what we're replacing
  - create_windows_3d shows our INPUT format — we must handle [n_samples, window_size, n_features] correctly

  **Acceptance Criteria**:

  ```
  QA Scenario: OSSEDataset returns correct tuple format
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import numpy as np, torch
         from pipeline.data.dataset import OSSEDataset
         n_samples, window_size, n_features = 20, 30, 8
         masked = np.random.randn(n_samples, window_size, n_features)
         masked[:, 5:10, :] = np.nan  # mask some positions
         original = np.random.randn(n_samples, window_size, n_features)
         # Create synthetic timestamps (1-minute intervals)
         from datetime import datetime, timedelta
         base = datetime(2025, 1, 1, 9, 0)
         timestamps = np.array([[base + timedelta(minutes=i) for i in range(window_size)] for _ in range(n_samples)])
         ds = OSSEDataset(masked, original, timestamps, freq='t')
         assert len(ds) == n_samples
         item = ds[0]
         assert len(item) == 4, f'Expected 4-tuple, got {len(item)}'
         masked_t, orig_t, tm_x, tm_y = item
         assert masked_t.shape == (window_size, n_features), f'masked shape: {masked_t.shape}'
         assert orig_t.shape == (window_size, n_features), f'original shape: {orig_t.shape}'
         assert tm_x.shape[0] == window_size, f'time marks rows: {tm_x.shape[0]}'
         assert tm_x.shape[1] >= 4, f'time mark features: {tm_x.shape[1]} (need >=4)'
         assert torch.isnan(masked_t[5, 0]), 'NaN should be preserved in masked data'
         print(f'Dataset: {len(ds)} samples')
         print(f'Shapes: masked={masked_t.shape}, orig={orig_t.shape}, tm_x={tm_x.shape}, tm_y={tm_y.shape}')
         print('OSSEDataset OK')
         "
    Expected Result: 4-tuple with correct shapes, NaN preserved in masked data, time marks have >=4 features
    Failure Indicators: Wrong tuple length, wrong shapes, NaN not preserved
    Evidence: .sisyphus/evidence/task-9-dataset.txt

  QA Scenario: DataLoader batching works
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import numpy as np
         from pipeline.data.dataset import OSSEDataset, create_dataloaders
         from datetime import datetime, timedelta
         n, T, F = 20, 30, 8
         masked = np.random.randn(n, T, F); masked[:, 5:10, :] = np.nan
         original = np.random.randn(n, T, F)
         base = datetime(2025, 1, 1, 9, 0)
         ts = np.array([[base + timedelta(minutes=i) for i in range(T)] for _ in range(n)])
         train_dl, val_dl = create_dataloaders(masked[:15], original[:15], ts[:15], masked[15:], original[15:], ts[15:], batch_size=4)
         batch = next(iter(train_dl))
         assert len(batch) == 4, f'Batch tuple length: {len(batch)}'
         assert batch[0].shape[0] == 4, f'Batch size: {batch[0].shape[0]}'
         print(f'Batch shapes: {[b.shape for b in batch]}')
         print('DataLoader OK')
         "
    Expected Result: DataLoader returns batches of 4-tuples with correct batch size
    Failure Indicators: Wrong batch size, wrong tuple structure
    Evidence: .sisyphus/evidence/task-9-dataloader.txt
  ```

  **Commit**: YES (groups with Commit B)

- [x] 10. TSRM Config YAML + Defaults

  **What to do**:
  - Create `configs/osse_default.yaml` containing the default TSRM configuration for OSSE imputation
  - This config should follow our YAML structure (not TSRM's native config format) — `build_tsrm_config()` from Task 2 will translate it

  **Required YAML structure**:
  ```yaml
  # configs/osse_default.yaml
  
  data:
    variables:
      - GDC_TEMPERATURE
      - GDC_TEMPERATURE_ION
      - GDC_TEMPERATURE_ELEC
      - GDC_VELOCITY_U
      - GDC_VELOCITY_V
      - GDC_DENSITY_ION_OP
      - GDC_DENSITY_NEUTRAL_O
      - GDC_DENSITY_NEUTRAL_O2
    block_boundaries: [0, 60, 120, 180, 240]
    cv_folds: 4
    cv_strategy: leave_one_block_out
    window_size: 30
    window_stride_train: 5
    window_stride_eval: 30

  normalization:
    density_vars:
      - GDC_DENSITY_ION_OP
      - GDC_DENSITY_NEUTRAL_O
      - GDC_DENSITY_NEUTRAL_O2
    log_epsilon: 1.0e-30
    scaler_type: nan_safe_standard
    fit_on_train_only: true

  masking:
    patterns: [point, subseq, block]
    missing_rates: [0.1, 0.2, 0.3]
    enforce_exact_rate: true
    augmentation_factor: 5
    seed: 42

  # TSRM model config — will be translated to TSRM dict by build_tsrm_config()
  tsrm:
    # Architecture
    encoding_size: 64
    h: 4                    # attention heads
    N: 2                    # encoder layers
    conv_dims: [[3, 1, 1], [5, 1, 1], [7, 1, 1]]  # kernel, dilation, groups
    attention_func: entmax15
    dropout: 0.1

    # Training
    batch_size: 32
    learning_rate: 0.001
    epochs: 100
    patience: 15            # early stopping patience

    # Loss
    loss_function_imputation: mse+mae
    loss_imputation_mode: imputation
    loss_weight_alpha: 0.5

    # Time embedding
    embed: timeF
    freq: t                 # minutely

    # Masking (for TSRM internals — ALWAYS these values for external masking)
    mask_size: 3
    mask_count: 1
    # revin: false          # Hardcoded in build_tsrm_config()
    # missing_ratio: 0.0    # Hardcoded in build_tsrm_config()
    # task: imputation      # Hardcoded in build_tsrm_config()
    # phase: downstream     # Hardcoded in build_tsrm_config()

  training:
    accelerator: auto       # auto-detect GPU
    precision: 16-mixed
    gradient_clip_val: 1.0
    num_workers: 4

  evaluation:
    baselines: [locf, linear]
    compute_per_variable: true
    skill_score_eps: 1.0e-10

  paths:
    data_dir: ${DATA_DIR}
    scratch_dir: ${SCRATCH_DIR}
  ```

  **Must NOT do**:
  - Do NOT set `revin: true` — this is hardcoded to False in `build_tsrm_config()`
  - Do NOT set `missing_ratio > 0` — this is hardcoded to 0 in `build_tsrm_config()`

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: YAML file creation with reference values
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 9, 11, 12)
  - **Blocks**: Task 12 (hyperparameter search grid needs base config)
  - **Blocked By**: Task 2 (config module must exist to load this)

  **References**:

  **Pattern References**:
  - `{SPACEW}/configs/default.yaml` — spacew config structure. Our YAML follows this pattern for data/normalization/masking sections.
  - `{TSRM}/experiments/configs/imputation_etth1.yml` — TSRM config example showing model-related keys (NOTE: extension is `.yml` not `.yaml`).

  **API/Type References**:
  - `{TSRM}/architecture/model.py:272-305` — TSRM.__init__ shows required config dict keys

  **WHY Each Reference Matters**:
  - spacew default.yaml shows the config structure our pipeline expects
  - TSRM configs show model-related keys needed for `build_tsrm_config()`

  **Acceptance Criteria**:

  ```
  QA Scenario: Config loads and builds valid TSRM dict
    Tool: Bash (python -c)
    Preconditions: configs/osse_default.yaml exists, pipeline/config.py exists
    Steps:
      1. Run: python -c "
         from pipeline.config import load_config, build_tsrm_config
         cfg = load_config('configs/osse_default.yaml')
         tsrm_cfg = build_tsrm_config(cfg)
         required = ['feature_dimension', 'seq_len', 'pred_len', 'encoding_size', 'h', 'N', 'conv_dims', 'attention_func', 'batch_size', 'dropout', 'revin', 'loss_function_imputation', 'loss_imputation_mode', 'missing_ratio', 'embed']
         missing = [k for k in required if k not in tsrm_cfg]
         assert not missing, f'Missing keys: {missing}'
         assert tsrm_cfg['revin'] == False, 'revin must be False'
         assert tsrm_cfg['missing_ratio'] == 0.0, 'missing_ratio must be 0'
         assert tsrm_cfg['feature_dimension'] == 8, 'Wrong feature dimension'
         assert tsrm_cfg['seq_len'] == 30, 'Wrong seq_len'
         print('Config loaded and validated')
         print(f'revin={tsrm_cfg[\"revin\"]}, missing_ratio={tsrm_cfg[\"missing_ratio\"]}')
         print('Config OK')
         "
    Expected Result: Config loads, TSRM dict has all required keys, revin=False, missing_ratio=0
    Failure Indicators: Missing keys, wrong values for revin/missing_ratio
    Evidence: .sisyphus/evidence/task-10-config-load.txt
  ```

  **Commit**: YES (groups with Commit B)

- [x] 11. Experiment Tracking Module

  **What to do**:
  - Copy `{SPACEW}/src/spacew/tracking/experiment.py` → `pipeline/tracking/experiment.py`
  - Adapt imports: `from spacew.config import ...` → `from pipeline.config import ...`
  - Keep core functionality:
    - `ExperimentTracker` class with atomic directory creation
    - Hybrid ID format: `{timestamp}_{name}`
    - Collision handling with numeric suffixes
    - Config snapshot after CLI overrides
    - Git metadata capture (commit hash, dirty state)
    - Environment capture (Python version, package versions)
    - Directory structure: `models/fold_N/`, `evaluation/`, `plots/`
  - Update any spacew-specific references to TSRM/pipeline

  **Must NOT do**:
  - Do NOT add new tracking features beyond what spacew has
  - Do NOT change directory structure or ID format

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct copy with import path changes
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 9, 10, 12)
  - **Blocks**: Task 13 (training script needs experiment tracking)
  - **Blocked By**: Task 1 (directory structure must exist)

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/tracking/experiment.py` — COMPLETE source file. Key class: `ExperimentTracker`. Key methods: `__init__`, `save_config`, `save_metadata`, `get_model_dir`, `get_eval_dir`.

  **WHY Each Reference Matters**:
  - experiment.py IS the source — copy faithfully, only change imports

  **Acceptance Criteria**:

  ```
  QA Scenario: ExperimentTracker creates experiment directory
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import tempfile, os
         from pipeline.tracking.experiment import ExperimentTracker
         with tempfile.TemporaryDirectory() as tmpdir:
             tracker = ExperimentTracker(tmpdir, 'test_exp', config={'test': True}, seed=42)
             assert os.path.isdir(tracker.experiment_dir)
             assert 'test_exp' in tracker.experiment_id
             print(f'Created: {tracker.experiment_dir}')
             print(f'ID: {tracker.experiment_id}')
             print('ExperimentTracker OK')
         "
    Expected Result: Experiment directory created with valid ID
    Failure Indicators: Directory not created, exception raised
    Evidence: .sisyphus/evidence/task-11-tracker.txt

  QA Scenario: No spacew imports remain
    Tool: Bash (grep)
    Steps:
      1. grep -n "spacew" pipeline/tracking/experiment.py
    Expected Result: Zero matches
    Evidence: .sisyphus/evidence/task-11-no-spacew-refs.txt
  ```

  **Commit**: YES (groups with Commit B)

- [x] 12. Hyperparameter Search Grid

  **What to do**:
  - Create `scripts/search_osse.py` — hyperparameter search script using ParameterGrid
  - Leverage TSRM's existing `experiments/scheduler.py` pattern for grid generation
  - Define search grid that covers:
    - `attention_func`: `["entmax15", "classic", "propsparse"]`
    - `N` (layers): `[1, 2, 3]`
    - `h` (heads): `[2, 4, 8]`
    - `encoding_size`: `[32, 64, 128]`
    - `conv_dims`: Valid configurations for seq_len=30 (receptive field ≤ 30)
      - `[[[3,1,1]], [[3,1,1],[5,1,1]], [[3,1,1],[5,2,1]], [[5,1,1],[7,1,1]]]`
    - `dropout`: `[0.0, 0.1, 0.25]`
    - `learning_rate`: `[0.0001, 0.001, 0.01]`
    - `batch_size`: `[32, 64]`
  - Total grid size: ~3 × 3 × 3 × 3 × 4 × 3 × 3 × 2 = ~5,832 configs — TOO LARGE
  - **Reduce to manageable size** by:
    - Using defaults for most params, only varying key ones
    - Or providing `--quick` mode with smaller grid
  - Script should:
    - Load base config from YAML
    - Generate ParameterGrid
    - For each config: train on fold 0 for a few epochs, record validation loss
    - Save results to CSV/JSON
    - Identify best config by aggregated validation loss across folds

  **Suggested manageable grid (~50 configs)**:
  ```python
  search_grid = {
      "attention_func": ["entmax15", "classic"],  # 2
      "N": [1, 2],                                # 2
      "h": [4, 8],                                # 2
      "encoding_size": [64, 128],                 # 2
      "dropout": [0.0, 0.1],                      # 2
      "learning_rate": [0.0005, 0.001],           # 2
      "batch_size": [32, 64],                     # 2
      # conv_dims: use default from YAML
  }
  # Total: 2^7 = 128 configs — still large, add --quick mode with 2^5 = 32
  ```

  **Script interface**:
  ```bash
  python scripts/search_osse.py --config configs/osse_default.yaml --folds 0,1 --epochs 10 --quick
  python scripts/search_osse.py --config configs/osse_default.yaml --folds all --epochs 50
  ```

  **Must NOT do**:
  - Do NOT implement Bayesian optimization — use ParameterGrid only
  - Do NOT run full 5000+ config grid by default
  - Do NOT modify existing TSRM experiment files

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Requires understanding ParameterGrid, designing a reasonable search space, and implementing a search loop with training
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 8, 9, 10, 11)
  - **Blocks**: Task 13 (training script can reuse search logic)
  - **Blocked By**: Task 10 (needs base config YAML)

  **References**:

  **Pattern References**:
  - `{TSRM}/experiments/scheduler.py` — `next_run()` generator using `ParameterGrid`. Copy this pattern for grid generation.
  - `{TSRM}/experiments/setups.py` — Experiment setup definitions showing how configs are defined and combined.
  - `{TSRM}/experiments/experiment_runner.py` — `ExperimentRun` class showing how to run experiments with different configs.
  - `{SPACEW}/scripts/train.py` — Training loop pattern for 4-fold CV.

  **API/Type References**:
  - `sklearn.model_selection.ParameterGrid` — Grid generation utility

  **WHY Each Reference Matters**:
  - scheduler.py shows the ParameterGrid pattern TSRM already uses
  - spacew train.py shows the 4-fold CV training loop we'll embed in the search

  **Acceptance Criteria**:

  ```
  QA Scenario: Search script generates valid grid
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         from scripts.search_osse import build_search_grid
         grid = build_search_grid(quick=True)
         print(f'Quick grid size: {len(grid)}')
         assert len(grid) <= 50, f'Quick grid too large: {len(grid)}'
         assert all('attention_func' in cfg for cfg in grid)
         print('Search grid OK')
         "
    Expected Result: Grid generated with manageable size
    Failure Indicators: Grid too large, missing required keys
    Evidence: .sisyphus/evidence/task-12-grid.txt

  QA Scenario: Search script has --help
    Tool: Bash
    Steps:
      1. python scripts/search_osse.py --help
    Expected Result: Help text displayed without error
    Failure Indicators: ImportError, no --help
    Evidence: .sisyphus/evidence/task-12-help.txt
  ```

  **Commit**: YES (groups with Commit B)

- [x] 13. Training Script with 4-Fold CV

  **What to do**:
  - Create `scripts/train_osse.py` — main training script implementing 4-fold leave-one-block-out CV

  **Script workflow**:
  ```bash
  python scripts/train_osse.py --config configs/osse_default.yaml --fold 0 --missing-pattern block --missing-rate 0.2
  python scripts/train_osse.py --config configs/osse_default.yaml --all-folds  # run all 4 folds
  ```

  **Implementation steps**:
  1. Parse args: `--config`, `--fold`, `--all-folds`, `--missing-pattern`, `--missing-rate`, `--epochs`, `--experiment-name`
  2. Load config via `pipeline.config.load_config()`
  3. Load OSSE data via `OSSEDataLoader`
  4. Fit preprocessor on TRAINING blocks only (per fold)
  5. Create windows for train/val/test
  6. Apply missing pattern to TRAINING data only (with augmentation factor)
  7. Apply same missing pattern to VALIDATION data (single mask, no augmentation)
  8. Create `OSSEDataset` instances and `DataLoader`s
  9. Build TSRM config via `build_tsrm_config()`
  10. Instantiate `TSRMImputationExternal`
  11. Use PyTorch Lightning `Trainer` with:
      - `max_epochs` from config
      - `callbacks=[EarlyStopping(patience), ModelCheckpoint]`
      - `accelerator='auto'`, `precision='16-mixed'`
      - `gradient_clip_val=1.0`
  12. Train: `trainer.fit(model, train_dataloader, val_dataloader)`
  13. Save model checkpoint and scaler to experiment directory
  14. Log training metrics

  **Critical data flow**:
  ```
  Raw OSSE data [6, 240, 8]
      ↓ (split by blocks based on fold)
  Train blocks [2, 60, 8] + Val block [6, 60, 8] + Test block [6, 60, 8]
      ↓ (fit scaler on train only, transform all)
  Normalized data
      ↓ (create windows with stride)
  Windows [n_samples, 30, 8]
      ↓ (apply missing pattern to train, val)
  Masked windows + original windows + masks
      ↓ (create OSSEDataset with timestamps)
  DataLoader batches
      ↓ (feed to TSRMImputationExternal)
  Trained model
  ```

  **Fold definitions (same as spacew)**:
  - Fold 0: Train=[2,3], Val=[1], Test=[0]
  - Fold 1: Train=[0,3], Val=[2], Test=[1]
  - Fold 2: Train=[0,1], Val=[3], Test=[2]
  - Fold 3: Train=[1,2], Val=[0], Test=[3]

  **Must NOT do**:
  - Do NOT fit scaler on val or test data
  - Do NOT apply augmentation to validation data
  - Do NOT create windows that cross block boundaries
  - Do NOT modify test data during training

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Orchestrates all pipeline components, implements complex data flow, requires understanding of CV strategy and Lightning training
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — this is the main integration task
  - **Parallel Group**: Wave 3 (sequential with Tasks 14, 15)
  - **Blocks**: Tasks 14, 17 (evaluation and E2E test depend on trained models)
  - **Blocked By**: Tasks 8, 9, 10, 11, 12 (model, dataset, config, tracking, search grid)

  **References**:

  **Pattern References**:
  - `{SPACEW}/scripts/train.py` — COMPLETE reference. Copy the structure: arg parsing → config loading → data loading → preprocessing → splitting → windowing → masking → model training → saving. Read every line.
  - `{TSRM}/experiments/experiment_runner.py` — `ExperimentRun.run_config()` shows how Lightning Trainer is configured and called.

  **API/Type References**:
  - `{TSRM}/architecture/model.py:340-360` — `TSRMImputation.training_step()` and `configure_optimizers()`. Our `TSRMImputationExternal` inherits these but overrides `_run()`.
  - `pytorch_lightning.Trainer` documentation — Understand callback configuration.

  **WHY Each Reference Matters**:
  - spacew train.py is the template — copy its structure exactly, only swap SAITS → TSRM
  - TSRM experiment_runner shows Lightning integration pattern

  **Acceptance Criteria**:

  ```
  QA Scenario: Training script runs for 1 fold with minimal epochs
    Tool: Bash
    Preconditions: All pipeline modules exist, configs/osse_default.yaml exists
    Steps:
      1. Run: python scripts/train_osse.py --config configs/osse_default.yaml --fold 0 --epochs 2 --missing-pattern point --missing-rate 0.1 --experiment-name test_run 2>&1 | head -100
    Expected Result: Script starts training, no ImportError, shows epoch progress
    Failure Indicators: ImportError, crash before training starts
    Evidence: .sisyphus/evidence/task-13-train-start.txt

  QA Scenario: Training produces model checkpoint
    Tool: Bash
    Preconditions: Training completed for at least 1 epoch
    Steps:
      1. After training run, check for checkpoint: python -c "
         from pathlib import Path
         checkpoints = list(Path('.').rglob('*.ckpt'))
         print(f'Found {len(checkpoints)} checkpoints')
         for ckpt in checkpoints[:5]: print(f'  {ckpt}')
         assert len(checkpoints) > 0, 'No checkpoints found'
         "
    Expected Result: At least one .ckpt file exists
    Failure Indicators: No checkpoints
    Evidence: .sisyphus/evidence/task-13-checkpoint.txt

  QA Scenario: Training script has --help
    Tool: Bash
    Steps:
      1. python scripts/train_osse.py --help
    Expected Result: Help text with all expected arguments
    Evidence: .sisyphus/evidence/task-13-help.txt
  ```

  **Commit**: YES (groups with Tasks 14-15 in Commit C)
  - Message: `feat(scripts): add training, evaluation, and visualization scripts`

- [x] 14. Evaluation Script

  **What to do**:
  - Create `scripts/evaluate_osse.py` — evaluation script comparing TSRM against baselines

  **Script workflow**:
  ```bash
  python scripts/evaluate_osse.py --config configs/osse_default.yaml --experiment-id 20260218_test
  python scripts/evaluate_osse.py --config configs/osse_default.yaml --fold 0 --missing-pattern block --missing-rate 0.2
  ```

  **Implementation steps**:
  1. Load trained model checkpoint from experiment directory
  2. Load OSSE data and preprocessor (fitted scaler from training)
  3. Get TEST block data for the fold
  4. Apply missing pattern to test data (same pattern/rate as training)
  5. Run TSRM imputation on masked test data
  6. Run baseline imputations (LOCF, linear) on same masked test data
  7. Compute shared evaluation mask (excludes NaN from any method)
  8. Compute metrics on shared mask:
     - MSE, MAE for each method
     - MSESS, MAESS (skill scores vs baselines)
     - Per-variable metrics
  9. Save results to JSON in experiment directory
  10. Optionally generate comparison plots

  **Output format** (`evaluation_results.json`):
  ```json
  {
    "fold": 0,
    "missing_pattern": "block",
    "missing_rate": 0.2,
    "tsrm": {"mse": 0.123, "mae": 0.234, "msess_vs_locf": 0.45, ...},
    "locf": {"mse": 0.456, "mae": 0.567},
    "linear": {"mse": 0.345, "mae": 0.456},
    "per_variable": {"GDC_TEMPERATURE": {...}, ...}
  }
  ```

  **Must NOT do**:
  - Do NOT compute metrics on training data
  - Do NOT use different masking for evaluation than was used for training
  - Do NOT skip shared evaluation mask computation

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integrates model inference, baseline methods, metrics computation, and result aggregation
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — depends on trained models
  - **Parallel Group**: Wave 3 (after Task 13)
  - **Blocks**: Tasks 15, 17 (visualization and E2E test depend on evaluation results)
  - **Blocked By**: Tasks 6, 7, 13 (metrics, baselines, training script)

  **References**:

  **Pattern References**:
  - `{SPACEW}/scripts/evaluate.py` — COMPLETE reference. Copy structure: load model → load test data → apply mask → impute → compute metrics → save results.
  - `{SPACEW}/src/spacew/evaluation/metrics.py` — `compute_shared_eval_mask()` and metric functions.

  **API/Type References**:
  - `{TSRM}/architecture/tsrm_external.py` — `TSRMImputationExternal.impute()` method (defined in Task 8) for inference

  **WHY Each Reference Matters**:
  - spacew evaluate.py is the template — copy its structure
  - shared eval mask is critical for fair comparison (spacew remediation plan)

  **Acceptance Criteria**:

  ```
  QA Scenario: Evaluation script loads model and runs inference
    Tool: Bash
    Preconditions: Trained model checkpoint exists
    Steps:
      1. Run: python scripts/evaluate_osse.py --config configs/osse_default.yaml --fold 0 --missing-pattern point --missing-rate 0.1 2>&1 | head -50
    Expected Result: Script loads model, runs inference, no crash
    Failure Indicators: ImportError, model loading failure, crash during inference
    Evidence: .sisyphus/evidence/task-14-eval-start.txt

  QA Scenario: Evaluation produces results JSON
    Tool: Bash
    Preconditions: Evaluation completed
    Steps:
      1. Run: python -c "
         from pathlib import Path
         results = list(Path('.').rglob('evaluation_results.json'))
         print(f'Found {len(results)} result files')
         for r in results[:3]: print(f'  {r}')
         assert len(results) > 0, 'No evaluation_results.json found'
         "
    Expected Result: evaluation_results.json exists with metrics
    Evidence: .sisyphus/evidence/task-14-results.txt

  QA Scenario: Results contain expected metrics
    Tool: Bash (python -c)
    Steps:
      1. Run: python -c "
         import json
         from pathlib import Path
         results_file = next(Path('.').rglob('evaluation_results.json'))
         data = json.load(open(results_file))
         assert 'tsrm' in data, 'Missing TSRM metrics'
         assert 'mse' in data['tsrm'], 'Missing MSE'
         assert 'mae' in data['tsrm'], 'Missing MAE'
         assert 'locf' in data, 'Missing LOCF baseline'
         print(f'TSRM MSE: {data[\"tsrm\"][\"mse\"]:.6f}')
         print(f'TSRM MAE: {data[\"tsrm\"][\"mae\"]:.6f}')
         print('Evaluation results OK')
         "
    Expected Result: JSON has TSRM and baseline metrics with MSE/MAE
    Failure Indicators: Missing keys, wrong structure
    Evidence: .sisyphus/evidence/task-14-metrics.txt
  ```

  **Commit**: YES (groups with Commit C)

- [x] 15. Visualization Module

  **What to do**:
  - Copy `{SPACEW}/src/spacew/visualization/plots.py` → `pipeline/visualization/plots.py`
  - Adapt imports
  - Keep core visualization functions:
    - Skill score bar charts (TSRM vs baselines)
    - MSE heatmap per variable
    - Time series comparison plots (original vs masked vs imputed)
    - Per-variable performance comparison
  - Update to work with our evaluation results format

  **Must NOT do**:
  - Do NOT add 3D plots, attention visualizations, or complex statistical plots
  - Do NOT add interactivity — static plots only

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct copy with import adaptations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — depends on evaluation results
  - **Parallel Group**: Wave 3 (after Task 14)
  - **Blocks**: Task 17 (E2E test may verify plots)
  - **Blocked By**: Task 14 (evaluation results)

  **References**:

  **Pattern References**:
  - `{SPACEW}/src/spacew/visualization/plots.py` — Source file to copy

  **Acceptance Criteria**:

  ```
  QA Scenario: Visualization module imports correctly
    Tool: Bash (python -c)
    Steps:
      1. python -c "from pipeline.visualization.plots import *; print('Visualization imports OK')"
    Expected Result: Module imports without error
    Evidence: .sisyphus/evidence/task-15-import.txt

  QA Scenario: Generates skill score plot from results
    Tool: Bash (python -c)
    Preconditions: evaluation_results.json exists
    Steps:
      1. python -c "
         from pipeline.visualization.plots import plot_skill_scores
         import json
         results = json.load(open('evaluation_results.json'))
         plot_skill_scores(results, 'test_plot.png')
         print('Plot generated')
         "
    Expected Result: Plot file created
    Evidence: .sisyphus/evidence/task-15-plot.txt
  ```

  **Commit**: YES (groups with Commit C)

- [x] 16. Smoke Tests

  **What to do**:
  - Create `tests/test_tsrm_external.py` — smoke tests for TSRMImputationExternal
  - Create `tests/test_osse_dataset.py` — smoke tests for OSSEDataset
  - Tests should be minimal but cover critical functionality:
    - Instantiation with valid config
    - Forward pass produces correct output shape
    - NaN handling in masked data
    - Time embedding generation
    - DataLoader batching

  **Test cases**:

  `tests/test_tsrm_external.py`:
  ```python
  def test_instantiation():
      """TSRMImputationExternal instantiates with required config keys."""
      
  def test_forward_shape():
      """Forward pass produces correct output shape."""
      
  def test_nan_handling():
      """NaN in masked data is handled correctly (filled with 0 for forward pass)."""
      
  def test_loss_on_masked_only():
      """Loss is computed on masked positions only."""
  ```

  `tests/test_osse_dataset.py`:
  ```python
  def test_dataset_length():
      """Dataset returns correct number of samples."""
      
  def test_getitem_format():
      """__getitem__ returns 4-tuple with correct shapes."""
      
  def test_nan_preservation():
      """NaN in masked windows is preserved in output tensor."""
      
  def test_time_embeddings():
      """Time embeddings have correct dimensionality."""
      
  def test_dataloader_batching():
      """DataLoader produces correctly batched outputs."""
  ```

  **Must NOT do**:
  - Do NOT write comprehensive unit tests — smoke tests only
  - Do NOT test edge cases exhaustively
  - Do NOT add performance benchmarks

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple pytest test cases for critical functionality
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 17)
  - **Blocks**: None
  - **Blocked By**: Tasks 8, 9 (modules under test must exist)

  **References**:

  **Test References**:
  - `{SPACEW}/tests/test_data.py` — Smoke test patterns
  - `{SPACEW}/tests/test_masking.py` — Test structure reference

  **Acceptance Criteria**:

  ```
  QA Scenario: All smoke tests pass
    Tool: Bash
    Preconditions: tests/ directory with test files
    Steps:
      1. pytest tests/ -v --tb=short
    Expected Result: All tests pass (0 failures)
    Failure Indicators: Any test failure
    Evidence: .sisyphus/evidence/task-16-tests.txt
  ```

  **Commit**: YES (groups with Task 17 in Commit D)
  - Message: `test: add smoke tests and E2E integration test`

- [x] 17. E2E Integration Test

  **What to do**:
  - Create end-to-end test that validates the full pipeline
  - This is a MANUAL QA task executed by the agent, not an automated pytest

  **Test workflow**:
  1. Create minimal test config (1 satellite, 10 timesteps, 2 variables, 1 fold, 2 epochs)
  2. Run: `python scripts/train_osse.py --config test_config.yaml --fold 0`
  3. Verify: model checkpoint created
  4. Run: `python scripts/evaluate_osse.py --config test_config.yaml --fold 0`
  5. Verify: evaluation_results.json created with valid metrics
  6. Run: `pytest tests/`
  7. Verify: all tests pass

  **Must NOT do**:
  - Do NOT run full-scale training (use minimal test config)
  - Do NOT skip any verification step

  **Recommended Agent Profile**:
  - **Category**: `deep`
    - Reason: Requires understanding full pipeline and debugging any integration issues
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO — depends on all previous tasks
  - **Parallel Group**: Wave 4 (after Task 16)
  - **Blocks**: Final Verification Wave
  - **Blocked By**: Tasks 13, 14, 15, 16 (all pipeline components)

  **References**:
  - All previous task references

  **Acceptance Criteria**:

  ```
  QA Scenario: E2E training completes
    Tool: Bash
    Preconditions: All modules exist
    Steps:
      1. Create minimal test config in temp directory
      2. Run: python scripts/train_osse.py --config test_config.yaml --fold 0 --epochs 2
      3. Verify checkpoint exists
    Expected Result: Training completes, checkpoint saved
    Failure Indicators: Training crash, no checkpoint
    Evidence: .sisyphus/evidence/task-17-e2e-train.txt

  QA Scenario: E2E evaluation produces results
    Tool: Bash
    Preconditions: E2E training completed
    Steps:
      1. Run: python scripts/evaluate_osse.py --config test_config.yaml --fold 0
      2. Verify evaluation_results.json exists
      3. Verify metrics are reasonable (MSE > 0, MAE > 0)
    Expected Result: Evaluation results JSON with valid metrics
    Failure Indicators: No results file, invalid metrics
    Evidence: .sisyphus/evidence/task-17-e2e-eval.txt

  QA Scenario: All tests pass after E2E
    Tool: Bash
    Steps:
      1. pytest tests/ -v
    Expected Result: All tests pass
    Evidence: .sisyphus/evidence/task-17-all-tests.txt
  ```

  **Commit**: YES (groups with Commit D)

---

## Final Verification Wave

> 4 review agents run in PARALLEL. ALL must APPROVE. Rejection → fix → re-run.

- [ ] F1. **Plan Compliance Audit** — `oracle`
  Read the plan end-to-end. For each "Must Have": verify implementation exists (read file, run command). For each "Must NOT Have": search codebase for forbidden patterns — reject with file:line if found. Check evidence files exist in `.sisyphus/evidence/`. Compare deliverables against plan.
  Output: `Must Have [N/N] | Must NOT Have [N/N] | Tasks [N/N] | VERDICT: APPROVE/REJECT`

- [ ] F2. **Code Quality Review** — `unspecified-high`
  Run linter + `pytest tests/`. Review all new files for: `as any`/`@ts-ignore` equivalents, empty catches, print statements in library code, commented-out code, unused imports. Check for AI slop: excessive comments, over-abstraction, generic variable names.
  Output: `Tests [N pass/N fail] | Files [N clean/N issues] | VERDICT`

- [ ] F3. **Real Manual QA** — `unspecified-high`
  Start from clean state. Execute EVERY QA scenario from EVERY task — follow exact steps, capture evidence. Test cross-task integration. Test edge cases: empty data, single satellite, missing .env. Save to `.sisyphus/evidence/final-qa/`.
  Output: `Scenarios [N/N pass] | Integration [N/N] | Edge Cases [N tested] | VERDICT`

- [ ] F4. **Scope Fidelity Check** — `deep`
  For each task: read "What to do", read actual implementation. Verify 1:1 — everything in spec was built, nothing beyond spec was built. Check "Must NOT do" compliance. Flag any modifications to existing TSRM files. Detect unaccounted changes.
  Output: `Tasks [N/N compliant] | Contamination [CLEAN/N issues] | Unaccounted [CLEAN/N files] | VERDICT`

---

## Commit Strategy

| Commit | Tasks | Message | Pre-commit Check |
|--------|-------|---------|-----------------|
| A | 1-7 | `feat(pipeline): add foundation modules adapted from spacew` | `python -c "from pipeline.config import load_config"` |
| B | 8-12 | `feat(pipeline): add TSRM adaptation layer (external masking, OSSE dataset, HP grid)` | `python -c "from architecture.tsrm_external import TSRMImputationExternal"` |
| C | 13-15 | `feat(scripts): add training, evaluation, and visualization scripts` | `python scripts/train_osse.py --help` |
| D | 16-17 | `test: add smoke tests and E2E integration test` | `pytest tests/ -v` |

---

## Success Criteria

### Verification Commands
```bash
# Pipeline imports work
python -c "from pipeline.config import load_config; from pipeline.data.loader import OSSEDataLoader; from pipeline.data.preprocessor import Preprocessor; from pipeline.data.masking import apply_missing_pattern; from pipeline.data.dataset import OSSEDataset; from pipeline.evaluation.metrics import compute_metrics; from pipeline.evaluation.baselines import locf_impute, linear_impute; print('All imports OK')"

# TSRM external subclass works
python -c "from architecture.tsrm_external import TSRMImputationExternal; print('TSRMImputationExternal imported OK')"

# Training script has --help
python scripts/train_osse.py --help

# Tests pass
pytest tests/ -v

# Config loads
python -c "from pipeline.config import load_config; cfg = load_config('configs/osse_default.yaml'); print(f'Config loaded: {len(cfg)} keys')"
```

### Final Checklist
- [ ] All "Must Have" items present and functional
- [ ] All "Must NOT Have" items absent (no existing TSRM files modified)
- [ ] All 15 new Python files created in correct locations
- [ ] configs/osse_default.yaml valid and loadable
- [ ] Training completes for at least 1 fold without error
- [ ] Evaluation produces MSE/MAE/MSESS/MAESS metrics
- [ ] Hyperparameter search grid defined and runnable
- [ ] All smoke tests pass
