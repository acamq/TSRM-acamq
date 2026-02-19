# Learnings - tsrm-imputation-pipeline

## Task 7: Baselines Module

### Date
2026-02-19

### Findings

**Baselines Module Copy (Task 7)**
- Successfully copied `baselines.py` from spacew project to `pipeline/evaluation/baselines.py`
- Source file: `C:\Users\aic\liua\work\projects\spacew_root\github\spacew\src\spacew\evaluation\baselines.py`
- Destination: `pipeline/evaluation/baselines.py`

**Module Characteristics**
- Self-contained module with only numpy and pandas dependencies
- No spacew imports present (verified)
- LSP diagnostics: No errors in the new file
- All functions preserved as-is:
  - `locf_impute(data)` - Last Observation Carried Forward with backward-fill for leading NaN
  - `linear_interp_impute(data)` - Pandas interpolate(method='linear', limit_direction='both')
  - `get_available_baselines()` - Returns ['locf', 'linear']
  - `apply_baseline(data, method)` - Main entry point with method validation

**Edge Case Handling**
- All-NaN features: Left as NaN (excluded from scoring)
- Leading NaN: Uses backward-fill from first observed value
- Input format: [n_samples, n_steps, n_features] with NaN for missing
- Output format: Imputed array with NaN only where entire feature is missing

**Key Observations**
- No modifications needed to the source code
- Module is completely independent from spacew
- Ready to use for baseline comparisons in the evaluation pipeline

## 2026-02-19: Project Scaffolding

### Successfully Created Structure
- Created `pipeline/` package with subdirectories:
  - `pipeline/data/` - data handling module
  - `pipeline/evaluation/` - evaluation metrics module
  - `pipeline/tracking/` - experiment tracking module
  - `pipeline/visualization/` - visualization module
- Created `scripts/` directory for utility scripts
- Created `tests/` directory for unit tests
- All directories contain empty `__init__.py` files

### Environment Configuration
- Created `.env` template following spacew pattern with:
  - `DATA_DIR` placeholder for OSSE data path
  - `SCRATCH_DIR` placeholder for output directory

### Dependencies
- Added `pytest~=7.4.0` to requirements.txt for testing
- Added `python-dotenv~=1.0.0` to requirements.txt for environment variable loading
- Existing requirements preserved (matplotlib, numpy, pandas, scikit-learn, scipy, torch, wfdb, lightning, PyYAML, tensorboardx, mlflow, seaborn)

### Verification
- All directories created successfully
- All `__init__.py` files exist
- `.env` file created
- `.sisyphus/evidence/` directory already existed from previous setup

### Notes
- No Python logic files created yet (per requirements)
- Existing TSRM files in architecture/ were not modified
- Ready for next phase: implementing pipeline modules

## Task 5: Masking Module Copy
Date: 2026-02-19 08:20:58

### Learnings
- Masking module is pure numpy - direct copy required no changes
- All functions return Tuple[np.ndarray, np.ndarray, float] (masked_data, artificial_mask, realized_rate)
- artificial_mask is boolean: True where artificially masked
- Three patterns implemented:
  - apply_point_mcar: Random scattered missing (MCAR)
  - apply_subseq_missing: Contiguous segments per feature
  - apply_block_missing: Time intervals across all features
- apply_missing_pattern: Main entry point that delegates to specific pattern functions
- No spacew imports - module is self-contained

## Task 2: Config Module

### Date
2026-02-19

### Findings

**Config Module Implementation**
- Created `pipeline/config.py` with two main functions:
  - `load_config(yaml_path)` - loads YAML config with .env support
  - `build_tsrm_config(yaml_cfg) -> dict` - translates YAML to TSRM config dict

**TSRM Config Structure**
- The TSRM model expects a specific dictionary structure with 20 required keys
- `task="imputation"` and `phase="downstream"` are REQUIRED by Transformations.__init__ at model.py:224
- Missing these keys causes runtime errors when initializing the model

**Hardcoded Overrides**
- `pred_len: 0` - Imputation task, not forecasting
- `revin: False` - External normalization only (RevIN handled externally)
- `missing_ratio: 0.0` - External masking only (missing values masked externally)
- `embed: "timeF"` - Time-based feature encoding
- `freq: "t"` - Time frequency (minutes)

**Key Mappings from YAML to TSRM Config**
- `feature_dimension` → direct mapping
- `seq_len` → direct mapping
- `encoding_size`, `h`, `N` → model architecture params
- `conv_dims` → list of convolution dimensions
- `attention_func` → attention function type
- `batch_size`, `dropout` → training params
- `loss_function_imputation`, `loss_imputation_mode`, `loss_weight_alpha` → loss configuration
- `mask_size`, `mask_count` → masking parameters

**Verification**
- LSP diagnostics: No errors on config.py
- All required keys present in build_tsrm_config()
- Proper type hints with Dict from typing module
- Clear docstrings for both functions

**Design Decisions**
- Used simple .get() with sensible defaults for all config values
- No complex validation logic (as per requirements)
- Load function validates file existence before loading
- Clean separation between loading (load_config) and translation (build_tsrm_config)

## Task 6: Metrics Module

### Date
2026-02-19

### Findings

**Metrics Module Copy (Task 6)**
- Successfully copied `metrics.py` from spacew project to `pipeline/evaluation/metrics.py`
- Source file: `C:\Users\aic\liua\work\projects\spacew_root\github\spacew\src\spacew\evaluation\metrics.py`
- Destination: `pipeline/evaluation/metrics.py`

**Module Characteristics**
- Self-contained module with only numpy dependency and typing
- No spacew imports present (verified via grep)
- LSP diagnostics: No errors in the new file
- All required functions preserved:
  - `compute_mse(imputed, truth, mask)` - MSE on masked positions
  - `compute_mae(imputed, truth, mask)` - MAE on masked positions
  - `compute_metrics(imputed, truth, mask)` - returns dict with mse, mae
  - `compute_skill_score(model_error, baseline_error, eps)` - skill score calculation
  - `compute_skill_scores_vs_baselines(model_metrics, baseline_metrics, eps)` - aggregate skill scores
  - `compute_shared_eval_mask(art_mask, truth, *method_outputs)` - shared mask excluding NaN
  - `compute_metrics_per_variable(imputed, truth, mask, variable_names, baselines, eps)` - per-variable metrics

**Skill Score Properties**
- `eps=1e-10` used for division-by-zero handling
- Both perfect (model_error < eps and baseline_error < eps): returns 0.0
- Baseline perfect (baseline_error < eps, model_error >= eps): returns -inf
- Skill score > 0: model outperforms baseline
- Skill score = 1.0: perfect model (zero error)

**Shared Evaluation Mask**
- Critical for fair comparison: excludes NaN from ANY method
- If method produces NaN where others produce values, those points excluded for ALL
- Returns: (valid_mask, n_excluded, exclusion_reason)
- exclusion_reason lists which methods caused exclusions (e.g., 'saits:5', 'locf:2')

**Per-Variable Metrics**
- Supports per-feature MSE, MAE
- Optionally computes skill scores against multiple baselines
- Returns dict: {variable_name: {'mse': ..., 'mae': ..., 'msess_baseline1': ..., 'maess_baseline1': ...}}

**Additional Functions Preserved**
- `compute_metrics_per_group()`: Aggregate metrics per (block, satellite) then macro-average
- Ensures statistical independence by treating each group as independent sample unit
- Returns mse, mse_std, mae, mae_std, per_group details, n_groups

**Verification**
- No spacew imports found
- All core functionality preserved (eps=1e-10, mask convention)
- LSP diagnostics: clean

## Task 4: Preprocessor Module Copy

### Date
2026-02-19

### Findings

**Preprocessor Module Copy**
- Successfully copied `preprocessor.py` from spacew project to `pipeline/data/preprocessor.py`
- Source file: `C:\Users\aic\liua\work\projects\spacew_root\github\spacew\src\spacew\data\preprocessor.py`
- Destination: `pipeline/data/preprocessor.py`

**Module Characteristics**
- Self-contained module with only numpy and standard library dependencies (pickle, pathlib, typing)
- No spacew imports present (verified)
- LSP diagnostics: No errors in the new file
- All core functionality preserved as-is:
  - `NaNSafeScaler` - Per-feature mean/std with [1, 1, F] shaped statistics
  - `Preprocessor` - Log transform + scaler orchestration
  - `prepare_splits_block_level()` - Block-level splitting for CV
  - `create_windows_3d()` - Windowing with stride for 3D output

**NaNSafeScaler Implementation**
- Computes per-feature statistics over ALL satellites and timesteps (axis=(0,1))
- Produces [1, 1, F] shaped mean/std for broadcasting
- Preserves NaN positions during transform (no NaN imputation in scaler)
- Guards against zero std (constant features) by replacing with 1.0
- No time-varying statistics - per-feature only

**Preprocessor Physics-Informed Normalization**
- Density variables get log1p transform with epsilon=1e-30 before scaling
- Log transform is reversible with expm1 for inverse_transform
- Other variables get standard scaling only
- Density variable names configured via `normalization.density_vars` in config dict

**Block-Level Splitting**
- `prepare_splits_block_level()` splits by BLOCKS, not within blocks
- Returns lists of arrays for clean separation between train/val/test
- Each split block maintains shape [N_sat, block_len, F]
- Cumulative timestep bounds included for reference

**3D Windowing**
- `create_windows_3d()` creates windows within each block only
- Output is 3D [n_samples, window_size, n_features]
- CRITICAL: Each satellite-window is a SEPARATE sample (no satellite dimension)
- Stride parameter controls step between consecutive windows
- Warning printed if block too small for window_size

**Key Observations**
- No modifications needed to the source code
- Module is completely independent from spacew
- Ready to use for data preprocessing in the pipeline
- All critical constraints preserved (axis=(0,1), log_epsilon=1e-30, no per-satellite normalization)

## Task 3: OSSEDataLoader Module

### Date
2026-02-19

### Findings

**OSSEDataLoader Module Copy (Task 3)**
- Successfully copied `loader.py` from spacew project to `pipeline/data/loader.py`
- Source file: `C:\Users\aic\liua\work\projects\spacew_root\github\spacew\src\spacew\data\loader.py`
- Destination: `pipeline/data/loader.py`

**Module Characteristics**
- Self-contained class with only numpy, pickle, pathlib, and typing dependencies
- No spacew imports present (verified via grep)
- LSP diagnostics: No errors in the new file
- All methods preserved:
  - `__init__(data_dir, variables=None)` - Initialize loader with data directory and optional variable list
  - `load_pickle()` - Load GDC_synthetic_observations.pkl file
  - `load_csv(variable_name)` - Load individual CSV file for one variable (supports 3-line header)
  - `get_metadata()` - Load or create metadata dict with block_boundaries, num_satellites
  - `get_block_boundaries()` - Return [0, 60, 120, 180, 240]
  - `to_multivariate_array()` - Convert dict to [6, 240, 8] array shape
  - `validate_no_cross_block_windows(window_indices)` - Verify no window spans block boundaries

**Data Structure**
- DEFAULT_VARIABLES: 8 ionospheric/thermospheric variables
  1. GDC_TEMPERATURE
  2. GDC_TEMPERATURE_ION
  3. GDC_TEMPERATURE_ELEC
  4. GDC_VELOCITY_U
  5. GDC_VELOCITY_V
  6. GDC_DENSITY_ION_OP
  7. GDC_DENSITY_NEUTRAL_O
  8. GDC_DENSITY_NEUTRAL_O2
- Block boundaries: [0, 60, 120, 180, 240] - 4 disjoint 1-hour blocks
- 6 satellites, 240 timesteps, 8 features
- to_multivariate_array() returns shape (N_satellites, T_timesteps, F_features)
- Data discontinuity: 3-4 hour gaps between observation blocks

**Key Observations**
- No modifications needed to source code
- Module is completely independent from spacew
- Ready to use for data loading in the TSRM imputation pipeline
- All critical constraints preserved (8 variables, 6 satellites, 4 blocks, [0, 60, 120, 180, 240] boundaries)
- Docstring updated to reference TSRM imputation pipeline instead of spacew
- validate_no_cross_block_windows() helps ensure windows don't span across discontinuous blocks
