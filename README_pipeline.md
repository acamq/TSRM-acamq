# TSRM Imputation Pipeline for OSSE Data

This pipeline provides a complete workflow for training and evaluating the TSRM (Time Series Restoration Model) on GDC OSSE satellite data using an external masking approach.

## Project Overview

The TSRM imputation pipeline is designed for GDC OSSE satellite data, which includes observations from 6 satellites across 240 timesteps with 8 physical variables. The pipeline uses an external masking approach where synthetic gaps are generated and applied to ground truth data for training and validation. A 4-fold cross-validation strategy is employed to ensure robust evaluation across different time blocks.

## Installation

Follow these steps to set up the environment:

```bash
# Clone the repository
git clone <repository-url>
cd TSRM-acamq

# Create and activate conda environment
conda create -n tsrm python=3.10
conda activate tsrm

# Install dependencies
pip install -r requirements.txt
pip install "setuptools<80"  # Required for lightning compatibility
```

## Configuration

The pipeline uses a combination of environment variables and YAML configuration files.

### Environment Setup
Create a `.env` file in the root directory:
- `DATA_DIR`: Path to the raw OSSE data files.
- `SCRATCH_DIR`: Path for temporary files and experiment outputs.

### YAML Config
The main configuration file is located at `configs/osse_default.yaml`. It's structured into several key sections:
- **data**: OSSE-specific parameters (satellites, variables, window size).
- **tsrm**: Model architecture parameters (layers, heads, dimensions).
- **training**: Optimization settings (learning rate, batch size, epochs).
- **evaluation**: Metrics and baseline configurations.

## Training

You can train the model for a single fold or across all folds.

```bash
# Train a single fold (e.g., fold 0)
python scripts/train_osse.py --config configs/osse_default.yaml --fold 0

# Train all 4 folds sequentially
python scripts/train_osse.py --config configs/osse_default.yaml --all-folds

# Training with custom missing patterns and epochs
python scripts/train_osse.py --config configs/osse_default.yaml --fold 0 \
    --missing-pattern point --missing-rate 0.2 --epochs 50
```

## Evaluation

Evaluate a trained model's performance against baselines using the evaluation script.

```bash
python scripts/evaluate_osse.py --config configs/osse_default.yaml \
    --experiment-dir outputs/experiments/tsrm_osse_xxx \
    --fold 0
```

## Hyperparameter Search

The search script supports both quick trials and full sweeps.

```bash
# Quick search (approximately 32 configurations)
python scripts/search_osse.py --config configs/osse_default.yaml --quick

# Full hyperparameter sweep
python scripts/search_osse.py --config configs/osse_default.yaml
```

## Testing

Ensure the pipeline is functioning correctly by running the test suite.

```bash
# Run all tests
pytest tests/ -v

# Run specific test files
pytest tests/test_tsrm_external.py -v
```

## Data Flow

The pipeline follows a structured data processing sequence:
1. **Load OSSE data**: Reads data for 6 satellites, 240 timesteps, and 8 variables.
2. **Block Splitting**: Divides data into 4 folds using a leave-one-block-out CV strategy.
3. **Normalization**: Fits a `NaNSafeScaler` on training blocks only to prevent data leakage.
4. **Windowing**: Creates overlapping time windows for model input.
5. **Masking**: Applies synthetic missingness patterns (point, subseq, or block).
6. **Model Training**: Trains the `TSRMImputationExternal` model to reconstruct masked values.
7. **Evaluation**: Computes MSE, MAE, and skill scores compared to baselines like LOCF and linear interpolation.

## Key Design Decisions

- **External Normalization**: RevIN is disabled in favor of external normalization to maintain consistency across the pipeline.
- **External Masking**: `missing_ratio` is set to 0 in the core TSRM config because masking is handled externally by the pipeline's data loader.
- **Task Configuration**: The model is configured with `task="imputation"` and `phase="downstream"` as required by the TSRM architecture.
- **Leakage Prevention**: Scalers are fitted per-fold on training data only.

## File Structure

```text
pipeline/
├── config.py           # Configuration loading and validation
├── data/
│   ├── loader.py       # OSSEDataLoader for raw data access
│   ├── preprocessor.py # NaNSafeScaler and windowing logic
│   ├── masking.py      # Missing pattern generators (point, subseq, block)
│   └── dataset.py      # OSSEDataset and dataloader creation
├── evaluation/
│   ├── metrics.py      # MSE, MAE, and physical skill scores
│   └── baselines.py    # LOCF and linear interpolation implementations
├── tracking/
│   └── experiment.py   # MLflow or local experiment tracking
└── visualization/
    └── plots.py        # Training curves and imputation visualizations

scripts/
├── train_osse.py       # Main training entry point
├── evaluate_osse.py    # Model evaluation and comparison
└── search_osse.py      # Hyperparameter optimization

architecture/
└── tsrm_external.py    # TSRMImputationExternal implementation
```
