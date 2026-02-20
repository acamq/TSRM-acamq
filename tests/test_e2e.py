from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("lightning")


class TestE2EPipeline:
    @pytest.fixture
    def synthetic_config(self, tmp_path):
        config = {
            "data": {
                "variables": ["var1", "var2", "var3"],
                "block_boundaries": [0, 10, 20, 30, 40],
                "window_size": 5,
                "window_stride_train": 2,
                "window_stride_eval": 5,
            },
            "normalization": {
                "density_vars": [],
                "log_epsilon": 1e-30,
                "scaler_type": "nan_safe_standard",
                "fit_on_train_only": True,
            },
            "masking": {
                "patterns": ["point"],
                "missing_rates": [0.1],
                "augmentation_factor": 2,
                "seed": 42,
            },
            "tsrm": {
                "encoding_size": 16,
                "h": 2,
                "N": 1,
                "conv_dims": [[3, 1, 1]],
                "attention_func": "classic",
                "batch_size": 4,
                "dropout": 0.1,
                "learning_rate": 0.01,
                "epochs": 2,
                "patience": 1,
                "loss_function_imputation": "mse+mae",
                "loss_imputation_mode": "imputation",
                "loss_weight_alpha": 0.5,
                "embed": "timeF",
                "freq": "t",
                "mask_size": 2,
                "mask_count": 1,
            },
            "training": {
                "accelerator": "cpu",
                "precision": "32",
                "gradient_clip_val": 1.0,
                "num_workers": 0,
            },
            "evaluation": {
                "baselines": ["locf", "linear"],
                "skill_score_eps": 1e-10,
            },
            "paths": {
                "data_dir": str(tmp_path / "data"),
                "scratch_dir": str(tmp_path / "scratch"),
            },
        }

        Path(config["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
        Path(config["paths"]["scratch_dir"]).mkdir(parents=True, exist_ok=True)

        return config

    def test_config_builds_tsrm_dict(self, synthetic_config):
        from pipeline.config import build_tsrm_config

        tsrm_cfg = build_tsrm_config(synthetic_config)

        assert tsrm_cfg["feature_dimension"] == 3
        assert tsrm_cfg["seq_len"] == 5
        assert tsrm_cfg["pred_len"] == 0
        assert tsrm_cfg["revin"] is False
        assert tsrm_cfg["missing_ratio"] == 0.0
        assert tsrm_cfg["task"] == "imputation"
        assert tsrm_cfg["phase"] == "downstream"

    def test_model_instantiates(self, synthetic_config):
        from architecture.tsrm_external import TSRMImputationExternal
        from pipeline.config import build_tsrm_config

        tsrm_cfg = build_tsrm_config(synthetic_config)
        model = TSRMImputationExternal(tsrm_cfg)

        assert model is not None
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_dataset_works(self):
        from pipeline.data.dataset import OSSEDataset

        n_samples, window_size, n_features = 10, 5, 3
        masked = np.random.randn(n_samples, window_size, n_features).astype(np.float32)
        masked[:, 1:3, :] = np.nan
        original = np.random.randn(n_samples, window_size, n_features).astype(np.float32)

        ds = OSSEDataset(masked, original, None, freq="t")

        assert len(ds) == n_samples
        item = ds[0]
        assert len(item) == 4

    def test_masking_works(self):
        from pipeline.data.masking import apply_missing_pattern

        data = np.random.randn(10, 5, 3)
        masked, mask, rate = apply_missing_pattern(data.copy(), "point", 0.2, seed=42)

        assert masked.shape == data.shape
        assert mask.shape == data.shape
        assert 0 < rate < 0.5

    def test_metrics_compute(self):
        from pipeline.evaluation.metrics import compute_metrics

        truth = np.ones((10, 5, 3))
        imputed = truth + 0.1
        mask = np.zeros((10, 5, 3), dtype=bool)
        mask[:, :2, :] = True

        metrics = compute_metrics(imputed, truth, mask)

        assert "mse" in metrics
        assert "mae" in metrics
        assert metrics["mse"] > 0
        assert metrics["mae"] > 0

    def test_baselines_work(self):
        from pipeline.evaluation.baselines import linear_interp_impute, locf_impute

        data = np.random.randn(10, 5, 3)
        data[:, 1:3, :] = np.nan

        locf_result = locf_impute(data.copy())
        linear_result = linear_interp_impute(data.copy())

        assert not np.isnan(locf_result).any()
        assert not np.isnan(linear_result).any()

    def test_e2e_pipeline(self):
        torch = pytest.importorskip("torch")
        lightning = pytest.importorskip("lightning")

        from architecture.tsrm_external import TSRMImputationExternal
        from pipeline.data.dataset import OSSEDataset
        from pipeline.data.masking import apply_missing_pattern
        from pipeline.evaluation.metrics import compute_metrics

        rng = np.random.default_rng(123)
        n_samples = 12
        window_size = 5
        n_features = 3
        batch_size = 4

        model_cfg = {
            "feature_dimension": n_features,
            "seq_len": window_size,
            "pred_len": 0,
            "encoding_size": 16,
            "h": 2,
            "N": 1,
            "conv_dims": [[3, 1, 1]],
            "attention_func": "classic",
            "batch_size": batch_size,
            "dropout": 0.1,
            "revin": False,
            "missing_ratio": 0.0,
            "loss_function_imputation": "mse+mae",
            "loss_imputation_mode": "imputation",
            "loss_weight_alpha": 0.5,
            "embed": "timeF",
            "freq": "t",
            "mask_size": 2,
            "mask_count": 1,
            "task": "imputation",
            "phase": "downstream",
        }

        full_data = rng.normal(size=(n_samples, window_size, n_features)).astype(np.float32)
        train_original = full_data[:8]
        val_original = full_data[8:10]
        test_original = full_data[10:]

        train_masked, _, _ = apply_missing_pattern(train_original.copy(), "point", 0.2, seed=1)
        val_masked, _, _ = apply_missing_pattern(val_original.copy(), "point", 0.2, seed=2)
        test_masked, test_mask, _ = apply_missing_pattern(test_original.copy(), "point", 0.2, seed=3)

        train_ds = OSSEDataset(train_masked, train_original, timestamps=None, freq="t")
        val_ds = OSSEDataset(val_masked, val_original, timestamps=None, freq="t")
        test_ds = OSSEDataset(test_masked, test_original, timestamps=None, freq="t")

        train_dl = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_dl = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_dl = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        model = TSRMImputationExternal(model_cfg)
        model.lr = 1e-3

        trainer = lightning.Trainer(
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
            limit_train_batches=1,
            limit_val_batches=1,
        )
        trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        masked_batch, truth_batch, time_marks_x, time_marks_y = next(iter(test_dl))
        imputed = model.impute(masked_batch, truth_batch, time_marks_x, time_marks_y)

        metrics = compute_metrics(
            imputed.detach().cpu().numpy(),
            truth_batch.detach().cpu().numpy(),
            test_mask,
        )

        assert imputed.shape == truth_batch.shape
        assert np.isfinite(metrics["mse"])
        assert np.isfinite(metrics["mae"])
