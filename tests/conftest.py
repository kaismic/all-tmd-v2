from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from all_tmd.config import PipelineConfig


@pytest.fixture
def config_factory(tmp_path):
    def factory(
        *,
        train_dataset: str = "us-tmd",
        sensors: dict[str, list[str]] | None = None,
        collector_minimum_sampling_rate: dict[str, float] | None = None,
        minimum_trip_seconds: int = 0,
        maximum_trip_seconds: int | None = 28_800,
        collector_max_sample_interval_ms: int | None = None,
        generic_minimum_sampling_rate: dict[str, float] | None = None,
        generic_maximum_sample_interval_ms: int | None = None,
        calibration_fraction: float | dict[str, float] = 0.5,
        evaluation_strategy: str = "session_holdout",
        weighting_strategy: str = "class_balanced",
        collector_domain_weight: float = 2.0,
        duration_balancing: str = "none",
        participant_inner_folds: int = 5,
        bootstrap_iterations: int = 0,
        selection_metric: str = "macro_f1",
        window_seconds: int = 1,
        step_seconds: int = 1,
        context_windows_seconds: list[int] | None = None,
        mlflow_enabled: bool = False,
        mlflow_tracking_uri: str | None = None,
        mlflow_artifact_location: str | None = None,
        run_name: str | None = None,
    ) -> PipelineConfig:
        sensors = sensors or {"accelerometer": ["mean"]}
        collector_minimum_sampling_rate = collector_minimum_sampling_rate or {
            sensor: 1 for sensor in sensors
        }
        data_root = tmp_path / "data"
        config_data = {
            "schema_version": 1,
            "sources": {
                "us-tmd": {
                    "input_dir": str(data_root / "us"),
                    "include_globs": ["**/*.csv"],
                    "chunk_rows": 100,
                },
                "nor-tmd": {
                    "input_dir": str(data_root / "nor"),
                    "include_globs": ["**/*.csv"],
                    "chunk_rows": 100,
                },
                "collector": {
                    "input_dir": str(data_root / "collector"),
                    "include_globs": ["**/*.json", "**/*.json.gz"],
                },
            },
            "dataset": {
                "work_dir": str(data_root / "work"),
                "minimum_trip_seconds": minimum_trip_seconds,
                "maximum_trip_seconds": maximum_trip_seconds,
                "collector_max_sample_interval_ms": (
                    collector_max_sample_interval_ms
                ),
            },
            "collector_minimum_sampling_rate": collector_minimum_sampling_rate,
            "training": {
                "n_jobs": 1,
                "timeout_seconds": None,
                "xgboost_device": "cpu",
            },
            "mlflow": {
                "enabled": mlflow_enabled,
                "experiment_name": "test",
                "tracking_uri": mlflow_tracking_uri,
                "artifact_location": mlflow_artifact_location,
            },
        }
        if generic_minimum_sampling_rate is not None:
            config_data["minimum_sampling_rate"] = generic_minimum_sampling_rate
        if generic_maximum_sample_interval_ms is not None:
            config_data["dataset"]["maximum_sample_interval_ms"] = (
                generic_maximum_sample_interval_ms
            )
        trial = {
            "train_dataset": train_dataset,
            "labels": {"bus": 0, "car": 1, "train": 2},
            "features": {
                "default_window_seconds": window_seconds,
                "default_step_seconds": step_seconds,
                "sensors": sensors,
            },
            "training": {
                "random_seed": 42,
                "optuna_trials": 1,
                "model_families": ["random_forest"],
                "calibration_fraction": calibration_fraction,
                "evaluation_strategy": evaluation_strategy,
                "weighting_strategy": weighting_strategy,
                "collector_domain_weight": collector_domain_weight,
                "duration_balancing": duration_balancing,
                "participant_inner_folds": participant_inner_folds,
                "bootstrap_iterations": bootstrap_iterations,
                "selection_metric": selection_metric,
            },
        }
        if run_name is not None:
            trial["run_name"] = run_name
        if context_windows_seconds is not None:
            trial["features"]["context_windows_seconds"] = context_windows_seconds
        config_path = tmp_path / f"{train_dataset}.yaml"
        trials_path = tmp_path / f"{train_dataset}.json"
        config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
        trials_path.write_text(json.dumps([trial]), encoding="utf-8")
        return PipelineConfig.from_files(config_path, trials_path)

    return factory
