from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from all_tmd.config import PipelineConfig
from all_tmd.mlflow_utils import dataset_digest
from all_tmd.models import fit_with_optional_sample_weight, model_from_params, suggest_model_params
from all_tmd.study import canonical_digest
from all_tmd.study_runner import read_feature_dataset
from all_tmd.windowing import feature_output_dir


LOCKED_PACKAGES = (
    "numpy",
    "pandas",
    "scikit-learn",
    "xgboost",
    "optuna",
    "mlflow",
)


def dependency_versions() -> dict[str, str]:
    return {name: importlib.metadata.version(name) for name in LOCKED_PACKAGES}


def tune_source(
    config: PipelineConfig,
    *,
    output_root: str | Path,
    trials: int = 45,
    seed: int = 42,
) -> dict[str, Any]:
    if trials != 45 or seed != 42:
        raise ValueError("the controlled study requires 45 trials and seed 42")
    source = read_feature_dataset(feature_output_dir(config, "nor-tmd"))
    if set(source["domain"].astype(str)) != {"nor-tmd"}:
        raise ValueError("source tuning accepts NOR-TMD features only")
    if source["participant_id"].nunique() < 5:
        raise ValueError("NOR-TMD tuning requires at least five participants")
    feature_names = config.trial.feature_names
    x = source[feature_names].to_numpy(dtype=np.float32)
    y = source["label"].to_numpy(dtype=np.int64)
    groups = source["participant_id"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, ("xgboost",))
        truth: list[np.ndarray] = []
        predicted: list[np.ndarray] = []
        for train_positions, valid_positions in folds:
            model = model_from_params(
                params,
                random_seed=seed,
                n_jobs=config.training.n_jobs,
                xgboost_device="cpu",
            )
            weights = compute_sample_weight("balanced", y[train_positions])
            fit_with_optional_sample_weight(
                model, x[train_positions], y[train_positions], weights
            )
            truth.append(y[valid_positions])
            predicted.append(model.predict(x[valid_positions]).astype(int))
        return float(
            f1_score(
                np.concatenate(truth),
                np.concatenate(predicted),
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        )

    mlflow = None
    if config.mlflow.enabled:
        import mlflow as mlflow_module

        mlflow = mlflow_module
    if config.mlflow.enabled and config.mlflow.tracking_uri:
        assert mlflow is not None
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    if config.mlflow.enabled:
        assert mlflow is not None
        mlflow.set_experiment(config.mlflow.experiment_name)
    context = (
        mlflow.start_run(run_name="nor-only-xgboost-model-selection")
        if config.mlflow.enabled
        else nullcontext(None)
    )
    with context as run:
        run_id = run.info.run_id if run is not None else uuid4().hex
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
        )
        study.optimize(objective, n_trials=trials)
        completed = [
            trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
        ]
        if not completed:
            raise RuntimeError("NOR-TMD tuning produced no completed trial")
        best = max(completed, key=lambda trial: (float(trial.value), -trial.number))
        result = {
            "schema_version": 1,
            "run_id": run_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "tuning_dataset": "nor-tmd",
            "sydney_rows_used": 0,
            "source_feature_digest": dataset_digest(source, feature_names),
            "feature_names": feature_names,
            "fold_method": "five-fold-participant-grouped",
            "trial_count": trials,
            "seed": seed,
            "selection_metric": "macro_f1",
            "best_trial_number": best.number,
            "best_cross_validation_macro_f1": float(best.value),
            "best_params": best.user_attrs["model_params"],
            "dependency_versions": dependency_versions(),
            "code_commit": os.environ.get("ALL_TMD_GIT_COMMIT", "unknown"),
        }
        output_dir = Path(output_root) / "tuning" / run_id
        output_dir.mkdir(parents=True, exist_ok=False)
        result_path = output_dir / "tuning-result.json"
        trials_path = output_dir / "optuna-trials.csv"
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        study.trials_dataframe().to_csv(trials_path, index=False)
        if config.mlflow.enabled:
            assert mlflow is not None
            mlflow.log_params(
                {
                    "study_phase": "nor-only-model-selection",
                    "sydney_rows_used": 0,
                    "trials": trials,
                    "seed": seed,
                    "source_feature_digest": result["source_feature_digest"],
                }
            )
            mlflow.log_metric(
                "best_cross_validation_macro_f1",
                result["best_cross_validation_macro_f1"],
            )
            mlflow.set_tags({"study_id": "sydney-transfer-v2", "run_level": "tuning"})
            mlflow.log_artifacts(str(output_dir))
        return result


def promote_model(
    run_id: str,
    *,
    output_root: str | Path,
    lock_path: str | Path,
) -> dict[str, Any]:
    result_path = Path(output_root) / "tuning" / run_id / "tuning-result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"tuning run was not found: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("tuning_dataset") != "nor-tmd" or result.get("sydney_rows_used") != 0:
        raise ValueError("only a Sydney-free NOR-TMD tuning run can be promoted")
    lock = {
        "schema_version": 1,
        "model_family": "xgboost",
        "params": result["best_params"],
        "feature_names": result["feature_names"],
        "tuning_dataset": "nor-tmd",
        "sydney_rows_used": 0,
        "source_feature_digest": result["source_feature_digest"],
        "selection_metric": "macro_f1",
        "cross_validation_score": result["best_cross_validation_macro_f1"],
        "tuning_run_id": result["run_id"],
        "dependency_versions": result["dependency_versions"],
        "code_commit": result.get("code_commit", "unknown"),
    }
    lock["lock_digest"] = canonical_digest(lock)
    destination = Path(lock_path)
    content = json.dumps(lock, indent=2) + "\n"
    if destination.exists():
        if destination.read_text(encoding="utf-8") != content:
            raise ValueError(f"existing immutable model lock differs: {destination}")
        return lock
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)
    return lock
