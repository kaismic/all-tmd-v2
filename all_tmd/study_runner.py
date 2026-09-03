from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import os
import importlib.metadata
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.utils.class_weight import compute_sample_weight

from all_tmd.config import PipelineConfig
from all_tmd.mlflow_utils import dataset_digest
from all_tmd.models import fit_with_optional_sample_weight, model_from_params
from all_tmd.study import (
    ParentRunSpec,
    StudyPlan,
    calibration_session_ids,
    canonical_digest,
    create_lopo_manifest,
    verify_snapshot,
)
from all_tmd.windowing import feature_output_dir


def read_feature_dataset(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"no feature parts found beneath {path}")
    return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)


def load_model_lock(
    path: str | Path,
    feature_names: list[str],
    *,
    source_feature_digest: str | None = None,
    validate_dependencies: bool = False,
) -> dict[str, Any]:
    lock_path = Path(path)
    if not lock_path.exists():
        raise FileNotFoundError(
            f"model lock is required before Sydney evaluation: {lock_path}"
        )
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("model_family") != "xgboost":
        raise ValueError("model lock must be a schema-v1 XGBoost lock")
    if lock.get("feature_names") != feature_names:
        raise ValueError("model lock feature contract differs from the study")
    if lock.get("tuning_dataset") != "nor-tmd" or lock.get("sydney_rows_used") != 0:
        raise ValueError("model lock was not selected using NOR-TMD only")
    params = lock.get("params")
    if not isinstance(params, dict) or params.get("family") != "xgboost":
        raise ValueError("model lock does not contain fixed XGBoost parameters")
    expected = lock.get("lock_digest")
    without_digest = {key: value for key, value in lock.items() if key != "lock_digest"}
    if expected != canonical_digest(without_digest):
        raise ValueError("model lock digest is invalid")
    if (
        source_feature_digest is not None
        and lock.get("source_feature_digest") != source_feature_digest
    ):
        raise ValueError("model lock NOR input digest differs from the study input")
    if validate_dependencies:
        locked_versions = lock.get("dependency_versions", {})
        current_versions = {
            package: importlib.metadata.version(package)
            for package in locked_versions
        }
        if current_versions != locked_versions:
            raise ValueError(
                "runtime dependency versions differ from the promoted model lock"
            )
    return lock


def run_study(
    config: PipelineConfig,
    plan: StudyPlan,
    *,
    output_root: str | Path,
    execution_backend: str = "local",
    spec_limit: int | None = None,
) -> list[dict[str, Any]]:
    snapshot = verify_snapshot(plan.snapshot_path)
    source = read_feature_dataset(feature_output_dir(config, "nor-tmd"))
    collector = read_feature_dataset(feature_output_dir(config, "collector"))
    eligible = set(snapshot["eligible_session_ids"])
    collector = collector.loc[collector["session_id"].astype(str).isin(eligible)].copy()
    unknown = set(collector["session_id"].astype(str)) - eligible
    if unknown:
        raise ValueError("collector features contain sessions outside the frozen snapshot")
    if collector.empty:
        raise ValueError("no frozen Sydney sessions survived feature extraction")

    feature_names = config.trial.feature_names
    for frame, name in ((source, "NOR-TMD"), (collector, "Sydney")):
        missing = sorted(set(feature_names) - set(frame.columns))
        if missing:
            raise ValueError(f"{name} features are missing: {', '.join(missing)}")
    source_digest = dataset_digest(source, config.trial.feature_names)
    model_lock = load_model_lock(
        plan.model_lock_path,
        feature_names,
        source_feature_digest=source_digest,
        validate_dependencies=True,
    )
    generated_lopo = create_lopo_manifest(collector)
    if not plan.lopo_manifest_path.exists():
        raise FileNotFoundError(
            f"create and freeze the LOPO manifest before evaluation: {plan.lopo_manifest_path}"
        )
    lopo = json.loads(plan.lopo_manifest_path.read_text(encoding="utf-8"))
    if lopo.get("manifest_digest") != generated_lopo["manifest_digest"]:
        raise ValueError("frozen LOPO manifest differs from the current Sydney features")
    results = []
    specs = plan.parent_specs()
    if spec_limit is not None:
        specs = specs[:spec_limit]
    for spec in specs:
        results.append(
            _run_parent(
                config,
                plan,
                spec,
                source,
                collector,
                lopo,
                model_lock,
                Path(output_root),
                execution_backend,
            )
        )
    return results


def _run_parent(
    config: PipelineConfig,
    plan: StudyPlan,
    spec: ParentRunSpec,
    source: pd.DataFrame,
    collector: pd.DataFrame,
    lopo: dict[str, Any],
    model_lock: dict[str, Any],
    output_root: Path,
    execution_backend: str,
) -> dict[str, Any]:
    tracking_enabled = config.mlflow.enabled
    mlflow = None
    if tracking_enabled:
        import mlflow as mlflow_module

        mlflow = mlflow_module
    if tracking_enabled and config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    if tracking_enabled:
        mlflow.set_experiment(config.mlflow.experiment_name)
    parent_context = (
        mlflow.start_run(run_name=spec.key)
        if tracking_enabled
        else nullcontext(None)
    )
    with parent_context as parent:
        run_id = parent.info.run_id if parent is not None else uuid4().hex
        output_dir = output_root / run_id
        if output_dir.exists():
            raise FileExistsError(f"result directory already exists: {output_dir}")
        output_dir.mkdir(parents=True)
        base_params = {
            "study_id": plan.study_id,
            "condition": spec.condition,
            "sydney_fraction": spec.sydney_fraction,
            "seed": spec.seed,
            "execution_backend": execution_backend,
            "evaluation_manifest_digest": lopo["manifest_digest"],
            "evaluation_session_digest": lopo["evaluation_digest"],
            "model_lock_digest": model_lock["lock_digest"],
            "code_commit": os.environ.get("ALL_TMD_GIT_COMMIT", "unknown"),
            "source_feature_digest": dataset_digest(source, config.trial.feature_names),
            "collector_feature_digest": dataset_digest(collector, config.trial.feature_names),
        }
        base_params["split_digest"] = canonical_digest(
            {
                "collector_feature_digest": base_params["collector_feature_digest"],
                "lopo_manifest_digest": lopo["manifest_digest"],
                "condition": spec.condition,
                "sydney_fraction": spec.sydney_fraction,
                "seed": spec.seed,
            }
        )
        if tracking_enabled:
            assert mlflow is not None
            mlflow.log_params(base_params)
            mlflow.set_tags(
                {
                    "study_id": plan.study_id,
                    "execution_backend": execution_backend,
                    "run_level": "parent",
                }
            )

        prediction_frames: list[pd.DataFrame] = []
        fold_reports: list[dict[str, Any]] = []
        shared_nor_model = None
        for fold_number, fold in enumerate(lopo["folds"]):
            selected_sessions: list[str] = []
            selection: dict[str, Any] = {"requested_fraction": 0.0}
            if spec.condition != "nor_only":
                selected_sessions, selection = calibration_session_ids(
                    collector, fold, spec.sydney_fraction, spec.seed
                )
            train = _training_frame(spec.condition, source, collector, selected_sessions)
            test = collector.loc[
                collector["session_id"].astype(str).isin(fold["test_session_ids"])
            ].copy()
            train_participants = set(
                train.loc[train["domain"].astype(str) == "collector", "participant_id"].astype(str)
            )
            held_out = str(fold["held_out_participant_id"])
            if held_out in train_participants:
                raise AssertionError("held-out Sydney participant leaked into training")
            if spec.condition == "nor_only" and shared_nor_model is not None:
                model = shared_nor_model
            else:
                model = _fit_fixed_model(config, model_lock["params"], train, spec.seed)
                if spec.condition == "nor_only":
                    shared_nor_model = model
            predictions = _prediction_frame(model, test, config.trial.feature_names)
            prediction_frames.append(predictions)
            metrics = classification_metrics(predictions, plan.labels)
            fold_report = {
                "fold": fold_number,
                "held_out_participant_id": held_out,
                "test_session_digest": fold["test_digest"],
                "training_session_digest": canonical_digest(selected_sessions),
                "training_rows": len(train),
                "source_training_rows": int(
                    (train["domain"].astype(str) == "nor-tmd").sum()
                ),
                "sydney_training_rows": int(
                    (train["domain"].astype(str) == "collector").sum()
                ),
                "test_rows": len(test),
                "train_frame_digest": dataset_digest(
                    train, config.trial.feature_names
                ),
                "test_frame_digest": dataset_digest(
                    test, config.trial.feature_names
                ),
                "selection": selection,
                "metrics": metrics,
            }
            fold_report["execution_identity"] = canonical_digest(
                {
                    "model_lock_digest": model_lock["lock_digest"],
                    "code_commit": base_params["code_commit"],
                    "condition": spec.condition,
                    "sydney_fraction": spec.sydney_fraction,
                    "seed": spec.seed,
                    "fold": fold_number,
                    "mlflow_run_id": run_id,
                }
            )
            fold_reports.append(fold_report)
            fold_artifact_dir = output_dir / "folds" / held_out
            fold_artifact_dir.mkdir(parents=True)
            predictions.to_parquet(
                fold_artifact_dir / "predictions.parquet", index=False
            )
            (fold_artifact_dir / "metrics.json").write_text(
                json.dumps(fold_report, indent=2) + "\n", encoding="utf-8"
            )
            if tracking_enabled:
                assert mlflow is not None
                with mlflow.start_run(
                    run_name=f"{spec.key}-{held_out}", nested=True
                ):
                    mlflow.log_params(
                        {
                            **base_params,
                            "held_out_participant_id": held_out,
                            "test_session_digest": fold["test_digest"],
                            "training_session_digest": canonical_digest(selected_sessions),
                            "training_rows": len(train),
                            "test_rows": len(test),
                        }
                    )
                    mlflow.set_tags({"study_id": plan.study_id, "run_level": "fold"})
                    _log_dataset_input(
                        mlflow,
                        train,
                        config.trial.feature_names,
                        name=f"{spec.key}-{held_out}-training",
                        context="training",
                    )
                    _log_dataset_input(
                        mlflow,
                        test,
                        config.trial.feature_names,
                        name=f"{spec.key}-{held_out}-evaluation",
                        context="evaluation",
                    )
                    _log_scalar_metrics(mlflow, metrics)
                    mlflow.log_artifacts(str(fold_artifact_dir))

        predictions = pd.concat(prediction_frames, ignore_index=True)
        if sorted(predictions["session_id"].astype(str).unique()) != lopo["evaluation_session_ids"]:
            raise AssertionError("parent run did not evaluate the frozen LOPO session union")
        pooled = classification_metrics(predictions, plan.labels)
        report = {
            "schema_version": 1,
            "run_id": run_id,
            "run_name": spec.key,
            **base_params,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "feature_names": config.trial.feature_names,
            "model_params": model_lock["params"],
            "folds": fold_reports,
            "metrics": pooled,
        }
        metrics_path = output_dir / "metrics.json"
        predictions_path = output_dir / "predictions.parquet"
        lopo_path = output_dir / "lopo-manifest.json"
        metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        predictions.to_parquet(predictions_path, index=False)
        lopo_path.write_text(json.dumps(lopo, indent=2) + "\n", encoding="utf-8")
        if tracking_enabled:
            assert mlflow is not None
            _log_scalar_metrics(mlflow, pooled)
            mlflow.log_artifacts(str(output_dir))
        return report


def _training_frame(
    condition: str,
    source: pd.DataFrame,
    collector: pd.DataFrame,
    selected_sessions: list[str],
) -> pd.DataFrame:
    selected = collector.loc[
        collector["session_id"].astype(str).isin(selected_sessions)
    ]
    if condition == "nor_only":
        result = source
    elif condition == "sydney_only":
        result = selected
    elif condition == "nor_plus_sydney":
        result = pd.concat([source, selected], ignore_index=True)
    else:
        raise ValueError(f"unsupported condition: {condition}")
    if result.empty or set(result["label"].astype(int)) != {0, 1, 2}:
        raise ValueError(f"training condition {condition} does not contain all labels")
    return result.reset_index(drop=True)


def _fit_fixed_model(
    config: PipelineConfig,
    params: dict[str, Any],
    train: pd.DataFrame,
    seed: int,
):
    x = train[config.trial.feature_names].to_numpy(dtype=np.float32)
    y = train["label"].to_numpy(dtype=np.int64)
    weights = compute_sample_weight("balanced", y).astype(np.float64)
    model = model_from_params(
        params,
        random_seed=seed,
        n_jobs=config.training.n_jobs,
        xgboost_device="cpu",
    )
    return fit_with_optional_sample_weight(model, x, y, weights)


def _prediction_frame(model, test: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    x = test[feature_names].to_numpy(dtype=np.float32)
    predicted = model.predict(x).astype(int)
    result = test[
        ["participant_id", "session_id", "vehicle_type", "label", "window_start_ms", "window_end_ms"]
    ].copy()
    result["prediction"] = predicted
    probabilities = model.predict_proba(x)
    for label in range(probabilities.shape[1]):
        result[f"probability_{label}"] = probabilities[:, label]
    return result


def classification_metrics(predictions: pd.DataFrame, labels: dict[str, int]) -> dict[str, Any]:
    truth = predictions["label"].to_numpy(dtype=int)
    predicted = predictions["prediction"].to_numpy(dtype=int)
    values = list(labels.values())
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=values, zero_division=0
    )
    return {
        "rows": len(predictions),
        "sessions": int(predictions["session_id"].nunique()),
        "participants": int(predictions["participant_id"].nunique()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(truth, predicted, labels=values).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(labels)
        },
    }


def _log_scalar_metrics(mlflow: Any, metrics: dict[str, Any]) -> None:
    values = {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
    }
    for name, detail in metrics["per_class"].items():
        values[f"{name}.precision"] = detail["precision"]
        values[f"{name}.recall"] = detail["recall"]
        values[f"{name}.f1"] = detail["f1"]
    mlflow.log_metrics(values)


def _log_dataset_input(
    mlflow: Any,
    frame: pd.DataFrame,
    feature_names: list[str],
    *,
    name: str,
    context: str,
) -> None:
    columns = [
        "domain",
        "group_id",
        "label",
        "session_id",
        "window_start_ms",
        "window_end_ms",
        *feature_names,
    ]
    selected = frame.loc[:, columns].reset_index(drop=True)
    dataset = mlflow.data.from_pandas(
        selected,
        name=name,
        digest=dataset_digest(selected, feature_names),
    )
    mlflow.log_input(dataset, context=context)
