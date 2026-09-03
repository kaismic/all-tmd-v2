from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from all_tmd.balancing import select_training_indices, training_sample_weights
from all_tmd.config import PipelineConfig
from all_tmd.windowing import feature_output_dir
from all_tmd.mlflow_utils import (
    log_artifact,
    log_confusion_matrix,
    log_metrics,
    start_run,
)
from all_tmd.models import (
    fit_with_optional_sample_weight,
    model_from_params,
    suggest_model_params,
)
from all_tmd.progress import progress
from all_tmd.splits import create_splits, write_splits


def train(config: PipelineConfig) -> dict[str, Any]:
    run_dir = config.run_dir()
    source_name = config.trial.train_dataset
    source_frame = _read_feature_dataset(feature_output_dir(config, source_name))
    collector_frame = _read_feature_dataset(feature_output_dir(config, "collector"))
    frame = pd.concat([source_frame, collector_frame], ignore_index=True)
    feature_names = config.trial.feature_names
    missing = sorted(set(feature_names) - set(frame.columns))
    if missing:
        raise ValueError("Feature dataset is missing columns: " + ", ".join(missing))
    manifest = create_splits(frame, config)
    split_path = write_splits(manifest, config)

    report_dir = config.report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    trial_snapshot_path = report_dir / "trial.json"
    trial_snapshot_path.write_text(
        json.dumps(config.trial.raw, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    x = frame[feature_names].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype=np.int64)

    with start_run(config, frame, manifest):
        if config.trial.training.evaluation_strategy == "participant_nested_cv":
            metrics, model, studies = _train_participant_nested(
                config, frame, x, y, manifest
            )
        else:
            metrics, model, studies = _train_session_holdout(
                config, frame, x, y, manifest
            )

        metrics.update(
            {
                "config_hash": config.config_hash,
                "trial_hash": config.trial_hash,
                "trial_index": config.trial_index,
                "run_name": config.trial.run_name,
                "train_dataset": source_name,
                "feature_names": feature_names,
                "rows": int(len(frame)),
                "source_rows": int(len(manifest["source_indices"])),
                "collector_rows": int(
                    (frame["domain"].astype(str) == "collector").sum()
                ),
                "groups": int(frame["group_id"].nunique()),
            }
        )
        metrics_path = report_dir / "metrics.json"
        trials_path = report_dir / "optuna-trials.csv"
        nested_trials_path = report_dir / "nested-optuna-trials.csv"
        model_path = report_dir / "model.joblib"
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        studies["final"].trials_dataframe().to_csv(trials_path, index=False)
        outer_studies = [
            study.trials_dataframe().assign(outer_participant_id=participant)
            for participant, study in studies.items()
            if participant != "final"
        ]
        if outer_studies:
            pd.concat(outer_studies, ignore_index=True).to_csv(
                nested_trials_path, index=False
            )
        joblib.dump(model, model_path)
        if config.mlflow.enabled:
            holdout_metrics = metrics["collector_holdout"]
            log_metrics(holdout_metrics, prefix="collector_holdout.")
            log_metrics(
                {"best_cross_validation_score": metrics["best_cross_validation_score"]}
            )
            _log_holdout_confusion_matrices(holdout_metrics, config)
            artifacts = [
                metrics_path,
                trials_path,
                model_path,
                split_path,
                trial_snapshot_path,
            ]
            if nested_trials_path.exists():
                artifacts.append(nested_trials_path)
            for artifact in artifacts:
                log_artifact(artifact)
    progress(f"Training complete: dataset={source_name}, reports={report_dir}")
    return metrics


def _train_session_holdout(
    config: PipelineConfig,
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, optuna.Study]]:
    source_indices = manifest["source_indices"]
    study = _optimize(
        config,
        frame,
        x,
        y,
        source_indices,
        manifest["collector_cv_folds"],
        label="session-holdout",
        apply_collector_domain_weight=(
            config.trial.training.weighting_strategy == "hierarchical"
        ),
    )
    best_params = _best_params(study)
    final_indices, selection = select_training_indices(
        frame,
        source_indices + manifest["collector_calibration_indices"],
        strategy=config.trial.training.duration_balancing,
        random_seed=config.trial.training.random_seed,
    )
    model = _fit_model(config, frame, x, y, best_params, final_indices)
    calibration_metrics = _evaluate_model(
        model,
        x,
        y,
        frame,
        manifest["collector_calibration_indices"],
        config,
    )
    holdout_metrics = _evaluate_model(
        model,
        x,
        y,
        frame,
        manifest["collector_holdout_indices"],
        config,
    )
    return (
        {
            "evaluation_strategy": "session_holdout",
            "best_cross_validation_score": float(study.best_value),
            **(
                {"best_cross_validation_macro_f1": float(study.best_value)}
                if config.trial.training.selection_metric == "macro_f1"
                else {}
            ),
            "selection_metric": config.trial.training.selection_metric,
            "cross_validation": {
                "method": "grouped_out_of_fold",
                "folds": int(len(manifest["collector_cv_folds"])),
                "rows": int(len(manifest["collector_calibration_indices"])),
                "score": float(study.best_value),
            },
            "best_params": best_params,
            "training_selection": selection,
            "collector_calibration": calibration_metrics,
            "collector_holdout": holdout_metrics,
        },
        model,
        {"final": study},
    )


def _train_participant_nested(
    config: PipelineConfig,
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    manifest: dict[str, Any],
) -> tuple[dict[str, Any], Any, dict[str, optuna.Study]]:
    source_indices = manifest["source_indices"]
    configured_labels = _label_values(config)
    collector_indices = manifest["collector_evaluation_indices"]
    prediction_by_index: dict[int, int] = {}
    probability_by_index: dict[int, np.ndarray] = {}
    outer_reports: list[dict[str, Any]] = []
    studies: dict[str, optuna.Study] = {}

    for outer_number, fold in enumerate(manifest["participant_outer_folds"]):
        participant = str(fold["held_out_participant_id"])
        progress(
            "Nested participant fold starting: "
            f"{outer_number + 1}/{len(manifest['participant_outer_folds'])}, "
            f"held_out={participant}"
        )
        study = _optimize(
            config,
            frame,
            x,
            y,
            source_indices,
            fold["inner_folds"],
            label=f"outer-{participant}",
            apply_collector_domain_weight=True,
        )
        studies[participant] = study
        train_indices, selection = select_training_indices(
            frame,
            source_indices + fold["train_indices"],
            strategy=config.trial.training.duration_balancing,
            random_seed=config.trial.training.random_seed + outer_number,
        )
        model = _fit_model(
            config,
            frame,
            x,
            y,
            _best_params(study),
            train_indices,
        )
        test_indices = [int(index) for index in fold["test_indices"]]
        predictions, probabilities = _predict(
            model,
            x[np.asarray(test_indices, dtype=np.int64)],
            configured_labels,
        )
        for position, index in enumerate(test_indices):
            prediction_by_index[index] = int(predictions[position])
            probability_by_index[index] = probabilities[position]
        fold_metrics = _evaluate_predictions(
            y[np.asarray(test_indices, dtype=np.int64)],
            predictions,
            probabilities,
            frame.loc[test_indices].reset_index(drop=True),
            config,
        )
        outer_reports.append(
            {
                "held_out_participant_id": participant,
                "best_inner_score": float(study.best_value),
                "best_params": _best_params(study),
                "training_selection": selection,
                "metrics": fold_metrics,
            }
        )

    missing_predictions = sorted(set(collector_indices) - set(prediction_by_index))
    if missing_predictions:
        raise RuntimeError("Nested evaluation did not predict every collector row")
    ordered_indices = np.asarray(collector_indices, dtype=np.int64)
    oof_predictions = np.asarray(
        [prediction_by_index[int(index)] for index in ordered_indices],
        dtype=np.int64,
    )
    oof_probabilities = np.vstack(
        [probability_by_index[int(index)] for index in ordered_indices]
    )
    holdout_metrics = _evaluate_predictions(
        y[ordered_indices],
        oof_predictions,
        oof_probabilities,
        frame.loc[collector_indices].reset_index(drop=True),
        config,
    )

    final_study = _optimize(
        config,
        frame,
        x,
        y,
        source_indices,
        manifest["final_participant_cv_folds"],
        label="final-participant-cv",
        apply_collector_domain_weight=True,
    )
    studies["final"] = final_study
    final_indices, final_selection = select_training_indices(
        frame,
        source_indices + collector_indices,
        strategy=config.trial.training.duration_balancing,
        random_seed=config.trial.training.random_seed,
    )
    final_model = _fit_model(
        config,
        frame,
        x,
        y,
        _best_params(final_study),
        final_indices,
    )
    return (
        {
            "evaluation_strategy": "participant_nested_cv",
            "best_cross_validation_score": float(final_study.best_value),
            **(
                {"best_cross_validation_macro_f1": float(final_study.best_value)}
                if config.trial.training.selection_metric == "macro_f1"
                else {}
            ),
            "selection_metric": config.trial.training.selection_metric,
            "cross_validation": {
                "method": "nested_participant_out_of_fold",
                "outer_folds": len(outer_reports),
                "inner_folds": config.trial.training.participant_inner_folds,
                "rows": len(collector_indices),
                "macro_f1": holdout_metrics["macro_f1"],
                "minimum_class_recall": holdout_metrics["minimum_class_recall"],
            },
            "best_params": _best_params(final_study),
            "training_selection": final_selection,
            "participant_outer_folds": outer_reports,
            "collector_holdout": holdout_metrics,
        },
        final_model,
        studies,
    )


def _optimize(
    config: PipelineConfig,
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    source_indices: list[int],
    cv_folds: list[dict[str, list[int]]],
    *,
    label: str,
    apply_collector_domain_weight: bool,
) -> optuna.Study:
    configured_labels = _label_values(config)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(trial, config.trial.training.model_families)
        truth_parts: list[np.ndarray] = []
        prediction_parts: list[np.ndarray] = []
        for fold_number, fold in enumerate(cv_folds):
            raw_train_indices = source_indices + fold["train_indices"]
            train_indices, _ = select_training_indices(
                frame,
                raw_train_indices,
                strategy=config.trial.training.duration_balancing,
                random_seed=config.trial.training.random_seed + fold_number,
            )
            valid_indices = np.asarray(fold["valid_indices"], dtype=np.int64)
            model = _fit_model(
                config,
                frame,
                x,
                y,
                params,
                train_indices,
                apply_collector_domain_weight=apply_collector_domain_weight,
            )
            predictions = model.predict(x[valid_indices]).astype(np.int64)
            truth_parts.append(y[valid_indices])
            prediction_parts.append(predictions)
            score = _selection_score(
                np.concatenate(truth_parts),
                np.concatenate(prediction_parts),
                configured_labels,
                config.trial.training.selection_metric,
            )
            if set(configured_labels).issubset(set(np.concatenate(truth_parts))):
                trial.report(score, step=fold_number)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return _selection_score(
            np.concatenate(truth_parts),
            np.concatenate(prediction_parts),
            configured_labels,
            config.trial.training.selection_metric,
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=config.trial.training.random_seed),
        pruner=optuna.pruners.HyperbandPruner(),
    )
    progress(
        f"Optuna optimization starting: scope={label}, "
        f"trials={config.trial.training.optuna_trials}"
    )
    study.optimize(
        objective,
        n_trials=config.trial.training.optuna_trials,
        timeout=config.training.timeout_seconds,
        callbacks=[_trial_progress(config.trial.training.optuna_trials, label)],
    )
    return study


def _fit_model(
    config: PipelineConfig,
    frame: pd.DataFrame,
    x: np.ndarray,
    y: np.ndarray,
    params: dict[str, Any],
    train_indices: Sequence[int],
    *,
    apply_collector_domain_weight: bool = True,
):
    indices = np.asarray(train_indices, dtype=np.int64)
    model = model_from_params(
        params,
        config.trial.training.random_seed,
        config.training.n_jobs,
        config.training.xgboost_device,
    )
    weights = training_sample_weights(
        frame,
        indices,
        strategy=config.trial.training.weighting_strategy,
        collector_domain_weight=config.trial.training.collector_domain_weight,
        apply_collector_domain_weight=apply_collector_domain_weight,
    )
    fit_with_optional_sample_weight(model, x[indices], y[indices], weights)
    return model


def _evaluate_model(
    model,
    x: np.ndarray,
    y: np.ndarray,
    frame: pd.DataFrame,
    indices: list[int],
    config: PipelineConfig,
) -> dict[str, Any]:
    if not indices:
        return {"rows": 0}
    idx = np.asarray(indices, dtype=np.int64)
    predictions, probabilities = _predict(model, x[idx], _label_values(config))
    return _evaluate_predictions(
        y[idx],
        predictions,
        probabilities,
        frame.loc[indices].reset_index(drop=True),
        config,
    )


def _evaluate_predictions(
    truth: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    config: PipelineConfig,
    *,
    include_session_level: bool = True,
) -> dict[str, Any]:
    labels = _label_values(config)
    label_names = _label_names(config)
    result = _base_metrics(truth, predictions, labels, label_names)
    result.update(
        {
            "rows": int(len(truth)),
            "groups": int(metadata["group_id"].nunique()),
            "sessions": int(metadata["session_id"].nunique()),
            "participants": int(metadata["participant_id"].nunique()),
            "by_phone_position": {},
            "by_participant": {},
        }
    )
    for position in sorted(metadata["phone_position"].astype(str).unique()):
        mask = metadata["phone_position"].astype(str).to_numpy() == position
        result["by_phone_position"][position] = _base_metrics(
            truth[mask], predictions[mask], labels, label_names
        )
    for participant in sorted(metadata["participant_id"].astype(str).unique()):
        mask = metadata["participant_id"].astype(str).to_numpy() == participant
        result["by_participant"][participant] = _base_metrics(
            truth[mask], predictions[mask], labels, label_names
        )
    iterations = config.trial.training.bootstrap_iterations
    if iterations:
        result["participant_cluster_95_ci"] = _cluster_confidence_intervals(
            truth,
            predictions,
            metadata["participant_id"].astype(str).to_numpy(),
            labels,
            label_names,
            iterations,
            config.trial.training.random_seed,
        )
    if include_session_level:
        session_truth, session_predictions, session_probabilities, session_metadata = (
            _aggregate_sessions(truth, probabilities, metadata, labels)
        )
        result["session_level"] = _evaluate_predictions(
            session_truth,
            session_predictions,
            session_probabilities,
            session_metadata,
            config,
            include_session_level=False,
        )
    return result


def _base_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    labels: Sequence[int],
    label_names: Sequence[str],
) -> dict[str, Any]:
    precision, recall, class_f1, support = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(np.mean(recall)),
        "macro_f1": float(np.mean(class_f1)),
        "minimum_class_recall": float(np.min(recall)),
        "classification_report": classification_report(
            truth,
            predictions,
            labels=labels,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            truth, predictions, labels=labels
        ).tolist(),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(class_f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(label_names)
        },
    }


def _aggregate_sessions(
    truth: np.ndarray,
    probabilities: np.ndarray,
    metadata: pd.DataFrame,
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    truth_values: list[int] = []
    prediction_values: list[int] = []
    probability_values: list[np.ndarray] = []
    metadata_rows: list[pd.Series] = []
    for _, positions in metadata.groupby("session_id", sort=True).indices.items():
        position_array = np.asarray(positions, dtype=np.int64)
        session_truth = np.unique(truth[position_array])
        if len(session_truth) != 1:
            raise ValueError("Each session must contain exactly one true label")
        mean_probability = probabilities[position_array].mean(axis=0)
        truth_values.append(int(session_truth[0]))
        prediction_values.append(int(labels[int(np.argmax(mean_probability))]))
        probability_values.append(mean_probability)
        metadata_rows.append(metadata.iloc[int(position_array[0])])
    return (
        np.asarray(truth_values, dtype=np.int64),
        np.asarray(prediction_values, dtype=np.int64),
        np.vstack(probability_values),
        pd.DataFrame(metadata_rows).reset_index(drop=True),
    )


def _cluster_confidence_intervals(
    truth: np.ndarray,
    predictions: np.ndarray,
    groups: np.ndarray,
    labels: Sequence[int],
    label_names: Sequence[str],
    iterations: int,
    random_seed: int,
) -> dict[str, Any]:
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(random_seed)
    values: dict[str, list[float]] = {
        "macro_f1": [],
        "balanced_accuracy": [],
        "minimum_class_recall": [],
        **{f"{name}.recall": [] for name in label_names},
    }
    for _ in range(iterations):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        positions = np.concatenate(
            [np.flatnonzero(groups == group) for group in sampled]
        )
        _, recall, class_f1, _ = precision_recall_fscore_support(
            truth[positions],
            predictions[positions],
            labels=labels,
            zero_division=0,
        )
        values["macro_f1"].append(float(np.mean(class_f1)))
        values["balanced_accuracy"].append(float(np.mean(recall)))
        values["minimum_class_recall"].append(float(np.min(recall)))
        for index, name in enumerate(label_names):
            if np.any(truth[positions] == labels[index]):
                values[f"{name}.recall"].append(float(recall[index]))
    return {
        metric: {
            "lower": float(np.percentile(samples, 2.5)),
            "upper": float(np.percentile(samples, 97.5)),
        }
        for metric, samples in values.items()
        if samples
    }


def _predict(
    model,
    x: np.ndarray,
    labels: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = model.predict(x).astype(np.int64)
    raw_probabilities = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=np.int64)
    probabilities = np.zeros((len(x), len(labels)), dtype=np.float64)
    label_positions = {int(label): index for index, label in enumerate(labels)}
    for source_position, label in enumerate(classes):
        if int(label) in label_positions:
            probabilities[:, label_positions[int(label)]] = raw_probabilities[
                :, source_position
            ]
    return predictions, probabilities


def _selection_score(
    truth: np.ndarray,
    predictions: np.ndarray,
    labels: Sequence[int],
    metric: str,
) -> float:
    _, recall, class_f1, _ = precision_recall_fscore_support(
        truth, predictions, labels=labels, zero_division=0
    )
    if metric == "minimum_class_recall":
        return float(np.min(recall))
    return float(np.mean(class_f1))


def _best_params(study: optuna.Study) -> dict[str, Any]:
    return dict(
        study.best_trial.user_attrs.get("model_params") or study.best_trial.params
    )


def _label_values(config: PipelineConfig) -> list[int]:
    return [
        value
        for _, value in sorted(config.trial.labels.items(), key=lambda item: item[1])
    ]


def _label_names(config: PipelineConfig) -> list[str]:
    return [
        name
        for name, _ in sorted(config.trial.labels.items(), key=lambda item: item[1])
    ]


def _read_feature_dataset(path: Path) -> pd.DataFrame:
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No feature Parquet parts found: {path}")
    return pd.concat((pd.read_parquet(part) for part in parts), ignore_index=True)


def _trial_progress(total_trials: int, label: str):
    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        try:
            best = f"{study.best_value:.4f}"
        except ValueError:
            best = "n/a"
        value = "n/a" if trial.value is None else f"{trial.value:.4f}"
        progress(
            f"Optuna {label} trial {trial.number + 1}/{total_trials}: "
            f"state={trial.state.name}, value={value}, best={best}"
        )

    return callback


def _log_holdout_confusion_matrices(
    holdout_metrics: dict[str, Any],
    config: PipelineConfig,
) -> None:
    if "confusion_matrix" not in holdout_metrics:
        return
    label_names = _label_names(config)
    log_confusion_matrix(
        holdout_metrics["confusion_matrix"],
        label_names,
        "evaluation/collector-holdout-confusion-matrix.png",
    )
    log_confusion_matrix(
        holdout_metrics["confusion_matrix"],
        label_names,
        "evaluation/collector-holdout-confusion-matrix-normalized.png",
        normalize=True,
    )
