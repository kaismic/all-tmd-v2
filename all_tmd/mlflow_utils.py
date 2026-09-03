from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from all_tmd.config import PipelineConfig


DATASET_ID_COLUMNS = (
    "domain",
    "group_id",
    "label",
    "session_id",
    "window_start_ms",
    "window_end_ms",
)
MLFLOW_DATASET_DIGEST_LENGTH = 36


@contextmanager
def start_run(
    config: PipelineConfig,
    frame: pd.DataFrame,
    split_manifest: dict[str, Any],
):
    if not config.mlflow.enabled:
        yield None
        return
    import mlflow

    if config.mlflow.tracking_uri:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    _set_experiment(mlflow, config)
    collector_digest, collector_count = collector_session_summary(frame)
    feature_params = {
        "sensors": ",".join(config.trial.features.sensors),
        "feature_names": ",".join(config.trial.feature_names),
        "context_windows_seconds": ",".join(
            str(value) for value in config.trial.features.context_windows_seconds
        ),
        **{
            f"features.{sensor}": ",".join(aggregations)
            for sensor, aggregations in config.trial.features.sensors.items()
        },
    }
    collector_quality_params = {
        "collector_max_sample_interval_ms": (
            config.dataset.collector_max_sample_interval_ms
            if config.dataset.collector_max_sample_interval_ms is not None
            else "disabled"
        ),
        **{
            f"collector_minimum_sampling_rate.{sensor}": (
                config.collector_minimum_sampling_rate[sensor]
            )
            for sensor in config.trial.features.sensors
        },
    }
    calibration_params = {
        f"calibration_fraction.{mode}": fraction
        for mode, fraction in config.trial.training.calibration_fraction.items()
    }
    run_name = mlflow_run_name(config, collector_count)
    with mlflow.start_run(
        run_name=run_name
    ) as run:
        run_params = {
            "config_hash": config.config_hash,
            "trial_hash": config.trial_hash,
            "trial_index": config.trial_index,
            "train_dataset": config.trial.train_dataset,
            "window_seconds": config.trial.features.default_window_seconds,
            "step_seconds": config.trial.features.default_step_seconds,
            **feature_params,
            **collector_quality_params,
            **calibration_params,
            "model_families": ",".join(config.trial.training.model_families),
            "optuna_trials": config.trial.training.optuna_trials,
            "evaluation_strategy": config.trial.training.evaluation_strategy,
            "weighting_strategy": config.trial.training.weighting_strategy,
            "collector_domain_weight": config.trial.training.collector_domain_weight,
            "duration_balancing": config.trial.training.duration_balancing,
            "selection_metric": config.trial.training.selection_metric,
            "participant_inner_folds": config.trial.training.participant_inner_folds,
            "collector_session_digest": collector_digest,
            "collector_session_count": collector_count,
        }
        if config.trial.run_name is not None:
            run_params["configured_run_name"] = config.trial.run_name
        mlflow.log_params(run_params)
        log_dataset_inputs(config, frame, split_manifest)
        yield run


def _set_experiment(mlflow: Any, config: PipelineConfig) -> None:
    artifact_location = config.mlflow.artifact_location
    if not artifact_location:
        mlflow.set_experiment(config.mlflow.experiment_name)
        return

    experiment = mlflow.get_experiment_by_name(config.mlflow.experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            config.mlflow.experiment_name,
            artifact_location=artifact_location,
        )
    else:
        if experiment.artifact_location != artifact_location:
            raise ValueError(
                f"MLflow experiment '{config.mlflow.experiment_name}' uses "
                f"artifact location {experiment.artifact_location!r}, expected "
                f"{artifact_location!r}"
            )
        experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


def mlflow_run_name(
    config: PipelineConfig,
    collector_session_count: int,
    started_at: datetime | None = None,
) -> str:
    """Build a run name containing its UTC start time and collector size."""
    start_time = started_at or datetime.now(timezone.utc)
    if start_time.tzinfo is None:
        raise ValueError("MLflow run start time must include timezone information")
    utc_time = start_time.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    generated_name = (
        f"{config.trial.train_dataset}-{config.trial_hash[:8]}-"
        f"{utc_time}-{collector_session_count}"
    )
    if config.trial.run_name is None:
        return generated_name
    return f"{config.trial.run_name}-{generated_name}"


def collector_session_summary(frame: pd.DataFrame) -> tuple[str, int]:
    collector = frame.loc[
        frame["domain"].astype(str) == "collector",
        "session_id",
    ]
    session_ids = sorted(set(collector.dropna().astype(str)))
    canonical = json.dumps(
        session_ids,
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest(), len(session_ids)


def dataset_digest(frame: pd.DataFrame, feature_names: Sequence[str]) -> str:
    columns = list(DATASET_ID_COLUMNS) + list(feature_names)
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(
            "MLflow dataset fingerprint is missing column(s): "
            + ", ".join(missing)
        )
    selected = frame.loc[:, columns]
    row_hashes = pd.util.hash_pandas_object(
        selected,
        index=False,
        categorize=True,
    ).to_numpy(dtype="uint64", copy=True)
    row_hashes.sort()
    header = json.dumps(
        {
            "columns": columns,
            "dtypes": [str(selected[column].dtype) for column in columns],
        },
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\n")
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()[:MLFLOW_DATASET_DIGEST_LENGTH]


def log_dataset_inputs(
    config: PipelineConfig,
    frame: pd.DataFrame,
    split_manifest: dict[str, Any],
) -> None:
    import mlflow

    from all_tmd.windowing import feature_output_dir

    def feature_source(source_name: str) -> Path:
        try:
            return feature_output_dir(config, source_name)
        except FileNotFoundError:
            # Unit-level callers may log an in-memory frame without materialized
            # feature artifacts. Production training always resolves the
            # content-addressed path before this function is reached.
            return config.run_dir() / "features" / "in-memory" / source_name

    feature_names = config.trial.feature_names
    source_dataset = (
        f"{config.trial.train_dataset}-training-features",
        "training",
        split_manifest["source_indices"],
        feature_source(config.trial.train_dataset),
    )
    if split_manifest.get("evaluation_strategy") == "participant_nested_cv":
        collector_indices = split_manifest["collector_evaluation_indices"]
        datasets = (
            source_dataset,
            (
                "collector-deployment-training-features",
                "calibration",
                collector_indices,
                feature_source("collector"),
            ),
            (
                "collector-participant-oof-evaluation-features",
                "evaluation",
                collector_indices,
                feature_source("collector"),
            ),
        )
    else:
        datasets = (
            source_dataset,
            (
                "collector-calibration-features",
                "calibration",
                split_manifest["collector_calibration_indices"],
                feature_source("collector"),
            ),
            (
                "collector-holdout-features",
                "evaluation",
                split_manifest["collector_holdout_indices"],
                feature_source("collector"),
            ),
        )
    columns = list(DATASET_ID_COLUMNS) + feature_names
    for name, context, indices, source in datasets:
        dataset_frame = frame.loc[indices, columns].reset_index(drop=True)
        dataset = mlflow.data.from_pandas(
            dataset_frame,
            source=str(source),
            name=name,
            digest=dataset_digest(dataset_frame, feature_names),
        )
        mlflow.log_input(dataset, context=context)


def log_metrics(metrics: dict[str, Any], prefix: str = "") -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            mlflow.log_metric(f"{prefix}{key}", float(value))


def log_artifact(path: Path) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    if path.exists():
        mlflow.log_artifact(str(path))


def log_confusion_matrix(
    matrix: Sequence[Sequence[int]],
    label_names: Sequence[str],
    artifact_file: str,
    *,
    normalize: bool = False,
) -> None:
    try:
        import mlflow
    except ModuleNotFoundError:
        return
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import ConfusionMatrixDisplay

    values = np.asarray(matrix)
    if values.shape != (len(label_names), len(label_names)):
        raise ValueError(
            "Confusion matrix dimensions must match the configured labels"
        )
    if normalize:
        values = values.astype(np.float64)
        row_totals = values.sum(axis=1, keepdims=True)
        values = np.divide(
            values,
            row_totals,
            out=np.zeros_like(values),
            where=row_totals != 0,
        )

    figure = Figure(figsize=(7, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    try:
        ConfusionMatrixDisplay(
            confusion_matrix=values,
            display_labels=list(label_names),
        ).plot(
            ax=axis,
            cmap="Blues",
            colorbar=False,
            values_format=".2f" if normalize else "d",
        )
        title_suffix = " (row normalized)" if normalize else ""
        axis.set_title(f"Collector holdout confusion matrix{title_suffix}")
        figure.tight_layout()
        mlflow.log_figure(figure, artifact_file)
    finally:
        figure.clear()
