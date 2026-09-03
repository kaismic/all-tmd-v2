from __future__ import annotations

from contextlib import nullcontext

import pandas as pd

from all_tmd.train import train
from all_tmd.windowing import feature_output_dir


def test_training_writes_required_reports(config_factory, monkeypatch):
    config = config_factory(mlflow_enabled=True, run_name="report-name")
    logged_artifacts = []
    logged_confusion_matrices = []
    monkeypatch.setattr(
        "all_tmd.train.start_run",
        lambda *_args: nullcontext(),
    )
    monkeypatch.setattr(
        "all_tmd.train.log_metrics",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "all_tmd.train.log_artifact",
        logged_artifacts.append,
    )
    monkeypatch.setattr(
        "all_tmd.train.log_confusion_matrix",
        lambda matrix, labels, artifact_file, **kwargs: (
            logged_confusion_matrices.append(
                (matrix, labels, artifact_file, kwargs)
            )
        ),
    )
    run_dir = config.run_dir()
    event_source = run_dir / "events" / "us-tmd"
    event_collector = run_dir / "events" / "collector"
    event_source.mkdir(parents=True)
    event_collector.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(event_source / "part-000000.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(event_collector / "part-000000.parquet")
    source_dir = feature_output_dir(config, "us-tmd")
    collector_dir = feature_output_dir(config, "collector")
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)

    source_rows = []
    for label in range(3):
        for index in range(4):
            source_rows.append(
                _feature_row(
                    "us-tmd",
                    f"source-{label}-{index}",
                    label,
                    label + index / 10,
                )
            )
    collector_rows = []
    for label in range(3):
        for index in range(4):
            collector_rows.append(
                _feature_row(
                    "collector",
                    f"collector-{label}-{index}",
                    label,
                    label + index / 10,
                )
            )
    pd.DataFrame(source_rows).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    pd.DataFrame(collector_rows).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )

    metrics = train(config)
    report_dir = config.report_dir()
    assert metrics["collector_holdout"]["rows"] == 6
    assert metrics["run_name"] == "report-name"
    assert (report_dir / "metrics.json").exists()
    assert (report_dir / "model.joblib").exists()
    assert (report_dir / "optuna-trials.csv").exists()
    assert {path.name for path in logged_artifacts} == {
        "metrics.json",
        "model.joblib",
        "optuna-trials.csv",
        f"{config.trial_hash}.json",
        "trial.json",
    }
    assert [
        (artifact_file, kwargs.get("normalize", False))
        for _, _, artifact_file, kwargs in logged_confusion_matrices
    ] == [
        ("evaluation/collector-holdout-confusion-matrix.png", False),
        (
            "evaluation/collector-holdout-confusion-matrix-normalized.png",
            True,
        ),
    ]
    assert all(
        labels == ["bus", "car", "train"]
        for _, labels, _, _ in logged_confusion_matrices
    )


def test_nested_participant_training_reports_unseen_participant_and_session_metrics(
    config_factory,
):
    config = config_factory(
        evaluation_strategy="participant_nested_cv",
        weighting_strategy="hierarchical",
        participant_inner_folds=2,
        bootstrap_iterations=10,
        selection_metric="minimum_class_recall",
    )
    run_dir = config.run_dir()
    event_source = run_dir / "events" / "us-tmd"
    event_collector = run_dir / "events" / "collector"
    event_source.mkdir(parents=True)
    event_collector.mkdir(parents=True)
    pd.DataFrame({"x": [1]}).to_parquet(event_source / "part-000000.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(event_collector / "part-000000.parquet")
    source_dir = feature_output_dir(config, "us-tmd")
    collector_dir = feature_output_dir(config, "collector")
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)

    source_rows = [
        _feature_row(
            "us-tmd",
            f"source-{label}-{index}",
            label,
            label + index / 100,
        )
        for label in range(3)
        for index in range(3)
    ]
    collector_rows = []
    for participant_number in range(3):
        participant = f"participant-{participant_number}"
        for label in range(3):
            row = _feature_row(
                "collector",
                f"{participant}-session-{label}",
                label,
                label + participant_number / 100,
            )
            row["participant_id"] = participant
            row["group_id"] = f"{participant}#{row['session_id']}"
            collector_rows.append(row)
    pd.DataFrame(source_rows).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    pd.DataFrame(collector_rows).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )

    metrics = train(config)

    assert metrics["evaluation_strategy"] == "participant_nested_cv"
    assert metrics["cross_validation"]["method"] == "nested_participant_out_of_fold"
    assert len(metrics["participant_outer_folds"]) == 3
    assert metrics["collector_holdout"]["rows"] == 9
    assert metrics["collector_holdout"]["participants"] == 3
    assert metrics["collector_holdout"]["session_level"]["rows"] == 9
    assert "participant_cluster_95_ci" in metrics["collector_holdout"]
    report_dir = config.report_dir()
    assert (report_dir / "nested-optuna-trials.csv").exists()


def _feature_row(
    domain: str,
    group_id: str,
    label: int,
    value: float,
) -> dict:
    return {
        "domain": domain,
        "participant_id": group_id,
        "device_id": "device",
        "session_id": group_id,
        "trip_id": group_id,
        "group_id": group_id,
        "vehicle_type": ("bus", "car", "train")[label],
        "label": label,
        "phone_position": "pocket",
        "window_start_ms": 0,
        "window_end_ms": 1000,
        "accelerometer#mean": value,
    }
