from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from all_tmd.study import StudyPlan, canonical_digest, create_lopo_manifest, verify_snapshot
from all_tmd.mlflow_utils import dataset_digest
from all_tmd.study_report import report_study
from all_tmd.study_runner import classification_metrics, run_study
from all_tmd.windowing import feature_output_dir


class _RoundedFeatureModel:
    def fit(self, _x, _y, **_kwargs):
        return self

    def predict(self, x):
        return np.clip(np.rint(x[:, 0]), 0, 2).astype(int)

    def predict_proba(self, x):
        predicted = self.predict(x)
        result = np.zeros((len(x), 3), dtype=float)
        result[np.arange(len(x)), predicted] = 1.0
        return result


def test_synthetic_study_keeps_outputs_isolated_and_conditions_paired(
    config_factory, tmp_path, monkeypatch
):
    config = config_factory(train_dataset="nor-tmd")
    run_dir = config.run_dir()
    for source_name in ("nor-tmd", "collector"):
        event_dir = run_dir / "events" / source_name
        event_dir.mkdir(parents=True)
        pd.DataFrame({"marker": [source_name]}).to_parquet(
            event_dir / "part-000000.parquet", index=False
        )

    source = pd.DataFrame(
        [
            _feature_row(
                "nor-tmd", f"nor-{participant}-{label}", f"nor-{participant}", label
            )
            for participant in range(5)
            for label in range(3)
        ]
    )
    snapshot_path = "manifests/sydney-166.json"
    eligible = set(verify_snapshot(snapshot_path)["eligible_session_ids"])
    snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
    collector = pd.DataFrame(
        [
            _feature_row(
                "collector",
                session["session_id"],
                session["participant_id"],
                {"bus": 0, "car": 1, "train": 2}[session["vehicle_type"]],
                session["vehicle_type"],
            )
            for session in snapshot["sessions"]
            if session["session_id"] in eligible
        ]
    )
    for name, frame in (("nor-tmd", source), ("collector", collector)):
        destination = feature_output_dir(config, name)
        destination.mkdir(parents=True)
        frame.to_parquet(destination / "part-000000.parquet", index=False)

    lopo = create_lopo_manifest(collector)
    lopo_path = tmp_path / "sydney-lopo.json"
    lopo_path.write_text(json.dumps(lopo), encoding="utf-8")
    lock = {
        "schema_version": 1,
        "model_family": "xgboost",
        "params": {"family": "xgboost", "n_estimators": 1},
        "feature_names": config.trial.feature_names,
        "tuning_dataset": "nor-tmd",
        "sydney_rows_used": 0,
        "source_feature_digest": dataset_digest(source, config.trial.feature_names),
        "selection_metric": "macro_f1",
        "cross_validation_score": 1.0,
        "tuning_run_id": "synthetic",
        "dependency_versions": {},
    }
    lock["lock_digest"] = canonical_digest(lock)
    lock_path = tmp_path / "model-lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    plan = replace(
        StudyPlan.load("study-plan.json"),
        lopo_manifest_path=lopo_path,
        model_lock_path=lock_path,
    )
    monkeypatch.setattr(
        "all_tmd.study_runner.model_from_params",
        lambda *_args, **_kwargs: _RoundedFeatureModel(),
    )

    reports = run_study(
        config, plan, output_root=tmp_path / "results", spec_limit=3
    )

    assert len(reports) == 3
    assert len({report["run_id"] for report in reports}) == 3
    assert len(list((tmp_path / "results").glob("*/metrics.json"))) == 3
    assert len({report["evaluation_manifest_digest"] for report in reports}) == 1
    assert all(
        fold["training_session_digest"] == canonical_digest([])
        for fold in reports[0]["folds"]
    )
    assert [fold["training_session_digest"] for fold in reports[1]["folds"]] == [
        fold["training_session_digest"] for fold in reports[2]["folds"]
    ]


def test_report_generates_complete_tables_figures_and_paired_intervals(tmp_path):
    plan = replace(StudyPlan.load("study-plan.json"), bootstrap_iterations=10)
    results_root = tmp_path / "results"
    predictions = pd.DataFrame(
        [
            {
                "participant_id": f"participant_{participant:03d}",
                "session_id": f"session-{participant}-{label}",
                "vehicle_type": mode,
                "label": label,
                "window_start_ms": 0,
                "window_end_ms": 60_000,
                "prediction": label,
            }
            for participant in range(7)
            for mode, label in plan.labels.items()
        ]
    )
    metrics = classification_metrics(predictions, plan.labels)
    for number, spec in enumerate(plan.parent_specs()):
        run_dir = results_root / f"run-{number:02d}"
        run_dir.mkdir(parents=True)
        predictions.to_parquet(run_dir / "predictions.parquet", index=False)
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "study_id": plan.study_id,
                    "run_name": spec.key,
                    "condition": spec.condition,
                    "sydney_fraction": spec.sydney_fraction,
                    "seed": spec.seed,
                    "evaluation_manifest_digest": "evaluation",
                    "model_lock_digest": "model",
                    "metrics": metrics,
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "report"
    summary = report_study(plan, results_root=results_root, output_dir=output)

    assert summary["valid"] is True
    assert summary["observed_parent_runs"] == 33
    for name in (
        "transfer-curve.csv",
        "transfer-curve.png",
        "transfer-curve.pdf",
        "per-class-performance.png",
        "controlled-comparison.tex",
        "paired-differences.csv",
        "paired-differences.tex",
        "study-summary.json",
    ):
        assert (output / name).is_file()
    paired = pd.read_csv(output / "paired-differences.csv")
    assert {"ci_lower", "ci_upper", "estimate"}.issubset(paired.columns)
    assert len(paired) == 20


def _feature_row(
    domain: str,
    session_id: str,
    participant_id: str,
    label: int,
    vehicle_type: str | None = None,
) -> dict[str, object]:
    return {
        "domain": domain,
        "participant_id": participant_id,
        "session_id": session_id,
        "group_id": f"{participant_id}#{session_id}",
        "vehicle_type": vehicle_type or ("bus", "car", "train")[label],
        "label": label,
        "window_start_ms": 0,
        "window_end_ms": 60_000,
        "accelerometer#mean": float(label),
    }
