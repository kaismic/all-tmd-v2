from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


@pytest.mark.parametrize("run_count,legacy", [(33, False), (33, True), (3, True)])
def test_report_generates_complete_tables_figures_and_paired_intervals(
    tmp_path, run_count, legacy
):
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
                "prediction": 1 if participant == 0 and label == 0 else label,
            }
            for participant in range(7)
            for mode, label in plan.labels.items()
        ]
    )
    metrics = classification_metrics(predictions, plan.labels)
    specs = plan.parent_specs()[:run_count]
    folds = [
        {
            "fold": number,
            "held_out_participant_id": participant,
            "metrics": classification_metrics(frame, plan.labels),
        }
        for number, (participant, frame) in enumerate(predictions.groupby("participant_id"))
    ]
    for number, spec in enumerate(specs):
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
                    **({} if legacy else {"run_id": f"id-{number:02d}", "folds": folds[::-1]}),
                }
            ),
            encoding="utf-8",
        )

    output = tmp_path / "report"
    if run_count < 33:
        with pytest.raises(ValueError, match="study is incomplete or inconsistent"):
            report_study(plan, results_root=results_root, output_dir=output)
        assert not output.exists()
    summary = report_study(
        plan, results_root=results_root, output_dir=output, allow_partial=run_count < 33
    )

    assert summary["valid"] is (run_count == 33)
    assert summary["observed_parent_runs"] == run_count
    assert len(list(output.iterdir())) == 10
    assert set(summary["generated_files"]) == {path.name for path in output.iterdir()}
    assert summary["curve_digest"] is not None
    for name in (
        "transfer-curve.csv",
        "transfer-curve.png",
        "transfer-curve.pdf",
        "per-class-performance.png",
        "controlled-comparison.tex",
        "paired-differences.csv",
        "paired-differences.tex",
        "study-summary.json",
        "per-run-mode-metrics.csv",
        "per-fold-mode-metrics.csv",
    ):
        assert (output / name).is_file()
    paired = pd.read_csv(output / "paired-differences.csv")
    assert {"ci_lower", "ci_upper", "estimate"}.issubset(paired.columns)
    if run_count == 33:
        assert len(paired) == 20

    parents = pd.read_csv(output / "per-run-mode-metrics.csv", keep_default_na=False)
    fold_metrics = pd.read_csv(output / "per-fold-mode-metrics.csv", keep_default_na=False)
    columns = [
        "study_id", "run_id", "run_name", "condition", "sydney_fraction", "seed",
        "model_lock_digest", "transport_mode", "support", "precision", "recall",
        "f1", "accuracy", "balanced_accuracy",
    ]
    assert list(parents.columns) == columns
    assert list(fold_metrics.columns) == columns[:7] + ["fold", "held_out_participant_id"] + columns[7:]
    assert len(parents) == run_count * 3
    assert len(fold_metrics) == (0 if legacy else run_count * 7 * 3)
    assert not parents.duplicated(["run_name", "transport_mode"]).any()
    assert parents["study_id"].eq(plan.study_id).all()
    assert parents["model_lock_digest"].eq("model").all()
    ordered_specs = sorted(specs, key=lambda spec: (spec.condition, spec.sydney_fraction, spec.seed))
    assert list(zip(parents["run_name"], parents["transport_mode"])) == [
        (spec.key, mode) for spec in ordered_specs for mode in plan.labels
    ]
    for number, spec in enumerate(specs):
        selected = parents.loc[parents["run_name"] == spec.key]
        assert selected["run_id"].eq("" if legacy else f"id-{number:02d}").all()
        assert selected["condition"].eq(spec.condition).all()
        assert selected["sydney_fraction"].eq(spec.sydney_fraction).all()
        assert selected["seed"].eq(spec.seed).all()
        assert selected["support"].eq(7).all()
        for row in selected.itertuples():
            for metric in ("precision", "recall", "f1"):
                assert getattr(row, metric) == pytest.approx(metrics["per_class"][row.transport_mode][metric])
    if not legacy:
        assert list(zip(fold_metrics["run_name"], fold_metrics["fold"], fold_metrics["transport_mode"])) == [
            (spec.key, fold["fold"], mode)
            for spec in ordered_specs for fold in folds for mode in plan.labels
        ]
        assert fold_metrics["support"].eq(1).all()
        for row in fold_metrics.itertuples():
            fold = folds[row.fold]
            assert row.held_out_participant_id == fold["held_out_participant_id"]
            assert row.run_id == parents.loc[parents["run_name"] == row.run_name, "run_id"].iloc[0]
            for metric in ("precision", "recall", "f1"):
                assert getattr(row, metric) == pytest.approx(fold["metrics"]["per_class"][row.transport_mode][metric])
        bus = fold_metrics.loc[(fold_metrics["fold"] == 0) & (fold_metrics["transport_mode"] == "bus")].iloc[0]
        assert bus.accuracy == pytest.approx(2 / 3)
        assert bus.balanced_accuracy == pytest.approx(0.5)


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
