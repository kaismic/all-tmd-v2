from dataclasses import replace
import json
import sys
from unittest.mock import Mock

import pandas as pd
import pytest

from all_tmd import study_cli, study_report
from all_tmd.study import StudyPlan
from all_tmd.study_runner import classification_metrics


METRIC_FILES = ["per-run-mode-metrics.csv", "per-fold-mode-metrics.csv"]
CURVE_FILES = [
    "transfer-curve.csv", "transfer-curve.png", "transfer-curve.pdf",
    "controlled-comparison.tex", "study-summary.json",
]
PAIRED_FILES = ["paired-differences.csv", "paired-differences.tex"]


@pytest.fixture
def report_inputs(tmp_path):
    plan = replace(StudyPlan.load("study-plan.json"), bootstrap_iterations=2)
    results = tmp_path / "results"
    predictions = pd.DataFrame([
        {
            "label": label, "prediction": label, "vehicle_type": mode,
            "participant_id": "p0", "session_id": f"s{label}",
            "window_start_ms": 0, "window_end_ms": 60_000,
        }
        for mode, label in plan.labels.items()
    ])
    metrics = classification_metrics(predictions, plan.labels)
    for number, spec in enumerate(plan.parent_specs()[:3]):
        directory = results / str(number)
        directory.mkdir(parents=True)
        (directory / "metrics.json").write_text(json.dumps({
            "study_id": plan.study_id, "run_id": f"id-{number}",
            "run_name": spec.key, "condition": spec.condition,
            "sydney_fraction": spec.sydney_fraction, "seed": spec.seed,
            "model_lock_digest": "model", "evaluation_manifest_digest": "evaluation",
            "metrics": metrics,
            "folds": [{"fold": 0, "held_out_participant_id": "p0", "metrics": metrics}],
        }), encoding="utf-8")
        predictions.to_parquet(directory / "predictions.parquet", index=False)
    return plan, results, tmp_path / "output"


@pytest.mark.parametrize("files", [
    *[[name] for name in study_report.REPORT_FILES],
    ["transfer-curve.csv", "transfer-curve.png", "transfer-curve.pdf", "study-summary.json",
     "paired-differences.csv", "paired-differences.tex", "per-run-mode-metrics.csv"],
])
def test_selected_artifacts_compute_only_required_dependencies(report_inputs, monkeypatch, files):
    plan, results, output = report_inputs
    curve = Mock(wraps=study_report._curve_rows)
    paired = Mock(wraps=study_report._paired_rows)
    monkeypatch.setattr(study_report, "_curve_rows", curve)
    monkeypatch.setattr(study_report, "_paired_rows", paired)
    summary = study_report.report_study(
        plan, results_root=results, output_dir=output, allow_partial=True, files=files
    )
    assert {path.name for path in output.iterdir()} == set(files)
    assert set(summary["generated_files"]) == set(files)
    assert curve.call_count == int(bool(set(files).intersection(CURVE_FILES)))
    assert paired.call_count == int(bool(set(files).intersection(PAIRED_FILES)))
    assert (summary["curve_digest"] is not None) == bool(curve.call_count)
    if "study-summary.json" in files:
        assert json.loads((output / "study-summary.json").read_text()) == summary


def test_metric_csvs_need_only_saved_metrics_and_preserve_unselected_files(report_inputs, monkeypatch):
    plan, results, output = report_inputs
    for path in results.rglob("predictions.parquet"):
        path.unlink()
    for name in ("_curve_rows", "_paired_rows", "_plot_transfer_curve", "_plot_per_class"):
        monkeypatch.setattr(study_report, name, Mock(side_effect=AssertionError(name)))
    monkeypatch.setattr(pd, "read_parquet", Mock(side_effect=AssertionError("read_parquet")))
    summary = study_report.report_study(
        plan, results_root=results, output_dir=output, allow_partial=True,
        files=[*METRIC_FILES, METRIC_FILES[0]],
    )
    assert {path.name for path in output.iterdir()} == set(METRIC_FILES)
    assert summary["generated_files"] == METRIC_FILES
    assert summary["curve_digest"] is None
    for name in METRIC_FILES:
        assert len(pd.read_csv(output / name)) == 9
    preserved = output / "study-summary.json"
    preserved.write_bytes(b"existing summary")
    (output / METRIC_FILES[0]).write_bytes(b"old csv")
    study_report.report_study(
        plan, results_root=results, output_dir=output, allow_partial=True, files=METRIC_FILES
    )
    assert preserved.read_bytes() == b"existing summary"
    assert len(pd.read_csv(output / METRIC_FILES[0])) == 9


def test_per_class_plot_does_not_require_predictions(report_inputs, monkeypatch):
    plan, results, output = report_inputs
    for path in results.rglob("predictions.parquet"):
        path.unlink()
    monkeypatch.setattr(pd, "read_parquet", Mock(side_effect=AssertionError("read_parquet")))
    study_report.report_study(
        plan, results_root=results, output_dir=output, allow_partial=True,
        files=["per-class-performance.png"],
    )
    assert {path.name for path in output.iterdir()} == {"per-class-performance.png"}


@pytest.mark.parametrize("filename", CURVE_FILES + PAIRED_FILES)
def test_prediction_dependent_outputs_reject_missing_predictions(report_inputs, filename):
    plan, results, output = report_inputs
    next(results.rglob("predictions.parquet")).unlink()
    with pytest.raises(FileNotFoundError, match="predictions are missing"):
        study_report.report_study(
            plan, results_root=results, output_dir=output, allow_partial=True, files=[filename]
        )
    assert not output.exists()


@pytest.mark.parametrize("files", [[], ["invalid.csv"], [METRIC_FILES[0], "../escape.csv"]])
def test_invalid_selection_is_rejected_before_reading_or_writing(tmp_path, monkeypatch, files):
    monkeypatch.setattr(study_report, "collect_runs", Mock(side_effect=AssertionError("collect_runs")))
    with pytest.raises(ValueError, match="report filename"):
        study_report.report_study(
            StudyPlan.load("study-plan.json"), results_root=tmp_path / "missing",
            output_dir=tmp_path / "output", files=files,
        )
    assert not (tmp_path / "output").exists()


def test_metrics_only_selection_preserves_study_validation(report_inputs):
    plan, results, output = report_inputs
    for path in results.rglob("predictions.parquet"):
        path.unlink()
    original = next(results.rglob("metrics.json"))
    duplicate = json.loads(original.read_text())
    duplicate["evaluation_manifest_digest"] = "different-evaluation"
    (results / "duplicate").mkdir()
    (results / "duplicate" / "metrics.json").write_text(json.dumps(duplicate))
    with pytest.raises(ValueError, match="study is incomplete or inconsistent"):
        study_report.report_study(plan, results_root=results, output_dir=output, files=METRIC_FILES)
    assert not output.exists()
    summary = study_report.report_study(
        plan, results_root=results, output_dir=output, files=METRIC_FILES, allow_partial=True
    )
    assert summary["valid"] is False
    assert summary["missing_run_names"]
    assert summary["duplicate_run_names"] == [duplicate["run_name"]]
    assert len(summary["evaluation_manifest_digests"]) == 2
    assert len(pd.read_csv(output / METRIC_FILES[0])) == 12


@pytest.mark.parametrize("files", [None, METRIC_FILES])
def test_cli_forwards_selection_and_prints_metadata(monkeypatch, capsys, files):
    args = ["study", "report", "--results-root", "results", "--output-dir", "output"]
    if files is not None:
        args.extend(["--files", *files])
    monkeypatch.setattr(sys, "argv", args)
    report = Mock(return_value={"generated_files": files or list(study_report.REPORT_FILES)})
    monkeypatch.setattr(study_cli, "report_study", report)
    study_cli.main()
    assert report.call_count == 1
    assert report.call_args.kwargs["files"] == files
    assert json.loads(capsys.readouterr().out) == report.return_value


@pytest.mark.parametrize("args", [["--files"], ["--files", "invalid.csv"]])
def test_cli_rejects_invalid_selection(args):
    with pytest.raises(SystemExit) as error:
        study_cli.build_parser().parse_args(["report", *args])
    assert error.value.code == 2
