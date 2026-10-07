from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from all_tmd.study import StudyPlan
from all_tmd.study_report import _mode_metric_rows, _write_mode_metrics
from all_tmd.study_runner import classification_metrics


def _metrics(matrix, labels):
    records = [
        {"label": truth, "prediction": predicted, "session_id": "s", "participant_id": "p"}
        for truth_index, truth in enumerate(labels.values())
        for predicted_index, predicted in enumerate(labels.values())
        for _ in range(matrix[truth_index][predicted_index])
    ]
    return classification_metrics(pd.DataFrame(records), labels)


def test_mode_metrics_use_one_vs_rest_counts_and_configured_label_order():
    # Matrix positions follow label insertion order, not numeric label values.
    labels = {"train": 2, "bus": 0, "car": 1}
    metrics = _metrics([[4, 1, 1], [2, 3, 0], [0, 2, 7]], labels)
    rows = _mode_metric_rows(metrics, labels)
    assert [row["transport_mode"] for row in rows] == list(labels)
    expected = [
        (6, 4 / 6, 4 / 6, 2 / 3, 16 / 20),
        (5, 3 / 6, 3 / 5, 6 / 11, 15 / 20),
        (9, 7 / 8, 7 / 9, 14 / 17, 17 / 20),
    ]
    for row, values in zip(rows, expected):
        assert [row[key] for key in (
            "support", "precision", "recall", "f1", "accuracy"
        )] == pytest.approx(values)
        assert row["accuracy"] != pytest.approx(metrics["accuracy"])
        assert row["accuracy"] != pytest.approx(row["recall"])


@pytest.mark.parametrize("matrix", [
    [[2, 0, 0], [1, 0, 0], [0, 0, 0]],  # absent train; no car predictions
    [[2, 1, 0], [0, 0, 0], [0, 0, 0]],  # only bus truth
])
def test_mode_metrics_handle_absent_classes_and_predictions(matrix):
    labels = {"bus": 0, "car": 1, "train": 2}
    metrics = _metrics(matrix, labels)
    bus, car, train = _mode_metric_rows(metrics, labels)
    assert train["support"] == 0
    assert train["precision"] == train["recall"] == train["f1"] == 0
    assert train["accuracy"] == 1
    assert car["precision"] == car["recall"] == car["f1"] == 0
    assert bus["accuracy"] == pytest.approx(2 / 3)


def test_csv_exports_preserve_runs_write_blanks_and_sort_deterministically(tmp_path):
    plan = replace(StudyPlan.load("study-plan.json"), labels={"train": 2, "bus": 0, "car": 1})
    metrics = _metrics([[2, 1, 0], [0, 0, 0], [0, 0, 0]], plan.labels)
    identity = {
        "study_id": plan.study_id, "run_name": "same-name", "condition": "nor_only",
        "sydney_fraction": 0, "seed": 42, "model_lock_digest": "model", "metrics": metrics,
    }
    empty = {
        "confusion_matrix": np.zeros((3, 3), dtype=int).tolist(),
        "per_class": {mode: {"precision": 0, "recall": 0, "f1": 0, "support": 0} for mode in plan.labels},
    }
    runs = [
        {**identity, "run_id": "b"},
        {**identity, "run_id": "a", "folds": [
            {"fold": 1, "held_out_participant_id": "p1", "metrics": empty},
            {"fold": 0, "held_out_participant_id": "p0", "metrics": metrics},
        ]},
    ]
    _write_mode_metrics(runs, plan, tmp_path)
    paths = [tmp_path / name for name in ("per-run-mode-metrics.csv", "per-fold-mode-metrics.csv")]
    original = [path.read_bytes() for path in paths]
    _write_mode_metrics(runs[::-1], plan, tmp_path)
    assert [path.read_bytes() for path in paths] == original
    parents, folds = [pd.read_csv(path, keep_default_na=False) for path in paths]
    assert parents["run_id"].tolist() == ["a"] * 3 + ["b"] * 3
    assert parents["transport_mode"].tolist() == list(plan.labels) * 2
    assert "balanced_accuracy" not in parents.columns
    assert "balanced_accuracy" not in folds.columns
    assert folds["fold"].tolist() == [0] * 3 + [1] * 3
    assert folds.loc[folds["fold"] == 1, "accuracy"].eq("").all()
    _write_mode_metrics([], plan, tmp_path)
    assert all(pd.read_csv(path).empty for path in paths)
