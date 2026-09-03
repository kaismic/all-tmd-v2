import json

import pandas as pd
import pytest

from all_tmd.source_tuning import promote_model
from all_tmd.study import (
    StudyPlan,
    assert_nested_subsets,
    calibration_session_ids,
    create_lopo_manifest,
    verify_snapshot,
)
from all_tmd.study_runner import load_model_lock


def test_frozen_snapshot_matches_published_membership():
    summary = verify_snapshot("manifests/sydney-166.json")
    assert summary["session_count"] == 166
    assert summary["participant_count"] == 7
    assert summary["eligible_session_count"] == 157
    assert summary["excluded_tram_count"] == 9
    assert len(summary["over_two_hour_session_ids"]) == 2
    assert set(summary["over_two_hour_session_ids"]).issubset(
        summary["eligible_session_ids"]
    )


def test_study_plan_generates_33_parent_runs():
    plan = StudyPlan.load("study-plan.json")
    specs = plan.parent_specs()
    assert len(specs) == 33
    assert sum(spec.condition == "nor_only" for spec in specs) == 3
    assert len({spec.key for spec in specs}) == 33


def test_lopo_and_nested_calibration_are_paired():
    collector = _collector_features()
    manifest = create_lopo_manifest(collector)
    assert manifest["participant_count"] == 7
    assert len(manifest["folds"]) == 7
    assert len(manifest["evaluation_session_ids"]) == 84
    for fold in manifest["folds"]:
        held_out = fold["held_out_participant_id"]
        test_participants = set(
            collector.loc[
                collector["session_id"].isin(fold["test_session_ids"]),
                "participant_id",
            ]
        )
        train_participants = set(
            collector.loc[
                collector["session_id"].isin(fold["training_pool_session_ids"]),
                "participant_id",
            ]
        )
        assert test_participants == {held_out}
        assert held_out not in train_participants
        assert_nested_subsets(collector, fold, (0.1, 0.25, 0.5, 0.75, 1.0), 42)
        sydney, first = calibration_session_ids(collector, fold, 0.5, 42)
        combined, second = calibration_session_ids(collector, fold, 0.5, 42)
        assert sydney == combined
        assert first["session_digest"] == second["session_digest"]


def test_model_promotion_and_validation(tmp_path, monkeypatch):
    run_id = "a" * 32
    result_dir = tmp_path / "results" / "tuning" / run_id
    result_dir.mkdir(parents=True)
    result = {
        "run_id": run_id,
        "tuning_dataset": "nor-tmd",
        "sydney_rows_used": 0,
        "source_feature_digest": "source",
        "feature_names": ["accelerometer#mean"],
        "best_cross_validation_macro_f1": 0.8,
        "best_params": {"family": "xgboost", "n_estimators": 200},
        "dependency_versions": {"xgboost": "3.3.0"},
    }
    (result_dir / "tuning-result.json").write_text(json.dumps(result), encoding="utf-8")
    lock_path = tmp_path / "model-lock.json"
    lock = promote_model(
        run_id, output_root=tmp_path / "results", lock_path=lock_path
    )
    assert load_model_lock(lock_path, ["accelerometer#mean"]) == lock
    lock["sydney_rows_used"] = 1
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="NOR-TMD only"):
        load_model_lock(lock_path, ["accelerometer#mean"])


def _collector_features() -> pd.DataFrame:
    rows = []
    for participant_number in range(7):
        participant = f"participant_{participant_number:03d}"
        for mode, label in {"bus": 0, "car": 1, "train": 2}.items():
            for session_number in range(4):
                session = f"{participant}-{mode}-{session_number}"
                rows.append(
                    {
                        "domain": "collector",
                        "participant_id": participant,
                        "session_id": session,
                        "vehicle_type": mode,
                        "label": label,
                        "window_start_ms": 0,
                        "window_end_ms": 60_000,
                    }
                )
    return pd.DataFrame(rows)
