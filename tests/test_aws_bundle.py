from datetime import datetime, timezone
import json

import pytest
import yaml

from all_tmd.aws_bundle import create_run_bundle


GIT_SHA = "a" * 40
COLLECTOR = {
    "collector_sessions_bucket": "transport-data-sessions-123456789012",
    "collector_sessions_table": "TransportSessions",
}


def test_full_bundle_contains_reproducibility_contracts(tmp_path):
    project = _project(tmp_path, evaluation=True)
    output = tmp_path / "bundle"
    manifest = create_run_bundle(
        project,
        output,
        run_id="20260903T120000Z-aaaaaaaa-full",
        git_repository="https://example.test/all-tmd-v2.git",
        git_commit=GIT_SHA,
        ntfy_topic="test-topic",
        created_at=datetime(2026, 9, 3, 12, tzinfo=timezone.utc),
        **COLLECTOR,
    )
    assert manifest["schema_version"] == 2
    assert manifest["expected_parent_runs"] == 33
    assert set(manifest["config_sha256"]) == {
        "model.config.yaml",
        "study-plan.json",
        "study-trial.json",
        "manifests/sydney-166.json",
        "manifests/sydney-lopo.json",
        "model-lock.json",
    }
    config = yaml.safe_load((output / "model.config.yaml").read_text())
    assert config["mlflow"]["tracking_uri"] == "sqlite:////mlflow-data/mlflow.db"


def test_tuning_bundle_does_not_require_model_or_lopo_lock(tmp_path):
    project = _project(tmp_path, evaluation=False)
    output = tmp_path / "bundle"
    manifest = create_run_bundle(
        project,
        output,
        run_id="tune-1",
        git_repository="https://example.test/all-tmd-v2.git",
        git_commit=GIT_SHA,
        mode="tune",
        **COLLECTOR,
    )
    assert manifest["expected_parent_runs"] == 0
    assert not (output / "model-lock.json").exists()


def test_full_bundle_requires_promoted_model_lock(tmp_path):
    project = _project(tmp_path, evaluation=False)
    with pytest.raises(FileNotFoundError, match="model-lock"):
        create_run_bundle(
            project,
            tmp_path / "bundle",
            run_id="full-1",
            git_repository="https://example.test/all-tmd-v2.git",
            git_commit=GIT_SHA,
            **COLLECTOR,
        )


def test_bundle_never_contains_notification_secret(tmp_path):
    output = tmp_path / "bundle"
    create_run_bundle(
        _project(tmp_path, evaluation=True),
        output,
        run_id="safe-run",
        git_repository="https://example.test/all-tmd-v2.git",
        git_commit=GIT_SHA,
        ntfy_token_parameter="/private/token",
        **COLLECTOR,
    )
    combined = "".join(path.read_text(errors="ignore") for path in output.rglob("*") if path.is_file())
    assert "secret-token-value" not in combined
    assert "/private/token" in combined


def _project(tmp_path, *, evaluation: bool):
    project = tmp_path / "project"
    (project / "manifests").mkdir(parents=True)
    (project / "model.config.yaml").write_text(
        "schema_version: 1\nmlflow:\n  enabled: false\n  experiment_name: test\n"
    )
    (project / "study-plan.json").write_text("{}")
    (project / "study-trial.json").write_text("[]")
    (project / "manifests" / "sydney-166.json").write_text("{}")
    if evaluation:
        (project / "model-lock.json").write_text("{}")
        (project / "manifests" / "sydney-lopo.json").write_text("{}")
    return project
