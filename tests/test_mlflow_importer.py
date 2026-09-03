from pathlib import Path

import pytest

from all_tmd.mlflow_importer import import_mlflow_runs


def test_import_is_idempotent_and_remaps_parent(tmp_path, monkeypatch):
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.chdir(tmp_path)
    source_database = tmp_path / "source.db"
    destination_database = tmp_path / "destination.db"
    source_uri = f"sqlite:///{source_database.as_posix()}"
    destination_uri = f"sqlite:///{destination_database.as_posix()}"
    client = mlflow.MlflowClient(tracking_uri=source_uri)
    experiment = client.create_experiment(
        "source", artifact_location=(tmp_path / "source-artifacts").as_uri()
    )
    parent = client.create_run(
        experiment,
        run_name="parent",
        tags={"study_id": "sydney-transfer-v2", "run_level": "parent"},
    )
    client.log_metric(parent.info.run_id, "macro_f1", 0.7, step=0)
    client.log_metric(parent.info.run_id, "macro_f1", 0.8, step=1)
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}")
    client.log_artifact(parent.info.run_id, str(artifact))
    child = client.create_run(
        experiment,
        run_name="child",
        tags={
            "study_id": "sydney-transfer-v2",
            "run_level": "fold",
            "mlflow.parentRunId": parent.info.run_id,
        },
    )
    client.set_terminated(child.info.run_id)
    client.set_terminated(parent.info.run_id)

    first = import_mlflow_runs(
        source_database,
        destination_uri,
        study_id="sydney-transfer-v2",
        sweep_id="aws-1",
    )
    second = import_mlflow_runs(
        source_database,
        destination_uri,
        study_id="sydney-transfer-v2",
        sweep_id="aws-1",
    )
    assert first["imported"] == 2
    assert second["imported"] == 0
    assert second["skipped"] == 2
    destination = mlflow.MlflowClient(tracking_uri=destination_uri)
    imported_parent = destination.get_run(first["run_id_mapping"][parent.info.run_id])
    imported_child = destination.get_run(first["run_id_mapping"][child.info.run_id])
    assert len(destination.get_metric_history(imported_parent.info.run_id, "macro_f1")) == 2
    assert imported_child.data.tags["mlflow.parentRunId"] == imported_parent.info.run_id
    assert destination.list_artifacts(imported_parent.info.run_id)[0].path == "metrics.json"
