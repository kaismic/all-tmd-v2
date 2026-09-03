from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Any


IMPORT_SOURCE_RUN_TAG = "import.source_run_id"
IMPORT_SOURCE_STORE_TAG = "import.source_store_digest"
IMPORT_SOURCE_SWEEP_TAG = "import.source_sweep_id"


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def import_mlflow_runs(
    source_database: str | Path,
    destination_uri: str,
    *,
    study_id: str,
    sweep_id: str,
    experiment_name: str = "ALL-TMD-V2 Sydney Transfer",
) -> dict[str, Any]:
    from mlflow import MlflowClient
    from mlflow.entities import Metric, Param, RunTag

    database = Path(source_database).resolve()
    if not database.exists():
        raise FileNotFoundError(database)
    source_uri = f"sqlite:///{database.as_posix()}"
    source = MlflowClient(tracking_uri=source_uri)
    destination = MlflowClient(tracking_uri=destination_uri)
    source_store_digest = file_digest(database)

    destination_experiment = destination.get_experiment_by_name(experiment_name)
    if destination_experiment is None:
        destination_experiment_id = destination.create_experiment(experiment_name)
    else:
        destination_experiment_id = destination_experiment.experiment_id

    existing = destination.search_runs(
        [destination_experiment_id],
        filter_string=f"tags.`{IMPORT_SOURCE_SWEEP_TAG}` = '{sweep_id}'",
        max_results=50000,
    )
    existing_ids = {
        run.data.tags.get(IMPORT_SOURCE_RUN_TAG): run.info.run_id
        for run in existing
        if run.data.tags.get(IMPORT_SOURCE_RUN_TAG)
    }

    source_runs = []
    for experiment in source.search_experiments():
        source_runs.extend(
            source.search_runs(
                [experiment.experiment_id],
                filter_string=f"tags.study_id = '{study_id}'",
                max_results=50000,
            )
        )
    source_runs.sort(
        key=lambda run: (
            bool(run.data.tags.get("mlflow.parentRunId")),
            run.info.start_time or 0,
            run.info.run_id,
        )
    )
    mapping = dict(existing_ids)
    imported = 0
    skipped = 0
    for source_run in source_runs:
        source_id = source_run.info.run_id
        if source_id in mapping:
            skipped += 1
            continue
        tags = {
            key: value
            for key, value in source_run.data.tags.items()
            if key not in {"mlflow.runName", "mlflow.parentRunId"}
        }
        source_parent = source_run.data.tags.get("mlflow.parentRunId")
        if source_parent:
            if source_parent not in mapping:
                raise ValueError(f"source parent {source_parent} was not imported first")
            tags["mlflow.parentRunId"] = mapping[source_parent]
        tags.update(
            {
                IMPORT_SOURCE_RUN_TAG: source_id,
                IMPORT_SOURCE_STORE_TAG: source_store_digest,
                IMPORT_SOURCE_SWEEP_TAG: sweep_id,
                "execution_backend": source_run.data.tags.get(
                    "execution_backend", "aws"
                ),
            }
        )
        created = destination.create_run(
            destination_experiment_id,
            start_time=source_run.info.start_time,
            tags=tags,
            run_name=source_run.info.run_name,
        )
        destination_id = created.info.run_id
        mapping[source_id] = destination_id
        metrics = []
        for key in source_run.data.metrics:
            metrics.extend(
                Metric(item.key, item.value, item.timestamp, item.step)
                for item in source.get_metric_history(source_id, key)
            )
        params = [Param(key, value) for key, value in source_run.data.params.items()]
        tag_entities = [RunTag(key, value) for key, value in tags.items()]
        for offset in range(0, max(len(metrics), len(params), len(tag_entities), 1), 500):
            destination.log_batch(
                destination_id,
                metrics=metrics[offset : offset + 500],
                params=params[offset : offset + 500],
                tags=tag_entities[offset : offset + 500],
            )
        if source_run.inputs and source_run.inputs.dataset_inputs:
            destination.log_inputs(
                destination_id, datasets=source_run.inputs.dataset_inputs
            )
        with tempfile.TemporaryDirectory() as temporary:
            downloaded = Path(source.download_artifacts(source_id, "", temporary))
            if downloaded.exists() and any(downloaded.iterdir()):
                destination.log_artifacts(destination_id, str(downloaded))
        destination.set_terminated(
            destination_id,
            status=source_run.info.status,
            end_time=source_run.info.end_time,
        )
        imported += 1
    return {
        "source_store_digest": source_store_digest,
        "source_run_count": len(source_runs),
        "imported": imported,
        "skipped": skipped,
        "run_id_mapping": mapping,
    }
