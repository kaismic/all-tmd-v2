from __future__ import annotations

import json
import hashlib
from pathlib import Path
import shutil
import time
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from all_tmd.config import PipelineConfig
from all_tmd.features import VECTOR_SENSOR_COLUMNS, ordered_sensor_features, sensor_series
from all_tmd.progress import progress


METADATA_COLUMNS = [
    "domain",
    "participant_id",
    "device_id",
    "session_id",
    "trip_id",
    "group_id",
    "vehicle_type",
    "label",
    "phone_position",
    "window_start_ms",
    "window_end_ms",
]
FEATURE_BUCKETS = 64
FEATURE_CHECKPOINT = "checkpoint.json"
FEATURE_POLICY = "feature-policy.json"
FEATURE_POLICY_SCHEMA_VERSION = 3
FEATURE_SESSION_HEARTBEAT_SECONDS = 10.0


def build_features(config: PipelineConfig) -> dict[str, Path]:
    source = config.trial.train_dataset
    return {
        source: _build_source_features(config, source, incremental=False),
        "collector": _build_source_features(
            config,
            "collector",
            incremental=config.dataset.snapshot_manifest is None,
        ),
    }


def feature_output_dir(config: PipelineConfig, source_name: str) -> Path:
    event_dir = config.run_dir() / "events" / source_name
    artifact_digest = _event_artifact_digest(event_dir)
    identity = {
        "schema_version": FEATURE_POLICY_SCHEMA_VERSION,
        "source_name": source_name,
        "event_artifact_digest": artifact_digest,
        "features": config.trial.raw["features"],
        "minimum_trip_seconds": config.dataset.minimum_trip_seconds,
        "maximum_trip_seconds": config.dataset.maximum_trip_seconds,
    }
    if source_name == "collector":
        identity["collector_sampling_quality"] = {
            "collector_minimum_sampling_rate": {
                sensor: config.collector_minimum_sampling_rate[sensor]
                for sensor in sorted(config.trial.features.sensors)
            },
            "collector_max_sample_interval_ms": (
                config.dataset.collector_max_sample_interval_ms
            ),
        }
    encoded = json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()
    return config.run_dir() / "features" / cache_key / source_name


def _event_artifact_digest(event_dir: Path) -> str:
    if not event_dir.exists():
        raise FileNotFoundError(f"Run ingestion before features: {event_dir}")
    digest = hashlib.sha256()
    parts = sorted(event_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"Run ingestion before features: {event_dir}")
    for path in parts:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _build_source_features(
    config: PipelineConfig,
    source_name: str,
    *,
    incremental: bool,
) -> Path:
    run_dir = config.run_dir()
    event_dir = run_dir / "events" / source_name
    if not event_dir.exists() or not any(event_dir.glob("part-*.parquet")):
        raise FileNotFoundError(f"Run ingestion before features: {event_dir}")
    if not incremental and not (event_dir / "_SUCCESS").exists():
        raise RuntimeError(
            f"Training event dataset is incomplete (missing _SUCCESS): {event_dir}"
        )
    final_output_dir = feature_output_dir(config, source_name)
    expected_policy = _feature_policy(config, source_name)
    if final_output_dir.exists() and not _feature_policy_matches(
        final_output_dir / FEATURE_POLICY,
        expected_policy,
    ):
        progress(f"Rebuilding incomplete/corrupt feature cache: {final_output_dir}")
        shutil.rmtree(final_output_dir)
    success_path = final_output_dir / "_SUCCESS"
    if not incremental and success_path.exists():
        progress(f"Training features already complete: {final_output_dir}")
        return final_output_dir
    if not incremental and final_output_dir.exists():
        progress(f"Removing incomplete training feature dataset: {final_output_dir}")
        shutil.rmtree(final_output_dir)
    building_dir: Path | None = None
    if incremental:
        output_dir = final_output_dir
    else:
        building_dir = final_output_dir.with_name(
            f".{source_name}.building-{uuid4().hex}"
        )
        output_dir = building_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    success_path = output_dir / "_SUCCESS"
    checkpoint_path = output_dir / FEATURE_CHECKPOINT
    _write_feature_policy(output_dir / FEATURE_POLICY, expected_policy)

    processed = _read_checkpoint(checkpoint_path) if incremental else set()
    if incremental:
        processed.update(_existing_feature_session_ids(output_dir, source_name))
    event_session_ids = _event_session_ids(event_dir, source_name)
    pending_ids = event_session_ids - processed
    progress(
        f"Feature extraction pending sessions: source={source_name}, "
        f"event_sessions={len(event_session_ids):,}, "
        f"processed_sessions={len(processed):,}, pending_sessions={len(pending_ids):,}"
    )
    if not pending_ids:
        if incremental and any(output_dir.glob("part-*.parquet")):
            _write_checkpoint(checkpoint_path, processed)
            progress(f"No new collector sessions require feature extraction: {output_dir}")
            return output_dir
        if not incremental and success_path.exists():
            return output_dir
        raise ValueError(f"No unprocessed event sessions found for {source_name}")

    temp_dir = output_dir / ".feature-build-tmp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()
    try:
        _bucket_pending_events(
            event_dir,
            temp_dir,
            pending_ids,
            _required_event_columns(config),
            source_name,
        )
        next_part = _next_part_number(output_dir)
        total_rows = 0
        completed_ids: set[str] = set()
        for bucket_id in range(FEATURE_BUCKETS):
            bucket_dir = temp_dir / f"bucket-{bucket_id:04d}"
            parts = sorted(bucket_dir.glob("part-*.parquet"))
            if not parts:
                continue
            progress(
                f"Feature bucket starting: source={source_name}, "
                f"bucket={bucket_id + 1}/{FEATURE_BUCKETS}, input_parts={len(parts):,}"
            )
            events = pd.concat(
                (pd.read_parquet(part) for part in parts),
                ignore_index=True,
            ).sort_values(["session_id", "timestamp_ms"])
            bucket_ids = set(events["session_id"].astype(str))
            features = feature_frame(
                events,
                config,
                source_name,
                report_progress=True,
            )
            if not features.empty:
                output_path = output_dir / f"part-{next_part:06d}.parquet"
                features.to_parquet(output_path, index=False)
                next_part += 1
                total_rows += len(features)
            completed_ids.update(bucket_ids)
            if incremental:
                processed.update(bucket_ids)
                _write_checkpoint(checkpoint_path, processed)
            progress(
                f"Feature bucket complete: source={source_name}, "
                f"bucket={bucket_id + 1}/{FEATURE_BUCKETS}, rows={len(features):,}"
            )

        if not completed_ids:
            raise ValueError(f"No event sessions could be processed for {source_name}")
        if not any(output_dir.glob("part-*.parquet")):
            raise ValueError(f"No feature rows produced for {source_name}")
        if not incremental:
            success_path.write_text("", encoding="utf-8")
            final_output_dir.parent.mkdir(parents=True, exist_ok=True)
            output_dir.replace(final_output_dir)
        progress(
            f"Feature extraction complete: source={source_name}, "
            f"new_rows={total_rows:,}, sessions={len(completed_ids):,}, output={final_output_dir}"
        )
        return final_output_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        if building_dir is not None and building_dir.exists():
            shutil.rmtree(building_dir)


def feature_frame(
    frame: pd.DataFrame,
    config: PipelineConfig,
    source_name: str,
    *,
    report_progress: bool = False,
) -> pd.DataFrame:
    available_sources = {config.trial.train_dataset, "collector"}
    if source_name not in available_sources:
        raise ValueError(
            f"Unsupported feature source '{source_name}'. "
            f"Available: {', '.join(sorted(available_sources))}"
        )
    domains = set(frame["domain"].dropna().astype(str))
    if domains and domains != {source_name}:
        raise ValueError(
            f"Feature source '{source_name}' received domain(s): "
            + ", ".join(sorted(domains))
        )
    feature_config = config.trial.features
    context_seconds = feature_config.context_windows_seconds
    if source_name == "collector":
        minimum_samples_by_context = {
            context: config.trial.minimum_samples_for_window(
                config.collector_minimum_sampling_rate,
                context,
            )
            for context in context_seconds
        }
        maximum_sample_interval_ms = (
            config.dataset.collector_max_sample_interval_ms
        )
    else:
        minimum_samples_by_context = {
            context: {sensor: 1 for sensor in feature_config.sensors}
            for context in context_seconds
        }
        maximum_sample_interval_ms = None
    window_ms = feature_config.default_window_seconds * 1000
    step_ms = feature_config.default_step_seconds * 1000
    rows: list[dict[str, Any]] = []

    session_groups = frame.groupby("session_id", sort=False)
    session_count = session_groups.ngroups
    for session_index, (_, session) in enumerate(session_groups, start=1):
        if report_progress:
            progress(
                f"Feature session starting: source={source_name}, "
                f"session={session_index:,}/{session_count:,}, event_rows={len(session):,}"
            )
        session = session.sort_values("timestamp_ms")
        start = int(session["timestamp_ms"].min())
        end = int(session["timestamp_ms"].max())
        duration = (end - start) / 1000.0
        if not (
            config.dataset.minimum_trip_seconds
            <= duration
            <= config.dataset.maximum_trip_seconds
        ):
            if report_progress:
                progress(
                    f"Feature session complete: source={source_name}, "
                    f"session={session_index:,}/{session_count:,}, "
                    "status=duration-filtered, windows=0, kept=0"
                )
            continue
        baselines = _session_baselines(
            session,
            set(feature_config.sensors),
        )
        timestamps = session["timestamp_ms"].to_numpy(dtype=np.int64)
        positive_diffs = np.diff(timestamps)
        positive_diffs = positive_diffs[positive_diffs > 0]
        sample_interval = (
            int(np.median(positive_diffs)) if positive_diffs.size else step_ms
        )
        exclusive_end = end + sample_interval
        cursor = start
        session_windows = 0
        session_rows_before = len(rows)
        last_heartbeat = time.monotonic()
        while cursor + window_ms <= exclusive_end:
            session_windows += 1
            window = session[
                (session["timestamp_ms"] >= cursor)
                & (session["timestamp_ms"] < cursor + window_ms)
            ]
            if all(
                _meets_window_quality_requirements(
                    window.loc[
                        window["timestamp_ms"] >= cursor + window_ms - context * 1000
                    ],
                    minimum_samples_by_context[context],
                    maximum_sample_interval_ms,
                    cursor + window_ms - context * 1000,
                    cursor + window_ms,
                )
                for context in context_seconds
            ):
                rows.append(
                    _window_features(
                        window,
                        config,
                        baselines,
                        cursor,
                        cursor + window_ms,
                    )
                )
            cursor += step_ms
            if (
                report_progress
                and time.monotonic() - last_heartbeat
                >= FEATURE_SESSION_HEARTBEAT_SECONDS
            ):
                progress(
                    f"Feature session progress: source={source_name}, "
                    f"session={session_index:,}/{session_count:,}, "
                    f"windows={session_windows:,}, "
                    f"kept={len(rows) - session_rows_before:,}"
                )
                last_heartbeat = time.monotonic()
        if report_progress:
            progress(
                f"Feature session complete: source={source_name}, "
                f"session={session_index:,}/{session_count:,}, status=processed, "
                f"windows={session_windows:,}, kept={len(rows) - session_rows_before:,}"
            )
    return pd.DataFrame(rows)


def _session_baselines(
    session: pd.DataFrame,
    configured_sensors: set[str],
) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for sensor in configured_sensors.intersection({"pressure", "magnetometer"}):
        values = sensor_series(sensor, session)
        if values.size:
            baselines[sensor] = (
                float(values[0])
                if sensor == "pressure"
                else float(np.median(values))
            )
    return baselines


def _window_features(
    window: pd.DataFrame,
    config: PipelineConfig,
    baselines: dict[str, float],
    window_start_ms: int,
    window_end_ms: int,
) -> dict[str, Any]:
    first = window.iloc[0]
    row: dict[str, Any] = {
        column: first[column]
        for column in METADATA_COLUMNS
        if column not in {"window_start_ms", "window_end_ms"}
    }
    row["label"] = int(first["label"])
    row["window_start_ms"] = int(window_start_ms)
    row["window_end_ms"] = int(window_end_ms)
    contexts = config.trial.features.context_windows_seconds
    single_default_context = contexts == (
        config.trial.features.default_window_seconds,
    )
    for context in contexts:
        context_rows = window.loc[
            window["timestamp_ms"] >= window_end_ms - context * 1000
        ]
        for sensor, aggregations in config.trial.features.sensors.items():
            values = ordered_sensor_features(
                sensor,
                context_rows,
                aggregations,
                session_baseline=baselines.get(sensor),
            )
            for aggregation, value in zip(aggregations, values):
                name = (
                    f"{sensor}#{aggregation}"
                    if single_default_context
                    else f"{sensor}#{aggregation}@{context}s"
                )
                row[name] = np.float32(value)
    return row


def _meets_window_quality_requirements(
    window: pd.DataFrame,
    minimum_samples: dict[str, int],
    maximum_sample_interval_ms: int | None,
    window_start_ms: int,
    window_end_ms: int,
) -> bool:
    for sensor, required in minimum_samples.items():
        if sensor == "pressure":
            valid = window["p"].notna()
        else:
            valid = window.loc[:, VECTOR_SENSOR_COLUMNS[sensor]].notna().all(axis=1)
        timestamps = np.sort(
            window.loc[valid, "timestamp_ms"].to_numpy(dtype=np.int64)
        )
        if timestamps.size < required:
            return False
        if maximum_sample_interval_ms is not None:
            bounded_timestamps = np.concatenate(
                (
                    np.array([window_start_ms], dtype=np.int64),
                    timestamps,
                    np.array([window_end_ms], dtype=np.int64),
                )
            )
            if (
                np.diff(bounded_timestamps).max(initial=0)
                > maximum_sample_interval_ms
            ):
                return False
    return True


def _feature_policy(
    config: PipelineConfig,
    source_name: str,
) -> dict[str, Any]:
    policy: dict[str, Any] = {
        "schema_version": FEATURE_POLICY_SCHEMA_VERSION,
        "source_name": source_name,
        "ingestion_hash": config.ingestion_hash,
        "features": config.trial.raw["features"],
        "event_artifact_digest": _event_artifact_digest(
            config.run_dir() / "events" / source_name
        ),
        "minimum_trip_seconds": config.dataset.minimum_trip_seconds,
        "maximum_trip_seconds": config.dataset.maximum_trip_seconds,
    }
    if source_name == "collector":
        policy["collector_sampling_quality"] = {
            "collector_minimum_sampling_rate": {
                sensor: config.collector_minimum_sampling_rate[sensor]
                for sensor in sorted(config.trial.features.sensors)
            },
            "collector_max_sample_interval_ms": (
                config.dataset.collector_max_sample_interval_ms
            ),
        }
    return policy


def _feature_policy_matches(path: Path, expected: dict[str, Any]) -> bool:
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def _write_feature_policy(path: Path, policy: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(policy, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _required_event_columns(config: PipelineConfig) -> list[str]:
    columns = {
        "domain",
        "participant_id",
        "device_id",
        "session_id",
        "trip_id",
        "group_id",
        "vehicle_type",
        "label",
        "phone_position",
        "timestamp_ms",
    }
    for sensor in config.trial.features.sensors:
        if sensor == "pressure":
            columns.add("p")
        else:
            columns.update(VECTOR_SENSOR_COLUMNS[sensor])
    return sorted(columns)


def _bucket_pending_events(
    event_dir: Path,
    temp_dir: Path,
    pending_ids: set[str],
    columns: list[str],
    source_name: str,
) -> None:
    parts = sorted(event_dir.glob("part-*.parquet"))
    bucketed_rows = 0
    progress(
        f"Feature event bucketing starting: source={source_name}, "
        f"input_parts={len(parts):,}, pending_sessions={len(pending_ids):,}"
    )
    for part_number, part in enumerate(parts):
        progress(
            f"Feature event part starting: source={source_name}, "
            f"part={part_number + 1:,}/{len(parts):,}, path={part}"
        )
        frame = pd.read_parquet(part, columns=columns)
        frame = frame[frame["session_id"].astype(str).isin(pending_ids)]
        if frame.empty:
            progress(
                f"Feature event part complete: source={source_name}, "
                f"part={part_number + 1:,}/{len(parts):,}, pending_rows=0, "
                f"bucketed_rows={bucketed_rows:,}"
            )
            continue
        bucketed_rows += len(frame)
        bucket_ids = (
            pd.util.hash_pandas_object(frame["session_id"].astype("string"), index=False)
            % FEATURE_BUCKETS
        ).astype("int64")
        for bucket_id, bucket in frame.groupby(bucket_ids, sort=False):
            bucket_dir = temp_dir / f"bucket-{int(bucket_id):04d}"
            bucket_dir.mkdir(exist_ok=True)
            bucket.to_parquet(
                bucket_dir / f"part-{part_number:06d}.parquet",
                index=False,
            )
        progress(
            f"Feature event part complete: source={source_name}, "
            f"part={part_number + 1:,}/{len(parts):,}, "
            f"pending_rows={len(frame):,}, bucketed_rows={bucketed_rows:,}"
        )
    progress(
        f"Feature event bucketing complete: source={source_name}, "
        f"input_parts={len(parts):,}, bucketed_rows={bucketed_rows:,}"
    )


def _event_session_ids(event_dir: Path, source_name: str) -> set[str]:
    result: set[str] = set()
    parts = sorted(event_dir.glob("part-*.parquet"))
    progress(
        f"Feature event session scan starting: source={source_name}, "
        f"input_parts={len(parts):,}"
    )
    for part_index, part in enumerate(parts, start=1):
        progress(
            f"Feature event session scan part starting: source={source_name}, "
            f"part={part_index:,}/{len(parts):,}, path={part}"
        )
        frame = pd.read_parquet(part, columns=["session_id"])
        result.update(frame["session_id"].dropna().astype(str))
        progress(
            f"Feature event session scan part complete: source={source_name}, "
            f"part={part_index:,}/{len(parts):,}, sessions={len(result):,}"
        )
    progress(
        f"Feature event session scan complete: source={source_name}, "
        f"input_parts={len(parts):,}, sessions={len(result):,}"
    )
    return result


def _existing_feature_session_ids(output_dir: Path, source_name: str) -> set[str]:
    result: set[str] = set()
    parts = sorted(output_dir.glob("part-*.parquet"))
    if parts:
        progress(
            f"Feature checkpoint reconciliation starting: source={source_name}, "
            f"feature_parts={len(parts):,}"
        )
    for part_index, part in enumerate(parts, start=1):
        frame = pd.read_parquet(part, columns=["session_id"])
        result.update(frame["session_id"].dropna().astype(str))
        progress(
            f"Feature checkpoint reconciliation progress: source={source_name}, "
            f"part={part_index:,}/{len(parts):,}, sessions={len(result):,}"
        )
    if parts:
        progress(
            f"Feature checkpoint reconciliation complete: source={source_name}, "
            f"feature_parts={len(parts):,}, sessions={len(result):,}"
        )
    return result


def _read_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise ValueError(f"Invalid feature checkpoint: {path}")
    return set(values)


def _write_checkpoint(path: Path, values: set[str]) -> None:
    path.write_text(json.dumps(sorted(values), indent=2) + "\n", encoding="utf-8")


def _next_part_number(output_dir: Path) -> int:
    values: list[int] = []
    for part in output_dir.glob("part-*.parquet"):
        try:
            values.append(int(part.stem.removeprefix("part-")))
        except ValueError:
            continue
    return max(values, default=-1) + 1
