from __future__ import annotations

from abc import ABC, abstractmethod
import csv
import errno
import gzip
import json
from pathlib import Path
import re
import shutil
import time
from typing import Any, ClassVar, Iterable, Iterator
from uuid import uuid4

import numpy as np
import pandas as pd

from all_tmd.config import PipelineConfig, SourceConfig
from all_tmd.progress import progress


EVENT_COLUMNS = [
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
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
    "mx",
    "my",
    "mz",
    "p",
]
SENSOR_COLUMNS = ("ax", "ay", "az", "gx", "gy", "gz", "mx", "my", "mz", "p")
US_RAW_COLUMNS = ["time", "sensor_type", "value_0", "value_1", "value_2"]
NOR_RAW_COLUMNS = [
    "installationId",
    "journeyNumber",
    "tripNumber",
    "manufacturer",
    "model",
    "timestamp",
    "transportType",
    "typeString",
    "deviceLocation",
    "value_0",
    "value_1",
    "value_2",
    "OS",
]
GRAVITY_METRES_PER_SECOND_SQUARED = 9.80665
SENSOR_TYPE_PATTERN = re.compile(r"[^a-zA-Z0-9._-]")
TIMESTAMP_PATTERN = re.compile(r"\d+")
COLLECTOR_CHECKPOINT = "checkpoint.json"
INCOMPLETE_DATASET_REMOVE_ATTEMPTS = 6
INCOMPLETE_DATASET_REMOVE_DELAY_SECONDS = 0.05


class TrainingDatasetAdapter(ABC):
    """Extension point for immutable source datasets.

    A new source only needs a concrete subclass with a unique ``dataset_name``
    and an implementation of ``normalized_frames``.
    """

    dataset_name: ClassVar[str]
    _registry: ClassVar[dict[str, type["TrainingDatasetAdapter"]]] = {}

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "dataset_name", "")
        if name:
            if name in cls._registry:
                raise ValueError(f"Duplicate training dataset adapter: {name}")
            cls._registry[name] = cls

    def __init__(self, source: SourceConfig):
        self.source = source

    @classmethod
    def for_config(cls, config: PipelineConfig) -> "TrainingDatasetAdapter":
        name = config.trial.train_dataset
        try:
            adapter_type = cls._registry[name]
        except KeyError as exc:
            available = ", ".join(sorted(cls._registry))
            raise ValueError(
                f"No ingestion adapter registered for '{name}'. Available: {available}"
            ) from exc
        return adapter_type(config.training_source)

    @abstractmethod
    def normalized_frames(
        self,
        labels: dict[str, int],
    ) -> Iterator[pd.DataFrame]:
        """Yield normalized frames in the common event schema."""


class USTMDAdapter(TrainingDatasetAdapter):
    dataset_name = "us-tmd"

    def normalized_frames(self, labels: dict[str, int]) -> Iterator[pd.DataFrame]:
        chunk_rows = self.source.chunk_rows or 50_000
        files = _data_files(self.source.input_path, self.source.include_globs)
        if not files:
            raise FileNotFoundError(
                f"No US-TMD CSV files found under {self.source.input_path}"
            )
        for file_index, path in enumerate(files, start=1):
            for chunk in _read_us_csv_chunks(path, chunk_rows):
                frame = normalize_us_frame(chunk, path, labels)
                if not frame.empty:
                    yield frame
            progress(f"US-TMD ingest file {file_index:,}/{len(files):,}: {path}")


class NORTMDAdapter(TrainingDatasetAdapter):
    dataset_name = "nor-tmd"

    def normalized_frames(self, labels: dict[str, int]) -> Iterator[pd.DataFrame]:
        chunk_rows = self.source.chunk_rows or 50_000
        files = _data_files(self.source.input_path, self.source.include_globs)
        if not files:
            raise FileNotFoundError(
                f"No NOR-TMD CSV files found at or under {self.source.input_path}"
            )
        for file_index, path in enumerate(files, start=1):
            for chunk in pd.read_csv(
                path,
                chunksize=chunk_rows,
                usecols=NOR_RAW_COLUMNS,
                low_memory=False,
            ):
                frame = normalize_nor_frame(chunk, labels)
                if not frame.empty:
                    yield frame
            progress(f"NOR-TMD ingest file {file_index:,}/{len(files):,}: {path}")


def ingest_training_dataset(config: PipelineConfig) -> Path:
    adapter = TrainingDatasetAdapter.for_config(config)
    output_dir = config.run_dir() / "events" / adapter.dataset_name
    success_path = output_dir / "_SUCCESS"
    if success_path.exists():
        progress(f"Training events already complete: {output_dir}")
        return output_dir
    if output_dir.exists():
        progress(f"Removing incomplete training event dataset: {output_dir}")
        _remove_incomplete_dataset(output_dir)
    temp_dir = output_dir.with_name(f".{output_dir.name}.building-{uuid4().hex}")
    temp_dir.mkdir(parents=True)

    try:
        rows = 0
        parts = 0
        for frame in adapter.normalized_frames(config.trial.labels):
            if frame.empty:
                continue
            part_path = temp_dir / f"part-{parts:06d}.parquet"
            frame.to_parquet(part_path, index=False)
            rows += len(frame)
            parts += 1
            progress(
                f"Training ingest progress: dataset={adapter.dataset_name}, "
                f"rows={rows:,}, parts={parts:,}"
            )
        if parts == 0:
            raise ValueError(
                f"{adapter.dataset_name} ingestion produced no rows for configured labels"
            )
        (temp_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "adapter": adapter.dataset_name,
                    "adapter_schema_version": 1,
                    "ingestion_hash": config.ingestion_hash,
                    "rows": rows,
                    "parts": parts,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (temp_dir / "_SUCCESS").write_text("", encoding="utf-8")
        temp_dir.replace(output_dir)
        progress(
            f"Training ingest complete: dataset={adapter.dataset_name}, "
            f"rows={rows:,}, parts={parts:,}, output={output_dir}"
        )
        return output_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def _remove_incomplete_dataset(path: Path) -> None:
    for attempt in range(INCOMPLETE_DATASET_REMOVE_ATTEMPTS):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            if not path.exists():
                return
            raise
        except OSError as error:
            if (
                error.errno != errno.ENOTEMPTY
                or attempt == INCOMPLETE_DATASET_REMOVE_ATTEMPTS - 1
            ):
                raise
            delay = INCOMPLETE_DATASET_REMOVE_DELAY_SECONDS * (2**attempt)
            progress(
                f"Incomplete dataset directory remained non-empty; retrying in "
                f"{delay:.2f}s: {path}"
            )
            time.sleep(delay)


def ingest_collector(config: PipelineConfig) -> Path:
    if config.dataset.snapshot_manifest is not None:
        return _ingest_frozen_collector(config)
    output_dir = config.run_dir() / "events" / "collector"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / COLLECTOR_CHECKPOINT
    processed = _read_checkpoint(checkpoint_path)
    # Reconcile an interrupted run where a part was committed before checkpoint.
    processed.update(_existing_session_ids(output_dir))

    source = config.sources.collector
    files = list(_collector_session_files(source.input_path, source.include_globs))
    next_part = _next_part_number(output_dir)
    new_parts = 0
    new_rows = 0
    already_processed_files = 0
    ignored_files = 0
    duration_filtered_files = 0
    progress(
        f"Collector ingest scan starting: files={len(files):,}, "
        f"checkpoint_sessions={len(processed):,}"
    )
    for index, path in enumerate(files, start=1):
        frame = normalize_collector_payload(path, config.trial.labels)
        if frame.empty:
            ignored_files += 1
            status = "ignored"
        else:
            frame = frame[~frame["session_id"].astype(str).isin(processed)].copy()
            if frame.empty:
                already_processed_files += 1
                status = "already-processed"
            else:
                frame = _duration_filter(frame, config)
                if frame.empty:
                    duration_filtered_files += 1
                    status = "duration-filtered"
                else:
                    part_path = output_dir / f"part-{next_part:06d}.parquet"
                    frame.to_parquet(part_path, index=False)
                    ingested_ids = set(frame["session_id"].astype(str))
                    processed.update(ingested_ids)
                    _write_checkpoint(checkpoint_path, processed)
                    next_part += 1
                    new_parts += 1
                    new_rows += len(frame)
                    status = "ingested"
        progress(
            f"Collector ingest file {index:,}/{len(files):,}: status={status}, "
            f"new_rows={new_rows:,}, new_parts={new_parts:,}, "
            f"already_processed={already_processed_files:,}, "
            f"ignored={ignored_files:,}, duration_filtered={duration_filtered_files:,}, "
            f"path={path}"
        )

    if not any(output_dir.glob("part-*.parquet")):
        raise ValueError("Collector ingestion produced no rows for configured labels")
    _write_checkpoint(checkpoint_path, processed)
    progress(
        f"Collector ingest complete: new_rows={new_rows:,}, "
        f"new_parts={new_parts:,}, processed_sessions={len(processed):,}, "
        f"already_processed_files={already_processed_files:,}, "
        f"ignored_files={ignored_files:,}, "
        f"duration_filtered_files={duration_filtered_files:,}, output={output_dir}"
    )
    return output_dir


def _ingest_frozen_collector(config: PipelineConfig) -> Path:
    from all_tmd.study import verify_snapshot, verify_snapshot_files

    manifest_path = config.dataset.snapshot_manifest
    assert manifest_path is not None
    summary = verify_snapshot_files(manifest_path, config.sources.collector.input_path)
    eligible = set(summary["eligible_session_ids"])
    output_dir = config.run_dir() / "events" / "collector"
    if (output_dir / "_SUCCESS").exists():
        progress(f"Frozen collector events already complete: {output_dir}")
        return output_dir
    if output_dir.exists():
        _remove_incomplete_dataset(output_dir)
    temp_dir = output_dir.with_name(f".{output_dir.name}.building-{uuid4().hex}")
    temp_dir.mkdir(parents=True)
    try:
        source = config.sources.collector
        files = [
            path
            for path in _collector_session_files(source.input_path, source.include_globs)
            if path.name.removesuffix(".gz").removesuffix(".json") in eligible
        ]
        written: set[str] = set()
        duration_filtered: set[str] = set()
        rows = 0
        for part, path in enumerate(files):
            frame = normalize_collector_payload(path, config.trial.labels)
            if frame.empty:
                continue
            normalized_ids = set(frame["session_id"].astype(str))
            frame = _duration_filter(frame, config)
            if frame.empty:
                duration_filtered.update(normalized_ids)
                continue
            session_ids = set(frame["session_id"].astype(str))
            if not session_ids.issubset(eligible):
                raise ValueError(f"collector payload is outside the frozen snapshot: {path}")
            frame.to_parquet(temp_dir / f"part-{part:06d}.parquet", index=False)
            written.update(session_ids)
            rows += len(frame)
        missing = eligible - written - duration_filtered
        if missing:
            raise ValueError(
                f"{len(missing)} frozen Sydney sessions produced no eligible events: {sorted(missing)[:5]}"
            )
        (temp_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "snapshot_digest": summary["snapshot_digest"],
                    "session_count": len(written),
                    "duration_filtered_session_count": len(duration_filtered),
                    "rows": rows,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        (temp_dir / "_SUCCESS").write_text("", encoding="utf-8")
        temp_dir.replace(output_dir)
        progress(f"Frozen collector ingest complete: sessions={len(written)}, rows={rows:,}")
        return output_dir
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def normalize_us_frame(
    frame: pd.DataFrame,
    path: Path,
    labels: dict[str, int],
) -> pd.DataFrame:
    vehicle_type = _transport_mode_from_path(path)
    if vehicle_type not in labels:
        return _empty_events()
    raw = frame.loc[:, US_RAW_COLUMNS].copy()
    raw["sensor"] = raw["sensor_type"].astype("string").map(_canonical_us_sensor)
    raw = raw[raw["sensor"].notna()]
    raw["timestamp_ms"] = raw["time"].astype("string").map(_parse_us_timestamp)
    for column in ("value_0", "value_1", "value_2"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(subset=["timestamp_ms", "sensor", "value_0"])
    raw = raw[
        (raw["sensor"] == "pressure")
        | raw[["value_1", "value_2"]].notna().all(axis=1)
    ]
    if raw.empty:
        return _empty_events()

    participant_id = _participant_id_from_path(path)
    session_id = path.stem
    out = _event_metadata_frame(
        raw.index,
        domain="us-tmd",
        participant_id=participant_id,
        device_id=participant_id,
        session_id=session_id,
        trip_id=session_id,
        vehicle_type=vehicle_type,
        label=labels[vehicle_type],
        phone_position="unknown",
        timestamp_ms=raw["timestamp_ms"],
    )
    _assign_sensor_columns(out, raw)
    return _coerce_event_frame(out)


def normalize_nor_frame(frame: pd.DataFrame, labels: dict[str, int]) -> pd.DataFrame:
    missing = sorted(set(NOR_RAW_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError("NOR-TMD CSV is missing columns: " + ", ".join(missing))
    raw = frame.loc[:, NOR_RAW_COLUMNS].copy()
    raw["transportType"] = raw["transportType"].astype("string").str.upper()
    raw = raw[raw["transportType"].isin({mode.upper() for mode in labels})]
    raw["sensor"] = raw["typeString"].astype("string").map(_canonical_nor_sensor)
    raw = raw[raw["sensor"].notna()]
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], errors="coerce", utc=True)
    for column in ("value_0", "value_1", "value_2"):
        raw[column] = pd.to_numeric(raw[column], errors="coerce")
    raw = raw.dropna(
        subset=["installationId", "timestamp", "transportType", "sensor", "value_0"]
    )
    platform = raw["OS"].astype("string").str.upper()
    ios_acceleration = (platform == "IOS") & (raw["sensor"] == "accelerometer")
    raw.loc[ios_acceleration, ["value_0", "value_1", "value_2"]] *= (
        GRAVITY_METRES_PER_SECOND_SQUARED
    )
    raw.loc[(platform == "IOS") & (raw["sensor"] == "pressure"), "value_0"] *= 10.0
    raw = raw[
        (raw["sensor"] == "pressure")
        | raw[["value_1", "value_2"]].notna().all(axis=1)
    ]
    if raw.empty:
        return _empty_events()

    participant = raw["installationId"].astype("string")
    trip_id = (
        raw["journeyNumber"].astype("string")
        + "#"
        + raw["tripNumber"].astype("string")
    )
    session_id = participant + "#" + trip_id
    device = (
        platform
        + ":"
        + raw["manufacturer"].astype("string").fillna("unknown")
        + ":"
        + raw["model"].astype("string").fillna("unknown")
    ).str.lower()
    vehicle = raw["transportType"].astype("string").str.lower()
    out = _event_metadata_frame(
        raw.index,
        domain="nor-tmd",
        participant_id=participant,
        device_id=device,
        session_id=session_id,
        trip_id=trip_id,
        vehicle_type=vehicle,
        label=vehicle.map(labels),
        phone_position=raw["deviceLocation"].astype("string").str.lower(),
        timestamp_ms=raw["timestamp"].astype("int64") // 1_000_000,
    )
    _assign_sensor_columns(out, raw)
    return _coerce_event_frame(out)


def normalize_collector_payload(path: Path, labels: dict[str, int]) -> pd.DataFrame:
    payload = _load_json(path)
    metadata = _load_sidecar_metadata(path)
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        return _empty_events()
    vehicle_type = str(
        payload.get("vehicle_type") or metadata.get("vehicle_type") or ""
    ).lower()
    if vehicle_type not in labels:
        return _empty_events()
    frame = pd.DataFrame(samples)
    if "ts" not in frame:
        raise ValueError(f"Collector sample file lacks ts column: {path}")
    session_id = str(
        payload.get("session_id") or metadata.get("session_id") or path.stem
    )
    participant_id = str(
        metadata.get("participant_id")
        or payload.get("participant_id")
        or "unknown_participant"
    )
    device_id = str(
        payload.get("device_uuid")
        or metadata.get("device_uuid")
        or metadata.get("device_id")
        or "unknown_device"
    )
    out = _event_metadata_frame(
        frame.index,
        domain="collector",
        participant_id=participant_id,
        device_id=device_id,
        session_id=session_id,
        trip_id=session_id,
        vehicle_type=vehicle_type,
        label=labels[vehicle_type],
        phone_position=str(
            payload.get("phone_position")
            or metadata.get("phone_position")
            or "unknown"
        ),
        timestamp_ms=pd.to_numeric(frame["ts"], errors="coerce"),
    )
    for column in SENSOR_COLUMNS:
        out[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            if column in frame
            else np.nan
        )
    trimmed_start = payload.get("trimmed_start_ms") or metadata.get("trimmed_start_ms")
    trimmed_end = payload.get("trimmed_end_ms") or metadata.get("trimmed_end_ms")
    if trimmed_start is not None and trimmed_end is not None:
        out = out[
            (out["timestamp_ms"] >= int(trimmed_start))
            & (out["timestamp_ms"] <= int(trimmed_end))
        ]
    return _coerce_event_frame(out)


def _event_metadata_frame(
    index,
    *,
    domain: str,
    participant_id,
    device_id,
    session_id,
    trip_id,
    vehicle_type,
    label,
    phone_position,
    timestamp_ms,
) -> pd.DataFrame:
    out = pd.DataFrame(index=index, columns=EVENT_COLUMNS)
    out["domain"] = domain
    out["participant_id"] = participant_id
    out["device_id"] = device_id
    out["session_id"] = session_id
    out["trip_id"] = trip_id
    out["vehicle_type"] = vehicle_type
    out["label"] = label
    out["phone_position"] = phone_position
    out["timestamp_ms"] = timestamp_ms
    out["group_id"] = make_group_id(out)
    return out


def _assign_sensor_columns(out: pd.DataFrame, raw: pd.DataFrame) -> None:
    for column in SENSOR_COLUMNS:
        out[column] = np.nan
    mappings = {
        "accelerometer": ("ax", "ay", "az"),
        "gyroscope": ("gx", "gy", "gz"),
        "magnetometer": ("mx", "my", "mz"),
    }
    for sensor, columns in mappings.items():
        mask = raw["sensor"] == sensor
        out.loc[mask, list(columns)] = raw.loc[
            mask, ["value_0", "value_1", "value_2"]
        ].to_numpy()
    pressure = raw["sensor"] == "pressure"
    out.loc[pressure, "p"] = raw.loc[pressure, "value_0"].to_numpy()


def make_group_id(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["participant_id"].astype(str)
        + "#"
        + frame["device_id"].astype(str)
        + "#"
        + frame["session_id"].astype(str)
    )


def _data_files(input_path: Path, include_globs: Iterable[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".csv" else []
    seen: set[Path] = set()
    files: list[Path] = []
    for pattern in include_globs:
        for path in input_path.glob(pattern):
            if path.is_file() and path.suffix.lower() == ".csv" and path not in seen:
                seen.add(path)
                files.append(path)
    return sorted(files)


def _read_us_csv_chunks(path: Path, chunk_rows: int) -> Iterator[pd.DataFrame]:
    rows: list[list[str | None]] = []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as stream:
        for raw_row in csv.reader(stream):
            row = list(raw_row[: len(US_RAW_COLUMNS)])
            row.extend([None] * (len(US_RAW_COLUMNS) - len(row)))
            rows.append(row)
            if len(rows) >= chunk_rows:
                yield pd.DataFrame(rows, columns=US_RAW_COLUMNS)
                rows = []
    if rows:
        yield pd.DataFrame(rows, columns=US_RAW_COLUMNS)


def _canonical_us_sensor(value: str) -> str | None:
    value = SENSOR_TYPE_PATTERN.sub("", str(value)).strip().lower().split(".")[-1]
    return {
        "accelerometer": "accelerometer",
        "gyroscope": "gyroscope",
        "magnetic_field": "magnetometer",
        "magnetometer": "magnetometer",
        "pressure": "pressure",
    }.get(value)


def _canonical_nor_sensor(value: str) -> str | None:
    value = str(value).strip().lower()
    if "pressure" in value or "altimeter" in value:
        return "pressure"
    return {
        "android.sensor.accelerometer": "accelerometer",
        "cmaccelerometerdata": "accelerometer",
        "android.sensor.gyroscope": "gyroscope",
        "cmgyrodata": "gyroscope",
        "android.sensor.magnetic_field": "magnetometer",
        "cmmagnetometerdata": "magnetometer",
    }.get(value)


def _parse_us_timestamp(value: str) -> int | None:
    match = TIMESTAMP_PATTERN.search(str(value))
    return int(match.group(0)) if match else None


def _transport_mode_from_path(path: Path) -> str:
    parts = path.name.split("_")
    return parts[2].lower() if len(parts) > 2 else ""


def _participant_id_from_path(path: Path) -> str:
    parts = path.name.split("_")
    return parts[1] if len(parts) > 1 else path.stem


def _collector_session_files(
    input_dir: Path,
    include_globs: Iterable[str],
) -> Iterator[Path]:
    seen: set[Path] = set()
    for pattern in include_globs:
        for path in sorted(input_dir.glob(pattern)):
            name = path.name.lower()
            if (
                path.is_file()
                and path not in seen
                and not name.endswith(".metadata.json")
                and not name.startswith(".")
            ):
                seen.add(path)
                yield path


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return json.load(stream)
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_sidecar_metadata(path: Path) -> dict[str, Any]:
    for candidate in (
        path.with_suffix(f"{path.suffix}.metadata.json"),
        path.with_suffix(".metadata.json"),
    ):
        if candidate.exists():
            with candidate.open("r", encoding="utf-8") as stream:
                return json.load(stream)
    return {}


def _read_checkpoint(path: Path) -> set[str]:
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(value, str) for value in data):
        raise ValueError(f"Invalid collector checkpoint: {path}")
    return set(data)


def _write_checkpoint(path: Path, session_ids: set[str]) -> None:
    path.write_text(
        json.dumps(sorted(session_ids), indent=2) + "\n",
        encoding="utf-8",
    )


def _existing_session_ids(output_dir: Path) -> set[str]:
    session_ids: set[str] = set()
    parts = sorted(output_dir.glob("part-*.parquet"))
    if parts:
        progress(
            f"Collector checkpoint reconciliation starting: "
            f"event_parts={len(parts):,}"
        )
    for part_index, part in enumerate(parts, start=1):
        frame = pd.read_parquet(part, columns=["session_id"])
        session_ids.update(frame["session_id"].dropna().astype(str))
        progress(
            f"Collector checkpoint reconciliation progress: "
            f"part={part_index:,}/{len(parts):,}, sessions={len(session_ids):,}"
        )
    if parts:
        progress(
            f"Collector checkpoint reconciliation complete: "
            f"event_parts={len(parts):,}, sessions={len(session_ids):,}"
        )
    return session_ids


def _next_part_number(output_dir: Path) -> int:
    numbers: list[int] = []
    for part in output_dir.glob("part-*.parquet"):
        try:
            numbers.append(int(part.stem.removeprefix("part-")))
        except ValueError:
            continue
    return max(numbers, default=-1) + 1


def _duration_filter(frame: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    keep: list[str] = []
    for session_id, group in frame.groupby("session_id", sort=False):
        duration = (
            group["timestamp_ms"].max() - group["timestamp_ms"].min()
        ) / 1000.0
        if duration >= config.dataset.minimum_trip_seconds and (
            config.dataset.maximum_trip_seconds is None
            or duration <= config.dataset.maximum_trip_seconds
        ):
            keep.append(str(session_id))
    return frame[frame["session_id"].astype(str).isin(keep)].reset_index(drop=True)


def _coerce_event_frame(frame: pd.DataFrame) -> pd.DataFrame:
    for column in ("timestamp_ms", *SENSOR_COLUMNS):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["timestamp_ms", "label", "group_id"]).copy()
    frame["timestamp_ms"] = frame["timestamp_ms"].astype("int64")
    frame["label"] = frame["label"].astype("int64")
    return (
        frame.loc[:, EVENT_COLUMNS]
        .sort_values(["session_id", "timestamp_ms"])
        .reset_index(drop=True)
    )


def _empty_events() -> pd.DataFrame:
    return pd.DataFrame(columns=EVENT_COLUMNS)
