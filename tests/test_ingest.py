from __future__ import annotations

import errno
import json
from pathlib import Path
import shutil

import pandas as pd

from all_tmd.ingest import (
    NORTMDAdapter,
    TrainingDatasetAdapter,
    ingest_collector,
    ingest_training_dataset,
    normalize_nor_frame,
    normalize_us_frame,
    _duration_filter,
)


def test_training_adapters_are_registered():
    assert set(TrainingDatasetAdapter._registry) >= {"us-tmd", "nor-tmd"}


def test_collector_duration_filter_has_no_upper_limit_when_disabled(config_factory):
    frame = pd.DataFrame(
        {
            "session_id": ["long", "long"],
            "timestamp_ms": [0, 10_800_000],
        }
    )
    unlimited = config_factory(maximum_trip_seconds=None)
    capped = config_factory(maximum_trip_seconds=7200)

    assert _duration_filter(frame, unlimited).empty is False
    assert _duration_filter(frame, capped).empty is True


def test_us_normalization_uses_common_domain():
    raw = pd.DataFrame(
        [
            [1000, "android.sensor.accelerometer", 1, 2, 3],
            [1001, "android.sensor.pressure", 1000, None, None],
        ],
        columns=["time", "sensor_type", "value_0", "value_1", "value_2"],
    )
    result = normalize_us_frame(
        raw,
        Path("sensorfile_U1_car_123.csv"),
        {"car": 1},
    )
    assert result["domain"].unique().tolist() == ["us-tmd"]
    assert result["session_id"].unique().tolist() == ["sensorfile_U1_car_123"]
    assert result["label"].unique().tolist() == [1]


def test_nor_normalization_uses_common_domain():
    raw = pd.DataFrame(
        [
            {
                "installationId": "p1",
                "journeyNumber": 1,
                "tripNumber": 2,
                "manufacturer": "m",
                "model": "x",
                "timestamp": "2024-01-01T00:00:00Z",
                "transportType": "CAR",
                "typeString": "android.sensor.accelerometer",
                "deviceLocation": "pocket",
                "value_0": 1,
                "value_1": 2,
                "value_2": 3,
                "OS": "ANDROID",
            }
        ]
    )
    result = normalize_nor_frame(raw, {"car": 1})
    assert result["domain"].tolist() == ["nor-tmd"]
    assert result["session_id"].tolist() == ["p1#1#2"]


def test_nor_adapter_accepts_one_csv_or_a_directory(config_factory, tmp_path):
    config = config_factory(train_dataset="nor-tmd")
    directory_adapter = NORTMDAdapter(config.training_source)
    assert directory_adapter.source.input_path == tmp_path / "data" / "nor"


def test_collector_ingest_uses_checkpoint_and_reports_progress(
    config_factory,
    monkeypatch,
):
    messages = []
    monkeypatch.setattr("all_tmd.ingest.progress", messages.append)
    config = config_factory(collector_max_sample_interval_ms=500)
    input_dir = config.sources.collector.input_path
    input_dir.mkdir(parents=True)
    _write_collector(input_dir / "one.json", "one")
    output = ingest_collector(config)
    first_parts = sorted(output.glob("part-*.parquet"))
    assert len(first_parts) == 1
    assert json.loads((output / "checkpoint.json").read_text()) == ["one"]

    ingest_collector(config)
    assert sorted(output.glob("part-*.parquet")) == first_parts

    _write_collector(input_dir / "two.json", "two")
    ingest_collector(config)
    assert len(list(output.glob("part-*.parquet"))) == 2
    assert json.loads((output / "checkpoint.json").read_text()) == ["one", "two"]
    assert any("Collector ingest scan starting" in message for message in messages)
    assert any(
        "Collector checkpoint reconciliation" in message
        for message in messages
    )
    assert any("status=ingested" in message for message in messages)
    assert any("status=already-processed" in message for message in messages)
    assert any("Collector ingest complete" in message for message in messages)


def test_collector_ingest_keeps_session_with_sample_gap(config_factory):
    config = config_factory(collector_max_sample_interval_ms=500)
    input_dir = config.sources.collector.input_path
    input_dir.mkdir(parents=True)
    _write_collector(input_dir / "gapped.json", "gapped")

    output = ingest_collector(config)

    events = pd.concat(
        (pd.read_parquet(path) for path in output.glob("part-*.parquet")),
        ignore_index=True,
    )
    assert events["session_id"].unique().tolist() == ["gapped"]


def test_training_ingest_writes_success_and_then_skips(config_factory):
    config = config_factory()
    input_dir = config.training_source.input_path
    input_dir.mkdir(parents=True)
    csv_path = input_dir / "sensorfile_U1_car_123.csv"
    csv_path.write_text(
        "0,android.sensor.accelerometer,1,2,3\n"
        "1000,android.sensor.accelerometer,2,3,4\n",
        encoding="utf-8",
    )
    output = ingest_training_dataset(config)
    first_parts = sorted(output.glob("part-*.parquet"))
    assert first_parts
    assert (output / "_SUCCESS").exists()

    csv_path.write_text("", encoding="utf-8")
    assert ingest_training_dataset(config) == output
    assert sorted(output.glob("part-*.parquet")) == first_parts


def test_training_ingest_retries_incomplete_directory_cleanup(
    config_factory,
    monkeypatch,
):
    config = config_factory()
    input_dir = config.training_source.input_path
    input_dir.mkdir(parents=True)
    (input_dir / "sensorfile_U1_car_123.csv").write_text(
        "0,android.sensor.accelerometer,1,2,3\n",
        encoding="utf-8",
    )
    output_dir = config.run_dir() / "events" / "us-tmd"
    output_dir.mkdir(parents=True)
    (output_dir / "stale.parquet").write_text("incomplete", encoding="utf-8")

    real_rmtree = shutil.rmtree
    calls = 0

    def transiently_busy_rmtree(path):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError(errno.ENOTEMPTY, "Directory not empty", path)
        real_rmtree(path)

    monkeypatch.setattr("all_tmd.ingest.shutil.rmtree", transiently_busy_rmtree)
    monkeypatch.setattr("all_tmd.ingest.time.sleep", lambda _seconds: None)

    output = ingest_training_dataset(config)

    assert calls == 2
    assert not (output / "stale.parquet").exists()
    assert (output / "_SUCCESS").exists()


def _write_collector(path: Path, session_id: str) -> None:
    samples = [
        {"ts": 0, "ax": 1, "ay": 2, "az": 3},
        {"ts": 1000, "ax": 2, "ay": 3, "az": 4},
    ]
    path.write_text(
        json.dumps(
            {
                "session_id": session_id,
                "participant_id": f"p-{session_id}",
                "device_uuid": "device",
                "vehicle_type": "car",
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
