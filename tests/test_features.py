from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from all_tmd.ingest import EVENT_COLUMNS
from all_tmd.windowing import build_features, feature_frame


def test_sampling_quality_is_collector_only(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"]},
        collector_minimum_sampling_rate={"accelerometer": 4},
        collector_max_sample_interval_ms=500,
    )
    source = _events("us-tmd", "source", [0])
    collector = _events("collector", "collector", [0])

    assert len(feature_frame(source, config, "us-tmd")) == 1
    assert feature_frame(collector, config, "collector").empty


def test_source_window_requires_one_value_for_each_sensor(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"], "pressure": ["range"]},
        collector_minimum_sampling_rate={"accelerometer": 4, "pressure": 2},
        collector_max_sample_interval_ms=500,
    )
    events = _events("us-tmd", "source", [0, 500])

    assert feature_frame(events, config, "us-tmd").empty


def test_feature_source_must_match_event_domain(config_factory):
    config = config_factory()
    events = _events("collector", "collector", [0])

    with pytest.raises(
        ValueError,
        match="Feature source 'us-tmd' received domain.*collector",
    ):
        feature_frame(events, config, "us-tmd")


def test_collector_window_rejects_internal_gap(config_factory):
    config = config_factory(
        collector_minimum_sampling_rate={"accelerometer": 4},
        collector_max_sample_interval_ms=500,
    )
    events = _events("collector", "gapped", [0, 100, 800, 900])

    assert feature_frame(events, config, "collector").empty


def test_window_accepts_gap_equal_to_threshold(config_factory):
    config = config_factory(
        collector_minimum_sampling_rate={"accelerometer": 2},
        collector_max_sample_interval_ms=500,
    )

    events = _events("collector", "collector", [0, 500])
    assert len(feature_frame(events, config, "collector")) == 1


def test_null_maximum_sample_interval_disables_continuity_check(config_factory):
    config = config_factory(
        collector_minimum_sampling_rate={"accelerometer": 2},
        collector_max_sample_interval_ms=None,
    )

    events = _events("collector", "collector", [0, 900])
    assert len(feature_frame(events, config, "collector")) == 1


@pytest.mark.parametrize(
    "middle_timestamps",
    [
        [1600, 1900],
        [1000, 1400],
    ],
)
def test_boundary_gap_rejects_only_affected_window(
    config_factory,
    middle_timestamps,
):
    config = config_factory(
        collector_minimum_sampling_rate={"accelerometer": 2},
        collector_max_sample_interval_ms=500,
    )
    timestamps = [0, 500, *middle_timestamps, 2000, 2500]

    features = feature_frame(
        _events("collector", "session", timestamps),
        config,
        "collector",
    )

    assert features["window_start_ms"].tolist() == [0, 2000]


def test_continuity_is_checked_per_sensor(config_factory):
    config = config_factory(
        sensors={"accelerometer": ["mean"], "pressure": ["range"]},
        collector_minimum_sampling_rate={"accelerometer": 4, "pressure": 2},
        collector_max_sample_interval_ms=500,
    )
    events = _events("collector", "collector", [0, 250, 500, 750])
    events["p"] = [1000.0, np.nan, np.nan, 1001.0]

    assert feature_frame(events, config, "collector").empty


def test_features_are_incremental_per_source(config_factory, monkeypatch):
    messages = []
    monkeypatch.setattr("all_tmd.windowing.progress", messages.append)
    config = config_factory()
    run_dir = config.run_dir()
    source_dir = run_dir / "events" / "us-tmd"
    collector_dir = run_dir / "events" / "collector"
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)
    _events("us-tmd", "source", [0, 1000]).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    (source_dir / "_SUCCESS").write_text("", encoding="utf-8")
    _events("collector", "collector-1", [0, 1000]).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )

    outputs = build_features(config)
    assert (outputs["us-tmd"] / "_SUCCESS").exists()
    collector_parts = sorted(outputs["collector"].glob("part-*.parquet"))
    assert len(collector_parts) == 1

    build_features(config)
    assert sorted(outputs["collector"].glob("part-*.parquet")) == collector_parts

    _events("collector", "collector-2", [0, 1000]).to_parquet(
        collector_dir / "part-000001.parquet",
        index=False,
    )
    rebuilt = build_features(config)
    assert rebuilt["collector"] != outputs["collector"]
    assert len(list(rebuilt["collector"].glob("part-*.parquet"))) == 2
    expected_progress = (
        "Feature event session scan starting",
        "Feature event part starting",
        "Feature event bucketing complete",
        "Feature bucket starting",
        "Feature session starting",
        "Feature session complete",
        "Feature extraction complete",
    )
    assert all(
        any(expected in message for message in messages)
        for expected in expected_progress
    )


def test_legacy_feature_policy_rebuilds_both_sources(config_factory):
    config = config_factory(collector_max_sample_interval_ms=500)
    run_dir = config.run_dir()
    source_dir = run_dir / "events" / "us-tmd"
    collector_dir = run_dir / "events" / "collector"
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)
    _events("us-tmd", "source", [0, 500, 1000, 1500]).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    (source_dir / "_SUCCESS").write_text("", encoding="utf-8")
    _events("collector", "collector", [0, 500, 1000, 1500]).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )
    outputs = build_features(config)
    sentinels = []
    for output in outputs.values():
        policy_path = output / "feature-policy.json"
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        policy["schema_version"] = 1
        policy_path.write_text(json.dumps(policy), encoding="utf-8")
        sentinel = output / "legacy-sentinel"
        sentinel.write_text("", encoding="utf-8")
        sentinels.append(sentinel)

    build_features(config)
    assert all(not sentinel.exists() for sentinel in sentinels)


def test_collector_policy_change_rebuilds_only_collector(config_factory):
    config = config_factory(collector_max_sample_interval_ms=500)
    run_dir = config.run_dir()
    source_dir = run_dir / "events" / "us-tmd"
    collector_dir = run_dir / "events" / "collector"
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)
    timestamps = [0, 500, 1000, 1500]
    _events("us-tmd", "source", timestamps).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    (source_dir / "_SUCCESS").write_text("", encoding="utf-8")
    _events("collector", "collector", timestamps).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )
    outputs = build_features(config)
    source_sentinel = outputs["us-tmd"] / "source-sentinel"
    collector_sentinel = outputs["collector"] / "collector-sentinel"
    source_sentinel.write_text("", encoding="utf-8")
    collector_sentinel.write_text("", encoding="utf-8")

    changed_config = config_factory(collector_max_sample_interval_ms=600)
    changed_outputs = build_features(changed_config)

    assert source_sentinel.exists()
    assert collector_sentinel.exists()
    assert changed_outputs["us-tmd"] == outputs["us-tmd"]
    assert changed_outputs["collector"] != outputs["collector"]


def test_corrupt_feature_policy_rebuilds_affected_source(config_factory):
    config = config_factory()
    run_dir = config.run_dir()
    source_dir = run_dir / "events" / "us-tmd"
    collector_dir = run_dir / "events" / "collector"
    source_dir.mkdir(parents=True)
    collector_dir.mkdir(parents=True)
    _events("us-tmd", "source", [0, 1000]).to_parquet(
        source_dir / "part-000000.parquet",
        index=False,
    )
    (source_dir / "_SUCCESS").write_text("", encoding="utf-8")
    _events("collector", "collector", [0, 1000]).to_parquet(
        collector_dir / "part-000000.parquet",
        index=False,
    )
    output = build_features(config)["collector"]
    sentinel = output / "corrupt-sentinel"
    sentinel.write_text("", encoding="utf-8")
    (output / "feature-policy.json").write_text("not json", encoding="utf-8")

    build_features(config)

    assert not sentinel.exists()


def test_multiscale_frequency_and_jerk_features(config_factory):
    config = config_factory(
        sensors={
            "pressure": [
                "mean_absolute_jerk",
                "spectral_energy",
                "dominant_frequency_hz",
                "spectral_entropy",
            ]
        },
        collector_minimum_sampling_rate={"pressure": 5},
        window_seconds=4,
        step_seconds=4,
        context_windows_seconds=[2, 4],
    )
    timestamps = list(range(0, 4100, 100))
    events = _events("us-tmd", "frequency", timestamps)
    seconds = np.asarray(timestamps, dtype=np.float64) / 1000.0
    events["p"] = 1000.0 + np.sin(2 * np.pi * seconds)

    features = feature_frame(events, config, "us-tmd")

    assert len(features) == 1
    assert features.loc[0, "pressure#mean_absolute_jerk@2s"] > 0
    assert features.loc[0, "pressure#spectral_energy@2s"] > 0
    assert features.loc[0, "pressure#dominant_frequency_hz@2s"] == pytest.approx(
        1.0,
        abs=0.1,
    )
    assert 0 <= features.loc[0, "pressure#spectral_entropy@4s"] <= 1


def _events(domain: str, session_id: str, timestamps: list[int]) -> pd.DataFrame:
    rows = []
    for index, timestamp in enumerate(timestamps):
        row = {column: np.nan for column in EVENT_COLUMNS}
        row.update(
            {
                "domain": domain,
                "participant_id": f"p-{session_id}",
                "device_id": "device",
                "session_id": session_id,
                "trip_id": session_id,
                "group_id": f"p-{session_id}#device#{session_id}",
                "vehicle_type": "car",
                "label": 1,
                "phone_position": "pocket",
                "timestamp_ms": timestamp,
                "ax": index + 1,
                "ay": index + 2,
                "az": index + 3,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)
