from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "aws"
    / "remote"
    / "sync-collector-sessions.py"
)
SPEC = importlib.util.spec_from_file_location("collector_cloud_sync", SCRIPT_PATH)
assert SPEC and SPEC.loader
collector_cloud_sync = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(collector_cloud_sync)


def test_sync_downloads_eligible_confirmed_sessions_and_checkpoints(tmp_path, monkeypatch):
    output_dir = tmp_path / "downloaded_sessions"
    sessions = [
        {
            "participant_id": "participant_001",
            "session_id": "session-1",
            "s3_key": "raw/participant_001/session-1.json.gz",
            "sync_key": "0000000000001#session-1",
        },
        {
            "participant_id": "test_001",
            "session_id": "test-session",
            "s3_key": "raw/test_001/test-session.json.gz",
            "sync_key": "0000000000002#test-session",
        },
    ]
    queried_after = []

    def fake_query(table, after_sync_key):
        queried_after.append((table, after_sync_key))
        return sessions

    def fake_download(bucket, destination, item):
        assert bucket == "collector-bucket"
        assert destination == output_dir
        return item["session_id"] == "session-1"

    monkeypatch.setattr(collector_cloud_sync, "query_sessions", fake_query)
    monkeypatch.setattr(collector_cloud_sync, "download_session", fake_download)

    result = collector_cloud_sync.sync(
        "collector-bucket", "TransportSessions", output_dir
    )

    assert result == {
        "discovered_count": 2,
        "eligible_count": 1,
        "downloaded_count": 1,
    }
    assert queried_after == [("TransportSessions", "")]
    checkpoint = json.loads(
        (output_dir / ".download_checkpoint.json").read_text(encoding="utf-8")
    )
    assert checkpoint["last_sync_key"] == "0000000000002#test-session"
    assert checkpoint["source_bucket"] == "collector-bucket"
    assert checkpoint["source_table"] == "TransportSessions"


def test_existing_checkpoint_limits_next_query(tmp_path, monkeypatch):
    output_dir = tmp_path / "downloaded_sessions"
    output_dir.mkdir()
    checkpoint = {
        "version": collector_cloud_sync.CHECKPOINT_VERSION,
        "last_sync_key": "0000000000042#session-42",
        "source_bucket": "collector-bucket",
        "source_table": "TransportSessions",
        "source_index": collector_cloud_sync.INDEX_NAME,
    }
    (output_dir / ".download_checkpoint.json").write_text(
        json.dumps(checkpoint), encoding="utf-8"
    )
    queried_after = []

    def fake_query(table, after_sync_key):
        queried_after.append(after_sync_key)
        return []

    monkeypatch.setattr(collector_cloud_sync, "query_sessions", fake_query)

    result = collector_cloud_sync.sync(
        "collector-bucket", "TransportSessions", output_dir
    )

    assert queried_after == ["0000000000042#session-42"]
    assert result["downloaded_count"] == 0


def test_frozen_manifest_queries_all_and_downloads_only_required_sessions(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "downloaded_sessions"
    sessions = [
        {
            "participant_id": "participant_001",
            "session_id": session_id,
            "s3_key": f"raw/participant_001/{session_id}.json.gz",
            "sync_key": f"0001#{session_id}",
        }
        for session_id in ("required", "later-upload")
    ]
    queried_after = []

    def fake_query(table, after_sync_key):
        queried_after.append(after_sync_key)
        return sessions

    def fake_download(bucket, destination, item):
        payload = collector_cloud_sync.destination_for(destination, item["s3_key"])
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"payload")
        payload.with_suffix(f"{payload.suffix}.metadata.json").write_text(
            json.dumps(item), encoding="utf-8"
        )
        return True

    monkeypatch.setattr(collector_cloud_sync, "query_sessions", fake_query)
    monkeypatch.setattr(collector_cloud_sync, "download_session", fake_download)

    result = collector_cloud_sync.sync(
        "collector-bucket",
        "TransportSessions",
        output_dir,
        {"required"},
    )

    assert queried_after == [""]
    assert result["eligible_count"] == 1
    assert result["downloaded_count"] == 1
    assert not any(output_dir.rglob("later-upload.json.gz"))


def test_eligibility_rejects_test_participants_and_mismatched_keys():
    assert collector_cloud_sync.is_eligible(
        {
            "participant_id": "participant_123",
            "s3_key": "raw/participant_123/session.json.gz",
        }
    )
    assert not collector_cloud_sync.is_eligible(
        {
            "participant_id": "test_123",
            "s3_key": "raw/test_123/session.json.gz",
        }
    )
    assert not collector_cloud_sync.is_eligible(
        {
            "participant_id": "participant_123",
            "s3_key": "raw/participant_999/session.json.gz",
        }
    )


def test_write_collector_snapshot_records_complete_stable_metadata(tmp_path):
    output_dir = tmp_path / "downloaded_sessions"
    payload = output_dir / "raw" / "participant_001" / "device-1" / "one.json.gz"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"payload")
    metadata = {
        "session_id": "one",
        "participant_id": "participant_001",
        "device_uuid": "device-1",
        "vehicle_type": "train",
        "phone_position": "pocket",
        "trimmed_start_ms": "1000",
        "trimmed_end_ms": "61000",
        "sample_count": "120",
        "uploaded_at_ms": "70000",
        "s3_key": "raw/participant_001/device-1/one.json.gz",
        "sync_key": "0001#one",
        "ignored_field": "not copied",
    }
    payload.with_suffix(f"{payload.suffix}.metadata.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )
    (output_dir / ".download_checkpoint.json").write_text(
        json.dumps(
            {
                "version": collector_cloud_sync.CHECKPOINT_VERSION,
                "last_sync_key": "0001#one",
                "source_bucket": "collector-bucket",
                "source_table": "TransportSessions",
                "source_index": collector_cloud_sync.INDEX_NAME,
            }
        ),
        encoding="utf-8",
    )
    snapshot_path = tmp_path / "collector-snapshot.json"

    snapshot = collector_cloud_sync.write_collector_snapshot(
        snapshot_path,
        run_id="run-1",
        bucket="collector-bucket",
        table="TransportSessions",
        output_dir=output_dir,
    )

    assert snapshot == json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["run_id"] == "run-1"
    captured_at = datetime.fromisoformat(snapshot["captured_at"])
    assert captured_at.tzinfo == timezone.utc
    assert snapshot["session_count"] == 1
    assert len(snapshot["session_id_digest"]) == 64
    assert snapshot["source"]["last_sync_key"] == "0001#one"
    assert snapshot["sessions"] == [
        {
            "session_id": "one",
            "participant_id": "participant_001",
            "device_uuid": "device-1",
            "vehicle_type": "train",
            "phone_position": "pocket",
            "s3_key": "raw/participant_001/device-1/one.json.gz",
            "sync_key": "0001#one",
            "trimmed_start_ms": 1000,
            "trimmed_end_ms": 61000,
            "uploaded_at_ms": 70000,
            "sample_count": 120,
            "duration_seconds": 60.0,
        }
    ]


def test_snapshot_rejects_sidecar_without_payload(tmp_path):
    output_dir = tmp_path / "downloaded_sessions"
    metadata_path = output_dir / "raw" / "participant_001" / "one.json.gz.metadata.json"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text(
        json.dumps(
            {
                "session_id": "one",
                "participant_id": "participant_001",
                "s3_key": "raw/participant_001/one.json.gz",
            }
        ),
        encoding="utf-8",
    )

    try:
        collector_cloud_sync.collector_snapshot_sessions(output_dir)
    except ValueError as error:
        assert "has no payload" in str(error)
    else:
        raise AssertionError("missing snapshot payload should fail")
