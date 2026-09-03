from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


SUPPORTED_CONDITIONS = ("nor_only", "sydney_only", "nor_plus_sydney")
TARGET_MODES = ("bus", "car", "train")
SNAPSHOT_DIGEST = "8931f5b287838320e9173639d4023da19f34271b6b8abdfdc63691102674124c"


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ParentRunSpec:
    condition: str
    sydney_fraction: float
    seed: int

    @property
    def key(self) -> str:
        fraction = f"{self.sydney_fraction:.2f}".replace(".", "p")
        return f"{self.condition}-sydney-{fraction}-seed-{self.seed}"


@dataclass(frozen=True)
class StudyPlan:
    study_id: str
    conditions: tuple[str, ...]
    sydney_fractions: tuple[float, ...]
    seeds: tuple[int, ...]
    labels: dict[str, int]
    snapshot_path: Path
    lopo_manifest_path: Path
    model_lock_path: Path
    analysis_seed: int
    bootstrap_iterations: int

    @classmethod
    def load(cls, path: str | Path) -> "StudyPlan":
        plan_path = Path(path)
        raw = json.loads(plan_path.read_text(encoding="utf-8"))
        conditions = tuple(str(value) for value in raw["conditions"])
        if conditions != SUPPORTED_CONDITIONS:
            raise ValueError(
                "conditions must be exactly: " + ", ".join(SUPPORTED_CONDITIONS)
            )
        fractions = tuple(float(value) for value in raw["sydney_fractions"])
        if fractions != (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
            raise ValueError("unexpected Sydney calibration fraction grid")
        seeds = tuple(int(value) for value in raw["seeds"])
        if seeds != (42, 43, 44):
            raise ValueError("study seeds must be 42, 43, and 44")
        labels = {str(key): int(value) for key, value in raw["labels"].items()}
        if tuple(labels) != TARGET_MODES or sorted(labels.values()) != [0, 1, 2]:
            raise ValueError("labels must map bus, car, and train to 0, 1, and 2")

        def relative(field: str) -> Path:
            candidate = Path(raw[field])
            return candidate if candidate.is_absolute() else plan_path.parent / candidate

        return cls(
            study_id=str(raw["study_id"]),
            conditions=conditions,
            sydney_fractions=fractions,
            seeds=seeds,
            labels=labels,
            snapshot_path=relative("snapshot_manifest"),
            lopo_manifest_path=relative("lopo_manifest"),
            model_lock_path=relative("model_lock"),
            analysis_seed=int(raw["analysis"]["random_seed"]),
            bootstrap_iterations=int(raw["analysis"]["bootstrap_iterations"]),
        )

    def parent_specs(self) -> list[ParentRunSpec]:
        result: list[ParentRunSpec] = []
        for seed in self.seeds:
            result.append(ParentRunSpec("nor_only", 0.0, seed))
            for fraction in self.sydney_fractions[1:]:
                result.append(ParentRunSpec("sydney_only", fraction, seed))
                result.append(ParentRunSpec("nor_plus_sydney", fraction, seed))
        return result


def verify_snapshot(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    sessions = raw.get("sessions")
    if not isinstance(sessions, list):
        raise ValueError("snapshot sessions must be an array")
    ids = [str(row["session_id"]) for row in sessions]
    if len(ids) != 166 or len(set(ids)) != 166:
        raise ValueError("Sydney snapshot must contain 166 unique sessions")
    if raw.get("session_count") != 166:
        raise ValueError("Sydney snapshot session_count must be 166")
    digest = canonical_digest(sorted(ids))
    # The collector snapshot format hashes the compact sorted session-id array.
    if digest != SNAPSHOT_DIGEST or raw.get("session_id_digest") != SNAPSHOT_DIGEST:
        raise ValueError("Sydney snapshot digest does not match the published snapshot")
    participants = {str(row["participant_id"]) for row in sessions}
    modes = [str(row["vehicle_type"]) for row in sessions]
    long_sessions = [
        str(row["session_id"])
        for row in sessions
        if str(row["vehicle_type"]) in TARGET_MODES
        and float(row.get("duration_seconds", 0.0)) > 7200
    ]
    eligible = [
        row
        for row in sessions
        if str(row["vehicle_type"]) in TARGET_MODES
        and float(row.get("duration_seconds", 0.0)) <= 7200
    ]
    if len(participants) != 7:
        raise ValueError("Sydney snapshot must contain seven participants")
    if {mode: modes.count(mode) for mode in (*TARGET_MODES, "tram")} != {
        "bus": 39,
        "car": 71,
        "train": 47,
        "tram": 9,
    }:
        raise ValueError("Sydney snapshot mode counts are not the published counts")
    if len(long_sessions) != 2 or len(eligible) != 155:
        raise ValueError("two-hour eligibility must exclude exactly two target sessions")
    return {
        "snapshot_digest": digest,
        "session_count": len(sessions),
        "participant_count": len(participants),
        "eligible_session_count": len(eligible),
        "excluded_tram_count": modes.count("tram"),
        "excluded_long_session_ids": sorted(long_sessions),
        "eligible_session_ids": sorted(str(row["session_id"]) for row in eligible),
    }


def verify_snapshot_files(snapshot_path: str | Path, input_dir: str | Path) -> dict[str, Any]:
    summary = verify_snapshot(snapshot_path)
    expected = set(summary["eligible_session_ids"])
    present: set[str] = set()
    root = Path(input_dir)
    for path in (*root.rglob("*.json"), *root.rglob("*.json.gz")):
        name = path.name.removesuffix(".gz").removesuffix(".json")
        if not name.endswith(".metadata"):
            present.add(name)
    missing = sorted(expected - present)
    if missing:
        raise FileNotFoundError(
            f"Sydney input is missing {len(missing)} manifest session(s): {missing[:5]}"
        )
    return {**summary, "present_eligible_count": len(expected), "ignored_extra_count": len(present - expected)}


def create_lopo_manifest(collector: pd.DataFrame) -> dict[str, Any]:
    required = {"participant_id", "session_id", "vehicle_type"}
    missing = sorted(required - set(collector.columns))
    if missing:
        raise ValueError("collector features are missing: " + ", ".join(missing))
    sessions = (
        collector.loc[collector["vehicle_type"].astype(str).isin(TARGET_MODES)]
        .groupby("session_id", sort=True)
        .agg(
            participant_id=("participant_id", "first"),
            vehicle_type=("vehicle_type", "first"),
            rows=("session_id", "size"),
            window_start_ms=("window_start_ms", "min"),
            window_end_ms=("window_end_ms", "max"),
        )
        .reset_index()
    )
    if sessions.empty:
        raise ValueError("no eligible Sydney feature sessions are available")
    participants = sorted(sessions["participant_id"].astype(str).unique())
    if len(participants) != 7:
        raise ValueError("LOPO evaluation requires exactly seven Sydney participants")
    folds = []
    evaluation_union: list[str] = []
    for participant in participants:
        test = sorted(
            sessions.loc[
                sessions["participant_id"].astype(str) == participant, "session_id"
            ].astype(str)
        )
        train_pool = sorted(
            sessions.loc[
                sessions["participant_id"].astype(str) != participant, "session_id"
            ].astype(str)
        )
        evaluation_union.extend(test)
        folds.append(
            {
                "held_out_participant_id": participant,
                "test_session_ids": test,
                "training_pool_session_ids": train_pool,
                "test_digest": canonical_digest(test),
                "training_pool_digest": canonical_digest(train_pool),
            }
        )
    evaluation = sorted(evaluation_union)
    payload = {
        "manifest_version": 1,
        "group_column": "participant_id",
        "participant_count": 7,
        "evaluation_session_ids": evaluation,
        "evaluation_digest": canonical_digest(evaluation),
        "folds": folds,
    }
    payload["manifest_digest"] = canonical_digest(payload)
    return payload


def calibration_session_ids(
    collector: pd.DataFrame,
    fold: dict[str, Any],
    fraction: float,
    seed: int,
) -> tuple[list[str], dict[str, Any]]:
    if not 0 < fraction <= 1:
        raise ValueError("Sydney calibration fraction must be greater than 0 and at most 1")
    pool_ids = set(str(value) for value in fold["training_pool_session_ids"])
    sessions = (
        collector.loc[collector["session_id"].astype(str).isin(pool_ids)]
        .groupby("session_id", sort=True)
        .agg(
            vehicle_type=("vehicle_type", "first"),
            rows=("session_id", "size"),
            window_start_ms=("window_start_ms", "min"),
            window_end_ms=("window_end_ms", "max"),
        )
        .reset_index()
    )
    chosen: list[str] = []
    counts: dict[str, int] = {}
    available: dict[str, int] = {}
    duration: dict[str, float] = {}
    windows: dict[str, int] = {}
    for mode in TARGET_MODES:
        mode_rows = sessions.loc[sessions["vehicle_type"].astype(str) == mode]
        records = mode_rows["session_id"].astype(str).to_numpy()
        if not len(records):
            raise ValueError(f"Sydney training pool has no {mode} sessions")
        mode_seed = int.from_bytes(
            hashlib.sha256(f"{seed}\0{mode}".encode()).digest()[:8], "big"
        )
        order = np.random.default_rng(mode_seed).permutation(records)
        count = len(records) if fraction == 1 else max(1, math.floor(len(records) * fraction))
        selected = [str(value) for value in order[:count]]
        chosen.extend(selected)
        selected_rows = mode_rows.loc[mode_rows["session_id"].astype(str).isin(selected)]
        counts[mode] = count
        available[mode] = len(records)
        duration[mode] = float(
            ((selected_rows["window_end_ms"] - selected_rows["window_start_ms"]) / 1000).sum()
        )
        windows[mode] = int(selected_rows["rows"].sum())
    chosen.sort()
    total_available_sessions = sum(available.values())
    total_selected_sessions = sum(counts.values())
    total_available_windows = int(sessions["rows"].sum())
    total_selected_windows = sum(windows.values())
    total_available_duration = float(
        ((sessions["window_end_ms"] - sessions["window_start_ms"]) / 1000).sum()
    )
    total_selected_duration = sum(duration.values())
    return chosen, {
        "requested_fraction": fraction,
        "session_counts_by_mode": counts,
        "available_sessions_by_mode": available,
        "duration_seconds_by_mode": duration,
        "window_rows_by_mode": windows,
        "effective_session_fraction": total_selected_sessions / total_available_sessions,
        "effective_window_fraction": total_selected_windows / total_available_windows,
        "effective_duration_fraction": (
            total_selected_duration / total_available_duration
            if total_available_duration
            else 0.0
        ),
        "session_digest": canonical_digest(chosen),
    }


def assert_nested_subsets(
    collector: pd.DataFrame,
    fold: dict[str, Any],
    fractions: Iterable[float],
    seed: int,
) -> None:
    previous: set[str] = set()
    for fraction in fractions:
        current, _ = calibration_session_ids(collector, fold, fraction, seed)
        if not previous.issubset(current):
            raise AssertionError("Sydney calibration fractions are not nested")
        previous = set(current)
