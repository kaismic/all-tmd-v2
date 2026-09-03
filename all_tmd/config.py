from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml

from all_tmd.features import normalize_aggregation_name


SUPPORTED_SENSORS = ("accelerometer", "gyroscope", "magnetometer", "pressure")
SUPPORTED_MODELS = ("random_forest", "xgboost", "mlp")
SUPPORTED_EVALUATION_STRATEGIES = ("session_holdout", "participant_nested_cv")
SUPPORTED_WEIGHTING_STRATEGIES = ("class_balanced", "hierarchical")
SUPPORTED_DURATION_BALANCING = ("none", "smallest_mode")
SUPPORTED_SELECTION_METRICS = ("macro_f1", "minimum_class_recall")
TRIAL_DISPLAY_FIELDS = frozenset({"run_name"})


@dataclass(frozen=True)
class SourceConfig:
    input_path: Path
    include_globs: tuple[str, ...]
    chunk_rows: int | None = None


@dataclass(frozen=True)
class SourcesConfig:
    training: dict[str, SourceConfig]
    collector: SourceConfig


@dataclass(frozen=True)
class DatasetConfig:
    work_dir: Path
    minimum_trip_seconds: int
    maximum_trip_seconds: int
    collector_max_sample_interval_ms: int | None
    input_manifest_digest: str
    snapshot_manifest: Path | None


@dataclass(frozen=True)
class GlobalTrainingConfig:
    n_jobs: int
    timeout_seconds: int | None
    xgboost_device: str


@dataclass(frozen=True)
class MlflowConfig:
    enabled: bool
    experiment_name: str
    tracking_uri: str | None
    artifact_location: str | None


@dataclass(frozen=True)
class FeatureConfig:
    default_window_seconds: int
    default_step_seconds: int
    context_windows_seconds: tuple[int, ...]
    sensors: dict[str, tuple[str, ...]]


@dataclass(frozen=True)
class TrialTrainingConfig:
    random_seed: int
    optuna_trials: int
    model_families: tuple[str, ...]
    calibration_fraction: dict[str, float]
    evaluation_strategy: str
    weighting_strategy: str
    collector_domain_weight: float
    duration_balancing: str
    participant_inner_folds: int
    bootstrap_iterations: int
    selection_metric: str


@dataclass(frozen=True)
class TrialConfig:
    run_name: str | None
    train_dataset: str
    labels: dict[str, int]
    features: FeatureConfig
    training: TrialTrainingConfig
    raw: dict[str, Any]

    @property
    def config_hash_input(self) -> dict[str, Any]:
        """Return the trial fields that affect ingestion and feature extraction."""
        excluded = TRIAL_DISPLAY_FIELDS | {"training"}
        return {key: value for key, value in self.raw.items() if key not in excluded}

    @property
    def trial_hash_input(self) -> dict[str, Any]:
        """Return behavior-affecting fields for the complete trial hash."""
        return {
            key: value
            for key, value in self.raw.items()
            if key not in TRIAL_DISPLAY_FIELDS
        }

    @property
    def config_hash(self) -> str:
        """Hash this trial's ingestion and feature-extraction configuration."""
        canonical = json.dumps(
            self.config_hash_input,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def trial_hash(self) -> str:
        """Hash the complete trial, including training-only settings."""
        canonical = json.dumps(
            self.trial_hash_input,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    @property
    def feature_names(self) -> list[str]:
        single_default_context = self.features.context_windows_seconds == (
            self.features.default_window_seconds,
        )
        return [
            (
                f"{sensor}#{aggregation}"
                if single_default_context
                else f"{sensor}#{aggregation}@{context_seconds}s"
            )
            for context_seconds in self.features.context_windows_seconds
            for sensor, aggregations in self.features.sensors.items()
            for aggregation in aggregations
        ]

    def minimum_samples(self, sampling_rates: dict[str, float]) -> dict[str, int]:
        return self.minimum_samples_for_window(
            sampling_rates,
            self.features.default_window_seconds,
        )

    def minimum_samples_for_window(
        self,
        sampling_rates: dict[str, float],
        window_seconds: int,
    ) -> dict[str, int]:
        return {
            sensor: max(
                1,
                math.ceil(
                    window_seconds * sampling_rates[sensor]
                ),
            )
            for sensor in self.features.sensors
        }


@dataclass(frozen=True)
class PipelineConfig:
    schema_version: int
    sources: SourcesConfig
    dataset: DatasetConfig
    collector_minimum_sampling_rate: dict[str, float]
    training: GlobalTrainingConfig
    mlflow: MlflowConfig
    trial: TrialConfig
    trial_index: int

    @classmethod
    def from_files(
        cls,
        config_path: str | Path = "model.config.yaml",
        trials_path: str | Path = "trials.json",
        trial_index: int = 0,
    ) -> "PipelineConfig":
        config_file = Path(config_path)
        trials_file = Path(trials_path)
        with config_file.open("r", encoding="utf-8") as stream:
            data: dict[str, Any] = yaml.safe_load(stream)
        with trials_file.open("r", encoding="utf-8") as stream:
            trials_data = json.load(stream)

        if not isinstance(trials_data, list) or not trials_data:
            raise ValueError("trials.json must contain a non-empty JSON array")
        if trial_index < 0 or trial_index >= len(trials_data):
            raise IndexError(
                f"Trial index {trial_index} is outside the available range "
                f"0..{len(trials_data) - 1}"
            )
        raw_trial = trials_data[trial_index]
        if not isinstance(raw_trial, dict):
            raise ValueError(f"Trial {trial_index} must be a JSON object")

        source_data = data["sources"]
        collector_data = source_data["collector"]
        training_sources = {
            str(name): _source_config(name, source)
            for name, source in source_data.items()
            if name != "collector"
        }
        trial = _trial_config(raw_trial)
        if trial.train_dataset not in training_sources:
            available = ", ".join(sorted(training_sources))
            raise ValueError(
                f"Trial train_dataset '{trial.train_dataset}' has no source "
                f"configuration. Available: {available}"
            )

        if "minimum_sampling_rate" in data:
            raise ValueError(
                "minimum_sampling_rate was renamed to "
                "collector_minimum_sampling_rate"
            )
        rates = {
            str(sensor): float(rate)
            for sensor, rate in data["collector_minimum_sampling_rate"].items()
        }
        invalid_rates = sorted(
            sensor
            for sensor, rate in rates.items()
            if not math.isfinite(rate) or rate <= 0
        )
        if invalid_rates:
            raise ValueError(
                "collector_minimum_sampling_rate must contain finite positive "
                "values for: " + ", ".join(invalid_rates)
            )
        missing_rates = sorted(set(trial.features.sensors) - set(rates))
        if missing_rates:
            raise ValueError(
                "collector_minimum_sampling_rate is missing configured sensor(s): "
                + ", ".join(missing_rates)
            )
        dataset = data["dataset"]
        training = data["training"]
        mlflow = data["mlflow"]
        if "maximum_sample_interval_ms" in dataset:
            raise ValueError(
                "dataset.maximum_sample_interval_ms was renamed to "
                "dataset.collector_max_sample_interval_ms"
            )
        collector_max_sample_interval_ms = _optional_positive_int(
            dataset.get("collector_max_sample_interval_ms"),
            "dataset.collector_max_sample_interval_ms",
        )
        return cls(
            schema_version=int(data["schema_version"]),
            sources=SourcesConfig(
                training=training_sources,
                collector=_source_config("collector", collector_data),
            ),
            dataset=DatasetConfig(
                work_dir=_path(dataset["work_dir"], "dataset.work_dir"),
                minimum_trip_seconds=int(dataset["minimum_trip_seconds"]),
                maximum_trip_seconds=int(dataset.get("maximum_trip_seconds", 28_800)),
                collector_max_sample_interval_ms=collector_max_sample_interval_ms,
                input_manifest_digest=str(dataset.get("input_manifest_digest", "unversioned")),
                snapshot_manifest=(
                    None
                    if dataset.get("snapshot_manifest") is None
                    else _path(dataset["snapshot_manifest"], "dataset.snapshot_manifest")
                ),
            ),
            collector_minimum_sampling_rate=rates,
            training=GlobalTrainingConfig(
                n_jobs=int(training["n_jobs"]),
                timeout_seconds=(
                    None
                    if training.get("timeout_seconds") is None
                    else int(training["timeout_seconds"])
                ),
                xgboost_device=str(training.get("xgboost_device", "cpu")),
            ),
            mlflow=MlflowConfig(
                enabled=bool(mlflow["enabled"]),
                experiment_name=str(mlflow["experiment_name"]),
                tracking_uri=mlflow.get("tracking_uri"),
                artifact_location=mlflow.get("artifact_location"),
            ),
            trial=trial,
            trial_index=trial_index,
        )

    @property
    def ingestion_hash_input(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "adapter_schema_version": 1,
            "train_dataset": self.trial.train_dataset,
            "labels": self.trial.labels,
            "input_manifest_digest": self.dataset.input_manifest_digest,
            "minimum_trip_seconds": self.dataset.minimum_trip_seconds,
            "maximum_trip_seconds": self.dataset.maximum_trip_seconds,
        }

    @property
    def ingestion_hash(self) -> str:
        return _canonical_hash(self.ingestion_hash_input)

    @property
    def feature_hash_input(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "feature_schema_version": 3,
            "ingestion_hash": self.ingestion_hash,
            "features": self.trial.raw["features"],
            "collector_max_sample_interval_ms": self.dataset.collector_max_sample_interval_ms,
            "collector_minimum_sampling_rate": {
                sensor: self.collector_minimum_sampling_rate[sensor]
                for sensor in sorted(self.trial.features.sensors)
            },
        }

    @property
    def feature_hash(self) -> str:
        return _canonical_hash(self.feature_hash_input)

    @property
    def config_hash(self) -> str:
        """Compatibility alias for the immutable ingestion-cache identity."""
        return self.ingestion_hash

    @property
    def trial_hash(self) -> str:
        return self.trial.trial_hash

    def run_dir(self) -> Path:
        run_dir = self.dataset.work_dir / self.config_hash
        run_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = run_dir / "cache-manifest.json"
        manifest = {
            "schema_version": 1,
            "ingestion_hash": self.ingestion_hash,
            "ingestion_hash_input": self.ingestion_hash_input,
        }
        canonical = json.dumps(
            manifest,
            sort_keys=True,
            indent=2,
            ensure_ascii=True,
        ) + "\n"
        if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != canonical:
            raise ValueError(f"Immutable cache manifest collision at {manifest_path}")
        if not manifest_path.exists():
            manifest_path.write_text(canonical, encoding="utf-8")
        return run_dir

    def report_dir(self) -> Path:
        return (
            self.run_dir()
            / "reports"
            / self.trial.train_dataset
            / self.trial_hash
        )

    def split_path(self) -> Path:
        return (
            self.run_dir()
            / "splits"
            / self.trial.train_dataset
            / f"{self.trial_hash}.json"
        )

    @property
    def training_source(self) -> SourceConfig:
        return self.sources.training[self.trial.train_dataset]


def _source_config(name: str, data: dict[str, Any]) -> SourceConfig:
    raw_path = data.get("input_path", data.get("input_dir", data.get("input_csv")))
    if raw_path is None:
        raise ValueError(f"sources.{name} requires input_dir, input_csv, or input_path")
    default_globs = (
        ("**/*.json", "**/*.json.gz")
        if name == "collector"
        else ("**/*.csv",)
    )
    chunk_rows = data.get("chunk_rows")
    return SourceConfig(
        input_path=_path(raw_path, f"sources.{name}.input_path"),
        include_globs=tuple(str(value) for value in data.get("include_globs", default_globs)),
        chunk_rows=None if chunk_rows is None else int(chunk_rows),
    )


def _trial_config(raw: dict[str, Any]) -> TrialConfig:
    required = {"train_dataset", "labels", "features", "training"}
    missing = sorted(required - set(raw))
    if missing:
        raise ValueError("Trial is missing field(s): " + ", ".join(missing))

    run_name_value = raw.get("run_name")
    if run_name_value is None:
        run_name = None
    elif not isinstance(run_name_value, str) or not run_name_value.strip():
        raise ValueError("Trial run_name must be a non-empty string when provided")
    else:
        run_name = run_name_value.strip()

    features = raw["features"]
    default_window_seconds = int(features["default_window_seconds"])
    default_step_seconds = int(features["default_step_seconds"])
    if default_window_seconds <= 0 or default_step_seconds <= 0:
        raise ValueError("Feature window and step seconds must be greater than zero")
    raw_contexts = features.get(
        "context_windows_seconds",
        [default_window_seconds],
    )
    if not isinstance(raw_contexts, list) or not raw_contexts:
        raise ValueError("features.context_windows_seconds must be a non-empty array")
    context_windows_seconds = tuple(int(value) for value in raw_contexts)
    if (
        any(value <= 0 or value > default_window_seconds for value in context_windows_seconds)
        or len(set(context_windows_seconds)) != len(context_windows_seconds)
    ):
        raise ValueError(
            "features.context_windows_seconds must contain unique positive values "
            "no greater than features.default_window_seconds"
        )
    context_windows_seconds = tuple(sorted(context_windows_seconds))
    if default_window_seconds not in context_windows_seconds:
        raise ValueError(
            "features.context_windows_seconds must include "
            "features.default_window_seconds"
        )
    sensors: dict[str, tuple[str, ...]] = {}
    for sensor, aggregations in features["sensors"].items():
        sensor_name = str(sensor).lower()
        if sensor_name not in SUPPORTED_SENSORS:
            raise ValueError(f"Unsupported sensor: {sensor_name}")
        normalized = tuple(normalize_aggregation_name(value) for value in aggregations)
        if not normalized:
            raise ValueError(f"Sensor '{sensor_name}' requires at least one aggregation")
        sensors[sensor_name] = normalized
    if not sensors:
        raise ValueError("Trial features.sensors cannot be empty")

    training = raw["training"]
    families = tuple(str(value).lower() for value in training["model_families"])
    unknown_families = sorted(set(families) - set(SUPPORTED_MODELS))
    if unknown_families:
        raise ValueError("Unsupported model family: " + ", ".join(unknown_families))
    optuna_trials = int(training["optuna_trials"])
    if optuna_trials < 1:
        raise ValueError("training.optuna_trials must be at least 1")

    labels = {str(key).lower(): int(value) for key, value in raw["labels"].items()}
    if len(set(labels.values())) != len(labels):
        raise ValueError("Trial label values must be unique")
    calibration_fraction = _calibration_fractions(
        training.get("calibration_fraction", 0.5),
        labels,
    )
    evaluation_strategy = str(
        training.get("evaluation_strategy", "session_holdout")
    ).lower()
    if evaluation_strategy not in SUPPORTED_EVALUATION_STRATEGIES:
        raise ValueError(
            "Unsupported training.evaluation_strategy: " + evaluation_strategy
        )
    weighting_strategy = str(
        training.get("weighting_strategy", "class_balanced")
    ).lower()
    if weighting_strategy not in SUPPORTED_WEIGHTING_STRATEGIES:
        raise ValueError(
            "Unsupported training.weighting_strategy: " + weighting_strategy
        )
    duration_balancing = str(
        training.get("duration_balancing", "none")
    ).lower()
    if duration_balancing not in SUPPORTED_DURATION_BALANCING:
        raise ValueError(
            "Unsupported training.duration_balancing: " + duration_balancing
        )
    collector_domain_weight = float(training.get("collector_domain_weight", 2.0))
    if not math.isfinite(collector_domain_weight) or collector_domain_weight <= 0:
        raise ValueError(
            "training.collector_domain_weight must be a finite positive number"
        )
    participant_inner_folds = int(training.get("participant_inner_folds", 5))
    if participant_inner_folds < 2:
        raise ValueError("training.participant_inner_folds must be at least 2")
    bootstrap_iterations = int(training.get("bootstrap_iterations", 0))
    if bootstrap_iterations < 0:
        raise ValueError("training.bootstrap_iterations cannot be negative")
    selection_metric = str(training.get("selection_metric", "macro_f1")).lower()
    if selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            "Unsupported training.selection_metric: " + selection_metric
        )
    return TrialConfig(
        run_name=run_name,
        train_dataset=str(raw["train_dataset"]).lower(),
        labels=labels,
        features=FeatureConfig(
            default_window_seconds=default_window_seconds,
            default_step_seconds=default_step_seconds,
            context_windows_seconds=context_windows_seconds,
            sensors=sensors,
        ),
        training=TrialTrainingConfig(
            random_seed=int(training["random_seed"]),
            optuna_trials=optuna_trials,
            model_families=families,
            calibration_fraction=calibration_fraction,
            evaluation_strategy=evaluation_strategy,
            weighting_strategy=weighting_strategy,
            collector_domain_weight=collector_domain_weight,
            duration_balancing=duration_balancing,
            participant_inner_folds=participant_inner_folds,
            bootstrap_iterations=bootstrap_iterations,
            selection_metric=selection_metric,
        ),
        raw=raw,
    )


def _path(value: Any, field: str) -> Path:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} cannot be empty")
    return Path(text)


def _optional_positive_int(value: Any, field: str) -> int | None:
    if value is None:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{field} must be greater than zero or null")
    return parsed


def _calibration_fractions(
    value: Any,
    labels: dict[str, int],
) -> dict[str, float]:
    field = "training.calibration_fraction"
    if isinstance(value, dict):
        normalized_keys = [str(key).lower() for key in value]
        if len(normalized_keys) != len(set(normalized_keys)):
            raise ValueError(f"{field} contains duplicate transport modes")
        normalized = {
            str(key).lower(): fraction
            for key, fraction in value.items()
        }
        missing = sorted(set(labels) - set(normalized))
        unknown = sorted(set(normalized) - set(labels))
        if missing:
            raise ValueError(
                f"{field} is missing transport mode(s): " + ", ".join(missing)
            )
        if unknown:
            raise ValueError(
                f"{field} contains unknown transport mode(s): "
                + ", ".join(unknown)
            )
        raw_fractions = normalized
    else:
        raw_fractions = {label: value for label in labels}

    fractions: dict[str, float] = {}
    for label in labels:
        try:
            fraction = float(raw_fractions[label])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{field}.{label} must be a number between 0 and 1"
            ) from exc
        if not math.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError(
                f"{field}.{label} must be a finite number between 0 and 1"
            )
        fractions[label] = fraction
    return fractions


def _canonical_hash(value: Any) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
