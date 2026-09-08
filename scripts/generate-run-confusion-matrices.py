"""Regenerate row-normalized confusion-matrix images for MLflow run IDs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence


SUMMARY_REPORT_KEYS = {"accuracy", "macro avg", "weighted avg"}


def output_filename(run_id: str, condition: str | None = None) -> str:
    """Return the concise image filename with run ID and optional condition."""
    condition_suffix = f"-{condition}" if condition else ""
    return f"conf-matrix-norm-{run_id[:7]}{condition_suffix}.png"


def load_condition_map(path: Path) -> dict[str, str]:
    """Load run-ID prefixes and article condition labels from JSON."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON ({error})") from error
    document = _as_mapping(document, "condition map", path)
    records = document.get("conditions")
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path}: conditions must be a non-empty JSON array")

    mapping: dict[str, str] = {}
    for index, raw_record in enumerate(records):
        record = _as_mapping(raw_record, f"conditions[{index}]", path)
        run_id = record.get("run_id")
        condition = record.get("condition")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError(f"{path}: conditions[{index}].run_id must be a string")
        if not isinstance(condition, str) or not condition:
            raise ValueError(
                f"{path}: conditions[{index}].condition must be a string"
            )
        if run_id in mapping:
            raise ValueError(f"{path}: duplicate run ID {run_id!r}")
        mapping[run_id] = condition
    return mapping


def condition_for_run(run_id: str, condition_map: dict[str, str]) -> str | None:
    """Resolve a full MLflow ID against exact IDs or unique stored prefixes."""
    matches = {
        condition
        for recorded_id, condition in condition_map.items()
        if run_id.startswith(recorded_id) or recorded_id.startswith(run_id)
    }
    if len(matches) > 1:
        raise ValueError(f"run ID {run_id!r} matches multiple conditions")
    return next(iter(matches), None)


def find_run_artifacts(
    results_root: Path,
    run_ids: Sequence[str],
) -> dict[str, list[Path]]:
    """Find artifact directories whose parent is an exact requested run ID."""
    requested = set(run_ids)
    matches: dict[str, list[Path]] = defaultdict(list)
    for metrics_path in results_root.rglob("metrics.json"):
        artifacts_dir = metrics_path.parent
        run_dir = artifacts_dir.parent
        experiment_dir = run_dir.parent
        if (
            artifacts_dir.name == "artifacts"
            and experiment_dir.parent.name == "mlartifacts"
            and experiment_dir.parent.parent.name == "mlflow"
            and run_dir.name in requested
        ):
            matches[run_dir.name].append(artifacts_dir)

    return {
        run_id: sorted(matches.get(run_id, []))
        for run_id in dict.fromkeys(run_ids)
    }


def _as_mapping(value: Any, description: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: {description} must be a JSON object")
    return value


def read_confusion_matrix(
    metrics_path: Path,
) -> tuple[list[list[float]], tuple[str, ...]]:
    """Read and validate the collector-holdout matrix and its label order."""
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{metrics_path}: invalid JSON ({error})") from error

    metrics = _as_mapping(metrics, "metrics", metrics_path)
    holdout = _as_mapping(
        metrics.get("collector_holdout"),
        "collector_holdout",
        metrics_path,
    )
    report = _as_mapping(
        holdout.get("classification_report"),
        "collector_holdout.classification_report",
        metrics_path,
    )
    labels = tuple(key for key in report if key not in SUMMARY_REPORT_KEYS)
    if not labels:
        raise ValueError(f"{metrics_path}: classification report has no labels")

    raw_matrix = holdout.get("confusion_matrix")
    if (
        not isinstance(raw_matrix, list)
        or len(raw_matrix) != len(labels)
        or any(
            not isinstance(row, list) or len(row) != len(labels)
            for row in raw_matrix
        )
    ):
        raise ValueError(
            f"{metrics_path}: collector_holdout.confusion_matrix dimensions "
            "do not match the classification report labels"
        )

    try:
        matrix = [[float(value) for value in row] for row in raw_matrix]
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{metrics_path}: confusion-matrix values must be numeric"
        ) from error
    if any(value < 0 or not math.isfinite(value) for row in matrix for value in row):
        raise ValueError(
            f"{metrics_path}: confusion-matrix values must be finite and non-negative"
        )
    return matrix, labels


def build_figure(
    matrix: Sequence[Sequence[float]],
    labels: Sequence[str],
    run_id: str,
    condition: str | None = None,
):
    """Build a row-normalized confusion-matrix figure for one MLflow run."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from sklearn.metrics import ConfusionMatrixDisplay

    values = np.asarray(matrix, dtype=np.float64)
    row_totals = values.sum(axis=1, keepdims=True)
    normalized = np.divide(
        values,
        row_totals,
        out=np.zeros_like(values),
        where=row_totals != 0,
    )

    figure = Figure(figsize=(7, 6))
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    ConfusionMatrixDisplay(
        confusion_matrix=normalized,
        display_labels=list(labels),
    ).plot(
        ax=axis,
        cmap="Blues",
        colorbar=False,
        values_format=".2f",
    )
    title_label = f"Condition {condition}" if condition else run_id[:7]
    axis.set_title(f"{title_label} (row normalized)")
    figure.tight_layout()
    return figure


def generate_image(
    artifacts_dir: Path,
    run_id: str,
    output_dir: Path,
    condition: str | None = None,
) -> Path:
    """Generate one normalized matrix image in the shared output directory."""
    matrix, labels = read_confusion_matrix(artifacts_dir / "metrics.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_filename(run_id, condition)
    figure = build_figure(matrix, labels, run_id, condition)
    try:
        figure.savefig(output_path)
    finally:
        figure.clear()
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate row-normalized collector-holdout confusion-matrix "
            "images for downloaded MLflow run IDs."
        )
    )
    parser.add_argument("run_ids", nargs="+", help="MLflow run IDs to process")
    parser.add_argument(
        "--results-root",
        type=Path,
        help="Downloaded results root (default: repository aws-results)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Image destination (default: <results-root>/confusion-matrices)"
        ),
    )
    parser.add_argument(
        "--condition-map",
        type=Path,
        help="JSON file mapping MLflow run IDs or prefixes to condition labels",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    results_root: Path | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    if results_root is None:
        results_root = (
            args.results_root
            if args.results_root is not None
            else Path(__file__).resolve().parents[1] / "aws-results"
        )
    elif args.results_root is not None:
        raise ValueError("results_root and --results-root cannot both be supplied")

    if not results_root.is_dir():
        print(f"error: results directory does not exist: {results_root}", file=sys.stderr)
        return 1

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else results_root / "confusion-matrices"
    )

    run_ids = list(dict.fromkeys(args.run_ids))
    try:
        condition_map = (
            load_condition_map(args.condition_map)
            if args.condition_map is not None
            else {}
        )
        matches = find_run_artifacts(results_root, run_ids)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    failed = False
    for run_id in run_ids:
        artifact_dirs = matches[run_id]
        if not artifact_dirs:
            print(f"error: run ID not found: {run_id}", file=sys.stderr)
            failed = True
            continue
        try:
            condition = condition_for_run(run_id, condition_map)
            output_path = generate_image(
                artifact_dirs[0], run_id, output_dir, condition
            )
        except (OSError, ValueError) as error:
            print(f"error: {error}", file=sys.stderr)
            failed = True
            continue
        print(output_path)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
