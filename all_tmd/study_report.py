from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from all_tmd.study import StudyPlan, canonical_digest
from all_tmd.study_runner import classification_metrics


METRICS = ("macro_f1", "balanced_accuracy")


def collect_runs(results_root: str | Path, study_id: str) -> list[dict[str, Any]]:
    runs = []
    for path in Path(results_root).rglob("metrics.json"):
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("study_id") != study_id or raw.get("condition") is None:
            continue
        prediction_path = path.with_name("predictions.parquet")
        if not prediction_path.exists():
            raise FileNotFoundError(f"predictions are missing for {path}")
        raw["metrics_path"] = str(path)
        raw["predictions_path"] = str(prediction_path)
        runs.append(raw)
    return runs


def report_study(
    plan: StudyPlan,
    *,
    results_root: str | Path,
    output_dir: str | Path,
    allow_partial: bool = False,
) -> dict[str, Any]:
    runs = collect_runs(results_root, plan.study_id)
    expected = {spec.key for spec in plan.parent_specs()}
    actual = {run["run_name"] for run in runs}
    missing = sorted(expected - actual)
    duplicates = sorted(name for name in actual if sum(run["run_name"] == name for run in runs) > 1)
    digests = {run["evaluation_manifest_digest"] for run in runs}
    valid = not missing and not duplicates and len(digests) == 1
    if not valid and not allow_partial:
        raise ValueError(
            f"study is incomplete or inconsistent: missing={len(missing)}, "
            f"duplicates={duplicates}, evaluation_digests={len(digests)}"
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    curve_rows = _curve_rows(runs, plan)
    curve = pd.DataFrame(curve_rows)
    curve.to_csv(destination / "transfer-curve.csv", index=False)
    _plot_transfer_curve(curve, destination)
    _plot_per_class(runs, plan, destination)
    _write_controlled_table(curve, destination / "controlled-comparison.tex")
    paired = _paired_rows(runs, plan)
    pd.DataFrame(paired).to_csv(destination / "paired-differences.csv", index=False)
    _write_paired_table(paired, destination / "paired-differences.tex")
    summary = {
        "schema_version": 1,
        "study_id": plan.study_id,
        "valid": valid,
        "expected_parent_runs": len(expected),
        "observed_parent_runs": len(runs),
        "missing_run_names": missing,
        "duplicate_run_names": duplicates,
        "evaluation_manifest_digests": sorted(digests),
        "model_lock_digests": sorted({run["model_lock_digest"] for run in runs}),
        "source_feature_digests": sorted(
            {run["source_feature_digest"] for run in runs if run.get("source_feature_digest")}
        ),
        "collector_feature_digests": sorted(
            {
                run["collector_feature_digest"]
                for run in runs
                if run.get("collector_feature_digest")
            }
        ),
        "split_digests": sorted(
            {run["split_digest"] for run in runs if run.get("split_digest")}
        ),
        "code_commits": sorted(
            {run["code_commit"] for run in runs if run.get("code_commit")}
        ),
        "snapshot_manifest_digest": canonical_digest(
            json.loads(plan.snapshot_path.read_text(encoding="utf-8"))
        ),
        "curve_digest": canonical_digest(curve_rows),
        "analysis_seed": plan.analysis_seed,
        "bootstrap_iterations": plan.bootstrap_iterations,
        "validation": {
            "complete_parent_grid": not missing and not duplicates,
            "shared_evaluation_manifest": len(digests) == 1,
        },
    }
    (destination / "study-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _curve_rows(runs: list[dict[str, Any]], plan: StudyPlan) -> list[dict[str, Any]]:
    rows = []
    grouped = _group_runs(runs)
    for (condition, fraction), condition_runs in sorted(grouped.items()):
        for metric in METRICS:
            values = [float(run["metrics"][metric]) for run in condition_runs]
            lower, upper = _bootstrap_interval(condition_runs, plan, metric)
            rows.append(
                {
                    "condition": condition,
                    "sydney_fraction": fraction,
                    "metric": metric,
                    "estimate": float(np.mean(values)),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "seeds": len(values),
                }
            )
    return rows


def _group_runs(runs: list[dict[str, Any]]) -> dict[tuple[str, float], list[dict[str, Any]]]:
    grouped: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for run in runs:
        grouped.setdefault((run["condition"], float(run["sydney_fraction"])), []).append(run)
    for values in grouped.values():
        values.sort(key=lambda run: int(run["seed"]))
    return grouped


def _bootstrap_interval(
    runs: list[dict[str, Any]], plan: StudyPlan, metric: str
) -> tuple[float, float]:
    predictions = {
        int(run["seed"]): pd.read_parquet(run["predictions_path"]) for run in runs
    }
    seeds = sorted(predictions)
    rng = np.random.default_rng(plan.analysis_seed + sum(ord(c) for c in metric))
    samples = []
    attempts = 0
    while len(samples) < plan.bootstrap_iterations:
        attempts += 1
        if attempts > plan.bootstrap_iterations * 20:
            raise RuntimeError("could not produce complete-class bootstrap samples")
        frames = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            frame = predictions[int(seed)]
            frames.append(frame.iloc[_hierarchical_indices(frame, rng)])
        sampled = pd.concat(frames, ignore_index=True)
        if set(sampled["label"].astype(int)) != {0, 1, 2}:
            continue
        samples.append(classification_metrics(sampled, plan.labels)[metric])
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _hierarchical_indices(frame: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Draw participants, then sessions, retaining every window in each draw."""

    participant_values = frame["participant_id"].astype(str)
    session_values = frame["session_id"].astype(str)
    participants = participant_values.unique()
    indices: list[int] = []
    for participant in rng.choice(participants, size=len(participants), replace=True):
        participant_indices = np.flatnonzero(
            participant_values.to_numpy() == str(participant)
        )
        sessions = session_values.iloc[participant_indices].unique()
        for session in rng.choice(sessions, size=len(sessions), replace=True):
            indices.extend(
                participant_indices[
                    session_values.iloc[participant_indices].to_numpy() == str(session)
                ].tolist()
            )
    return np.asarray(indices, dtype=int)


def _plot_transfer_curve(curve: pd.DataFrame, output_dir: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 4.5), layout="constrained")
    FigureCanvasAgg(figure)
    labels = {"sydney_only": "Sydney only", "nor_plus_sydney": "NOR-TMD + Sydney"}
    for axis, metric in zip(figure.subplots(1, 2), METRICS):
        baseline = curve.loc[
            (curve["condition"] == "nor_only") & (curve["metric"] == metric)
        ]
        if not baseline.empty:
            axis.axhline(float(baseline["estimate"].iloc[0]), color="black", linestyle="--", label="NOR-TMD only")
        for condition, label in labels.items():
            data = curve.loc[(curve["condition"] == condition) & (curve["metric"] == metric)].sort_values("sydney_fraction")
            if data.empty:
                continue
            x = data["sydney_fraction"].to_numpy() * 100
            y = data["estimate"].to_numpy()
            axis.plot(x, y, marker="o", label=label)
            axis.fill_between(x, data["ci_lower"], data["ci_upper"], alpha=0.18)
        axis.set_xlabel("Sydney calibration sessions (%)")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.savefig(output_dir / "transfer-curve.png", dpi=180)
    figure.savefig(output_dir / "transfer-curve.pdf")


def _plot_per_class(runs: list[dict[str, Any]], plan: StudyPlan, output_dir: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(10, 7), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 1)
    for axis, metric in zip(axes, ("recall", "f1")):
        for condition in ("sydney_only", "nor_plus_sydney"):
            for mode in plan.labels:
                points = []
                for fraction in plan.sydney_fractions[1:]:
                    matching = [run for run in runs if run["condition"] == condition and float(run["sydney_fraction"]) == fraction]
                    if matching:
                        points.append((fraction * 100, np.mean([run["metrics"]["per_class"][mode][metric] for run in matching])))
                if points:
                    axis.plot(*zip(*points), marker="o", label=f"{condition}: {mode}")
        axis.set_ylabel(metric.title())
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("Sydney calibration sessions (%)")
    figure.savefig(output_dir / "per-class-performance.png", dpi=180)


def _write_controlled_table(curve: pd.DataFrame, path: Path) -> None:
    lines = ["\\begin{tabular}{lrr}", "\\hline", "Condition & Macro F1 & Balanced accuracy \\\\", "\\hline"]
    labels = {"nor_only": "NOR-TMD only", "sydney_only": "Sydney only", "nor_plus_sydney": "NOR-TMD + Sydney"}
    for condition in labels:
        fraction = 0.0 if condition == "nor_only" else 1.0
        selected = curve.loc[(curve["condition"] == condition) & (curve["sydney_fraction"] == fraction)]
        values = {row.metric: row.estimate for row in selected.itertuples()}
        if values:
            lines.append(f"{labels[condition]} & {values['macro_f1']:.4f} & {values['balanced_accuracy']:.4f} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _paired_rows(runs: list[dict[str, Any]], plan: StudyPlan) -> list[dict[str, Any]]:
    grouped = _group_runs(runs)
    rows = []
    baseline = grouped.get(("nor_only", 0.0), [])
    baseline_by_seed = {int(run["seed"]): run for run in baseline}
    for fraction in plan.sydney_fractions[1:]:
        combined = {int(run["seed"]): run for run in grouped.get(("nor_plus_sydney", fraction), [])}
        sydney = {int(run["seed"]): run for run in grouped.get(("sydney_only", fraction), [])}
        for comparison, reference in (("combined_minus_nor", baseline_by_seed), ("combined_minus_sydney", sydney)):
            common = sorted(set(combined) & set(reference))
            for metric in METRICS:
                deltas = [combined[seed]["metrics"][metric] - reference[seed]["metrics"][metric] for seed in common]
                if deltas:
                    lower, upper = _paired_bootstrap_interval(
                        combined, reference, common, plan, comparison, metric
                    )
                    rows.append(
                        {
                            "sydney_fraction": fraction,
                            "comparison": comparison,
                            "metric": metric,
                            "estimate": float(np.mean(deltas)),
                            "ci_lower": lower,
                            "ci_upper": upper,
                            "seeds": len(common),
                        }
                    )
    return rows


def _paired_bootstrap_interval(
    combined: dict[int, dict[str, Any]],
    reference: dict[int, dict[str, Any]],
    common_seeds: list[int],
    plan: StudyPlan,
    comparison: str,
    metric: str,
) -> tuple[float, float]:
    pairs: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    identity_columns = [
        "participant_id",
        "session_id",
        "window_start_ms",
        "window_end_ms",
        "label",
    ]
    for seed in common_seeds:
        left = pd.read_parquet(combined[seed]["predictions_path"]).sort_values(
            identity_columns, kind="stable"
        ).reset_index(drop=True)
        right = pd.read_parquet(reference[seed]["predictions_path"]).sort_values(
            identity_columns, kind="stable"
        ).reset_index(drop=True)
        if not left[identity_columns].equals(right[identity_columns]):
            raise ValueError(
                f"paired predictions are not aligned for {comparison}, seed={seed}"
            )
        pairs[seed] = (left, right)

    salt = sum(ord(character) for character in f"{comparison}:{metric}")
    rng = np.random.default_rng(plan.analysis_seed + salt)
    samples: list[float] = []
    attempts = 0
    while len(samples) < plan.bootstrap_iterations:
        attempts += 1
        if attempts > plan.bootstrap_iterations * 20:
            raise RuntimeError("could not produce complete-class paired bootstrap samples")
        left_frames = []
        right_frames = []
        for seed in rng.choice(common_seeds, size=len(common_seeds), replace=True):
            left, right = pairs[int(seed)]
            indices = _hierarchical_indices(left, rng)
            left_frames.append(left.iloc[indices])
            right_frames.append(right.iloc[indices])
        sampled_left = pd.concat(left_frames, ignore_index=True)
        sampled_right = pd.concat(right_frames, ignore_index=True)
        if set(sampled_left["label"].astype(int)) != {0, 1, 2}:
            continue
        left_metric = classification_metrics(sampled_left, plan.labels)[metric]
        right_metric = classification_metrics(sampled_right, plan.labels)[metric]
        samples.append(float(left_metric - right_metric))
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def _write_paired_table(rows: list[dict[str, Any]], path: Path) -> None:
    lines = ["\\begin{tabular}{lllrrr}", "\\hline", "Sydney fraction & Comparison & Metric & Difference & 95\\% CI low & 95\\% CI high \\\\", "\\hline"]
    for row in rows:
        lines.append(
            f"{row['sydney_fraction']:.0%} & "
            f"{row['comparison'].replace('_', ' ')} & "
            f"{row['metric'].replace('_', ' ')} & {row['estimate']:+.4f} & "
            f"{row['ci_lower']:+.4f} & {row['ci_upper']:+.4f} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
