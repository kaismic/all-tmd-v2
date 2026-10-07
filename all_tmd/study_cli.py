from __future__ import annotations

import argparse
import json
from pathlib import Path

from all_tmd.config import PipelineConfig
from all_tmd.ingest import ingest_collector, ingest_training_dataset
from all_tmd.mlflow_importer import import_mlflow_runs
from all_tmd.source_tuning import promote_model, tune_source
from all_tmd.study import StudyPlan, create_lopo_manifest, verify_snapshot, verify_snapshot_files
from all_tmd.study_report import REPORT_FILES, report_study
from all_tmd.study_runner import read_feature_dataset, run_study
from all_tmd.windowing import build_features, feature_output_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ALL-TMD v2 controlled Sydney transfer study")
    parser.add_argument("--config", default="model.config.yaml")
    parser.add_argument("--trial", default="study-trial.json")
    parser.add_argument("--plan", default="study-plan.json")
    subparsers = parser.add_subparsers(dest="command", required=True)
    snapshot = subparsers.add_parser("snapshot")
    snapshot.add_argument("action", choices=("verify",))
    snapshot.add_argument("--input-dir")
    subparsers.add_parser("prepare-data")
    subparsers.add_parser("create-lopo")
    tune = subparsers.add_parser("tune-source")
    tune.add_argument("--output-root", default="/data/all-tmd-v2-results")
    promote = subparsers.add_parser("promote-model")
    promote.add_argument("--run-id", required=True)
    promote.add_argument("--output-root", default="/data/all-tmd-v2-results")
    run = subparsers.add_parser("run-study")
    run.add_argument("--output-root", default="/data/all-tmd-v2-results")
    run.add_argument("--execution-backend", choices=("local", "aws"), default="local")
    run.add_argument("--limit", type=int)
    importer = subparsers.add_parser("import-mlflow")
    importer.add_argument("--source-database", required=True)
    importer.add_argument("--destination-uri", required=True)
    importer.add_argument("--sweep-id", required=True)
    report = subparsers.add_parser("report")
    report.add_argument("--results-root", default="/data/all-tmd-v2-results")
    report.add_argument("--output-dir", default="report-artifacts")
    report.add_argument("--allow-partial", action="store_true")
    report.add_argument(
        "--files", nargs="+", choices=REPORT_FILES, metavar="FILENAME",
        help="Generate only these report files (default: all). Choices: %(choices)s",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plan = StudyPlan.load(args.plan)
    if args.command == "snapshot":
        result = (
            verify_snapshot_files(plan.snapshot_path, args.input_dir)
            if args.input_dir
            else verify_snapshot(plan.snapshot_path)
        )
    elif args.command == "import-mlflow":
        result = import_mlflow_runs(
            args.source_database,
            args.destination_uri,
            study_id=plan.study_id,
            sweep_id=args.sweep_id,
        )
    elif args.command == "report":
        result = report_study(
            plan,
            results_root=args.results_root,
            output_dir=args.output_dir,
            allow_partial=args.allow_partial,
            files=args.files,
        )
    else:
        config = PipelineConfig.from_files(args.config, args.trial)
        if args.command == "prepare-data":
            verify_snapshot_files(plan.snapshot_path, config.sources.collector.input_path)
            result = {
                "source_events": str(ingest_training_dataset(config)),
                "collector_events": str(ingest_collector(config)),
                "features": {key: str(value) for key, value in build_features(config).items()},
            }
        elif args.command == "create-lopo":
            collector = read_feature_dataset(feature_output_dir(config, "collector"))
            eligible = set(verify_snapshot(plan.snapshot_path)["eligible_session_ids"])
            collector = collector.loc[collector["session_id"].astype(str).isin(eligible)]
            result = create_lopo_manifest(collector)
            if plan.lopo_manifest_path.exists():
                existing = json.loads(plan.lopo_manifest_path.read_text(encoding="utf-8"))
                if existing != result:
                    raise ValueError("existing immutable LOPO manifest differs")
            else:
                plan.lopo_manifest_path.write_text(
                    json.dumps(result, indent=2) + "\n", encoding="utf-8"
                )
        elif args.command == "tune-source":
            result = tune_source(config, output_root=args.output_root)
        elif args.command == "promote-model":
            result = promote_model(
                args.run_id,
                output_root=args.output_root,
                lock_path=plan.model_lock_path,
            )
        elif args.command == "run-study":
            result = run_study(
                config,
                plan,
                output_root=args.output_root,
                execution_backend=args.execution_backend,
                spec_limit=args.limit,
            )
        else:
            raise AssertionError(args.command)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
