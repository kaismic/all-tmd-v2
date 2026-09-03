from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import yaml


RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
COMMON_FILES = (
    "model.config.yaml",
    "study-plan.json",
    "study-trial.json",
    "manifests/sydney-166.json",
)
EVALUATION_FILES = ("model-lock.json", "manifests/sydney-lopo.json")
AWS_MLFLOW_TRACKING_URI = "sqlite:////mlflow-data/mlflow.db"
AWS_MLFLOW_ARTIFACT_LOCATION = "file:///mlflow-data/mlartifacts"


def create_run_bundle(
    project_root: str | Path,
    output_dir: str | Path,
    *,
    run_id: str,
    git_repository: str,
    git_commit: str,
    mode: str = "full",
    ntfy_server: str = "https://ntfy.sh",
    ntfy_topic: str = "",
    ntfy_events: str = "all-trials",
    ntfy_token_parameter: str = "/all-tmd-v2/ntfy-token",
    collector_sessions_bucket: str = "",
    collector_sessions_table: str = "",
    auto_stop: bool = True,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("invalid run_id")
    if mode not in {"full", "smoke", "tune"}:
        raise ValueError("mode must be 'full', 'smoke', or 'tune'")
    if not git_repository.strip() or not re.fullmatch(r"[0-9a-fA-F]{40}", git_commit):
        raise ValueError("a repository and full Git commit are required")
    if not collector_sessions_bucket.strip() or not collector_sessions_table.strip():
        raise ValueError("collector bucket and table are required")
    root = Path(project_root)
    required = COMMON_FILES + (() if mode == "tune" else EVALUATION_FILES)
    for name in required:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required run configuration is missing: {root / name}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    for name in required:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root / name, target)
    model_config = yaml.safe_load((destination / "model.config.yaml").read_text(encoding="utf-8"))
    model_config["mlflow"].update(
        {
            "enabled": True,
            "tracking_uri": AWS_MLFLOW_TRACKING_URI,
            "artifact_location": AWS_MLFLOW_ARTIFACT_LOCATION,
        }
    )
    (destination / "model.config.yaml").write_text(
        yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8"
    )
    timestamp = created_at or datetime.now(timezone.utc)
    manifest = {
        "schema_version": 2,
        "run_id": run_id,
        "mode": mode,
        "created_at": timestamp.astimezone(timezone.utc).isoformat(),
        "git_repository": git_repository,
        "git_commit": git_commit.lower(),
        "expected_parent_runs": 1 if mode == "smoke" else (0 if mode == "tune" else 33),
        "config_sha256": {name: _sha256(destination / name) for name in required},
        "notifications": {
            "server": ntfy_server,
            "topic": ntfy_topic,
            "events": ntfy_events,
            "token_parameter": ntfy_token_parameter,
        },
        "collector_sessions": {
            "bucket": collector_sessions_bucket,
            "table": collector_sessions_table,
        },
        "auto_stop": auto_stop,
    }
    (destination / "run-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Create an AWS ALL-TMD v2 study bundle")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--git-repository", required=True)
    parser.add_argument("--git-commit", required=True)
    parser.add_argument("--mode", choices=("full", "smoke", "tune"), default="full")
    parser.add_argument("--ntfy-server", default="https://ntfy.sh")
    parser.add_argument("--ntfy-topic", default="")
    parser.add_argument("--ntfy-events", default="all-trials")
    parser.add_argument("--collector-sessions-bucket", required=True)
    parser.add_argument("--collector-sessions-table", required=True)
    parser.add_argument("--ntfy-token-parameter", default="/all-tmd-v2/ntfy-token")
    parser.add_argument("--no-auto-stop", action="store_true")
    args = parser.parse_args(argv)
    manifest = create_run_bundle(
        args.project_root,
        args.output,
        run_id=args.run_id,
        git_repository=args.git_repository,
        git_commit=args.git_commit,
        mode=args.mode,
        ntfy_server=args.ntfy_server,
        ntfy_topic=args.ntfy_topic,
        ntfy_events=args.ntfy_events,
        ntfy_token_parameter=args.ntfy_token_parameter,
        collector_sessions_bucket=args.collector_sessions_bucket,
        collector_sessions_table=args.collector_sessions_table,
        auto_stop=not args.no_auto_stop,
    )
    print(f"Created {manifest['mode']} run bundle {manifest['run_id']}")
    return 0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
