#!/usr/bin/env bash
set -Eeuo pipefail
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$project_root"
export ALL_TMD_GIT_COMMIT
ALL_TMD_GIT_COMMIT=$(git -c safe.directory="$project_root" rev-parse HEAD)
docker compose build study
docker compose --profile mlflow up -d --wait mlflow
docker compose run --rm study "$@"
