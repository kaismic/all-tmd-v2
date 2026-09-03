#!/usr/bin/env bash

set -Eeuo pipefail

state_dir=/etc/all-tmd-v2
install_dir=/usr/local/lib/all-tmd-v2
service_name=all-tmd-trials.service
data_dir=/mnt/all-tmd-data

usage() {
    printf '%s\n' \
        "Usage:" \
        "  run-trials-cloud.sh install --bucket BUCKET --run-id RUN_ID" \
        "  run-trials-cloud.sh execute" \
        "  run-trials-cloud.sh start-mlflow --run-id RUN_ID" \
        "  run-trials-cloud.sh stop-mlflow --run-id RUN_ID"
}

validate_bucket() {
    [[ $1 =~ ^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$ ]]
}

validate_run_id() {
    [[ $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]
}

install_service() {
    local bucket=
    local run_id=
    while (($#)); do
        case "$1" in
            --bucket)
                bucket=${2:-}
                shift 2
                ;;
            --run-id)
                run_id=${2:-}
                shift 2
                ;;
            *)
                usage >&2
                return 2
                ;;
        esac
    done
    if ! validate_bucket "$bucket" || ! validate_run_id "$run_id"; then
        printf '%s\n' "Invalid S3 bucket or run ID." >&2
        return 2
    fi
    if systemctl is-active --quiet "$service_name"; then
        printf '%s\n' "An All-TMD trial service is already active." >&2
        return 1
    fi

    install -d -m 0755 "$state_dir" "$install_dir"
    install -m 0755 "$0" "$install_dir/run-trials-cloud.sh"
    {
        printf 'ALL_TMD_AWS_BUCKET=%q\n' "$bucket"
        printf 'ALL_TMD_RUN_ID=%q\n' "$run_id"
    } >"$state_dir/run.env"
    chmod 0600 "$state_dir/run.env"

    cat >/etc/systemd/system/$service_name <<EOF
[Unit]
Description=All-TMD AWS trial sweep
After=docker.service network-online.target
Wants=network-online.target
Requires=docker.service
RequiresMountsFor=$data_dir

[Service]
Type=oneshot
EnvironmentFile=$state_dir/run.env
ExecStart=$install_dir/run-trials-cloud.sh execute
TimeoutStartSec=infinity
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl reset-failed "$service_name" 2>/dev/null || true
    systemctl start --no-block "$service_name"
    printf 'Started %s for run %s.\n' "$service_name" "$run_id"
}

metadata_document() {
    local token
    token=$(curl --fail --silent --show-error --request PUT \
        --header "X-aws-ec2-metadata-token-ttl-seconds: 60" \
        http://169.254.169.254/latest/api/token)
    curl --fail --silent --show-error \
        --header "X-aws-ec2-metadata-token: $token" \
        http://169.254.169.254/latest/dynamic/instance-identity/document
}

manage_mlflow_server() {
    local action=$1
    shift
    local expected_run_id=
    while (($#)); do
        case "$1" in
            --run-id)
                expected_run_id=${2:-}
                shift 2
                ;;
            *)
                usage >&2
                return 2
                ;;
        esac
    done
    validate_run_id "$expected_run_id" || {
        printf '%s\n' "Invalid run ID." >&2
        return 2
    }
    [[ -r $state_dir/run.env ]] || {
        printf '%s\n' "Run state is missing: $state_dir/run.env" >&2
        return 1
    }
    # shellcheck disable=SC1091
    source "$state_dir/run.env"
    if [[ $ALL_TMD_RUN_ID != "$expected_run_id" ]]; then
        printf 'Active run is %s, not %s.\n' \
            "$ALL_TMD_RUN_ID" "$expected_run_id" >&2
        return 1
    fi

    local run_state_dir="$data_dir/cloud-runs/$ALL_TMD_RUN_ID"
    local manifest="$run_state_dir/config/run-manifest.json"
    [[ -f $manifest ]] || {
        printf '%s\n' "Active run manifest is not available yet." >&2
        return 1
    }
    local git_commit
    git_commit=$(jq -r .git_commit "$manifest")
    [[ $git_commit =~ ^[0-9a-f]{40}$ ]] || {
        printf '%s\n' "Active run manifest has an invalid Git commit." >&2
        return 1
    }
    local checkout="/opt/all-tmd-v2/checkouts/$git_commit"
    [[ -f $checkout/docker-compose.yml && -f $checkout/.env ]] || {
        printf '%s\n' "Active run checkout is not ready for MLflow." >&2
        return 1
    }

    case "$action" in
        start)
            (cd "$checkout" && \
                docker compose --profile mlflow up -d --wait mlflow)
            ;;
        stop)
            (cd "$checkout" && \
                docker compose --profile mlflow stop mlflow)
            ;;
        *)
            return 2
            ;;
    esac
}

execute_run() {
    if [[ ! -r $state_dir/run.env ]]; then
        printf '%s\n' "Run state is missing: $state_dir/run.env" >&2
        return 1
    fi
    # shellcheck disable=SC1091
    source "$state_dir/run.env"
    validate_bucket "$ALL_TMD_AWS_BUCKET"
    validate_run_id "$ALL_TMD_RUN_ID"

    # These must outlive execute_run because the EXIT trap runs after the
    # function has unwound on an early failure.
    config_prefix="s3://$ALL_TMD_AWS_BUCKET/all-tmd-v2/config/$ALL_TMD_RUN_ID"
    result_prefix="s3://$ALL_TMD_AWS_BUCKET/all-tmd-v2/results/$ALL_TMD_RUN_ID"
    run_state_dir="$data_dir/cloud-runs/$ALL_TMD_RUN_ID"
    bundle_dir="$run_state_dir/config"
    log_path="$run_state_dir/run.log"
    resource_path="$run_state_dir/resource-usage.txt"
    start_epoch=
    checkout=
    final_status=1
    start_epoch=$(date +%s)
    mkdir -p "$bundle_dir"
    touch "$log_path"
    exec > >(tee -a "$log_path") 2>&1

    finish() {
        local trapped_status=$?
        local end_epoch
        local manifest_commit=unknown
        local manifest_mode=unknown
        local expected_parent_runs=0
        set +e
        trap - EXIT
        if ((final_status == 1 && trapped_status != 0)); then
            final_status=$trapped_status
        fi
        end_epoch=$(date +%s)
        if [[ -f $bundle_dir/run-manifest.json ]]; then
            manifest_commit=$(jq -r '.git_commit // "unknown"' "$bundle_dir/run-manifest.json")
            manifest_mode=$(jq -r '.mode // "unknown"' "$bundle_dir/run-manifest.json")
            expected_parent_runs=$(jq -r '.expected_parent_runs // 0' "$bundle_dir/run-manifest.json")
        fi
        if [[ -n $checkout && -f $checkout/docker-compose.yml ]]; then
            (cd "$checkout" && docker compose --profile mlflow down) || true
        fi
        jq -n \
            --arg run_id "$ALL_TMD_RUN_ID" \
            --arg git_commit "$manifest_commit" \
            --arg mode "$manifest_mode" \
            --argjson expected_parent_runs "$expected_parent_runs" \
            --argjson exit_code "$final_status" \
            --arg started_at "$(date --date="@$start_epoch" --iso-8601=seconds)" \
            --arg completed_at "$(date --date="@$end_epoch" --iso-8601=seconds)" \
            --argjson duration_seconds "$((end_epoch - start_epoch))" \
            --argjson logical_processors "$(nproc)" \
            --argjson memory_mib "$(awk '/MemTotal/ {print int($2 / 1024)}' /proc/meminfo)" \
            '{schema_version: 1, run_id: $run_id, git_commit: $git_commit,
              mode: $mode, expected_parent_runs: $expected_parent_runs, exit_code: $exit_code,
              started_at: $started_at, completed_at: $completed_at,
              duration_seconds: $duration_seconds,
              logical_processors: $logical_processors, memory_mib: $memory_mib}' \
            >"$run_state_dir/run-summary.json"

        if [[ -d $run_state_dir/results ]]; then
            aws s3 sync "$run_state_dir/results" \
                "$result_prefix/results" --only-show-errors || true
        fi
        if [[ -d $run_state_dir/mlflow ]]; then
            aws s3 sync "$run_state_dir/mlflow" \
                "$result_prefix/mlflow" --only-show-errors || true
        fi
        aws s3 sync "$run_state_dir" "$result_prefix/run" \
            --exclude "config/*" --exclude "mlflow/*" \
            --only-show-errors || true

        local auto_stop=false
        if [[ -f $bundle_dir/run-manifest.json ]]; then
            auto_stop=$(jq -r '.auto_stop // true' "$bundle_dir/run-manifest.json")
        fi
        if [[ $auto_stop == true ]]; then
            local identity
            local instance_id
            local region
            identity=$(metadata_document)
            instance_id=$(jq -r .instanceId <<<"$identity")
            region=$(jq -r .region <<<"$identity")
            printf 'Stopping EC2 instance %s after result upload.\n' "$instance_id"
            aws ec2 stop-instances --region "$region" \
                --instance-ids "$instance_id" >/dev/null || true
        fi
        exit "$final_status"
    }
    trap finish EXIT

    printf 'Preparing All-TMD AWS run %s.\n' "$ALL_TMD_RUN_ID"
    aws s3 sync "$config_prefix" "$bundle_dir" --only-show-errors
    local manifest="$bundle_dir/run-manifest.json"
    [[ -f $manifest ]]
    [[ $(jq -r .schema_version "$manifest") == 2 ]]
    [[ $(jq -r .run_id "$manifest") == "$ALL_TMD_RUN_ID" ]]
    while IFS= read -r name; do
        local expected
        local actual
        expected=$(jq -r --arg name "$name" '.config_sha256[$name]' "$manifest")
        actual=$(sha256sum "$bundle_dir/$name" | awk '{print $1}')
        [[ $expected == "$actual" ]] || {
            printf 'Checksum mismatch for %s.\n' "$name" >&2
            return 1
        }
    done < <(jq -r '.config_sha256 | keys[]' "$manifest")

    local git_repository
    local git_commit
    git_repository=$(jq -r .git_repository "$manifest")
    git_commit=$(jq -r .git_commit "$manifest")
    [[ $git_commit =~ ^[0-9a-f]{40}$ ]]
    checkout="/opt/all-tmd-v2/checkouts/$git_commit"
    if [[ ! -d $checkout/.git ]]; then
        mkdir -p "$(dirname "$checkout")"
        git clone --no-checkout "$git_repository" "$checkout"
    fi
    git -C "$checkout" fetch --depth 1 origin "$git_commit"
    git -C "$checkout" checkout --detach --force "$git_commit"
    install -m 0644 "$bundle_dir/model.config.yaml" "$checkout/model.config.yaml"
    install -m 0644 "$bundle_dir/study-plan.json" "$checkout/study-plan.json"
    install -m 0644 "$bundle_dir/study-trial.json" "$checkout/study-trial.json"
    rm -rf "$checkout/manifests"
    cp -R "$bundle_dir/manifests" "$checkout/manifests"
    if [[ -f $bundle_dir/model-lock.json ]]; then
        install -m 0644 "$bundle_dir/model-lock.json" "$checkout/model-lock.json"
    fi

    mkdir -p \
        "$data_dir/nor-tmd-data" \
        "$data_dir/us-tmd-data" \
        "$data_dir/downloaded_sessions" \
        "$data_dir/all-tmd-v2-work" \
        "$run_state_dir/mlflow/mlartifacts" \
        "$run_state_dir/results"
    for source in nor-tmd-data us-tmd-data downloaded_sessions; do
        if [[ $source == downloaded_sessions ]]; then
            continue
        fi
        aws s3 sync \
            "s3://$ALL_TMD_AWS_BUCKET/all-tmd-v2/inputs/$source" \
            "$data_dir/$source" --only-show-errors
    done
    python3 "$bundle_dir/sync-collector-sessions.py" \
        --bucket "$(jq -r .collector_sessions.bucket "$manifest")" \
        --table "$(jq -r .collector_sessions.table "$manifest")" \
        --output-dir "$data_dir/downloaded_sessions" \
        --required-manifest "$bundle_dir/manifests/sydney-166.json" \
        --snapshot-path "$run_state_dir/collector-snapshot.json" \
        --run-id "$ALL_TMD_RUN_ID"

    local ntfy_server
    local ntfy_topic
    local ntfy_events
    local token_parameter
    local ntfy_token=
    ntfy_server=$(jq -r '.notifications.server // "https://ntfy.sh"' "$manifest")
    ntfy_topic=$(jq -r '.notifications.topic // ""' "$manifest")
    ntfy_events=$(jq -r '.notifications.events // "all-trials"' "$manifest")
    token_parameter=$(jq -r '.notifications.token_parameter // ""' "$manifest")
    if [[ -n $ntfy_topic && -n $token_parameter ]]; then
        if ! ntfy_token=$(aws ssm get-parameter --name "$token_parameter" \
            --with-decryption --query Parameter.Value --output text 2>&1); then
            if [[ $ntfy_token == *ParameterNotFound* ]]; then
                printf 'Notification token %s is absent; continuing without one.\n' \
                    "$token_parameter"
                ntfy_token=
            else
                printf '%s\n' "$ntfy_token" >&2
                return 1
            fi
        fi
    fi
    if [[ $ntfy_server == *$'\n'* || $ntfy_topic == *$'\n'* \
        || $ntfy_events == *$'\n'* || $ntfy_token == *$'\n'* ]]; then
        printf '%s\n' "Notification settings cannot contain newlines." >&2
        return 1
    fi
    {
        printf 'ALL_TMD_DATA_DIR=%s\n' "$data_dir"
        printf 'ALL_TMD_MLFLOW_DATA_DIR=%s\n' "$run_state_dir/mlflow"
        printf 'MLFLOW_TRACKING_URI=%s\n' \
            'sqlite:////mlflow-data/mlflow.db'
        printf 'MLFLOW_BACKEND_STORE_URI=%s\n' \
            'sqlite:////mlflow-data/mlflow.db'
        printf 'MLFLOW_ARTIFACTS_DESTINATION=%s\n' \
            '/mlflow-data/mlartifacts'
        printf 'ALL_TMD_GIT_COMMIT=%s\n' "$(jq -r .git_commit "$manifest")"
        printf 'NTFY_SERVER=%s\n' "$ntfy_server"
        printf 'NTFY_TOPIC=%s\n' "$ntfy_topic"
        printf 'NTFY_TOKEN=%s\n' "$ntfy_token"
        printf 'NTFY_EVENTS=%s\n' "$ntfy_events"
    } >"$checkout/.env"
    chmod 0600 "$checkout/.env"

    local mode
    mode=$(jq -r .mode "$manifest")
    printf 'Starting ALL-TMD v2 %s study run.\n' "$mode"
    export ALL_TMD_SKIP_MLFLOW_SERVER=true
    set +e
    (
        cd "$checkout"
        docker compose build
        docker compose run --rm study snapshot verify \
            --input-dir /data/downloaded_sessions
        docker compose run --rm study prepare-data
        if [[ $mode == tune ]]; then
            /usr/bin/time -v -o "$resource_path" docker compose run --rm study \
                tune-source --output-root "/data/cloud-runs/$ALL_TMD_RUN_ID/results"
        else
            docker compose run --rm study create-lopo
            local limit_args=()
            if [[ $mode == smoke ]]; then
                limit_args=(--limit 1)
            fi
            /usr/bin/time -v -o "$resource_path" docker compose run --rm study \
                run-study --execution-backend aws \
                --output-root "/data/cloud-runs/$ALL_TMD_RUN_ID/results" "${limit_args[@]}"
        fi
    )
    final_status=$?
    set -e
    return "$final_status"
}

case ${1:-} in
    install)
        shift
        install_service "$@"
        ;;
    execute)
        execute_run
        ;;
    start-mlflow)
        shift
        manage_mlflow_server start "$@"
        ;;
    stop-mlflow)
        shift
        manage_mlflow_server stop "$@"
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac
