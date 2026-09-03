from pathlib import Path


AWS_SCRIPTS = Path(__file__).parents[1] / "scripts" / "aws"


def test_worker_setup_waits_for_ssm_instead_of_ec2_status_checks():
    for name in ("deploy.ps1", "start-run.ps1"):
        script = (AWS_SCRIPTS / name).read_text(encoding="utf-8")

        assert "Wait-AllTmdSsmOnline" in script
        assert '"instance-status-ok"' not in script


def test_deploy_starts_an_existing_stopped_worker_before_validation():
    deploy = (AWS_SCRIPTS / "deploy.ps1").read_text(encoding="utf-8")

    assert 'if ($instanceState -eq "stopped")' in deploy
    assert '"ec2", "start-instances"' in deploy
    assert "Wait-AllTmdSsmOnline -InstanceId $instanceId" in deploy


def test_ssm_wait_detects_an_externally_stopped_instance():
    common = (AWS_SCRIPTS / "common.ps1").read_text(encoding="utf-8")

    assert "Get-AllTmdEc2InstanceState" in common
    assert '"stopping", "stopped", "shutting-down", "terminated"' in common
    assert "It may have been stopped externally." in common


def test_aws_run_scripts_reject_a_commit_missing_from_the_remote():
    common = (AWS_SCRIPTS / "common.ps1").read_text(encoding="utf-8")
    preparer = (AWS_SCRIPTS / "prepare-run.ps1").read_text(encoding="utf-8")
    launcher = (AWS_SCRIPTS / "start-run.ps1").read_text(encoding="utf-8")

    assert "function Assert-AllTmdGitCommitAvailable" in common
    assert "fetch --dry-run --depth 1" in common
    assert "Push it before preparing or starting an AWS run." in common
    assert "Assert-AllTmdGitCommitAvailable" in preparer
    assert '"s3", "cp", $manifestUri, "-"' in launcher
    assert "Assert-AllTmdGitCommitAvailable" in launcher


def test_cloud_runner_syncs_collector_backend_directly():
    runner = (AWS_SCRIPTS / "remote" / "run-trials-cloud.sh").read_text(
        encoding="utf-8"
    )
    uploader = (AWS_SCRIPTS / "upload-inputs.ps1").read_text(encoding="utf-8")

    assert 'python3 "$bundle_dir/sync-collector-sessions.py"' in runner
    assert '--output-dir "$data_dir/downloaded_sessions"' in runner
    assert '--snapshot-path "$run_state_dir/collector-snapshot.json"' in runner
    assert '--run-id "$ALL_TMD_RUN_ID"' in runner
    assert '@("nor-tmd-data", "us-tmd-data")' in uploader
    assert 'all-tmd-v2/inputs/$source' in uploader


def test_cloud_runner_uses_serverless_run_specific_mlflow_storage():
    runner = (AWS_SCRIPTS / "remote" / "run-trials-cloud.sh").read_text(
        encoding="utf-8"
    )
    trial_runner = (AWS_SCRIPTS.parent / "run-trials.sh").read_text(
        encoding="utf-8"
    )

    assert "ALL_TMD_SKIP_MLFLOW_SERVER=true" in runner
    assert '[[ ${ALL_TMD_SKIP_MLFLOW_SERVER:-false} != true ]]' in trial_runner
    assert 'ALL_TMD_MLFLOW_DATA_DIR=%s' in runner
    assert 'MLFLOW_TRACKING_URI=%s' in runner
    assert 'aws s3 sync "$run_state_dir/mlflow"' in runner
    assert '"$result_prefix/mlflow" --only-show-errors' in runner
    assert '$data_dir/all-tmd-work/mlflow.db' not in runner
    assert '$data_dir/all-tmd-work/mlartifacts' not in runner


def test_port_forward_starts_and_stops_on_demand_mlflow():
    forwarder = (AWS_SCRIPTS / "port-forward-mlflow.ps1").read_text(
        encoding="utf-8"
    )
    runner = (AWS_SCRIPTS / "remote" / "run-trials-cloud.sh").read_text(
        encoding="utf-8"
    )

    assert '[Parameter(Mandatory = $true)]' in forwarder
    assert "start-mlflow --run-id '$RunId'" in forwarder
    assert "stop-mlflow --run-id '$RunId'" in forwarder
    assert "start-mlflow)" in runner
    assert "stop-mlflow)" in runner


def test_downloaded_results_viewer_uses_pinned_local_mlflow_server():
    viewer = (AWS_SCRIPTS / "view-results.ps1").read_text(encoding="utf-8")

    assert "[int]$LocalPort = 5003" in viewer
    assert "ghcr.io/mlflow/mlflow:v3.14.0" in viewer
    assert '"127.0.0.1:${LocalPort}:5002"' in viewer
    assert "sqlite:////data/all-tmd-work/mlflow.db" in viewer
    assert '"--artifacts-destination", "/data/all-tmd-work/mlartifacts"' in viewer
    assert "target=/mlflow-data" in viewer
    assert '"--label", "$viewerLabel=true"' in viewer
    assert '"--rm"' in viewer


def test_downloaded_results_viewer_validates_and_auto_selects_runs():
    viewer = (AWS_SCRIPTS / "view-results.ps1").read_text(encoding="utf-8")

    assert "RunId contains unsupported characters." in viewer
    assert 'Sort-Object LastWriteTime -Descending' in viewer
    assert 'Join-Path $_.FullName "mlflow\\mlflow.db"' in viewer
    assert "MLflow may contain only partial results." in viewer
