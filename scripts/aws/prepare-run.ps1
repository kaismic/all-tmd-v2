param(
    [ValidateSet("Full", "Smoke", "Tune")]
    [string]$Mode = "Full",
    [string]$RunId = "",
    [string]$NtfyTopic = "",
    [string]$NtfyServer = "https://ntfy.sh",
    [string]$NtfyEvents = "all-trials",
    [string]$StackName = "all-tmd-v2-worker",
    [string]$Region = "ap-southeast-2",
    [string]$Profile = "",
    [switch]$NoAutoStop,
    [switch]$AllowDirtyWorktree
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
. (Join-Path $PSScriptRoot "common.ps1")
Initialize-AwsContext -Region $Region -Profile $Profile

if (-not $AllowDirtyWorktree) {
    $status = & git -c "safe.directory=$($projectRoot.Replace('\', '/'))" `
        -C $projectRoot status --porcelain
    if ($LASTEXITCODE -ne 0) { throw "Could not inspect the Git worktree." }
    if ($status) {
        throw "The all-tmd-v2 worktree is dirty. Commit the run code or pass -AllowDirtyWorktree intentionally."
    }
}
$gitCommit = (& git -c "safe.directory=$($projectRoot.Replace('\', '/'))" `
    -C $projectRoot rev-parse HEAD).Trim()
$gitRepository = (& git -c "safe.directory=$($projectRoot.Replace('\', '/'))" `
    -C $projectRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0) { throw "Could not resolve the Git commit and repository." }
Assert-AllTmdGitCommitAvailable `
    -ProjectRoot $projectRoot `
    -Repository $gitRepository `
    -Commit $gitCommit

$modeValue = $Mode.ToLowerInvariant()
if (-not $RunId) {
    $timestamp = [DateTime]::UtcNow.ToString("yyyyMMddTHHmmssZ")
    $RunId = "$timestamp-$($gitCommit.Substring(0, 8))-$modeValue"
}
if ($RunId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$') {
    throw "RunId contains unsupported characters."
}

$outputs = Get-AllTmdStackOutputs -StackName $StackName
$bundleDir = Join-Path ([System.IO.Path]::GetTempPath()) "all-tmd-$RunId"
if (Test-Path -LiteralPath $bundleDir) {
    throw "Temporary run bundle already exists: $bundleDir"
}
try {
    $arguments = @(
        "-m", "all_tmd.aws_bundle",
        "--project-root", $projectRoot,
        "--output", $bundleDir,
        "--run-id", $RunId,
        "--git-repository", $gitRepository,
        "--git-commit", $gitCommit,
        "--mode", $modeValue,
        "--ntfy-server", $NtfyServer,
        "--ntfy-topic", $NtfyTopic,
        "--ntfy-events", $NtfyEvents,
        "--ntfy-token-parameter", $outputs.NtfyTokenParameterName
        "--collector-sessions-bucket", $outputs.CollectorSessionsBucketName
        "--collector-sessions-table", $outputs.CollectorSessionsTableName
    )
    if ($NoAutoStop) { $arguments += "--no-auto-stop" }
    Push-Location $projectRoot
    try {
        & python @arguments
        if ($LASTEXITCODE -ne 0) { throw "Run bundle generation failed." }
    }
    finally {
        Pop-Location
    }
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "remote\run-trials-cloud.sh") `
        -Destination $bundleDir
    Copy-Item -LiteralPath (Join-Path $PSScriptRoot "remote\sync-collector-sessions.py") `
        -Destination $bundleDir
    $prefix = "s3://$($outputs.BucketName)/all-tmd-v2/config/$RunId/"
    Invoke-AllTmdAws -Arguments @(
        "s3", "sync", $bundleDir, $prefix, "--only-show-errors"
    ) -AllowEmpty
    Write-Host "Prepared $Mode run: $RunId"
    Write-Host "Start it with: .\scripts\aws\start-run.ps1 -RunId $RunId"
}
finally {
    if (Test-Path -LiteralPath $bundleDir) {
        Remove-Item -LiteralPath $bundleDir -Recurse -Force
    }
}
